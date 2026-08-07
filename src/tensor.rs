//! hos-tensor: a from-scratch tensor + reverse-mode autograd core.
//!
//! Unlike a scalar micrograd, this is tensor-level: each `Tensor` holds an
//! n-dim f32 array, its gradient, and (if it was produced by an op) a closure
//! that pushes gradient to its parents. `backward()` walks the graph in reverse
//! topological order. This is the foundation HOS trains on — extend it with new
//! ops, a GPU backend (reuse the Metal kernels), and your own model format.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

use rayon::prelude::*;

use crate::metal_be::{download_buf, Gpu, GpuBuf};

// Monotonic, never-reused tensor id — the stable key for caching a frozen
// tensor's GPU buffer (a freed Rc address could be reused with different data; an
// id never is).
static NEXT_TENSOR_ID: AtomicU64 = AtomicU64::new(1);

// A per-batch matmul only goes to the GPU when it's big enough to amortize the
// dispatch cost; below this, the CPU path is faster (and bit-for-bit with the
// reference). Small attention bmms (short sequences) stay on CPU; large ones
// (big models / long context) use the GPU kernels.
const GPU_BMM_MIN: usize = 1 << 20;

// Optional GPU backend for training matmuls. Thread-local (training is single-threaded);
// when enabled, Tensor::matmul runs its forward + backward on the GPU.
thread_local! {
    static GPU: RefCell<Option<Gpu>> = const { RefCell::new(None) };
}
/// Turn the GPU matmul backend on/off for training.
pub fn use_gpu(on: bool) {
    // GPU training is macOS-only; on other targets this is a no-op (CPU matmul).
    if !cfg!(target_os = "macos") {
        return;
    }
    GPU.with(|g| {
        let mut b = g.borrow_mut();
        if on {
            if b.is_none() {
                *b = Some(Gpu::new());
            }
        } else {
            *b = None;
        }
    });
}
fn gpu_on() -> bool {
    GPU.with(|g| g.borrow().is_some())
}
fn gmm(a: &[f32], m: usize, k: usize, b: &[f32], n: usize, b_key: Option<u64>) -> Vec<f32> {
    GPU.with(|g| {
        g.borrow()
            .as_ref()
            .unwrap()
            .matmul_f32_keyed(a, m, k, b, n, b_key)
    })
}
/// Drop any cached GPU buffer for tensor `id` — called when its data is mutated,
/// so a stale frozen-weight buffer can't survive an in-place edit.
fn gpu_evict(id: u64) {
    GPU.with(|g| {
        if let Some(gpu) = g.borrow().as_ref() {
            gpu.evict(id);
        }
    });
}
/// Empty the GPU keyed-buffer cache — call once per training step so transient
/// non-grad activations (which get a fresh id every step) don't accumulate.
pub fn gpu_clear_cache() {
    GPU.with(|g| {
        if let Some(gpu) = g.borrow().as_ref() {
            gpu.clear_cache();
        }
    });
}
/// C[m,k] = A[m,n] @ B[k,n]^T on the GPU (matmul backward dA), reusing B's cached
/// forward buffer when `b_key` is set.
fn gmm_abt(a: &[f32], m: usize, n: usize, b: &[f32], k: usize, b_key: Option<u64>) -> Vec<f32> {
    GPU.with(|g| {
        g.borrow()
            .as_ref()
            .unwrap()
            .matmul_abt_keyed(a, m, n, b, k, b_key)
    })
}
/// C[k,n] = A[m,k]^T @ B[m,n] on the GPU (matmul/bmm backward dB), no CPU transpose.
fn gmm_atb(a: &[f32], k: usize, m: usize, b: &[f32], n: usize, a_key: Option<u64>) -> Vec<f32> {
    GPU.with(|g| {
        g.borrow()
            .as_ref()
            .unwrap()
            .matmul_atb_keyed(a, k, m, b, n, a_key)
    })
}

#[allow(clippy::too_many_arguments)]
fn gpu_conv3d_bn_relu(
    x: &[f32],
    w: &[f32],
    scale: &[f32],
    shift: &[f32],
    shape: [usize; 5],
    k: [usize; 3],
    stride: [usize; 3],
    pad: [usize; 3],
    cout: usize,
    w_key: u64,
) -> Vec<f32> {
    GPU.with(|g| {
        g.borrow()
            .as_ref()
            .unwrap()
            .conv3d_bn_relu_infer(x, w, scale, shift, shape, k, stride, pad, cout, w_key)
    })
}

#[allow(clippy::too_many_arguments)]
fn gpu_conv3d_bn_relu_backward(
    x: &[f32],
    w: &[f32],
    scale: &[f32],
    out: &[f32],
    gout: &[f32],
    shape: [usize; 5],
    k: [usize; 3],
    stride: [usize; 3],
    pad: [usize; 3],
    cout: usize,
    w_key: u64,
    need_dx: bool,
    need_dw: bool,
) -> (Vec<f32>, Vec<f32>) {
    GPU.with(|gpu| {
        gpu.borrow().as_ref().unwrap().conv3d_bn_relu_backward(
            x, w, scale, out, gout, shape, k, stride, pad, cout, w_key, need_dx, need_dw,
        )
    })
}

fn gpu_depthwise_conv3d(
    x: &[f32],
    w: &[f32],
    shape: [usize; 5],
    k: [usize; 3],
    stride: [usize; 3],
    pad: [usize; 3],
    w_key: u64,
) -> Vec<f32> {
    GPU.with(|g| {
        g.borrow()
            .as_ref()
            .unwrap()
            .depthwise_conv3d_forward(x, w, shape, k, stride, pad, w_key)
    })
}

fn gpu_depthwise_conv3d_backward(
    x: &[f32],
    w: &[f32],
    gout: &[f32],
    shape: [usize; 5],
    k: [usize; 3],
    stride: [usize; 3],
    pad: [usize; 3],
    w_key: u64,
    need_dx: bool,
    need_dw: bool,
) -> (Vec<f32>, Vec<f32>) {
    GPU.with(|gpu| {
        gpu.borrow()
            .as_ref()
            .unwrap()
            .depthwise_conv3d_backward(x, w, gout, shape, k, stride, pad, w_key, need_dx, need_dw)
    })
}
#[derive(Clone)]
pub struct Tensor(Rc<RefCell<Inner>>);

struct Inner {
    id: u64,
    data: Vec<f32>,
    shape: Vec<usize>,
    grad: Vec<f32>,
    requires_grad: bool,
    parents: Vec<Tensor>,
    // given this node's grad, scatter into parents' grads
    backward: Option<Box<dyn Fn(&[f32], &[Tensor])>>,
    // GPU-resident copy of `data`. When `Some`, IT is authoritative and `data`
    // may be stale until `realize_cpu()` syncs it back. Lets activations stay on
    // the GPU between ops instead of round-tripping. Always `None` off macOS.
    gpu: Option<GpuBuf>,
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Error function via Abramowitz & Stegun 7.1.26 (max abs error ~1.5e-7).
/// Rust std has no `erf`; computed in f64 for accuracy, returned as f32.
/// Used by `gelu_erf` for the exact erf-based GELU (matches PyTorch nn.GELU).
fn erf(x: f32) -> f32 {
    let x = x as f64;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * ax);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-ax * ax).exp();
    (sign * y) as f32
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: &[usize], requires_grad: bool) -> Tensor {
        assert_eq!(data.len(), numel(shape));
        let n = data.len();
        Tensor(Rc::new(RefCell::new(Inner {
            id: NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed),
            data,
            shape: shape.to_vec(),
            grad: vec![0.0; n],
            requires_grad,
            parents: Vec::new(),
            backward: None,
            gpu: None,
        })))
    }

    pub fn param(data: Vec<f32>, shape: &[usize]) -> Tensor {
        Tensor::new(data, shape, true)
    }
    pub fn constant(data: Vec<f32>, shape: &[usize]) -> Tensor {
        Tensor::new(data, shape, false)
    }

    /// Xavier-ish uniform init via a seeded xorshift (dependency-free).
    pub fn randn(shape: &[usize], seed: &mut u64) -> Tensor {
        let n = numel(shape);
        let fan = shape.last().copied().unwrap_or(1) as f32;
        let scale = (1.0 / fan).sqrt();
        let data = (0..n)
            .map(|_| {
                let mut x = *seed;
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                *seed = x;
                ((x >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
            })
            .collect();
        Tensor::param(data, shape)
    }

    pub fn shape(&self) -> Vec<usize> {
        self.0.borrow().shape.clone()
    }
    /// Bring CPU `data` up to date from the resident GPU buffer, if any, then
    /// clear residency (CPU becomes authoritative). Cheap no-op when not resident.
    /// Every CPU op must call this on its inputs before reading `.data`.
    fn realize_cpu(&self) {
        if self.0.borrow().gpu.is_none() {
            return;
        }
        let mut b = self.0.borrow_mut();
        if let Some(buf) = b.gpu.take() {
            let n = b.data.len();
            b.data = download_buf(&buf, n);
        }
    }

    pub fn data(&self) -> Vec<f32> {
        self.realize_cpu();
        self.0.borrow().data.clone()
    }
    fn add_grad(&self, g: &[f32]) {
        let mut b = self.0.borrow_mut();
        for i in 0..g.len() {
            b.grad[i] += g[i];
        }
    }
    fn req(&self) -> bool {
        self.0.borrow().requires_grad
    }
    fn build(
        data: Vec<f32>,
        shape: &[usize],
        parents: Vec<Tensor>,
        bw: impl Fn(&[f32], &[Tensor]) + 'static,
    ) -> Tensor {
        let req = parents.iter().any(|p| p.0.borrow().requires_grad);
        let out = Tensor::new(data, shape, req);
        if req {
            let mut b = out.0.borrow_mut();
            b.parents = parents;
            b.backward = Some(Box::new(bw));
        }
        out
    }

    // ---- ops ----

    /// 2D matmul: [m,k] @ [k,n] -> [m,n]
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let (a, b) = (self.0.borrow(), other.0.borrow());
        let (m, k) = (a.shape[0], a.shape[1]);
        let n = b.shape[1];
        assert_eq!(b.shape[0], k, "matmul shape mismatch");
        // Cache the GPU buffer for B iff it's a frozen weight (never mutated):
        // a base constant is identical every step, so upload it once (keyed by the
        // tensor's stable id).
        let b_key = if b.requires_grad { None } else { Some(b.id) };
        let out = if gpu_on() {
            gmm(&a.data, m, k, &b.data, n, b_key)
        } else {
            let mut out = vec![0.0; m * n];
            for i in 0..m {
                for p in 0..k {
                    let av = a.data[i * k + p];
                    for j in 0..n {
                        out[i * n + j] += av * b.data[p * n + j];
                    }
                }
            }
            out
        };
        drop(a);
        drop(b);
        Tensor::build(
            out,
            &[m, n],
            vec![self.clone(), other.clone()],
            move |g, par| {
                // Only compute the gradients that are actually needed. When the
                // weight is a frozen constant (PEFT base), skipping dB roughly
                // halves the backward cost.
                let need_a = par[0].req();
                let need_b = par[1].req();
                let a = par[0].0.borrow();
                let b = par[1].0.borrow();
                // dA = g[m,n] @ B^T[n,k] ;  dB = A^T[k,m] @ g[m,n]
                let da = if !need_a {
                    Vec::new()
                } else if gpu_on() {
                    // dA = g @ B^T, reading B's cached forward buffer (no transpose)
                    let bk = if b.requires_grad { None } else { Some(b.id) };
                    gmm_abt(g, m, n, &b.data, k, bk)
                } else {
                    let mut da = vec![0.0; m * k];
                    for i in 0..m {
                        for j in 0..n {
                            let gv = g[i * n + j];
                            for p in 0..k {
                                da[i * k + p] += gv * b.data[p * n + j];
                            }
                        }
                    }
                    da
                };
                let db = if !need_b {
                    Vec::new()
                } else if gpu_on() {
                    // dB = A^T @ g on the GPU (no CPU transpose)
                    let ak = if a.requires_grad { None } else { Some(a.id) };
                    gmm_atb(&a.data, k, m, g, n, ak)
                } else {
                    let mut db = vec![0.0; k * n];
                    for i in 0..m {
                        for p in 0..k {
                            let av = a.data[i * k + p];
                            for j in 0..n {
                                db[p * n + j] += av * g[i * n + j];
                            }
                        }
                    }
                    db
                };
                drop(a);
                drop(b);
                if need_a {
                    par[0].add_grad(&da);
                }
                if need_b {
                    par[1].add_grad(&db);
                }
            },
        )
    }

    /// add, with row-broadcast of a 1-D bias: [m,f] + [f] -> [m,f]
    pub fn add(&self, other: &Tensor) -> Tensor {
        let (a, b) = (self.0.borrow(), other.0.borrow());
        let broadcast = b.shape.len() == 1 && a.shape.len() == 2 && b.shape[0] == a.shape[1];
        let (m, f) = if broadcast {
            (a.shape[0], a.shape[1])
        } else {
            (a.data.len(), 1)
        };
        let mut out = a.data.clone();
        if broadcast {
            for i in 0..m {
                for j in 0..f {
                    out[i * f + j] += b.data[j];
                }
            }
        } else {
            for i in 0..out.len() {
                out[i] += b.data[i];
            }
        }
        let shape = a.shape.clone();
        drop(a);
        drop(b);
        Tensor::build(
            out,
            &shape,
            vec![self.clone(), other.clone()],
            move |g, par| {
                par[0].add_grad(g);
                if broadcast {
                    let mut db = vec![0.0; f];
                    for i in 0..m {
                        for j in 0..f {
                            db[j] += g[i * f + j];
                        }
                    }
                    par[1].add_grad(&db);
                } else {
                    par[1].add_grad(g);
                }
            },
        )
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        let a = self.0.borrow();
        let b = other.0.borrow();
        let out: Vec<f32> = a.data.iter().zip(&b.data).map(|(x, y)| x - y).collect();
        let shape = a.shape.clone();
        drop(a);
        drop(b);
        Tensor::build(
            out,
            &shape,
            vec![self.clone(), other.clone()],
            move |g, par| {
                par[0].add_grad(g);
                let neg: Vec<f32> = g.iter().map(|x| -x).collect();
                par[1].add_grad(&neg);
            },
        )
    }

    /// elementwise multiply (same shape)
    pub fn mul(&self, other: &Tensor) -> Tensor {
        let a = self.0.borrow();
        let b = other.0.borrow();
        let out: Vec<f32> = a.data.iter().zip(&b.data).map(|(x, y)| x * y).collect();
        let shape = a.shape.clone();
        drop(a);
        drop(b);
        Tensor::build(
            out,
            &shape,
            vec![self.clone(), other.clone()],
            move |g, par| {
                let a = par[0].0.borrow();
                let b = par[1].0.borrow();
                let ga: Vec<f32> = g.iter().zip(&b.data).map(|(x, y)| x * y).collect();
                let gb: Vec<f32> = g.iter().zip(&a.data).map(|(x, y)| x * y).collect();
                drop(a);
                drop(b);
                par[0].add_grad(&ga);
                par[1].add_grad(&gb);
            },
        )
    }

    pub fn relu(&self) -> Tensor {
        let a = self.0.borrow();
        let out: Vec<f32> = a.data.iter().map(|x| x.max(0.0)).collect();
        let shape = a.shape.clone();
        drop(a);
        Tensor::build(out, &shape, vec![self.clone()], move |g, par| {
            let a = par[0].0.borrow();
            let ga: Vec<f32> = g
                .iter()
                .zip(&a.data)
                .map(|(gv, x)| if *x > 0.0 { *gv } else { 0.0 })
                .collect();
            drop(a);
            par[0].add_grad(&ga);
        })
    }

    pub fn tanh(&self) -> Tensor {
        let a = self.0.borrow();
        let out: Vec<f32> = a.data.iter().map(|x| x.tanh()).collect();
        let shape = a.shape.clone();
        drop(a);
        let saved = out.clone();
        Tensor::build(out, &shape, vec![self.clone()], move |g, par| {
            let ga: Vec<f32> = g
                .iter()
                .zip(&saved)
                .map(|(gv, t)| gv * (1.0 - t * t))
                .collect();
            par[0].add_grad(&ga);
        })
    }

    /// GELU via the sigmoid approximation x·σ(1.702x) — cheap, accurate enough,
    /// and its derivative is closed-form in σ.
    pub fn gelu(&self) -> Tensor {
        let a = self.0.borrow();
        let xs = a.data.clone();
        let out: Vec<f32> = xs.iter().map(|x| x / (1.0 + (-1.702 * x).exp())).collect();
        let shape = a.shape.clone();
        drop(a);
        Tensor::build(out, &shape, vec![self.clone()], move |g, par| {
            let ga: Vec<f32> = g
                .iter()
                .zip(&xs)
                .map(|(gv, x)| {
                    let s = 1.0 / (1.0 + (-1.702 * x).exp());
                    gv * (s + x * 1.702 * s * (1.0 - s))
                })
                .collect();
            par[0].add_grad(&ga);
        })
    }

    /// Exact (erf) GELU: y = 0.5·x·(1 + erf(x/√2)). This matches PyTorch
    /// `nn.GELU` (the "gelu" variant), as opposed to the cheaper sigmoid/tanh
    /// approximation in `gelu`. Backward is the closed-form derivative
    /// 0.5·(1+erf(x/√2)) + x·φ(x), with φ(x)=exp(-x²/2)/√(2π) the std-normal pdf.
    pub fn gelu_erf(&self) -> Tensor {
        let a = self.0.borrow();
        let xs = a.data.clone();
        const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2; // 1/√2
        const INV_SQRT_2PI: f32 = 0.398_942_28_f32; // 1/√(2π)
        let out: Vec<f32> = xs
            .iter()
            .map(|&x| 0.5 * x * (1.0 + erf(x * INV_SQRT2)))
            .collect();
        let shape = a.shape.clone();
        drop(a);
        Tensor::build(out, &shape, vec![self.clone()], move |g, par| {
            let ga: Vec<f32> = g
                .iter()
                .zip(&xs)
                .map(|(gv, &x)| {
                    let cdf = 0.5 * (1.0 + erf(x * INV_SQRT2));
                    let pdf = INV_SQRT_2PI * (-0.5 * x * x).exp();
                    gv * (cdf + x * pdf)
                })
                .collect();
            par[0].add_grad(&ga);
        })
    }

    /// 1D valid convolution. self = input [L, cin]; weight [cout, k*cin]
    /// (each output channel is a flattened k×cin filter). Returns [L-k+1, cout].
    pub fn conv1d(&self, weight: &Tensor, k: usize, cin: usize, cout: usize) -> Tensor {
        let a = self.0.borrow();
        let w = weight.0.borrow();
        let l = a.shape[0];
        let lout = l - k + 1;
        let mut out = vec![0.0; lout * cout];
        for t in 0..lout {
            for co in 0..cout {
                let mut acc = 0.0;
                for kk in 0..k {
                    for ci in 0..cin {
                        acc += a.data[(t + kk) * cin + ci] * w.data[co * (k * cin) + kk * cin + ci];
                    }
                }
                out[t * cout + co] = acc;
            }
        }
        drop(a);
        drop(w);
        Tensor::build(
            out,
            &[lout, cout],
            vec![self.clone(), weight.clone()],
            move |g, par| {
                let a = par[0].0.borrow();
                let w = par[1].0.borrow();
                let mut di = vec![0.0; l * cin];
                let mut dw = vec![0.0; cout * k * cin];
                for t in 0..lout {
                    for co in 0..cout {
                        let gv = g[t * cout + co];
                        for kk in 0..k {
                            for ci in 0..cin {
                                di[(t + kk) * cin + ci] +=
                                    gv * w.data[co * (k * cin) + kk * cin + ci];
                                dw[co * (k * cin) + kk * cin + ci] +=
                                    gv * a.data[(t + kk) * cin + ci];
                            }
                        }
                    }
                }
                drop(a);
                drop(w);
                par[0].add_grad(&di);
                par[1].add_grad(&dw);
            },
        )
    }

    /// max over rows of [r, c] -> [1, c] (per-column max; gradient routes to argmax)
    pub fn max_rows(&self) -> Tensor {
        let a = self.0.borrow();
        let (r, c) = (a.shape[0], a.shape[1]);
        let mut out = vec![f32::NEG_INFINITY; c];
        let mut argmax = vec![0usize; c];
        for i in 0..r {
            for j in 0..c {
                let val = a.data[i * c + j];
                if val > out[j] {
                    out[j] = val;
                    argmax[j] = i;
                }
            }
        }
        drop(a);
        Tensor::build(out, &[1, c], vec![self.clone()], move |g, par| {
            let mut gi = vec![0.0; r * c];
            for j in 0..c {
                gi[argmax[j] * c + j] = g[j];
            }
            par[0].add_grad(&gi);
        })
    }

    /// 2D valid convolution. self = input [N, H, W, Cin] (NHWC); weight
    /// [Cout, Kh*Kw*Cin] (each output channel a flattened Kh×Kw×Cin filter).
    /// Returns [N, H-Kh+1, W-Kw+1, Cout]. Same flattened-filter convention as conv1d.
    pub fn conv2d(&self, weight: &Tensor, kh: usize, kw: usize, cout: usize) -> Tensor {
        let a = self.0.borrow();
        let wt = weight.0.borrow();
        let (n, h, w, cin) = (a.shape[0], a.shape[1], a.shape[2], a.shape[3]);
        let (ho, wo) = (h - kh + 1, w - kw + 1);
        let fsz = kh * kw * cin;
        let mut out = vec![0.0; n * ho * wo * cout];
        for ni in 0..n {
            for oh in 0..ho {
                for ow in 0..wo {
                    for co in 0..cout {
                        let mut acc = 0.0;
                        for ki in 0..kh {
                            for kj in 0..kw {
                                for ci in 0..cin {
                                    let iv = a.data[((ni * h + oh + ki) * w + ow + kj) * cin + ci];
                                    acc += iv * wt.data[co * fsz + (ki * kw + kj) * cin + ci];
                                }
                            }
                        }
                        out[((ni * ho + oh) * wo + ow) * cout + co] = acc;
                    }
                }
            }
        }
        drop(a);
        drop(wt);
        Tensor::build(
            out,
            &[n, ho, wo, cout],
            vec![self.clone(), weight.clone()],
            move |g, par| {
                let a = par[0].0.borrow();
                let wt = par[1].0.borrow();
                let mut gin = vec![0.0; a.data.len()];
                let mut gwt = vec![0.0; wt.data.len()];
                for ni in 0..n {
                    for oh in 0..ho {
                        for ow in 0..wo {
                            for co in 0..cout {
                                let go = g[((ni * ho + oh) * wo + ow) * cout + co];
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        for ci in 0..cin {
                                            let ii = ((ni * h + oh + ki) * w + ow + kj) * cin + ci;
                                            let wi = co * fsz + (ki * kw + kj) * cin + ci;
                                            gin[ii] += go * wt.data[wi];
                                            gwt[wi] += go * a.data[ii];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                drop(a);
                drop(wt);
                par[0].add_grad(&gin);
                par[1].add_grad(&gwt);
            },
        )
    }

    /// Non-overlapping 2D max pool. self = [N, H, W, C]; returns [N, H/kh, W/kw, C].
    pub fn maxpool2d(&self, kh: usize, kw: usize) -> Tensor {
        let a = self.0.borrow();
        let (n, h, w, c) = (a.shape[0], a.shape[1], a.shape[2], a.shape[3]);
        let (ho, wo) = (h / kh, w / kw);
        let mut out = vec![0.0; n * ho * wo * c];
        let mut argmax = vec![0usize; n * ho * wo * c]; // flat input index of each max
        for ni in 0..n {
            for oh in 0..ho {
                for ow in 0..wo {
                    for ci in 0..c {
                        let mut best = f32::NEG_INFINITY;
                        let mut bi = 0;
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let ii = ((ni * h + oh * kh + ki) * w + ow * kw + kj) * c + ci;
                                if a.data[ii] > best {
                                    best = a.data[ii];
                                    bi = ii;
                                }
                            }
                        }
                        let oi = ((ni * ho + oh) * wo + ow) * c + ci;
                        out[oi] = best;
                        argmax[oi] = bi;
                    }
                }
            }
        }
        let inlen = a.data.len();
        drop(a);
        Tensor::build(out, &[n, ho, wo, c], vec![self.clone()], move |g, par| {
            let mut gin = vec![0.0; inlen];
            for (oi, &ii) in argmax.iter().enumerate() {
                gin[ii] += g[oi];
            }
            par[0].add_grad(&gin);
        })
    }

    // ---- detector (conv/pool/shape) ops -------------------------------------
    // Layout conventions, identical to `conv2d`/`maxpool2d` above:
    //   * activations are NHWC: [N, H, W, C], row-major.
    //   * conv weight is [cout, kh*kw*cin_per_group], inner order (kh, kw, cin).
    //   * conv bias (when present) is [cout], broadcast over N,H,W.
    // These exist for INFERENCE of a fused detector (BN folded into conv), but
    // they still register a full, correct backward that mirrors the forward
    // index mapping — consistent with the rest of the engine.

    /// Grouped 2D convolution with stride + zero-padding (NHWC).
    ///
    /// Input  [N, H, W, Cin]; weight [Cout, kh*kw*(Cin/groups)] inner (kh,kw,cin);
    /// optional bias [Cout]. Output
    /// [N, (H+2*pad-kh)/stride + 1, (W+2*pad-kw)/stride + 1, Cout].
    /// `groups==1` is dense conv; `groups==Cin` (with Cout a multiple of Cin) is
    /// depthwise. Zero pad is applied on all four spatial sides.
    pub fn conv2d_spg(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        kh: usize,
        kw: usize,
        cout: usize,
        stride: usize,
        pad: usize,
        groups: usize,
    ) -> Tensor {
        let a = self.0.borrow();
        let wt = weight.0.borrow();
        let (n, h, w, cin) = (a.shape[0], a.shape[1], a.shape[2], a.shape[3]);
        assert!(cin % groups == 0, "conv2d_spg: cin not divisible by groups");
        assert!(
            cout % groups == 0,
            "conv2d_spg: cout not divisible by groups"
        );
        let cin_g = cin / groups;
        let cout_g = cout / groups;
        let fsz = kh * kw * cin_g;
        assert_eq!(
            wt.data.len(),
            cout * fsz,
            "conv2d_spg: weight size mismatch"
        );
        let ho = (h + 2 * pad - kh) / stride + 1;
        let wo = (w + 2 * pad - kw) / stride + 1;
        let has_bias = bias.is_some();
        let bias_data: Vec<f32> = match bias {
            Some(b) => {
                let bb = b.0.borrow();
                assert_eq!(bb.data.len(), cout, "conv2d_spg: bias size mismatch");
                bb.data.clone()
            }
            None => Vec::new(),
        };
        let mut out = vec![0.0; n * ho * wo * cout];
        for ni in 0..n {
            for oh in 0..ho {
                for ow in 0..wo {
                    for co in 0..cout {
                        let grp = co / cout_g;
                        let ci0 = grp * cin_g;
                        let mut acc = if has_bias { bias_data[co] } else { 0.0 };
                        for ki in 0..kh {
                            // signed input row
                            let ih = oh * stride + ki;
                            if ih < pad || ih >= h + pad {
                                continue;
                            }
                            let ih = ih - pad;
                            for kj in 0..kw {
                                let iw = ow * stride + kj;
                                if iw < pad || iw >= w + pad {
                                    continue;
                                }
                                let iw = iw - pad;
                                for cg in 0..cin_g {
                                    let iv = a.data[((ni * h + ih) * w + iw) * cin + ci0 + cg];
                                    let wv = wt.data[co * fsz + (ki * kw + kj) * cin_g + cg];
                                    acc += iv * wv;
                                }
                            }
                        }
                        out[((ni * ho + oh) * wo + ow) * cout + co] = acc;
                    }
                }
            }
        }
        let inlen = a.data.len();
        drop(a);
        drop(wt);
        let mut parents = vec![self.clone(), weight.clone()];
        if let Some(b) = bias {
            parents.push(b.clone());
        }
        Tensor::build(out, &[n, ho, wo, cout], parents, move |g, par| {
            let a = par[0].0.borrow();
            let wt = par[1].0.borrow();
            let mut gin = vec![0.0; inlen];
            let mut gwt = vec![0.0; wt.data.len()];
            let mut gb = vec![0.0; cout];
            for ni in 0..n {
                for oh in 0..ho {
                    for ow in 0..wo {
                        for co in 0..cout {
                            let grp = co / cout_g;
                            let ci0 = grp * cin_g;
                            let go = g[((ni * ho + oh) * wo + ow) * cout + co];
                            gb[co] += go;
                            for ki in 0..kh {
                                let ih = oh * stride + ki;
                                if ih < pad || ih >= h + pad {
                                    continue;
                                }
                                let ih = ih - pad;
                                for kj in 0..kw {
                                    let iw = ow * stride + kj;
                                    if iw < pad || iw >= w + pad {
                                        continue;
                                    }
                                    let iw = iw - pad;
                                    for cg in 0..cin_g {
                                        let ii = ((ni * h + ih) * w + iw) * cin + ci0 + cg;
                                        let wi = co * fsz + (ki * kw + kj) * cin_g + cg;
                                        gin[ii] += go * wt.data[wi];
                                        gwt[wi] += go * a.data[ii];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            drop(a);
            drop(wt);
            par[0].add_grad(&gin);
            par[1].add_grad(&gwt);
            if par.len() > 2 {
                par[2].add_grad(&gb);
            }
        })
    }

    /// Dense (groups=1) strided + padded 2D conv with optional bias. Generalizes
    /// `conv2d` (which equals `conv2d_sp(.., stride=1, pad=0, bias=None)`).
    pub fn conv2d_sp(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        kh: usize,
        kw: usize,
        cout: usize,
        stride: usize,
        pad: usize,
    ) -> Tensor {
        self.conv2d_spg(weight, bias, kh, kw, cout, stride, pad, 1)
    }

    /// Max pool with stride + padding (NHWC). Padded cells are treated as
    /// NEGATIVE INFINITY so they never win the max. With k=5, stride=1, pad=2
    /// (SPPF) the spatial dims are preserved. Output
    /// [N, (H+2*pad-k)/stride + 1, (W+2*pad-k)/stride + 1, C].
    pub fn maxpool2d_sp(&self, k: usize, stride: usize, pad: usize) -> Tensor {
        let a = self.0.borrow();
        let (n, h, w, c) = (a.shape[0], a.shape[1], a.shape[2], a.shape[3]);
        let ho = (h + 2 * pad - k) / stride + 1;
        let wo = (w + 2 * pad - k) / stride + 1;
        let mut out = vec![0.0; n * ho * wo * c];
        // Flat input index of each max; usize::MAX means "won from a padded
        // cell" (only possible if a whole window is padding) → no grad routed.
        let mut argmax = vec![usize::MAX; n * ho * wo * c];
        for ni in 0..n {
            for oh in 0..ho {
                for ow in 0..wo {
                    for ci in 0..c {
                        let mut best = f32::NEG_INFINITY;
                        let mut bi = usize::MAX;
                        for ki in 0..k {
                            let ih = oh * stride + ki;
                            if ih < pad || ih >= h + pad {
                                continue;
                            }
                            let ih = ih - pad;
                            for kj in 0..k {
                                let iw = ow * stride + kj;
                                if iw < pad || iw >= w + pad {
                                    continue;
                                }
                                let iw = iw - pad;
                                let ii = ((ni * h + ih) * w + iw) * c + ci;
                                if a.data[ii] > best {
                                    best = a.data[ii];
                                    bi = ii;
                                }
                            }
                        }
                        let oi = ((ni * ho + oh) * wo + ow) * c + ci;
                        out[oi] = best;
                        argmax[oi] = bi;
                    }
                }
            }
        }
        let inlen = a.data.len();
        drop(a);
        Tensor::build(out, &[n, ho, wo, c], vec![self.clone()], move |g, par| {
            let mut gin = vec![0.0; inlen];
            for (oi, &ii) in argmax.iter().enumerate() {
                if ii != usize::MAX {
                    gin[ii] += g[oi];
                }
            }
            par[0].add_grad(&gin);
        })
    }

    /// Nearest-neighbour 2x upsample in H and W (NHWC). Each input pixel is
    /// copied into a 2x2 output block. Output [N, 2H, 2W, C]. Matches
    /// `F.interpolate(scale_factor=2, mode='nearest')`.
    pub fn upsample_nearest2x(&self) -> Tensor {
        let a = self.0.borrow();
        let (n, h, w, c) = (a.shape[0], a.shape[1], a.shape[2], a.shape[3]);
        let (ho, wo) = (h * 2, w * 2);
        let mut out = vec![0.0; n * ho * wo * c];
        for ni in 0..n {
            for oh in 0..ho {
                let ih = oh / 2;
                for ow in 0..wo {
                    let iw = ow / 2;
                    for ci in 0..c {
                        out[((ni * ho + oh) * wo + ow) * c + ci] =
                            a.data[((ni * h + ih) * w + iw) * c + ci];
                    }
                }
            }
        }
        let inlen = a.data.len();
        drop(a);
        Tensor::build(out, &[n, ho, wo, c], vec![self.clone()], move |g, par| {
            let mut gin = vec![0.0; inlen];
            for ni in 0..n {
                for oh in 0..ho {
                    let ih = oh / 2;
                    for ow in 0..wo {
                        let iw = ow / 2;
                        for ci in 0..c {
                            gin[((ni * h + ih) * w + iw) * c + ci] +=
                                g[((ni * ho + oh) * wo + ow) * c + ci];
                        }
                    }
                }
            }
            par[0].add_grad(&gin);
        })
    }

    /// Concatenate NHWC tensors along the channel (last) dim. All inputs must
    /// share N, H, W. Output [N, H, W, sum(C_i)].
    pub fn concat_channels(tensors: &[&Tensor]) -> Tensor {
        assert!(!tensors.is_empty(), "concat_channels: empty input");
        let b0 = tensors[0].0.borrow();
        let (n, h, w) = (b0.shape[0], b0.shape[1], b0.shape[2]);
        drop(b0);
        let mut chans = Vec::with_capacity(tensors.len());
        let mut ctot = 0usize;
        for t in tensors {
            let b = t.0.borrow();
            assert_eq!(b.shape[0], n, "concat_channels: N mismatch");
            assert_eq!(b.shape[1], h, "concat_channels: H mismatch");
            assert_eq!(b.shape[2], w, "concat_channels: W mismatch");
            chans.push(b.shape[3]);
            ctot += b.shape[3];
        }
        let mut out = vec![0.0; n * h * w * ctot];
        let mut coff = 0usize;
        for (ti, t) in tensors.iter().enumerate() {
            let b = t.0.borrow();
            let ct = chans[ti];
            for sp in 0..(n * h * w) {
                for ci in 0..ct {
                    out[sp * ctot + coff + ci] = b.data[sp * ct + ci];
                }
            }
            coff += ct;
        }
        let parents: Vec<Tensor> = tensors.iter().map(|t| (*t).clone()).collect();
        let chans2 = chans.clone();
        Tensor::build(out, &[n, h, w, ctot], parents, move |g, par| {
            let mut coff = 0usize;
            for (ti, p) in par.iter().enumerate() {
                let ct = chans2[ti];
                let mut gin = vec![0.0; n * h * w * ct];
                for sp in 0..(n * h * w) {
                    for ci in 0..ct {
                        gin[sp * ct + ci] = g[sp * ctot + coff + ci];
                    }
                }
                p.add_grad(&gin);
                coff += ct;
            }
        })
    }

    /// Channel-concat convenience for two tensors (see `concat_channels`).
    pub fn cat_c(&self, other: &Tensor) -> Tensor {
        Tensor::concat_channels(&[self, other])
    }

    /// Concatenate tensors along their last dimension, for any rank. All leading
    /// dimensions must match. This is the channel concat used by NDHWC 3D-conv
    /// networks.
    pub fn concat_last(tensors: &[&Tensor]) -> Tensor {
        assert!(!tensors.is_empty(), "concat_last: empty input");
        let shape0 = tensors[0].shape();
        assert!(!shape0.is_empty(), "concat_last: scalar input");
        let rank = shape0.len();
        let outer: usize = shape0[..rank - 1].iter().product();
        let mut widths = Vec::with_capacity(tensors.len());
        let mut total = 0usize;
        for t in tensors {
            let sh = t.shape();
            assert_eq!(sh.len(), rank, "concat_last: rank mismatch");
            assert_eq!(
                &sh[..rank - 1],
                &shape0[..rank - 1],
                "concat_last: leading shape mismatch"
            );
            widths.push(sh[rank - 1]);
            total += sh[rank - 1];
        }
        let mut out = vec![0.0f32; outer * total];
        let mut offset = 0usize;
        for (ti, t) in tensors.iter().enumerate() {
            let data = t.data();
            let width = widths[ti];
            for row in 0..outer {
                out[row * total + offset..row * total + offset + width]
                    .copy_from_slice(&data[row * width..(row + 1) * width]);
            }
            offset += width;
        }
        let mut out_shape = shape0;
        out_shape[rank - 1] = total;
        let parents: Vec<Tensor> = tensors.iter().map(|t| (*t).clone()).collect();
        Tensor::build(out, &out_shape, parents, move |g, par| {
            let mut offset = 0usize;
            for (ti, p) in par.iter().enumerate() {
                let width = widths[ti];
                let mut gin = vec![0.0f32; outer * width];
                for row in 0..outer {
                    gin[row * width..(row + 1) * width]
                        .copy_from_slice(&g[row * total + offset..row * total + offset + width]);
                }
                p.add_grad(&gin);
                offset += width;
            }
        })
    }

    // ---- general N-d ops: arbitrary-rank movement / reduction / broadcasting,
    // so models aren't limited to the transformer-shaped ops. Index math is
    // explicit via row-major strides; backward mirrors the forward mapping.

    fn strides(shape: &[usize]) -> Vec<usize> {
        let nd = shape.len();
        let mut s = vec![1usize; nd];
        for i in (0..nd.saturating_sub(1)).rev() {
            s[i] = s[i + 1] * shape[i + 1];
        }
        s
    }

    fn permute_data(data: &[f32], shape: &[usize], axes: &[usize]) -> Vec<f32> {
        let nd = shape.len();
        let istr = Self::strides(shape);
        let osh: Vec<usize> = axes.iter().map(|&ax| shape[ax]).collect();
        let ostr = Self::strides(&osh);
        let mut out = vec![0.0; data.len()];
        for fi in 0..data.len() {
            let mut rem = fi;
            let mut ic = vec![0usize; nd];
            for d in 0..nd {
                ic[d] = rem / istr[d];
                rem %= istr[d];
            }
            let mut of = 0;
            for k in 0..nd {
                of += ic[axes[k]] * ostr[k];
            }
            out[of] = data[fi];
        }
        out
    }

    /// General axis permutation (N-d transpose). `axes` is a permutation of 0..ndim.
    pub fn permute(&self, axes: &[usize]) -> Tensor {
        let a = self.0.borrow();
        let osh: Vec<usize> = axes.iter().map(|&ax| a.shape[ax]).collect();
        let out = Self::permute_data(&a.data, &a.shape, axes);
        let ish = a.shape.clone();
        drop(a);
        let axv = axes.to_vec();
        Tensor::build(out, &osh, vec![self.clone()], move |g, par| {
            let nd = axv.len();
            let mut inv = vec![0usize; nd]; // grad routes through the inverse permutation
            for k in 0..nd {
                inv[axv[k]] = k;
            }
            let osh: Vec<usize> = axv.iter().map(|&ax| ish[ax]).collect();
            let gin = Self::permute_data(g, &osh, &inv);
            par[0].add_grad(&gin);
        })
    }

    /// Sum over one axis, dropping it: shape with `axis` removed (min rank 1).
    pub fn sum_axis(&self, axis: usize) -> Tensor {
        let a = self.0.borrow();
        let ish = a.shape.clone();
        let (out, osh) = Self::reduce_sum(&a.data, &ish, axis);
        drop(a);
        Tensor::build(out, &osh, vec![self.clone()], move |g, par| {
            let n = par[0].0.borrow().data.len();
            let istr = Self::strides(&ish);
            let kept: Vec<usize> = (0..ish.len()).filter(|&d| d != axis).collect();
            let mut osh: Vec<usize> = kept.iter().map(|&d| ish[d]).collect();
            if osh.is_empty() {
                osh.push(1);
            }
            let ostr = Self::strides(&osh);
            let mut ofull = vec![0usize; ish.len()];
            for (oi, &d) in kept.iter().enumerate() {
                ofull[d] = ostr[oi];
            }
            let mut gin = vec![0.0; n];
            for fi in 0..n {
                let mut rem = fi;
                let mut of = 0;
                for d in 0..ish.len() {
                    let c = rem / istr[d];
                    rem %= istr[d];
                    of += c * ofull[d];
                }
                gin[fi] += g[of];
            }
            par[0].add_grad(&gin);
        })
    }

    fn reduce_sum(data: &[f32], ish: &[usize], axis: usize) -> (Vec<f32>, Vec<usize>) {
        let istr = Self::strides(ish);
        let kept: Vec<usize> = (0..ish.len()).filter(|&d| d != axis).collect();
        let mut osh: Vec<usize> = kept.iter().map(|&d| ish[d]).collect();
        if osh.is_empty() {
            osh.push(1);
        }
        let ostr = Self::strides(&osh);
        let mut ofull = vec![0usize; ish.len()];
        for (oi, &d) in kept.iter().enumerate() {
            ofull[d] = ostr[oi];
        }
        let mut out = vec![0.0; osh.iter().product()];
        for fi in 0..data.len() {
            let mut rem = fi;
            let mut of = 0;
            for d in 0..ish.len() {
                let c = rem / istr[d];
                rem %= istr[d];
                of += c * ofull[d];
            }
            out[of] += data[fi];
        }
        (out, osh)
    }

    /// Broadcast to a target shape (numpy rules: align right; each source dim must
    /// equal the target or be 1). Backward sums grad back over the broadcast dims.
    pub fn broadcast_to(&self, target: &[usize]) -> Tensor {
        let a = self.0.borrow();
        let tnd = target.len();
        let snd = a.shape.len();
        let mut ssh = vec![1usize; tnd];
        for i in 0..snd {
            ssh[tnd - snd + i] = a.shape[i];
        }
        let base = Self::strides(&ssh);
        let sstr: Vec<usize> = (0..tnd)
            .map(|i| {
                if ssh[i] == 1 && target[i] != 1 {
                    0
                } else {
                    base[i]
                }
            })
            .collect();
        let tstr = Self::strides(target);
        let tsize: usize = target.iter().product();
        let mut out = vec![0.0; tsize];
        for ti in 0..tsize {
            let mut rem = ti;
            let mut si = 0;
            for d in 0..tnd {
                let c = rem / tstr[d];
                rem %= tstr[d];
                si += c * sstr[d];
            }
            out[ti] = a.data[si];
        }
        let inlen = a.data.len();
        drop(a);
        let tgt = target.to_vec();
        Tensor::build(out, target, vec![self.clone()], move |g, par| {
            let tstr = Self::strides(&tgt);
            let mut gin = vec![0.0; inlen];
            for ti in 0..g.len() {
                let mut rem = ti;
                let mut si = 0;
                for d in 0..tgt.len() {
                    let c = rem / tstr[d];
                    rem %= tstr[d];
                    si += c * sstr[d];
                }
                gin[si] += g[ti];
            }
            par[0].add_grad(&gin);
        })
    }

    /// 3D valid convolution. self = [N, D, H, W, Cin] (NDHWC); weight
    /// [Cout, Kd*Kh*Kw*Cin]. Returns [N, D-Kd+1, H-Kh+1, W-Kw+1, Cout].
    pub fn conv3d(&self, weight: &Tensor, kd: usize, kh: usize, kw: usize, cout: usize) -> Tensor {
        let a = self.0.borrow();
        let wt = weight.0.borrow();
        let (n, dd, h, w, cin) = (a.shape[0], a.shape[1], a.shape[2], a.shape[3], a.shape[4]);
        let (od, oh, ow) = (dd - kd + 1, h - kh + 1, w - kw + 1);
        let fsz = kd * kh * kw * cin;
        let idx = move |ni, z, y, x, ci| (((ni * dd + z) * h + y) * w + x) * cin + ci;
        let widx = move |co, a3, ki, kj, ci| co * fsz + ((a3 * kh + ki) * kw + kj) * cin + ci;
        let oidx = move |ni, z, y, x, co| (((ni * od + z) * oh + y) * ow + x) * cout + co;
        let mut out = vec![0.0; n * od * oh * ow * cout];
        for ni in 0..n {
            for z in 0..od {
                for y in 0..oh {
                    for x in 0..ow {
                        for co in 0..cout {
                            let mut acc = 0.0;
                            for a3 in 0..kd {
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        for ci in 0..cin {
                                            acc += a.data[idx(ni, z + a3, y + ki, x + kj, ci)]
                                                * wt.data[widx(co, a3, ki, kj, ci)];
                                        }
                                    }
                                }
                            }
                            out[oidx(ni, z, y, x, co)] = acc;
                        }
                    }
                }
            }
        }
        drop(a);
        drop(wt);
        Tensor::build(
            out,
            &[n, od, oh, ow, cout],
            vec![self.clone(), weight.clone()],
            move |g, par| {
                let a = par[0].0.borrow();
                let wt = par[1].0.borrow();
                let mut gin = vec![0.0; a.data.len()];
                let mut gwt = vec![0.0; wt.data.len()];
                for ni in 0..n {
                    for z in 0..od {
                        for y in 0..oh {
                            for x in 0..ow {
                                for co in 0..cout {
                                    let go = g[oidx(ni, z, y, x, co)];
                                    for a3 in 0..kd {
                                        for ki in 0..kh {
                                            for kj in 0..kw {
                                                for ci in 0..cin {
                                                    let ii = idx(ni, z + a3, y + ki, x + kj, ci);
                                                    let wi = widx(co, a3, ki, kj, ci);
                                                    gin[ii] += go * wt.data[wi];
                                                    gwt[wi] += go * a.data[ii];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                drop(a);
                drop(wt);
                par[0].add_grad(&gin);
                par[1].add_grad(&gwt);
            },
        )
    }

    /// Grouped 3D convolution with per-axis stride + zero-padding (NDHWC).
    ///
    /// Input  [N, D, H, W, Cin]; weight [Cout, kd*kh*kw*(Cin/groups)] inner
    /// order (kd,kh,kw,cin); optional bias [Cout]. Output
    /// [N, (D+2*pd-kd)/sd+1, (H+2*ph-kh)/sh+1, (W+2*pw-kw)/sw+1, Cout].
    /// `stride`/`pad` are per-axis [d,h,w]. `groups==1` is dense conv;
    /// `groups==Cin` is depthwise. Zero pad on all six faces. The 3D analogue of
    /// `conv2d_spg`; generalizes `conv3d` (stride 1, pad 0, groups 1).
    #[allow(clippy::too_many_arguments)]
    pub fn conv3d_spg(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        kd: usize,
        kh: usize,
        kw: usize,
        cout: usize,
        stride: [usize; 3],
        pad: [usize; 3],
        groups: usize,
    ) -> Tensor {
        let a = self.0.borrow();
        let wt = weight.0.borrow();
        let (n, dd, h, w, cin) = (a.shape[0], a.shape[1], a.shape[2], a.shape[3], a.shape[4]);
        assert!(cin % groups == 0, "conv3d_spg: cin not divisible by groups");
        assert!(
            cout % groups == 0,
            "conv3d_spg: cout not divisible by groups"
        );
        let (sd, sh, sw) = (stride[0], stride[1], stride[2]);
        let (pd, ph, pw) = (pad[0], pad[1], pad[2]);
        let cin_g = cin / groups;
        let cout_g = cout / groups;
        let fsz = kd * kh * kw * cin_g;
        assert_eq!(
            wt.data.len(),
            cout * fsz,
            "conv3d_spg: weight size mismatch"
        );
        let od = (dd + 2 * pd - kd) / sd + 1;
        let oh = (h + 2 * ph - kh) / sh + 1;
        let ow = (w + 2 * pw - kw) / sw + 1;
        let has_bias = bias.is_some();
        let bias_data: Vec<f32> = match bias {
            Some(b) => {
                let bb = b.0.borrow();
                assert_eq!(bb.data.len(), cout, "conv3d_spg: bias size mismatch");
                bb.data.clone()
            }
            None => Vec::new(),
        };
        let iidx = move |ni: usize, z: usize, y: usize, x: usize, ci: usize| {
            (((ni * dd + z) * h + y) * w + x) * cin + ci
        };
        let widx = move |co: usize, a3: usize, ki: usize, kj: usize, cg: usize| {
            co * fsz + ((a3 * kh + ki) * kw + kj) * cin_g + cg
        };
        let oidx = move |ni: usize, z: usize, y: usize, x: usize, co: usize| {
            (((ni * od + z) * oh + y) * ow + x) * cout + co
        };
        let mut out = vec![0.0; n * od * oh * ow * cout];
        for ni in 0..n {
            for oz in 0..od {
                for oy in 0..oh {
                    for ox in 0..ow {
                        for co in 0..cout {
                            let ci0 = (co / cout_g) * cin_g;
                            let mut acc = if has_bias { bias_data[co] } else { 0.0 };
                            for a3 in 0..kd {
                                let iz = oz * sd + a3;
                                if iz < pd || iz >= dd + pd {
                                    continue;
                                }
                                let iz = iz - pd;
                                for ki in 0..kh {
                                    let iy = oy * sh + ki;
                                    if iy < ph || iy >= h + ph {
                                        continue;
                                    }
                                    let iy = iy - ph;
                                    for kj in 0..kw {
                                        let ix = ox * sw + kj;
                                        if ix < pw || ix >= w + pw {
                                            continue;
                                        }
                                        let ix = ix - pw;
                                        for cg in 0..cin_g {
                                            let iv = a.data[iidx(ni, iz, iy, ix, ci0 + cg)];
                                            let wv = wt.data[widx(co, a3, ki, kj, cg)];
                                            acc += iv * wv;
                                        }
                                    }
                                }
                            }
                            out[oidx(ni, oz, oy, ox, co)] = acc;
                        }
                    }
                }
            }
        }
        let inlen = a.data.len();
        drop(a);
        drop(wt);
        let mut parents = vec![self.clone(), weight.clone()];
        if let Some(b) = bias {
            parents.push(b.clone());
        }
        Tensor::build(out, &[n, od, oh, ow, cout], parents, move |g, par| {
            let a = par[0].0.borrow();
            let wt = par[1].0.borrow();
            let mut gin = vec![0.0; inlen];
            let mut gwt = vec![0.0; wt.data.len()];
            let mut gb = vec![0.0; cout];
            for ni in 0..n {
                for oz in 0..od {
                    for oy in 0..oh {
                        for ox in 0..ow {
                            for co in 0..cout {
                                let ci0 = (co / cout_g) * cin_g;
                                let go = g[oidx(ni, oz, oy, ox, co)];
                                gb[co] += go;
                                for a3 in 0..kd {
                                    let iz = oz * sd + a3;
                                    if iz < pd || iz >= dd + pd {
                                        continue;
                                    }
                                    let iz = iz - pd;
                                    for ki in 0..kh {
                                        let iy = oy * sh + ki;
                                        if iy < ph || iy >= h + ph {
                                            continue;
                                        }
                                        let iy = iy - ph;
                                        for kj in 0..kw {
                                            let ix = ox * sw + kj;
                                            if ix < pw || ix >= w + pw {
                                                continue;
                                            }
                                            let ix = ix - pw;
                                            for cg in 0..cin_g {
                                                let ii = iidx(ni, iz, iy, ix, ci0 + cg);
                                                let wi = widx(co, a3, ki, kj, cg);
                                                gin[ii] += go * wt.data[wi];
                                                gwt[wi] += go * a.data[ii];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            drop(a);
            drop(wt);
            par[0].add_grad(&gin);
            par[1].add_grad(&gwt);
            if par.len() > 2 {
                par[2].add_grad(&gb);
            }
        })
    }

    /// Depthwise NDHWC Conv3D with trainable weights `[Cin, Kd*Kh*Kw]`.
    /// This is the fast path for depthwise 3D convolution. With Metal enabled and
    /// `HOS_METAL_DEPTHWISE` not set to `off`, forward/dInput/dWeight run in
    /// dedicated depthwise kernels; otherwise it falls back to grouped Conv3D.
    pub fn conv3d_depthwise_sp(
        &self,
        weight: &Tensor,
        kd: usize,
        kh: usize,
        kw: usize,
        stride: [usize; 3],
        pad: [usize; 3],
    ) -> Tensor {
        let sh = self.shape();
        assert_eq!(sh.len(), 5, "conv3d_depthwise_sp: input must be NDHWC");
        let cin = sh[4];
        assert_eq!(
            weight.data().len(),
            cin * kd * kh * kw,
            "conv3d_depthwise_sp: weight must be [Cin,Kd*Kh*Kw]"
        );
        let metal_on = gpu_on()
            && std::env::var("HOS_METAL_DEPTHWISE")
                .map(|v| v != "off" && v != "0")
                .unwrap_or(true);
        if !metal_on {
            return self.conv3d_spg(weight, None, kd, kh, kw, cin, stride, pad, cin);
        }
        let shape = [sh[0], sh[1], sh[2], sh[3], sh[4]];
        let od = (shape[1] + 2 * pad[0] - kd) / stride[0] + 1;
        let oh = (shape[2] + 2 * pad[1] - kh) / stride[1] + 1;
        let ow = (shape[3] + 2 * pad[2] - kw) / stride[2] + 1;
        self.realize_cpu();
        weight.realize_cpu();
        let a = self.0.borrow();
        let w = weight.0.borrow();
        let w_key = w.id;
        let out = gpu_depthwise_conv3d(&a.data, &w.data, shape, [kd, kh, kw], stride, pad, w_key);
        drop(a);
        drop(w);
        Tensor::build(
            out,
            &[shape[0], od, oh, ow, cin],
            vec![self.clone(), weight.clone()],
            move |g, par| {
                par[0].realize_cpu();
                par[1].realize_cpu();
                let a = par[0].0.borrow();
                let w = par[1].0.borrow();
                let need_dx = a.requires_grad;
                let need_dw = w.requires_grad;
                let (dx, dw) = gpu_depthwise_conv3d_backward(
                    &a.data,
                    &w.data,
                    g,
                    shape,
                    [kd, kh, kw],
                    stride,
                    pad,
                    w.id,
                    need_dx,
                    need_dw,
                );
                drop(a);
                drop(w);
                if need_dx {
                    par[0].add_grad(&dx);
                }
                if need_dw {
                    par[1].add_grad(&dw);
                }
            },
        )
    }

    /// Dense (groups=1) strided + padded 3D conv with optional bias. Generalizes
    /// `conv3d` (which equals `conv3d_sp(.., [1,1,1], [0,0,0], None)`).
    #[allow(clippy::too_many_arguments)]
    pub fn conv3d_sp(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        kd: usize,
        kh: usize,
        kw: usize,
        cout: usize,
        stride: [usize; 3],
        pad: [usize; 3],
    ) -> Tensor {
        self.conv3d_spg(weight, bias, kd, kh, kw, cout, stride, pad, 1)
    }

    /// Frozen inference-only NDHWC Conv3D + BatchNorm affine + ReLU. On Metal,
    /// this dispatches a direct fused kernel and never materializes im2col; the
    /// CPU fallback uses the verified unfold/matmul path. `scale` and `shift`
    /// are the folded inference-BN coefficients per output channel.
    #[allow(clippy::too_many_arguments)]
    pub fn conv3d_bn_relu_infer(
        &self,
        weight: &Tensor,
        scale: &[f32],
        shift: &[f32],
        k: [usize; 3],
        stride: [usize; 3],
        pad: [usize; 3],
        cout: usize,
    ) -> Tensor {
        let sh = self.shape();
        assert_eq!(sh.len(), 5, "conv3d_bn_relu_infer: input must be NDHWC");
        let shape = [sh[0], sh[1], sh[2], sh[3], sh[4]];
        let od = (shape[1] + 2 * pad[0] - k[0]) / stride[0] + 1;
        let oh = (shape[2] + 2 * pad[1] - k[1]) / stride[1] + 1;
        let ow = (shape[3] + 2 * pad[2] - k[2]) / stride[2] + 1;
        assert_eq!(scale.len(), cout);
        assert_eq!(shift.len(), cout);
        let out = if gpu_on() {
            self.realize_cpu();
            weight.realize_cpu();
            let a = self.0.borrow();
            let w = weight.0.borrow();
            gpu_conv3d_bn_relu(
                &a.data, &w.data, scale, shift, shape, k, stride, pad, cout, w.id,
            )
        } else {
            let cols = self.unfold3d(k[0], k[1], k[2], stride, pad);
            let raw = cols.matmul(weight).data();
            raw.into_iter()
                .enumerate()
                .map(|(i, v)| (v * scale[i % cout] + shift[i % cout]).max(0.0))
                .collect()
        };
        Tensor::constant(out, &[shape[0], od, oh, ow, cout])
    }

    /// Trainable direct NDHWC Conv3D with frozen inference-BN affine and ReLU.
    /// Metal uses direct forward/dInput/dWeight kernels without im2col. The CPU
    /// reference composes verified unfold3d + matmul autograd primitives.
    #[allow(clippy::too_many_arguments)]
    pub fn conv3d_bn_relu_train(
        &self,
        weight: &Tensor,
        scale: &[f32],
        shift: &[f32],
        k: [usize; 3],
        stride: [usize; 3],
        pad: [usize; 3],
        cout: usize,
    ) -> Tensor {
        let sh = self.shape();
        let shape = [sh[0], sh[1], sh[2], sh[3], sh[4]];
        let od = (shape[1] + 2 * pad[0] - k[0]) / stride[0] + 1;
        let oh = (shape[2] + 2 * pad[1] - k[1]) / stride[1] + 1;
        let ow = (shape[3] + 2 * pad[2] - k[2]) / stride[2] + 1;
        let out_shape = [shape[0], od, oh, ow, cout];
        if !gpu_on() {
            let raw = self
                .unfold3d(k[0], k[1], k[2], stride, pad)
                .matmul(weight)
                .reshape(&out_shape);
            let sc = Tensor::constant(scale.to_vec(), &[1, 1, 1, 1, cout]).broadcast_to(&out_shape);
            let bs = Tensor::constant(shift.to_vec(), &[1, 1, 1, 1, cout]).broadcast_to(&out_shape);
            return raw.mul(&sc).add(&bs).relu();
        }
        self.realize_cpu();
        weight.realize_cpu();
        let a = self.0.borrow();
        let w = weight.0.borrow();
        let w_key = w.id;
        let out = gpu_conv3d_bn_relu(
            &a.data, &w.data, scale, shift, shape, k, stride, pad, cout, w_key,
        );
        drop(a);
        drop(w);
        let saved_out = out.clone();
        let scale = scale.to_vec();
        Tensor::build(
            out,
            &out_shape,
            vec![self.clone(), weight.clone()],
            move |g, par| {
                par[0].realize_cpu();
                par[1].realize_cpu();
                let a = par[0].0.borrow();
                let w = par[1].0.borrow();
                let need_dx = a.requires_grad;
                let need_dw = w.requires_grad;
                let (dx, dw) = gpu_conv3d_bn_relu_backward(
                    &a.data, &w.data, &scale, &saved_out, g, shape, k, stride, pad, cout, w.id,
                    need_dx, need_dw,
                );
                drop(a);
                drop(w);
                if need_dx {
                    par[0].add_grad(&dx);
                }
                if need_dw {
                    par[1].add_grad(&dw);
                }
            },
        )
    }

    /// Asymmetric padding for NDHWC tensors. Backward crops the upstream
    /// gradient back to the unpadded input, preserving TensorFlow-SAME graphs.
    pub fn pad5d_ndhwc(&self, before: [usize; 3], after: [usize; 3], fill: f32) -> Tensor {
        let a = self.0.borrow();
        assert_eq!(a.shape.len(), 5, "pad5d_ndhwc expects rank 5");
        let (n, d, h, w, c) = (a.shape[0], a.shape[1], a.shape[2], a.shape[3], a.shape[4]);
        let (od, oh, ow) = (
            d + before[0] + after[0],
            h + before[1] + after[1],
            w + before[2] + after[2],
        );
        let mut out = vec![fill; n * od * oh * ow * c];
        for ni in 0..n {
            for z in 0..d {
                for y in 0..h {
                    let si = (((ni * d + z) * h + y) * w) * c;
                    let di =
                        (((ni * od + z + before[0]) * oh + y + before[1]) * ow + before[2]) * c;
                    out[di..di + w * c].copy_from_slice(&a.data[si..si + w * c]);
                }
            }
        }
        drop(a);
        Tensor::build(
            out,
            &[n, od, oh, ow, c],
            vec![self.clone()],
            move |g, par| {
                let mut gin = vec![0.0f32; n * d * h * w * c];
                for ni in 0..n {
                    for z in 0..d {
                        for y in 0..h {
                            let si = (((ni * od + z + before[0]) * oh + y + before[1]) * ow
                                + before[2])
                                * c;
                            let di = (((ni * d + z) * h + y) * w) * c;
                            gin[di..di + w * c].copy_from_slice(&g[si..si + w * c]);
                        }
                    }
                }
                par[0].add_grad(&gin);
            },
        )
    }

    /// 3D max pool with per-axis kernel / stride / padding (NDHWC). Padded cells
    /// are treated as NEGATIVE INFINITY so they never win the max (a window that
    /// is all padding routes no gradient). Output
    /// [N, (D+2*pd-kd)/sd+1, (H+2*ph-kh)/sh+1, (W+2*pw-kw)/sw+1, C].
    pub fn maxpool3d_sp(&self, k: [usize; 3], stride: [usize; 3], pad: [usize; 3]) -> Tensor {
        let a = self.0.borrow();
        let (n, dd, h, w, c) = (a.shape[0], a.shape[1], a.shape[2], a.shape[3], a.shape[4]);
        let (kd, kh, kw) = (k[0], k[1], k[2]);
        let (sd, sh, sw) = (stride[0], stride[1], stride[2]);
        let (pd, ph, pw) = (pad[0], pad[1], pad[2]);
        let od = (dd + 2 * pd - kd) / sd + 1;
        let oh = (h + 2 * ph - kh) / sh + 1;
        let ow = (w + 2 * pw - kw) / sw + 1;
        let iidx = move |ni: usize, z: usize, y: usize, x: usize, ci: usize| {
            (((ni * dd + z) * h + y) * w + x) * c + ci
        };
        let oidx = move |ni: usize, z: usize, y: usize, x: usize, ci: usize| {
            (((ni * od + z) * oh + y) * ow + x) * c + ci
        };
        let mut out = vec![0.0; n * od * oh * ow * c];
        let mut argmax = vec![usize::MAX; n * od * oh * ow * c];
        for ni in 0..n {
            for oz in 0..od {
                for oy in 0..oh {
                    for ox in 0..ow {
                        for ci in 0..c {
                            let mut best = f32::NEG_INFINITY;
                            let mut bi = usize::MAX;
                            for a3 in 0..kd {
                                let iz = oz * sd + a3;
                                if iz < pd || iz >= dd + pd {
                                    continue;
                                }
                                let iz = iz - pd;
                                for ki in 0..kh {
                                    let iy = oy * sh + ki;
                                    if iy < ph || iy >= h + ph {
                                        continue;
                                    }
                                    let iy = iy - ph;
                                    for kj in 0..kw {
                                        let ix = ox * sw + kj;
                                        if ix < pw || ix >= w + pw {
                                            continue;
                                        }
                                        let ix = ix - pw;
                                        let ii = iidx(ni, iz, iy, ix, ci);
                                        if a.data[ii] > best {
                                            best = a.data[ii];
                                            bi = ii;
                                        }
                                    }
                                }
                            }
                            let oi = oidx(ni, oz, oy, ox, ci);
                            out[oi] = if bi == usize::MAX { 0.0 } else { best };
                            argmax[oi] = bi;
                        }
                    }
                }
            }
        }
        let inlen = a.data.len();
        drop(a);
        Tensor::build(
            out,
            &[n, od, oh, ow, c],
            vec![self.clone()],
            move |g, par| {
                let mut gin = vec![0.0; inlen];
                for (oi, &ii) in argmax.iter().enumerate() {
                    if ii != usize::MAX {
                        gin[ii] += g[oi];
                    }
                }
                par[0].add_grad(&gin);
            },
        )
    }

    /// 3D average pool with per-axis kernel / stride / padding (NDHWC).
    /// Padded cells are excluded from the denominator, matching TensorFlow-style
    /// average pooling semantics for SAME padding. Output shape follows the
    /// explicit padded convolution formula.
    pub fn avgpool3d_sp(&self, k: [usize; 3], stride: [usize; 3], pad: [usize; 3]) -> Tensor {
        let a = self.0.borrow();
        let (n, dd, h, w, c) = (a.shape[0], a.shape[1], a.shape[2], a.shape[3], a.shape[4]);
        let (kd, kh, kw) = (k[0], k[1], k[2]);
        let (sd, sh, sw) = (stride[0], stride[1], stride[2]);
        let (pd, ph, pw) = (pad[0], pad[1], pad[2]);
        let od = (dd + 2 * pd - kd) / sd + 1;
        let oh = (h + 2 * ph - kh) / sh + 1;
        let ow = (w + 2 * pw - kw) / sw + 1;
        let iidx = move |ni: usize, z: usize, y: usize, x: usize, ci: usize| {
            (((ni * dd + z) * h + y) * w + x) * c + ci
        };
        let oidx = move |ni: usize, z: usize, y: usize, x: usize, ci: usize| {
            (((ni * od + z) * oh + y) * ow + x) * c + ci
        };
        let mut out = vec![0.0; n * od * oh * ow * c];
        let mut counts = vec![0u16; od * oh * ow];
        for oz in 0..od {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut cnt = 0u16;
                    for a3 in 0..kd {
                        let iz = oz * sd + a3;
                        if iz < pd || iz >= dd + pd {
                            continue;
                        }
                        for ki in 0..kh {
                            let iy = oy * sh + ki;
                            if iy < ph || iy >= h + ph {
                                continue;
                            }
                            for kj in 0..kw {
                                let ix = ox * sw + kj;
                                if ix < pw || ix >= w + pw {
                                    continue;
                                }
                                cnt += 1;
                            }
                        }
                    }
                    counts[(oz * oh + oy) * ow + ox] = cnt.max(1);
                }
            }
        }
        for ni in 0..n {
            for oz in 0..od {
                for oy in 0..oh {
                    for ox in 0..ow {
                        let inv = 1.0 / counts[(oz * oh + oy) * ow + ox] as f32;
                        for a3 in 0..kd {
                            let iz = oz * sd + a3;
                            if iz < pd || iz >= dd + pd {
                                continue;
                            }
                            let iz = iz - pd;
                            for ki in 0..kh {
                                let iy = oy * sh + ki;
                                if iy < ph || iy >= h + ph {
                                    continue;
                                }
                                let iy = iy - ph;
                                for kj in 0..kw {
                                    let ix = ox * sw + kj;
                                    if ix < pw || ix >= w + pw {
                                        continue;
                                    }
                                    let ix = ix - pw;
                                    for ci in 0..c {
                                        out[oidx(ni, oz, oy, ox, ci)] +=
                                            a.data[iidx(ni, iz, iy, ix, ci)] * inv;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let inlen = a.data.len();
        drop(a);
        Tensor::build(
            out,
            &[n, od, oh, ow, c],
            vec![self.clone()],
            move |g, par| {
                let mut gin = vec![0.0; inlen];
                for ni in 0..n {
                    for oz in 0..od {
                        for oy in 0..oh {
                            for ox in 0..ow {
                                let inv = 1.0 / counts[(oz * oh + oy) * ow + ox] as f32;
                                for a3 in 0..kd {
                                    let iz = oz * sd + a3;
                                    if iz < pd || iz >= dd + pd {
                                        continue;
                                    }
                                    let iz = iz - pd;
                                    for ki in 0..kh {
                                        let iy = oy * sh + ki;
                                        if iy < ph || iy >= h + ph {
                                            continue;
                                        }
                                        let iy = iy - ph;
                                        for kj in 0..kw {
                                            let ix = ox * sw + kj;
                                            if ix < pw || ix >= w + pw {
                                                continue;
                                            }
                                            let ix = ix - pw;
                                            for ci in 0..c {
                                                gin[iidx(ni, iz, iy, ix, ci)] +=
                                                    g[oidx(ni, oz, oy, ox, ci)] * inv;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                par[0].add_grad(&gin);
            },
        )
    }

    /// im2col for 3D conv (NDHWC): unfold every sliding window into a row so that
    /// a convolution becomes `unfold3d(x).matmul(W[k³·Cin, Cout])`. This routes the
    /// heavy `Cout` accumulation through HOS's optimized (GPU-capable) matmul
    /// instead of naive conv loop-nests. Input [N,D,H,W,Cin]; output
    /// [N·Od·Oh·Ow, kd·kh·kw·Cin] with column order (kd,kh,kw,cin) and zero-fill
    /// for padded taps. Backward is col2im (scatter-add).
    pub fn unfold3d(
        &self,
        kd: usize,
        kh: usize,
        kw: usize,
        stride: [usize; 3],
        pad: [usize; 3],
    ) -> Tensor {
        let a = self.0.borrow();
        let (n, dd, h, w, cin) = (a.shape[0], a.shape[1], a.shape[2], a.shape[3], a.shape[4]);
        let (sd, sh, sw) = (stride[0], stride[1], stride[2]);
        let (pd, ph, pw) = (pad[0], pad[1], pad[2]);
        let od = (dd + 2 * pd - kd) / sd + 1;
        let oh = (h + 2 * ph - kh) / sh + 1;
        let ow = (w + 2 * pw - kw) / sw + 1;
        let patch = kd * kh * kw * cin;
        let rows = n * od * oh * ow;
        let iidx = move |ni: usize, z: usize, y: usize, x: usize, ci: usize| {
            (((ni * dd + z) * h + y) * w + x) * cin + ci
        };
        // Decode a flat output-row index into (ni, oz, oy, ox).
        let decode = move |row: usize| -> (usize, usize, usize, usize) {
            let ox = row % ow;
            let t = row / ow;
            let oy = t % oh;
            let t = t / oh;
            let oz = t % od;
            let ni = t / od;
            (ni, oz, oy, ox)
        };
        let mut out = vec![0.0f32; rows * patch];
        {
            let adata: &[f32] = &a.data;
            // Rows are independent -> parallelize the im2col fill (the CPU cost).
            out.par_chunks_mut(patch)
                .enumerate()
                .for_each(|(row, rowbuf)| {
                    let (ni, oz, oy, ox) = decode(row);
                    for a3 in 0..kd {
                        let iz = oz * sd + a3;
                        let iz_ok = iz >= pd && iz < dd + pd;
                        let iz = iz.wrapping_sub(pd);
                        for ki in 0..kh {
                            let iy = oy * sh + ki;
                            let iy_ok = iy >= ph && iy < h + ph;
                            let iy = iy.wrapping_sub(ph);
                            for kj in 0..kw {
                                let ix = ox * sw + kj;
                                let ix_ok = ix >= pw && ix < w + pw;
                                let ix = ix.wrapping_sub(pw);
                                let cbase = ((a3 * kh + ki) * kw + kj) * cin;
                                if iz_ok && iy_ok && ix_ok {
                                    let ibase = iidx(ni, iz, iy, ix, 0);
                                    rowbuf[cbase..cbase + cin]
                                        .copy_from_slice(&adata[ibase..ibase + cin]);
                                }
                            }
                        }
                    }
                });
        }
        let inlen = a.data.len();
        drop(a);
        let rows_per_n = od * oh * ow;
        let in_per_n = dd * h * w * cin;
        Tensor::build(out, &[rows, patch], vec![self.clone()], move |g, par| {
            // col2im, parallelized over the BATCH axis: each sample's gradient
            // lands in a DISJOINT slice of gin, so no races and no reduction.
            // Within a sample, rows overlap in the input -> that part stays serial.
            let mut gin = vec![0.0f32; inlen];
            gin.par_chunks_mut(in_per_n)
                .enumerate()
                .for_each(|(ni, gslice)| {
                    for r in 0..rows_per_n {
                        let row = ni * rows_per_n + r;
                        let oz = r / (oh * ow);
                        let oy = (r / ow) % oh;
                        let ox = r % ow;
                        let rbase = row * patch;
                        for a3 in 0..kd {
                            let iz = oz * sd + a3;
                            let iz_ok = iz >= pd && iz < dd + pd;
                            let iz = iz.wrapping_sub(pd);
                            for ki in 0..kh {
                                let iy = oy * sh + ki;
                                let iy_ok = iy >= ph && iy < h + ph;
                                let iy = iy.wrapping_sub(ph);
                                for kj in 0..kw {
                                    let ix = ox * sw + kj;
                                    let ix_ok = ix >= pw && ix < w + pw;
                                    let ix = ix.wrapping_sub(pw);
                                    if iz_ok && iy_ok && ix_ok {
                                        let cbase = rbase + ((a3 * kh + ki) * kw + kj) * cin;
                                        let ibase = (((iz) * h + iy) * w + ix) * cin; // local to sample
                                        for c in 0..cin {
                                            gslice[ibase + c] += g[cbase + c];
                                        }
                                    }
                                }
                            }
                        }
                    }
                });
            par[0].add_grad(&gin);
        })
    }

    /// SiLU / swish: x * sigmoid(x)
    pub fn silu(&self) -> Tensor {
        let a = self.0.borrow();
        let out: Vec<f32> = a.data.iter().map(|x| x / (1.0 + (-x).exp())).collect();
        let shape = a.shape.clone();
        drop(a);
        Tensor::build(out, &shape, vec![self.clone()], move |g, par| {
            let a = par[0].0.borrow();
            let ga: Vec<f32> = g
                .iter()
                .zip(&a.data)
                .map(|(gv, x)| {
                    let s = 1.0 / (1.0 + (-x).exp());
                    gv * (s + x * s * (1.0 - s))
                })
                .collect();
            drop(a);
            par[0].add_grad(&ga);
        })
    }

    /// Logistic sigmoid, elementwise. Used by RGA's regulatory gates.
    pub fn sigmoid(&self) -> Tensor {
        let a = self.0.borrow();
        let out: Vec<f32> = a.data.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
        let shape = a.shape.clone();
        drop(a);
        let saved = out.clone();
        Tensor::build(out, &shape, vec![self.clone()], move |g, par| {
            let dx: Vec<f32> = saved
                .iter()
                .zip(g)
                .map(|(&s, &gv)| gv * s * (1.0 - s))
                .collect();
            par[0].add_grad(&dx);
        })
    }

    /// mean over all elements -> scalar [1]
    pub fn mean(&self) -> Tensor {
        let a = self.0.borrow();
        let n = a.data.len();
        let s: f32 = a.data.iter().sum::<f32>() / n as f32;
        drop(a);
        Tensor::build(vec![s], &[1], vec![self.clone()], move |g, par| {
            let scale = g[0] / n as f32;
            par[0].add_grad(&vec![scale; n]);
        })
    }

    pub fn square(&self) -> Tensor {
        self.mul(self)
    }

    /// multiply by a scalar constant
    pub fn scale(&self, c: f32) -> Tensor {
        let a = self.0.borrow();
        let out: Vec<f32> = a.data.iter().map(|x| x * c).collect();
        let shape = a.shape.clone();
        drop(a);
        Tensor::build(out, &shape, vec![self.clone()], move |g, par| {
            let gg: Vec<f32> = g.iter().map(|x| x * c).collect();
            par[0].add_grad(&gg);
        })
    }

    /// 2D transpose [m,n] -> [n,m]
    pub fn transpose(&self) -> Tensor {
        let a = self.0.borrow();
        let (m, n) = (a.shape[0], a.shape[1]);
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                out[j * m + i] = a.data[i * n + j];
            }
        }
        drop(a);
        Tensor::build(out, &[n, m], vec![self.clone()], move |g, par| {
            let mut gi = vec![0.0; m * n];
            for i in 0..m {
                for j in 0..n {
                    gi[i * n + j] = g[j * m + i];
                }
            }
            par[0].add_grad(&gi);
        })
    }

    /// reshape (same data, new shape) — identity gradient
    pub fn reshape(&self, shape: &[usize]) -> Tensor {
        let a = self.0.borrow();
        assert_eq!(numel(shape), a.data.len(), "reshape numel mismatch");
        let data = a.data.clone();
        drop(a);
        Tensor::build(data, shape, vec![self.clone()], move |g, par| {
            par[0].add_grad(g);
        })
    }

    /// swap the last two dims of a 3D tensor [b, m, n] -> [b, n, m]
    pub fn transpose_last2(&self) -> Tensor {
        let a = self.0.borrow();
        let (bs, m, n) = (a.shape[0], a.shape[1], a.shape[2]);
        let mut out = vec![0.0; bs * m * n];
        for bi in 0..bs {
            for i in 0..m {
                for j in 0..n {
                    out[bi * n * m + j * m + i] = a.data[bi * m * n + i * n + j];
                }
            }
        }
        drop(a);
        Tensor::build(out, &[bs, n, m], vec![self.clone()], move |g, par| {
            let mut gi = vec![0.0; bs * m * n];
            for bi in 0..bs {
                for i in 0..m {
                    for j in 0..n {
                        gi[bi * m * n + i * n + j] = g[bi * n * m + j * m + i];
                    }
                }
            }
            par[0].add_grad(&gi);
        })
    }

    /// batched matmul: self [b, m, k] @ other [b, k, n] -> [b, m, n]
    pub fn bmm(&self, other: &Tensor) -> Tensor {
        let a = self.0.borrow();
        let b = other.0.borrow();
        let (bs, m, k) = (a.shape[0], a.shape[1], a.shape[2]);
        let n = b.shape[2];
        assert_eq!(b.shape[0], bs);
        assert_eq!(b.shape[1], k, "bmm inner dim mismatch");
        let big = gpu_on() && m * k * n >= GPU_BMM_MIN;
        let mut out = vec![0.0; bs * m * n];
        if big {
            // each batch is an independent matmul on the GPU
            for bi in 0..bs {
                let a_s = &a.data[bi * m * k..(bi + 1) * m * k];
                let b_s = &b.data[bi * k * n..(bi + 1) * k * n];
                let c = gmm(a_s, m, k, b_s, n, None);
                out[bi * m * n..(bi + 1) * m * n].copy_from_slice(&c);
            }
        } else {
            for bi in 0..bs {
                let (ao, bo, oo) = (bi * m * k, bi * k * n, bi * m * n);
                for i in 0..m {
                    for p in 0..k {
                        let av = a.data[ao + i * k + p];
                        for j in 0..n {
                            out[oo + i * n + j] += av * b.data[bo + p * n + j];
                        }
                    }
                }
            }
        }
        drop(a);
        drop(b);
        Tensor::build(
            out,
            &[bs, m, n],
            vec![self.clone(), other.clone()],
            move |g, par| {
                let a = par[0].0.borrow();
                let b = par[1].0.borrow();
                let mut da = vec![0.0; bs * m * k];
                let mut db = vec![0.0; bs * k * n];
                if big {
                    for bi in 0..bs {
                        let a_s = &a.data[bi * m * k..(bi + 1) * m * k];
                        let b_s = &b.data[bi * k * n..(bi + 1) * k * n];
                        let g_s = &g[bi * m * n..(bi + 1) * m * n];
                        // dA[bi] = g @ B^T -> [m,k] ;  dB[bi] = A^T @ g -> [k,n]
                        da[bi * m * k..(bi + 1) * m * k]
                            .copy_from_slice(&gmm_abt(g_s, m, n, b_s, k, None));
                        db[bi * k * n..(bi + 1) * k * n]
                            .copy_from_slice(&gmm_atb(a_s, k, m, g_s, n, None));
                    }
                } else {
                    for bi in 0..bs {
                        let (ao, bo, oo) = (bi * m * k, bi * k * n, bi * m * n);
                        for i in 0..m {
                            for p in 0..k {
                                for j in 0..n {
                                    da[ao + i * k + p] +=
                                        g[oo + i * n + j] * b.data[bo + p * n + j];
                                    db[bo + p * n + j] +=
                                        a.data[ao + i * k + p] * g[oo + i * n + j];
                                }
                            }
                        }
                    }
                }
                drop(a);
                drop(b);
                par[0].add_grad(&da);
                par[1].add_grad(&db);
            },
        )
    }

    /// swap axes 1 and 2 of a 4D tensor [a,b,c,e] -> [a,c,b,e] (for multi-head)
    pub fn transpose12_4d(&self) -> Tensor {
        let a = self.0.borrow();
        let (d0, d1, d2, d3) = (a.shape[0], a.shape[1], a.shape[2], a.shape[3]);
        let mut out = vec![0.0; d0 * d1 * d2 * d3];
        for i in 0..d0 {
            for j in 0..d1 {
                for k in 0..d2 {
                    for l in 0..d3 {
                        out[((i * d2 + k) * d1 + j) * d3 + l] =
                            a.data[((i * d1 + j) * d2 + k) * d3 + l];
                    }
                }
            }
        }
        drop(a);
        Tensor::build(out, &[d0, d2, d1, d3], vec![self.clone()], move |g, par| {
            let mut gi = vec![0.0; d0 * d1 * d2 * d3];
            for i in 0..d0 {
                for j in 0..d1 {
                    for k in 0..d2 {
                        for l in 0..d3 {
                            gi[((i * d1 + j) * d2 + k) * d3 + l] =
                                g[((i * d2 + k) * d1 + j) * d3 + l];
                        }
                    }
                }
            }
            par[0].add_grad(&gi);
        })
    }

    /// Per-row-weighted softmax cross-entropy over [N, C] logits. `weights[i]`
    /// scales row i's loss and gradient; the total is normalized by the sum of
    /// weights (a weighted mean). Useful to counter per-class imbalance.
    pub fn cross_entropy_weighted(&self, targets: &[usize], weights: &[f32]) -> Tensor {
        let a = self.0.borrow();
        let (n, vocab) = (a.shape[0], a.shape[1]);
        assert_eq!(targets.len(), n, "ce_weighted: targets len");
        assert_eq!(weights.len(), n, "ce_weighted: weights len");
        let wsum: f32 = weights.iter().sum::<f32>().max(1e-8);
        let mut probs = vec![0.0; n * vocab];
        let mut loss = 0.0f32;
        for i in 0..n {
            let row = &a.data[i * vocab..i * vocab + vocab];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for j in 0..vocab {
                let e = (row[j] - mx).exp();
                probs[i * vocab + j] = e;
                sum += e;
            }
            for j in 0..vocab {
                probs[i * vocab + j] /= sum;
            }
            loss += weights[i] * -(probs[i * vocab + targets[i]].max(1e-12)).ln();
        }
        loss /= wsum;
        drop(a);
        let targets = targets.to_vec();
        let weights = weights.to_vec();
        Tensor::build(vec![loss], &[1], vec![self.clone()], move |g, par| {
            let mut gi = probs.clone();
            for i in 0..n {
                gi[i * vocab + targets[i]] -= 1.0;
                let s = g[0] * weights[i] / wsum;
                for j in 0..vocab {
                    gi[i * vocab + j] *= s;
                }
            }
            par[0].add_grad(&gi);
        })
    }

    /// Numerically stable binary cross-entropy with logits, averaged over all
    /// elements. `targets` (0.0/1.0) has one entry per element (row-major). For a
    /// logit z and target y: `max(z,0) - z*y + log(1+exp(-|z|))`, grad
    /// `(sigmoid(z) - y)/N`. Used for the hand (L/R) and impact (landed/missed) heads.
    pub fn bce_with_logits(&self, targets: &[f32]) -> Tensor {
        let a = self.0.borrow();
        let n = a.data.len();
        assert_eq!(targets.len(), n, "bce: targets len");
        let mut loss = 0.0f32;
        for i in 0..n {
            let z = a.data[i];
            let y = targets[i];
            loss += z.max(0.0) - z * y + (1.0 + (-z.abs()).exp()).ln();
        }
        loss /= n as f32;
        drop(a);
        let targets = targets.to_vec();
        Tensor::build(vec![loss], &[1], vec![self.clone()], move |g, par| {
            let a = par[0].0.borrow();
            let mut gi = vec![0.0; n];
            let scale = g[0] / n as f32;
            for i in 0..n {
                let s = 1.0 / (1.0 + (-a.data[i]).exp());
                gi[i] = (s - targets[i]) * scale;
            }
            drop(a);
            par[0].add_grad(&gi);
        })
    }

    /// column j of [n, c] -> [n, 1]
    pub fn col(&self, j: usize) -> Tensor {
        let a = self.0.borrow();
        let (n, c) = (a.shape[0], a.shape[1]);
        let out: Vec<f32> = (0..n).map(|i| a.data[i * c + j]).collect();
        drop(a);
        Tensor::build(out, &[n, 1], vec![self.clone()], move |g, par| {
            let mut gi = vec![0.0; n * c];
            for i in 0..n {
                gi[i * c + j] = g[i];
            }
            par[0].add_grad(&gi);
        })
    }

    /// broadcast-multiply: a[n,c] * b[n,1] -> [n,c]
    pub fn mul_broadcast(&self, b: &Tensor) -> Tensor {
        let a = self.0.borrow();
        let bb = b.0.borrow();
        let (n, c) = (a.shape[0], a.shape[1]);
        let mut out = vec![0.0; n * c];
        for i in 0..n {
            for j in 0..c {
                out[i * c + j] = a.data[i * c + j] * bb.data[i];
            }
        }
        drop(a);
        drop(bb);
        Tensor::build(
            out,
            &[n, c],
            vec![self.clone(), b.clone()],
            move |g, par| {
                let a = par[0].0.borrow();
                let bb = par[1].0.borrow();
                let mut da = vec![0.0; n * c];
                let mut db = vec![0.0; n];
                for i in 0..n {
                    for j in 0..c {
                        da[i * c + j] = g[i * c + j] * bb.data[i];
                        db[i] += g[i * c + j] * a.data[i * c + j];
                    }
                }
                drop(a);
                drop(bb);
                par[0].add_grad(&da);
                par[1].add_grad(&db);
            },
        )
    }

    /// embedding lookup: self is the table [vocab, d]; returns [ids.len(), d]
    pub fn embedding(&self, ids: &[usize]) -> Tensor {
        let a = self.0.borrow();
        let d = a.shape[1];
        let mut out = vec![0.0; ids.len() * d];
        for (i, &id) in ids.iter().enumerate() {
            out[i * d..i * d + d].copy_from_slice(&a.data[id * d..id * d + d]);
        }
        let vocab = a.shape[0];
        drop(a);
        let ids = ids.to_vec();
        Tensor::build(out, &[ids.len(), d], vec![self.clone()], move |g, par| {
            let mut gw = vec![0.0; vocab * d];
            for (i, &id) in ids.iter().enumerate() {
                for j in 0..d {
                    gw[id * d + j] += g[i * d + j];
                }
            }
            par[0].add_grad(&gw);
        })
    }

    /// row-wise softmax over the last dim of a 2D tensor [r, c]
    pub fn softmax_rows(&self) -> Tensor {
        let a = self.0.borrow();
        let (r, c) = (a.shape[0], a.shape[1]);
        let mut out = vec![0.0; r * c];
        for i in 0..r {
            let row = &a.data[i * c..i * c + c];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for j in 0..c {
                let e = (row[j] - mx).exp();
                out[i * c + j] = e;
                sum += e;
            }
            for j in 0..c {
                out[i * c + j] /= sum;
            }
        }
        drop(a);
        let probs = out.clone();
        Tensor::build(out, &[r, c], vec![self.clone()], move |g, par| {
            let mut gi = vec![0.0; r * c];
            for i in 0..r {
                let s = &probs[i * c..i * c + c];
                let go = &g[i * c..i * c + c];
                let dot: f32 = (0..c).map(|j| go[j] * s[j]).sum();
                for j in 0..c {
                    gi[i * c + j] = s[j] * (go[j] - dot);
                }
            }
            par[0].add_grad(&gi);
        })
    }

    /// RMSNorm over the last dim of [r, d], scaled by weight[d]
    pub fn rmsnorm(&self, weight: &Tensor) -> Tensor {
        self.rmsnorm_eps(weight, 1e-5)
    }

    /// RMSNorm with an explicit epsilon (config, not hardwired — Llama uses 1e-5,
    /// Qwen2 1e-6, etc.).
    pub fn rmsnorm_eps(&self, weight: &Tensor, eps: f32) -> Tensor {
        let a = self.0.borrow();
        let w = weight.0.borrow();
        let (r, d) = (a.shape[0], a.shape[1]);
        let mut out = vec![0.0; r * d];
        let mut invs = vec![0.0; r];
        for i in 0..r {
            let row = &a.data[i * d..i * d + d];
            let ss = row.iter().map(|x| x * x).sum::<f32>() / d as f32;
            let inv = 1.0 / (ss + eps).sqrt();
            invs[i] = inv;
            for j in 0..d {
                out[i * d + j] = row[j] * inv * w.data[j];
            }
        }
        drop(a);
        drop(w);
        Tensor::build(
            out,
            &[r, d],
            vec![self.clone(), weight.clone()],
            move |g, par| {
                let a = par[0].0.borrow();
                let w = par[1].0.borrow();
                let mut gx = vec![0.0; r * d];
                let mut gw = vec![0.0; d];
                for i in 0..r {
                    let x = &a.data[i * d..i * d + d];
                    let inv = invs[i];
                    let go = &g[i * d..i * d + d];
                    // sum_j g_j w_j x_j  (for the x_i correction term)
                    let sgwx: f32 = (0..d).map(|j| go[j] * w.data[j] * x[j]).sum();
                    for j in 0..d {
                        gx[i * d + j] =
                            go[j] * w.data[j] * inv - inv * inv * inv * x[j] / d as f32 * sgwx;
                        gw[j] += go[j] * x[j] * inv;
                    }
                }
                drop(a);
                drop(w);
                par[0].add_grad(&gx);
                par[1].add_grad(&gw);
            },
        )
    }

    /// LayerNorm over the last dim with the default eps=1e-5.
    pub fn layernorm_default(&self, weight: &Tensor, bias: &Tensor) -> Tensor {
        self.layernorm(weight, bias, 1e-5)
    }

    /// BatchNorm over the CHANNEL (last) axis of an NDHWC tensor: each channel is
    /// normalized using its mean/var over ALL other positions (N·D·H·W), then
    /// scaled/shifted by per-channel `gamma`/`beta`. Uses BATCH statistics (no
    /// running stats) — well-defined for any N since the reduction includes the
    /// spatial/temporal positions. Unlike per-sample LayerNorm, this couples
    /// samples across the batch, which prevents the "all outputs identical"
    /// collapse. Full BN backward w.r.t. x, gamma, beta.
    pub fn batchnorm(&self, gamma: &Tensor, beta: &Tensor, eps: f32) -> Tensor {
        let a = self.0.borrow();
        let shape = a.shape.clone();
        let c = *shape.last().unwrap();
        let m = a.data.len() / c;
        let mut mean = vec![0f32; c];
        for i in 0..m {
            for j in 0..c {
                mean[j] += a.data[i * c + j];
            }
        }
        for v in mean.iter_mut() {
            *v /= m as f32;
        }
        let mut var = vec![0f32; c];
        for i in 0..m {
            for j in 0..c {
                let d = a.data[i * c + j] - mean[j];
                var[j] += d * d;
            }
        }
        for v in var.iter_mut() {
            *v /= m as f32;
        }
        let invstd: Vec<f32> = (0..c).map(|j| 1.0 / (var[j] + eps).sqrt()).collect();
        let g = gamma.0.borrow();
        let b = beta.0.borrow();
        let mut out = vec![0f32; a.data.len()];
        let mut xhat = vec![0f32; a.data.len()];
        for i in 0..m {
            for j in 0..c {
                let xh = (a.data[i * c + j] - mean[j]) * invstd[j];
                xhat[i * c + j] = xh;
                out[i * c + j] = g.data[j] * xh + b.data[j];
            }
        }
        let gamma_data = g.data.clone();
        drop(a);
        drop(g);
        drop(b);
        Tensor::build(
            out,
            &shape,
            vec![self.clone(), gamma.clone(), beta.clone()],
            move |dy, par| {
                let mut dgamma = vec![0f32; c];
                let mut dbeta = vec![0f32; c];
                let mut sum_dxhat = vec![0f32; c];
                let mut sum_dxhat_xhat = vec![0f32; c];
                for i in 0..m {
                    for j in 0..c {
                        let d = dy[i * c + j];
                        dbeta[j] += d;
                        dgamma[j] += d * xhat[i * c + j];
                        let dxh = d * gamma_data[j];
                        sum_dxhat[j] += dxh;
                        sum_dxhat_xhat[j] += dxh * xhat[i * c + j];
                    }
                }
                let mut dx = vec![0f32; m * c];
                let mf = m as f32;
                for i in 0..m {
                    for j in 0..c {
                        let dxh = dy[i * c + j] * gamma_data[j];
                        // dx = invstd/M * (M*dxhat - Σdxhat - xhat*Σ(dxhat*xhat))
                        dx[i * c + j] = invstd[j] / mf
                            * (mf * dxh - sum_dxhat[j] - xhat[i * c + j] * sum_dxhat_xhat[j]);
                    }
                }
                par[0].add_grad(&dx);
                par[1].add_grad(&dgamma);
                par[2].add_grad(&dbeta);
            },
        )
    }

    /// LayerNorm over the last dimension (mean-centred, population variance),
    /// with per-element affine weight and bias — the standard transformer
    /// LayerNorm (as in BERT-style models). `self` is rows of length D = last dim
    /// (same row treatment as `rmsnorm_eps`); `weight` and `bias` are [D].
    ///   m = mean(x); var = mean((x-m)²); inv = 1/√(var+eps)
    ///   y_i = (x_i - m)·inv·w_i + b_i
    pub fn layernorm(&self, weight: &Tensor, bias: &Tensor, eps: f32) -> Tensor {
        let a = self.0.borrow();
        let w = weight.0.borrow();
        let b = bias.0.borrow();
        let (r, d) = (a.shape[0], a.shape[1]);
        let mut out = vec![0.0; r * d];
        // Per-row mean and inverse-std, cached for the backward pass.
        let mut means = vec![0.0; r];
        let mut invs = vec![0.0; r];
        for i in 0..r {
            let row = &a.data[i * d..i * d + d];
            let m = row.iter().sum::<f32>() / d as f32;
            let var = row.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / d as f32;
            let inv = 1.0 / (var + eps).sqrt();
            means[i] = m;
            invs[i] = inv;
            for j in 0..d {
                let xhat = (row[j] - m) * inv;
                out[i * d + j] = xhat * w.data[j] + b.data[j];
            }
        }
        drop(a);
        drop(w);
        drop(b);
        Tensor::build(
            out,
            &[r, d],
            vec![self.clone(), weight.clone(), bias.clone()],
            move |g, par| {
                let a = par[0].0.borrow();
                let w = par[1].0.borrow();
                let mut gx = vec![0.0; r * d];
                let mut gw = vec![0.0; d];
                let mut gb = vec![0.0; d];
                for i in 0..r {
                    let x = &a.data[i * d..i * d + d];
                    let m = means[i];
                    let inv = invs[i];
                    let go = &g[i * d..i * d + d];
                    // mean(gw) and mean(gw·xhat) over the row, where gw_j = g_j·w_j.
                    let mut mean_gw = 0.0f32;
                    let mut mean_gwxh = 0.0f32;
                    for j in 0..d {
                        let xhat = (x[j] - m) * inv;
                        let gwj = go[j] * w.data[j];
                        mean_gw += gwj;
                        mean_gwxh += gwj * xhat;
                    }
                    mean_gw /= d as f32;
                    mean_gwxh /= d as f32;
                    for j in 0..d {
                        let xhat = (x[j] - m) * inv;
                        let gwj = go[j] * w.data[j];
                        gx[i * d + j] = inv * (gwj - mean_gw - xhat * mean_gwxh);
                        gw[j] += go[j] * xhat;
                        gb[j] += go[j];
                    }
                }
                drop(a);
                drop(w);
                par[0].add_grad(&gx);
                par[1].add_grad(&gw);
                par[2].add_grad(&gb);
            },
        )
    }

    /// softmax cross-entropy: self = logits [n, vocab], targets = class id per row.
    /// Returns mean loss as a scalar [1].
    pub fn cross_entropy(&self, targets: &[usize]) -> Tensor {
        let a = self.0.borrow();
        let (n, vocab) = (a.shape[0], a.shape[1]);
        let mut probs = vec![0.0; n * vocab];
        let mut loss = 0.0f32;
        for i in 0..n {
            let row = &a.data[i * vocab..i * vocab + vocab];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for j in 0..vocab {
                let e = (row[j] - mx).exp();
                probs[i * vocab + j] = e;
                sum += e;
            }
            for j in 0..vocab {
                probs[i * vocab + j] /= sum;
            }
            loss += -(probs[i * vocab + targets[i]].max(1e-12)).ln();
        }
        loss /= n as f32;
        drop(a);
        let targets = targets.to_vec();
        Tensor::build(vec![loss], &[1], vec![self.clone()], move |g, par| {
            // dL/dlogits = (softmax - onehot) / n  * upstream
            let mut gi = probs.clone();
            for i in 0..n {
                gi[i * vocab + targets[i]] -= 1.0;
            }
            let scale = g[0] / n as f32;
            for v in gi.iter_mut() {
                *v *= scale;
            }
            par[0].add_grad(&gi);
        })
    }

    /// Rotary position embedding. `self` is `[rows, n_heads*head_dim]`; `pos[r]`
    /// is row r's absolute position. `neox` selects rotate-halves (Qwen/Gemma/Phi)
    /// vs interleaved-pairs (Llama/Mistral) — a configurable knob, not hardwired.
    /// RoPE is a fixed orthogonal rotation per (row, head, frequency), so the
    /// backward is the inverse rotation applied to the gradient.
    pub fn rope(
        &self,
        n_heads: usize,
        head_dim: usize,
        base: f32,
        neox: bool,
        pos: &[usize],
    ) -> Tensor {
        let a = self.0.borrow();
        let rows = a.shape[0];
        let width = a.shape[1];
        assert_eq!(width, n_heads * head_dim, "rope width mismatch");
        assert_eq!(pos.len(), rows, "rope needs one position per row");
        let half = head_dim / 2;
        // (sin, cos) per (row, frequency) — shared across heads
        let sc: Vec<(f32, f32)> = {
            let mut v = vec![(0.0f32, 1.0f32); rows * half];
            for r in 0..rows {
                for i in 0..half {
                    let freq = 1.0 / base.powf(2.0 * i as f32 / head_dim as f32);
                    v[r * half + i] = (pos[r] as f32 * freq).sin_cos();
                }
            }
            v
        };
        let mut out = a.data.clone();
        for r in 0..rows {
            for h in 0..n_heads {
                let off = r * width + h * head_dim;
                for i in 0..half {
                    let (s, c) = sc[r * half + i];
                    let (ia, ib) = if neox {
                        (i, i + half)
                    } else {
                        (2 * i, 2 * i + 1)
                    };
                    let (xa, xb) = (out[off + ia], out[off + ib]);
                    out[off + ia] = xa * c - xb * s;
                    out[off + ib] = xa * s + xb * c;
                }
            }
        }
        drop(a);
        Tensor::build(out, &[rows, width], vec![self.clone()], move |g, par| {
            let mut dx = vec![0.0; g.len()];
            for r in 0..rows {
                for h in 0..n_heads {
                    let off = r * width + h * head_dim;
                    for i in 0..half {
                        let (s, c) = sc[r * half + i];
                        let (ia, ib) = if neox {
                            (i, i + half)
                        } else {
                            (2 * i, 2 * i + 1)
                        };
                        let (ga, gb) = (g[off + ia], g[off + ib]);
                        // inverse rotation (R^T): [c, s; -s, c]
                        dx[off + ia] = ga * c + gb * s;
                        dx[off + ib] = -ga * s + gb * c;
                    }
                }
            }
            par[0].add_grad(&dx);
        })
    }

    /// Repeat each slice along dim 0 `g` times consecutively (repeat-interleave):
    /// `[n, ...]` -> `[n*g, ...]`, `out[i] = self[i / g]`. This is grouped-query
    /// attention's K/V head expansion. Backward sums each group's grads back.
    pub fn repeat_interleave_dim0(&self, g: usize) -> Tensor {
        let a = self.0.borrow();
        let d0 = a.shape[0];
        let block: usize = a.shape[1..].iter().product();
        let mut out = vec![0.0; d0 * g * block];
        for i in 0..d0 * g {
            let src = (i / g) * block;
            out[i * block..i * block + block].copy_from_slice(&a.data[src..src + block]);
        }
        let mut new_shape = a.shape.clone();
        new_shape[0] = d0 * g;
        drop(a);
        Tensor::build(out, &new_shape, vec![self.clone()], move |grad, par| {
            let mut din = vec![0.0; d0 * block];
            for i in 0..d0 * g {
                let dst = (i / g) * block;
                for j in 0..block {
                    din[dst + j] += grad[i * block + j];
                }
            }
            par[0].add_grad(&din);
        })
    }

    /// reverse-mode backward from this (scalar) node
    pub fn backward(&self) {
        // topological order via DFS over parents
        let mut topo: Vec<Tensor> = Vec::new();
        let mut visited: Vec<*const RefCell<Inner>> = Vec::new();
        fn dfs(t: &Tensor, topo: &mut Vec<Tensor>, visited: &mut Vec<*const RefCell<Inner>>) {
            let ptr = Rc::as_ptr(&t.0);
            if visited.contains(&ptr) {
                return;
            }
            visited.push(ptr);
            let parents = t.0.borrow().parents.clone();
            for p in &parents {
                dfs(p, topo, visited);
            }
            topo.push(t.clone());
        }
        dfs(self, &mut topo, &mut visited);

        // seed grad of the output (scalar) with 1
        {
            let mut b = self.0.borrow_mut();
            for g in b.grad.iter_mut() {
                *g = 1.0;
            }
        }
        // propagate in reverse topo order
        for node in topo.iter().rev() {
            let grad = node.0.borrow().grad.clone();
            let parents = node.0.borrow().parents.clone();
            let has_bw = node.0.borrow().backward.is_some();
            if has_bw {
                let b = node.0.borrow();
                if let Some(bw) = &b.backward {
                    bw(&grad, &parents);
                }
            }
        }
    }

    pub fn zero_grad(&self) {
        let mut b = self.0.borrow_mut();
        for g in b.grad.iter_mut() {
            *g = 0.0;
        }
    }

    /// SGD step on a parameter: data -= lr * grad
    pub fn sgd_step(&self, lr: f32) {
        let mut b = self.0.borrow_mut();
        for i in 0..b.data.len() {
            b.data[i] -= lr * b.grad[i];
        }
    }

    pub fn grad(&self) -> Vec<f32> {
        self.0.borrow().grad.clone()
    }
    /// data -= update[i]
    pub fn step_raw(&self, update: &[f32]) {
        let mut b = self.0.borrow_mut();
        for i in 0..update.len() {
            b.data[i] -= update[i];
        }
        let id = b.id;
        drop(b);
        gpu_evict(id);
    }
    /// overwrite the data (used when loading weights)
    pub fn set_data(&self, d: &[f32]) {
        let mut b = self.0.borrow_mut();
        b.data.copy_from_slice(d);
        let id = b.id;
        drop(b);
        gpu_evict(id); // a cached frozen buffer would now be stale
    }
}

/// AdamW optimizer — decoupled weight decay. Holds per-parameter moment state
/// (m, v) and a step counter; all of which can be checkpointed to `.hos`.
pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub wd: f32,
    pub t: u64,
    pub m: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}

impl AdamW {
    pub fn new(params: &[&Tensor], lr: f32, wd: f32) -> AdamW {
        let m = params.iter().map(|p| vec![0.0; p.data().len()]).collect();
        let v = params.iter().map(|p| vec![0.0; p.data().len()]).collect();
        AdamW {
            lr,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            wd,
            t: 0,
            m,
            v,
        }
    }

    /// One optimizer step. `decay[i]` = apply weight decay to params[i]
    /// (true for weight matrices, false for norms/biases per modern practice).
    pub fn step(&mut self, params: &[&Tensor], decay: &[bool]) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for (i, p) in params.iter().enumerate() {
            let g = p.grad();
            let data = p.data();
            let mi = &mut self.m[i];
            let vi = &mut self.v[i];
            let mut upd = vec![0.0; g.len()];
            for j in 0..g.len() {
                mi[j] = self.beta1 * mi[j] + (1.0 - self.beta1) * g[j];
                vi[j] = self.beta2 * vi[j] + (1.0 - self.beta2) * g[j] * g[j];
                let mh = mi[j] / bc1;
                let vh = vi[j] / bc2;
                let mut u = self.lr * mh / (vh.sqrt() + self.eps);
                if decay[i] {
                    u += self.lr * self.wd * data[j];
                }
                upd[j] = u;
            }
            p.step_raw(&upd);
        }
    }
}
