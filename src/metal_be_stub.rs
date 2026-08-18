//! CPU-only stub of the Metal backend for non-macOS targets (Windows/Linux/WASM).
//!
//! The GPU path is gated to `cfg(target_os = "macos")` everywhere it's *decided*
//! (the `use_gpu` flags), so on these targets a `Gpu`/`GpuRunner` is never
//! constructed and these methods are never called. They exist only so the rest
//! of the engine — which mentions `Weight::Gpu`, `Option<GpuRunner>`, etc. —
//! type-checks unchanged. Reaching any of them is a bug, hence the panics.

use crate::model::Model;

const NO_GPU: &str = "the GPU backend is macOS-only; this is a CPU-only build";

/// Placeholder for the GPU buffer handle off macOS — never constructed, since the
/// residency path is gated off. Exists so `Tensor` type-checks unchanged.
pub type GpuBuf = ();

/// Never reached off macOS (no resident buffers exist to download).
pub fn download_buf(_buf: &GpuBuf, _n: usize) -> Vec<f32> {
    unreachable!("{NO_GPU}")
}

/// A weight matrix that would be GPU-resident on macOS. Never built here.
#[derive(Default)]
pub struct GpuMatrix {
    pub n_rows: usize,
    pub in_dim: usize,
    pub ggml_type: u32,
}

/// Stub Metal context. `new` panics; it is only reached if the GPU gate is
/// bypassed, which cannot happen off macOS.
#[derive(Default)]
pub struct Gpu;

impl Gpu {
    pub fn new() -> Gpu {
        panic!("{NO_GPU}")
    }
    pub fn clear_cache(&self) {}
    pub fn matmul_f32(&self, _a: &[f32], _m: usize, _k: usize, _b: &[f32], _n: usize) -> Vec<f32> {
        unreachable!("{NO_GPU}")
    }
    pub fn upload_matrix(&self, _w: &[f32], _n_rows: usize, _in_dim: usize) -> GpuMatrix {
        unreachable!("{NO_GPU}")
    }
    pub fn upload_quant(
        &self,
        _bytes: &[u8],
        _ggml_type: u32,
        _n_rows: usize,
        _in_dim: usize,
    ) -> GpuMatrix {
        unreachable!("{NO_GPU}")
    }
    pub fn matvec_into(&self, _w: &GpuMatrix, _x: &[f32], _out: &mut [f32]) {
        unreachable!("{NO_GPU}")
    }
    pub fn matvec(&self, _w: &GpuMatrix, _x: &[f32]) -> Vec<f32> {
        unreachable!("{NO_GPU}")
    }
    pub fn matmul_q4k_prefill_into(
        &self,
        _w: &GpuMatrix,
        _x: &[f32],
        _ntok: usize,
        _out: &mut [f32],
    ) {
        unreachable!("{NO_GPU}")
    }
    // Batched/2-token/small K-quant matmuls used by qwen35's `Weight::matvec_batch`.
    // Only ever dispatched on a `Weight::Gpu`, which is never built off macOS — the
    // CPU path uses `cpu_matmat`. Present so the engine type-checks on Windows/Linux.
    pub fn matmul_q6k_prefill_into(
        &self,
        _w: &GpuMatrix,
        _x: &[f32],
        _ntok: usize,
        _out: &mut [f32],
    ) {
        unreachable!("{NO_GPU}")
    }
    pub fn matmul_q4k_small_into(
        &self,
        _w: &GpuMatrix,
        _x: &[f32],
        _ntok: usize,
        _out: &mut [f32],
    ) {
        unreachable!("{NO_GPU}")
    }
    pub fn matmul_q6k_small_into(
        &self,
        _w: &GpuMatrix,
        _x: &[f32],
        _ntok: usize,
        _out: &mut [f32],
    ) {
        unreachable!("{NO_GPU}")
    }
    pub fn matmul_q4k_2tok_into(&self, _w: &GpuMatrix, _x: &[f32], _out: &mut [f32]) {
        unreachable!("{NO_GPU}")
    }
    pub fn matmul_q6k_2tok_into(&self, _w: &GpuMatrix, _x: &[f32], _out: &mut [f32]) {
        unreachable!("{NO_GPU}")
    }
    // keyed GEMM entry points used by tensor.rs autograd (gated behind the GPU
    // residency flag, so unreachable off macOS — present only so the lib builds).
    pub fn matmul_f32_keyed(
        &self,
        _a: &[f32],
        _m: usize,
        _k: usize,
        _b: &[f32],
        _n: usize,
        _b_key: Option<u64>,
    ) -> Vec<f32> {
        unreachable!("{NO_GPU}")
    }
    pub fn matmul_abt_keyed(
        &self,
        _a: &[f32],
        _m: usize,
        _n: usize,
        _b: &[f32],
        _k: usize,
        _b_key: Option<u64>,
    ) -> Vec<f32> {
        unreachable!("{NO_GPU}")
    }
    pub fn matmul_atb_keyed(
        &self,
        _a: &[f32],
        _k: usize,
        _m: usize,
        _b: &[f32],
        _n: usize,
        _a_key: Option<u64>,
    ) -> Vec<f32> {
        unreachable!("{NO_GPU}")
    }
    #[allow(clippy::too_many_arguments)]
    pub fn conv3d_bn_relu_infer(
        &self,
        _x: &[f32],
        _w: &[f32],
        _scale: &[f32],
        _shift: &[f32],
        _shape: [usize; 5],
        _k: [usize; 3],
        _stride: [usize; 3],
        _pad: [usize; 3],
        _cout: usize,
        _w_key: u64,
    ) -> Vec<f32> {
        unreachable!("{NO_GPU}")
    }
    #[allow(clippy::too_many_arguments)]
    pub fn conv3d_bn_relu_backward(
        &self,
        _x: &[f32],
        _w: &[f32],
        _scale: &[f32],
        _out: &[f32],
        _gout: &[f32],
        _shape: [usize; 5],
        _k: [usize; 3],
        _stride: [usize; 3],
        _pad: [usize; 3],
        _cout: usize,
        _w_key: u64,
        _need_dx: bool,
        _need_dw: bool,
    ) -> (Vec<f32>, Vec<f32>) {
        unreachable!("{NO_GPU}")
    }
    #[allow(clippy::too_many_arguments)]
    pub fn depthwise_conv3d_forward(
        &self,
        _x: &[f32],
        _w: &[f32],
        _shape: [usize; 5],
        _k: [usize; 3],
        _stride: [usize; 3],
        _pad: [usize; 3],
        _w_key: u64,
    ) -> Vec<f32> {
        unreachable!("{NO_GPU}")
    }
    #[allow(clippy::too_many_arguments)]
    pub fn depthwise_conv3d_backward(
        &self,
        _x: &[f32],
        _w: &[f32],
        _gout: &[f32],
        _shape: [usize; 5],
        _k: [usize; 3],
        _stride: [usize; 3],
        _pad: [usize; 3],
        _w_key: u64,
        _need_dx: bool,
        _need_dw: bool,
    ) -> (Vec<f32>, Vec<f32>) {
        unreachable!("{NO_GPU}")
    }
    pub fn evict(&self, _key: u64) {
        unreachable!("{NO_GPU}")
    }
}

/// Stub resident runner. Never constructed off macOS.
pub struct GpuRunner;

impl GpuRunner {
    pub fn new(_gpu: Gpu, _model: &Model) -> GpuRunner {
        unreachable!("{NO_GPU}")
    }
    pub fn forward(&self, _model: &Model, _token: u32, _pos: usize) -> Vec<f32> {
        unreachable!("{NO_GPU}")
    }
    pub fn prefill_step(&self, _model: &Model, _token: u32, _pos: usize) {
        unreachable!("{NO_GPU}")
    }
    pub fn forward_prefill_gpu(
        &self,
        _model: &Model,
        _tokens: &[u32],
        _pos0: usize,
    ) -> Option<Vec<f32>> {
        unreachable!("{NO_GPU}")
    }
}

/// No-op autorelease wrapper off macOS (no Metal objects to drain).
pub fn autorelease<R>(f: impl FnOnce() -> R) -> R {
    f()
}
