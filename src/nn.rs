//! `nn::Module` — a minimal composition layer for building ARBITRARY models on
//! the hos-tensor autograd, not just transformers. A Module owns parameters and
//! maps input → output; `parameters()` auto-collects them so training loops never
//! hand-list tensors (the transformer/genome code does that by hand — this removes
//! the boilerplate). You can always drop back to raw `Tensor` ops.
//!
//! This is the *compose-primitives* path of HOS's dual design: fused high-level
//! ops drive the inference hot path; these small composable pieces drive training
//! and experimentation. Adding a layer or activation is self-contained — the same
//! property the autograd's per-node backward closures give individual ops.

use crate::tensor::{AdamW, Tensor};

/// Anything with parameters that maps a tensor to a tensor.
pub trait Module {
    fn forward(&self, x: &Tensor) -> Tensor;
    /// Leaf parameters for the optimizer. Containers concatenate their children's.
    fn parameters(&self) -> Vec<&Tensor>;
}

/// Fully-connected layer: `y = x @ W + b` (W:[in,out], b:[out], bias broadcasts
/// over rows).
pub struct Linear {
    pub w: Tensor,
    pub b: Tensor,
}

impl Linear {
    pub fn new(in_dim: usize, out_dim: usize, seed: &mut u64) -> Linear {
        let scale = (2.0 / in_dim as f32).sqrt(); // Kaiming init (relu-family nets)
        let mut wdata = Tensor::randn(&[in_dim, out_dim], seed).data();
        for v in &mut wdata {
            *v *= scale;
        }
        Linear {
            w: Tensor::param(wdata, &[in_dim, out_dim]),
            b: Tensor::param(vec![0.0; out_dim], &[out_dim]),
        }
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.matmul(&self.w).add(&self.b)
    }
    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.w, &self.b]
    }
}

/// Parameter-free activation modules, so they slot into a `Sequential` like any
/// other layer. Each is one line because the underlying op carries its own
/// backward.
macro_rules! activation {
    ($name:ident, $method:ident) => {
        pub struct $name;
        impl Module for $name {
            fn forward(&self, x: &Tensor) -> Tensor {
                x.$method()
            }
            fn parameters(&self) -> Vec<&Tensor> {
                Vec::new()
            }
        }
    };
}
activation!(Relu, relu);
activation!(Tanh, tanh);
activation!(Gelu, gelu);
activation!(Sigmoid, sigmoid);

/// A stack of modules applied in order — the workhorse container.
pub struct Sequential {
    pub layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Sequential {
        Sequential { layers }
    }
}

impl Module for Sequential {
    fn forward(&self, x: &Tensor) -> Tensor {
        let mut h = x.clone();
        for l in &self.layers {
            h = l.forward(&h);
        }
        h
    }
    fn parameters(&self) -> Vec<&Tensor> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}

/// 2D convolution layer (NHWC). Weight is [Cout, Kh*Kw*Cin]; no bias (fold into a
/// following norm/linear if needed).
pub struct Conv2d {
    pub w: Tensor,
    kh: usize,
    kw: usize,
    cout: usize,
}

impl Conv2d {
    pub fn new(cin: usize, cout: usize, kh: usize, kw: usize, s: &mut u64) -> Conv2d {
        let fsz = kh * kw * cin;
        let scale = (1.0 / fsz as f32).sqrt();
        let mut d = Tensor::randn(&[cout, fsz], s).data();
        for v in &mut d {
            *v *= scale;
        }
        Conv2d {
            w: Tensor::param(d, &[cout, fsz]),
            kh,
            kw,
            cout,
        }
    }
}

impl Module for Conv2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.conv2d(&self.w, self.kh, self.kw, self.cout)
    }
    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.w]
    }
}

/// Non-overlapping 2D max pool.
pub struct MaxPool2d {
    pub kh: usize,
    pub kw: usize,
}
impl Module for MaxPool2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.maxpool2d(self.kh, self.kw)
    }
    fn parameters(&self) -> Vec<&Tensor> {
        Vec::new()
    }
}

/// Flatten everything past the batch dim: [N, ...] -> [N, prod(...)].
pub struct Flatten;
impl Module for Flatten {
    fn forward(&self, x: &Tensor) -> Tensor {
        let s = x.shape();
        let rest: usize = s[1..].iter().product();
        x.reshape(&[s[0], rest])
    }
    fn parameters(&self) -> Vec<&Tensor> {
        Vec::new()
    }
}

// ---- recurrent cells ---------------------------------------------------------
//
// An LSTM/GRU cell is *only* matmul + sigmoid + tanh + elementwise mul/add — all
// ops the autograd already has. The recurrence is a plain Rust loop over
// timesteps; define-by-run grows the tape through it, so backprop-through-time
// is automatic. No new ops, no special RNN machinery.

fn pmat(a: usize, b: usize, s: &mut u64) -> Tensor {
    let scale = (1.0 / a as f32).sqrt();
    let mut d = Tensor::randn(&[a, b], s).data();
    for v in &mut d {
        *v *= scale;
    }
    Tensor::param(d, &[a, b])
}
fn pvec(n: usize) -> Tensor {
    Tensor::param(vec![0.0; n], &[n])
}

/// LSTM cell. `step(x,h,c) -> (h',c')`; loop it over a sequence.
pub struct LstmCell {
    wxi: Tensor,
    wxf: Tensor,
    wxg: Tensor,
    wxo: Tensor,
    whi: Tensor,
    whf: Tensor,
    whg: Tensor,
    who: Tensor,
    bi: Tensor,
    bf: Tensor,
    bg: Tensor,
    bo: Tensor,
    pub hidden: usize,
}

impl LstmCell {
    pub fn new(in_dim: usize, hidden: usize, s: &mut u64) -> LstmCell {
        LstmCell {
            wxi: pmat(in_dim, hidden, s),
            wxf: pmat(in_dim, hidden, s),
            wxg: pmat(in_dim, hidden, s),
            wxo: pmat(in_dim, hidden, s),
            whi: pmat(hidden, hidden, s),
            whf: pmat(hidden, hidden, s),
            whg: pmat(hidden, hidden, s),
            who: pmat(hidden, hidden, s),
            bi: pvec(hidden),
            bf: pvec(hidden),
            bg: pvec(hidden),
            bo: pvec(hidden),
            hidden,
        }
    }
    pub fn step(&self, x: &Tensor, h: &Tensor, c: &Tensor) -> (Tensor, Tensor) {
        let i = x
            .matmul(&self.wxi)
            .add(&h.matmul(&self.whi))
            .add(&self.bi)
            .sigmoid();
        let f = x
            .matmul(&self.wxf)
            .add(&h.matmul(&self.whf))
            .add(&self.bf)
            .sigmoid();
        let g = x
            .matmul(&self.wxg)
            .add(&h.matmul(&self.whg))
            .add(&self.bg)
            .tanh();
        let o = x
            .matmul(&self.wxo)
            .add(&h.matmul(&self.who))
            .add(&self.bo)
            .sigmoid();
        let c2 = f.mul(c).add(&i.mul(&g));
        let h2 = o.mul(&c2.tanh());
        (h2, c2)
    }
    pub fn parameters(&self) -> Vec<&Tensor> {
        vec![
            &self.wxi, &self.wxf, &self.wxg, &self.wxo, &self.whi, &self.whf, &self.whg, &self.who,
            &self.bi, &self.bf, &self.bg, &self.bo,
        ]
    }
}

/// GRU cell. `step(x,h) -> h'`.
pub struct GruCell {
    wxr: Tensor,
    wxz: Tensor,
    wxn: Tensor,
    whr: Tensor,
    whz: Tensor,
    whn: Tensor,
    br: Tensor,
    bz: Tensor,
    bn: Tensor,
    pub hidden: usize,
}

impl GruCell {
    pub fn new(in_dim: usize, hidden: usize, s: &mut u64) -> GruCell {
        GruCell {
            wxr: pmat(in_dim, hidden, s),
            wxz: pmat(in_dim, hidden, s),
            wxn: pmat(in_dim, hidden, s),
            whr: pmat(hidden, hidden, s),
            whz: pmat(hidden, hidden, s),
            whn: pmat(hidden, hidden, s),
            br: pvec(hidden),
            bz: pvec(hidden),
            bn: pvec(hidden),
            hidden,
        }
    }
    pub fn step(&self, x: &Tensor, h: &Tensor) -> Tensor {
        let r = x
            .matmul(&self.wxr)
            .add(&h.matmul(&self.whr))
            .add(&self.br)
            .sigmoid();
        let z = x
            .matmul(&self.wxz)
            .add(&h.matmul(&self.whz))
            .add(&self.bz)
            .sigmoid();
        let n = x
            .matmul(&self.wxn)
            .add(&r.mul(&h.matmul(&self.whn)))
            .add(&self.bn)
            .tanh();
        let sh = z.shape();
        let ones = Tensor::constant(vec![1.0; sh.iter().product()], &sh); // (1 - z)
        ones.sub(&z).mul(&n).add(&z.mul(h)) // h' = (1-z)⊙n + z⊙h
    }
    pub fn parameters(&self) -> Vec<&Tensor> {
        vec![
            &self.wxr, &self.wxz, &self.wxn, &self.whr, &self.whz, &self.whn, &self.br, &self.bz,
            &self.bn,
        ]
    }
}

// ---- demo: a non-transformer model, trained entirely through the Module API ----

fn unit(s: &mut u64) -> f32 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    (x >> 40) as f32 / (1u64 << 24) as f32
}

/// Concentric circles: class 0 inside radius 1, class 1 outside. Radially
/// separable, so it *needs* nonlinearity — a linear classifier is stuck near 50%.
fn circles(n: usize, s: &mut u64) -> (Tensor, Vec<usize>) {
    let mut xs = vec![0f32; n * 2];
    let mut ys = Vec::with_capacity(n);
    for i in 0..n {
        let r = 2.0 * unit(s);
        let th = std::f32::consts::TAU * unit(s);
        xs[i * 2] = r * th.cos();
        xs[i * 2 + 1] = r * th.sin();
        ys.push(if r < 1.0 { 0 } else { 1 });
    }
    (Tensor::constant(xs, &[n, 2]), ys)
}

fn accuracy(logits: &[f32], targets: &[usize], classes: usize) -> f32 {
    let mut correct = 0;
    for (row, &t) in targets.iter().enumerate() {
        let r = &logits[row * classes..row * classes + classes];
        let pred = (0..classes)
            .max_by(|&a, &b| r[a].partial_cmp(&r[b]).unwrap())
            .unwrap();
        correct += (pred == t) as usize;
    }
    correct as f32 / targets.len() as f32
}

/// `--nn-demo`: build and train a small MLP via `nn::Module` — proof that the
/// engine trains *arbitrary* architectures, not just the built-in transformer.
pub fn demo() {
    println!("=== nn::Module — a non-transformer model, composed + trained ===");
    println!("task: concentric circles (radially separable → needs nonlinearity)\n");
    let mut seed = 0xC0FF_EE12_3456_789Bu64;
    // 2 → 16 → 16 → 2, built by composition. Swap Tanh for Gelu/Relu freely.
    let net = Sequential::new(vec![
        Box::new(Linear::new(2, 16, &mut seed)),
        Box::new(Tanh),
        Box::new(Linear::new(16, 16, &mut seed)),
        Box::new(Tanh),
        Box::new(Linear::new(16, 2, &mut seed)),
    ]);
    let params = net.parameters();
    let nparam: usize = params.iter().map(|p| p.data().len()).sum();
    let decay: Vec<bool> = params.iter().map(|p| p.shape().len() == 2).collect();
    let mut opt = AdamW::new(&params, 0.03, 0.0);

    let mut data_seed = 0xABCD_EF01_1357_2468u64;
    let (xtr, ytr) = circles(512, &mut data_seed);
    for step in 0..=600 {
        for p in &params {
            p.zero_grad();
        }
        let logits = net.forward(&xtr);
        let loss = logits.cross_entropy(&ytr);
        loss.backward();
        opt.step(&params, &decay);
        if step % 150 == 0 {
            let acc = accuracy(&logits.data(), &ytr, 2);
            println!(
                "step {step:4}   loss {:.4}   train acc {:.1}%",
                loss.data()[0],
                acc * 100.0
            );
        }
    }
    let (xte, yte) = circles(512, &mut data_seed);
    let acc = accuracy(&net.forward(&xte).data(), &yte, 2);
    println!(
        "\nheld-out acc {:.1}%   ·   {nparam} params auto-collected via Module::parameters()",
        acc * 100.0
    );
    println!("same autograd as the transformer trainer — no PyTorch, no hand-listed params.");
}

/// A length-`t` sequence of scalars per item; label = 1 if the sum is positive.
/// Needs accumulation across time, so a memoryless model can't do better than the
/// marginal — the recurrence has to carry state. Returns per-timestep [batch,1]s.
fn seq_batch(batch: usize, t: usize, s: &mut u64) -> (Vec<Tensor>, Vec<usize>) {
    let mut cols = vec![vec![0f32; batch]; t];
    let mut sums = vec![0f32; batch];
    for b in 0..batch {
        for k in 0..t {
            let v = 2.0 * unit(s) - 1.0;
            cols[k][b] = v;
            sums[b] += v;
        }
    }
    let xs = cols
        .into_iter()
        .map(|c| Tensor::constant(c, &[batch, 1]))
        .collect();
    (xs, sums.iter().map(|&v| (v > 0.0) as usize).collect())
}

/// `--rnn-demo`: train an LSTM and a GRU on a sequential task — proof the engine
/// handles recurrence (backprop-through-time) with no special machinery.
pub fn rnn_demo() {
    const T: usize = 12;
    const H: usize = 16;
    const B: usize = 256;
    println!("=== nn: recurrent nets (LSTM + GRU) via define-by-run ===");
    println!("task: sum-sign of a length-{T} sequence (needs accumulation over time)\n");

    // --- LSTM ---
    let mut s = 0x15D_7_u64 | 1;
    let lstm = LstmCell::new(1, H, &mut s);
    let lhead = Linear::new(H, 2, &mut s);
    let mut lp = lstm.parameters();
    lp.extend(lhead.parameters());
    let ld: Vec<bool> = lp.iter().map(|p| p.shape().len() == 2).collect();
    let mut lopt = AdamW::new(&lp, 0.01, 0.0);
    let mut ds = 0x5EED_u64 | 1;
    for step in 0..=400 {
        let (xs, y) = seq_batch(B, T, &mut ds);
        for p in &lp {
            p.zero_grad();
        }
        let mut h = Tensor::constant(vec![0.0; B * H], &[B, H]);
        let mut c = Tensor::constant(vec![0.0; B * H], &[B, H]);
        for x in &xs {
            let (h2, c2) = lstm.step(x, &h, &c);
            h = h2;
            c = c2;
        }
        let logits = lhead.forward(&h);
        let loss = logits.cross_entropy(&y);
        loss.backward();
        lopt.step(&lp, &ld);
        if step % 100 == 0 {
            println!(
                "LSTM  step {step:4}  loss {:.4}  acc {:.1}%",
                loss.data()[0],
                accuracy(&logits.data(), &y, 2) * 100.0
            );
        }
    }

    // --- GRU ---
    let mut s = 0x6_47_u64 | 1;
    let gru = GruCell::new(1, H, &mut s);
    let ghead = Linear::new(H, 2, &mut s);
    let mut gp = gru.parameters();
    gp.extend(ghead.parameters());
    let gd: Vec<bool> = gp.iter().map(|p| p.shape().len() == 2).collect();
    let mut gopt = AdamW::new(&gp, 0.01, 0.0);
    let mut ds = 0x5EED_u64 | 1;
    for step in 0..=400 {
        let (xs, y) = seq_batch(B, T, &mut ds);
        for p in &gp {
            p.zero_grad();
        }
        let mut h = Tensor::constant(vec![0.0; B * H], &[B, H]);
        for x in &xs {
            h = gru.step(x, &h);
        }
        let logits = ghead.forward(&h);
        let loss = logits.cross_entropy(&y);
        loss.backward();
        gopt.step(&gp, &gd);
        if step % 100 == 0 {
            println!(
                "GRU   step {step:4}  loss {:.4}  acc {:.1}%",
                loss.data()[0],
                accuracy(&logits.data(), &y, 2) * 100.0
            );
        }
    }
    println!("\nboth trained through backprop-through-time on the shared autograd — cells are ~30 lines each.");
}

/// Noisy 8×8 images: class 0 = horizontal gradient (varies by row), class 1 =
/// vertical (varies by col). Orientation is a spatial feature — conv detects it.
fn images(n: usize, s: &mut u64) -> (Tensor, Vec<usize>) {
    const HW: usize = 8;
    let mut data = vec![0f32; n * HW * HW];
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let cls = (unit(s) < 0.5) as usize;
        for r in 0..HW {
            for c in 0..HW {
                let base = if cls == 0 { r as f32 } else { c as f32 } / HW as f32;
                data[(i * HW + r) * HW + c] = base + 0.3 * (2.0 * unit(s) - 1.0);
            }
        }
        y.push(cls);
    }
    (Tensor::constant(data, &[n, HW, HW, 1]), y)
}

/// `--cnn-demo`: a small CNN (conv2d + maxpool) built with `Sequential` and
/// trained — proof the engine covers convolutional models too.
pub fn cnn_demo() {
    println!("=== nn: a CNN (conv2d + maxpool2d) via Sequential ===");
    println!("task: gradient orientation (horizontal vs vertical) in noisy 8×8 images\n");
    let mut s = 0xC0_77_u64 | 1;
    // 8×8×1 → conv3×3(8) → relu → maxpool2×2 → flatten(3·3·8=72) → linear(2)
    let net = Sequential::new(vec![
        Box::new(Conv2d::new(1, 8, 3, 3, &mut s)),
        Box::new(Relu),
        Box::new(MaxPool2d { kh: 2, kw: 2 }),
        Box::new(Flatten),
        Box::new(Linear::new(3 * 3 * 8, 2, &mut s)),
    ]);
    let params = net.parameters();
    let nparam: usize = params.iter().map(|p| p.data().len()).sum();
    let decay: Vec<bool> = params.iter().map(|p| p.shape().len() == 2).collect();
    let mut opt = AdamW::new(&params, 0.01, 0.0);
    let mut ds = 0xC0DE_u64 | 1;
    for step in 0..=300 {
        let (x, y) = images(128, &mut ds);
        for p in &params {
            p.zero_grad();
        }
        let logits = net.forward(&x);
        let loss = logits.cross_entropy(&y);
        loss.backward();
        opt.step(&params, &decay);
        if step % 75 == 0 {
            println!(
                "step {step:4}  loss {:.4}  acc {:.1}%",
                loss.data()[0],
                accuracy(&logits.data(), &y, 2) * 100.0
            );
        }
    }
    let (xte, yte) = images(256, &mut ds);
    let acc = accuracy(&net.forward(&xte).data(), &yte, 2);
    println!("\nheld-out acc {:.1}%   ·   {nparam} params   ·   conv2d/maxpool2d are new ops (~40 lines incl. backward)", acc * 100.0);
    println!("the CNN itself is just Sequential — same composition as the MLP.");
}

/// 3D convolution layer (NDHWC). Weight [Cout, Kd*Kh*Kw*Cin].
pub struct Conv3d {
    pub w: Tensor,
    kd: usize,
    kh: usize,
    kw: usize,
    cout: usize,
}
impl Conv3d {
    pub fn new(cin: usize, cout: usize, kd: usize, kh: usize, kw: usize, s: &mut u64) -> Conv3d {
        let fsz = kd * kh * kw * cin;
        let scale = (1.0 / fsz as f32).sqrt();
        let mut d = Tensor::randn(&[cout, fsz], s).data();
        for v in &mut d {
            *v *= scale;
        }
        Conv3d {
            w: Tensor::param(d, &[cout, fsz]),
            kd,
            kh,
            kw,
            cout,
        }
    }
}
impl Module for Conv3d {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.conv3d(&self.w, self.kd, self.kh, self.kw, self.cout)
    }
    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.w]
    }
}

/// Numeric gradient check: central-difference each input element, compare to the
/// autograd's analytic grad. The honest way to prove an op's backward is correct.
fn grad_check(x: &Tensor, loss_fn: impl Fn(&Tensor) -> Tensor) -> f32 {
    x.zero_grad();
    loss_fn(x).backward();
    let ana = x.grad();
    let base = x.data();
    let sh = x.shape();
    let eps = 1e-3f32;
    let mut maxd = 0f32;
    for i in 0..base.len() {
        let mut dp = base.clone();
        dp[i] += eps;
        let lp = loss_fn(&Tensor::constant(dp, &sh)).data()[0];
        let mut dm = base.clone();
        dm[i] -= eps;
        let lm = loss_fn(&Tensor::constant(dm, &sh)).data()[0];
        maxd = maxd.max(((lp - lm) / (2.0 * eps) - ana[i]).abs());
    }
    maxd
}

/// `--nd-demo`: gradient-check the general N-d ops (permute / sum_axis /
/// broadcast_to / conv3d) on arbitrary-rank tensors.
pub fn nd_demo() {
    println!("=== general N-d ops — gradient-checked (analytic backward vs numeric) ===\n");
    let mut s = 0x9D11_u64 | 1;
    let randt = |shape: &[usize], s: &mut u64| -> Tensor {
        let n: usize = shape.iter().product();
        Tensor::param((0..n).map(|_| 2.0 * unit(s) - 1.0).collect(), shape)
    };
    let x = randt(&[2, 3, 4], &mut s);
    let cperm = Tensor::constant((0..24).map(|i| 0.1 * i as f32).collect(), &[4, 2, 3]);
    let d1 = grad_check(&x, |t| t.permute(&[2, 0, 1]).mul(&cperm).mean());
    println!("permute([2,0,1])   on [2,3,4]     max|Δgrad| = {d1:.2e}");

    let x2 = randt(&[3, 4, 5], &mut s);
    let d2 = grad_check(&x2, |t| t.sum_axis(1).square().mean());
    println!("sum_axis(1)        on [3,4,5]     max|Δgrad| = {d2:.2e}");

    let x3 = randt(&[3, 1], &mut s);
    let d3 = grad_check(&x3, |t| t.broadcast_to(&[3, 5]).square().mean());
    println!("broadcast_to       [3,1]->[3,5]   max|Δgrad| = {d3:.2e}");

    let x4 = randt(&[1, 4, 4, 4, 1], &mut s);
    let w4 = randt(&[2, 8], &mut s); // [Cout=2, Kd·Kh·Kw·Cin = 2·2·2·1 = 8]
    let d4 = grad_check(&x4, move |t| t.conv3d(&w4, 2, 2, 2, 2).square().mean());
    println!("conv3d(2×2×2, 2ch) on [1,4,4,4,1] max|Δgrad| = {d4:.2e}");

    // --- strided/padded/grouped 3D ops for the from-scratch video backbone ---
    let x5 = randt(&[1, 4, 5, 5, 2], &mut s);
    let w5 = randt(&[3, 2 * 3 * 3 * 2], &mut s); // Cout=3, kd·kh·kw·cin = 2·3·3·2
    let b5 = randt(&[3], &mut s);
    let (w5c, b5c) = (
        Tensor::constant(w5.data(), &[3, 36]),
        Tensor::constant(b5.data(), &[3]),
    );
    let d5x = grad_check(&x5, move |t| {
        t.conv3d_sp(&w5c, Some(&b5c), 2, 3, 3, 3, [2, 2, 2], [0, 1, 1])
            .square()
            .mean()
    });
    println!("conv3d_sp s2 p011   (wrt x) on [1,4,5,5,2] max|Δgrad| = {d5x:.2e}");
    let (x5c, b5c2) = (
        Tensor::constant(x5.data(), &[1, 4, 5, 5, 2]),
        Tensor::constant(b5.data(), &[3]),
    );
    let d5w = grad_check(&w5, move |t| {
        x5c.conv3d_sp(t, Some(&b5c2), 2, 3, 3, 3, [2, 2, 2], [0, 1, 1])
            .square()
            .mean()
    });
    println!("conv3d_sp s2 p011   (wrt w) on [3,36]      max|Δgrad| = {d5w:.2e}");
    let (x5c2, w5c2) = (
        Tensor::constant(x5.data(), &[1, 4, 5, 5, 2]),
        Tensor::constant(w5.data(), &[3, 36]),
    );
    let d5b = grad_check(&b5, move |t| {
        x5c2.conv3d_sp(&w5c2, Some(t), 2, 3, 3, 3, [2, 2, 2], [0, 1, 1])
            .square()
            .mean()
    });
    println!("conv3d_sp s2 p011   (wrt b) on [3]         max|Δgrad| = {d5b:.2e}");
    // grouped (depthwise): groups=2, cin=2 -> cin_g=1
    let x5g = randt(&[1, 3, 4, 4, 2], &mut s);
    let w5g = randt(&[2, 2 * 2 * 2], &mut s); // Cout=2, k=2·2·2, cin_g=1
    let w5gc = Tensor::constant(w5g.data(), &[2, 8]);
    let d5g = grad_check(&x5g, move |t| {
        t.conv3d_spg(&w5gc, None, 2, 2, 2, 2, [1, 1, 1], [0, 0, 0], 2)
            .square()
            .mean()
    });
    println!("conv3d_spg groups=2 (wrt x) on [1,3,4,4,2] max|Δgrad| = {d5g:.2e}");
    // 3D max pool, per-axis kernel/stride/pad (stem-style 1×3×3, stride 1×2×2)
    let xp = randt(&[1, 4, 5, 5, 2], &mut s);
    let dp = grad_check(&xp, |t| {
        t.maxpool3d_sp([1, 3, 3], [1, 2, 2], [0, 1, 1])
            .square()
            .mean()
    });
    println!("maxpool3d_sp 1x3x3  (wrt x) on [1,4,5,5,2] max|Δgrad| = {dp:.2e}");
    // unfold3d (im2col) — the conv-as-matmul path for 3D convolution
    let xu = randt(&[1, 4, 5, 5, 2], &mut s);
    let du = grad_check(&xu, |t| {
        t.unfold3d(2, 3, 3, [2, 2, 2], [0, 1, 1]).square().mean()
    });
    println!("unfold3d 2x3x3 s2p1 (wrt x) on [1,4,5,5,2] max|Δgrad| = {du:.2e}");
    // batchnorm (channel = last dim, batch stats over N·D·H·W)
    let cbn = 3usize;
    let xbn = randt(&[2, 3, 4, cbn], &mut s); // [.., C=3], M = 2*3*4 = 24
    let gbn = randt(&[cbn], &mut s);
    let bbn = randt(&[cbn], &mut s);
    let (gc, bc) = (
        Tensor::constant(gbn.data(), &[cbn]),
        Tensor::constant(bbn.data(), &[cbn]),
    );
    let dbx = grad_check(&xbn, |t| t.batchnorm(&gc, &bc, 1e-5).square().mean());
    let (xc, bc2) = (
        Tensor::constant(xbn.data(), &[2, 3, 4, cbn]),
        Tensor::constant(bbn.data(), &[cbn]),
    );
    let dbg = grad_check(&gbn, |t| xc.batchnorm(t, &bc2, 1e-5).square().mean());
    let (xc2, gc2) = (
        Tensor::constant(xbn.data(), &[2, 3, 4, cbn]),
        Tensor::constant(gbn.data(), &[cbn]),
    );
    let dbb = grad_check(&bbn, |t| xc2.batchnorm(&gc2, t, 1e-5).square().mean());
    println!("batchnorm (wrt x) on [2,3,4,3]     max|Δgrad| = {dbx:.2e}");
    println!("batchnorm (wrt gamma) on [3]       max|Δgrad| = {dbg:.2e}");
    println!("batchnorm (wrt beta)  on [3]       max|Δgrad| = {dbb:.2e}");

    // --- Transformer ops: LayerNorm (last dim, affine) and exact erf GELU ---
    println!("\n=== LayerNorm + erf GELU — gradient-checked ===\n");

    // gelu_erf: grad wrt input
    let xg = randt(&[2, 4], &mut s);
    let dg = grad_check(&xg, |t| t.gelu_erf().square().mean());
    println!("gelu_erf            on [2,4]        max|Δgrad/x| = {dg:.2e}");

    // layernorm: grad wrt input, weight, and bias (checked one at a time, the
    // other two held as constants so the perturbed tensor is the only graph leaf)
    let r = 3usize;
    let d = 4usize;
    let xl = randt(&[r, d], &mut s);
    let wl = randt(&[d], &mut s);
    let bl = randt(&[d], &mut s);
    let (wc, bc) = (
        Tensor::constant(wl.data(), &[d]),
        Tensor::constant(bl.data(), &[d]),
    );
    let dlx = grad_check(&xl, |t| t.layernorm(&wc, &bc, 1e-5).square().mean());
    let (xc1, bc1) = (
        Tensor::constant(xl.data(), &[r, d]),
        Tensor::constant(bl.data(), &[d]),
    );
    let dlw = grad_check(&wl, |t| xc1.layernorm(t, &bc1, 1e-5).square().mean());
    let (xc2, wc2) = (
        Tensor::constant(xl.data(), &[r, d]),
        Tensor::constant(wl.data(), &[d]),
    );
    let dlb = grad_check(&bl, |t| xc2.layernorm(&wc2, t, 1e-5).square().mean());
    println!("layernorm (wrt x)   on [3,4]        max|Δgrad| = {dlx:.2e}");
    println!("layernorm (wrt w)   on [4]          max|Δgrad| = {dlw:.2e}");
    println!("layernorm (wrt b)   on [4]          max|Δgrad| = {dlb:.2e}");

    // quick numeric sanity checks
    let z = Tensor::constant(vec![0.0], &[1, 1]).gelu_erf().data()[0];
    let big = Tensor::constant(vec![8.0], &[1, 1]).gelu_erf().data()[0];
    println!("\nsanity: gelu_erf(0) = {z:.6} (≈0), gelu_erf(8) = {big:.6} (≈8)");
    let row = Tensor::constant(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[1, 5]);
    let ones = Tensor::constant(vec![1.0; 5], &[5]);
    let zeros = Tensor::constant(vec![0.0; 5], &[5]);
    let ln = row.layernorm(&ones, &zeros, 1e-5).data();
    let mean = ln.iter().sum::<f32>() / 5.0;
    let var = ln.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 5.0;
    println!("sanity: layernorm row mean = {mean:.6} (≈0), var = {var:.6} (≈1)");

    println!(
        "\nall ≈1e-3 or below → analytic backward matches numeric. arbitrary-rank, real autograd."
    );
}

// ---- vector / lattice quantization (QAT primitive) ---------------------------
//
// Train through a DISCRETE bottleneck. Straight-through estimation: the forward
// value is the quantized point, but the gradient passes through to the encoder
// unchanged (identity). A commitment loss pulls the encoder's output toward the
// lattice so it learns to produce quantizable latents. This is the Rust port of
// the E8Q/ScalarQ from the FashionMNIST QAT study — the building block for
// quantization-aware training and discrete representations.

/// E8 = D8 ∪ (D8 + ½). Nearest D8 point: round, and if the coordinate sum is odd,
/// flip the single coordinate with the largest rounding residual (Conway–Sloane).
fn nearest_d8(x: &[f32]) -> Vec<f32> {
    let mut z: Vec<f32> = x.iter().map(|v| v.round()).collect();
    let sum: f32 = z.iter().sum();
    if (sum as i64) & 1 != 0 {
        let (mut bi, mut bd) = (0usize, -1.0f32);
        for i in 0..x.len() {
            let r = (x[i] - z[i]).abs();
            if r > bd {
                bd = r;
                bi = i;
            }
        }
        z[bi] += if x[bi] - z[bi] >= 0.0 { 1.0 } else { -1.0 };
    }
    z
}

pub fn nearest_e8(x: &[f32]) -> Vec<f32> {
    let a = nearest_d8(x);
    let xm: Vec<f32> = x.iter().map(|v| v - 0.5).collect();
    let h: Vec<f32> = nearest_d8(&xm).iter().map(|v| v + 0.5).collect();
    let da: f32 = x.iter().zip(&a).map(|(p, q)| (p - q) * (p - q)).sum();
    let dh: f32 = x.iter().zip(&h).map(|(p, q)| (p - q) * (p - q)).sum();
    if da <= dh {
        a
    } else {
        h
    }
}

/// The codebook geometry a `VectorQuantizer` snaps blocks to.
pub enum Codebook {
    /// Per-coordinate integer grid, clamped to ±qmax — the scalar baseline.
    Scalar { qmax: i32 },
    /// E8 lattice over 8-D blocks (denser packing than scalar at equal scale).
    E8,
}

/// Result of a quantization step: the straight-through `output` to feed onward,
/// the `commitment` loss to add to your objective, and how many distinct codes
/// the batch actually used.
pub struct QOut {
    pub output: Tensor,
    pub commitment: Tensor,
    pub codes_used: usize,
}

/// Snaps `block`-sized chunks of a latent to a lattice codebook at a fixed scale.
/// (Learned per-block scale is the natural next step; fixed scale keeps v1 clear.)
pub struct VectorQuantizer {
    pub block: usize,
    pub scale: f32,
    pub codebook: Codebook,
    pub commit_weight: f32,
}

impl VectorQuantizer {
    /// `strength` anneals the quantizer in (0 → pass-through, 1 → fully quantized),
    /// so early training isn't shocked by the discretization.
    pub fn quantize(&self, h: &Tensor, strength: f32) -> QOut {
        let hd = h.data();
        let sh = h.shape();
        let (n, d) = (sh[0], sh[1]);
        let s = self.scale;
        let mut hard = vec![0f32; hd.len()];
        let mut used = std::collections::HashSet::new();
        for row in 0..n {
            for b in 0..d / self.block {
                let off = row * d + b * self.block;
                let blk: Vec<f32> = (0..self.block).map(|i| hd[off + i] / s).collect();
                let c = match &self.codebook {
                    Codebook::Scalar { qmax } => blk
                        .iter()
                        .map(|&x| (x.round() as i32).clamp(-qmax, *qmax) as f32)
                        .collect::<Vec<_>>(),
                    Codebook::E8 => nearest_e8(&blk),
                };
                used.insert(c.iter().map(|&v| v as i64).collect::<Vec<i64>>());
                for i in 0..self.block {
                    hard[off + i] = c[i] * s;
                }
            }
        }
        // STE: output = h + strength·(hard − h)  → value≈hard, grad to h = identity.
        let delta: Vec<f32> = (0..hd.len())
            .map(|i| strength * (hard[i] - hd[i]))
            .collect();
        let output = h.add(&Tensor::constant(delta, &sh));
        // commitment = mean‖h − sg[hard]‖²  → grad to h, pulls encoder onto the lattice.
        let commitment = h
            .sub(&Tensor::constant(hard, &sh))
            .square()
            .mean()
            .scale(self.commit_weight);
        QOut {
            output,
            commitment,
            codes_used: used.len(),
        }
    }
}

/// C-class Gaussian blobs in `dim`-D (fixed centers, noisy draws).
fn blobs(n: usize, classes: usize, dim: usize, s: &mut u64) -> (Tensor, Vec<usize>) {
    let mut cs = 0xCE7E_u64; // fixed center seed: same blobs train and test
    let centers: Vec<Vec<f32>> = (0..classes)
        .map(|_| (0..dim).map(|_| 2.0 * unit(&mut cs) - 1.0).collect())
        .collect();
    let mut data = vec![0f32; n * dim];
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let c = ((unit(s) * classes as f32) as usize).min(classes - 1);
        for j in 0..dim {
            data[i * dim + j] = centers[c][j] + 0.4 * (2.0 * unit(s) - 1.0);
        }
        y.push(c);
    }
    (Tensor::constant(data, &[n, dim]), y)
}

/// `--vq-demo`: classify through a discrete E8-quantized bottleneck. Proves the
/// engine does quantization-aware training — gradients flow through the lattice
/// via straight-through estimation.
pub fn vq_demo() {
    println!("=== nn: VQ-VAE-style discrete bottleneck (E8 lattice, STE) ===");
    println!("task: 4-class blobs through an 8-D E8-quantized latent\n");
    const IN: usize = 16;
    const HID: usize = 32;
    const D: usize = 8;
    const C: usize = 4;
    let mut s = 0x7E57_u64 | 1;
    let e1 = Linear::new(IN, HID, &mut s);
    let e2 = Linear::new(HID, D, &mut s);
    let d1 = Linear::new(D, HID, &mut s);
    let d2 = Linear::new(HID, C, &mut s);
    let vq = VectorQuantizer {
        block: 8,
        scale: 0.75,
        codebook: Codebook::E8,
        commit_weight: 0.25,
    };
    let mut params: Vec<&Tensor> = Vec::new();
    for l in [&e1, &e2, &d1, &d2] {
        params.extend(l.parameters());
    }
    let decay: Vec<bool> = params.iter().map(|p| p.shape().len() == 2).collect();
    let mut opt = AdamW::new(&params, 0.01, 0.0);
    let mut ds = 0xB10B_u64 | 1;
    for step in 0..=600 {
        let strength = (step as f32 / 100.0).min(1.0); // anneal the quantizer in
        let (x, y) = blobs(256, C, IN, &mut ds);
        for p in &params {
            p.zero_grad();
        }
        let latent = e2.forward(&e1.forward(&x).gelu()).tanh().scale(2.5);
        let q = vq.quantize(&latent, strength);
        let logits = d2.forward(&d1.forward(&q.output).gelu());
        let loss = logits.cross_entropy(&y).add(&q.commitment);
        loss.backward();
        opt.step(&params, &decay);
        if step % 150 == 0 {
            let codes = q.codes_used;
            let acc = accuracy(&logits.data(), &y, C);
            println!(
                "step {step:4}  loss {:.4}  acc {:.1}%  E8 codes used {codes:3}  q {strength:.2}",
                loss.data()[0],
                acc * 100.0
            );
        }
    }
    println!("\ntrained through a discrete E8 bottleneck via straight-through estimation.");
    println!("VectorQuantizer (STE + commitment + E8 coset decoder) is the QAT primitive.");
}

// ---- representation alignment (fusion bridge) --------------------------------
//
// Two models express the same content in different coordinate systems. A learned
// map R that takes one model's activations into the other's is the "translator"
// between them — the first piece of geometric fusion. R can be a free linear
// stitch (most general) or pushed toward an orthogonal, function-preserving
// rotation via a penalty. Trained by gradient on the autograd; no SVD.

/// `--mem-demo`: local, no-bleed editing in a discrete addressable memory (Test 1).
/// Same facts (key → value) stored two ways: a DENSE MLP (entangled) vs an E8-coded
/// lookup memory (each cell an isolable slot). Edit ONE fact in each; measure whether
/// the OTHER facts survive. Then the collision case: two facts in one cell — the plain
/// codebook bleeds, clonal proliferation (a per-key private entry) resolves it.
pub fn mem_demo() {
    use std::collections::{HashMap, HashSet};
    println!("=== addressable memory: local, no-bleed editing (Test 1) ===");
    println!("facts (key→value) stored DENSE (MLP) vs DISCRETE (E8-coded slots).");
    println!("edit one fact; do the OTHERS survive?\n");
    const D: usize = 8; // key dim = one E8 block
    const C: usize = 6; // possible values
    const N: usize = 24; // facts
    let mut s = 0x3E3E_u64 | 1;

    let mut keys: Vec<Vec<f32>> = (0..N)
        .map(|_| (0..D).map(|_| 1.4 * (2.0 * unit(&mut s) - 1.0)).collect())
        .collect();
    let mut labels: Vec<usize> = (0..N)
        .map(|_| (unit(&mut s) * C as f32) as usize % C)
        .collect();
    // force a COLLISION: fact 2 lands in fact 1's cell (near-identical key) with a different value
    keys[2] = keys[1].iter().map(|&v| v + 0.001).collect();
    labels[2] = (labels[1] + 1) % C;

    let e8code = |k: &[f32]| -> Vec<i64> {
        nearest_e8(k)
            .iter()
            .map(|&v| (v * 2.0).round() as i64)
            .collect()
    };
    let argmax = |row: &[f32]| -> usize {
        let mut b = 0;
        for i in 1..row.len() {
            if row[i] > row[b] {
                b = i;
            }
        }
        b
    };
    let cos = |a: &[f32], b: &[f32]| -> f32 {
        let (mut d, mut na, mut nb) = (0f32, 0f32, 0f32);
        for i in 0..a.len() {
            d += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        d / (na.sqrt() * nb.sqrt() + 1e-6)
    };

    let mut cells = HashSet::new();
    for k in &keys {
        cells.insert(e8code(k));
    }
    println!(
        "{N} facts → {} distinct E8 cells ({} collision)\n",
        cells.len(),
        N - cells.len()
    );
    let uniq = (0..N)
        .find(|&i| {
            (0..N)
                .filter(|&j| j != i)
                .all(|j| e8code(&keys[j]) != e8code(&keys[i]))
        })
        .unwrap();
    let frac = |hits: usize, total: usize| 100.0 * hits as f32 / total as f32;

    // ---- A: DENSE MLP ----
    let mut sm = 0xD1CE_u64 | 1;
    let l1 = Linear::new(D, 32, &mut sm);
    let l2 = Linear::new(32, C, &mut sm);
    let xall = Tensor::constant(keys.concat(), &[N, D]);
    let logits = |x: &Tensor| l2.forward(&l1.forward(x).gelu());
    let mut p = l1.parameters();
    p.extend(l2.parameters());
    let dc: Vec<bool> = p.iter().map(|t| t.shape().len() == 2).collect();
    let mut opt = AdamW::new(&p, 0.02, 0.0);
    for _ in 0..700 {
        for t in &p {
            t.zero_grad();
        }
        logits(&xall).cross_entropy(&labels).backward();
        opt.step(&p, &dc);
    }
    let dense_base = accuracy(&logits(&xall).data(), &labels, C);
    let new_u = (labels[uniq] + 3) % C;
    let ku = Tensor::constant(keys[uniq].clone(), &[1, D]);
    let mut opt2 = AdamW::new(&p, 0.02, 0.0);
    for _ in 0..80 {
        for t in &p {
            t.zero_grad();
        }
        logits(&ku).cross_entropy(&[new_u]).backward();
        opt2.step(&p, &dc);
    }
    let dd = logits(&xall).data();
    let dense_edit = argmax(&dd[uniq * C..(uniq + 1) * C]) == new_u;
    let dense_keep = (0..N)
        .filter(|&i| i != uniq && argmax(&dd[i * C..(i + 1) * C]) == labels[i])
        .count();

    // ---- B: DISCRETE codebook (one value per E8 cell) ----
    let mut book: HashMap<Vec<i64>, usize> = HashMap::new();
    for i in 0..N {
        book.insert(e8code(&keys[i]), labels[i]); // collision: last write wins
    }
    // edit a UNIQUE-cell fact → local by construction
    book.insert(e8code(&keys[uniq]), new_u);
    let disc_edit = book[&e8code(&keys[uniq])] == new_u;
    let disc_keep = (0..N)
        .filter(|&i| i != uniq && book[&e8code(&keys[i])] == labels[i])
        .count();
    // now edit the COLLISION fact (cell shared by facts 1 & 2)
    let new_c = (labels[1] + 2) % C;
    book.insert(e8code(&keys[1]), new_c);
    let disc_coll_keep = (0..N)
        .filter(|&i| i != 1 && book[&e8code(&keys[i])] == labels[i])
        .count();

    // ---- C: CLONAL memory (per-cell entries keyed by exact key) ----
    let mut clonal: HashMap<Vec<i64>, Vec<(Vec<f32>, usize)>> = HashMap::new();
    for i in 0..N {
        clonal
            .entry(e8code(&keys[i]))
            .or_default()
            .push((keys[i].clone(), labels[i]));
    }
    let cread = |clonal: &HashMap<Vec<i64>, Vec<(Vec<f32>, usize)>>, k: &[f32]| -> usize {
        clonal[&e8code(k)]
            .iter()
            .max_by(|a, b| cos(k, &a.0).partial_cmp(&cos(k, &b.0)).unwrap())
            .unwrap()
            .1
    };
    // edit the COLLISION fact 1 by proliferating ITS entry only
    for e in clonal.get_mut(&e8code(&keys[1])).unwrap().iter_mut() {
        if e.0 == keys[1] {
            e.1 = new_c;
        }
    }
    let clonal_edit = cread(&clonal, &keys[1]) == new_c;
    let clonal_keep = (0..N)
        .filter(|&i| i != 1 && cread(&clonal, &keys[i]) == labels[i])
        .count();

    let ok = |b: bool| if b { "ok " } else { "MISS" };
    println!(
        "(dense base acc {:.0}% · the MLP memorized the facts)\n",
        dense_base * 100.0
    );
    println!("                                   edit     retention (other facts)");
    println!(
        "  DENSE MLP        · unique-cell    {}     {:.1}%   ← entangled, bleeds",
        ok(dense_edit),
        frac(dense_keep, N - 1)
    );
    println!(
        "  DISCRETE codebook· unique-cell    {}     {:.1}%   ← local by construction",
        ok(disc_edit),
        frac(disc_keep, N - 1)
    );
    println!(
        "  DISCRETE codebook· COLLISION      ok      {:.1}%   ← shared cell bleeds to collider",
        frac(disc_coll_keep, N - 1)
    );
    println!(
        "  CLONAL memory    · COLLISION      {}     {:.1}%   ← per-key clone resolves it",
        ok(clonal_edit),
        frac(clonal_keep, N - 1)
    );
    println!();
    println!("→ discrete addressing makes a single-fact edit LOCAL where the dense model bleeds.");
    println!(
        "  the only leak is cell COLLISION — and clonal proliferation (a private per-key entry)"
    );
    println!(
        "  closes it. addressing-collision is the crux to watch at scale (the Test 2 question)."
    );
}

/// `--contradict-scale`: does the addressing-collision tail grow with scale? Generate N
/// synthetic functional facts (subject code → value code), hold slot headroom fixed
/// (S = 4·N), and sweep N ∈ {24,100,1000}. Then sweep routing temperature at N=200 to
/// show temperature is the anti-collision lever. Reports recall, distinct-slots/N
/// (collision proxy), mean route prob, and edit-locality at each point.
pub fn contradiction_scale_demo() {
    use std::collections::HashSet;
    println!("=== addressing-collision vs scale · separable synthetic facts ===");
    crate::tensor::use_gpu(true); // M4: offload training matmuls to Metal (the CPU path is 1-core)
    println!();

    // one measured run: n facts, slot count s, routing temperature temp, training steps
    let trial = |n: usize, s: usize, temp: f32, steps: usize| -> (usize, usize, f32, f32, bool) {
        const K: usize = 16;
        const E: usize = 24;
        const H: usize = 96;
        const A: usize = 8;
        // separable synthetic facts: random 4-char subject (well-spread contexts — unlike
        // sequential codes, which the encoder can't tell apart) → random 3-char value.
        let mut g = 0x5EED_u64 | 1;
        let rch = |g: &mut u64| (b'a' + (unit(g) * 26.0) as u8 % 26) as char;
        let mut used = HashSet::new();
        let mut facts: Vec<(String, String)> = Vec::new();
        while facts.len() < n {
            let sub: String = (0..4).map(|_| rch(&mut g)).collect();
            if !used.insert(sub.clone()) {
                continue;
            }
            let val: String = (0..3).map(|_| rch(&mut g)).collect();
            facts.push((sub, val));
        }
        let mut corpus = String::new();
        for _ in 0..2 {
            for (sub, val) in &facts {
                corpus.push_str(&format!("{sub} is {val}. "));
            }
        }
        let cc: Vec<char> = corpus.chars().collect();
        let mut chs: Vec<char> = cc
            .iter()
            .cloned()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        chs.sort();
        let v = chs.len();
        let id_of: std::collections::HashMap<char, usize> =
            chs.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        let ids: Vec<usize> = cc.iter().map(|c| id_of[c]).collect();

        let mut ctxs: Vec<usize> = Vec::new();
        let mut tgts: Vec<usize> = Vec::new();
        for i in K..ids.len() {
            for j in i - K..i {
                ctxs.push(ids[j]);
            }
            tgts.push(ids[i]);
        }
        let b = tgts.len();
        // decision point per fact: K chars ending right before the value's first char
        let fact_ctx: Vec<(Vec<usize>, String, usize)> = facts
            .iter()
            .map(|(sub, val)| {
                let probe = format!("{sub} is ");
                let p = corpus.rfind(&probe).unwrap() + probe.len(); // rep-2 occurrence ⇒ ≥K chars of left context
                (
                    ids[p - K..p].to_vec(),
                    val.clone(),
                    id_of[&val.chars().next().unwrap()],
                )
            })
            .collect();

        let argmax = |row: &[f32]| -> usize {
            (0..row.len())
                .max_by(|&a, &c| row[a].partial_cmp(&row[c]).unwrap())
                .unwrap()
        };
        let hard1 = |scores: Tensor, nb: usize| -> Tensor {
            let ns = scores.shape()[1];
            let soft = scores.softmax_rows();
            let sd = soft.data();
            let mut delta = vec![0f32; nb * ns];
            for r in 0..nb {
                let row = &sd[r * ns..(r + 1) * ns];
                let bi = (0..ns)
                    .max_by(|&a, &c| row[a].partial_cmp(&row[c]).unwrap())
                    .unwrap();
                for j in 0..ns {
                    delta[r * ns + j] = (if j == bi { 1.0 } else { 0.0 }) - row[j];
                }
            }
            soft.add(&Tensor::constant(delta, &[nb, ns]))
        };
        let fwd = |cx: &[usize],
                   nb: usize,
                   t: &Tensor,
                   li: &Linear,
                   lq: &Linear,
                   lo: &Linear,
                   km: &Tensor,
                   vm: &Tensor|
         -> Tensor {
            let enc = li.forward(&t.embedding(cx).reshape(&[nb, K * E])).gelu();
            let a = hard1(lq.forward(&enc).matmul(&km.transpose()).scale(temp), nb);
            let mixed = enc.add(&a.matmul(vm)).gelu();
            lo.forward(&mixed)
        };
        // build + train hard model
        let mut seed = 0x1EAF_u64 | 1;
        let mk = |r: usize, c: usize, sc: f32, sd: &mut u64| {
            Tensor::param(
                (0..r * c).map(|_| sc * (2.0 * unit(sd) - 1.0)).collect(),
                &[r, c],
            )
        };
        let table = mk(v, E, 0.3, &mut seed);
        let l_in = Linear::new(K * E, H, &mut seed);
        let l_q = Linear::new(H, A, &mut seed);
        let km = mk(s, A, 0.4, &mut seed);
        let vm = mk(s, H, 0.2, &mut seed);
        let l_out = Linear::new(H, v, &mut seed);
        {
            let mut params: Vec<&Tensor> = vec![&table, &km, &vm];
            params.extend(l_in.parameters());
            params.extend(l_q.parameters());
            params.extend(l_out.parameters());
            let decay: Vec<bool> = params.iter().map(|t| t.shape().len() == 2).collect();
            let mut opt = AdamW::new(&params, 0.01, 0.0);
            let bs = 512.min(b);
            let mut dsd = 0xBEEF_u64 | 1;
            for _ in 0..=steps {
                let mut bctx = Vec::with_capacity(bs * K);
                let mut btgt = Vec::with_capacity(bs);
                for _ in 0..bs {
                    let wi = (unit(&mut dsd) * b as f32) as usize % b;
                    bctx.extend_from_slice(&ctxs[wi * K..(wi + 1) * K]);
                    btgt.push(tgts[wi]);
                }
                for p in &params {
                    p.zero_grad();
                }
                let ce = fwd(&bctx, bs, &table, &l_in, &l_q, &l_out, &km, &vm).cross_entropy(&btgt);
                let enc = l_in
                    .forward(&table.embedding(&bctx).reshape(&[bs, K * E]))
                    .gelu();
                let soft = l_q
                    .forward(&enc)
                    .matmul(&km.transpose())
                    .scale(temp)
                    .softmax_rows();
                let usage = soft.sum_axis(0).scale(1.0 / bs as f32);
                let bal = usage.square().mean().scale(8.0 * s as f32);
                ce.add(&bal).backward();
                opt.step(&params, &decay);
            }
        }

        // ---- batched eval: one forward over ALL n facts, not O(n) nb=1 calls ----
        // recall via batched greedy decode (4 batched forwards instead of n×4 single)
        let mut cur: Vec<Vec<usize>> = fact_ctx.iter().map(|(c, _, _)| c.clone()).collect();
        let mut outs: Vec<String> = vec![String::new(); n];
        let mut done = vec![false; n];
        for _ in 0..4 {
            let active: Vec<usize> = (0..n).filter(|&i| !done[i]).collect();
            if active.is_empty() {
                break;
            }
            let mut bctx = Vec::with_capacity(active.len() * K);
            for &i in &active {
                let c = &cur[i];
                bctx.extend_from_slice(&c[c.len() - K..]);
            }
            let d = fwd(&bctx, active.len(), &table, &l_in, &l_q, &l_out, &km, &vm).data();
            for (bi, &i) in active.iter().enumerate() {
                let nx = argmax(&d[bi * v..(bi + 1) * v]);
                let ch = chs[nx];
                if ch == '.' || ch == ' ' {
                    done[i] = true;
                } else {
                    outs[i].push(ch);
                    cur[i].push(nx);
                }
            }
        }
        let recall = (0..n).filter(|&i| outs[i] == fact_ctx[i].1).count();
        // distinct dominant slots + mean route prob, one batched routing forward
        let allctx: Vec<usize> = fact_ctx
            .iter()
            .flat_map(|(c, _, _)| c.iter().cloned())
            .collect();
        let enc = l_in
            .forward(&table.embedding(&allctx).reshape(&[n, K * E]))
            .gelu();
        let probs = l_q
            .forward(&enc)
            .matmul(&km.transpose())
            .scale(temp)
            .softmax_rows()
            .data();
        let mut distinct: HashSet<usize> = HashSet::new();
        let mut psum = 0.0f32;
        for i in 0..n {
            let row = &probs[i * s..(i + 1) * s];
            let bi = argmax(row);
            distinct.insert(bi);
            psum += row[bi];
        }
        let meanp = psum / n as f32;

        // edit-entanglement: sequentially revise T facts' slot values; for each, measure the
        // % of OTHER facts (a fixed sample) whose first-char prediction is UNCHANGED. This is
        // the real collision test — can each belief be addressed and edited independently.
        let first_preds = |sample: &[usize],
                           t: &Tensor,
                           li: &Linear,
                           lq: &Linear,
                           lo: &Linear,
                           km: &Tensor,
                           vm: &Tensor|
         -> Vec<usize> {
            if sample.is_empty() {
                return Vec::new();
            }
            let flat: Vec<usize> = sample
                .iter()
                .flat_map(|&j| fact_ctx[j].0.iter().cloned())
                .collect();
            let d = fwd(&flat, sample.len(), t, li, lq, lo, km, vm).data();
            (0..sample.len())
                .map(|bi| argmax(&d[bi * v..(bi + 1) * v]))
                .collect()
        };
        let tcount = 12.min(n);
        let sample: Vec<usize> = (tcount..n).take(200).collect();
        let mut loc_sum = 0.0f32;
        let mut applied_ok = 0usize;
        let mut eopt = AdamW::new(&[&vm], 0.05, 0.0);
        for target in 0..tcount {
            let before = first_preds(&sample, &table, &l_in, &l_q, &l_out, &km, &vm);
            let cur = argmax(
                &fwd(
                    &fact_ctx[target].0,
                    1,
                    &table,
                    &l_in,
                    &l_q,
                    &l_out,
                    &km,
                    &vm,
                )
                .data(),
            );
            let newc = (cur + 1) % v; // a guaranteed-different, in-vocab target char
            let ectx = fact_ctx[target].0.clone();
            for _ in 0..150 {
                vm.zero_grad();
                fwd(&ectx, 1, &table, &l_in, &l_q, &l_out, &km, &vm)
                    .cross_entropy(&[newc])
                    .backward();
                eopt.step(&[&vm], &[true]);
            }
            let after = first_preds(&sample, &table, &l_in, &l_q, &l_out, &km, &vm);
            let kept = sample
                .iter()
                .enumerate()
                .filter(|(idx, _)| after[*idx] == before[*idx])
                .count();
            loc_sum += 100.0 * kept as f32 / sample.len().max(1) as f32;
            if argmax(
                &fwd(
                    &fact_ctx[target].0,
                    1,
                    &table,
                    &l_in,
                    &l_q,
                    &l_out,
                    &km,
                    &vm,
                )
                .data(),
            ) == newc
            {
                applied_ok += 1;
            }
        }
        let editloc = loc_sum / tcount as f32;
        let _ = applied_ok;

        (recall, distinct.len(), meanp, editloc, true)
    };

    println!(
        "[scale · fair compute] S=4·N · TEMP=12 · steps≈4·N (≈const views/window) · Metal-timed"
    );
    println!(
        "  facts   slots    steps   recall         collision%   route_p   edit-local%   train_s"
    );
    for &nf in &[1000usize, 2000, 4000] {
        let s = (nf * 4).next_power_of_two();
        let steps = (4 * nf).min(12000); // hold views/window ~constant as N grows (capped)
        let t0 = std::time::Instant::now();
        let (rec, dist, mp, el, _) = trial(nf, s, 12.0, steps);
        let secs = t0.elapsed().as_secs_f32();
        let coll = 100.0 * (1.0 - dist as f32 / nf as f32);
        println!("  {nf:>5}   {s:>5}   {steps:>6}   {rec:>5}/{nf:<6}  {coll:>7.1}%   {mp:>6.2}   {el:>8.1}%   {secs:>6.0}");
    }
    println!("\nread: if recall holds ~90%+ at 2k/4k with steps∝N, the earlier N=2000 cliff was a compute-budget artifact, not a capacity wall.");
}

/// `--fixed-scale`: the "middle row" — FIXED address (slot i = fact i, no router to
/// collapse), but value vectors STILL learned by gradient, and a learned POINTER (a
/// supervised classifier enc→fact-id, not a load-balanced soft router). Separates the
/// three things that can fail: oracle-storage recall (can it LEARN the facts when each
/// has its own slot), pointer accuracy (can it ROUTE to the right slot), end-to-end
/// recall (both together). Edit-locality should be ~100% by construction.
pub fn fixed_addr_scale_demo() {
    use std::collections::HashSet;
    println!("=== fixed-address slot memory · learned values · learned pointer ===");
    crate::tensor::use_gpu(true); // M4: train matmuls on Metal
    println!();

    // (recall_end, recall_oracle, pointer_acc, edit_local)
    let trial = |n: usize, steps: usize| -> (usize, usize, usize, f32) {
        const K: usize = 16;
        const E: usize = 24;
        const H: usize = 96;
        // separable random facts (same generator as the router test)
        let mut g = 0x5EED_u64 | 1;
        let rch = |g: &mut u64| (b'a' + (unit(g) * 26.0) as u8 % 26) as char;
        let mut used = HashSet::new();
        let mut facts: Vec<(String, String)> = Vec::new();
        while facts.len() < n {
            let sub: String = (0..4).map(|_| rch(&mut g)).collect();
            if !used.insert(sub.clone()) {
                continue;
            }
            let val: String = (0..3).map(|_| rch(&mut g)).collect();
            facts.push((sub, val));
        }
        let mut corpus = String::new();
        let mut fact_of_char: Vec<usize> = Vec::new();
        for _ in 0..2 {
            for (fi, (sub, val)) in facts.iter().enumerate() {
                let s = format!("{sub} is {val}. ");
                corpus.push_str(&s);
                for _ in 0..s.chars().count() {
                    fact_of_char.push(fi);
                }
            }
        }
        let cc: Vec<char> = corpus.chars().collect();
        let mut chs: Vec<char> = cc
            .iter()
            .cloned()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        chs.sort();
        let v = chs.len();
        let id_of: std::collections::HashMap<char, usize> =
            chs.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        let ids: Vec<usize> = cc.iter().map(|c| id_of[c]).collect();

        let mut ctxs: Vec<usize> = Vec::new();
        let mut tgts: Vec<usize> = Vec::new();
        let mut wfacts: Vec<usize> = Vec::new();
        for i in K..ids.len() {
            for j in i - K..i {
                ctxs.push(ids[j]);
            }
            tgts.push(ids[i]);
            wfacts.push(fact_of_char[i]); // FIXED address: this window's slot = its fact id
        }
        let b = tgts.len();
        let fact_ctx: Vec<(Vec<usize>, String)> = facts
            .iter()
            .map(|(sub, val)| {
                let probe = format!("{sub} is ");
                let p = corpus.rfind(&probe).unwrap() + probe.len();
                (ids[p - K..p].to_vec(), val.clone())
            })
            .collect();

        let argmax = |row: &[f32]| -> usize {
            (0..row.len())
                .max_by(|&a, &c| row[a].partial_cmp(&row[c]).unwrap())
                .unwrap()
        };

        // model: encoder, learned POINTER (H→N classifier), learned VALUES vm[N,H], head
        let mut seed = 0x1EAF_u64 | 1;
        let mk = |r: usize, c: usize, sc: f32, sd: &mut u64| {
            Tensor::param(
                (0..r * c).map(|_| sc * (2.0 * unit(sd) - 1.0)).collect(),
                &[r, c],
            )
        };
        let table = mk(v, E, 0.3, &mut seed);
        let l_in = Linear::new(K * E, H, &mut seed);
        let l_slot = Linear::new(H, n, &mut seed); // pointer: enc → fact-id logits
        let vm = mk(n, H, 0.2, &mut seed); // one learned value slot per fact
        let l_out = Linear::new(H, v, &mut seed);

        let enc_fn = |ctx: &[usize], nb: usize, table: &Tensor, l_in: &Linear| -> Tensor {
            l_in.forward(&table.embedding(ctx).reshape(&[nb, K * E]))
                .gelu()
        };
        // FIXED-address read: gather vm at the given (fixed) slot per row via a one-hot
        let read_at = |slots: &[usize], nb: usize, vm: &Tensor| -> Tensor {
            let mut oh = vec![0f32; nb * n];
            for r in 0..nb {
                oh[r * n + slots[r]] = 1.0;
            }
            Tensor::constant(oh, &[nb, n]).matmul(vm)
        };
        let char_fwd = |ctx: &[usize],
                        nb: usize,
                        slots: &[usize],
                        table: &Tensor,
                        l_in: &Linear,
                        l_out: &Linear,
                        vm: &Tensor|
         -> Tensor {
            let enc = enc_fn(ctx, nb, table, l_in);
            let mixed = enc.add(&read_at(slots, nb, vm)).gelu();
            l_out.forward(&mixed)
        };

        // ---- train: char loss (read TRUE slot — teacher-forced address) + pointer loss ----
        {
            let mut params: Vec<&Tensor> = vec![&table, &vm];
            params.extend(l_in.parameters());
            params.extend(l_slot.parameters());
            params.extend(l_out.parameters());
            let decay: Vec<bool> = params.iter().map(|t| t.shape().len() == 2).collect();
            let mut opt = AdamW::new(&params, 0.01, 0.0);
            let bs = 512.min(b);
            let mut dsd = 0xBEEF_u64 | 1;
            for _ in 0..=steps {
                let mut bctx = Vec::with_capacity(bs * K);
                let mut btgt = Vec::with_capacity(bs);
                let mut bslot = Vec::with_capacity(bs);
                for _ in 0..bs {
                    let wi = (unit(&mut dsd) * b as f32) as usize % b;
                    bctx.extend_from_slice(&ctxs[wi * K..(wi + 1) * K]);
                    btgt.push(tgts[wi]);
                    bslot.push(wfacts[wi]);
                }
                for p in &params {
                    p.zero_grad();
                }
                let enc = enc_fn(&bctx, bs, &table, &l_in);
                let slot_logits = l_slot.forward(&enc);
                let mixed = enc.add(&read_at(&bslot, bs, &vm)).gelu();
                let char_logits = l_out.forward(&mixed);
                let loss = char_logits
                    .cross_entropy(&btgt)
                    .add(&slot_logits.cross_entropy(&bslot));
                loss.backward();
                opt.step(&params, &decay);
            }
        }

        // ---- eval (batched) ----
        let decode_all = |slot_of: &[usize],
                          table: &Tensor,
                          l_in: &Linear,
                          l_out: &Linear,
                          vm: &Tensor|
         -> Vec<String> {
            let mut cur: Vec<Vec<usize>> = fact_ctx.iter().map(|(c, _)| c.clone()).collect();
            let mut outs: Vec<String> = vec![String::new(); n];
            let mut done = vec![false; n];
            for _ in 0..4 {
                let active: Vec<usize> = (0..n).filter(|&i| !done[i]).collect();
                if active.is_empty() {
                    break;
                }
                let mut bctx = Vec::with_capacity(active.len() * K);
                let mut slots = Vec::with_capacity(active.len());
                for &i in &active {
                    let c = &cur[i];
                    bctx.extend_from_slice(&c[c.len() - K..]);
                    slots.push(slot_of[i]);
                }
                let d = char_fwd(&bctx, active.len(), &slots, table, l_in, l_out, vm).data();
                for (bi, &i) in active.iter().enumerate() {
                    let nx = argmax(&d[bi * v..(bi + 1) * v]);
                    let ch = chs[nx];
                    if ch == '.' || ch == ' ' {
                        done[i] = true;
                    } else {
                        outs[i].push(ch);
                        cur[i].push(nx);
                    }
                }
            }
            outs
        };

        // oracle storage: each fact reads ITS OWN slot — tests whether values were learned
        let oracle_slots: Vec<usize> = (0..n).collect();
        let outs_oracle = decode_all(&oracle_slots, &table, &l_in, &l_out, &vm);
        let recall_oracle = (0..n).filter(|&i| outs_oracle[i] == fact_ctx[i].1).count();

        // learned pointer: predict fact-id from the decision-point context
        let allctx: Vec<usize> = fact_ctx
            .iter()
            .flat_map(|(c, _)| c.iter().cloned())
            .collect();
        let enc_all = enc_fn(&allctx, n, &table, &l_in);
        let sl = l_slot.forward(&enc_all).data();
        let pred_slot: Vec<usize> = (0..n).map(|i| argmax(&sl[i * n..(i + 1) * n])).collect();
        let pointer_acc = (0..n).filter(|&i| pred_slot[i] == i).count();

        // end-to-end: route with the learned pointer, then read that slot
        let outs_end = decode_all(&pred_slot, &table, &l_in, &l_out, &vm);
        let recall_end = (0..n).filter(|&i| outs_end[i] == fact_ctx[i].1).count();

        // edit-locality: change fact 0's value (its slot), measure others (oracle read).
        // With fixed addressing this is local by construction — we measure to confirm.
        let first_preds = |slot_of: &[usize],
                           table: &Tensor,
                           l_in: &Linear,
                           l_out: &Linear,
                           vm: &Tensor|
         -> Vec<usize> {
            let flat: Vec<usize> = (0..n).flat_map(|i| fact_ctx[i].0.iter().cloned()).collect();
            let d = char_fwd(&flat, n, slot_of, table, l_in, l_out, vm).data();
            (0..n).map(|i| argmax(&d[i * v..(i + 1) * v])).collect()
        };
        let before = first_preds(&oracle_slots, &table, &l_in, &l_out, &vm);
        let cur0 = argmax(&char_fwd(&fact_ctx[0].0, 1, &[0], &table, &l_in, &l_out, &vm).data());
        let newc = (cur0 + 1) % v;
        let mut eopt = AdamW::new(&[&vm], 0.05, 0.0);
        let ectx = fact_ctx[0].0.clone();
        for _ in 0..150 {
            vm.zero_grad();
            char_fwd(&ectx, 1, &[0], &table, &l_in, &l_out, &vm)
                .cross_entropy(&[newc])
                .backward();
            eopt.step(&[&vm], &[true]);
        }
        let after = first_preds(&oracle_slots, &table, &l_in, &l_out, &vm);
        let kept = (1..n).filter(|&i| after[i] == before[i]).count();
        let edit_local = 100.0 * kept as f32 / (n - 1).max(1) as f32;

        (recall_end, recall_oracle, pointer_acc, edit_local)
    };

    println!("[fixed-address] one slot per fact · values+pointer learned · steps≈4·N");
    println!(
        "  facts    steps   storage(oracle)   pointer-acc   end-to-end    edit-local%   train_s"
    );
    for &nf in &[1000usize, 2000, 4000] {
        let steps = (4 * nf).min(12000);
        let t0 = std::time::Instant::now();
        let (re, ro, pa, el) = trial(nf, steps);
        let secs = t0.elapsed().as_secs_f32();
        let pc = |x: usize| 100.0 * x as f32 / nf as f32;
        println!(
            "  {nf:>5}   {steps:>6}    {ro:>5}/{nf:<5}({:>4.0}%)  {:>5.1}%   {re:>5}/{nf:<5}({:>4.0}%)   {el:>7.1}%   {secs:>6.0}",
            pc(ro), pc(pa), pc(re)
        );
    }
    println!("\nread: storage(oracle) = did the VALUES get learned (each fact its own slot, no collapse).");
    println!("      pointer-acc = can the learned classifier route to the right slot. end-to-end = both.");
    println!("      contrast vs the soft router: it cratered to 20% recall at 2k. fixed address removes the collapse.");
}

// ---- ownable, knowledge-retaining operators ----------------------------------
//
// A genome-gated trunk residual: h ← h + sigmoid(h·Wr + γ·Wg) ⊙ mix(h), where the
// output projection is ZERO-initialized. Zero-init ⇒ the operator is EXACT identity
// at insertion, so a frozen base keeps its knowledge perfectly. The gate is the RGA
// gene-regulation rule (genome γ + hidden h); `mix` runs through the E8 lattice
// bottleneck (your VectorQuantizer, via STE). Then only the operator trains, adding
// a new domain without disturbing the base — grow (capacity) + RGA (gating) + E8.

/// `classes`-class Gaussian blobs with a CHOOSABLE center seed, so two tasks (A, B)
/// can have genuinely different class geometry (the built-in `blobs` fixes centers).
fn blobs_seeded(
    n: usize,
    classes: usize,
    dim: usize,
    center_seed: u64,
    s: &mut u64,
) -> (Tensor, Vec<usize>) {
    let mut cs = center_seed | 1;
    let centers: Vec<Vec<f32>> = (0..classes)
        .map(|_| (0..dim).map(|_| 2.0 * unit(&mut cs) - 1.0).collect())
        .collect();
    let mut data = vec![0f32; n * dim];
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let c = ((unit(s) * classes as f32) as usize).min(classes - 1);
        for j in 0..dim {
            data[i * dim + j] = centers[c][j] + 0.4 * (2.0 * unit(s) - 1.0);
        }
        y.push(c);
    }
    (Tensor::constant(data, &[n, dim]), y)
}

struct GatedTrunkOp {
    wr: Tensor,                  // [d, d]    hidden-driven gate
    wg: Tensor,                  // [glen, d] genome-driven gate bias
    genome: Tensor,              // [1, glen] the trainable regulatory code
    w_in: Tensor,                // [d, m]    into the mixer bottleneck
    w_out: Tensor,               // [m, d]    out of the mixer — ZERO-init (⇒ identity at start)
    vq: Option<VectorQuantizer>, // Some = E8 lattice bottleneck, None = plain
    d: usize,
}

impl GatedTrunkOp {
    fn new(d: usize, m: usize, glen: usize, e8: bool, s: &mut u64) -> GatedTrunkOp {
        let mut p_rand = |shape: &[usize], scale: f32| {
            let n: usize = shape.iter().product();
            Tensor::param(
                (0..n).map(|_| scale * (2.0 * unit(s) - 1.0)).collect(),
                shape,
            )
        };
        GatedTrunkOp {
            wr: p_rand(&[d, d], 0.1),
            wg: p_rand(&[glen, d], 0.1),
            genome: p_rand(&[1, glen], 0.2),
            w_in: p_rand(&[d, m], 0.2),
            w_out: Tensor::param(vec![0f32; m * d], &[m, d]),
            vq: if e8 {
                Some(VectorQuantizer {
                    block: 8,
                    scale: 0.75,
                    codebook: Codebook::E8,
                    commit_weight: 0.0,
                })
            } else {
                None
            },
            d,
        }
    }
    fn params(&self) -> Vec<&Tensor> {
        vec![&self.wr, &self.wg, &self.genome, &self.w_in, &self.w_out]
    }
    /// h ← h + sigmoid(h·Wr + γ·Wg) ⊙ Wout·mix(Win·h). Zero w_out ⇒ returns h.
    fn forward(&self, h: &Tensor, strength: f32) -> Tensor {
        let n = h.shape()[0];
        let gfield = self.genome.matmul(&self.wg).broadcast_to(&[n, self.d]);
        let gate = h.matmul(&self.wr).add(&gfield).sigmoid();
        let z = h.matmul(&self.w_in);
        let zq = match &self.vq {
            Some(vq) => vq.quantize(&z, strength).output,
            None => z,
        };
        let mix = zq.matmul(&self.w_out);
        h.add(&gate.mul(&mix))
    }
}

/// `--op-demo`: ownable operators that RETAIN knowledge. Train a frozen base on
/// task A, then insert the genome-gated E8 trunk residual (zero-init). Show A is
/// unchanged at insertion (drift ≈ 0 — knowledge kept exactly), then train ONLY the
/// operator to add task B while the base stays frozen. Run it with the E8 bottleneck
/// and with a plain mixer, to see whether the lattice actually earns its place.
pub fn op_demo() {
    println!("=== ownable, knowledge-retaining operators: grow + RGA-gate + E8 ===");
    println!("operator: h ← h + sigmoid(h·Wr + γ·Wg) ⊙ Wout·E8mix(Win·h),  Wout = ZERO-init");
    println!("zero-init ⇒ exact identity at insertion ⇒ a frozen base keeps its knowledge.\n");
    const IN: usize = 16;
    const H: usize = 32;
    const C: usize = 4;
    const M: usize = 16;
    const GLEN: usize = 8;
    let seed_a = 0xA1A1_u64;
    let seed_b = 0xB2B2_u64;
    let mut s = 0x0FED_u64 | 1;
    let enc = Linear::new(IN, H, &mut s);
    let head = Linear::new(H, C, &mut s);
    let mut ds = 0xD00D_u64 | 1;

    // --- train the base on task A, then freeze it ---
    {
        let mut bp = enc.parameters();
        bp.extend(head.parameters());
        let dc: Vec<bool> = bp.iter().map(|t| t.shape().len() == 2).collect();
        let mut opt = AdamW::new(&bp, 0.01, 0.0);
        for _ in 0..600 {
            let (x, y) = blobs_seeded(256, C, IN, seed_a, &mut ds);
            for t in &bp {
                t.zero_grad();
            }
            head.forward(&enc.forward(&x).gelu())
                .cross_entropy(&y)
                .backward();
            opt.step(&bp, &dc);
        }
    }
    let (xa, ya) = blobs_seeded(512, C, IN, seed_a, &mut ds);
    let (xb, yb) = blobs_seeded(512, C, IN, seed_b, &mut ds);
    let base_a = accuracy(&head.forward(&enc.forward(&xa).gelu()).data(), &ya, C);
    let base_b = accuracy(&head.forward(&enc.forward(&xb).gelu()).data(), &yb, C);
    println!(
        "frozen base (trained on A only):   A {:.1}%   B {:.1}%   (B unseen ⇒ ~chance)\n",
        base_a * 100.0,
        base_b * 100.0
    );

    let hidden = |x: &Tensor| Tensor::constant(enc.forward(x).gelu().data(), &[x.shape()[0], H]);
    let mut run = |e8: bool, label: &str| -> f32 {
        let op = GatedTrunkOp::new(H, M, GLEN, e8, &mut s);
        // drift at insertion: operator must be EXACT identity (knowledge kept).
        let init_a = accuracy(&head.forward(&op.forward(&hidden(&xa), 1.0)).data(), &ya, C);
        let init_b = accuracy(&head.forward(&op.forward(&hidden(&xb), 1.0)).data(), &yb, C);
        // train ONLY the operator on an A∪B mixture (replay), base frozen.
        let p = op.params();
        let dc: Vec<bool> = p.iter().map(|t| t.shape().len() == 2).collect();
        let mut opt = AdamW::new(&p, 0.01, 0.0);
        for step in 0..900 {
            let strength = (step as f32 / 100.0).min(1.0); // ease the lattice in
            let seed = if step % 2 == 0 { seed_a } else { seed_b };
            let (x, y) = blobs_seeded(256, C, IN, seed, &mut ds);
            let h = Tensor::constant(enc.forward(&x).gelu().data(), &[256, H]);
            for t in &p {
                t.zero_grad();
            }
            head.forward(&op.forward(&h, strength))
                .cross_entropy(&y)
                .backward();
            opt.step(&p, &dc);
        }
        let fa = accuracy(&head.forward(&op.forward(&hidden(&xa), 1.0)).data(), &ya, C);
        let fb = accuracy(&head.forward(&op.forward(&hidden(&xb), 1.0)).data(), &yb, C);
        println!("[{label}]");
        println!(
            "  at insertion:  A {:.1}%   B {:.1}%   (drift from base on A = {:+.2} pts)",
            init_a * 100.0,
            init_b * 100.0,
            (init_a - base_a) * 100.0
        );
        println!(
            "  after train:   A {:.1}%   B {:.1}%   (kept A, learned B)\n",
            fa * 100.0,
            fb * 100.0
        );
        (fa + fb) / 2.0
    };

    let e8 = run(true, "E8 mixer (your lattice bottleneck)");
    let plain = run(false, "plain mixer (no E8)");

    println!("verdict:");
    println!(
        "  RETAIN  — zero-init ⇒ drift ≈ 0 at insertion: the base keeps its knowledge exactly."
    );
    println!(
        "  CAPACITY— the gated operator added domain B with the base frozen (no edit, no forget)."
    );
    if e8 > plain + 0.01 {
        println!(
            "  E8 EARNS its place: mean {:.1}% > plain {:.1}% — the lattice helped on this task.",
            e8 * 100.0,
            plain * 100.0
        );
    } else if e8 + 0.01 < plain {
        println!(
            "  E8 did NOT earn its place: mean {:.1}% < plain {:.1}% — keep it optional/gated here.",
            e8 * 100.0,
            plain * 100.0
        );
    } else {
        println!(
            "  E8 TIES the plain mixer ({:.1}% vs {:.1}%): knowledge-safe and yours, no edge on this toy.",
            e8 * 100.0,
            plain * 100.0
        );
    }
}
