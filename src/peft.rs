//! T2 — parameter-efficient finetuning on a FROZEN base.
//!
//! Two methods share one substrate, so they can be compared head-to-head:
//!
//!  - **LoRA** (the matched baseline): `W·x + (α/r)·B(A·x)` at chosen
//!    projections; A small-random, B zero, so training starts from the base.
//!
//!  - **RGA** (Regulatory Genome Adapters — the genomic alternative to LoRA):
//!    a compact trainable *genome* regulates sparse expression of a shared bank
//!    of tiny adapter modules. Per layer, `h ← h + Σ_k r_k(h, γ)·A_k(h)`, where
//!    the gate `r_k = sigmoid(h·W_r + γ·W_g)` is driven by the genome and the
//!    hidden state. This is the M3 genome-regulated-mixture mechanism, applied
//!    as a PEFT adapter on a frozen pretrained transformer.
//!
//! The base weights are autograd **constants** (no gradient, no optimizer state),
//! so only the adapters train — this fits where full finetuning will not, and the
//! `.hos` artifact stores only the adapter (its lineage points to the base).

use crate::model::{Config, Model, Weight};
use crate::tensor::Tensor;

// ---- frozen base ----

fn t_const(w: &Weight, out: usize, inn: usize) -> Tensor {
    let d = w.to_f32();
    let mut t = vec![0.0; out * inn];
    for o in 0..out {
        for i in 0..inn {
            t[i * out + o] = d[o * inn + i];
        }
    }
    Tensor::constant(t, &[inn, out])
}

struct BaseLayer {
    attn_norm: Tensor,
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wo: Tensor,
    bq: Option<Tensor>,
    bk: Option<Tensor>,
    bv: Option<Tensor>,
    ffn_norm: Tensor,
    w_gate: Tensor,
    w_up: Tensor,
    w_down: Tensor,
}

// ---- adapters ----

/// LoRA factors for the Q and V projections of one layer.
struct LoraLayer {
    aq: Tensor,
    bq: Tensor,
    av: Tensor,
    bv: Tensor,
    scale: f32,
}

/// RGA gene bank + gate for one layer (the genome is global, in `Rga`).
struct RgaLayer {
    wr: Tensor,        // [dim, K] hidden -> gate
    down: Vec<Tensor>, // K x [dim, bottleneck]
    up: Vec<Tensor>,   // K x [bottleneck, dim]
}

/// A private, trainable bank cloned from selected genes — clonal selection's
/// "proliferate the best antibodies, then somatically mutate the copies." Only
/// the new task expresses it; the shared bank (and old tasks) stay untouched.
struct CloneLayer {
    wr: Tensor,        // [dim, nc] gate (copied from selected genes' gate cols)
    down: Vec<Tensor>, // nc x [dim, bottleneck]
    up: Vec<Tensor>,   // nc x [bottleneck, dim]
}

struct Rga {
    genomes: Vec<Tensor>, // one compact regulatory code per domain (M3: shared bank, per-regime genome)
    wg: Tensor,           // [GLEN, K] genome -> per-gene gate bias
    k: usize,
    layers: Vec<RgaLayer>,
    clones: Option<Vec<CloneLayer>>, // private proliferated genes (clonal selection)
}

/// **Net2Net-style FFN widening** — `delta` extra *trainable* intermediate
/// dimensions grafted onto the frozen base FFN of one layer. Unlike a gene (a
/// gated side-module on the residual), this widens the hidden layer itself: it
/// shares the FFN's normed input and SwiGLU structure. `down_new` is ZERO at init,
/// so `[inter | inter_new] · [[down];[down_new]] = inter · down` exactly — the
/// function is unchanged at init (function-preserving growth). Training the new
/// dims adds shared capacity *in* the hidden layer.
struct GrowLayer {
    gate_new: Tensor, // [dim, delta]
    up_new: Tensor,   // [dim, delta]
    down_new: Tensor, // [delta, dim], zero-init
}

enum Method {
    Lora(Vec<LoraLayer>),
    Rga(Rga),
}

pub struct PeftModel {
    cfg: Config,
    tok_embd: Tensor,
    base: Vec<BaseLayer>,
    out_norm: Tensor,
    out: Tensor,
    method: Method,
    pub method_name: String,
    /// When set, the forward also expresses the private cloned genes (clonal
    /// selection). Off for the seen domains, so they're literally unaffected.
    clones_active: std::cell::Cell<bool>,
    /// Net2Net-grown FFN dims (empty = ungrown). Shared trainable capacity *in*
    /// the hidden layers — distinct from genes (parallel side-modules).
    grow: Vec<GrowLayer>,
}

/// Adapter hyperparameters (from CLI/env).
pub struct PeftCfg {
    pub rank: usize,
    pub genes: usize,
    pub bottleneck: usize,
    pub lora_alpha: f32,
}

impl PeftModel {
    pub fn build(m: &Model, method: &str, h: &PeftCfg, seed: u64) -> Result<PeftModel, String> {
        Self::build_multi(m, method, h, 1, seed)
    }

    /// `n_genomes` (RGA only): one genome per domain over a shared gene bank — the
    /// M3 configuration (frozen bank re-regulated per regime). LoRA ignores it.
    pub fn build_multi(
        m: &Model,
        method: &str,
        h: &PeftCfg,
        n_genomes: usize,
        seed: u64,
    ) -> Result<PeftModel, String> {
        let c = &m.cfg;
        if !matches!(
            c.arch,
            crate::model::Arch::Llama | crate::model::Arch::Mistral | crate::model::Arch::Qwen2
        ) {
            return Err(format!(
                "PEFT supports Llama/Mistral/Qwen2 today; {:?} not yet",
                c.arch
            ));
        }
        if m.layers.iter().any(|l| l.moe.is_some()) {
            return Err("MoE PEFT not supported yet".into());
        }
        let (dim, qd, kvd, ffn) = (
            c.dim,
            c.n_heads * c.head_dim,
            c.n_kv_heads * c.head_dim,
            c.ffn_dim,
        );
        let vc = |v: &[f32], s: &[usize]| Tensor::constant(v.to_vec(), s);
        let ob = |o: &Option<Vec<f32>>| o.as_ref().map(|b| vc(b, &[b.len()]));
        let base: Vec<BaseLayer> = m
            .layers
            .iter()
            .map(|l| BaseLayer {
                attn_norm: vc(&l.attn_norm, &[dim]),
                wq: t_const(&l.wq, qd, dim),
                wk: t_const(&l.wk, kvd, dim),
                wv: t_const(&l.wv, kvd, dim),
                wo: t_const(&l.wo, dim, qd),
                bq: ob(&l.bq),
                bk: ob(&l.bk),
                bv: ob(&l.bv),
                ffn_norm: vc(&l.ffn_norm, &[dim]),
                w_gate: t_const(&l.w_gate, ffn, dim),
                w_up: t_const(&l.w_up, ffn, dim),
                w_down: t_const(&l.w_down, dim, ffn),
            })
            .collect();

        let method_name = method.to_string();
        let mut s = seed | 1;
        let mut rnd = |sh: &[usize]| Tensor::randn(sh, &mut s);
        let zeros = |sh: &[usize]| Tensor::param(vec![0.0; sh.iter().product()], sh);

        let method = match method {
            "lora" => {
                let r = h.rank;
                let layers = (0..c.n_layers)
                    .map(|_| LoraLayer {
                        aq: rnd(&[dim, r]),
                        bq: zeros(&[r, qd]), // B = 0  -> adapter starts at the base
                        av: rnd(&[dim, r]),
                        bv: zeros(&[r, kvd]),
                        scale: h.lora_alpha / r as f32,
                    })
                    .collect();
                Method::Lora(layers)
            }
            "rga" => {
                let (k, b, glen) = (h.genes, h.bottleneck, 8usize);
                let layers = (0..c.n_layers)
                    .map(|_| RgaLayer {
                        wr: rnd(&[dim, k]),
                        down: (0..k).map(|_| rnd(&[dim, b])).collect(),
                        up: (0..k).map(|_| zeros(&[b, dim])).collect(), // up = 0 -> starts at base
                    })
                    .collect();
                Method::Rga(Rga {
                    genomes: (0..n_genomes.max(1)).map(|_| rnd(&[1, glen])).collect(),
                    wg: rnd(&[glen, k]),
                    k,
                    layers,
                    clones: None,
                })
            }
            other => return Err(format!("unknown PEFT method '{other}' (lora | rga)")),
        };

        Ok(PeftModel {
            cfg: c.clone(),
            tok_embd: vc(&m.tok_embd, &[c.vocab_size, dim]),
            base,
            out_norm: vc(&m.output_norm, &[dim]),
            out: t_const(&m.output, c.vocab_size, dim),
            method,
            method_name,
            clones_active: std::cell::Cell::new(false),
            grow: Vec::new(),
        })
    }

    /// **Net2Net widen** — add `delta` trainable intermediate dimensions to every
    /// layer's FFN. `down_new` is zero-initialised, so the model's function is
    /// unchanged at init (function-preserving); training the new dims grows shared
    /// capacity in the hidden layers. This is "adding dimensions in the hidden
    /// layers," as opposed to genes (parallel gated side-modules).
    pub fn grow_ffn(&mut self, delta: usize, seed: u64) {
        if delta == 0 {
            return;
        }
        let dim = self.cfg.dim;
        let ffn = self.cfg.ffn_dim;
        let mut s = seed | 1;
        // True Net2Net: COPY existing FFN intermediate neurons into the new slots
        // (gate/up columns), and ZERO the new down rows. The copies start in a
        // useful region (warm start), and down_new=0 keeps the function unchanged
        // at init — training then specializes the duplicated neurons.
        self.grow = (0..self.cfg.n_layers)
            .map(|li| {
                let wg = self.base[li].w_gate.data(); // [dim, ffn]
                let wu = self.base[li].w_up.data(); // [dim, ffn]
                let mut gate = vec![0.0f32; dim * delta];
                let mut up = vec![0.0f32; dim * delta];
                for c in 0..delta {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    let j = (s >> 16) as usize % ffn; // a real neuron to duplicate
                    for i in 0..dim {
                        gate[i * delta + c] = wg[i * ffn + j];
                        up[i * delta + c] = wu[i * ffn + j];
                    }
                }
                GrowLayer {
                    gate_new: Tensor::param(gate, &[dim, delta]),
                    up_new: Tensor::param(up, &[dim, delta]),
                    down_new: Tensor::param(vec![0.0; delta * dim], &[delta, dim]),
                }
            })
            .collect();
    }

    /// Grown FFN width (0 if ungrown) — for save/reload of a widened body.
    pub fn grow_delta(&self) -> usize {
        self.grow
            .first()
            .map(|g| g.gate_new.shape()[1])
            .unwrap_or(0)
    }

    /// All adapter parameters (base frozen): bank/gates + every genome.
    pub fn params(&self) -> Vec<&Tensor> {
        let mut p = self.params_bank();
        if let Method::Rga(r) = &self.method {
            p.extend(r.genomes.iter());
        }
        p
    }

    /// The shared, reusable infrastructure — LoRA factors, or the RGA gene bank
    /// plus its gate weights — i.e. everything EXCEPT the per-domain genomes.
    /// Freeze this and train only a genome to adapt a new domain with zero
    /// interference on the others.
    pub fn params_bank(&self) -> Vec<&Tensor> {
        let mut p = Vec::new();
        match &self.method {
            Method::Lora(ls) => {
                for l in ls {
                    p.extend([&l.aq, &l.bq, &l.av, &l.bv]);
                }
            }
            Method::Rga(r) => {
                p.push(&r.wg);
                for l in &r.layers {
                    p.push(&l.wr);
                    p.extend(l.down.iter());
                    p.extend(l.up.iter());
                }
            }
        }
        // grown FFN dims are shared trainable capacity — consolidated like the bank
        for gl in &self.grow {
            p.push(&gl.gate_new);
            p.push(&gl.up_new);
            p.push(&gl.down_new);
        }
        p
    }

    /// Just the genome for domain `gi` (RGA) — the tiny per-domain code.
    pub fn params_genome(&self, gi: usize) -> Vec<&Tensor> {
        match &self.method {
            Method::Rga(r) => vec![&r.genomes[gi]],
            Method::Lora(_) => Vec::new(),
        }
    }

    /// A domain's per-gene regulatory drive (`genome · Wg`, length K) — how much
    /// that domain's genome "wants" each gene. Clonal selection picks the genes
    /// with the highest drive to clone + mutate for a new (out-of-span) task.
    pub fn genome_gate_bias(&self, gi: usize) -> Vec<f32> {
        match &self.method {
            Method::Rga(r) => r.genomes[gi.min(r.genomes.len() - 1)]
                .matmul(&r.wg)
                .reshape(&[r.k])
                .data(),
            Method::Lora(_) => Vec::new(),
        }
    }

    /// The module tensors (down/up across all layers) for a set of gene indices —
    /// the "clones" to unfreeze and somatically mutate in clonal selection.
    pub fn params_genes(&self, ks: &[usize]) -> Vec<&Tensor> {
        let mut p = Vec::new();
        if let Method::Rga(r) = &self.method {
            for l in &r.layers {
                for &kk in ks {
                    if kk < r.k {
                        p.push(&l.down[kk]);
                        p.push(&l.up[kk]);
                    }
                }
            }
        }
        p
    }

    /// Clonal selection — *proliferate*: copy the chosen genes (gate column plus
    /// down/up across every layer) into a fresh, private, trainable bank. The
    /// originals are untouched; the copies are then mutated by training.
    pub fn proliferate(&mut self, genes: &[usize]) {
        let dim = self.cfg.dim;
        if let Method::Rga(r) = &mut self.method {
            let nc = genes.len();
            let kk = r.k;
            let clones: Vec<CloneLayer> = r
                .layers
                .iter()
                .map(|l| {
                    let wr = l.wr.data();
                    let mut wc = vec![0.0f32; dim * nc];
                    for i in 0..dim {
                        for (j, &g) in genes.iter().enumerate() {
                            wc[i * nc + j] = wr[i * kk + g];
                        }
                    }
                    CloneLayer {
                        wr: Tensor::param(wc, &[dim, nc]),
                        down: genes
                            .iter()
                            .map(|&g| Tensor::param(l.down[g].data(), &l.down[g].shape()))
                            .collect(),
                        up: genes
                            .iter()
                            .map(|&g| Tensor::param(l.up[g].data(), &l.up[g].shape()))
                            .collect(),
                    }
                })
                .collect();
            r.clones = Some(clones);
        }
    }

    /// The trainable parameters of the private cloned bank (to mutate them).
    pub fn params_clones(&self) -> Vec<&Tensor> {
        let mut p = Vec::new();
        if let Method::Rga(r) = &self.method {
            if let Some(cl) = &r.clones {
                for l in cl {
                    p.push(&l.wr);
                    p.extend(l.down.iter());
                    p.extend(l.up.iter());
                }
            }
        }
        p
    }

    /// Express the private clones in the forward (the new task) or not (old tasks).
    pub fn set_clones(&self, on: bool) {
        self.clones_active.set(on);
    }

    pub fn n_genomes(&self) -> usize {
        match &self.method {
            Method::Rga(r) => r.genomes.len(),
            Method::Lora(_) => 1,
        }
    }

    /// Read / write a genome's values — for recombination (set a child genome to a
    /// crossover of two parents, then evaluate it).
    pub fn genome_data(&self, gi: usize) -> Vec<f32> {
        match &self.method {
            Method::Rga(r) => r.genomes[gi].data(),
            Method::Lora(_) => Vec::new(),
        }
    }
    pub fn set_genome(&self, gi: usize, data: &[f32]) {
        if let Method::Rga(r) = &self.method {
            r.genomes[gi].set_data(data);
        }
    }

    pub fn n_trainable(&self) -> usize {
        self.params().iter().map(|t| t.data().len()).sum()
    }

    /// Loss using genome 0 (single-domain).
    pub fn loss(&self, ids: &[usize], targets: &[usize], lambda: f32) -> Tensor {
        self.loss_g(ids, targets, lambda, 0)
    }

    /// Loss for one sequence using domain `gi`'s genome: cross-entropy + (RGA only)
    /// an L1 sparsity penalty on the gates.
    pub fn loss_g(&self, ids: &[usize], targets: &[usize], lambda: f32, gi: usize) -> Tensor {
        let (logits, gate_means) = self.forward(ids, gi);
        let ce = logits.cross_entropy(targets);
        if lambda > 0.0 && !gate_means.is_empty() {
            let mut s = gate_means[0].clone();
            for g in &gate_means[1..] {
                s = s.add(g);
            }
            ce.add(&s.scale(lambda / gate_means.len() as f32))
        } else {
            ce
        }
    }

    /// Logits only — for the init-parity check against the base forward.
    pub fn logits(&self, ids: &[usize]) -> Tensor {
        self.forward(ids, 0).0
    }

    /// Logits for a specific genome `gi` (clones honored if active) — used to
    /// greedily sample qualitative before/after generations.
    pub fn logits_g(&self, ids: &[usize], gi: usize) -> Tensor {
        self.forward(ids, gi).0
    }

    /// Forward through the frozen base with adapters, using domain `gi`'s genome
    /// -> (logits `[t, vocab]`, per-layer RGA gate means for the sparsity penalty).
    fn forward(&self, ids: &[usize], gi: usize) -> (Tensor, Vec<Tensor>) {
        let c = &self.cfg;
        let (t, hd, nh, nkv) = (ids.len(), c.head_dim, c.n_heads, c.n_kv_heads);
        let (qd, group, scale) = (nh * hd, nh / nkv, 1.0 / (hd as f32).sqrt());
        let pos: Vec<usize> = (0..t).collect();
        let mask = {
            let mut m = vec![0.0f32; nh * t * t];
            for h in 0..nh {
                for i in 0..t {
                    for j in (i + 1)..t {
                        m[(h * t + i) * t + j] = -1e9;
                    }
                }
            }
            Tensor::constant(m, &[nh * t, t])
        };
        // domain gi's genome -> per-gene gate bias [K] (the regulator)
        let g_bias = match &self.method {
            Method::Rga(r) => Some(
                r.genomes[gi.min(r.genomes.len() - 1)]
                    .matmul(&r.wg)
                    .reshape(&[r.k]),
            ),
            _ => None,
        };

        let mut x = self.tok_embd.embedding(ids);
        let mut gate_means: Vec<Tensor> = Vec::new();
        for (li, bl) in self.base.iter().enumerate() {
            // ---- attention (frozen base + optional LoRA on Q,V) ----
            let xn = x.rmsnorm_eps(&bl.attn_norm, c.rms_eps);
            let mut q = xn.matmul(&bl.wq);
            let mut k = xn.matmul(&bl.wk);
            let mut v = xn.matmul(&bl.wv);
            if let Method::Lora(ls) = &self.method {
                let l = &ls[li];
                q = q.add(&xn.matmul(&l.aq).matmul(&l.bq).scale(l.scale));
                v = v.add(&xn.matmul(&l.av).matmul(&l.bv).scale(l.scale));
            }
            if let Some(b) = &bl.bq {
                q = q.add(b);
            }
            if let Some(b) = &bl.bk {
                k = k.add(b);
            }
            if let Some(b) = &bl.bv {
                v = v.add(b);
            }
            let q = q.rope(nh, hd, c.rope_base, c.rope_neox, &pos);
            let k = k.rope(nkv, hd, c.rope_base, c.rope_neox, &pos);
            let qh = q
                .reshape(&[1, t, nh, hd])
                .transpose12_4d()
                .reshape(&[nh, t, hd]);
            let kh = k
                .reshape(&[1, t, nkv, hd])
                .transpose12_4d()
                .reshape(&[nkv, t, hd]);
            let vh = v
                .reshape(&[1, t, nkv, hd])
                .transpose12_4d()
                .reshape(&[nkv, t, hd]);
            let kh = kh.repeat_interleave_dim0(group);
            let vh = vh.repeat_interleave_dim0(group);
            let scores = qh.bmm(&kh.transpose_last2()).scale(scale);
            let att = scores
                .reshape(&[nh * t, t])
                .add(&mask)
                .softmax_rows()
                .reshape(&[nh, t, t]);
            let ctx = att
                .bmm(&vh)
                .reshape(&[1, nh, t, hd])
                .transpose12_4d()
                .reshape(&[t, qd]);
            x = x.add(&ctx.matmul(&bl.wo));
            // ---- FFN (frozen base) ----
            let xn2 = x.rmsnorm_eps(&bl.ffn_norm, c.rms_eps);
            x = x.add(
                &xn2.matmul(&bl.w_gate)
                    .silu()
                    .mul(&xn2.matmul(&bl.w_up))
                    .matmul(&bl.w_down),
            );
            // ---- Net2Net-grown FFN dims (same normed input + SwiGLU; down_new=0
            // at init so this is function-preserving until the new dims train) ----
            if !self.grow.is_empty() {
                let gl = &self.grow[li];
                let inter = xn2.matmul(&gl.gate_new).silu().mul(&xn2.matmul(&gl.up_new));
                x = x.add(&inter.matmul(&gl.down_new));
            }
            // ---- RGA: genome-regulated sparse gene expression on the residual ----
            if let Method::Rga(r) = &self.method {
                let rl = &r.layers[li];
                let gate = x.matmul(&rl.wr).add(g_bias.as_ref().unwrap()).sigmoid(); // [t, K]
                let mut upd: Option<Tensor> = None;
                for kk in 0..r.k {
                    let gene = x.matmul(&rl.down[kk]).relu().matmul(&rl.up[kk]); // [t, dim]
                    let gated = gene.mul_broadcast(&gate.col(kk));
                    upd = Some(match upd {
                        Some(acc) => acc.add(&gated),
                        None => gated,
                    });
                }
                x = x.add(&upd.unwrap());
                gate_means.push(gate.mean());
                // private cloned genes — expressed only for the new (clone) task,
                // so the seen domains (clones off) are literally unaffected.
                if self.clones_active.get() {
                    if let Some(cl) = &r.clones {
                        let cll = &cl[li];
                        let gate_c = x.matmul(&cll.wr).sigmoid(); // [t, nc]
                        let mut updc: Option<Tensor> = None;
                        for kk in 0..cll.down.len() {
                            let gene = x.matmul(&cll.down[kk]).relu().matmul(&cll.up[kk]);
                            let gated = gene.mul_broadcast(&gate_c.col(kk));
                            updc = Some(match updc {
                                Some(a) => a.add(&gated),
                                None => gated,
                            });
                        }
                        if let Some(u) = updc {
                            x = x.add(&u);
                        }
                    }
                }
            }
        }
        let logits = x.rmsnorm_eps(&self.out_norm, c.rms_eps).matmul(&self.out);
        (logits, gate_means)
    }

    /// (gene count K, genome count) for an RGA model — lets a reloader rebuild a
    /// matching shell before `load_adapter`. `None` for LoRA.
    pub fn rga_shape(&self) -> Option<(usize, usize)> {
        match &self.method {
            Method::Rga(r) => Some((r.k, r.genomes.len())),
            Method::Lora(_) => None,
        }
    }

    /// Restore an adapter snapshot (the inverse of [`adapter_tensors`]) into a
    /// freshly-built model of matching shape — used to reload a fused `.hos` body
    /// and re-express its genes for verification.
    pub fn load_adapter(&self, snap: &[(String, Vec<usize>, Vec<f32>)]) {
        if let Method::Rga(r) = &self.method {
            for (name, _shape, data) in snap {
                let parts: Vec<&str> = name.split('.').collect();
                match parts.as_slice() {
                    ["genome", gi] => {
                        if let Ok(gi) = gi.parse::<usize>() {
                            if gi < r.genomes.len() {
                                r.genomes[gi].set_data(data);
                            }
                        }
                    }
                    ["wg"] => r.wg.set_data(data),
                    ["rga", li, "wr"] => {
                        if let Ok(li) = li.parse::<usize>() {
                            r.layers[li].wr.set_data(data);
                        }
                    }
                    ["rga", li, "down", kk] => {
                        if let (Ok(li), Ok(kk)) = (li.parse::<usize>(), kk.parse::<usize>()) {
                            r.layers[li].down[kk].set_data(data);
                        }
                    }
                    ["rga", li, "up", kk] => {
                        if let (Ok(li), Ok(kk)) = (li.parse::<usize>(), kk.parse::<usize>()) {
                            r.layers[li].up[kk].set_data(data);
                        }
                    }
                    _ => {}
                }
            }
        }
        // grown FFN dims (must have been re-allocated via grow_ffn first)
        for (name, _shape, data) in snap {
            if let ["grow", li, which] = name.split('.').collect::<Vec<_>>().as_slice() {
                if let Ok(li) = li.parse::<usize>() {
                    if let Some(gl) = self.grow.get(li) {
                        match *which {
                            "gate" => gl.gate_new.set_data(data),
                            "up" => gl.up_new.set_data(data),
                            "down" => gl.down_new.set_data(data),
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    /// **Regime-A fusion** — import two homologous parents' gene banks into this
    /// (freshly-built, K = Ka+Kb) shared body, and copy their genomes in.
    ///
    /// Both parents regulate the *same frozen base*, so alignment is the identity:
    /// their weight-deltas already live in one coordinate frame and the bank is a
    /// literal concatenation (span-union). `genome_src[d] = (from_b, idx)` says
    /// where fused domain `d`'s genome comes from (parent A or B, which slot).
    /// After this, a brief per-domain genome refit + replay consolidates the
    /// imported organs into one body with no forgetting.
    pub fn fuse(&self, a: &PeftModel, b: &PeftModel, genome_src: &[(bool, usize)]) {
        let (sr, ar, br) = match (&self.method, &a.method, &b.method) {
            (Method::Rga(s), Method::Rga(x), Method::Rga(y)) => (s, x, y),
            _ => return,
        };
        let (ka, kb, kf) = (ar.k, br.k, sr.k);
        assert_eq!(kf, ka + kb, "fused K must equal Ka+Kb");
        let glen = sr.wg.shape()[0];
        // gate weights wg: [glen, Kf] = [A.wg | B.wg] column-concat
        let (awg, bwg) = (ar.wg.data(), br.wg.data());
        let mut wg = vec![0.0f32; glen * kf];
        for g in 0..glen {
            for c in 0..ka {
                wg[g * kf + c] = awg[g * ka + c];
            }
            for c in 0..kb {
                wg[g * kf + ka + c] = bwg[g * kb + c];
            }
        }
        sr.wg.set_data(&wg);
        // per-layer bank: wr columns concat, down/up gene lists concat
        let dim = self.cfg.dim;
        for li in 0..sr.layers.len() {
            let (al, bl, sl) = (&ar.layers[li], &br.layers[li], &sr.layers[li]);
            let (awr, bwr) = (al.wr.data(), bl.wr.data());
            let mut wr = vec![0.0f32; dim * kf];
            for i in 0..dim {
                for c in 0..ka {
                    wr[i * kf + c] = awr[i * ka + c];
                }
                for c in 0..kb {
                    wr[i * kf + ka + c] = bwr[i * kb + c];
                }
            }
            sl.wr.set_data(&wr);
            for c in 0..ka {
                sl.down[c].set_data(&al.down[c].data());
                sl.up[c].set_data(&al.up[c].data());
            }
            for c in 0..kb {
                sl.down[ka + c].set_data(&bl.down[c].data());
                sl.up[ka + c].set_data(&bl.up[c].data());
            }
        }
        // genomes: copy each fused domain's genome from its source parent
        for (d, &(from_b, idx)) in genome_src.iter().enumerate() {
            if d >= sr.genomes.len() {
                break;
            }
            let src = if from_b { &br.genomes } else { &ar.genomes };
            sr.genomes[d].set_data(&src[idx.min(src.len() - 1)].data());
        }
    }

    /// Adapter snapshot for saving (name, shape, data) — adapter-only capsule.
    pub fn adapter_tensors(&self) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        let mut v = Vec::new();
        match &self.method {
            Method::Lora(ls) => {
                for (i, l) in ls.iter().enumerate() {
                    for (n, t) in [("aq", &l.aq), ("bq", &l.bq), ("av", &l.av), ("bv", &l.bv)] {
                        v.push((format!("lora.{i}.{n}"), t.shape(), t.data()));
                    }
                }
            }
            Method::Rga(r) => {
                for (gi, g) in r.genomes.iter().enumerate() {
                    v.push((format!("genome.{gi}"), g.shape(), g.data()));
                }
                v.push(("wg".into(), r.wg.shape(), r.wg.data()));
                for (i, l) in r.layers.iter().enumerate() {
                    v.push((format!("rga.{i}.wr"), l.wr.shape(), l.wr.data()));
                    for kk in 0..r.k {
                        v.push((
                            format!("rga.{i}.down.{kk}"),
                            l.down[kk].shape(),
                            l.down[kk].data(),
                        ));
                        v.push((
                            format!("rga.{i}.up.{kk}"),
                            l.up[kk].shape(),
                            l.up[kk].data(),
                        ));
                    }
                }
            }
        }
        // grown FFN dims (Net2Net width)
        for (i, gl) in self.grow.iter().enumerate() {
            v.push((
                format!("grow.{i}.gate"),
                gl.gate_new.shape(),
                gl.gate_new.data(),
            ));
            v.push((format!("grow.{i}.up"), gl.up_new.shape(), gl.up_new.data()));
            v.push((
                format!("grow.{i}.down"),
                gl.down_new.shape(),
                gl.down_new.data(),
            ));
        }
        v
    }
}
