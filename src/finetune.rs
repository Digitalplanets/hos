//! T1 — full-parameter finetuning of a real pretrained transformer.
//!
//! Loads a model's weights as trainable autograd `Tensor`s (reusing the same
//! loader the inference path uses, so GGUF *and* HuggingFace both work) and
//! builds a real Llama-family forward — RMSNorm, configurable RoPE, grouped-query
//! attention, SwiGLU — out of `tensor.rs` ops, so gradients flow to every weight.
//!
//! Scope (T1): Llama / Mistral / Qwen2 (the configurable knobs `rope_neox` and
//! `attn_bias` cover the differences). Gemma soft-caps / sandwich norms, Phi-3
//! fusion, and MoE are deferred. RMSNorm eps is the autograd default (1e-5) —
//! exact for Llama/SmolLM2/Mistral.
//!
//! The flexibility principle: RoPE style and the optimizer are *config*, not
//! hardwired — when finetuning a foreign model we derive them from its metadata,
//! we do not bend HOS's design around it.

use crate::model::{Config, Model, Weight};
use crate::tensor::Tensor;

fn param(data: Vec<f32>, shape: &[usize]) -> Tensor {
    Tensor::param(data, shape)
}

/// Transpose a row-major `[out, in]` weight to `[in, out]` so it can be used as
/// `x[T,in] @ W[in,out]` in the autograd matmul.
fn transposed_param(w: &Weight, out: usize, inn: usize) -> Tensor {
    let d = w.to_f32();
    let mut t = vec![0.0; out * inn];
    for o in 0..out {
        for i in 0..inn {
            t[i * out + o] = d[o * inn + i];
        }
    }
    param(t, &[inn, out])
}

struct FtLayer {
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

pub struct FtModel {
    pub cfg: Config,
    tok_embd: Tensor,
    layers: Vec<FtLayer>,
    out_norm: Tensor,
    out: Tensor, // [dim, vocab]
}

impl FtModel {
    /// Wrap an already-loaded (CPU) `Model`'s weights as trainable tensors.
    pub fn from_model(m: &Model) -> Result<FtModel, String> {
        let c = &m.cfg;
        if !matches!(
            c.arch,
            crate::model::Arch::Llama | crate::model::Arch::Mistral | crate::model::Arch::Qwen2
        ) {
            return Err(format!(
                "finetuning supports Llama/Mistral/Qwen2 today; {:?} needs its extra ops (Gemma soft-cap, Phi-3 fusion, MoE)",
                c.arch
            ));
        }
        if m.layers.iter().any(|l| l.moe.is_some()) {
            return Err("MoE finetuning not supported yet".into());
        }
        let (dim, qd, kvd, ffn) = (
            c.dim,
            c.n_heads * c.head_dim,
            c.n_kv_heads * c.head_dim,
            c.ffn_dim,
        );
        let ob = |o: &Option<Vec<f32>>| o.as_ref().map(|b| param(b.clone(), &[b.len()]));
        let layers = m
            .layers
            .iter()
            .map(|l| FtLayer {
                attn_norm: param(l.attn_norm.clone(), &[dim]),
                wq: transposed_param(&l.wq, qd, dim),
                wk: transposed_param(&l.wk, kvd, dim),
                wv: transposed_param(&l.wv, kvd, dim),
                wo: transposed_param(&l.wo, dim, qd),
                bq: ob(&l.bq),
                bk: ob(&l.bk),
                bv: ob(&l.bv),
                ffn_norm: param(l.ffn_norm.clone(), &[dim]),
                w_gate: transposed_param(&l.w_gate, ffn, dim),
                w_up: transposed_param(&l.w_up, ffn, dim),
                w_down: transposed_param(&l.w_down, dim, ffn),
            })
            .collect();
        Ok(FtModel {
            cfg: c.clone(),
            tok_embd: param(m.tok_embd.clone(), &[c.vocab_size, dim]),
            layers,
            out_norm: param(m.output_norm.clone(), &[dim]),
            out: transposed_param(&m.output, c.vocab_size, dim),
        })
    }

    /// Net2Net **function-preserving** FFN widening. Seeds from `m` (via
    /// `from_model`), then grows every layer's FFN from `ffn_dim` to `new_ffn`:
    /// each new unit replicates a random existing unit, and the replicated units'
    /// down-projection rows are split (÷ replica count) so the layer's output is
    /// **unchanged at init** — capability inherited losslessly. Tiny noise on the
    /// new gate columns breaks symmetry so the copies diverge during training.
    pub fn from_model_grown(m: &Model, new_ffn: usize) -> Result<FtModel, String> {
        let mut ft = FtModel::from_model(m)?;
        let dim = ft.cfg.dim;
        let ffn = ft.cfg.ffn_dim;
        if new_ffn <= ffn {
            return Ok(ft);
        }
        let extra = new_ffn - ffn;
        let mut s = 0x9E3779B97F4A7C15u64;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for l in &mut ft.layers {
            let gate = l.w_gate.data(); // [dim, ffn]
            let up = l.w_up.data(); // [dim, ffn]
            let down = l.w_down.data(); // [ffn, dim]
            let mut rep = vec![0usize; extra];
            let mut count = vec![1usize; ffn];
            for k in 0..extra {
                let j = (rng() as usize) % ffn;
                rep[k] = j;
                count[j] += 1;
            }
            // widened gate/up: [dim, new_ffn], original cols copied, new cols replicate
            let mut ng = vec![0f32; dim * new_ffn];
            let mut nu = vec![0f32; dim * new_ffn];
            for i in 0..dim {
                for u in 0..ffn {
                    ng[i * new_ffn + u] = gate[i * ffn + u];
                    nu[i * new_ffn + u] = up[i * ffn + u];
                }
                for (k, &j) in rep.iter().enumerate() {
                    let u = ffn + k;
                    let noise = 0.001 * ((rng() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0);
                    ng[i * new_ffn + u] = gate[i * ffn + j] * (1.0 + noise);
                    nu[i * new_ffn + u] = up[i * ffn + j];
                }
            }
            // widened down: [new_ffn, dim], rows split by replica count (preserves sum)
            let mut nd = vec![0f32; new_ffn * dim];
            for u in 0..ffn {
                let c = count[u] as f32;
                for jd in 0..dim {
                    nd[u * dim + jd] = down[u * dim + jd] / c;
                }
            }
            for (k, &j) in rep.iter().enumerate() {
                let u = ffn + k;
                let c = count[j] as f32;
                for jd in 0..dim {
                    nd[u * dim + jd] = down[j * dim + jd] / c;
                }
            }
            l.w_gate = param(ng, &[dim, new_ffn]);
            l.w_up = param(nu, &[dim, new_ffn]);
            l.w_down = param(nd, &[new_ffn, dim]);
        }
        ft.cfg.ffn_dim = new_ffn;
        Ok(ft)
    }

    /// Every trainable parameter (for the optimizer).
    pub fn params(&self) -> Vec<&Tensor> {
        let mut p = vec![&self.tok_embd, &self.out_norm, &self.out];
        for l in &self.layers {
            p.extend([
                &l.attn_norm,
                &l.wq,
                &l.wk,
                &l.wv,
                &l.wo,
                &l.ffn_norm,
                &l.w_gate,
                &l.w_up,
                &l.w_down,
            ]);
            for b in [&l.bq, &l.bk, &l.bv].into_iter().flatten() {
                p.push(b);
            }
        }
        p
    }

    /// Snapshot of the (trained) weights for saving: (name, shape, data). Weights
    /// are stored transposed `[in, out]`; this is a lineaged provenance capsule of
    /// the finetune, tagged `hos-finetune` in its card.
    pub fn named_tensors(&self) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        let mut v = vec![
            (
                "tok_embd".to_string(),
                self.tok_embd.shape(),
                self.tok_embd.data(),
            ),
            (
                "output_norm".to_string(),
                self.out_norm.shape(),
                self.out_norm.data(),
            ),
            ("output".to_string(), self.out.shape(), self.out.data()),
        ];
        for (i, l) in self.layers.iter().enumerate() {
            for (suf, t) in [
                ("attn_norm", &l.attn_norm),
                ("wq", &l.wq),
                ("wk", &l.wk),
                ("wv", &l.wv),
                ("wo", &l.wo),
                ("ffn_norm", &l.ffn_norm),
                ("w_gate", &l.w_gate),
                ("w_up", &l.w_up),
                ("w_down", &l.w_down),
            ] {
                v.push((format!("blk.{i}.{suf}"), t.shape(), t.data()));
            }
            for (suf, b) in [("bq", &l.bq), ("bk", &l.bk), ("bv", &l.bv)] {
                if let Some(b) = b {
                    v.push((format!("blk.{i}.{suf}"), b.shape(), b.data()));
                }
            }
        }
        v
    }

    /// Final hidden state `[T, dim]` for `ids` (full autograd graph). `forward` and
    /// `forward_vq` add the output head on top.
    pub fn hidden(&self, ids: &[usize]) -> Tensor {
        let c = &self.cfg;
        let (t, hd, nh, nkv) = (ids.len(), c.head_dim, c.n_heads, c.n_kv_heads);
        let qd = nh * hd;
        let pos: Vec<usize> = (0..t).collect();
        let scale = 1.0 / (hd as f32).sqrt();
        let group = nh / nkv;

        // tiled causal mask [nh*t, t]
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

        let mut x = self.tok_embd.embedding(ids); // [t, dim]
        for l in &self.layers {
            // --- attention ---
            let xn = x.rmsnorm_eps(&l.attn_norm, c.rms_eps);
            let mut q = xn.matmul(&l.wq);
            let mut k = xn.matmul(&l.wk);
            let v = xn.matmul(&l.wv);
            if let Some(b) = &l.bq {
                q = q.add(b);
            }
            if let Some(b) = &l.bk {
                k = k.add(b);
            }
            let v = if let Some(b) = &l.bv { v.add(b) } else { v };
            let q = q.rope(nh, hd, c.rope_base, c.rope_neox, &pos);
            let k = k.rope(nkv, hd, c.rope_base, c.rope_neox, &pos);
            // [t, H*hd] -> [H, t, hd]
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
            let kh = kh.repeat_interleave_dim0(group); // GQA expand -> [H, t, hd]
            let vh = vh.repeat_interleave_dim0(group);
            let scores = qh.bmm(&kh.transpose_last2()).scale(scale); // [H, t, t]
            let att = scores
                .reshape(&[nh * t, t])
                .add(&mask)
                .softmax_rows()
                .reshape(&[nh, t, t]);
            let ctx = att.bmm(&vh); // [H, t, hd]
            let ctx = ctx
                .reshape(&[1, nh, t, hd])
                .transpose12_4d()
                .reshape(&[t, qd]);
            x = x.add(&ctx.matmul(&l.wo));
            // --- SwiGLU FFN ---
            let xn2 = x.rmsnorm_eps(&l.ffn_norm, c.rms_eps);
            let ff = xn2
                .matmul(&l.w_gate)
                .silu()
                .mul(&xn2.matmul(&l.w_up))
                .matmul(&l.w_down);
            x = x.add(&ff);
        }
        x.rmsnorm_eps(&self.out_norm, self.cfg.rms_eps) // [t, dim]
    }

    /// Standard logits `[t, vocab]` (Llama-family forward).
    pub fn forward(&self, ids: &[usize]) -> Tensor {
        self.hidden(ids).matmul(&self.out)
    }

    /// flwr forward: E8-quantize the final hidden (STE through the lattice) before
    /// the head. Returns (logits, commitment loss); `strength` anneals quant 0->1.
    pub fn forward_vq(
        &self,
        ids: &[usize],
        vq: &crate::nn::VectorQuantizer,
        strength: f32,
    ) -> (Tensor, Tensor) {
        let q = vq.quantize(&self.hidden(ids), strength);
        (q.output.matmul(&self.out), q.commitment)
    }
}

/// Correctness gate: the autograd-built forward must reproduce the hand-coded
/// inference forward (`forward.rs`) at the last position. Returns max abs diff
/// over the vocab logits — should be ~0 (same f32 weights, same math).
pub fn check_parity(m: &Model, ids: &[usize]) -> f32 {
    use crate::forward::{forward, State};
    let mut st = State::new(&m.cfg);
    let mut last = Vec::new();
    for (pos, &tok) in ids.iter().enumerate() {
        last = forward(m, &mut st, tok as u32, pos, None);
    }
    let ft = FtModel::from_model(m).expect("build FtModel");
    let logits = ft.forward(ids).data();
    let (t, vocab) = (ids.len(), m.cfg.vocab_size);
    let row = &logits[(t - 1) * vocab..t * vocab];
    let mut md = 0.0f32;
    for j in 0..vocab {
        md = md.max((row[j] - last[j]).abs());
    }
    md
}
