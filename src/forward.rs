//! Transformer forward pass with a single-sequence KV cache.
//!
//! GGUF Llama weights are permuted at conversion time to match ggml's interleaved
//! ("GPT-J"-style) RoPE, so we rotate adjacent pairs (x[2i], x[2i+1]).

use crate::metal_be::Gpu;
use crate::model::{cpu_matmat, cpu_matmul, Config, Model};
use std::cell::RefCell;

thread_local! {
    static CALIB: RefCell<Option<Calib>> = const { RefCell::new(None) };
}

/// Per-layer input-channel activation salience for AWQ calibration. Opt-in
/// (off by default), so the inference path is unaffected when not calibrating.
#[derive(Default)]
pub struct Calib {
    pub attn_in: Vec<Vec<f64>>, // [layer][ch] Σ|attn-norm out|        (input to wq/wk/wv)
    pub ffn_in: Vec<Vec<f64>>,  // [layer][ch] Σ|ffn-norm out|         (input to gate/up)
    pub ffn_inter: Vec<Vec<f64>>, // [layer][ch] Σ|silu(gate)·up|, len ffn (input to down)
    pub attn_out: Vec<Vec<f64>>, // [layer][ch] Σ|attention context|, len q_dim (input to wo)
    pub tokens: usize,
}

/// Begin recording activation salience for the next forward passes. `attn_out`
/// is lazily sized on first record (its length q_dim isn't always derivable
/// from GGUF metadata alone).
pub fn calib_start(n_layers: usize, dim: usize, ffn: usize) {
    CALIB.with(|c| {
        *c.borrow_mut() = Some(Calib {
            attn_in: vec![vec![0.0; dim]; n_layers],
            ffn_in: vec![vec![0.0; dim]; n_layers],
            ffn_inter: vec![vec![0.0; ffn]; n_layers],
            attn_out: vec![Vec::new(); n_layers],
            tokens: 0,
        });
    });
}

/// Stop recording and return the accumulated salience.
pub fn calib_take() -> Option<Calib> {
    CALIB.with(|c| c.borrow_mut().take())
}

/// Per-token representation capture: the residual stream after each layer plus the
/// final post-output-norm hidden. Overwritten every token, so after a prompt it
/// holds the LAST token's representation at every depth.
struct HTap {
    per_layer: Vec<Vec<f32>>,
    final_h: Vec<f32>,
}

thread_local! {
    static HTAP: RefCell<Option<HTap>> = const { RefCell::new(None) };
}

/// Arm the hidden-state tap: the next `forward()` calls record the residual at
/// every layer and the final hidden state. Opt-in and off by default, so the
/// inference path is unaffected. Used by representation-alignment tooling
/// (`--stitch`) to pull per-depth representation vectors out of a frozen model.
pub fn htap_arm() {
    HTAP.with(|h| {
        *h.borrow_mut() = Some(HTap {
            per_layer: Vec::new(),
            final_h: Vec::new(),
        })
    });
}

/// Take the final (post-output-norm, pre-lm-head) hidden captured since arming.
pub fn htap_take() -> Option<Vec<f32>> {
    HTAP.with(|h| h.borrow_mut().take().map(|t| t.final_h))
}

/// Take the per-layer residual stream captured since arming (one vector per layer).
pub fn htap_take_layers() -> Option<Vec<Vec<f32>>> {
    HTAP.with(|h| h.borrow_mut().take().map(|t| t.per_layer))
}

#[inline]
fn htap_rec_layer(l: usize, x: &[f32]) {
    HTAP.with(|h| {
        if let Some(t) = h.borrow_mut().as_mut() {
            if l == t.per_layer.len() {
                t.per_layer.push(x.to_vec());
            } else if l < t.per_layer.len() {
                let s = &mut t.per_layer[l];
                s.clear();
                s.extend_from_slice(x);
            }
        }
    });
}

#[inline]
fn htap_rec_final(x: &[f32]) {
    HTAP.with(|h| {
        if let Some(t) = h.borrow_mut().as_mut() {
            t.final_h.clear();
            t.final_h.extend_from_slice(x);
        }
    });
}

/// flwr arch: snap the final hidden state onto the E8 lattice (block=8, scale=1.0)
/// before the lm-head — the inference twin of `FtModel::forward_vq` at strength 1.
///
/// `pub(crate)` so the Metal head can replay the EXACT same snap on the hidden it
/// downloads from the GPU — sharing this one definition keeps the CPU and GPU
/// paths from drifting (block size, lattice, and order must match bit-for-bit).
pub(crate) fn flwr_e8_quant(x: &mut [f32]) {
    const B: usize = 8;
    let mut i = 0;
    while i + B <= x.len() {
        let c = crate::nn::nearest_e8(&x[i..i + B]);
        x[i..i + B].copy_from_slice(&c);
        i += B;
    }
}

#[inline]
fn calib_rec(which: usize, l: usize, v: &[f32]) {
    CALIB.with(|c| {
        if let Some(cal) = c.borrow_mut().as_mut() {
            let acc = match which {
                0 => &mut cal.attn_in,
                1 => &mut cal.ffn_in,
                2 => &mut cal.ffn_inter,
                _ => &mut cal.attn_out,
            };
            if let Some(row) = acc.get_mut(l) {
                if row.is_empty() {
                    row.resize(v.len(), 0.0); // lazy-size (attn_out)
                }
                for (a, &x) in row.iter_mut().zip(v) {
                    *a += x.abs() as f64;
                }
            }
            if which == 0 && l == 0 {
                cal.tokens += 1;
            }
        }
    });
}

pub struct State {
    // KV cache: [layer][pos * (n_kv_heads*head_dim) + ...]
    k_cache: Vec<Vec<f32>>,
    v_cache: Vec<Vec<f32>>,
    pub pos: usize,
}

impl State {
    pub fn new(cfg: &Config) -> State {
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        State {
            k_cache: vec![vec![0.0; cfg.ctx_len * kv_dim]; cfg.n_layers],
            v_cache: vec![vec![0.0; cfg.ctx_len * kv_dim]; cfg.n_layers],
            pos: 0,
        }
    }
}

/// RMSNorm. `add_one` applies the weight as (1 + w) — Gemma's convention.
/// Safe to call in place (out may alias x).
fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32, add_one: bool) {
    let n = x.len();
    let ss = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let scale = 1.0 / (ss + eps).sqrt();
    for i in 0..n {
        let w = if add_one { weight[i] + 1.0 } else { weight[i] };
        out[i] = x[i] * scale * w;
    }
}

/// In-place RMSNorm (no add_one) — used for OLMoE's QK-norm on the full q/k vectors.
fn rmsnorm_inplace(v: &mut [f32], weight: &[f32], eps: f32) {
    let ss = v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32;
    let scale = 1.0 / (ss + eps).sqrt();
    for i in 0..v.len() {
        v[i] *= scale * weight[i];
    }
}

/// GELU (tanh approximation) — Gemma's GeGLU gate activation.
fn gelu(x: f32) -> f32 {
    let c = (2.0f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

/// Soft-cap: c·tanh(v/c) when c > 0, else identity (Gemma2 logit capping).
#[inline]
fn softcap(v: f32, c: f32) -> f32 {
    if c > 0.0 {
        c * (v / c).tanh()
    } else {
        v
    }
}

/// RoPE per head at `pos`. `neox` selects rotate-halves (Qwen2/NEOX) vs
/// rotate-adjacent-pairs (Llama/interleaved).
fn rope(v: &mut [f32], head_dim: usize, pos: usize, base: f32, neox: bool) {
    let n_heads = v.len() / head_dim;
    let half = head_dim / 2;
    for h in 0..n_heads {
        let off = h * head_dim;
        for i in 0..half {
            let freq = 1.0 / base.powf(2.0 * i as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let (s, c) = angle.sin_cos();
            let (a, b) = if neox {
                (off + i, off + i + half)
            } else {
                (off + 2 * i, off + 2 * i + 1)
            };
            let (x0, x1) = (v[a], v[b]);
            v[a] = x0 * c - x1 * s;
            v[b] = x0 * s + x1 * c;
        }
    }
}

fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// Batched prefill for the dense CPU path: ingest all `tokens` in one pass,
/// filling the KV cache at positions `start..start+T` and returning the LAST
/// token's logits (what generation samples from). Returns `None` to signal the
/// caller to fall back to the token-by-token loop — for GPU-resident weights
/// (`gpu` set) or mixture-of-experts layers, which this path doesn't batch.
///
/// The win: the projection matmuls run through `cpu_matmat`, reading each weight
/// once and reusing it across all T tokens, instead of re-reading the whole model
/// T times. Every per-token op (norm/RoPE/QK-norm/attention/SwiGLU) matches
/// `forward` exactly, so the last-token logits are bit-identical to the loop.
#[allow(clippy::needless_range_loop)]
pub fn forward_prefill(
    model: &Model,
    state: &mut State,
    tokens: &[u32],
    start: usize,
    gpu: Option<&Gpu>,
) -> Option<Vec<f32>> {
    if gpu.is_some() || tokens.is_empty() {
        return None;
    }
    if model.layers.iter().any(|l| l.moe.is_some()) {
        return None;
    }
    let cfg = &model.cfg;
    let (dim, hd, ff) = (cfg.dim, cfg.head_dim, cfg.ffn_dim);
    let kv_dim = cfg.n_kv_heads * hd;
    let q_dim = cfg.n_heads * hd;
    let kv_mul = cfg.n_heads / cfg.n_kv_heads;
    let nrm = cfg.norm_add_one;
    let t = tokens.len();
    if start + t > cfg.ctx_len {
        return None;
    }

    // embed all tokens -> x[T, dim]
    let mut x = vec![0.0f32; t * dim];
    for (i, &tok) in tokens.iter().enumerate() {
        let dst = &mut x[i * dim..(i + 1) * dim];
        dst.copy_from_slice(&model.tok_embd[tok as usize * dim..(tok as usize + 1) * dim]);
        if cfg.embed_scale != 1.0 {
            for v in dst.iter_mut() {
                *v *= cfg.embed_scale;
            }
        }
    }
    let mut xb = vec![0.0f32; t * dim];
    let mut q = vec![0.0f32; t * q_dim];
    let mut k = vec![0.0f32; t * kv_dim];
    let mut v = vec![0.0f32; t * kv_dim];
    let mut att = vec![0.0f32; t * q_dim];
    let scale = 1.0 / (hd as f32).sqrt();

    for (l, layer) in model.layers.iter().enumerate() {
        // attention input norm (per row) -> projections (batched, weight-reused)
        for i in 0..t {
            rmsnorm(
                &mut xb[i * dim..(i + 1) * dim],
                &x[i * dim..(i + 1) * dim],
                &layer.attn_norm,
                cfg.rms_eps,
                nrm,
            );
        }
        cpu_matmat(&mut q, &layer.wq, &xb, t);
        cpu_matmat(&mut k, &layer.wk, &xb, t);
        cpu_matmat(&mut v, &layer.wv, &xb, t);
        // per-row bias + QK-norm + RoPE (pos = start + i)
        for i in 0..t {
            let qr = &mut q[i * q_dim..(i + 1) * q_dim];
            if let Some(b) = &layer.bq {
                qr.iter_mut().zip(b).for_each(|(a, b)| *a += b);
            }
            if let Some(qn) = &layer.q_norm {
                rmsnorm_inplace(qr, qn, cfg.rms_eps);
            }
            rope(qr, hd, start + i, cfg.rope_base, cfg.rope_neox);
            let kr = &mut k[i * kv_dim..(i + 1) * kv_dim];
            if let Some(b) = &layer.bk {
                kr.iter_mut().zip(b).for_each(|(a, b)| *a += b);
            }
            if let Some(kn) = &layer.k_norm {
                rmsnorm_inplace(kr, kn, cfg.rms_eps);
            }
            rope(kr, hd, start + i, cfg.rope_base, cfg.rope_neox);
            if let Some(b) = &layer.bv {
                v[i * kv_dim..(i + 1) * kv_dim]
                    .iter_mut()
                    .zip(b)
                    .for_each(|(a, b)| *a += b);
            }
        }
        // write all K,V into the cache, then attend (causal: token i sees 0..=start+i)
        let kc = &mut state.k_cache[l];
        let vc = &mut state.v_cache[l];
        for i in 0..t {
            let p = start + i;
            kc[p * kv_dim..(p + 1) * kv_dim].copy_from_slice(&k[i * kv_dim..(i + 1) * kv_dim]);
            vc[p * kv_dim..(p + 1) * kv_dim].copy_from_slice(&v[i * kv_dim..(i + 1) * kv_dim]);
        }
        for i in 0..t {
            let pos = start + i;
            for h in 0..cfg.n_heads {
                let qh = &q[i * q_dim + h * hd..i * q_dim + (h + 1) * hd];
                let kvh = h / kv_mul;
                let mut scores = vec![0.0f32; pos + 1];
                for tt in 0..=pos {
                    let kh = &kc[tt * kv_dim + kvh * hd..tt * kv_dim + (kvh + 1) * hd];
                    let mut d = 0.0;
                    for z in 0..hd {
                        d += qh[z] * kh[z];
                    }
                    scores[tt] = softcap(d * scale, cfg.attn_softcap);
                }
                softmax(&mut scores);
                let oh = &mut att[i * q_dim + h * hd..i * q_dim + (h + 1) * hd];
                oh.iter_mut().for_each(|o| *o = 0.0);
                for tt in 0..=pos {
                    let vh = &vc[tt * kv_dim + kvh * hd..tt * kv_dim + (kvh + 1) * hd];
                    let w = scores[tt];
                    for z in 0..hd {
                        oh[z] += w * vh[z];
                    }
                }
            }
        }
        // attention output + residual (with optional Gemma sandwich norm)
        cpu_matmat(&mut xb, &layer.wo, &att, t);
        for i in 0..t {
            if let Some(pn) = &layer.post_attn_norm {
                let mut nb = vec![0.0f32; dim];
                rmsnorm(&mut nb, &xb[i * dim..(i + 1) * dim], pn, cfg.rms_eps, nrm);
                xb[i * dim..(i + 1) * dim].copy_from_slice(&nb);
            }
            for z in 0..dim {
                x[i * dim + z] += xb[i * dim + z];
            }
        }
        // dense FFN (SwiGLU/GeGLU), batched
        for i in 0..t {
            rmsnorm(
                &mut xb[i * dim..(i + 1) * dim],
                &x[i * dim..(i + 1) * dim],
                &layer.ffn_norm,
                cfg.rms_eps,
                nrm,
            );
        }
        let mut gate = vec![0.0f32; t * ff];
        let mut up = vec![0.0f32; t * ff];
        cpu_matmat(&mut gate, &layer.w_gate, &xb, t);
        cpu_matmat(&mut up, &layer.w_up, &xb, t);
        for j in 0..t * ff {
            let g = gate[j];
            let act = if cfg.geglu {
                gelu(g)
            } else {
                g / (1.0 + (-g).exp())
            };
            gate[j] = act * up[j];
        }
        let mut ffn_out = vec![0.0f32; t * dim];
        cpu_matmat(&mut ffn_out, &layer.w_down, &gate, t);
        for i in 0..t {
            if let Some(pn) = &layer.post_ffn_norm {
                let mut nb = vec![0.0f32; dim];
                rmsnorm(
                    &mut nb,
                    &ffn_out[i * dim..(i + 1) * dim],
                    pn,
                    cfg.rms_eps,
                    nrm,
                );
                ffn_out[i * dim..(i + 1) * dim].copy_from_slice(&nb);
            }
            for z in 0..dim {
                x[i * dim + z] += ffn_out[i * dim + z];
            }
        }
    }

    // final norm + lm head on the LAST token only (what generation samples)
    let mut xbn = vec![0.0f32; dim];
    rmsnorm(
        &mut xbn,
        &x[(t - 1) * dim..t * dim],
        &model.output_norm,
        cfg.rms_eps,
        nrm,
    );
    if matches!(cfg.arch, crate::model::Arch::Flwr) {
        flwr_e8_quant(&mut xbn); // flwr: E8 lattice bottleneck before the head
    }
    let mut logits = vec![0.0f32; cfg.vocab_size];
    model.output.matvec(None, &xbn, &mut logits);
    if cfg.final_softcap > 0.0 {
        for v in logits.iter_mut() {
            *v = softcap(*v, cfg.final_softcap);
        }
    }
    state.pos = start + t;
    Some(logits)
}

/// Run one token through the model, return logits over the vocabulary.
/// Matmuls run on `gpu` when the model's weights are GPU-resident; the cheap
/// ops (norm, rope, attention, swiglu) stay on CPU for now.
pub fn forward(
    model: &Model,
    state: &mut State,
    token: u32,
    pos: usize,
    gpu: Option<&Gpu>,
) -> Vec<f32> {
    let cfg = &model.cfg;
    let dim = cfg.dim;
    let hd = cfg.head_dim;
    let kv_dim = cfg.n_kv_heads * hd;
    let q_dim = cfg.n_heads * hd;
    let kv_mul = cfg.n_heads / cfg.n_kv_heads; // GQA group size

    let nrm = cfg.norm_add_one;
    let mut x = model.tok_embd[token as usize * dim..(token as usize + 1) * dim].to_vec();
    if cfg.embed_scale != 1.0 {
        for v in &mut x {
            *v *= cfg.embed_scale;
        }
    }
    let mut xb = vec![0.0f32; dim];
    let mut nb = vec![0.0f32; dim]; // scratch for sandwich (post-block) norms
    let mut q = vec![0.0f32; q_dim];
    let mut k = vec![0.0f32; kv_dim];
    let mut v = vec![0.0f32; kv_dim];
    let mut att_out = vec![0.0f32; q_dim];

    for (l, layer) in model.layers.iter().enumerate() {
        // --- attention ---
        rmsnorm(&mut xb, &x, &layer.attn_norm, cfg.rms_eps, nrm);
        calib_rec(0, l, &xb);
        layer.wq.matvec(gpu, &xb, &mut q);
        layer.wk.matvec(gpu, &xb, &mut k);
        layer.wv.matvec(gpu, &xb, &mut v);
        if let Some(b) = &layer.bq {
            q.iter_mut().zip(b).for_each(|(x, b)| *x += b);
        }
        if let Some(b) = &layer.bk {
            k.iter_mut().zip(b).for_each(|(x, b)| *x += b);
        }
        if let Some(b) = &layer.bv {
            v.iter_mut().zip(b).for_each(|(x, b)| *x += b);
        }
        // OLMoE QK-norm: RMSNorm the full q/k projections before RoPE
        if let Some(qn) = &layer.q_norm {
            rmsnorm_inplace(&mut q, qn, cfg.rms_eps);
        }
        if let Some(kn) = &layer.k_norm {
            rmsnorm_inplace(&mut k, kn, cfg.rms_eps);
        }
        rope(&mut q, hd, pos, cfg.rope_base, cfg.rope_neox);
        rope(&mut k, hd, pos, cfg.rope_base, cfg.rope_neox);

        // write k,v into cache at this position
        let kc = &mut state.k_cache[l];
        let vc = &mut state.v_cache[l];
        kc[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&k);
        vc[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&v);

        let scale = 1.0 / (hd as f32).sqrt();
        for h in 0..cfg.n_heads {
            let qh = &q[h * hd..(h + 1) * hd];
            let kvh = h / kv_mul; // which kv head this query head attends to
            let mut scores = vec![0.0f32; pos + 1];
            for t in 0..=pos {
                let kh = &kc[t * kv_dim + kvh * hd..t * kv_dim + (kvh + 1) * hd];
                let mut dot = 0.0;
                for i in 0..hd {
                    dot += qh[i] * kh[i];
                }
                scores[t] = softcap(dot * scale, cfg.attn_softcap);
            }
            softmax(&mut scores);
            let oh = &mut att_out[h * hd..(h + 1) * hd];
            oh.iter_mut().for_each(|o| *o = 0.0);
            for t in 0..=pos {
                let vh = &vc[t * kv_dim + kvh * hd..t * kv_dim + (kvh + 1) * hd];
                let w = scores[t];
                for i in 0..hd {
                    oh[i] += w * vh[i];
                }
            }
        }
        calib_rec(3, l, &att_out);
        layer.wo.matvec(gpu, &att_out, &mut xb);
        // Gemma2 "sandwich": normalize the attention block output before the residual
        if let Some(pn) = &layer.post_attn_norm {
            rmsnorm(&mut nb, &xb, pn, cfg.rms_eps, nrm);
            xb.copy_from_slice(&nb);
        }
        for i in 0..dim {
            x[i] += xb[i];
        }

        // --- feed-forward: dense (SwiGLU/GeGLU) or mixture-of-experts ---
        rmsnorm(&mut xb, &x, &layer.ffn_norm, cfg.rms_eps, nrm);
        calib_rec(1, l, &xb);
        let mut ffn_out = vec![0.0f32; dim];
        if let Some(m) = &layer.moe {
            // router -> softmax -> top-k experts -> renormalized weighted sum
            let (ne, topk, ff) = (cfg.n_experts, cfg.n_experts_used, cfg.ffn_dim);
            let mut logits = vec![0.0f32; ne];
            cpu_matmul(&mut logits, &m.router, &xb);
            softmax(&mut logits);
            let mut idx: Vec<usize> = (0..ne).collect();
            idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
            idx.truncate(topk);
            // OLMoE uses the raw softmax probabilities of the selected experts
            // (norm_topk_prob = false), i.e. no renormalization to sum 1.
            let mut g = vec![0.0f32; ff];
            let mut u = vec![0.0f32; ff];
            let mut de = vec![0.0f32; dim];
            let mut wbuf = vec![0.0f32; ff * dim]; // one dequantized expert matrix
            for &e in &idx {
                let w = logits[e];
                m.gate.dequant_expert(e, &mut wbuf);
                cpu_matmul(&mut g, &wbuf[..ff * dim], &xb);
                m.up.dequant_expert(e, &mut wbuf);
                cpu_matmul(&mut u, &wbuf[..ff * dim], &xb);
                for j in 0..ff {
                    let s = g[j] / (1.0 + (-g[j]).exp());
                    g[j] = s * u[j];
                }
                m.down.dequant_expert(e, &mut wbuf);
                cpu_matmul(&mut de, &wbuf[..dim * ff], &g);
                for j in 0..dim {
                    ffn_out[j] += w * de[j];
                }
            }
        } else {
            let mut gate = vec![0.0f32; cfg.ffn_dim];
            let mut up = vec![0.0f32; cfg.ffn_dim];
            layer.w_gate.matvec(gpu, &xb, &mut gate);
            layer.w_up.matvec(gpu, &xb, &mut up);
            for i in 0..cfg.ffn_dim {
                let g = gate[i];
                let act = if cfg.geglu {
                    gelu(g)
                } else {
                    g / (1.0 + (-g).exp())
                };
                gate[i] = act * up[i];
            }
            calib_rec(2, l, &gate);
            layer.w_down.matvec(gpu, &gate, &mut ffn_out);
        }
        if let Some(pn) = &layer.post_ffn_norm {
            rmsnorm(&mut nb, &ffn_out, pn, cfg.rms_eps, nrm);
            ffn_out.copy_from_slice(&nb);
        }
        for i in 0..dim {
            x[i] += ffn_out[i];
        }
        // representation tap: residual stream after this layer (per-depth anchor).
        htap_rec_layer(l, &x);
    }

    rmsnorm(&mut xb, &x, &model.output_norm, cfg.rms_eps, nrm);
    if matches!(cfg.arch, crate::model::Arch::Flwr) {
        flwr_e8_quant(&mut xb); // flwr: E8 lattice bottleneck before the head
    }
    // representation tap: the model's final hidden state for this token.
    htap_rec_final(&xb);
    let mut logits = vec![0.0f32; cfg.vocab_size];
    model.output.matvec(gpu, &xb, &mut logits);
    if cfg.final_softcap > 0.0 {
        for v in logits.iter_mut() {
            *v = softcap(*v, cfg.final_softcap);
        }
    }
    state.pos = pos + 1;
    logits
}
