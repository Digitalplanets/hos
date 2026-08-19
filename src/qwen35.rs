//! Experimental: Qwen3.5 (`qwen35`) hybrid architecture.
//!
//! 32 blocks = QK-norm attention every `full_attention_interval`-th layer +
//! Gated-DeltaNet-style SSM blocks for the rest, each followed by a SwiGLU FFN.
//!
//! Status: B2 step 1 — config + per-block structure validation. The forward
//! pass (esp. the gated delta-rule recurrence) is WIP and built/verified
//! incrementally. Kept fully separate from the supported transformer path.

use crate::chat::{self, ChatFamily, Message};
use crate::error::{HosError, Result};
use crate::gguf::Gguf;
use crate::model::Arch;
use crate::tokenizer::Tokenizer;
use rayon::prelude::*;

#[derive(Debug)]
pub struct Cfg {
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub vocab: usize,
    pub rms_eps: f32,
    pub rope_base: f32,
    pub rope_dim: usize,
    pub full_attn_interval: usize,
    // ssm
    pub conv_kernel: usize,
    pub state_size: usize,
    pub group_count: usize,
    pub inner_size: usize,
    pub time_step_rank: usize,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum BlockKind {
    Attention,
    Ssm,
}

impl Cfg {
    pub fn from_gguf<S: crate::model::ModelSource>(g: &S) -> Result<Cfg> {
        let k = |s: &str| format!("qwen35.{s}");
        let need = |key: &str| {
            g.meta_u64(&k(key))
                .ok_or_else(|| HosError::MissingMeta(k(key)))
        };
        let dim = need("embedding_length")? as usize;
        // block_count includes trailing NextN/MTP draft-head blocks, which are
        // loaded for speculative decoding but NOT run in the main forward pass
        // (llama.cpp: n_layer() = n_layer_all - nextn_predict_layers).
        let n_blocks = need("block_count")? as usize;
        let n_nextn = g.meta_u64(&k("nextn_predict_layers")).unwrap_or(0) as usize;
        Ok(Cfg {
            dim,
            n_layers: n_blocks.saturating_sub(n_nextn),
            n_heads: need("attention.head_count")? as usize,
            n_kv_heads: need("attention.head_count_kv")? as usize,
            head_dim: g
                .meta_u64(&k("attention.key_length"))
                .unwrap_or(dim as u64 / 16) as usize,
            ffn_dim: need("feed_forward_length")? as usize,
            // vocab = embed element count / hidden; via ModelSource::raw so this works
            // for a GGUF, an HF checkpoint, or a .hos capsule alike (no `.tensors`).
            vocab: g
                .raw("token_embd.weight")
                .map(|(_, _, n)| n / dim)
                .unwrap_or(0),
            rms_eps: g
                .meta_f32(&k("attention.layer_norm_rms_epsilon"))
                .unwrap_or(1e-6),
            rope_base: g.meta_f32(&k("rope.freq_base")).unwrap_or(1e7),
            rope_dim: g.meta_u64(&k("rope.dimension_count")).unwrap_or(64) as usize,
            full_attn_interval: g.meta_u64(&k("full_attention_interval")).unwrap_or(4) as usize,
            conv_kernel: g.meta_u64(&k("ssm.conv_kernel")).unwrap_or(4) as usize,
            state_size: g.meta_u64(&k("ssm.state_size")).unwrap_or(128) as usize,
            group_count: g.meta_u64(&k("ssm.group_count")).unwrap_or(16) as usize,
            inner_size: g.meta_u64(&k("ssm.inner_size")).unwrap_or(0) as usize,
            time_step_rank: g.meta_u64(&k("ssm.time_step_rank")).unwrap_or(0) as usize,
        })
    }
}

/// Detect each block's kind by which tensors it carries.
pub fn block_kinds<S: crate::model::ModelSource>(g: &S, n_layers: usize) -> Vec<BlockKind> {
    (0..n_layers)
        .map(|i| {
            if g.has(&format!("blk.{i}.ssm_a")) {
                BlockKind::Ssm
            } else {
                BlockKind::Attention
            }
        })
        .collect()
}

/// Load + validate structure: confirm every expected tensor exists per block
/// kind and report the layout. Foundation for the forward pass.
pub fn validate(g: &Gguf) -> Result<()> {
    let cfg = Cfg::from_gguf(g)?;
    eprintln!("[qwen35] {cfg:#?}");

    let kinds = block_kinds(g, cfg.n_layers);
    let n_attn = kinds.iter().filter(|k| **k == BlockKind::Attention).count();
    let n_ssm = kinds.iter().filter(|k| **k == BlockKind::Ssm).count();

    let pattern: String = kinds
        .iter()
        .map(|k| if *k == BlockKind::Attention { 'A' } else { 's' })
        .collect();
    eprintln!(
        "[qwen35] {} layers: {n_attn} attention, {n_ssm} SSM",
        cfg.n_layers
    );
    eprintln!("[qwen35] layout (A=attn, s=ssm): {pattern}");

    let attn_tensors = [
        "attn_norm.weight",
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_q_norm.weight",
        "attn_k_norm.weight",
        "attn_output.weight",
        "post_attention_norm.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
    ];
    let ssm_tensors = [
        "attn_norm.weight",
        "attn_qkv.weight",
        "attn_gate.weight",
        "ssm_a",
        "ssm_conv1d.weight",
        "ssm_dt.bias",
        "ssm_alpha.weight",
        "ssm_beta.weight",
        "ssm_norm.weight",
        "ssm_out.weight",
        "post_attention_norm.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
    ];

    let mut missing = 0;
    for (i, kind) in kinds.iter().enumerate() {
        let expected: &[&str] = if *kind == BlockKind::Attention {
            &attn_tensors
        } else {
            &ssm_tensors
        };
        for t in expected {
            let name = format!("blk.{i}.{t}");
            if !g.has(&name) {
                eprintln!("[qwen35] MISSING blk.{i} ({kind:?}): {t}");
                missing += 1;
            }
        }
    }

    // print one example of each block kind's tensor shapes
    for &(kind, idx) in &[
        (
            BlockKind::Attention,
            kinds.iter().position(|k| *k == BlockKind::Attention),
        ),
        (
            BlockKind::Ssm,
            kinds.iter().position(|k| *k == BlockKind::Ssm),
        ),
    ]
    .iter()
    .filter_map(|(k, o)| o.map(|i| (*k, i)))
    .collect::<Vec<_>>()
    {
        eprintln!("[qwen35] --- example {kind:?} block (blk.{idx}) shapes ---");
        let mut names: Vec<&String> = g
            .tensors
            .keys()
            .filter(|n| n.starts_with(&format!("blk.{idx}.")))
            .collect();
        names.sort();
        for n in names {
            let t = &g.tensors[n];
            eprintln!(
                "[qwen35]   {}  dims={:?} type={}",
                n.strip_prefix(&format!("blk.{idx}.")).unwrap(),
                t.dims,
                t.ggml_type
            );
        }
    }

    if missing == 0 {
        eprintln!(
            "[qwen35] ✅ structure validated — all expected tensors present for every block."
        );
    } else {
        eprintln!(
            "[qwen35] ⚠️  {missing} expected tensors missing — layout assumption needs revisiting."
        );
    }
    Ok(())
}

// ============================================================================
// Forward pass (CPU). Port of llama.cpp's qwen35 + gated delta-net.
// ============================================================================

use crate::metal_be::Gpu;
use crate::model::{cpu_matmul, Weight};

// Big matmul weights are `Weight` (CPU f32 or GPU f16, runtime-selected).
// Norms, conv, ssm_a/dt, and the tiny beta/alpha projections stay on CPU.
struct AttnW {
    attn_norm: Vec<f32>,
    wq: Weight,
    wk: Weight,
    wv: Weight,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    wo: Weight,
    post_norm: Vec<f32>,
    ffn_gate: Weight,
    ffn_up: Weight,
    ffn_down: Weight,
}
struct LinW {
    attn_norm: Vec<f32>,
    wqkv: Weight,
    wz: Weight,
    ssm_beta: Vec<f32>,
    ssm_alpha: Vec<f32>,
    ssm_a: Vec<f32>,
    ssm_dt: Vec<f32>,
    conv1d: Vec<f32>,
    ssm_norm: Vec<f32>,
    ssm_out: Weight,
    post_norm: Vec<f32>,
    ffn_gate: Weight,
    ffn_up: Weight,
    ffn_down: Weight,
}
enum Block {
    Attn(AttnW),
    Lin(LinW),
}

/// The NextN / MTP (Multi-Token-Prediction) draft head: one full-attention
/// decoder block plus the fusion that folds the backbone's hidden state and the
/// last token's embedding into it. It reuses the backbone's `tok_embd` + `output`
/// (lm_head), so it is ~one layer of extra weight. Used for self-speculative
/// decoding — drafts the next token, the backbone verifies a 2-token batch.
struct MtpW {
    attn: AttnW,
    eh_proj: Weight,          // [2*dim -> dim] fuse(enorm(embed) ‖ hnorm(hidden))
    enorm: Vec<f32>,          // norm applied to the token embedding
    hnorm: Vec<f32>,          // norm applied to the backbone hidden state
    shared_head_norm: Vec<f32>, // final norm before the shared lm_head
}

pub struct Qwen35 {
    pub cfg: Cfg,
    tok_embd: Vec<f32>,
    blocks: Vec<Block>,
    output_norm: Vec<f32>,
    output: Weight,
    mtp: Option<MtpW>,
}

pub struct State {
    k_cache: Vec<Vec<f32>>, // per layer (attn): max_seq * n_kv_heads*head_dim
    v_cache: Vec<Vec<f32>>,
    conv: Vec<Vec<f32>>, // per layer (lin): conv_channels * (K-1)
    ssm: Vec<Vec<f32>>,  // per layer (lin): num_v_heads * head_v_dim^2
    // MTP draft-head KV cache (one extra attention layer, used only for drafting).
    mtp_k: Vec<f32>,
    mtp_v: Vec<f32>,
    // Per-layer SSM/conv snapshot taken AFTER the first token of a 2-token verify,
    // so a rejected draft rolls back to "after the confirmed token" with no reprocess.
    snap_ssm: Vec<Vec<f32>>,
    snap_conv: Vec<Vec<f32>>,
    pub pos: usize,
}

const MAX_SEQ: usize = 4096;

impl State {
    pub fn new(m: &Qwen35) -> State {
        let c = &m.cfg;
        let kv_dim = c.n_kv_heads * c.head_dim;
        let conv_ch = c.inner_size + 2 * c.group_count * c.state_size;
        let hv = c.inner_size / c.time_step_rank; // head_v_dim
        State {
            k_cache: vec![vec![0.0; MAX_SEQ * kv_dim]; c.n_layers],
            v_cache: vec![vec![0.0; MAX_SEQ * kv_dim]; c.n_layers],
            conv: vec![vec![0.0; conv_ch * (c.conv_kernel - 1)]; c.n_layers],
            ssm: vec![vec![0.0; c.time_step_rank * hv * hv]; c.n_layers],
            mtp_k: vec![0.0; MAX_SEQ * kv_dim],
            mtp_v: vec![0.0; MAX_SEQ * kv_dim],
            snap_ssm: vec![Vec::new(); c.n_layers],
            snap_conv: vec![Vec::new(); c.n_layers],
            pos: 0,
        }
    }

    /// Snapshot the recurrent SSM + conv state (per layer) for MTP rollback. The
    /// attention KV cache doesn't need snapshotting — rejected positions are simply
    /// overwritten by the next real token at the same position.
    fn snapshot_ssm(&self) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        (self.ssm.clone(), self.conv.clone())
    }
    fn restore_ssm(&mut self, snap: (Vec<Vec<f32>>, Vec<Vec<f32>>)) {
        self.ssm = snap.0;
        self.conv = snap.1;
    }
}

fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let ss = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let s = 1.0 / (ss + eps).sqrt();
    (0..n).map(|i| x[i] * s * w[i]).collect()
}
fn rmsnorm_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    let n = x.len();
    let ss = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let s = 1.0 / (ss + eps).sqrt();
    for i in 0..n {
        x[i] = x[i] * s * w[i];
    }
}
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
fn mm(w: &[f32], x: &[f32], out_dim: usize) -> Vec<f32> {
    let mut y = vec![0.0; out_dim];
    cpu_matmul(&mut y, w, x);
    y
}
fn mmw(w: &Weight, x: &[f32], out_dim: usize, gpu: Option<&Gpu>) -> Vec<f32> {
    let mut y = vec![0.0; out_dim];
    w.matvec(gpu, x, &mut y);
    y
}
// NEOX partial rope: rotate first n_rot dims of each head (pairs i, i+n_rot/2)
fn rope(v: &mut [f32], n_heads: usize, head_dim: usize, n_rot: usize, pos: usize, base: f32) {
    let half = n_rot / 2;
    for h in 0..n_heads {
        let off = h * head_dim;
        for i in 0..half {
            let freq = 1.0 / base.powf(2.0 * i as f32 / n_rot as f32);
            let (s, c) = (pos as f32 * freq).sin_cos();
            let (a, b) = (off + i, off + i + half);
            let (x0, x1) = (v[a], v[b]);
            v[a] = x0 * c - x1 * s;
            v[b] = x0 * s + x1 * c;
        }
    }
}

impl Qwen35 {
    pub fn load<S: crate::model::ModelSource>(g: &S, gpu: Option<&Gpu>) -> Result<Qwen35> {
        let cfg = Cfg::from_gguf(g)?;
        let kinds = block_kinds(g, cfg.n_layers);
        let tok_embd = g.dequant("token_embd.weight")?;
        // big matmul weight: GPU (f16 resident) when available, else CPU f32
        let w = |name: &str, cols: usize| -> Result<Weight> {
            Ok(match gpu {
                // upload native quantized bytes (coalesced K-quant kernels handle them)
                Some(gp) => {
                    let (bytes, ty, n) = g.raw(name)?;
                    Weight::Gpu(gp.upload_quant(bytes, ty, n / cols, cols))
                }
                None => Weight::cpu(g.dequant(name)?, cols),
            })
        };
        let dim = cfg.dim;
        let ffn = cfg.ffn_dim;
        let conv_ch = cfg.inner_size + 2 * cfg.group_count * cfg.state_size;

        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for (i, kind) in kinds.iter().enumerate() {
            let p = |s: &str| format!("blk.{i}.{s}");
            blocks.push(match kind {
                BlockKind::Attention => Block::Attn(AttnW {
                    attn_norm: g.dequant(&p("attn_norm.weight"))?,
                    wq: w(&p("attn_q.weight"), dim)?,
                    wk: w(&p("attn_k.weight"), dim)?,
                    wv: w(&p("attn_v.weight"), dim)?,
                    q_norm: g.dequant(&p("attn_q_norm.weight"))?,
                    k_norm: g.dequant(&p("attn_k_norm.weight"))?,
                    wo: w(&p("attn_output.weight"), cfg.n_heads * cfg.head_dim)?,
                    post_norm: g.dequant(&p("post_attention_norm.weight"))?,
                    ffn_gate: w(&p("ffn_gate.weight"), dim)?,
                    ffn_up: w(&p("ffn_up.weight"), dim)?,
                    ffn_down: w(&p("ffn_down.weight"), ffn)?,
                }),
                BlockKind::Ssm => Block::Lin(LinW {
                    attn_norm: g.dequant(&p("attn_norm.weight"))?,
                    wqkv: w(&p("attn_qkv.weight"), dim)?,
                    wz: w(&p("attn_gate.weight"), dim)?,
                    ssm_beta: g.dequant(&p("ssm_beta.weight"))?,
                    ssm_alpha: g.dequant(&p("ssm_alpha.weight"))?,
                    ssm_a: g.dequant(&p("ssm_a"))?,
                    ssm_dt: g.dequant(&p("ssm_dt.bias"))?,
                    conv1d: g.dequant(&p("ssm_conv1d.weight"))?,
                    ssm_norm: g.dequant(&p("ssm_norm.weight"))?,
                    ssm_out: w(&p("ssm_out.weight"), cfg.inner_size)?,
                    post_norm: g.dequant(&p("post_attention_norm.weight"))?,
                    ffn_gate: w(&p("ffn_gate.weight"), dim)?,
                    ffn_up: w(&p("ffn_up.weight"), dim)?,
                    ffn_down: w(&p("ffn_down.weight"), ffn)?,
                }),
            });
        }
        let _ = conv_ch;
        let output_norm = g.dequant("output_norm.weight")?;
        let output = if g.has("output.weight") {
            w("output.weight", dim)?
        } else {
            w("token_embd.weight", dim)?
        };
        eprintln!(
            "[qwen35] loaded: {} blocks, vocab {} ({})",
            cfg.n_layers,
            cfg.vocab,
            if gpu.is_some() { "GPU matmuls" } else { "CPU" }
        );
        if std::env::var("HOS_QWEN35_DEBUG").is_ok() {
            if let Some(Block::Lin(lw)) = blocks.iter().find(|b| matches!(b, Block::Lin(_))) {
                let (mn, mx) = lw
                    .ssm_a
                    .iter()
                    .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
                eprintln!(
                    "[qwen35] ssm_a range: min={mn:.4} max={mx:.4} (negative => already A; positive => A_log needs -exp)"
                );
            }
        }
        // Load the trailing NextN/MTP draft block (index = n_layers) if present.
        // Used for self-speculative decoding; absent => plain decode still works.
        let mi = cfg.n_layers;
        let mp = |s: &str| format!("blk.{mi}.{s}");
        let mtp = if g.has(&mp("nextn.eh_proj.weight")) {
            Some(MtpW {
                attn: AttnW {
                    attn_norm: g.dequant(&mp("attn_norm.weight"))?,
                    wq: w(&mp("attn_q.weight"), dim)?,
                    wk: w(&mp("attn_k.weight"), dim)?,
                    wv: w(&mp("attn_v.weight"), dim)?,
                    q_norm: g.dequant(&mp("attn_q_norm.weight"))?,
                    k_norm: g.dequant(&mp("attn_k_norm.weight"))?,
                    wo: w(&mp("attn_output.weight"), cfg.n_heads * cfg.head_dim)?,
                    post_norm: g.dequant(&mp("post_attention_norm.weight"))?,
                    ffn_gate: w(&mp("ffn_gate.weight"), dim)?,
                    ffn_up: w(&mp("ffn_up.weight"), dim)?,
                    ffn_down: w(&mp("ffn_down.weight"), ffn)?,
                },
                eh_proj: w(&mp("nextn.eh_proj.weight"), 2 * dim)?,
                enorm: g.dequant(&mp("nextn.enorm.weight"))?,
                hnorm: g.dequant(&mp("nextn.hnorm.weight"))?,
                shared_head_norm: g.dequant(&mp("nextn.shared_head_norm.weight"))?,
            })
        } else {
            None
        };
        if mtp.is_some() {
            eprintln!("[qwen35] MTP draft head loaded (self-speculative decoding available)");
        }
        Ok(Qwen35 {
            cfg,
            tok_embd,
            blocks,
            output_norm,
            output,
            mtp,
        })
    }

    /// True if the NextN/MTP draft head is available (enables speculative decode).
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    pub fn dim(&self) -> usize {
        self.cfg.dim
    }

    /// Public wrapper over the MTP draft head (correctness checks / resident loop).
    pub fn mtp_draft_logits(
        &self,
        hidden: &[f32],
        token: u32,
        mpos: usize,
        st: &mut State,
        gpu: Option<&Gpu>,
    ) -> Vec<f32> {
        self.mtp_draft(hidden, token, mpos, st, gpu)
    }

    fn ffn(
        &self,
        x: &[f32],
        gate: &Weight,
        up: &Weight,
        down: &Weight,
        gpu: Option<&Gpu>,
    ) -> Vec<f32> {
        let f = self.cfg.ffn_dim;
        let mut g = mmw(gate, x, f, gpu);
        let u = mmw(up, x, f, gpu);
        for i in 0..f {
            g[i] = silu(g[i]) * u[i];
        }
        mmw(down, &g, self.cfg.dim, gpu)
    }

    fn attn_block(
        &self,
        w: &AttnW,
        x: &[f32],
        st: &mut State,
        il: usize,
        pos: usize,
        gpu: Option<&Gpu>,
    ) -> Vec<f32> {
        let c = &self.cfg;
        let (hd, nh, nkv) = (c.head_dim, c.n_heads, c.n_kv_heads);
        let kv_dim = nkv * hd;
        let qfull = mmw(&w.wq, x, nh * hd * 2, gpu);
        let k = mmw(&w.wk, x, kv_dim, gpu);
        let v = mmw(&w.wv, x, kv_dim, gpu);
        let att = self.attn_core(w, &qfull, &k, &v, st, il, pos);
        mmw(&w.wo, &att, c.dim, gpu)
    }

    /// Attention per-token core given ALREADY-PROJECTED q/gate (`qfull`), `k`, `v`:
    /// q/gate split, q/k norm, RoPE, KV-cache write, causal attention, output gate.
    /// Returns the pre-output-projection `[n_heads*head_dim]` vector. Defined ONCE
    /// and shared by per-token `attn_block` and batched `attn_block_prefill`.
    fn attn_core(
        &self,
        w: &AttnW,
        qfull: &[f32],
        k_in: &[f32],
        v_in: &[f32],
        st: &mut State,
        il: usize,
        pos: usize,
    ) -> Vec<f32> {
        let c = &self.cfg;
        let (hd, nh, nkv) = (c.head_dim, c.n_heads, c.n_kv_heads);
        let kv_dim = nkv * hd;
        let mut q = vec![0.0f32; nh * hd];
        let mut gate = vec![0.0f32; nh * hd];
        for h in 0..nh {
            let base = h * hd * 2;
            q[h * hd..h * hd + hd].copy_from_slice(&qfull[base..base + hd]);
            gate[h * hd..h * hd + hd].copy_from_slice(&qfull[base + hd..base + 2 * hd]);
        }
        for h in 0..nh {
            rmsnorm_inplace(&mut q[h * hd..h * hd + hd], &w.q_norm, c.rms_eps);
        }
        let mut k = k_in.to_vec();
        for h in 0..nkv {
            rmsnorm_inplace(&mut k[h * hd..h * hd + hd], &w.k_norm, c.rms_eps);
        }
        rope(&mut q, nh, hd, c.rope_dim, pos, c.rope_base);
        rope(&mut k, nkv, hd, c.rope_dim, pos, c.rope_base);
        st.k_cache[il][pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&k);
        st.v_cache[il][pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(v_in);
        let scale = 1.0 / (hd as f32).sqrt();
        let kv_mul = nh / nkv;
        let mut att = vec![0.0f32; nh * hd];
        for h in 0..nh {
            let kvh = h / kv_mul;
            let qh = &q[h * hd..h * hd + hd];
            let mut scores = vec![0.0f32; pos + 1];
            for t in 0..=pos {
                let kh = &st.k_cache[il][t * kv_dim + kvh * hd..t * kv_dim + kvh * hd + hd];
                scores[t] = qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f32>() * scale;
            }
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for s in scores.iter_mut() {
                *s = (*s - mx).exp();
                sum += *s;
            }
            let oh = &mut att[h * hd..h * hd + hd];
            for t in 0..=pos {
                let vh = &st.v_cache[il][t * kv_dim + kvh * hd..t * kv_dim + kvh * hd + hd];
                let wt = scores[t] / sum;
                for i in 0..hd {
                    oh[i] += wt * vh[i];
                }
            }
        }
        for i in 0..nh * hd {
            att[i] *= 1.0 / (1.0 + (-gate[i]).exp());
        } // sigmoid output gate (attn_output_gate) — per the reference forward
        att
    }

    /// Batched attention prefill: project q/gate, k, v for all `n` tokens at once
    /// (weights read once), run the sequential `attn_core` per token (KV cache
    /// carries), then project the output for all tokens at once. `[n, dim]`.
    fn attn_block_prefill(
        &self,
        w: &AttnW,
        xn: &[f32],
        st: &mut State,
        il: usize,
        n: usize,
        start_pos: usize,
        gpu: Option<&Gpu>,
    ) -> Vec<f32> {
        let c = &self.cfg;
        let d = c.dim;
        let (hd, nh, nkv) = (c.head_dim, c.n_heads, c.n_kv_heads);
        let kv_dim = nkv * hd;
        let qw = nh * hd * 2;
        let mut qfull_all = vec![0.0f32; n * qw];
        w.wq.matvec_batch(gpu, xn, n, &mut qfull_all);
        let mut k_all = vec![0.0f32; n * kv_dim];
        w.wk.matvec_batch(gpu, xn, n, &mut k_all);
        let mut v_all = vec![0.0f32; n * kv_dim];
        w.wv.matvec_batch(gpu, xn, n, &mut v_all);
        let mut att_all = vec![0.0f32; n * nh * hd];
        for t in 0..n {
            // Absolute sequence position, so mid-sequence verify attends correctly.
            let att = self.attn_core(
                w,
                &qfull_all[t * qw..(t + 1) * qw],
                &k_all[t * kv_dim..(t + 1) * kv_dim],
                &v_all[t * kv_dim..(t + 1) * kv_dim],
                st,
                il,
                start_pos + t,
            );
            att_all[t * nh * hd..(t + 1) * nh * hd].copy_from_slice(&att);
        }
        let mut result = vec![0.0f32; n * d];
        w.wo.matvec_batch(gpu, &att_all, n, &mut result);
        result
    }

    fn lin_block(
        &self,
        w: &LinW,
        x: &[f32],
        st: &mut State,
        il: usize,
        gpu: Option<&Gpu>,
    ) -> Vec<f32> {
        let c = &self.cfg;
        let conv_ch = c.inner_size + 2 * c.group_count * c.state_size;
        let qkv = mmw(&w.wqkv, x, conv_ch, gpu);
        let z = mmw(&w.wz, x, c.inner_size, gpu);
        let out = self.lin_core(w, &qkv, &z, x, st, il);
        mmw(&w.ssm_out, &out, c.dim, gpu)
    }

    /// Gated-DeltaNet per-token core given ALREADY-PROJECTED `qkv`/`z` (and `x` for
    /// the small beta/alpha projections): conv → split → scan → gated norm, carrying
    /// the conv + SSM state. Returns the pre-output-projection `[inner_size]` vector.
    /// Defined ONCE and shared by per-token `lin_block` and batched
    /// `lin_block_prefill`.
    fn lin_core(
        &self,
        w: &LinW,
        qkv: &[f32],
        z: &[f32],
        x: &[f32],
        st: &mut State,
        il: usize,
    ) -> Vec<f32> {
        let c = &self.cfg;
        let hk = c.state_size; // head_k_dim = 128
        let nk = c.group_count; // num_k_heads = 16
        let nv = c.time_step_rank; // num_v_heads = 48
        let hv = c.inner_size / nv; // head_v_dim = 128
        let conv_ch = c.inner_size + 2 * nk * hk;
        let kk = c.conv_kernel; // 4

        let beta: Vec<f32> = mm(&w.ssm_beta, x, nv)
            .iter()
            .map(|a| 1.0 / (1.0 + (-a).exp()))
            .collect();
        let alpha = mm(&w.ssm_alpha, x, nv);
        let decay: Vec<f32> = (0..nv)
            .map(|h| {
                let sp = {
                    let z = alpha[h] + w.ssm_dt[h];
                    // softplus
                    if z > 20.0 {
                        z
                    } else {
                        (1.0 + z.exp()).ln()
                    }
                };
                w.ssm_a[h] * sp
            })
            .collect();

        // causal depthwise conv1d over conv_ch channels, kernel kk, then silu
        let cs = &mut st.conv[il]; // conv_ch * (kk-1)
        let mut conv_out = vec![0.0f32; conv_ch];
        for ch in 0..conv_ch {
            let mut acc = 0.0;
            // window = [state(kk-1) , new]
            for t in 0..kk - 1 {
                acc += cs[ch * (kk - 1) + t] * w.conv1d[ch * kk + t];
            }
            acc += qkv[ch] * w.conv1d[ch * kk + (kk - 1)];
            conv_out[ch] = silu(acc);
            // shift state: keep last (kk-1) of [state.., new]
            for t in 0..kk - 2 {
                cs[ch * (kk - 1) + t] = cs[ch * (kk - 1) + t + 1];
            }
            cs[ch * (kk - 1) + (kk - 2)] = qkv[ch];
        }

        // qkv is concatenated by type: [q(nk*hk), k(nk*hk), v(nv*hv)].
        let q0 = 0;
        let k0 = nk * hk;
        let v0 = 2 * nk * hk;
        let l2 = |s: &[f32]| {
            let n = s.iter().map(|x| x * x).sum::<f32>();
            let inv = 1.0 / (n + c.rms_eps).sqrt();
            s.iter().map(|x| x * inv).collect::<Vec<f32>>()
        };
        let qscale = 1.0 / (hk as f32).sqrt();
        // The v-heads are independent (each owns its hv*hv state block), so the
        // Gated-DeltaNet scan parallelizes across heads with no math change.
        st.ssm[il]
            .par_chunks_mut(hv * hv)
            .enumerate()
            .flat_map(|(hvh, s)| {
                let khh = hvh % nk; // v-head -> k-head via ggml_repeat tile
                let qh: Vec<f32> = l2(&conv_out[q0 + khh * hk..q0 + khh * hk + hk])
                    .iter()
                    .map(|x| x * qscale)
                    .collect();
                let kh = l2(&conv_out[k0 + khh * hk..k0 + khh * hk + hk]);
                let vh = &conv_out[v0 + hvh * hv..v0 + hvh * hv + hv];
                let g = decay[hvh].exp();
                for x in s.iter_mut() {
                    *x *= g;
                }
                let kq: f32 = (0..hk).map(|i| kh[i] * qh[i]).sum();
                let mut d = vec![0.0f32; hv];
                let mut o = vec![0.0f32; hv];
                for j in 0..hv {
                    let row = &s[j * hv..j * hv + hv];
                    let sk: f32 = (0..hk).map(|i| row[i] * kh[i]).sum();
                    let sq: f32 = (0..hk).map(|i| row[i] * qh[i]).sum();
                    d[j] = beta[hvh] * (vh[j] - sk);
                    o[j] = sq + d[j] * kq;
                }
                for i in 0..hv {
                    let di = d[i];
                    let si = &mut s[i * hv..i * hv + hv];
                    for j in 0..hv {
                        si[j] += di * kh[j];
                    }
                }
                // RMSNormGated (llama.cpp build_norm_gated): normalize FIRST, then
                // gate. out = (rmsnorm(o) * ssm_norm.weight) * silu(z).
                let zh = &z[hvh * hv..hvh * hv + hv];
                let on = rmsnorm(&o, &w.ssm_norm, c.rms_eps);
                (0..hv).map(|j| on[j] * silu(zh[j])).collect::<Vec<f32>>()
            })
            .collect()
    }

    /// Batched SSM prefill: project qkv/z for all `n` tokens at once (weights read
    /// once via `matvec_batch`), run the sequential `lin_core` per token (conv + scan
    /// carry state), then project the output for all tokens at once. `[n, dim]`.
    #[allow(clippy::too_many_arguments)]
    fn lin_block_prefill(
        &self,
        w: &LinW,
        xn: &[f32],
        st: &mut State,
        il: usize,
        n: usize,
        snap: bool,
        gpu: Option<&Gpu>,
    ) -> Vec<f32> {
        let c = &self.cfg;
        let d = c.dim;
        let inner = c.inner_size;
        let conv_ch = inner + 2 * c.group_count * c.state_size;
        let mut qkv_all = vec![0.0f32; n * conv_ch];
        w.wqkv.matvec_batch(gpu, xn, n, &mut qkv_all);
        let mut z_all = vec![0.0f32; n * inner];
        w.wz.matvec_batch(gpu, xn, n, &mut z_all);
        let mut out_all = vec![0.0f32; n * inner];
        for t in 0..n {
            let o = self.lin_core(
                w,
                &qkv_all[t * conv_ch..(t + 1) * conv_ch],
                &z_all[t * inner..(t + 1) * inner],
                &xn[t * d..(t + 1) * d],
                st,
                il,
            );
            out_all[t * inner..(t + 1) * inner].copy_from_slice(&o);
            // Verify snapshot: capture this layer's SSM+conv state right after the
            // confirmed token (t==0), so a rejected draft rolls back with no reprocess.
            if snap && t == 0 {
                st.snap_ssm[il].clone_from(&st.ssm[il]);
                st.snap_conv[il].clone_from(&st.conv[il]);
            }
        }
        let mut result = vec![0.0f32; n * d];
        w.ssm_out.matvec_batch(gpu, &out_all, n, &mut result);
        result
    }

    pub fn forward(&self, st: &mut State, token: u32, pos: usize, gpu: Option<&Gpu>) -> Vec<f32> {
        let c = &self.cfg;
        let x = self.tok_embd[token as usize * c.dim..(token as usize + 1) * c.dim].to_vec();
        self.forward_x(st, x, pos, gpu, true)
    }

    /// Forward from a precomputed input embedding (image-token splice): identical
    /// to `forward` but the residual stream starts from `x` instead of a token
    /// embedding lookup. Used to feed vision embeddings at `<|image_pad|>` slots.
    pub fn forward_embed(
        &self,
        st: &mut State,
        embed: &[f32],
        pos: usize,
        gpu: Option<&Gpu>,
    ) -> Vec<f32> {
        self.forward_x(st, embed.to_vec(), pos, gpu, true)
    }

    /// Prefill step — advances the KV/SSM state for `token` but SKIPS the vocab
    /// projection (the ~152k-wide lm_head), which is only needed for the last
    /// prompt token. Returns nothing; saves the full lm_head matmul per prompt
    /// token (huge for the 576-token image block).
    pub fn prefill(&self, st: &mut State, token: u32, pos: usize, gpu: Option<&Gpu>) {
        let c = &self.cfg;
        let x = self.tok_embd[token as usize * c.dim..(token as usize + 1) * c.dim].to_vec();
        self.forward_x(st, x, pos, gpu, false);
    }

    /// Prefill step from a precomputed embedding (image tokens), no lm_head.
    pub fn prefill_embed(&self, st: &mut State, embed: &[f32], pos: usize, gpu: Option<&Gpu>) {
        self.forward_x(st, embed.to_vec(), pos, gpu, false);
    }

    fn forward_x(
        &self,
        st: &mut State,
        mut x: Vec<f32>,
        pos: usize,
        gpu: Option<&Gpu>,
        want_logits: bool,
    ) -> Vec<f32> {
        let c = &self.cfg;
        // Data-driven localization: on the first token, print the residual-stream
        // norm after each layer's mixer and FFN. Wherever it explodes or collapses
        // is the broken stage. Enable with HOS_QWEN35_TRACE.
        let trace = pos == 0 && std::env::var("HOS_QWEN35_TRACE").is_ok();
        let l2 = |v: &[f32]| (v.iter().map(|a| a * a).sum::<f32>()).sqrt();
        if trace {
            eprintln!("[trace] embed         ||x||={:.3}", l2(&x));
        }
        for il in 0..c.n_layers {
            let (cur, post, fg, fu, fd) = match &self.blocks[il] {
                Block::Attn(w) => {
                    let xn = rmsnorm(&x, &w.attn_norm, c.rms_eps);
                    (
                        self.attn_block(w, &xn, st, il, pos, gpu),
                        &w.post_norm,
                        &w.ffn_gate,
                        &w.ffn_up,
                        &w.ffn_down,
                    )
                }
                Block::Lin(w) => {
                    let xn = rmsnorm(&x, &w.attn_norm, c.rms_eps);
                    (
                        self.lin_block(w, &xn, st, il, gpu),
                        &w.post_norm,
                        &w.ffn_gate,
                        &w.ffn_up,
                        &w.ffn_down,
                    )
                }
            };
            let kind = if matches!(self.blocks[il], Block::Attn(_)) {
                'A'
            } else {
                's'
            };
            let mix_n = l2(&cur);
            for i in 0..c.dim {
                x[i] += cur[i];
            }
            let after_mix = l2(&x);
            let xpn = rmsnorm(&x, post, c.rms_eps);
            let ffn = self.ffn(&xpn, fg, fu, fd, gpu);
            for i in 0..c.dim {
                x[i] += ffn[i];
            }
            if trace {
                eprintln!(
                    "[trace] L{il:02} {kind}  mixer={mix_n:.3}  ->x={after_mix:.3}  +ffn->x={:.3}",
                    l2(&x)
                );
            }
        }
        if !want_logits {
            return Vec::new(); // prefill: skip the wide lm_head projection
        }
        rmsnorm_inplace(&mut x, &self.output_norm, c.rms_eps);
        if trace {
            eprintln!("[trace] final norm    ||x||={:.3}", l2(&x));
        }
        mmw(&self.output, &x, c.vocab, gpu)
    }

    /// Batched prefill: ingest all `n` prompt token embeddings (`x0` = `[n*dim]`,
    /// text tokens already looked up + image tokens spliced) in one pass, and
    /// return ONLY the last token's logits. The SSM/attention MIXER stays on the
    /// exact proven per-token path (its state is sequential), but the FFN — the
    /// widest matmuls and ~63% of the model's weights — is BATCHED across all
    /// tokens so each FFN weight is read once instead of `n` times. This is the
    /// prefill memory-bandwidth win (dominant cost for the 576-token image block).
    /// Additive: `forward`/`forward_embed`/`prefill` are unchanged; this is a new
    /// entry point, and `matvec_batch` falls back to per-token for non-q4_k weights.
    pub fn forward_prefill(
        &self,
        st: &mut State,
        x0: &[f32],
        n: usize,
        gpu: Option<&Gpu>,
    ) -> Vec<f32> {
        let c = &self.cfg;
        let d = c.dim;
        let x = self.run_layers(st, x0, n, 0, gpu);
        // Only the last token needs logits.
        let mut xl = x[(n - 1) * d..n * d].to_vec();
        rmsnorm_inplace(&mut xl, &self.output_norm, c.rms_eps);
        mmw(&self.output, &xl, c.vocab, gpu)
    }

    /// Like `forward_prefill` but also returns the last token's PRE-output-norm
    /// hidden state — the seed the MTP draft head needs to start speculating.
    pub fn forward_prefill_h(
        &self,
        st: &mut State,
        x0: &[f32],
        n: usize,
        gpu: Option<&Gpu>,
    ) -> (Vec<f32>, Vec<f32>) {
        let c = &self.cfg;
        let d = c.dim;
        let x = self.run_layers(st, x0, n, 0, gpu);
        let hidden = x[(n - 1) * d..n * d].to_vec();
        let mut xl = hidden.clone();
        rmsnorm_inplace(&mut xl, &self.output_norm, c.rms_eps);
        (mmw(&self.output, &xl, c.vocab, gpu), hidden)
    }

    /// Prefill from token ids (builds the embedding batch), returning
    /// `(last_logits, last_hidden)` — the seed for `decode_speculative`.
    pub fn forward_prefill_tokens(
        &self,
        st: &mut State,
        ids: &[u32],
        gpu: Option<&Gpu>,
    ) -> (Vec<f32>, Vec<f32>) {
        let d = self.cfg.dim;
        let mut x0 = vec![0.0f32; ids.len() * d];
        for (i, &t) in ids.iter().enumerate() {
            x0[i * d..(i + 1) * d]
                .copy_from_slice(&self.tok_embd[t as usize * d..(t as usize + 1) * d]);
        }
        let r = self.forward_prefill_h(st, &x0, ids.len(), gpu);
        st.pos = ids.len();
        r
    }

    /// Prefill for MTP decoding: runs the backbone AND pre-fills the draft head's KV
    /// cache over the prompt (each position p uses hidden[p-1] + token[p]), so the
    /// very first drafts have real context — without this, acceptance is ~nil.
    pub fn forward_prefill_mtp(
        &self,
        st: &mut State,
        ids: &[u32],
        gpu: Option<&Gpu>,
    ) -> (Vec<f32>, Vec<f32>) {
        let c = &self.cfg;
        let d = c.dim;
        let n = ids.len();
        let mut x0 = vec![0.0f32; n * d];
        for (i, &t) in ids.iter().enumerate() {
            x0[i * d..(i + 1) * d]
                .copy_from_slice(&self.tok_embd[t as usize * d..(t as usize + 1) * d]);
        }
        self.forward_prefill_mtp_x0(st, &x0, ids, gpu)
    }

    /// Like `forward_prefill_mtp` but the caller supplies the `[n, dim]` input
    /// embeddings `x0` directly (chat splices image embeddings into their slots).
    /// `ids` still drives the MTP KV token embeddings; at image positions the pad
    /// token is used for the draft KV only (slightly lower acceptance there — the
    /// backbone verify stays lossless regardless).
    pub fn forward_prefill_mtp_x0(
        &self,
        st: &mut State,
        x0: &[f32],
        ids: &[u32],
        gpu: Option<&Gpu>,
    ) -> (Vec<f32>, Vec<f32>) {
        let d = self.cfg.dim;
        let n = ids.len();
        let x = self.run_layers(st, x0, n, 0, gpu); // all per-position hiddens
        // Fill the MTP KV over the prompt: position p attends hidden[p-1] + token[p].
        for p in 1..n {
            let _ = self.mtp_fuse_attn(&x[(p - 1) * d..p * d], ids[p], p, st, gpu);
        }
        st.pos = n;
        let hidden = x[(n - 1) * d..n * d].to_vec();
        let logits = self.head(&hidden, gpu);
        (logits, hidden)
    }

    /// Public wrapper over `head` (final norm + shared lm_head → logits), for the
    /// resident MTP loop / correctness checks that hold a bare hidden state.
    pub fn logits_from_hidden(&self, hidden: &[f32], gpu: Option<&Gpu>) -> Vec<f32> {
        self.head(hidden, gpu)
    }

    /// Project a hidden state through the final norm + shared lm_head → logits.
    fn head(&self, hidden: &[f32], gpu: Option<&Gpu>) -> Vec<f32> {
        let mut h = hidden.to_vec();
        rmsnorm_inplace(&mut h, &self.output_norm, self.cfg.rms_eps);
        mmw(&self.output, &h, self.cfg.vocab, gpu)
    }

    /// MTP draft head: predict the NEXT token's logits from the backbone's pre-norm
    /// `hidden` at position p and the just-emitted `token` (at p+1). One cheap layer
    /// (fuse → attn → FFN → shared lm_head). `mpos` is its own KV slot.
    /// The MTP fuse + attention (writes its KV at `mpos`), returning the post-attn
    /// residual. Shared by drafting and by prompt KV pre-fill.
    fn mtp_fuse_attn(
        &self,
        hidden: &[f32],
        token: u32,
        mpos: usize,
        st: &mut State,
        gpu: Option<&Gpu>,
    ) -> Vec<f32> {
        let mtp = self.mtp.as_ref().expect("mtp without an MTP head");
        let c = &self.cfg;
        let d = c.dim;
        let emb = &self.tok_embd[token as usize * d..(token as usize + 1) * d];
        let e_norm = rmsnorm(emb, &mtp.enorm, c.rms_eps);
        let h_norm = rmsnorm(hidden, &mtp.hnorm, c.rms_eps);
        let mut cat = Vec::with_capacity(2 * d);
        cat.extend_from_slice(&e_norm);
        cat.extend_from_slice(&h_norm);
        let mut cur = mmw(&mtp.eh_proj, &cat, d, gpu);
        let xn = rmsnorm(&cur, &mtp.attn.attn_norm, c.rms_eps);
        let att = self.mtp_attn(&mtp.attn, &xn, st, mpos, gpu);
        for i in 0..d {
            cur[i] += att[i];
        }
        cur
    }

    fn mtp_draft(
        &self,
        hidden: &[f32],
        token: u32,
        mpos: usize,
        st: &mut State,
        gpu: Option<&Gpu>,
    ) -> Vec<f32> {
        let mtp = self.mtp.as_ref().expect("mtp_draft without an MTP head");
        let c = &self.cfg;
        let d = c.dim;
        let mut cur = self.mtp_fuse_attn(hidden, token, mpos, st, gpu);
        let xpn = rmsnorm(&cur, &mtp.attn.post_norm, c.rms_eps);
        let ffn = self.ffn(&xpn, &mtp.attn.ffn_gate, &mtp.attn.ffn_up, &mtp.attn.ffn_down, gpu);
        for i in 0..d {
            cur[i] += ffn[i];
        }
        let xh = rmsnorm(&cur, &mtp.shared_head_norm, c.rms_eps);
        mmw(&self.output, &xh, c.vocab, gpu)
    }

    /// The MTP layer's attention over its own KV cache (`mtp_k`/`mtp_v`).
    fn mtp_attn(
        &self,
        w: &AttnW,
        xn: &[f32],
        st: &mut State,
        pos: usize,
        gpu: Option<&Gpu>,
    ) -> Vec<f32> {
        let c = &self.cfg;
        let (hd, nh, nkv) = (c.head_dim, c.n_heads, c.n_kv_heads);
        let kv_dim = nkv * hd;
        let qfull = mmw(&w.wq, xn, nh * hd * 2, gpu);
        let mut q = vec![0.0f32; nh * hd];
        let mut gate = vec![0.0f32; nh * hd];
        for h in 0..nh {
            let base = h * hd * 2;
            q[h * hd..h * hd + hd].copy_from_slice(&qfull[base..base + hd]);
            gate[h * hd..h * hd + hd].copy_from_slice(&qfull[base + hd..base + 2 * hd]);
        }
        for h in 0..nh {
            rmsnorm_inplace(&mut q[h * hd..h * hd + hd], &w.q_norm, c.rms_eps);
        }
        let mut k = mmw(&w.wk, xn, kv_dim, gpu);
        let v = mmw(&w.wv, xn, kv_dim, gpu);
        for h in 0..nkv {
            rmsnorm_inplace(&mut k[h * hd..h * hd + hd], &w.k_norm, c.rms_eps);
        }
        rope(&mut q, nh, hd, c.rope_dim, pos, c.rope_base);
        rope(&mut k, nkv, hd, c.rope_dim, pos, c.rope_base);
        st.mtp_k[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&k);
        st.mtp_v[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&v);
        let scale = 1.0 / (hd as f32).sqrt();
        let kv_mul = nh / nkv;
        let mut att = vec![0.0f32; nh * hd];
        for h in 0..nh {
            let kvh = h / kv_mul;
            let qh = &q[h * hd..h * hd + hd];
            let mut scores = vec![0.0f32; pos + 1];
            for t in 0..=pos {
                let kh = &st.mtp_k[t * kv_dim + kvh * hd..t * kv_dim + kvh * hd + hd];
                scores[t] = qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f32>() * scale;
            }
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for s in scores.iter_mut() {
                *s = (*s - mx).exp();
                sum += *s;
            }
            let oh = &mut att[h * hd..h * hd + hd];
            for t in 0..=pos {
                let vh = &st.mtp_v[t * kv_dim + kvh * hd..t * kv_dim + kvh * hd + hd];
                let wt = scores[t] / sum;
                for i in 0..hd {
                    oh[i] += wt * vh[i];
                }
            }
        }
        for i in 0..nh * hd {
            att[i] *= 1.0 / (1.0 + (-gate[i]).exp());
        }
        mmw(&w.wo, &att, c.dim, gpu)
    }

    /// Self-speculative decode via the MTP head. Each step: draft the next token
    /// (cheap 1-layer head), then the backbone verifies `[confirmed, draft]` in ONE
    /// batched 2-token pass (weights read once for both). LOSSLESS: the emitted
    /// token is always a fresh sample from the backbone's own distribution — the
    /// draft only lets us skip a full pass when it matches. On mismatch the SSM/conv
    /// state is rolled back and the confirmed token is reprocessed. `on(token)` is
    /// called per emitted token; returns the count.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_speculative(
        &self,
        st: &mut State,
        mut logits: Vec<f32>,
        mut hidden: Vec<f32>,
        max_tokens: usize,
        temp: f32,
        top_k: usize,
        top_p: f32,
        rep_penalty: f32,
        repeat_last_n: usize,
        seed: u64,
        stops: &[u32],
        gpu: Option<&Gpu>,
        mut on: impl FnMut(u32) -> bool,
    ) -> usize {
        let c = &self.cfg;
        let d = c.dim;
        let argmax = |l: &[f32]| -> u32 {
            let mut bi = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &v) in l.iter().enumerate() {
                if v > bv {
                    bv = v;
                    bi = i;
                }
            }
            bi as u32
        };
        let mut rng = if seed != 0 { seed } else { 42 };
        let mut recent: Vec<u32> = Vec::new();
        let mut emitted = 0usize;
        let mut verifies = 0usize;
        let mut accepts = 0usize;
        while emitted < max_tokens {
            let from = recent.len().saturating_sub(repeat_last_n);
            let t = crate::sample(&logits, temp, top_k, top_p, rep_penalty, &recent[from..], &mut rng);
            if stops.contains(&t) {
                break;
            }
            recent.push(t);
            emitted += 1;
            if on(t) {
                break;
            }
            let p = st.pos;
            // Draft the token after t, then verify [t, draft] in one batched pass
            // (snapshotting each layer's SSM state after t, for zero-cost rejection).
            let draft = argmax(&self.mtp_draft(&hidden, t, p, st, gpu));
            let mut x2 = vec![0.0f32; 2 * d];
            x2[0..d].copy_from_slice(&self.tok_embd[t as usize * d..(t as usize + 1) * d]);
            x2[d..2 * d].copy_from_slice(&self.tok_embd[draft as usize * d..(draft as usize + 1) * d]);
            let hids = self.run_layers_snap(st, &x2, 2, p, true, gpu);
            let logits_t = self.head(&hids[0..d], gpu); // backbone dist at p+1
            let from2 = recent.len().saturating_sub(repeat_last_n);
            let real =
                crate::sample(&logits_t, temp, top_k, top_p, rep_penalty, &recent[from2..], &mut rng);
            verifies += 1;
            if real == draft {
                accepts += 1;
                // Accept: draft == the fresh backbone sample. Two tokens for one pass.
                recent.push(draft);
                emitted += 1;
                st.pos = p + 2;
                if stops.contains(&draft) || on(draft) {
                    break;
                }
                logits = self.head(&hids[d..2 * d], gpu);
                hidden = hids[d..2 * d].to_vec();
            } else {
                // Reject: roll each layer's SSM/conv back to the post-t snapshot (no
                // reprocess), keep t confirmed, and let the next iteration sample p+1
                // from the backbone distribution. hids[0] is already the hidden after t.
                for il in 0..self.cfg.n_layers {
                    if !st.snap_ssm[il].is_empty() {
                        st.ssm[il].clone_from(&st.snap_ssm[il]);
                        st.conv[il].clone_from(&st.snap_conv[il]);
                    }
                }
                hidden = hids[0..d].to_vec();
                logits = logits_t;
                st.pos = p + 1;
            }
        }
        if verifies > 0 && std::env::var("HOS_MTP_STATS").is_ok() {
            eprintln!(
                "[mtp] acceptance {:.0}% ({}/{} drafts)",
                100.0 * accepts as f64 / verifies as f64,
                accepts,
                verifies
            );
        }
        emitted
    }

    /// MTP self-speculative decode with the GPU-resident 2-token verify. Same
    /// lossless contract as `decode_speculative`, but the backbone verify runs
    /// entirely on the resident runner (`forward2`) instead of CPU recurrence — the
    /// SSM/attention scan no longer caps the speedup. The caller seeds the resident
    /// backbone state from `st` (via `upload_state`) before the first token; the CPU
    /// `st` supplies only the MTP KV cache (`mtp_draft`) and `st.pos` from here on.
    #[cfg(target_os = "macos")]
    #[allow(clippy::too_many_arguments)]
    pub fn decode_speculative_resident(
        &self,
        st: &mut State,
        rgpu: &Qwen35Gpu,
        mut logits: Vec<f32>,
        mut hidden: Vec<f32>,
        max_tokens: usize,
        temp: f32,
        top_k: usize,
        top_p: f32,
        rep_penalty: f32,
        repeat_last_n: usize,
        seed: u64,
        stops: &[u32],
        gpu: Option<&Gpu>,
        mut on: impl FnMut(u32) -> bool,
    ) -> usize {
        let c = &self.cfg;
        let d = c.dim;
        let argmax = |l: &[f32]| -> u32 {
            let mut bi = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &v) in l.iter().enumerate() {
                if v > bv {
                    bv = v;
                    bi = i;
                }
            }
            bi as u32
        };
        let mut rng = if seed != 0 { seed } else { 42 };
        let mut recent: Vec<u32> = Vec::new();
        let mut emitted = 0usize;
        let mut verifies = 0usize;
        let mut accepts = 0usize;
        while emitted < max_tokens {
            let from = recent.len().saturating_sub(repeat_last_n);
            let t = crate::sample(&logits, temp, top_k, top_p, rep_penalty, &recent[from..], &mut rng);
            if stops.contains(&t) {
                break;
            }
            recent.push(t);
            emitted += 1;
            if on(t) {
                break;
            }
            let p = st.pos;
            let draft = argmax(&self.mtp_draft(&hidden, t, p, st, gpu));
            // Resident 2-token verify: [t @ p, draft @ p+1] in one command buffer,
            // snapshotting conv/SSM after t for zero-cost rejection.
            let hids = rgpu.forward2(self, t, draft, p);
            let logits_t = self.head(&hids[0..d], gpu);
            let from2 = recent.len().saturating_sub(repeat_last_n);
            let real =
                crate::sample(&logits_t, temp, top_k, top_p, rep_penalty, &recent[from2..], &mut rng);
            verifies += 1;
            if real == draft {
                accepts += 1;
                recent.push(draft);
                emitted += 1;
                st.pos = p + 2;
                if stops.contains(&draft) || on(draft) {
                    break;
                }
                logits = self.head(&hids[d..2 * d], gpu);
                hidden = hids[d..2 * d].to_vec();
            } else {
                rgpu.restore_snap(self);
                hidden = hids[0..d].to_vec();
                logits = logits_t;
                st.pos = p + 1;
            }
        }
        if verifies > 0 && std::env::var("HOS_MTP_STATS").is_ok() {
            eprintln!(
                "[mtp] acceptance {:.0}% ({}/{} drafts)",
                100.0 * accepts as f64 / verifies as f64,
                accepts,
                verifies
            );
        }
        emitted
    }

    /// Run all backbone layers over an `[n, dim]` batch, advancing SSM/conv/KV
    /// state, and return the per-position PRE-output-norm hidden states `[n, dim]`.
    /// Shared by `forward_prefill` (last-token logits) and the MTP verify (which
    /// needs both tokens' hidden + logits). Weights are read once across the batch.
    fn run_layers(
        &self,
        st: &mut State,
        x0: &[f32],
        n: usize,
        start_pos: usize,
        gpu: Option<&Gpu>,
    ) -> Vec<f32> {
        self.run_layers_snap(st, x0, n, start_pos, false, gpu)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_layers_snap(
        &self,
        st: &mut State,
        x0: &[f32],
        n: usize,
        start_pos: usize,
        snap: bool,
        gpu: Option<&Gpu>,
    ) -> Vec<f32> {
        let c = &self.cfg;
        let d = c.dim;
        let mut x = x0.to_vec();
        for il in 0..c.n_layers {
            let (post, fg, fu, fd) = match &self.blocks[il] {
                Block::Attn(w) => {
                    let xn = self.batched_rmsnorm(&x, &w.attn_norm, n, d);
                    let mix = self.attn_block_prefill(w, &xn, st, il, n, start_pos, gpu);
                    for i in 0..n * d {
                        x[i] += mix[i];
                    }
                    (&w.post_norm, &w.ffn_gate, &w.ffn_up, &w.ffn_down)
                }
                Block::Lin(w) => {
                    let xn = self.batched_rmsnorm(&x, &w.attn_norm, n, d);
                    let mix = self.lin_block_prefill(w, &xn, st, il, n, snap, gpu);
                    for i in 0..n * d {
                        x[i] += mix[i];
                    }
                    (&w.post_norm, &w.ffn_gate, &w.ffn_up, &w.ffn_down)
                }
            };
            let xpn = self.batched_rmsnorm(&x, post, n, d);
            let ffn = self.ffn_prefill(&xpn, fg, fu, fd, n, gpu);
            for i in 0..n * d {
                x[i] += ffn[i];
            }
        }
        x
    }

    /// Per-token RMSNorm over an `[n, d]` batch (parallel over tokens).
    fn batched_rmsnorm(&self, x: &[f32], w: &[f32], n: usize, d: usize) -> Vec<f32> {
        let eps = self.cfg.rms_eps;
        (0..n)
            .into_par_iter()
            .flat_map(|t| rmsnorm(&x[t * d..t * d + d], w, eps))
            .collect()
    }

    /// Batched SwiGLU FFN over an `[n, dim]` batch: gate/up/down each read once
    /// across all tokens (`matvec_batch`). Returns `[n, dim]`.
    fn ffn_prefill(
        &self,
        xn: &[f32],
        gate: &Weight,
        up: &Weight,
        down: &Weight,
        n: usize,
        gpu: Option<&Gpu>,
    ) -> Vec<f32> {
        let c = &self.cfg;
        let (d, ff) = (c.dim, c.ffn_dim);
        let mut g = vec![0.0f32; n * ff];
        gate.matvec_batch(gpu, xn, n, &mut g);
        let mut u = vec![0.0f32; n * ff];
        up.matvec_batch(gpu, xn, n, &mut u);
        for i in 0..n * ff {
            g[i] = silu(g[i]) * u[i];
        }
        let mut out = vec![0.0f32; n * d];
        down.matvec_batch(gpu, &g, n, &mut out);
        out
    }
}

// ============================================================================
// Qwen35Gpu: fully GPU-resident runner. Activations + KV/conv/ssm state live on
// the GPU; one command buffer + one encoder per token (no CPU round-trips).
// ============================================================================

/// CPU-only stub of the qwen35 GPU runner for non-macOS targets — never built
/// (the qwen35 GPU gate is `cfg(target_os="macos")`), present only to type-check.
#[cfg(not(target_os = "macos"))]
pub struct Qwen35Gpu;
#[cfg(not(target_os = "macos"))]
impl Qwen35Gpu {
    pub fn new(_gpu: &Gpu, _m: &Qwen35) -> Qwen35Gpu {
        unreachable!("the qwen35 GPU runner is macOS-only")
    }
    pub fn forward(&self, _m: &Qwen35, _token: u32, _pos: usize) -> Vec<f32> {
        unreachable!("the qwen35 GPU runner is macOS-only")
    }
}

#[cfg(target_os = "macos")]
use crate::metal_be::GpuMatrix;
#[cfg(target_os = "macos")]
use metal::{
    Buffer, BufferRef, CommandQueue, ComputeCommandEncoderRef, ComputePipelineState, Device,
    MTLResourceOptions, MTLSize, ResourceRef,
};
#[cfg(target_os = "macos")]
use std::ffi::c_void;

#[cfg(target_os = "macos")]
pub struct Qwen35Gpu {
    #[allow(dead_code)] // held to keep the Metal device alive for the runner's lifetime
    device: Device,
    queue: CommandQueue,
    p_mv_f16: ComputePipelineState,
    p_mv_q4k: ComputePipelineState,
    p_mv_q5k: ComputePipelineState,
    p_mv_q6k: ComputePipelineState,
    p_mv_q8: ComputePipelineState,
    p_rms: ComputePipelineState,
    p_rms_h: ComputePipelineState,
    p_l2_h: ComputePipelineState,
    p_sig_mul: ComputePipelineState,
    p_sig_ip: ComputePipelineState,
    p_decay: ComputePipelineState,
    p_conv: ComputePipelineState,
    p_delta: ComputePipelineState,
    p_gnorm: ComputePipelineState,
    p_rope: ComputePipelineState,
    p_store: ComputePipelineState,
    p_attn: ComputePipelineState,
    p_swiglu: ComputePipelineState,
    p_add: ComputePipelineState,
    p_qgate: ComputePipelineState,
    // 2-token (MTP verify) matmuls + state snapshot copy
    p_mm_q4k_2tok: ComputePipelineState,
    p_mm_q6k_2tok: ComputePipelineState,
    p_copy: ComputePipelineState,
    // activation buffers
    x: Buffer,
    xb: Buffer,
    qfull: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    att: Buffer,
    gate: Buffer,
    qkv: Buffer,
    z: Buffer,
    conv_out: Buffer,
    beta: Buffer,
    decay: Buffer,
    dout: Buffer,
    gbuf: Buffer,
    ubuf: Buffer,
    logits: Buffer,
    // per-layer norms / small params (None where not applicable)
    attn_norm: Vec<Buffer>,
    post_norm: Vec<Buffer>,
    q_norm: Vec<Option<Buffer>>,
    k_norm: Vec<Option<Buffer>>,
    ssm_norm: Vec<Option<Buffer>>,
    conv_w: Vec<Option<Buffer>>,
    ssm_a: Vec<Option<Buffer>>,
    ssm_dt: Vec<Option<Buffer>>,
    ssm_beta: Vec<Option<GpuMatrix>>,
    ssm_alpha: Vec<Option<GpuMatrix>>,
    output_norm: Buffer,
    // state
    kcache: Vec<Buffer>,
    vcache: Vec<Buffer>,
    conv_state: Vec<Buffer>,
    ssm_state: Vec<Buffer>,
    // post-token-t snapshots for MTP verify rollback (per block; empty for attn blocks)
    snap_conv: Vec<Buffer>,
    snap_ssm: Vec<Buffer>,
}

#[cfg(target_os = "macos")]
impl Qwen35Gpu {
    pub fn new(gpu: &Gpu, m: &Qwen35) -> Qwen35Gpu {
        let device = gpu.device().clone();
        let queue = gpu.queue().clone();
        let lib = gpu.fused_library();
        let pl = |n: &str| {
            device
                .new_compute_pipeline_state_with_function(&lib.get_function(n, None).unwrap())
                .unwrap()
        };
        let c = &m.cfg;
        let nb =
            |n: usize| device.new_buffer((n * 4) as u64, MTLResourceOptions::StorageModeShared);
        let up = |d: &[f32]| {
            device.new_buffer_with_data(
                d.as_ptr() as *const c_void,
                (d.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };

        let kv_dim = c.n_kv_heads * c.head_dim;
        let conv_ch = c.inner_size + 2 * c.group_count * c.state_size;
        let hv = c.inner_size / c.time_step_rank;
        let kinds = block_kinds_from(m);

        let mut attn_norm = Vec::new();
        let mut post_norm = Vec::new();
        let mut q_norm = Vec::new();
        let mut k_norm = Vec::new();
        let mut ssm_norm = Vec::new();
        let mut conv_w = Vec::new();
        let mut ssm_a = Vec::new();
        let mut ssm_dt = Vec::new();
        let mut ssm_beta = Vec::new();
        let mut ssm_alpha = Vec::new();
        for (i, blk) in m.blocks.iter().enumerate() {
            let _ = i;
            match blk {
                Block::Attn(w) => {
                    attn_norm.push(up(&w.attn_norm));
                    post_norm.push(up(&w.post_norm));
                    q_norm.push(Some(up(&w.q_norm)));
                    k_norm.push(Some(up(&w.k_norm)));
                    ssm_norm.push(None);
                    conv_w.push(None);
                    ssm_a.push(None);
                    ssm_dt.push(None);
                    ssm_beta.push(None);
                    ssm_alpha.push(None);
                }
                Block::Lin(w) => {
                    attn_norm.push(up(&w.attn_norm));
                    post_norm.push(up(&w.post_norm));
                    q_norm.push(None);
                    k_norm.push(None);
                    ssm_norm.push(Some(up(&w.ssm_norm)));
                    conv_w.push(Some(up(&w.conv1d)));
                    ssm_a.push(Some(up(&w.ssm_a)));
                    ssm_dt.push(Some(up(&w.ssm_dt)));
                    ssm_beta.push(Some(gpu.upload_matrix(
                        &w.ssm_beta,
                        c.time_step_rank,
                        c.dim,
                    )));
                    ssm_alpha.push(Some(gpu.upload_matrix(
                        &w.ssm_alpha,
                        c.time_step_rank,
                        c.dim,
                    )));
                }
            }
        }
        let mut kcache = Vec::new();
        let mut vcache = Vec::new();
        let mut conv_state = Vec::new();
        let mut ssm_state = Vec::new();
        let mut snap_conv = Vec::new();
        let mut snap_ssm = Vec::new();
        for kind in &kinds {
            match kind {
                BlockKind::Attention => {
                    kcache.push(nb(MAX_SEQ * kv_dim));
                    vcache.push(nb(MAX_SEQ * kv_dim));
                    conv_state.push(nb(1));
                    ssm_state.push(nb(1));
                    snap_conv.push(nb(1));
                    snap_ssm.push(nb(1));
                }
                BlockKind::Ssm => {
                    kcache.push(nb(1));
                    vcache.push(nb(1));
                    conv_state.push(nb(conv_ch * (c.conv_kernel - 1)));
                    ssm_state.push(nb(c.time_step_rank * hv * hv));
                    snap_conv.push(nb(conv_ch * (c.conv_kernel - 1)));
                    snap_ssm.push(nb(c.time_step_rank * hv * hv));
                }
            }
        }
        eprintln!("[qwen35] GPU-resident runner ready");
        Qwen35Gpu {
            p_mv_f16: pl("matvec_f16"),
            p_mv_q4k: pl("matvec_q4k_co"),
            p_mv_q5k: pl("matvec_q5k_co"),
            p_mv_q6k: pl("matvec_q6k_co"),
            p_mv_q8: pl("matvec_q8_0_co"),
            p_rms: pl("rmsnorm"),
            p_rms_h: pl("rmsnorm_heads"),
            p_l2_h: pl("l2norm_heads"),
            p_sig_mul: pl("sigmoid_mul"),
            p_sig_ip: pl("sigmoid_inplace"),
            p_decay: pl("ssm_decay"),
            p_conv: pl("conv1d"),
            p_delta: pl("deltanet_multi"),
            p_gnorm: pl("gated_norm"),
            p_rope: pl("rope_partial"),
            p_store: pl("store_kv"),
            p_attn: pl("attention"),
            p_swiglu: pl("swiglu"),
            p_add: pl("add_inplace"),
            p_qgate: pl("extract_qgate"),
            p_mm_q4k_2tok: pl("matmul_q4k_co2"),
            p_mm_q6k_2tok: pl("matmul_q6k_co2"),
            p_copy: pl("copy_buf"),
            // Buffers hold up to 2 tokens ([tok0 | tok1]) so the same runner serves
            // the single-token decode (first slice only) and the MTP 2-token verify.
            x: nb(2 * c.dim),
            xb: nb(2 * c.dim),
            qfull: nb(2 * c.n_heads * c.head_dim * 2),
            q: nb(2 * c.n_heads * c.head_dim),
            k: nb(2 * kv_dim),
            v: nb(2 * kv_dim),
            att: nb(2 * c.n_heads * c.head_dim),
            gate: nb(2 * c.n_heads * c.head_dim),
            qkv: nb(2 * conv_ch),
            z: nb(2 * c.inner_size),
            conv_out: nb(2 * conv_ch),
            beta: nb(2 * c.time_step_rank),
            decay: nb(2 * c.time_step_rank),
            dout: nb(2 * c.inner_size),
            gbuf: nb(2 * c.ffn_dim),
            ubuf: nb(2 * c.ffn_dim),
            logits: nb(c.vocab),
            attn_norm,
            post_norm,
            q_norm,
            k_norm,
            ssm_norm,
            conv_w,
            ssm_a,
            ssm_dt,
            ssm_beta,
            ssm_alpha,
            output_norm: up(&m.output_norm),
            kcache,
            vcache,
            conv_state,
            ssm_state,
            snap_conv,
            snap_ssm,
            device,
            queue,
        }
    }

    fn bar(enc: &ComputeCommandEncoderRef, outs: &[&BufferRef]) {
        let r: Vec<&ResourceRef> = outs
            .iter()
            .map(|&b| {
                let x: &ResourceRef = b;
                x
            })
            .collect();
        enc.memory_barrier_with_resources(&r);
    }
    fn set_u(enc: &ComputeCommandEncoderRef, idx: u64, val: u32) {
        enc.set_bytes(idx, 4, &val as *const u32 as *const c_void);
    }
    fn set_f(enc: &ComputeCommandEncoderRef, idx: u64, val: f32) {
        enc.set_bytes(idx, 4, &val as *const f32 as *const c_void);
    }
    // matvec y = W x ; coalesced (32-lane/row) for K-quants, scalar for f16
    fn mv(&self, enc: &ComputeCommandEncoderRef, w: &GpuMatrix, x: &BufferRef, y: &BufferRef) {
        use crate::gguf::{GGML_F16, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0};
        let (p, coalesced) = match w.ggml_type {
            GGML_Q4_K => (&self.p_mv_q4k, true),
            GGML_Q5_K => (&self.p_mv_q5k, true),
            GGML_Q6_K => (&self.p_mv_q6k, true),
            GGML_Q8_0 => (&self.p_mv_q8, true),
            GGML_F16 => (&self.p_mv_f16, false),
            t => {
                eprintln!("[qwen35] no resident matvec for type {t}");
                std::process::exit(2);
            }
        };
        enc.set_compute_pipeline_state(p);
        enc.set_buffer(0, Some(w.buffer()), 0);
        enc.set_buffer(1, Some(x), 0);
        enc.set_buffer(2, Some(y), 0);
        Self::set_u(enc, 3, w.in_dim as u32);
        if coalesced {
            // coalesced kernels process NDST=2 rows per 32-lane simdgroup and
            // need n_rows in buffer(4) for the tail guard (must match metal_be).
            const NDST: u64 = 2;
            Self::set_u(enc, 4, w.n_rows as u32);
            let simdgroups = (w.n_rows as u64).div_ceil(NDST);
            enc.dispatch_threads(MTLSize::new(simdgroups * 32, 1, 1), MTLSize::new(32, 1, 1));
        } else {
            enc.dispatch_threads(MTLSize::new(w.n_rows as u64, 1, 1), MTLSize::new(256, 1, 1));
        }
        Self::bar(enc, &[y]);
    }
    // matvec y = W x with byte offsets into x and y (one token's slice of a 2-token buffer)
    fn mv_off(
        &self,
        enc: &ComputeCommandEncoderRef,
        w: &GpuMatrix,
        x: &BufferRef,
        xoff: u64,
        y: &BufferRef,
        yoff: u64,
    ) {
        use crate::gguf::{GGML_F16, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0};
        let (p, coalesced) = match w.ggml_type {
            GGML_Q4_K => (&self.p_mv_q4k, true),
            GGML_Q5_K => (&self.p_mv_q5k, true),
            GGML_Q6_K => (&self.p_mv_q6k, true),
            GGML_Q8_0 => (&self.p_mv_q8, true),
            GGML_F16 => (&self.p_mv_f16, false),
            t => {
                eprintln!("[qwen35] no resident matvec for type {t}");
                std::process::exit(2);
            }
        };
        enc.set_compute_pipeline_state(p);
        enc.set_buffer(0, Some(w.buffer()), 0);
        enc.set_buffer(1, Some(x), xoff);
        enc.set_buffer(2, Some(y), yoff);
        Self::set_u(enc, 3, w.in_dim as u32);
        if coalesced {
            const NDST: u64 = 2;
            Self::set_u(enc, 4, w.n_rows as u32);
            let simdgroups = (w.n_rows as u64).div_ceil(NDST);
            enc.dispatch_threads(MTLSize::new(simdgroups * 32, 1, 1), MTLSize::new(32, 1, 1));
        } else {
            enc.dispatch_threads(MTLSize::new(w.n_rows as u64, 1, 1), MTLSize::new(256, 1, 1));
        }
        Self::bar(enc, &[y]);
    }
    // 2-token batched matvec: Y=[y_t | y_d] = W · [x_t | x_d]. For q4_k/q6_k the co2
    // kernel reads each weight row ONCE for both tokens (the MTP verify win); other
    // types fall back to two per-token matvecs (correct, weight read twice).
    fn mv2(&self, enc: &ComputeCommandEncoderRef, w: &GpuMatrix, x: &BufferRef, y: &BufferRef) {
        use crate::gguf::{GGML_Q4_K, GGML_Q6_K};
        let p = match w.ggml_type {
            GGML_Q4_K => &self.p_mm_q4k_2tok,
            GGML_Q6_K => &self.p_mm_q6k_2tok,
            _ => {
                self.mv_off(enc, w, x, 0, y, 0);
                self.mv_off(enc, w, x, (w.in_dim as u64) * 4, y, (w.n_rows as u64) * 4);
                return;
            }
        };
        enc.set_compute_pipeline_state(p);
        enc.set_buffer(0, Some(w.buffer()), 0);
        enc.set_buffer(1, Some(x), 0);
        enc.set_buffer(2, Some(y), 0);
        Self::set_u(enc, 3, w.in_dim as u32);
        Self::set_u(enc, 4, w.n_rows as u32);
        const NDST: u64 = 2;
        let simdgroups = (w.n_rows as u64).div_ceil(NDST);
        enc.dispatch_threads(MTLSize::new(simdgroups * 32, 1, 1), MTLSize::new(32, 1, 1));
        Self::bar(enc, &[y]);
    }
    // GPU buffer-to-buffer copy of `n` f32 (state snapshot for MTP rollback)
    fn copy(&self, enc: &ComputeCommandEncoderRef, src: &BufferRef, dst: &BufferRef, n: usize) {
        enc.set_compute_pipeline_state(&self.p_copy);
        enc.set_buffer(0, Some(src), 0);
        enc.set_buffer(1, Some(dst), 0);
        enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(256, 1, 1));
        Self::bar(enc, &[dst]);
    }
    fn rms(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &BufferRef,
        w: &BufferRef,
        out: &BufferRef,
        n: usize,
        eps: f32,
    ) {
        enc.set_compute_pipeline_state(&self.p_rms);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(w), 0);
        enc.set_buffer(2, Some(out), 0);
        Self::set_u(enc, 3, n as u32);
        Self::set_f(enc, 4, eps);
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        Self::bar(enc, &[out]);
    }
    fn elt(
        &self,
        enc: &ComputeCommandEncoderRef,
        p: &ComputePipelineState,
        a: &BufferRef,
        b: Option<&BufferRef>,
        n: usize,
    ) {
        enc.set_compute_pipeline_state(p);
        enc.set_buffer(0, Some(a), 0);
        if let Some(bb) = b {
            enc.set_buffer(1, Some(bb), 0);
        }
        let tg = p.max_total_threads_per_threadgroup().min(256);
        enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(tg, 1, 1));
        Self::bar(enc, &[a]);
    }

    pub fn forward(&self, m: &Qwen35, token: u32, pos: usize) -> Vec<f32> {
        let c = &m.cfg;
        let (dim, hd, nh, nkv) = (c.dim, c.head_dim, c.n_heads, c.n_kv_heads);
        let kv_dim = nkv * hd;
        let kv_mul = nh / nkv;
        let hv = c.inner_size / c.time_step_rank;
        let conv_ch = c.inner_size + 2 * c.group_count * c.state_size;
        let q0b = 0u64;
        let k0b = (c.group_count * c.state_size * 4) as u64;
        let v0b = (2 * c.group_count * c.state_size * 4) as u64;
        let f4 = 4u64;

        // embedding -> x
        let row = &m.tok_embd[token as usize * dim..(token as usize + 1) * dim];
        unsafe {
            std::ptr::copy_nonoverlapping(row.as_ptr(), self.x.contents() as *mut f32, dim);
        }

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        for (il, blk) in m.blocks.iter().enumerate() {
            self.rms(enc, &self.x, &self.attn_norm[il], &self.xb, dim, c.rms_eps);
            match blk {
                Block::Attn(w) => {
                    self.mv(enc, w.wq.as_gpu(), &self.xb, &self.qfull);
                    // extract q + gate
                    enc.set_compute_pipeline_state(&self.p_qgate);
                    enc.set_buffer(0, Some(&self.qfull), 0);
                    enc.set_buffer(1, Some(&self.q), 0);
                    enc.set_buffer(2, Some(&self.gate), 0);
                    Self::set_u(enc, 3, hd as u32);
                    enc.dispatch_threads(
                        MTLSize::new((nh * hd) as u64, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    Self::bar(enc, &[&self.q, &self.gate]);
                    self.mv(enc, w.wk.as_gpu(), &self.xb, &self.k);
                    self.mv(enc, w.wv.as_gpu(), &self.xb, &self.v);
                    // qk-norm per head
                    enc.set_compute_pipeline_state(&self.p_rms_h);
                    enc.set_buffer(0, Some(&self.q), 0);
                    enc.set_buffer(1, Some(self.q_norm[il].as_ref().unwrap()), 0);
                    Self::set_u(enc, 2, hd as u32);
                    Self::set_f(enc, 3, c.rms_eps);
                    enc.dispatch_thread_groups(
                        MTLSize::new(nh as u64, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    Self::bar(enc, &[&self.q]);
                    enc.set_compute_pipeline_state(&self.p_rms_h);
                    enc.set_buffer(0, Some(&self.k), 0);
                    enc.set_buffer(1, Some(self.k_norm[il].as_ref().unwrap()), 0);
                    Self::set_u(enc, 2, hd as u32);
                    Self::set_f(enc, 3, c.rms_eps);
                    enc.dispatch_thread_groups(
                        MTLSize::new(nkv as u64, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    Self::bar(enc, &[&self.k]);
                    // partial rope
                    for (buf, heads) in [(&self.q, nh), (&self.k, nkv)] {
                        enc.set_compute_pipeline_state(&self.p_rope);
                        enc.set_buffer(0, Some(buf), 0);
                        Self::set_u(enc, 1, hd as u32);
                        Self::set_u(enc, 2, c.rope_dim as u32);
                        Self::set_u(enc, 3, pos as u32);
                        Self::set_f(enc, 4, c.rope_base);
                        enc.dispatch_threads(
                            MTLSize::new((heads * c.rope_dim / 2) as u64, 1, 1),
                            MTLSize::new(64, 1, 1),
                        );
                        Self::bar(enc, &[buf]);
                    }
                    // store kv
                    enc.set_compute_pipeline_state(&self.p_store);
                    enc.set_buffer(0, Some(&self.k), 0);
                    enc.set_buffer(1, Some(&self.v), 0);
                    enc.set_buffer(2, Some(&self.kcache[il]), 0);
                    enc.set_buffer(3, Some(&self.vcache[il]), 0);
                    Self::set_u(enc, 4, kv_dim as u32);
                    Self::set_u(enc, 5, pos as u32);
                    enc.dispatch_threads(
                        MTLSize::new(kv_dim as u64, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    Self::bar(enc, &[&self.kcache[il], &self.vcache[il]]);
                    // attention
                    enc.set_compute_pipeline_state(&self.p_attn);
                    enc.set_buffer(0, Some(&self.q), 0);
                    enc.set_buffer(1, Some(&self.kcache[il]), 0);
                    enc.set_buffer(2, Some(&self.vcache[il]), 0);
                    enc.set_buffer(3, Some(&self.att), 0);
                    Self::set_u(enc, 4, hd as u32);
                    Self::set_u(enc, 5, kv_dim as u32);
                    Self::set_u(enc, 6, kv_mul as u32);
                    Self::set_u(enc, 7, pos as u32);
                    // new attention kernel: one threadgroup per head, hd threads
                    enc.dispatch_thread_groups(
                        MTLSize::new(nh as u64, 1, 1),
                        MTLSize::new(hd as u64, 1, 1),
                    );
                    Self::bar(enc, &[&self.att]);
                    self.elt(enc, &self.p_sig_mul, &self.att, Some(&self.gate), nh * hd);
                    self.mv(enc, w.wo.as_gpu(), &self.att, &self.xb);
                }
                Block::Lin(w) => {
                    self.mv(enc, w.wqkv.as_gpu(), &self.xb, &self.qkv);
                    self.mv(enc, w.wz.as_gpu(), &self.xb, &self.z);
                    self.mv(
                        enc,
                        self.ssm_beta[il].as_ref().unwrap(),
                        &self.xb,
                        &self.beta,
                    );
                    self.elt(enc, &self.p_sig_ip, &self.beta, None, c.time_step_rank);
                    self.mv(
                        enc,
                        self.ssm_alpha[il].as_ref().unwrap(),
                        &self.xb,
                        &self.decay,
                    );
                    enc.set_compute_pipeline_state(&self.p_decay);
                    enc.set_buffer(0, Some(&self.decay), 0);
                    enc.set_buffer(1, Some(self.ssm_a[il].as_ref().unwrap()), 0);
                    enc.set_buffer(2, Some(self.ssm_dt[il].as_ref().unwrap()), 0);
                    enc.dispatch_threads(
                        MTLSize::new(c.time_step_rank as u64, 1, 1),
                        MTLSize::new(32, 1, 1),
                    );
                    Self::bar(enc, &[&self.decay]);
                    // conv1d
                    enc.set_compute_pipeline_state(&self.p_conv);
                    enc.set_buffer(0, Some(&self.qkv), 0);
                    enc.set_buffer(1, Some(&self.conv_state[il]), 0);
                    enc.set_buffer(2, Some(self.conv_w[il].as_ref().unwrap()), 0);
                    enc.set_buffer(3, Some(&self.conv_out), 0);
                    Self::set_u(enc, 4, c.conv_kernel as u32);
                    enc.dispatch_threads(
                        MTLSize::new(conv_ch as u64, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    Self::bar(enc, &[&self.conv_out, &self.conv_state[il]]);
                    // l2 norm q and k regions
                    for off in [q0b, k0b] {
                        enc.set_compute_pipeline_state(&self.p_l2_h);
                        enc.set_buffer(0, Some(&self.conv_out), off);
                        Self::set_u(enc, 1, c.state_size as u32);
                        Self::set_f(enc, 2, c.rms_eps);
                        enc.dispatch_thread_groups(
                            MTLSize::new(c.group_count as u64, 1, 1),
                            MTLSize::new(128, 1, 1),
                        );
                        Self::bar(enc, &[&self.conv_out]);
                    }
                    // deltanet
                    enc.set_compute_pipeline_state(&self.p_delta);
                    enc.set_buffer(0, Some(&self.ssm_state[il]), 0);
                    enc.set_buffer(1, Some(&self.conv_out), q0b);
                    enc.set_buffer(2, Some(&self.conv_out), k0b);
                    enc.set_buffer(3, Some(&self.conv_out), v0b);
                    enc.set_buffer(4, Some(&self.decay), 0);
                    enc.set_buffer(5, Some(&self.beta), 0);
                    enc.set_buffer(6, Some(&self.dout), 0);
                    Self::set_u(enc, 7, hv as u32);
                    Self::set_u(enc, 8, c.state_size as u32);
                    Self::set_u(enc, 9, c.group_count as u32);
                    Self::set_f(enc, 10, 1.0 / (c.state_size as f32).sqrt());
                    enc.dispatch_thread_groups(
                        MTLSize::new(c.time_step_rank as u64, 1, 1),
                        MTLSize::new(hv as u64, 1, 1),
                    );
                    Self::bar(enc, &[&self.dout, &self.ssm_state[il]]);
                    // gated norm
                    enc.set_compute_pipeline_state(&self.p_gnorm);
                    enc.set_buffer(0, Some(&self.dout), 0);
                    enc.set_buffer(1, Some(self.ssm_norm[il].as_ref().unwrap()), 0);
                    enc.set_buffer(2, Some(&self.z), 0);
                    Self::set_u(enc, 3, hv as u32);
                    Self::set_f(enc, 4, c.rms_eps);
                    enc.dispatch_thread_groups(
                        MTLSize::new(c.time_step_rank as u64, 1, 1),
                        MTLSize::new(hv as u64, 1, 1),
                    );
                    Self::bar(enc, &[&self.dout]);
                    self.mv(enc, w.ssm_out.as_gpu(), &self.dout, &self.xb);
                }
            }
            self.elt(enc, &self.p_add, &self.x, Some(&self.xb), dim);
            // FFN
            let (fg, fu, fd) = match blk {
                Block::Attn(w) => (&w.ffn_gate, &w.ffn_up, &w.ffn_down),
                Block::Lin(w) => (&w.ffn_gate, &w.ffn_up, &w.ffn_down),
            };
            self.rms(enc, &self.x, &self.post_norm[il], &self.xb, dim, c.rms_eps);
            self.mv(enc, fg.as_gpu(), &self.xb, &self.gbuf);
            self.mv(enc, fu.as_gpu(), &self.xb, &self.ubuf);
            self.elt(enc, &self.p_swiglu, &self.gbuf, Some(&self.ubuf), c.ffn_dim);
            self.mv(enc, fd.as_gpu(), &self.gbuf, &self.xb);
            self.elt(enc, &self.p_add, &self.x, Some(&self.xb), dim);
            let _ = f4;
        }
        self.rms(enc, &self.x, &self.output_norm, &self.xb, dim, c.rms_eps);
        self.mv(enc, m.output.as_gpu(), &self.xb, &self.logits);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        unsafe { std::slice::from_raw_parts(self.logits.contents() as *const f32, c.vocab) }
            .to_vec()
    }

    /// 2-token MTP verify on the resident runner: process confirmed token `t` (at
    /// `pos`) and draft token `d` (at `pos+1`) through the whole backbone in ONE
    /// command buffer. The matmuls are batched (co2 reads each weight row once for
    /// both tokens — the bandwidth win), while the per-token SSM/conv/attention run
    /// twice with per-token buffer offsets. Conv/SSM state is snapshotted after
    /// token t (into snap_conv/snap_ssm) so a rejected draft rolls back via
    /// `restore_snap` with no reprocess. Returns both pre-output-norm hidden states
    /// `[hidden_t (dim) | hidden_d (dim)]`; the caller applies `head` for logits.
    /// Additive: single-token `forward` is untouched.
    pub fn forward2(&self, m: &Qwen35, t: u32, d: u32, pos: usize) -> Vec<f32> {
        let c = &m.cfg;
        let (dim, hd, nh, nkv) = (c.dim, c.head_dim, c.n_heads, c.n_kv_heads);
        let kv_dim = nkv * hd;
        let kv_mul = nh / nkv;
        let hv = c.inner_size / c.time_step_rank;
        let conv_ch = c.inner_size + 2 * c.group_count * c.state_size;
        let q0b = 0u64;
        let k0b = (c.group_count * c.state_size * 4) as u64;
        let v0b = (2 * c.group_count * c.state_size * 4) as u64;
        let eps = c.rms_eps;
        // per-token byte strides
        let ox = (dim * 4) as u64;
        let oqf = (nh * hd * 2 * 4) as u64;
        let oq = (nh * hd * 4) as u64;
        let okv = (kv_dim * 4) as u64;
        let och = (conv_ch * 4) as u64;
        let oin = (c.inner_size * 4) as u64;
        let otsr = (c.time_step_rank * 4) as u64;
        let offn = (c.ffn_dim * 4) as u64;

        // embed both tokens into x = [emb(t) | emb(d)]
        for (slot, &tok) in [t, d].iter().enumerate() {
            let row = &m.tok_embd[tok as usize * dim..(tok as usize + 1) * dim];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    row.as_ptr(),
                    (self.x.contents() as *mut f32).add(slot * dim),
                    dim,
                );
            }
        }

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        // per-token RMSNorm helper (offset-aware, single-token dispatch)
        let rms_o = |x: &BufferRef, xo: u64, w: &BufferRef, out: &BufferRef, oo: u64, n: usize| {
            enc.set_compute_pipeline_state(&self.p_rms);
            enc.set_buffer(0, Some(x), xo);
            enc.set_buffer(1, Some(w), 0);
            enc.set_buffer(2, Some(out), oo);
            Self::set_u(enc, 3, n as u32);
            Self::set_f(enc, 4, eps);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
            Self::bar(enc, &[out]);
        };

        for (il, blk) in m.blocks.iter().enumerate() {
            for tau in 0..2u64 {
                rms_o(&self.x, tau * ox, &self.attn_norm[il], &self.xb, tau * ox, dim);
            }
            match blk {
                Block::Attn(w) => {
                    self.mv2(enc, w.wq.as_gpu(), &self.xb, &self.qfull);
                    for tau in 0..2u64 {
                        enc.set_compute_pipeline_state(&self.p_qgate);
                        enc.set_buffer(0, Some(&self.qfull), tau * oqf);
                        enc.set_buffer(1, Some(&self.q), tau * oq);
                        enc.set_buffer(2, Some(&self.gate), tau * oq);
                        Self::set_u(enc, 3, hd as u32);
                        enc.dispatch_threads(
                            MTLSize::new((nh * hd) as u64, 1, 1),
                            MTLSize::new(256, 1, 1),
                        );
                    }
                    Self::bar(enc, &[&self.q, &self.gate]);
                    self.mv2(enc, w.wk.as_gpu(), &self.xb, &self.k);
                    self.mv2(enc, w.wv.as_gpu(), &self.xb, &self.v);
                    for tau in 0..2u64 {
                        enc.set_compute_pipeline_state(&self.p_rms_h);
                        enc.set_buffer(0, Some(&self.q), tau * oq);
                        enc.set_buffer(1, Some(self.q_norm[il].as_ref().unwrap()), 0);
                        Self::set_u(enc, 2, hd as u32);
                        Self::set_f(enc, 3, eps);
                        enc.dispatch_thread_groups(
                            MTLSize::new(nh as u64, 1, 1),
                            MTLSize::new(256, 1, 1),
                        );
                        enc.set_compute_pipeline_state(&self.p_rms_h);
                        enc.set_buffer(0, Some(&self.k), tau * okv);
                        enc.set_buffer(1, Some(self.k_norm[il].as_ref().unwrap()), 0);
                        Self::set_u(enc, 2, hd as u32);
                        Self::set_f(enc, 3, eps);
                        enc.dispatch_thread_groups(
                            MTLSize::new(nkv as u64, 1, 1),
                            MTLSize::new(256, 1, 1),
                        );
                    }
                    Self::bar(enc, &[&self.q, &self.k]);
                    // partial rope per token (token tau at pos+tau)
                    for tau in 0..2u64 {
                        for (buf, off, heads) in
                            [(&self.q, tau * oq, nh), (&self.k, tau * okv, nkv)]
                        {
                            enc.set_compute_pipeline_state(&self.p_rope);
                            enc.set_buffer(0, Some(buf), off);
                            Self::set_u(enc, 1, hd as u32);
                            Self::set_u(enc, 2, c.rope_dim as u32);
                            Self::set_u(enc, 3, (pos + tau as usize) as u32);
                            Self::set_f(enc, 4, c.rope_base);
                            enc.dispatch_threads(
                                MTLSize::new((heads * c.rope_dim / 2) as u64, 1, 1),
                                MTLSize::new(64, 1, 1),
                            );
                        }
                    }
                    Self::bar(enc, &[&self.q, &self.k]);
                    // store kv then attend, sequenced per token (d attends over t's kv)
                    for tau in 0..2u64 {
                        enc.set_compute_pipeline_state(&self.p_store);
                        enc.set_buffer(0, Some(&self.k), tau * okv);
                        enc.set_buffer(1, Some(&self.v), tau * okv);
                        enc.set_buffer(2, Some(&self.kcache[il]), 0);
                        enc.set_buffer(3, Some(&self.vcache[il]), 0);
                        Self::set_u(enc, 4, kv_dim as u32);
                        Self::set_u(enc, 5, (pos + tau as usize) as u32);
                        enc.dispatch_threads(
                            MTLSize::new(kv_dim as u64, 1, 1),
                            MTLSize::new(256, 1, 1),
                        );
                        Self::bar(enc, &[&self.kcache[il], &self.vcache[il]]);
                        enc.set_compute_pipeline_state(&self.p_attn);
                        enc.set_buffer(0, Some(&self.q), tau * oq);
                        enc.set_buffer(1, Some(&self.kcache[il]), 0);
                        enc.set_buffer(2, Some(&self.vcache[il]), 0);
                        enc.set_buffer(3, Some(&self.att), tau * oq);
                        Self::set_u(enc, 4, hd as u32);
                        Self::set_u(enc, 5, kv_dim as u32);
                        Self::set_u(enc, 6, kv_mul as u32);
                        Self::set_u(enc, 7, (pos + tau as usize) as u32);
                        enc.dispatch_thread_groups(
                            MTLSize::new(nh as u64, 1, 1),
                            MTLSize::new(hd as u64, 1, 1),
                        );
                        Self::bar(enc, &[&self.att]);
                    }
                    for tau in 0..2u64 {
                        enc.set_compute_pipeline_state(&self.p_sig_mul);
                        enc.set_buffer(0, Some(&self.att), tau * oq);
                        enc.set_buffer(1, Some(&self.gate), tau * oq);
                        enc.dispatch_threads(
                            MTLSize::new((nh * hd) as u64, 1, 1),
                            MTLSize::new(256, 1, 1),
                        );
                    }
                    Self::bar(enc, &[&self.att]);
                    self.mv2(enc, w.wo.as_gpu(), &self.att, &self.xb);
                }
                Block::Lin(w) => {
                    self.mv2(enc, w.wqkv.as_gpu(), &self.xb, &self.qkv);
                    self.mv2(enc, w.wz.as_gpu(), &self.xb, &self.z);
                    self.mv2(enc, self.ssm_beta[il].as_ref().unwrap(), &self.xb, &self.beta);
                    self.mv2(enc, self.ssm_alpha[il].as_ref().unwrap(), &self.xb, &self.decay);
                    for tau in 0..2u64 {
                        enc.set_compute_pipeline_state(&self.p_sig_ip);
                        enc.set_buffer(0, Some(&self.beta), tau * otsr);
                        enc.dispatch_threads(
                            MTLSize::new(c.time_step_rank as u64, 1, 1),
                            MTLSize::new(256, 1, 1),
                        );
                        enc.set_compute_pipeline_state(&self.p_decay);
                        enc.set_buffer(0, Some(&self.decay), tau * otsr);
                        enc.set_buffer(1, Some(self.ssm_a[il].as_ref().unwrap()), 0);
                        enc.set_buffer(2, Some(self.ssm_dt[il].as_ref().unwrap()), 0);
                        enc.dispatch_threads(
                            MTLSize::new(c.time_step_rank as u64, 1, 1),
                            MTLSize::new(32, 1, 1),
                        );
                    }
                    Self::bar(enc, &[&self.beta, &self.decay]);
                    // conv1d: recurrent -> token t, snapshot conv state, token d
                    for tau in 0..2u64 {
                        enc.set_compute_pipeline_state(&self.p_conv);
                        enc.set_buffer(0, Some(&self.qkv), tau * och);
                        enc.set_buffer(1, Some(&self.conv_state[il]), 0);
                        enc.set_buffer(2, Some(self.conv_w[il].as_ref().unwrap()), 0);
                        enc.set_buffer(3, Some(&self.conv_out), tau * och);
                        Self::set_u(enc, 4, c.conv_kernel as u32);
                        enc.dispatch_threads(
                            MTLSize::new(conv_ch as u64, 1, 1),
                            MTLSize::new(256, 1, 1),
                        );
                        Self::bar(enc, &[&self.conv_out, &self.conv_state[il]]);
                        if tau == 0 {
                            self.copy(
                                enc,
                                &self.conv_state[il],
                                &self.snap_conv[il],
                                conv_ch * (c.conv_kernel - 1),
                            );
                        }
                    }
                    for tau in 0..2u64 {
                        for off in [q0b, k0b] {
                            enc.set_compute_pipeline_state(&self.p_l2_h);
                            enc.set_buffer(0, Some(&self.conv_out), tau * och + off);
                            Self::set_u(enc, 1, c.state_size as u32);
                            Self::set_f(enc, 2, eps);
                            enc.dispatch_thread_groups(
                                MTLSize::new(c.group_count as u64, 1, 1),
                                MTLSize::new(128, 1, 1),
                            );
                        }
                    }
                    Self::bar(enc, &[&self.conv_out]);
                    // deltanet: recurrent -> token t, snapshot ssm state, token d
                    for tau in 0..2u64 {
                        enc.set_compute_pipeline_state(&self.p_delta);
                        enc.set_buffer(0, Some(&self.ssm_state[il]), 0);
                        enc.set_buffer(1, Some(&self.conv_out), tau * och + q0b);
                        enc.set_buffer(2, Some(&self.conv_out), tau * och + k0b);
                        enc.set_buffer(3, Some(&self.conv_out), tau * och + v0b);
                        enc.set_buffer(4, Some(&self.decay), tau * otsr);
                        enc.set_buffer(5, Some(&self.beta), tau * otsr);
                        enc.set_buffer(6, Some(&self.dout), tau * oin);
                        Self::set_u(enc, 7, hv as u32);
                        Self::set_u(enc, 8, c.state_size as u32);
                        Self::set_u(enc, 9, c.group_count as u32);
                        Self::set_f(enc, 10, 1.0 / (c.state_size as f32).sqrt());
                        enc.dispatch_thread_groups(
                            MTLSize::new(c.time_step_rank as u64, 1, 1),
                            MTLSize::new(hv as u64, 1, 1),
                        );
                        Self::bar(enc, &[&self.dout, &self.ssm_state[il]]);
                        if tau == 0 {
                            self.copy(
                                enc,
                                &self.ssm_state[il],
                                &self.snap_ssm[il],
                                c.time_step_rank * hv * hv,
                            );
                        }
                    }
                    for tau in 0..2u64 {
                        enc.set_compute_pipeline_state(&self.p_gnorm);
                        enc.set_buffer(0, Some(&self.dout), tau * oin);
                        enc.set_buffer(1, Some(self.ssm_norm[il].as_ref().unwrap()), 0);
                        enc.set_buffer(2, Some(&self.z), tau * oin);
                        Self::set_u(enc, 3, hv as u32);
                        Self::set_f(enc, 4, eps);
                        enc.dispatch_thread_groups(
                            MTLSize::new(c.time_step_rank as u64, 1, 1),
                            MTLSize::new(hv as u64, 1, 1),
                        );
                    }
                    Self::bar(enc, &[&self.dout]);
                    self.mv2(enc, w.ssm_out.as_gpu(), &self.dout, &self.xb);
                }
            }
            for tau in 0..2u64 {
                enc.set_compute_pipeline_state(&self.p_add);
                enc.set_buffer(0, Some(&self.x), tau * ox);
                enc.set_buffer(1, Some(&self.xb), tau * ox);
                enc.dispatch_threads(MTLSize::new(dim as u64, 1, 1), MTLSize::new(256, 1, 1));
            }
            Self::bar(enc, &[&self.x]);
            // FFN
            let (fg, fu, fd) = match blk {
                Block::Attn(w) => (&w.ffn_gate, &w.ffn_up, &w.ffn_down),
                Block::Lin(w) => (&w.ffn_gate, &w.ffn_up, &w.ffn_down),
            };
            for tau in 0..2u64 {
                rms_o(&self.x, tau * ox, &self.post_norm[il], &self.xb, tau * ox, dim);
            }
            self.mv2(enc, fg.as_gpu(), &self.xb, &self.gbuf);
            self.mv2(enc, fu.as_gpu(), &self.xb, &self.ubuf);
            for tau in 0..2u64 {
                enc.set_compute_pipeline_state(&self.p_swiglu);
                enc.set_buffer(0, Some(&self.gbuf), tau * offn);
                enc.set_buffer(1, Some(&self.ubuf), tau * offn);
                enc.dispatch_threads(
                    MTLSize::new(c.ffn_dim as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            }
            Self::bar(enc, &[&self.gbuf]);
            self.mv2(enc, fd.as_gpu(), &self.gbuf, &self.xb);
            for tau in 0..2u64 {
                enc.set_compute_pipeline_state(&self.p_add);
                enc.set_buffer(0, Some(&self.x), tau * ox);
                enc.set_buffer(1, Some(&self.xb), tau * ox);
                enc.dispatch_threads(MTLSize::new(dim as u64, 1, 1), MTLSize::new(256, 1, 1));
            }
            Self::bar(enc, &[&self.x]);
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        // return both pre-output-norm hidden states [hidden_t | hidden_d]
        unsafe { std::slice::from_raw_parts(self.x.contents() as *const f32, 2 * dim) }.to_vec()
    }

    /// Roll conv/SSM state back to the post-token-t snapshot (after a rejected MTP
    /// draft). KV cache needs no rollback: the draft's stale slot at pos+1 is simply
    /// overwritten on the next step.
    pub fn restore_snap(&self, m: &Qwen35) {
        let c = &m.cfg;
        let hv = c.inner_size / c.time_step_rank;
        let conv_ch = c.inner_size + 2 * c.group_count * c.state_size;
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        for (il, blk) in m.blocks.iter().enumerate() {
            if let Block::Lin(_) = blk {
                self.copy(
                    enc,
                    &self.snap_conv[il],
                    &self.conv_state[il],
                    conv_ch * (c.conv_kernel - 1),
                );
                self.copy(
                    enc,
                    &self.snap_ssm[il],
                    &self.ssm_state[il],
                    c.time_step_rank * hv * hv,
                );
            }
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Load a CPU `State` (from batched prefill) into the resident GPU buffers, so
    /// decode can continue on the fast resident runner. Buffer layouts match the
    /// CPU forward exactly (attn→KV cache, lin→conv/ssm state), verified by the
    /// per-block sizing. Purely additive: the CPU decode path is untouched.
    pub fn upload_state(&self, m: &Qwen35, st: &State, pos: usize) {
        let c = &m.cfg;
        let kv_dim = c.n_kv_heads * c.head_dim;
        let up = |buf: &Buffer, data: &[f32]| unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buf.contents() as *mut f32, data.len());
        };
        for (il, blk) in m.blocks.iter().enumerate() {
            match blk {
                Block::Attn(_) => {
                    let n = pos * kv_dim;
                    up(&self.kcache[il], &st.k_cache[il][..n]);
                    up(&self.vcache[il], &st.v_cache[il][..n]);
                }
                Block::Lin(_) => {
                    up(&self.conv_state[il], &st.conv[il]);
                    up(&self.ssm_state[il], &st.ssm[il]);
                }
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn block_kinds_from(m: &Qwen35) -> Vec<BlockKind> {
    m.blocks
        .iter()
        .map(|b| match b {
            Block::Attn(_) => BlockKind::Attention,
            Block::Lin(_) => BlockKind::Ssm,
        })
        .collect()
}

// ============================================================================
// ChatSession: the front-door chat backend for qwen35, mirroring Engine::chat.
// Renders history through the detected chat template (ChatML) so control tokens
// splice in atomically, then samples with stop tokens + boundary-safe UTF-8
// streaming. Used by both the `hos` CLI (--chat) and `flwr` serve/chat/run.
// Correctness-first path: GPU matmuls + CPU recurrence (owns Option<Gpu>).
// ============================================================================
/// Reasoning depth, matching the model's `reasoning_effort` (xhigh default).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Effort {
    Low,
    Medium,
    Xhigh,
}

impl Effort {
    pub fn parse(s: &str) -> Option<Effort> {
        match s.to_ascii_lowercase().as_str() {
            "low" => Some(Effort::Low),
            "medium" | "mid" | "med" => Some(Effort::Medium),
            "xhigh" | "high" => Some(Effort::Xhigh),
            _ => None,
        }
    }
}

/// Thinking-mode configuration for a turn. `on=false` => non-thinking (the
/// assistant turn is pre-closed with an empty think block).
#[derive(Clone, Copy, Debug)]
pub struct Think {
    pub on: bool,
    pub effort: Effort,
}

impl Default for Think {
    fn default() -> Think {
        Think {
            on: true,
            effort: Effort::Xhigh,
        }
    }
}

/// The `reasoning_effort` system instruction the model's own chat template
/// injects. Medium is the balanced default and adds nothing.
fn reasoning_instructions(effort: Effort) -> &'static str {
    match effort {
        Effort::Xhigh => "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
        Effort::Low => "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
        Effort::Medium => "",
    }
}

/// A streamed piece of a reply: model reasoning (inside `<think>…</think>`) vs the
/// user-facing answer. Lets callers show/hide/relocate the reasoning.
#[derive(Clone, Copy, Debug)]
pub enum Chunk<'a> {
    Reasoning(&'a str),
    Answer(&'a str),
}

pub struct ChatSession {
    model: Qwen35,
    tok: Tokenizer,
    state: State,
    gpu: Option<Gpu>,
    fam: ChatFamily,
    ctx: usize,
    think_close: Option<u32>,
    vision: Option<crate::qwen35_vision::VisionTower>,
    // Fast resident-GPU decode runner (batched prefill hands its state to this via
    // upload_state, then decode continues here at ~+26%). macOS+GPU only.
    #[cfg(target_os = "macos")]
    resident: Option<Qwen35Gpu>,
}

impl ChatSession {
    /// Load a qwen35 GGUF into a ready chat session. `gpu` uses Metal matmuls
    /// (macOS) for the projections; the SSM recurrence runs on CPU.
    pub fn load<S: crate::model::ModelSource>(g: &S, tok: Tokenizer, gpu: bool) -> Result<ChatSession> {
        let ctx = g
            .meta_u64("qwen35.context_length")
            .unwrap_or(32768)
            .min(32768) as usize;
        let gpu = if gpu && cfg!(target_os = "macos") {
            Some(Gpu::new())
        } else {
            None
        };
        let model = Qwen35::load(g, gpu.as_ref())?;
        let state = State::new(&model);
        let fam = ChatFamily::detect(&tok, Arch::Qwen35Hybrid);
        let think_close = tok.special_id("</think>");
        // Build the resident-GPU decode runner (macOS+GPU). Prefill stays batched;
        // its state is handed to this runner for faster per-token decode.
        #[cfg(target_os = "macos")]
        let resident = gpu.as_ref().map(|g| Qwen35Gpu::new(g, &model));
        Ok(ChatSession {
            model,
            tok,
            state,
            gpu,
            fam,
            ctx,
            think_close,
            vision: None,
            #[cfg(target_os = "macos")]
            resident,
        })
    }

    /// One decode step: use the resident GPU runner when available (its state was
    /// seeded from the batched prefill), else the CPU-recurrence forward.
    fn step(&mut self, tok: u32, pos: usize) -> Vec<f32> {
        #[cfg(target_os = "macos")]
        if let Some(r) = &self.resident {
            return r.forward(&self.model, tok, pos);
        }
        self.model.forward(&mut self.state, tok, pos, self.gpu.as_ref())
    }

    /// Attach a vision tower from an mmproj GGUF so this session can answer about
    /// images (`chat_img`). Optional — text-only sessions never load it.
    pub fn attach_vision<S: crate::model::ModelSource>(&mut self, mmproj: &S) -> Result<()> {
        self.vision = Some(crate::qwen35_vision::VisionTower::load(mmproj)?);
        Ok(())
    }

    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    /// Encode an image into spliceable embeddings (content-addressed disk cache).
    /// Returns an error if no vision tower is attached.
    pub fn encode_image(&self, path: &std::path::Path) -> Result<Vec<f32>> {
        self.vision
            .as_ref()
            .ok_or_else(|| HosError::Format("no vision tower attached".into()))?
            .encode_image_cached(path)
    }

    /// Encode raw image bytes (base64 `image_url` from an OpenAI request).
    pub fn encode_image_bytes(&self, bytes: &[u8]) -> Result<Vec<f32>> {
        self.vision
            .as_ref()
            .ok_or_else(|| HosError::Format("no vision tower attached".into()))?
            .encode_image_bytes(bytes)
    }

    /// Render `msgs` into the token stream, matching the model's chat template:
    /// splice ChatML control tokens + `<think>` atomically, inject the
    /// `reasoning_effort` instruction into the system turn, and open (or pre-close)
    /// the assistant `<think>` per `think.on`. Past assistant turns are stored
    /// answer-only, so no stale reasoning is replayed (preserve_thinking=false).
    /// Render the prompt. When `n_img > 0`, a vision block
    /// (`<|vision_start|>` + n_img·`<|image_pad|>` + `<|vision_end|>`) is spliced at
    /// the front of the LAST user turn's content; the returned `Vec<usize>` lists
    /// the sequence positions of those image-pad tokens (in order) so the caller
    /// can feed vision embeddings there. Empty when `n_img == 0`.
    fn build_prompt(&self, msgs: &[Message], think: Think, n_img: usize) -> (Vec<u32>, Vec<usize>) {
        let mut ids = Vec::new();
        let mut img_pos = Vec::new();
        let sp = |ids: &mut Vec<u32>, s: &str| {
            if let Some(id) = self.tok.special_id(s) {
                ids.push(id);
            }
        };
        let tx = |ids: &mut Vec<u32>, s: &str| ids.extend(self.tok.encode(s, false));

        let instr = if think.on {
            reasoning_instructions(think.effort)
        } else {
            ""
        };
        let sys = msgs.first().filter(|m| m.role == "system");
        let sys_content = match (sys, instr.is_empty()) {
            (Some(m), false) => Some(format!("{}\n\n{}", instr, m.content)),
            (Some(m), true) => Some(m.content.clone()),
            (None, false) => Some(instr.to_string()),
            (None, true) => None,
        };
        if let Some(sc) = sys_content {
            sp(&mut ids, "<|im_start|>");
            tx(&mut ids, &format!("system\n{sc}"));
            sp(&mut ids, "<|im_end|>");
            tx(&mut ids, "\n");
        }
        let start = if sys.is_some() { 1 } else { 0 };
        let last_user = msgs
            .iter()
            .enumerate()
            .rev()
            .find(|(_, m)| m.role == "user")
            .map(|(i, _)| i);
        let img_pad = self.tok.special_id("<|image_pad|>");
        for (i, m) in msgs.iter().enumerate().skip(start) {
            sp(&mut ids, "<|im_start|>");
            tx(&mut ids, &format!("{}\n", m.role));
            // Splice the vision block into the last user turn.
            if n_img > 0 && Some(i) == last_user {
                sp(&mut ids, "<|vision_start|>");
                if let Some(pad) = img_pad {
                    for _ in 0..n_img {
                        img_pos.push(ids.len());
                        ids.push(pad);
                    }
                }
                sp(&mut ids, "<|vision_end|>");
            }
            tx(&mut ids, &m.content);
            sp(&mut ids, "<|im_end|>");
            tx(&mut ids, "\n");
        }
        // Generation prompt: open <think> (thinking) or pre-close it (non-thinking).
        sp(&mut ids, "<|im_start|>");
        tx(&mut ids, "assistant\n");
        if think.on {
            sp(&mut ids, "<think>");
            tx(&mut ids, "\n");
        } else {
            sp(&mut ids, "<think>");
            tx(&mut ids, "\n\n");
            sp(&mut ids, "</think>");
            tx(&mut ids, "\n\n");
        }
        (ids, img_pos)
    }

    pub fn family_label(&self) -> &'static str {
        self.fam.label()
    }

    /// (dim, n_layers, n_heads, n_kv_heads, ctx) for banner/status display.
    pub fn dims(&self) -> (usize, usize, usize, usize, usize) {
        let c = &self.model.cfg;
        (c.dim, c.n_layers, c.n_heads, c.n_kv_heads, self.ctx)
    }

    /// Run one assistant turn over `msgs`, streaming reasoning + answer chunks to
    /// `on`. `</think>` is a single atomic token, so the split is exact: everything
    /// before it is `Chunk::Reasoning`, everything after is `Chunk::Answer`. Resets
    /// state and re-renders history each call. Returns tokens emitted.
    #[allow(clippy::too_many_arguments)]
    pub fn chat(
        &mut self,
        msgs: &[Message],
        think: Think,
        max_tokens: usize,
        temp: f32,
        top_k: usize,
        top_p: f32,
        rep_penalty: f32,
        repeat_last_n: usize,
        seed: u64,
        on: impl FnMut(Chunk),
    ) -> usize {
        self.chat_img(
            msgs, None, think, max_tokens, temp, top_k, top_p, rep_penalty, repeat_last_n, seed, on,
        )
    }

    /// Like `chat`, but with optional pre-encoded image embeddings (flat
    /// `[n_img * dim]`) spliced at the `<|image_pad|>` slots of the last user turn.
    #[allow(clippy::too_many_arguments)]
    pub fn chat_img(
        &mut self,
        msgs: &[Message],
        img_emb: Option<&[f32]>,
        think: Think,
        max_tokens: usize,
        temp: f32,
        top_k: usize,
        top_p: f32,
        rep_penalty: f32,
        repeat_last_n: usize,
        seed: u64,
        mut on: impl FnMut(Chunk),
    ) -> usize {
        let dim = self.model.cfg.dim;
        let n_img = img_emb.map(|e| e.len() / dim).unwrap_or(0);
        let (ids, img_pos) = self.build_prompt(msgs, think, n_img);
        let stops = chat::stop_ids(&self.tok, self.fam);
        self.state = State::new(&self.model);
        let gpu = self.gpu.as_ref();

        // Build the [N, dim] input embeddings: token lookups, with the image
        // embeddings spliced at their <|image_pad|> slots. Then batched prefill
        // (FFN read once across all tokens) returns the last token's logits.
        let mut x0 = vec![0.0f32; ids.len() * dim];
        let mut img_slot = 0usize;
        for (i, &t) in ids.iter().enumerate() {
            if img_slot < img_pos.len() && img_pos[img_slot] == i {
                let e = img_emb.unwrap();
                x0[i * dim..i * dim + dim]
                    .copy_from_slice(&e[img_slot * dim..img_slot * dim + dim]);
                img_slot += 1;
            } else {
                let tb = &self.model.tok_embd[t as usize * dim..t as usize * dim + dim];
                x0[i * dim..i * dim + dim].copy_from_slice(tb);
            }
        }
        // MTP self-speculative decode on the resident runner: lossless (byte-identical
        // to greedy) and ~1.6x faster. Active when the model has an MTP head and the
        // resident GPU runner is present; opt-out via HOS_QWEN35_NO_MTP.
        #[cfg(target_os = "macos")]
        if self.model.has_mtp()
            && self.resident.is_some()
            && std::env::var("HOS_QWEN35_NO_MTP").is_err()
        {
            let (logits, hidden) =
                self.model
                    .forward_prefill_mtp_x0(&mut self.state, &x0, &ids, gpu);
            let pos0 = ids.len();
            let rgpu = self.resident.as_ref().unwrap();
            rgpu.upload_state(&self.model, &self.state, pos0);
            let model = &self.model;
            let tok = &self.tok;
            let think_close = self.think_close;
            let ctx = self.ctx;
            let mut reasoning = think.on;
            let mut answer_started = false;
            let mut pending: Vec<u8> = Vec::new();
            let mut buf: Vec<u8> = Vec::new();
            let mut pos = pos0;
            let _t_dec = std::time::Instant::now();
            let n = model.decode_speculative_resident(
                &mut self.state, rgpu, logits, hidden, max_tokens, temp, top_k, top_p,
                rep_penalty, repeat_last_n, seed, &stops, gpu,
                |next| {
                    if pos >= ctx {
                        return true;
                    }
                    if reasoning && Some(next) == think_close {
                        if !pending.is_empty() {
                            on(Chunk::Reasoning(&String::from_utf8_lossy(&pending)));
                            pending.clear();
                        }
                        reasoning = false;
                        pos += 1;
                        return false;
                    }
                    buf.clear();
                    tok.decode_into(next, &mut buf);
                    pending.extend_from_slice(&buf);
                    let valid = match std::str::from_utf8(&pending) {
                        Ok(s) => s.len(),
                        Err(e) => e.valid_up_to(),
                    };
                    if valid > 0 {
                        let mut text = std::str::from_utf8(&pending[..valid]).unwrap();
                        if reasoning {
                            on(Chunk::Reasoning(text));
                        } else {
                            if !answer_started {
                                text = text.trim_start_matches(['\n', ' ', '\t']);
                                if text.is_empty() {
                                    pending.drain(..valid);
                                    pos += 1;
                                    return false;
                                }
                                answer_started = true;
                            }
                            on(Chunk::Answer(text));
                        }
                        pending.drain(..valid);
                    }
                    pos += 1;
                    false
                },
            );
            if !pending.is_empty() {
                let tail = String::from_utf8_lossy(&pending);
                if reasoning {
                    on(Chunk::Reasoning(&tail));
                } else {
                    on(Chunk::Answer(&tail));
                }
            }
            if std::env::var("HOS_QWEN35_TIMING").is_ok() {
                let s = _t_dec.elapsed().as_secs_f64();
                eprintln!("[qwen35-timing] {n} tokens in {s:.2}s = {:.1} tok/s (resident MTP)", n as f64 / s.max(1e-9));
            }
            return n;
        }

        let mut logits = self.model.forward_prefill(&mut self.state, &x0, ids.len(), gpu);
        let mut pos = ids.len();
        // Hand the batched-prefill state to the resident GPU runner for fast decode.
        #[cfg(target_os = "macos")]
        if let Some(r) = &self.resident {
            r.upload_state(&self.model, &self.state, pos);
        }

        let mut rng = if seed != 0 { seed } else { 42 };
        let mut recent: Vec<u32> = Vec::new();
        let mut buf: Vec<u8> = Vec::new();
        let mut pending: Vec<u8> = Vec::new();
        // In thinking mode the prompt opened <think>, so we start inside reasoning
        // until the atomic </think> token; non-thinking is pre-closed => answer.
        let mut reasoning = think.on;
        let mut answer_started = false;
        let mut n = 0usize;
        for _ in 0..max_tokens {
            let from = recent.len().saturating_sub(repeat_last_n);
            let next = crate::sample(
                &logits,
                temp,
                top_k,
                top_p,
                rep_penalty,
                &recent[from..],
                &mut rng,
            );
            recent.push(next);
            if stops.contains(&next) || pos >= self.ctx {
                break;
            }
            // The reasoning/answer boundary is the atomic </think> token.
            if reasoning && Some(next) == self.think_close {
                reasoning = false;
                // flush any held reasoning bytes, then skip the tag itself
                if !pending.is_empty() {
                    on(Chunk::Reasoning(&String::from_utf8_lossy(&pending)));
                    pending.clear();
                }
                logits = self.step(next, pos);
                pos += 1;
                n += 1;
                continue;
            }
            buf.clear();
            self.tok.decode_into(next, &mut buf);
            pending.extend_from_slice(&buf);
            let valid = match std::str::from_utf8(&pending) {
                Ok(s) => s.len(),
                Err(e) => e.valid_up_to(),
            };
            if valid > 0 {
                let mut text = std::str::from_utf8(&pending[..valid]).unwrap();
                if reasoning {
                    on(Chunk::Reasoning(text));
                } else {
                    // Trim the leading whitespace the template puts after </think>.
                    if !answer_started {
                        text = text.trim_start_matches(['\n', ' ', '\t']);
                        if text.is_empty() {
                            pending.drain(..valid);
                            logits = self.step(next, pos);
                            pos += 1;
                            n += 1;
                            continue;
                        }
                        answer_started = true;
                    }
                    on(Chunk::Answer(text));
                }
                pending.drain(..valid);
            }
            logits = self.step(next, pos);
            pos += 1;
            n += 1;
        }
        if !pending.is_empty() {
            let tail = String::from_utf8_lossy(&pending);
            if reasoning {
                on(Chunk::Reasoning(&tail));
            } else {
                on(Chunk::Answer(&tail));
            }
        }
        n
    }
}
