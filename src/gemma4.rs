//! Gemma 4 12B "unified" — native TEXT decoder inference (self-contained arch).
//!
//! A NEW, additive HOS arch path (does not touch the generic Llama `forward.rs`
//! nor the existing Gemma `(1+w)` norm). Mirrors the `qwen35.rs` pattern: its own
//! `Cfg`, its own weight loading straight from the HF safetensors dir, and its own
//! CPU forward pass that reuses HOS's parallel `Weight`/matvec for the big linears.
//!
//! Implements the verified forward spec in `models/gemma4/SPEC.md`:
//!   * heterogeneous layers: SLIDING (256-dim GQA, local RoPE θ=1e4, window 1024)
//!     vs GLOBAL every 6th layer (512-dim MQA, p-RoPE θ=1e6 / 0.25, k_eq_v, full causal)
//!   * dual RoPE (NEOX rotate-half), q_norm/k_norm (raw-weight RMSNorm over head_dim,
//!     BEFORE rope), scale-free v_norm (no rope), attention scaling = 1.0
//!   * RAW-weight RMSNorm (NO +1), sandwich norms, per-layer `layer_scalar` output scale
//!   * embed * bf16(sqrt(3840)), gelu-tanh SwiGLU, tied lm-head, final softcap 30.
//!
//! Text weights live under `model.language_model.*`; the 11 vision/audio tensors
//! are skipped. bf16 on disk is loaded to f32.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;

use crate::error::Result;
use crate::gguf::{GGML_Q4_K, GGML_Q5_K, GGML_Q6_K};
use crate::metal_be::Gpu;
use crate::model::Weight;
use crate::safetensors::SafeTensors;

// ============================================================================
// Q4_K / Q5_K / Q6_K quantized load (ADDITIVE — the f32 path is untouched).
//
// When `HOS_GEMMA4_QUANT=q4k|q5k|q6k` is set (wired to `--q4k`/`--q5k`/`--q6k`),
// every BIG linear (per-layer wq/wk/wv/wo, gate/up/down, and the tied
// embed/lm-head) is stored quantized-resident as `Weight::Quant`; the per-row
// dequant is fused into `matvec`. Norms/q_norm/k_norm/layer_scalar stay f32, and
// the embedding LOOKUP keeps a full-precision f32 copy (`embed_cpu`). All Gemma4
// linears have `cols` that are a clean multiple of QK_K=256, so every big tensor
// K-quantizes; a non-256 tensor would fall back to f32 (see `make_big`).
//
// Result: ~12B params at ~0.56 B/param ≈ 6.8 GB (+4 GB f32 embed for lookup)
// instead of ~48 GB f32. The quantized bytes are cached to
// `<model_dir>/gemma4-<tag>.hosw` so subsequent launches skip both the bf16
// read of the big tensors AND the requantize (disable with HOS_GEMMA4_NOCACHE=1).
// ============================================================================

/// Selected quant type from the environment (None = the default f32 path).
fn quant_type_from_env() -> Option<u32> {
    match std::env::var("HOS_GEMMA4_QUANT").ok().as_deref() {
        Some("q4k") | Some("q4_k") | Some("Q4_K") => Some(GGML_Q4_K),
        Some("q5k") | Some("q5_k") | Some("Q5_K") => Some(GGML_Q5_K),
        Some("q6k") | Some("q6_k") | Some("Q6_K") => Some(GGML_Q6_K),
        Some("hq4") | Some("HQ4") => Some(crate::model::HQ4_TYPE),
        Some("e8") | Some("E8") => Some(crate::model::E8_TYPE),
        _ => None,
    }
}

fn quant_tag(ty: u32) -> &'static str {
    match ty {
        GGML_Q4_K => "q4k",
        GGML_Q5_K => "q5k",
        GGML_Q6_K => "q6k",
        crate::gguf::GGML_Q8_0 => "q8",
        crate::model::HQ4_TYPE => "hq4",
        crate::model::E8_TYPE => "e8",
        _ => "q",
    }
}

/// Quant type for the tied embed/lm-head. This matvec maps STRAIGHT to logits, so
/// 4-bit noise here flips borderline greedy tokens (e.g. P1 " Paris" vs " a").
/// Default to near-lossless Q8_0 regardless of the layer quant; override with
/// `HOS_GEMMA4_HEAD=q4k|q5k|q6k|q8`.
fn head_quant_type() -> u32 {
    match std::env::var("HOS_GEMMA4_HEAD").ok().as_deref() {
        Some("q4k") | Some("q4_k") => GGML_Q4_K,
        Some("q5k") | Some("q5_k") => GGML_Q5_K,
        Some("q6k") | Some("q6_k") => GGML_Q6_K,
        _ => crate::gguf::GGML_Q8_0,
    }
}

/// Parallel per-row quantize of a [rows, cols] f32 matrix. Each row (cols is a
/// multiple of the block size) is an independent set of K-quant super-blocks, so
/// rows quantize in parallel and concatenate in order — bit-identical to a serial
/// `gguf_write::quantize` over the whole tensor.
fn quantize_par(data: &[f32], cols: usize, ty: u32) -> Vec<u8> {
    use rayon::prelude::*;
    let rows = data.len() / cols;
    let parts: Vec<Vec<u8>> = (0..rows)
        .into_par_iter()
        .map(|r| {
            let row = &data[r * cols..(r + 1) * cols];
            // hq4 is HOS-native (not a ggml block type); gguf_write::quantize has
            // no arm for it and would silently emit q8_0, so encode it directly.
            if ty == crate::model::HQ4_TYPE {
                crate::hos_quant::encode_hq4(row)
            } else if ty == crate::model::E8_TYPE {
                crate::hos_quant::encode_e8(row)
            } else {
                crate::gguf_write::quantize(row, ty)
            }
        })
        .collect();
    parts.concat()
}

/// Quantize [rows, cols] f32 -> `Weight::Quant` (does NOT keep the f32).
fn quantize_weight(data: &[f32], cols: usize, ty: u32) -> Weight {
    let rows = data.len() / cols;
    Weight::Quant {
        bytes: quantize_par(data, cols, ty),
        ggml_type: ty,
        rows,
        cols,
    }
}

/// Build a big linear: quantize (consuming the f32) when a block-aligned quant
/// type is requested, else keep it as a plain f32 `Weight::Cpu`.
fn make_big(data: Vec<f32>, cols: usize, ty: Option<u32>) -> Weight {
    // hq4 uses its own 64-element block; every other type is a ggml block type.
    let block = |t: u32| {
        if t == crate::model::HQ4_TYPE {
            crate::hos_quant::hq4_block()
        } else if t == crate::model::E8_TYPE {
            crate::hos_quant::e8_block()
        } else {
            crate::gguf_write::block_size(t)
        }
    };
    match ty {
        Some(t) if cols % block(t) == 0 => quantize_weight(&data, cols, t),
        _ => Weight::cpu(data, cols),
    }
}

/// Load a big linear either from the quant cache (by `key`) or by reading+quantizing
/// the safetensors tensor `st_name`.
fn big_weight(
    st: &SafeTensors,
    cache: &mut Option<HashMap<String, Weight>>,
    st_name: &str,
    key: &str,
    cols: usize,
    ty: Option<u32>,
) -> Result<Weight> {
    if let Some(map) = cache.as_mut() {
        if let Some(w) = map.remove(key) {
            return Ok(w);
        }
    }
    Ok(make_big(st.to_f32(st_name)?, cols, ty))
}

const QCACHE_MAGIC: &[u8; 8] = b"HOSWQK01";

/// Read a `.hosw` quant cache into a name -> `Weight::Quant` map. Returns None on
/// any mismatch (wrong magic / different quant type / truncation) so the caller
/// silently rebuilds.
fn read_qcache(path: &Path, want_ty: u32) -> Option<HashMap<String, Weight>> {
    let f = std::fs::File::open(path).ok()?;
    let mut r = std::io::BufReader::new(f);
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic).ok()?;
    if &magic != QCACHE_MAGIC {
        return None;
    }
    let ru32 = |r: &mut std::io::BufReader<std::fs::File>| -> Option<u32> {
        let mut b = [0u8; 4];
        r.read_exact(&mut b).ok()?;
        Some(u32::from_le_bytes(b))
    };
    let ru64 = |r: &mut std::io::BufReader<std::fs::File>| -> Option<u64> {
        let mut b = [0u8; 8];
        r.read_exact(&mut b).ok()?;
        Some(u64::from_le_bytes(b))
    };
    if ru32(&mut r)? != want_ty {
        return None;
    }
    let count = ru32(&mut r)?;
    let mut map = HashMap::new();
    for _ in 0..count {
        let nl = ru32(&mut r)? as usize;
        let mut nb = vec![0u8; nl];
        r.read_exact(&mut nb).ok()?;
        let name = String::from_utf8(nb).ok()?;
        let ty = ru32(&mut r)?;
        let rows = ru64(&mut r)? as usize;
        let cols = ru64(&mut r)? as usize;
        let blen = ru64(&mut r)? as usize;
        let mut bytes = vec![0u8; blen];
        r.read_exact(&mut bytes).ok()?;
        map.insert(
            name,
            Weight::Quant {
                bytes,
                ggml_type: ty,
                rows,
                cols,
            },
        );
    }
    Some(map)
}

/// Append one quant weight record to the cache writer.
fn write_qentry<W: Write>(w: &mut W, name: &str, weight: &Weight) -> std::io::Result<()> {
    if let Weight::Quant {
        bytes,
        ggml_type,
        rows,
        cols,
    } = weight
    {
        w.write_all(&(name.len() as u32).to_le_bytes())?;
        w.write_all(name.as_bytes())?;
        w.write_all(&ggml_type.to_le_bytes())?;
        w.write_all(&(*rows as u64).to_le_bytes())?;
        w.write_all(&(*cols as u64).to_le_bytes())?;
        w.write_all(&(bytes.len() as u64).to_le_bytes())?;
        w.write_all(bytes)?;
    }
    Ok(())
}

// ---- fixed Gemma-4-12B text dims (SPEC.md / config.json text_config) ----
pub const HIDDEN: usize = 3840;
pub const N_LAYERS: usize = 48;
pub const N_HEADS: usize = 16;
pub const N_KV_SLIDING: usize = 8; // GQA
pub const N_KV_GLOBAL: usize = 1; // MQA
pub const HEAD_DIM_SLIDING: usize = 256;
pub const HEAD_DIM_GLOBAL: usize = 512;
pub const INTER: usize = 15360;
pub const VOCAB: usize = 262144;
pub const RMS_EPS: f32 = 1e-6;
pub const ROPE_THETA_LOCAL: f64 = 10_000.0;
pub const ROPE_THETA_GLOBAL: f64 = 1_000_000.0;
pub const SLIDING_WINDOW: usize = 1024;
pub const FINAL_SOFTCAP: f32 = 30.0;

/// Minimum prompt length for the batched GPU prefill to engage (below this the
/// per-token sequential path's lower fixed cost wins). Opt-in on `--gpu` + q4_k.
pub const PREFILL_BATCH_THRESHOLD: usize = 16;

#[derive(Debug, Clone)]
pub struct Cfg {
    pub hidden: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub inter: usize,
    pub vocab: usize,
    pub rms_eps: f32,
    pub sliding_window: usize,
    pub final_softcap: f32,
}

impl Default for Cfg {
    fn default() -> Cfg {
        Cfg {
            hidden: HIDDEN,
            n_layers: N_LAYERS,
            n_heads: N_HEADS,
            inter: INTER,
            vocab: VOCAB,
            rms_eps: RMS_EPS,
            sliding_window: SLIDING_WINDOW,
            final_softcap: FINAL_SOFTCAP,
        }
    }
}

/// Layer l is GLOBAL/full_attention iff (l+1) % 6 == 0 (l ∈ {5,11,17,23,29,35,41,47}).
#[inline]
pub fn is_global(l: usize) -> bool {
    (l + 1) % 6 == 0
}

struct LayerW {
    global: bool,
    // sandwich norms (raw weight, over hidden)
    input_ln: Vec<f32>,
    post_attn_ln: Vec<f32>,
    pre_ff_ln: Vec<f32>,
    post_ff_ln: Vec<f32>,
    // attention
    q_norm: Vec<f32>, // over head_dim
    k_norm: Vec<f32>, // over head_dim
    wq: Weight,
    wk: Weight,
    wv: Option<Weight>, // absent on global layers (k_eq_v)
    wo: Weight,
    // mlp (SwiGLU, gelu-tanh)
    gate: Weight,
    up: Weight,
    down: Weight,
    layer_scalar: f32,
}

impl LayerW {
    fn head_dim(&self) -> usize {
        if self.global {
            HEAD_DIM_GLOBAL
        } else {
            HEAD_DIM_SLIDING
        }
    }
    fn n_kv(&self) -> usize {
        if self.global {
            N_KV_GLOBAL
        } else {
            N_KV_SLIDING
        }
    }
}

pub struct Gemma4 {
    pub cfg: Cfg,
    embed: Weight, // [vocab, hidden] — used for lookup AND tied lm-head matvec
    /// On the GPU path `embed` becomes a `Weight::Gpu` (used for the tied lm-head
    /// matvec), so the f32 rows needed for the embedding LOOKUP are kept here.
    /// `None` on the CPU path (lookup reads `embed` directly).
    embed_cpu: Option<Vec<f32>>,
    final_norm: Vec<f32>,
    layers: Vec<LayerW>,
    embed_scale: f32,
    inv_freq_local: Vec<f32>,  // len HEAD_DIM_SLIDING/2 = 128
    inv_freq_global: Vec<f32>, // len HEAD_DIM_GLOBAL/2 = 256 (first 64 nonzero)
    /// Emulate the oracle's bf16 compute (round activations to bf16 at each op
    /// boundary). The reference ran the whole decoder in bf16; matching its
    /// argmax on borderline prompts (e.g. P1 " Paris" vs " a") requires this.
    /// Toggle off with HOS_GEMMA4_F32=1 for the pure-f32 path.
    bf16_emul: bool,
}

/// Snapshot of the last-position hidden state at each validation stage.
pub struct Fwd {
    pub h_embed: Vec<f32>,
    pub h0: Vec<f32>,
    pub h24: Vec<f32>,
    pub logits: Vec<f32>,
}

// ---------------- primitives ----------------

/// Round an f32 to bfloat16 precision (round-to-nearest-even), back to f32.
fn bf16(x: f32) -> f32 {
    let b = x.to_bits();
    let lsb = (b >> 16) & 1;
    let rounded = (b + 0x7fff + lsb) >> 16;
    f32::from_bits(rounded << 16)
}

/// Gemma4 RMSNorm — RAW weight (NO +1), fp32. out = x * rms(x)^-1 * weight.
fn rmsnorm_raw(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let ms = x.iter().map(|v| v * v).sum::<f32>() / n as f32 + eps;
    let s = ms.powf(-0.5);
    (0..n).map(|i| x[i] * s * w[i]).collect()
}

/// Scale-free RMSNorm (v_norm) — just normalize, no weight.
fn rmsnorm_noscale(x: &mut [f32], eps: f32) {
    let n = x.len();
    let ms = x.iter().map(|v| v * v).sum::<f32>() / n as f32 + eps;
    let s = ms.powf(-0.5);
    for v in x.iter_mut() {
        *v *= s;
    }
}

/// In-place raw-weight RMSNorm over a single head slice.
fn rmsnorm_raw_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    let n = x.len();
    let ms = x.iter().map(|v| v * v).sum::<f32>() / n as f32 + eps;
    let s = ms.powf(-0.5);
    for i in 0..n {
        x[i] = x[i] * s * w[i];
    }
}

/// NEOX rotate-half RoPE on a single head vector (len d), pairing (j, j+d/2).
/// When `bf`, cos/sin are computed in fp32 then rounded to bf16 (matching torch's
/// `cos.to(x.dtype)`), and each rotated output is rounded to bf16.
fn rope_head(v: &mut [f32], inv_freq: &[f32], pos: usize, bf: bool) {
    let half = inv_freq.len();
    for j in 0..half {
        let ang = pos as f32 * inv_freq[j];
        let (mut s, mut c) = ang.sin_cos();
        if bf {
            c = bf16(c);
            s = bf16(s);
        }
        let a = v[j];
        let b = v[j + half];
        if bf {
            // torch computes (x*cos)+(rotate_half(x)*sin) fully in bf16:
            // each product is rounded to bf16, then the sum is rounded.
            v[j] = bf16(bf16(a * c) + bf16(-b * s));
            v[j + half] = bf16(bf16(b * c) + bf16(a * s));
        } else {
            v[j] = a * c - b * s;
            v[j + half] = b * c + a * s;
        }
    }
}

/// Round a slice to bf16 in place (no-op when `bf` is false).
#[inline]
fn rvec(v: &mut [f32], bf: bool) {
    if bf {
        for x in v.iter_mut() {
            *x = bf16(*x);
        }
    }
}

#[inline]
fn gelu_tanh(x: f32) -> f32 {
    const K: f32 = 0.797_884_560_802_865_4; // sqrt(2/pi)
    0.5 * x * (1.0 + (K * (x + 0.044715 * x * x * x)).tanh())
}

fn matvec(w: &Weight, x: &[f32], out_dim: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; out_dim];
    w.matvec(None, x, &mut y);
    y
}

/// GPU-aware matvec: dispatches to Metal when `w` is a `Weight::Gpu` (and `gpu`
/// is `Some`), else runs the CPU rayon matvec. Used by the KV-cached decode path.
fn matvec_g(w: &Weight, x: &[f32], out_dim: usize, gpu: Option<&Gpu>) -> Vec<f32> {
    let mut y = vec![0.0f32; out_dim];
    w.matvec(gpu, x, &mut y);
    y
}

impl Gemma4 {
    pub fn load(dir: &Path) -> Result<Gemma4> {
        let st = SafeTensors::open_dir(dir)?;
        let cfg = Cfg::default();
        let p = "model.language_model.";

        // Optional K-quant path (additive; f32 stays the default). When enabled we
        // reuse a `<dir>/gemma4-<tag>.hosw` cache of the quantized big linears so a
        // relaunch skips both the bf16 read and the requantize.
        let quant_ty = quant_type_from_env();
        let use_cache = quant_ty.is_some()
            && !std::env::var("HOS_GEMMA4_NOCACHE").is_ok_and(|v| v != "0" && !v.is_empty());
        let cache_path = quant_ty.map(|t| dir.join(format!("gemma4-{}.hosw", quant_tag(t))));
        let mut cache: Option<HashMap<String, Weight>> = None;
        if use_cache {
            if let Some(cp) = &cache_path {
                if cp.exists() {
                    let tq = std::time::Instant::now();
                    cache = read_qcache(cp, quant_ty.unwrap());
                    if cache.is_some() {
                        eprintln!(
                            "[gemma4] loaded quant cache {} ({:.1}s)",
                            cp.display(),
                            tq.elapsed().as_secs_f32()
                        );
                    }
                }
            }
        }
        let had_cache = cache.is_some();

        match quant_ty {
            Some(t) => eprintln!(
                "[gemma4] loading text weights as {} (from {}){}",
                quant_tag(t).to_uppercase(),
                dir.display(),
                if had_cache {
                    " [cache]"
                } else {
                    " [quantizing bf16->f32->kquant]"
                }
            ),
            None => eprintln!(
                "[gemma4] loading text weights (bf16->f32) from {}",
                dir.display()
            ),
        }

        // Embedding: keep the full-precision f32 rows for the LOOKUP; on the quant
        // path ALSO hold a quantized copy for the tied lm-head matvec.
        let embed_f32 = st.to_f32(&format!("{p}embed_tokens.weight"))?;
        let (embed, embed_cpu) = if quant_ty.is_some() {
            let head_ty = head_quant_type();
            let w = cache
                .as_mut()
                .and_then(|m| m.remove("embed"))
                .unwrap_or_else(|| quantize_weight(&embed_f32, cfg.hidden, head_ty));
            (w, Some(embed_f32))
        } else {
            (Weight::cpu(embed_f32, cfg.hidden), None)
        };
        let final_norm = st.to_f32(&format!("{p}norm.weight"))?;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            let g = is_global(l);
            let lp = format!("{p}layers.{l}.");
            let hd = if g { HEAD_DIM_GLOBAL } else { HEAD_DIM_SLIDING };
            let n_kv = if g { N_KV_GLOBAL } else { N_KV_SLIDING };
            let o_in = cfg.n_heads * hd; // o_proj in-dim (4096 sliding / 8192 global)

            let wq = big_weight(
                &st,
                &mut cache,
                &format!("{lp}self_attn.q_proj.weight"),
                &format!("l{l}.wq"),
                cfg.hidden,
                quant_ty,
            )?;
            let wk = big_weight(
                &st,
                &mut cache,
                &format!("{lp}self_attn.k_proj.weight"),
                &format!("l{l}.wk"),
                cfg.hidden,
                quant_ty,
            )?;
            let wv = if g {
                None
            } else {
                Some(big_weight(
                    &st,
                    &mut cache,
                    &format!("{lp}self_attn.v_proj.weight"),
                    &format!("l{l}.wv"),
                    cfg.hidden,
                    quant_ty,
                )?)
            };
            let wo = big_weight(
                &st,
                &mut cache,
                &format!("{lp}self_attn.o_proj.weight"),
                &format!("l{l}.wo"),
                o_in,
                quant_ty,
            )?;
            let _ = n_kv;

            layers.push(LayerW {
                global: g,
                input_ln: st.to_f32(&format!("{lp}input_layernorm.weight"))?,
                post_attn_ln: st.to_f32(&format!("{lp}post_attention_layernorm.weight"))?,
                pre_ff_ln: st.to_f32(&format!("{lp}pre_feedforward_layernorm.weight"))?,
                post_ff_ln: st.to_f32(&format!("{lp}post_feedforward_layernorm.weight"))?,
                q_norm: st.to_f32(&format!("{lp}self_attn.q_norm.weight"))?,
                k_norm: st.to_f32(&format!("{lp}self_attn.k_norm.weight"))?,
                wq,
                wk,
                wv,
                wo,
                gate: big_weight(
                    &st,
                    &mut cache,
                    &format!("{lp}mlp.gate_proj.weight"),
                    &format!("l{l}.gate"),
                    cfg.hidden,
                    quant_ty,
                )?,
                up: big_weight(
                    &st,
                    &mut cache,
                    &format!("{lp}mlp.up_proj.weight"),
                    &format!("l{l}.up"),
                    cfg.hidden,
                    quant_ty,
                )?,
                down: big_weight(
                    &st,
                    &mut cache,
                    &format!("{lp}mlp.down_proj.weight"),
                    &format!("l{l}.down"),
                    cfg.inter,
                    quant_ty,
                )?,
                layer_scalar: st.to_f32(&format!("{lp}layer_scalar"))?[0],
            });
        }

        // RoPE inv_freq tables.
        // local: full 256-dim rotation, θ=1e4 → 128 pairs.
        let inv_freq_local: Vec<f32> = (0..HEAD_DIM_SLIDING / 2)
            .map(|j| (ROPE_THETA_LOCAL.powf(-2.0 * j as f64 / HEAD_DIM_SLIDING as f64)) as f32)
            .collect();
        // global p-RoPE: head_dim 512 → 256 pairs, but partial_rotary_factor 0.25 ⇒ only
        // the first 64 pairs rotate (inv_freq[j]=1e6^(-2j/512)); the remaining 192 pairs
        // have inv_freq=0 (NoPE: cos=1,sin=0 → identity).
        let inv_freq_global: Vec<f32> = (0..HEAD_DIM_GLOBAL / 2)
            .map(|j| {
                if j < 64 {
                    (ROPE_THETA_GLOBAL.powf(-2.0 * j as f64 / HEAD_DIM_GLOBAL as f64)) as f32
                } else {
                    0.0
                }
            })
            .collect();

        let embed_scale = bf16((cfg.hidden as f32).sqrt());
        let bf16_emul = !std::env::var("HOS_GEMMA4_F32").is_ok_and(|v| v != "0" && !v.is_empty());
        eprintln!(
            "[gemma4] loaded {} layers, vocab {}, embed_scale(bf16 sqrt {})={}, bf16_emul={}",
            cfg.n_layers, cfg.vocab, cfg.hidden, embed_scale, bf16_emul
        );

        let m = Gemma4 {
            cfg,
            embed,
            embed_cpu,
            final_norm,
            layers,
            embed_scale,
            inv_freq_local,
            inv_freq_global,
            bf16_emul,
        };

        // First-time quant build: persist the quantized big linears so relaunches
        // skip the bf16 read + requantize. Best-effort (a write failure just means
        // the next run requantizes).
        if quant_ty.is_some() && use_cache && !had_cache {
            if let Some(cp) = &cache_path {
                let tw = std::time::Instant::now();
                match m.write_qcache(cp, quant_ty.unwrap()) {
                    Ok(()) => eprintln!(
                        "[gemma4] wrote quant cache {} ({:.1}s)",
                        cp.display(),
                        tw.elapsed().as_secs_f32()
                    ),
                    Err(e) => eprintln!("[gemma4] quant cache write skipped: {e}"),
                }
            }
        }
        Ok(m)
    }

    /// Serialize every quantized big linear (embed + per-layer wq/wk/wv/wo,
    /// gate/up/down) to a `.hosw` cache. Norms stay in the safetensors and are
    /// re-read cheaply on cache load.
    fn write_qcache(&self, path: &Path, ty: u32) -> std::io::Result<()> {
        let tmp = path.with_extension("hosw.tmp");
        let mut entries: Vec<(String, &Weight)> = Vec::new();
        entries.push(("embed".to_string(), &self.embed));
        for (l, lw) in self.layers.iter().enumerate() {
            entries.push((format!("l{l}.wq"), &lw.wq));
            entries.push((format!("l{l}.wk"), &lw.wk));
            if let Some(wv) = &lw.wv {
                entries.push((format!("l{l}.wv"), wv));
            }
            entries.push((format!("l{l}.wo"), &lw.wo));
            entries.push((format!("l{l}.gate"), &lw.gate));
            entries.push((format!("l{l}.up"), &lw.up));
            entries.push((format!("l{l}.down"), &lw.down));
        }
        // Only truly quantized weights are cached; if the caller ran f32 nothing writes.
        let quant_entries: Vec<_> = entries
            .iter()
            .filter(|(_, w)| matches!(w, Weight::Quant { .. }))
            .collect();
        {
            let f = std::fs::File::create(&tmp)?;
            let mut w = std::io::BufWriter::new(f);
            w.write_all(QCACHE_MAGIC)?;
            w.write_all(&ty.to_le_bytes())?;
            w.write_all(&(quant_entries.len() as u32).to_le_bytes())?;
            for (name, weight) in &quant_entries {
                write_qentry(&mut w, name, weight)?;
            }
            w.flush()?;
        }
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Sum of resident weight bytes (quantized bytes / f32 data / GPU buffers),
    /// including the f32 embedding-lookup copy and all norm vectors — the memory
    /// figure the bench reports.
    pub fn resident_bytes(&self) -> usize {
        let mut n = self.embed.nbytes();
        if let Some(e) = &self.embed_cpu {
            n += e.len() * 4;
        }
        n += self.final_norm.len() * 4;
        for lw in &self.layers {
            n += lw.wq.nbytes() + lw.wk.nbytes() + lw.wo.nbytes();
            n += lw.gate.nbytes() + lw.up.nbytes() + lw.down.nbytes();
            if let Some(wv) = &lw.wv {
                n += wv.nbytes();
            }
            n += (lw.input_ln.len()
                + lw.post_attn_ln.len()
                + lw.pre_ff_ln.len()
                + lw.post_ff_ln.len()
                + lw.q_norm.len()
                + lw.k_norm.len())
                * 4;
        }
        n
    }

    fn embed_row(&self, id: u32) -> &[f32] {
        let h = self.cfg.hidden;
        let src = match &self.embed_cpu {
            Some(e) => e.as_slice(),
            None => self.embed.cpu_data(),
        };
        &src[id as usize * h..(id as usize + 1) * h]
    }

    /// One decoder layer over the whole sequence. `h` is [seq][hidden], mutated in place.
    fn layer(&self, lw: &LayerW, h: &mut [Vec<f32>]) {
        let seq = h.len();
        let hidden = self.cfg.hidden;
        let eps = self.cfg.rms_eps;
        let hd = lw.head_dim();
        let n_kv = lw.n_kv();
        let nh = self.cfg.n_heads;
        let n_rep = nh / n_kv;
        let inv_freq: &[f32] = if lw.global {
            &self.inv_freq_global
        } else {
            &self.inv_freq_local
        };
        let bf = self.bf16_emul;

        // ---- projections + per-head norms + rope, for every token ----
        // q: seq*nh*hd ; k,v: seq*n_kv*hd
        let mut q = vec![0.0f32; seq * nh * hd];
        let mut k = vec![0.0f32; seq * n_kv * hd];
        let mut v = vec![0.0f32; seq * n_kv * hd];
        for t in 0..seq {
            let mut a = rmsnorm_raw(&h[t], &lw.input_ln, eps);
            rvec(&mut a, bf); // input_layernorm output -> bf16
                              // q
            let mut qt = matvec(&lw.wq, &a, nh * hd);
            rvec(&mut qt, bf); // q_proj -> bf16
            let qslot = &mut q[t * nh * hd..(t + 1) * nh * hd];
            qslot.copy_from_slice(&qt);
            for hh in 0..nh {
                let head = &mut qslot[hh * hd..(hh + 1) * hd];
                rmsnorm_raw_inplace(head, &lw.q_norm, eps); // q_norm BEFORE rope
                rvec(head, bf);
                rope_head(head, inv_freq, t, bf);
            }
            // k source
            let mut k_src = matvec(&lw.wk, &a, n_kv * hd);
            rvec(&mut k_src, bf); // k_proj -> bf16
                                  // v source: global => v = k_src (k_eq_v); sliding => v_proj(a)
            let mut v_src = if lw.global {
                k_src.clone()
            } else {
                matvec(lw.wv.as_ref().unwrap(), &a, n_kv * hd)
            };
            rvec(&mut v_src, bf); // v_proj (or k_eq_v) -> bf16
            let kslot = &mut k[t * n_kv * hd..(t + 1) * n_kv * hd];
            kslot.copy_from_slice(&k_src);
            for hh in 0..n_kv {
                let head = &mut kslot[hh * hd..(hh + 1) * hd];
                rmsnorm_raw_inplace(head, &lw.k_norm, eps); // k_norm BEFORE rope
                rvec(head, bf);
                rope_head(head, inv_freq, t, bf);
            }
            let vslot = &mut v[t * n_kv * hd..(t + 1) * n_kv * hd];
            vslot.copy_from_slice(&v_src);
            for hh in 0..n_kv {
                let head = &mut vslot[hh * hd..(hh + 1) * hd];
                rmsnorm_noscale(head, eps); // v_norm: scale-free, NO rope
                rvec(head, bf);
            }
        }

        // ---- attention (scaling = 1.0), then o_proj, then post_attention_layernorm + residual ----
        let o_in = nh * hd;
        for t in 0..seq {
            // lowest attendable key index (sliding window; global => 0)
            let lo = if lw.global {
                0
            } else {
                (t + 1).saturating_sub(self.cfg.sliding_window)
            };
            let mut merged = vec![0.0f32; o_in];
            for hh in 0..nh {
                let kvh = hh / n_rep;
                let qh = &q[t * nh * hd + hh * hd..t * nh * hd + hh * hd + hd];
                // scores over allowed keys j in [lo, t]
                let mut scores = vec![0.0f32; t - lo + 1];
                for (idx, j) in (lo..=t).enumerate() {
                    let kh = &k[j * n_kv * hd + kvh * hd..j * n_kv * hd + kvh * hd + hd];
                    let mut s = 0.0f32;
                    for d in 0..hd {
                        s += qh[d] * kh[d];
                    }
                    scores[idx] = s; // scaling = 1.0
                }
                // softmax (fp32) then cast probs to bf16 (eager: attn_weights.to(dtype)),
                // then the value-matmul; attention output rounded to bf16 below.
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                for s in scores.iter_mut() {
                    *s /= sum;
                    if bf {
                        *s = bf16(*s);
                    }
                }
                let out = &mut merged[hh * hd..(hh + 1) * hd];
                for (idx, j) in (lo..=t).enumerate() {
                    let vh = &v[j * n_kv * hd + kvh * hd..j * n_kv * hd + kvh * hd + hd];
                    let w = scores[idx];
                    for d in 0..hd {
                        out[d] += w * vh[d];
                    }
                }
            }
            rvec(&mut merged, bf); // attn_output -> bf16 (eager output dtype)
            let mut attn = matvec(&lw.wo, &merged, hidden);
            rvec(&mut attn, bf); // o_proj -> bf16
            let mut post = rmsnorm_raw(&attn, &lw.post_attn_ln, eps);
            rvec(&mut post, bf);
            for i in 0..hidden {
                h[t][i] += post[i];
            }
            rvec(&mut h[t], bf); // residual add -> bf16
        }

        // ---- MLP block: r + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(x))) ----
        for t in 0..seq {
            let mut pf = rmsnorm_raw(&h[t], &lw.pre_ff_ln, eps);
            rvec(&mut pf, bf);
            let mut g = matvec(&lw.gate, &pf, self.cfg.inter);
            rvec(&mut g, bf); // gate_proj -> bf16
            let mut u = matvec(&lw.up, &pf, self.cfg.inter);
            rvec(&mut u, bf); // up_proj -> bf16
            for i in 0..self.cfg.inter {
                let gi = if bf {
                    bf16(gelu_tanh(g[i]))
                } else {
                    gelu_tanh(g[i])
                };
                g[i] = gi * u[i];
                if bf {
                    g[i] = bf16(g[i]);
                }
            }
            let mut m = matvec(&lw.down, &g, hidden);
            rvec(&mut m, bf); // down_proj -> bf16
            let mut post = rmsnorm_raw(&m, &lw.post_ff_ln, eps);
            rvec(&mut post, bf);
            for i in 0..hidden {
                h[t][i] += post[i];
            }
            rvec(&mut h[t], bf); // residual add -> bf16
                                 // ---- layer_scalar scales the WHOLE residual stream ----
            let ls = if bf {
                bf16(lw.layer_scalar)
            } else {
                lw.layer_scalar
            };
            for i in 0..hidden {
                h[t][i] *= ls;
            }
            rvec(&mut h[t], bf); // *= layer_scalar -> bf16
        }
    }

    /// Full forward on `ids` (BOS already included). Captures last-position
    /// snapshots after embed, layer 0, layer 24, and the final post-softcap logits.
    pub fn forward(&self, ids: &[u32]) -> Fwd {
        let seq = ids.len();
        let hidden = self.cfg.hidden;
        let last = seq - 1;

        // embed * bf16(sqrt(hidden)); the product is stored bf16 in the oracle.
        let bf = self.bf16_emul;
        let mut h: Vec<Vec<f32>> = ids
            .iter()
            .map(|&id| {
                self.embed_row(id)
                    .iter()
                    .map(|&x| {
                        let p = x * self.embed_scale;
                        if bf {
                            bf16(p)
                        } else {
                            p
                        }
                    })
                    .collect()
            })
            .collect();

        let h_embed = h[last].clone();
        let mut h0 = Vec::new();
        let mut h24 = Vec::new();

        // Optional per-layer last-position dump for bisection debugging.
        let dump = std::env::var("HOS_GEMMA4_DUMP").ok();
        let mut perlayer: Vec<f32> = Vec::new();
        if dump.is_some() {
            perlayer.extend_from_slice(&h[last]); // index 0 = after embed
        }

        for (l, lw) in self.layers.iter().enumerate() {
            self.layer(lw, &mut h);
            if dump.is_some() {
                perlayer.extend_from_slice(&h[last]);
            }
            if l == 0 {
                h0 = h[last].clone();
            }
            if l == 24 {
                h24 = h[last].clone();
            }
        }
        if let Some(path) = dump {
            // raw little-endian f32, shape [n_layers+1, hidden], row-major.
            let mut bytes = Vec::with_capacity(perlayer.len() * 4);
            for &x in &perlayer {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
            std::fs::write(&path, &bytes).ok();
        }

        // final norm (raw weight) on last position, then tied lm-head + softcap
        let mut hn = rmsnorm_raw(&h[last], &self.final_norm, self.cfg.rms_eps);
        rvec(&mut hn, bf);
        let mut logits = matvec(&self.embed, &hn, self.cfg.vocab);
        rvec(&mut logits, bf); // lm_head -> bf16
        let cap = self.cfg.final_softcap;
        for v in logits.iter_mut() {
            *v = cap * (*v / cap).tanh();
            if bf {
                *v = bf16(*v);
            }
        }

        let _ = hidden;
        Fwd {
            h_embed,
            h0,
            h24,
            logits,
        }
    }

    /// Last-position post-softcap logits only (for greedy generation).
    pub fn forward_logits(&self, ids: &[u32]) -> Vec<f32> {
        self.forward(ids).logits
    }

    /// Greedy-generate `n` next token ids (argmax), appending as we go.
    pub fn generate(&self, ids: &[u32], n: usize) -> Vec<u32> {
        let mut seq = ids.to_vec();
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let logits = self.forward_logits(&seq);
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &l) in logits.iter().enumerate() {
                if l > bv {
                    bv = l;
                    best = i;
                }
            }
            out.push(best as u32);
            seq.push(best as u32);
        }
        out
    }

    // ======================================================================
    // KV-cached incremental decoding (Part 1). Bit-identical to `forward` on
    // the CPU bf16 path: reuses the exact same per-op math + bf16 rounding as
    // `layer`, but processes ONE token at a time against a growing per-layer
    // KV cache instead of recomputing the whole prefix every step (O(n²)→O(n)).
    // ======================================================================

    /// Upload the big linear weights (wq/wk/wv/wo, gate/up/down for every layer,
    /// and the tied `embed`/lm-head) to the GPU as f16, replacing the CPU f32
    /// copies (frees their RAM). Disables bf16 emulation — the GPU does f16/f32
    /// math, so per-op bf16 rounding (a CPU-only oracle-matching device) is off.
    /// Norms/rope/attention/softmax stay on the CPU. ≈24 GB resident (12B×f16).
    pub fn upload_to_gpu(&mut self, gpu: &Gpu) {
        // f32 weights upload as f16; quantized weights upload in their native
        // K-quant bytes (the coalesced Metal K-quant kernels dequant in-kernel).
        // The native HOS quants (hq4, e8) have no Metal matvec kernel, so they stay
        // resident on the CPU as `Weight::Quant`; `matvec_g` dispatches them to the
        // fused CPU decoder even when a Gpu context is present. Only f32 and the
        // ggml K-quants upload.
        let up = |w: &Weight, gpu: &Gpu| -> Weight {
            match w {
                Weight::Cpu { data, rows, cols } => {
                    Weight::Gpu(gpu.upload_matrix(data, *rows, *cols))
                }
                Weight::Quant {
                    bytes,
                    ggml_type,
                    rows,
                    cols,
                } if *ggml_type == crate::model::HQ4_TYPE
                    || *ggml_type == crate::model::E8_TYPE =>
                {
                    Weight::Quant {
                        bytes: bytes.clone(),
                        ggml_type: *ggml_type,
                        rows: *rows,
                        cols: *cols,
                    }
                }
                Weight::Quant {
                    bytes,
                    ggml_type,
                    rows,
                    cols,
                } => Weight::Gpu(gpu.upload_quant(bytes, *ggml_type, *rows, *cols)),
                Weight::Gpu(_) => panic!("[gemma4] upload_to_gpu: weight already on GPU"),
            }
        };
        for lw in self.layers.iter_mut() {
            lw.wq = up(&lw.wq, gpu);
            lw.wk = up(&lw.wk, gpu);
            if let Some(wv) = lw.wv.as_ref() {
                lw.wv = Some(up(wv, gpu));
            }
            lw.wo = up(&lw.wo, gpu);
            lw.gate = up(&lw.gate, gpu);
            lw.up = up(&lw.up, gpu);
            lw.down = up(&lw.down, gpu);
        }
        // embed → GPU for the tied lm-head matvec, but KEEP the f32 rows on the CPU
        // for the embedding lookup. f32 path: move the data out (no double alloc)
        // into `embed_cpu`. Quant path: `embed_cpu` already holds the f32 lookup
        // rows, so just upload the quantized bytes.
        let old = std::mem::replace(&mut self.embed, Weight::cpu(Vec::new(), self.cfg.hidden));
        match old {
            Weight::Cpu { data, rows, cols } => {
                self.embed = Weight::Gpu(gpu.upload_matrix(&data, rows, cols));
                self.embed_cpu = Some(data);
            }
            Weight::Quant {
                bytes,
                ggml_type,
                rows,
                cols,
            } if ggml_type == crate::model::HQ4_TYPE || ggml_type == crate::model::E8_TYPE => {
                // No Metal kernel for the native quants — keep the lm-head on CPU.
                self.embed = Weight::Quant {
                    bytes,
                    ggml_type,
                    rows,
                    cols,
                };
            }
            Weight::Quant {
                bytes,
                ggml_type,
                rows,
                cols,
            } => {
                self.embed = Weight::Gpu(gpu.upload_quant(&bytes, ggml_type, rows, cols));
            }
            Weight::Gpu(_) => panic!("[gemma4] upload_to_gpu: embed already on GPU"),
        }
        self.bf16_emul = false;
        eprintln!("[gemma4] uploaded big linears to GPU (bf16_emul disabled)");
    }

    /// Process ONE token (absolute position = current cache length) through all
    /// layers using + growing the KV cache, and return its post-softcap logits.
    /// Mirrors `layer` op-for-op for a single token, so on the CPU bf16 path the
    /// result is identical to `forward` over the same prefix.
    pub fn decode_step(&self, cache: &mut Gemma4Cache, id: u32, gpu: Option<&Gpu>) -> Vec<f32> {
        let bf = self.bf16_emul;
        let pos = cache.len;

        // embed * bf16(sqrt(hidden))
        let mut h: Vec<f32> = self
            .embed_row(id)
            .iter()
            .map(|&x| {
                let p = x * self.embed_scale;
                if bf {
                    bf16(p)
                } else {
                    p
                }
            })
            .collect();

        for l in 0..self.layers.len() {
            self.layer_step(l, cache, &mut h, pos, gpu);
        }
        cache.len = pos + 1;

        // final norm + tied lm-head + softcap (same as `forward`)
        let mut hn = rmsnorm_raw(&h, &self.final_norm, self.cfg.rms_eps);
        rvec(&mut hn, bf);
        let mut logits = matvec_g(&self.embed, &hn, self.cfg.vocab, gpu);
        rvec(&mut logits, bf);
        let cap = self.cfg.final_softcap;
        for v in logits.iter_mut() {
            *v = cap * (*v / cap).tanh();
            if bf {
                *v = bf16(*v);
            }
        }
        logits
    }

    /// One decoder layer over a SINGLE new token at absolute position `pos`,
    /// appending its processed k/v to `cache` and attending its q over the
    /// cached k/v. Identical per-op math + bf16 rounding to `layer`.
    fn layer_step(
        &self,
        l: usize,
        cache: &mut Gemma4Cache,
        h: &mut Vec<f32>,
        pos: usize,
        gpu: Option<&Gpu>,
    ) {
        let lw = &self.layers[l];
        let hidden = self.cfg.hidden;
        let eps = self.cfg.rms_eps;
        let hd = lw.head_dim();
        let n_kv = lw.n_kv();
        let nh = self.cfg.n_heads;
        let n_rep = nh / n_kv;
        let inv_freq: &[f32] = if lw.global {
            &self.inv_freq_global
        } else {
            &self.inv_freq_local
        };
        let bf = self.bf16_emul;

        // ---- projections + per-head norms + rope for this one token ----
        let mut a = rmsnorm_raw(h, &lw.input_ln, eps);
        rvec(&mut a, bf); // input_layernorm output -> bf16
        let mut q = matvec_g(&lw.wq, &a, nh * hd, gpu);
        rvec(&mut q, bf); // q_proj -> bf16
        for hh in 0..nh {
            let head = &mut q[hh * hd..(hh + 1) * hd];
            rmsnorm_raw_inplace(head, &lw.q_norm, eps); // q_norm BEFORE rope
            rvec(head, bf);
            rope_head(head, inv_freq, pos, bf);
        }
        let mut k_src = matvec_g(&lw.wk, &a, n_kv * hd, gpu);
        rvec(&mut k_src, bf); // k_proj -> bf16
        let mut v_src = if lw.global {
            k_src.clone() // k_eq_v
        } else {
            matvec_g(lw.wv.as_ref().unwrap(), &a, n_kv * hd, gpu)
        };
        rvec(&mut v_src, bf); // v_proj (or k_eq_v) -> bf16
        for hh in 0..n_kv {
            let head = &mut k_src[hh * hd..(hh + 1) * hd];
            rmsnorm_raw_inplace(head, &lw.k_norm, eps); // k_norm BEFORE rope
            rvec(head, bf);
            rope_head(head, inv_freq, pos, bf);
        }
        for hh in 0..n_kv {
            let head = &mut v_src[hh * hd..(hh + 1) * hd];
            rmsnorm_noscale(head, eps); // v_norm: scale-free, NO rope
            rvec(head, bf);
        }
        // append processed k/v for this token (its index in the cache == pos)
        cache.k[l].extend_from_slice(&k_src);
        cache.v[l].extend_from_slice(&v_src);
        let kc = &cache.k[l];
        let vc = &cache.v[l];

        // ---- attention (scaling = 1.0) over cached keys [lo, pos] ----
        let o_in = nh * hd;
        let lo = if lw.global {
            0
        } else {
            (pos + 1).saturating_sub(self.cfg.sliding_window)
        };
        let mut merged = vec![0.0f32; o_in];
        for hh in 0..nh {
            let kvh = hh / n_rep;
            let qh = &q[hh * hd..hh * hd + hd];
            let mut scores = vec![0.0f32; pos - lo + 1];
            for (idx, j) in (lo..=pos).enumerate() {
                let kh = &kc[j * n_kv * hd + kvh * hd..j * n_kv * hd + kvh * hd + hd];
                let mut s = 0.0f32;
                for d in 0..hd {
                    s += qh[d] * kh[d];
                }
                scores[idx] = s; // scaling = 1.0
            }
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - mx).exp();
                sum += *s;
            }
            for s in scores.iter_mut() {
                *s /= sum;
                if bf {
                    *s = bf16(*s);
                }
            }
            let out = &mut merged[hh * hd..(hh + 1) * hd];
            for (idx, j) in (lo..=pos).enumerate() {
                let vh = &vc[j * n_kv * hd + kvh * hd..j * n_kv * hd + kvh * hd + hd];
                let w = scores[idx];
                for d in 0..hd {
                    out[d] += w * vh[d];
                }
            }
        }
        rvec(&mut merged, bf); // attn_output -> bf16
        let mut attn = matvec_g(&lw.wo, &merged, hidden, gpu);
        rvec(&mut attn, bf); // o_proj -> bf16
        let mut post = rmsnorm_raw(&attn, &lw.post_attn_ln, eps);
        rvec(&mut post, bf);
        for i in 0..hidden {
            h[i] += post[i];
        }
        rvec(h, bf); // residual add -> bf16

        // ---- MLP block ----
        let mut pf = rmsnorm_raw(h, &lw.pre_ff_ln, eps);
        rvec(&mut pf, bf);
        let mut g = matvec_g(&lw.gate, &pf, self.cfg.inter, gpu);
        rvec(&mut g, bf); // gate_proj -> bf16
        let mut u = matvec_g(&lw.up, &pf, self.cfg.inter, gpu);
        rvec(&mut u, bf); // up_proj -> bf16
        for i in 0..self.cfg.inter {
            let gi = if bf {
                bf16(gelu_tanh(g[i]))
            } else {
                gelu_tanh(g[i])
            };
            g[i] = gi * u[i];
            if bf {
                g[i] = bf16(g[i]);
            }
        }
        let mut m = matvec_g(&lw.down, &g, hidden, gpu);
        rvec(&mut m, bf); // down_proj -> bf16
        let mut post = rmsnorm_raw(&m, &lw.post_ff_ln, eps);
        rvec(&mut post, bf);
        for i in 0..hidden {
            h[i] += post[i];
        }
        rvec(h, bf); // residual add -> bf16
        let ls = if bf {
            bf16(lw.layer_scalar)
        } else {
            lw.layer_scalar
        };
        for i in 0..hidden {
            h[i] *= ls;
        }
        rvec(h, bf); // *= layer_scalar -> bf16
    }

    /// One decoder layer over the WHOLE prompt at once, writing each token's
    /// processed k/v into `cache`. Same per-op math + bf16 rounding as `layer_step`,
    /// but the seven big projections (wq/wk/wv/wo/gate/up/down) go through
    /// `cpu_matmat`, which dequantizes each weight row ONCE and dots it against all
    /// `seq` tokens — so a P-token prompt reads each weight once instead of P times.
    /// Bit-identical to looping `layer_step` (same dequant + `dot_f32` + rounding).
    /// CPU only (`matvec`/`cpu_matmat`, no `gpu`); the GPU prefill is separate.
    fn layer_prefill_cpu(&self, l: usize, cache: &mut Gemma4Cache, h: &mut [Vec<f32>], start: usize) {
        let lw = &self.layers[l];
        let hidden = self.cfg.hidden;
        let eps = self.cfg.rms_eps;
        let hd = lw.head_dim();
        let n_kv = lw.n_kv();
        let nh = self.cfg.n_heads;
        let n_rep = nh / n_kv;
        let inv_freq: &[f32] = if lw.global {
            &self.inv_freq_global
        } else {
            &self.inv_freq_local
        };
        let bf = self.bf16_emul;
        let seq = h.len();

        // input_layernorm -> bf16, stacked as A[seq, hidden]
        let mut a_all = vec![0.0f32; seq * hidden];
        for t in 0..seq {
            let mut a = rmsnorm_raw(&h[t], &lw.input_ln, eps);
            rvec(&mut a, bf);
            a_all[t * hidden..(t + 1) * hidden].copy_from_slice(&a);
        }

        // batched projections (each weight row dequantized once, dotted over all t)
        let mut q = vec![0.0f32; seq * nh * hd];
        crate::model::cpu_matmat(&mut q, &lw.wq, &a_all, seq);
        let mut k = vec![0.0f32; seq * n_kv * hd];
        crate::model::cpu_matmat(&mut k, &lw.wk, &a_all, seq);
        // q_proj/k_proj -> bf16, then per-head q/k-norm BEFORE rope
        for t in 0..seq {
            let qslot = &mut q[t * nh * hd..(t + 1) * nh * hd];
            rvec(qslot, bf);
            for hh in 0..nh {
                let head = &mut qslot[hh * hd..(hh + 1) * hd];
                rmsnorm_raw_inplace(head, &lw.q_norm, eps);
                rvec(head, bf);
                rope_head(head, inv_freq, start + t, bf);
            }
            let kslot = &mut k[t * n_kv * hd..(t + 1) * n_kv * hd];
            rvec(kslot, bf);
        }
        // v source: global => v = k_src (k_eq_v, taken AFTER k's bf16 round, BEFORE
        // k-norm/rope); sliding => v_proj(a). Then k-norm+rope on k, v-norm on v.
        let mut v = vec![0.0f32; seq * n_kv * hd];
        if lw.global {
            v.copy_from_slice(&k);
        } else {
            crate::model::cpu_matmat(&mut v, lw.wv.as_ref().unwrap(), &a_all, seq);
            for t in 0..seq {
                rvec(&mut v[t * n_kv * hd..(t + 1) * n_kv * hd], bf);
            }
        }
        for t in 0..seq {
            let kslot = &mut k[t * n_kv * hd..(t + 1) * n_kv * hd];
            for hh in 0..n_kv {
                let head = &mut kslot[hh * hd..(hh + 1) * hd];
                rmsnorm_raw_inplace(head, &lw.k_norm, eps);
                rvec(head, bf);
                rope_head(head, inv_freq, start + t, bf);
            }
            let vslot = &mut v[t * n_kv * hd..(t + 1) * n_kv * hd];
            for hh in 0..n_kv {
                let head = &mut vslot[hh * hd..(hh + 1) * hd];
                rmsnorm_noscale(head, eps);
                rvec(head, bf);
            }
        }
        // append this whole prompt's processed k/v to the cache (positions start..start+seq)
        cache.k[l].extend_from_slice(&k);
        cache.v[l].extend_from_slice(&v);

        // attention (scaling = 1.0) then o_proj (batched) + post_attn_ln + residual
        let o_in = nh * hd;
        let mut merged_all = vec![0.0f32; seq * o_in];
        for t in 0..seq {
            let abs = start + t;
            let lo = if lw.global {
                0
            } else {
                (abs + 1).saturating_sub(self.cfg.sliding_window)
            };
            let merged = &mut merged_all[t * o_in..(t + 1) * o_in];
            for hh in 0..nh {
                let kvh = hh / n_rep;
                let qh = &q[t * nh * hd + hh * hd..t * nh * hd + hh * hd + hd];
                let mut scores = vec![0.0f32; abs - lo + 1];
                for (idx, j) in (lo..=abs).enumerate() {
                    let kh = &k[j * n_kv * hd + kvh * hd..j * n_kv * hd + kvh * hd + hd];
                    let mut s = 0.0f32;
                    for d in 0..hd {
                        s += qh[d] * kh[d];
                    }
                    scores[idx] = s;
                }
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                for s in scores.iter_mut() {
                    *s /= sum;
                    if bf {
                        *s = bf16(*s);
                    }
                }
                let out = &mut merged[hh * hd..(hh + 1) * hd];
                for (idx, j) in (lo..=abs).enumerate() {
                    let vh = &v[j * n_kv * hd + kvh * hd..j * n_kv * hd + kvh * hd + hd];
                    let w = scores[idx];
                    for d in 0..hd {
                        out[d] += w * vh[d];
                    }
                }
            }
            rvec(merged, bf); // attn_output -> bf16
        }
        let mut attn_all = vec![0.0f32; seq * hidden];
        crate::model::cpu_matmat(&mut attn_all, &lw.wo, &merged_all, seq);
        for t in 0..seq {
            let attn = &mut attn_all[t * hidden..(t + 1) * hidden];
            rvec(attn, bf); // o_proj -> bf16
            let mut post = rmsnorm_raw(attn, &lw.post_attn_ln, eps);
            rvec(&mut post, bf);
            for i in 0..hidden {
                h[t][i] += post[i];
            }
            rvec(&mut h[t], bf); // residual add -> bf16
        }

        // MLP: pre_ff_ln -> gate/up (batched) -> gelu*up -> down (batched) -> post_ff_ln
        let mut pf_all = vec![0.0f32; seq * hidden];
        for t in 0..seq {
            let mut pf = rmsnorm_raw(&h[t], &lw.pre_ff_ln, eps);
            rvec(&mut pf, bf);
            pf_all[t * hidden..(t + 1) * hidden].copy_from_slice(&pf);
        }
        let inter = self.cfg.inter;
        let mut g_all = vec![0.0f32; seq * inter];
        crate::model::cpu_matmat(&mut g_all, &lw.gate, &pf_all, seq);
        let mut u_all = vec![0.0f32; seq * inter];
        crate::model::cpu_matmat(&mut u_all, &lw.up, &pf_all, seq);
        for t in 0..seq {
            let g = &mut g_all[t * inter..(t + 1) * inter];
            let u = &u_all[t * inter..(t + 1) * inter];
            rvec(g, bf); // gate_proj -> bf16
            // NOTE: up is rounded in place below on its own slice
            let _ = u;
        }
        for t in 0..seq {
            rvec(&mut u_all[t * inter..(t + 1) * inter], bf); // up_proj -> bf16
        }
        for t in 0..seq {
            let g = &mut g_all[t * inter..(t + 1) * inter];
            let u = &u_all[t * inter..(t + 1) * inter];
            for i in 0..inter {
                let gi = if bf { bf16(gelu_tanh(g[i])) } else { gelu_tanh(g[i]) };
                g[i] = gi * u[i];
                if bf {
                    g[i] = bf16(g[i]);
                }
            }
        }
        let mut m_all = vec![0.0f32; seq * hidden];
        crate::model::cpu_matmat(&mut m_all, &lw.down, &g_all, seq);
        for t in 0..seq {
            let m = &mut m_all[t * hidden..(t + 1) * hidden];
            rvec(m, bf); // down_proj -> bf16
            let mut post = rmsnorm_raw(m, &lw.post_ff_ln, eps);
            rvec(&mut post, bf);
            for i in 0..hidden {
                h[t][i] += post[i];
            }
            rvec(&mut h[t], bf); // residual add -> bf16
            let ls = if bf { bf16(lw.layer_scalar) } else { lw.layer_scalar };
            for i in 0..hidden {
                h[t][i] *= ls;
            }
            rvec(&mut h[t], bf); // *= layer_scalar -> bf16
        }
    }

    /// Batched CPU prefill: process the whole prompt in one pass per layer (each
    /// weight read once) and return the last-position post-softcap logits. Produces
    /// byte-identical logits to the per-token `prefill`, at a fraction of the weight
    /// bandwidth on x86/Windows. Starts from a fresh cache (position 0).
    fn prefill_cpu_batched(&self, cache: &mut Gemma4Cache, ids: &[u32]) -> Vec<f32> {
        let bf = self.bf16_emul;
        let start = cache.len;
        // embed * bf16(sqrt(hidden)) for every token
        let mut h: Vec<Vec<f32>> = ids.iter().map(|&id| self.text_embed(id)).collect();
        for l in 0..self.layers.len() {
            self.layer_prefill_cpu(l, cache, &mut h, start);
        }
        cache.len = start + ids.len();
        // final norm + tied lm-head + softcap on the last position
        let last = &h[h.len() - 1];
        let mut hn = rmsnorm_raw(last, &self.final_norm, self.cfg.rms_eps);
        rvec(&mut hn, bf);
        let mut logits = matvec_g(&self.embed, &hn, self.cfg.vocab, None);
        rvec(&mut logits, bf);
        let cap = self.cfg.final_softcap;
        for vv in logits.iter_mut() {
            *vv = cap * (*vv / cap).tanh();
            if bf {
                *vv = bf16(*vv);
            }
        }
        logits
    }

    /// Prefill: process the whole prompt through the cache, returning the
    /// last-position post-softcap logits (== `forward(ids).logits`). On CPU with a
    /// multi-token prompt this uses the batched (weight-read-once) path; the GPU
    /// path and tiny prompts fall back to the per-token loop.
    pub fn prefill(&self, cache: &mut Gemma4Cache, ids: &[u32], gpu: Option<&Gpu>) -> Vec<f32> {
        if gpu.is_none()
            && ids.len() >= 8
            && cache.len == 0
            && std::env::var("HOS_GEMMA4_NOBATCH").is_err()
        {
            return self.prefill_cpu_batched(cache, ids);
        }
        let mut logits = Vec::new();
        for &id in ids {
            logits = self.decode_step(cache, id, gpu);
        }
        logits
    }

    /// KV-cached greedy generation: prefill the prompt, then decode `n` tokens
    /// one at a time. On the CPU bf16 path this yields the SAME token ids as
    /// `generate`, at O(n) instead of O(n²).
    pub fn generate_cached(&self, ids: &[u32], n: usize, gpu: Option<&Gpu>) -> Vec<u32> {
        let mut cache = Gemma4Cache::new(self);
        let mut logits = self.prefill(&mut cache, ids, gpu);
        let argmax = |lg: &[f32]| -> u32 {
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &l) in lg.iter().enumerate() {
                if l > bv {
                    bv = l;
                    best = i;
                }
            }
            best as u32
        };
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let best = argmax(&logits);
            out.push(best);
            if out.len() == n {
                break;
            }
            logits = self.decode_step(&mut cache, best, gpu);
        }
        out
    }

    // ======================================================================
    // IMAGE path (additive) — splice UNSCALED vision soft-tokens into the
    // input embeddings + BIDIRECTIONAL attention within each image-token span.
    // Text-only prompts have no image tokens -> `spans` empty -> mask identical
    // to the causal text path (so --gemma4-selftest/-kv-check are unaffected).
    // ======================================================================

    /// Public helper: raw text embedding row for `id`, scaled by
    /// `embed_scale`(bf16) and bf16-rounded when bf16 emulation is on. Matches
    /// exactly what `forward`/`decode_step` build for a text token.
    pub fn text_embed(&self, id: u32) -> Vec<f32> {
        let bf = self.bf16_emul;
        self.embed_row(id)
            .iter()
            .map(|&x| {
                let p = x * self.embed_scale;
                if bf {
                    bf16(p)
                } else {
                    p
                }
            })
            .collect()
    }

    /// Build the spliced input embeddings for an image prompt: text tokens are
    /// `embed[id]*embed_scale` (as usual); each `image_token_id` position is
    /// OVERWRITTEN with the corresponding UNSCALED vision soft-token row
    /// (row-major 1:1). `soft_tokens` is `[n_img, hidden]` flat, where `n_img`
    /// == number of `image_token_id` positions in `ids`. Returns
    /// `(embeds, spans)` where `spans` are the contiguous `[start,end)` ranges
    /// of image-token positions (for bidirectional attention).
    pub fn build_image_embeds(
        &self,
        ids: &[u32],
        soft_tokens: &[f32],
        image_token_id: u32,
    ) -> (Vec<Vec<f32>>, Vec<(usize, usize)>) {
        let hidden = self.cfg.hidden;
        let mut embeds: Vec<Vec<f32>> = Vec::with_capacity(ids.len());
        let mut spans: Vec<(usize, usize)> = Vec::new();
        let mut img_idx = 0usize;
        let mut cur_start: Option<usize> = None;
        for (pos, &id) in ids.iter().enumerate() {
            if id == image_token_id {
                let row = &soft_tokens[img_idx * hidden..(img_idx + 1) * hidden];
                embeds.push(row.to_vec()); // UNSCALED
                img_idx += 1;
                if cur_start.is_none() {
                    cur_start = Some(pos);
                }
            } else {
                if let Some(s) = cur_start.take() {
                    spans.push((s, pos));
                }
                embeds.push(self.text_embed(id));
            }
        }
        if let Some(s) = cur_start.take() {
            spans.push((s, ids.len()));
        }
        (embeds, spans)
    }

    /// One decoder layer over the WHOLE sequence with pre-built embeddings and
    /// BIDIRECTIONAL attention inside image spans; stores per-token k/v into
    /// `cache` (in position order) so subsequent text tokens can be decoded
    /// causally with `decode_step`. Op-for-op identical to `layer` except the
    /// attention allowed-key set is (causal ∪ same-span future keys).
    fn layer_image(
        &self,
        l: usize,
        lw: &LayerW,
        h: &mut [Vec<f32>],
        span_of: &[Option<usize>],
        spans: &[(usize, usize)],
        cache: &mut Gemma4Cache,
        gpu: Option<&Gpu>,
    ) {
        let seq = h.len();
        let hidden = self.cfg.hidden;
        let eps = self.cfg.rms_eps;
        let hd = lw.head_dim();
        let n_kv = lw.n_kv();
        let nh = self.cfg.n_heads;
        let n_rep = nh / n_kv;
        let inv_freq: &[f32] = if lw.global {
            &self.inv_freq_global
        } else {
            &self.inv_freq_local
        };
        let bf = self.bf16_emul;

        let mut q = vec![0.0f32; seq * nh * hd];
        let mut k = vec![0.0f32; seq * n_kv * hd];
        let mut v = vec![0.0f32; seq * n_kv * hd];
        for t in 0..seq {
            let mut a = rmsnorm_raw(&h[t], &lw.input_ln, eps);
            rvec(&mut a, bf);
            let mut qt = matvec_g(&lw.wq, &a, nh * hd, gpu);
            rvec(&mut qt, bf);
            let qslot = &mut q[t * nh * hd..(t + 1) * nh * hd];
            qslot.copy_from_slice(&qt);
            for hh in 0..nh {
                let head = &mut qslot[hh * hd..(hh + 1) * hd];
                rmsnorm_raw_inplace(head, &lw.q_norm, eps);
                rvec(head, bf);
                rope_head(head, inv_freq, t, bf);
            }
            let mut k_src = matvec_g(&lw.wk, &a, n_kv * hd, gpu);
            rvec(&mut k_src, bf);
            let mut v_src = if lw.global {
                k_src.clone()
            } else {
                matvec_g(lw.wv.as_ref().unwrap(), &a, n_kv * hd, gpu)
            };
            rvec(&mut v_src, bf);
            let kslot = &mut k[t * n_kv * hd..(t + 1) * n_kv * hd];
            kslot.copy_from_slice(&k_src);
            for hh in 0..n_kv {
                let head = &mut kslot[hh * hd..(hh + 1) * hd];
                rmsnorm_raw_inplace(head, &lw.k_norm, eps);
                rvec(head, bf);
                rope_head(head, inv_freq, t, bf);
            }
            let vslot = &mut v[t * n_kv * hd..(t + 1) * n_kv * hd];
            vslot.copy_from_slice(&v_src);
            for hh in 0..n_kv {
                let head = &mut vslot[hh * hd..(hh + 1) * hd];
                rmsnorm_noscale(head, eps);
                rvec(head, bf);
            }
        }
        // Persist processed k/v into the cache in position order (0..seq).
        cache.k[l].extend_from_slice(&k);
        cache.v[l].extend_from_slice(&v);

        let o_in = nh * hd;
        for t in 0..seq {
            let lo = if lw.global {
                0
            } else {
                (t + 1).saturating_sub(self.cfg.sliding_window)
            };
            // Allowed keys: causal [lo, t] plus (if t is inside an image span)
            // the same span's FUTURE keys (t+1 .. span_end) — bidirectional.
            let mut keys: Vec<usize> = (lo..=t).collect();
            if let Some(sid) = span_of[t] {
                let (s, e) = spans[sid];
                let lo_s = s.max(lo);
                // include any same-span keys below `lo` (window edge) too
                for j in lo_s..lo.min(e) {
                    keys.push(j);
                }
                for j in (t + 1)..e {
                    keys.push(j);
                }
            }
            let mut merged = vec![0.0f32; o_in];
            for hh in 0..nh {
                let kvh = hh / n_rep;
                let qh = &q[t * nh * hd + hh * hd..t * nh * hd + hh * hd + hd];
                let mut scores = vec![0.0f32; keys.len()];
                for (idx, &j) in keys.iter().enumerate() {
                    let kh = &k[j * n_kv * hd + kvh * hd..j * n_kv * hd + kvh * hd + hd];
                    let mut s = 0.0f32;
                    for d in 0..hd {
                        s += qh[d] * kh[d];
                    }
                    scores[idx] = s;
                }
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                for s in scores.iter_mut() {
                    *s /= sum;
                    if bf {
                        *s = bf16(*s);
                    }
                }
                let out = &mut merged[hh * hd..(hh + 1) * hd];
                for (idx, &j) in keys.iter().enumerate() {
                    let vh = &v[j * n_kv * hd + kvh * hd..j * n_kv * hd + kvh * hd + hd];
                    let w = scores[idx];
                    for d in 0..hd {
                        out[d] += w * vh[d];
                    }
                }
            }
            rvec(&mut merged, bf);
            let mut attn = matvec_g(&lw.wo, &merged, hidden, gpu);
            rvec(&mut attn, bf);
            let mut post = rmsnorm_raw(&attn, &lw.post_attn_ln, eps);
            rvec(&mut post, bf);
            for i in 0..hidden {
                h[t][i] += post[i];
            }
            rvec(&mut h[t], bf);
        }

        for t in 0..seq {
            let mut pf = rmsnorm_raw(&h[t], &lw.pre_ff_ln, eps);
            rvec(&mut pf, bf);
            let mut g = matvec_g(&lw.gate, &pf, self.cfg.inter, gpu);
            rvec(&mut g, bf);
            let mut u = matvec_g(&lw.up, &pf, self.cfg.inter, gpu);
            rvec(&mut u, bf);
            for i in 0..self.cfg.inter {
                let gi = if bf {
                    bf16(gelu_tanh(g[i]))
                } else {
                    gelu_tanh(g[i])
                };
                g[i] = gi * u[i];
                if bf {
                    g[i] = bf16(g[i]);
                }
            }
            let mut m = matvec_g(&lw.down, &g, hidden, gpu);
            rvec(&mut m, bf);
            let mut post = rmsnorm_raw(&m, &lw.post_ff_ln, eps);
            rvec(&mut post, bf);
            for i in 0..hidden {
                h[t][i] += post[i];
            }
            rvec(&mut h[t], bf);
            let ls = if bf {
                bf16(lw.layer_scalar)
            } else {
                lw.layer_scalar
            };
            for i in 0..hidden {
                h[t][i] *= ls;
            }
            rvec(&mut h[t], bf);
        }
    }

    /// Prefill the whole image prompt (pre-built spliced embeddings) with
    /// bidirectional image attention, populating `cache`, and return the
    /// last-position post-softcap logits.
    pub fn prefill_image(
        &self,
        cache: &mut Gemma4Cache,
        embeds: &[Vec<f32>],
        spans: &[(usize, usize)],
        gpu: Option<&Gpu>,
    ) -> Vec<f32> {
        let seq = embeds.len();
        let mut span_of = vec![None; seq];
        for (sid, &(s, e)) in spans.iter().enumerate() {
            for pos in s..e {
                span_of[pos] = Some(sid);
            }
        }
        let mut h: Vec<Vec<f32>> = embeds.to_vec();
        for (l, lw) in self.layers.iter().enumerate() {
            self.layer_image(l, lw, &mut h, &span_of, spans, cache, gpu);
        }
        cache.len = seq;

        let bf = self.bf16_emul;
        let last = seq - 1;
        let mut hn = rmsnorm_raw(&h[last], &self.final_norm, self.cfg.rms_eps);
        rvec(&mut hn, bf);
        let mut logits = matvec_g(&self.embed, &hn, self.cfg.vocab, gpu);
        rvec(&mut logits, bf);
        let cap = self.cfg.final_softcap;
        for val in logits.iter_mut() {
            *val = cap * (*val / cap).tanh();
            if bf {
                *val = bf16(*val);
            }
        }
        logits
    }

    // ======================================================================
    // BATCHED GPU PREFILL (additive, opt-in on --gpu + q4_k). Processes the WHOLE
    // prompt in ONE pass: every q/k/v/o and gate/up/down projection becomes a
    // single batched matmul (X[seq,in]·W^T -> [seq,out]) on Metal — the weight is
    // read+dequanted ONCE and reused across all `seq` tokens, replacing `seq`
    // sequential per-token `matvec`s (the 90% win). Norms/qk-norm/rope/v-norm and
    // attention stay on the CPU (cheap, O(seq·hidden)/O(seq²)), rayon-parallel.
    //
    // Numerically this differs from the sequential GPU prefill only in the float
    // summation order of the projection matmul (f32 dequant-then-GEMM vs the
    // per-token coalesced matvec) — top-1 identical, cosine >0.999. The sequential
    // path (`layer_image`/`prefill_image`) is untouched and stays the fallback.
    // ======================================================================

    /// True iff every per-layer projection weight is a resident q4_k GpuMatrix, so
    /// the batched q4_k prefill matmul applies. (embed/lm-head may be a different
    /// type — the head runs as a single last-position matvec, not batched.)
    pub fn can_batch_prefill(&self) -> bool {
        let is_q4k_gpu =
            |w: &Weight| matches!(w, Weight::Gpu(m) if m.ggml_type == crate::gguf::GGML_Q4_K);
        self.layers.iter().all(|l| {
            is_q4k_gpu(&l.wq)
                && is_q4k_gpu(&l.wk)
                && is_q4k_gpu(&l.wo)
                && is_q4k_gpu(&l.gate)
                && is_q4k_gpu(&l.up)
                && is_q4k_gpu(&l.down)
                && l.wv.as_ref().is_none_or(is_q4k_gpu)
        })
    }

    /// One decoder layer over the WHOLE sequence with the projections done as
    /// batched GPU matmuls. Op-for-op equivalent to `layer_image` (same
    /// bidirectional-image / sliding / global masks, same qk-norm/rope/v-norm,
    /// same cache population) — only the projection compute is batched. `gpu` must
    /// be present and `can_batch_prefill()` true. bf16 emulation is OFF here (GPU
    /// path), so no per-op bf16 rounding — matching the sequential GPU prefill.
    fn layer_batched(
        &self,
        l: usize,
        lw: &LayerW,
        h: &mut [Vec<f32>],
        span_of: &[Option<usize>],
        spans: &[(usize, usize)],
        cache: &mut Gemma4Cache,
        gpu: &Gpu,
    ) {
        use rayon::prelude::*;
        let seq = h.len();
        let hidden = self.cfg.hidden;
        let eps = self.cfg.rms_eps;
        let hd = lw.head_dim();
        let n_kv = lw.n_kv();
        let nh = self.cfg.n_heads;
        let n_rep = nh / n_kv;
        let inter = self.cfg.inter;
        let inv_freq: &[f32] = if lw.global {
            &self.inv_freq_global
        } else {
            &self.inv_freq_local
        };

        // Batched projection helper: Y[seq,out] = X[seq,in] @ W^T on the GPU.
        let proj = |w: &Weight, x: &[f32]| -> Vec<f32> {
            let gm = w.as_gpu();
            let mut y = vec![0.0f32; seq * gm.n_rows];
            gpu.matmul_q4k_prefill_into(gm, x, seq, &mut y);
            y
        };

        // ---- A = input_layernorm(h), per row (rayon) ----
        let mut a_flat = vec![0.0f32; seq * hidden];
        a_flat
            .par_chunks_mut(hidden)
            .zip(h.par_iter())
            .for_each(|(dst, ht)| {
                dst.copy_from_slice(&rmsnorm_raw(ht, &lw.input_ln, eps));
            });

        // ---- q/k/v projections (batched), then per-head norms + rope (rayon) ----
        let qo = nh * hd;
        let kvo = n_kv * hd;
        let mut q = proj(&lw.wq, &a_flat);
        let mut k = proj(&lw.wk, &a_flat);
        let mut v = if lw.global {
            k.clone() // k_eq_v (global MQA): v source == raw k projection
        } else {
            proj(lw.wv.as_ref().unwrap(), &a_flat)
        };
        q.par_chunks_mut(qo).enumerate().for_each(|(t, qslot)| {
            for hh in 0..nh {
                let head = &mut qslot[hh * hd..(hh + 1) * hd];
                rmsnorm_raw_inplace(head, &lw.q_norm, eps); // q_norm BEFORE rope
                rope_head(head, inv_freq, t, false);
            }
        });
        k.par_chunks_mut(kvo).enumerate().for_each(|(t, kslot)| {
            for hh in 0..n_kv {
                let head = &mut kslot[hh * hd..(hh + 1) * hd];
                rmsnorm_raw_inplace(head, &lw.k_norm, eps); // k_norm BEFORE rope
                rope_head(head, inv_freq, t, false);
            }
        });
        v.par_chunks_mut(kvo).for_each(|vslot| {
            for hh in 0..n_kv {
                let head = &mut vslot[hh * hd..(hh + 1) * hd];
                rmsnorm_noscale(head, eps); // v_norm: scale-free, NO rope
            }
        });
        // Persist processed k/v into the cache in position order (0..seq).
        cache.k[l].extend_from_slice(&k);
        cache.v[l].extend_from_slice(&v);

        // ---- attention (scaling = 1.0), per query row (rayon). Same allowed-key
        //      set as layer_image: causal [lo,t] ∪ same-span future keys. ----
        let o_in = nh * hd;
        let mut merged = vec![0.0f32; seq * o_in];
        merged
            .par_chunks_mut(o_in)
            .enumerate()
            .for_each(|(t, mrow)| {
                let lo = if lw.global {
                    0
                } else {
                    (t + 1).saturating_sub(self.cfg.sliding_window)
                };
                let mut keys: Vec<usize> = (lo..=t).collect();
                if let Some(sid) = span_of[t] {
                    let (s, e) = spans[sid];
                    let lo_s = s.max(lo);
                    for j in lo_s..lo.min(e) {
                        keys.push(j);
                    }
                    for j in (t + 1)..e {
                        keys.push(j);
                    }
                }
                for hh in 0..nh {
                    let kvh = hh / n_rep;
                    let qh = &q[t * qo + hh * hd..t * qo + hh * hd + hd];
                    let mut scores = vec![0.0f32; keys.len()];
                    for (idx, &j) in keys.iter().enumerate() {
                        let kh = &k[j * kvo + kvh * hd..j * kvo + kvh * hd + hd];
                        let mut sc = 0.0f32;
                        for d in 0..hd {
                            sc += qh[d] * kh[d];
                        }
                        scores[idx] = sc;
                    }
                    let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for sc in scores.iter_mut() {
                        *sc = (*sc - mx).exp();
                        sum += *sc;
                    }
                    for sc in scores.iter_mut() {
                        *sc /= sum;
                    }
                    let out = &mut mrow[hh * hd..(hh + 1) * hd];
                    for (idx, &j) in keys.iter().enumerate() {
                        let vh = &v[j * kvo + kvh * hd..j * kvo + kvh * hd + hd];
                        let wsc = scores[idx];
                        for d in 0..hd {
                            out[d] += wsc * vh[d];
                        }
                    }
                }
            });

        // ---- o_proj (batched), post_attention_layernorm + residual (rayon) ----
        let o = proj(&lw.wo, &merged);
        h.par_iter_mut().enumerate().for_each(|(t, ht)| {
            let post = rmsnorm_raw(&o[t * hidden..(t + 1) * hidden], &lw.post_attn_ln, eps);
            for i in 0..hidden {
                ht[i] += post[i];
            }
        });

        // ---- MLP: r + post_ff_ln(down(gelu(gate(pf)) * up(pf))), * layer_scalar ----
        let mut pf = vec![0.0f32; seq * hidden];
        pf.par_chunks_mut(hidden)
            .zip(h.par_iter())
            .for_each(|(dst, ht)| {
                dst.copy_from_slice(&rmsnorm_raw(ht, &lw.pre_ff_ln, eps));
            });
        let mut g = proj(&lw.gate, &pf);
        let u = proj(&lw.up, &pf);
        g.par_iter_mut().zip(u.par_iter()).for_each(|(gi, &ui)| {
            *gi = gelu_tanh(*gi) * ui;
        });
        let _ = inter;
        let m = proj(&lw.down, &g);
        let ls = lw.layer_scalar;
        h.par_iter_mut().enumerate().for_each(|(t, ht)| {
            let post = rmsnorm_raw(&m[t * hidden..(t + 1) * hidden], &lw.post_ff_ln, eps);
            for i in 0..hidden {
                ht[i] = (ht[i] + post[i]) * ls;
            }
        });
    }

    /// Batched-prefill core (shared by text + image): run `layer_batched` over all
    /// layers, populate `cache`, and return the last-position post-softcap logits.
    /// The head (final_norm + tied lm-head + softcap) runs as a single matvec at
    /// the last position — not batched.
    fn prefill_core_batched(
        &self,
        cache: &mut Gemma4Cache,
        embeds: &[Vec<f32>],
        spans: &[(usize, usize)],
        gpu: &Gpu,
    ) -> Vec<f32> {
        let seq = embeds.len();
        let mut span_of = vec![None; seq];
        for (sid, &(s, e)) in spans.iter().enumerate() {
            for pos in s..e {
                span_of[pos] = Some(sid);
            }
        }
        let mut h: Vec<Vec<f32>> = embeds.to_vec();
        for (l, lw) in self.layers.iter().enumerate() {
            self.layer_batched(l, lw, &mut h, &span_of, spans, cache, gpu);
        }
        cache.len = seq;

        let last = seq - 1;
        let hn = rmsnorm_raw(&h[last], &self.final_norm, self.cfg.rms_eps);
        let mut logits = matvec_g(&self.embed, &hn, self.cfg.vocab, Some(gpu));
        let cap = self.cfg.final_softcap;
        for val in logits.iter_mut() {
            *val = cap * (*val / cap).tanh();
        }
        logits
    }

    /// Batched image prefill: identical result to `prefill_image` (bidirectional
    /// image spans) but projections batched on the GPU.
    pub fn prefill_image_batched(
        &self,
        cache: &mut Gemma4Cache,
        embeds: &[Vec<f32>],
        spans: &[(usize, usize)],
        gpu: &Gpu,
    ) -> Vec<f32> {
        self.prefill_core_batched(cache, embeds, spans, gpu)
    }

    /// Batched TEXT prefill: identical result to `prefill` (plain causal) but
    /// projections batched on the GPU.
    pub fn prefill_batched(&self, cache: &mut Gemma4Cache, ids: &[u32], gpu: &Gpu) -> Vec<f32> {
        let embeds: Vec<Vec<f32>> = ids.iter().map(|&id| self.text_embed(id)).collect();
        self.prefill_core_batched(cache, &embeds, &[], gpu)
    }

    /// Dispatcher: use the batched GPU prefill when a GPU is present, the prompt is
    /// long enough to amortize, and all projections are q4_k-on-GPU; otherwise fall
    /// back to the exact sequential `prefill_image`. Drop-in for the classifier.
    pub fn prefill_image_fast(
        &self,
        cache: &mut Gemma4Cache,
        embeds: &[Vec<f32>],
        spans: &[(usize, usize)],
        gpu: Option<&Gpu>,
    ) -> Vec<f32> {
        if let Some(g) = gpu {
            if embeds.len() >= PREFILL_BATCH_THRESHOLD && self.can_batch_prefill() {
                return self.prefill_image_batched(cache, embeds, spans, g);
            }
        }
        self.prefill_image(cache, embeds, spans, gpu)
    }

    /// End-to-end greedy image generation: splice `soft_tokens` at the
    /// `image_token_id` positions of `ids`, prefill with bidirectional image
    /// attention, then greedily decode `n` further tokens (plain causal over the
    /// cache). Returns the `n` generated ids.
    pub fn generate_image(
        &self,
        ids: &[u32],
        soft_tokens: &[f32],
        image_token_id: u32,
        n: usize,
        gpu: Option<&Gpu>,
    ) -> Vec<u32> {
        let (embeds, spans) = self.build_image_embeds(ids, soft_tokens, image_token_id);
        let mut cache = Gemma4Cache::new(self);
        // Use the batched GPU prefill when eligible (--gpu + q4_k, long prompt);
        // falls back to the exact sequential prefill otherwise.
        let mut logits = self.prefill_image_fast(&mut cache, &embeds, &spans, gpu);
        let argmax = |lg: &[f32]| -> u32 {
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &l) in lg.iter().enumerate() {
                if l > bv {
                    bv = l;
                    best = i;
                }
            }
            best as u32
        };
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let best = argmax(&logits);
            out.push(best);
            if out.len() == n {
                break;
            }
            logits = self.decode_step(&mut cache, best, gpu);
        }
        out
    }

    // ======================================================================
    // CONSTRAINED CLASSIFIER (additive) — label-scoring over a fixed set.
    //
    // Given a KV cache already prefilled to the point where the model is about
    // to emit the answer word (e.g. image + preamble prefilled once, then a
    // per-question suffix appended), score EACH candidate label by its
    // teacher-forced summed log-probability (sum over the label's token ids of
    // log-softmax of the post-softcap logits, conditioned on the prompt +
    // preceding label tokens). Pick argmax; confidence = softmax over the
    // candidates' summed logprobs. Guarantees a valid label from the set.
    // ======================================================================

    /// Feed `suffix_ids` onto a CLONE of `base_cache` (leaving `base_cache`
    /// untouched so it can be reused across questions), then score each
    /// `candidates[i]` (a pre-tokenized label token-id sequence) by teacher-forced
    /// summed log-prob. Returns `(best_idx, summed_logprobs, softmax_confidence)`.
    pub fn classify(
        &self,
        base_cache: &Gemma4Cache,
        suffix_ids: &[u32],
        candidates: &[Vec<u32>],
        gpu: Option<&Gpu>,
    ) -> (usize, Vec<f32>, Vec<f32>) {
        // Advance a private clone of the base cache through the question suffix.
        // The LAST step's logits are the distribution over the first answer token.
        let mut c = base_cache.clone_cache();
        let mut first_logits: Vec<f32> = Vec::new();
        for &id in suffix_ids {
            first_logits = self.decode_step(&mut c, id, gpu);
        }

        let mut sums: Vec<f32> = Vec::with_capacity(candidates.len());
        for cand in candidates {
            debug_assert!(!cand.is_empty(), "classify: empty candidate tokenization");
            // First label token scored against the post-prompt logits.
            let mut lp = logprob_of(&first_logits, cand[0]);
            // Remaining tokens: teacher-force on a throwaway clone of `c`.
            if cand.len() > 1 {
                let mut cc = c.clone_cache();
                for w in 0..cand.len() - 1 {
                    let lg = self.decode_step(&mut cc, cand[w], gpu);
                    lp += logprob_of(&lg, cand[w + 1]);
                }
            }
            sums.push(lp as f32);
        }

        let best = sums
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let conf = softmax_vec(&sums);
        (best, sums, conf)
    }
}

/// Log-softmax at a single index over the (post-softcap) logits: computed in f64
/// for numerical stability across the 262k-way vocabulary.
fn logprob_of(logits: &[f32], id: u32) -> f64 {
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut sum = 0.0f64;
    for &l in logits {
        sum += ((l as f64) - mx).exp();
    }
    (logits[id as usize] as f64 - mx) - sum.ln()
}

/// Numerically-stable softmax over a small score vector (candidate logprob sums).
fn softmax_vec(xs: &[f32]) -> Vec<f32> {
    let mx = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut e: Vec<f32> = xs.iter().map(|&x| (x - mx).exp()).collect();
    let s: f32 = e.iter().sum();
    if s > 0.0 {
        for v in e.iter_mut() {
            *v /= s;
        }
    }
    e
}

/// Growing per-layer KV cache for incremental decoding. Stores the SAME
/// post-q_norm/k_norm+rope `k` and post-v_norm `v` that `layer` computes for
/// each token, so cached decoding is bit-identical to the full recompute.
/// Keep-all (no window eviction); sliding layers mask by `lo` at attention
/// time — identical results, memory ≈0.67 MB/token across all 48 layers.
pub struct Gemma4Cache {
    k: Vec<Vec<f32>>, // per layer: len * n_kv(l) * head_dim(l)
    v: Vec<Vec<f32>>,
    len: usize, // tokens processed so far (== absolute position of next token)
}

impl Gemma4Cache {
    pub fn new(m: &Gemma4) -> Gemma4Cache {
        Gemma4Cache {
            k: vec![Vec::new(); m.cfg.n_layers],
            v: vec![Vec::new(); m.cfg.n_layers],
            len: 0,
        }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    /// Deep-copy the cache (per-layer k/v + position) so a prefilled prompt state
    /// can be branched — used by the constrained classifier to score several
    /// candidate labels from the same post-prompt state without re-prefilling.
    pub fn clone_cache(&self) -> Gemma4Cache {
        Gemma4Cache {
            k: self.k.clone(),
            v: self.v.clone(),
            len: self.len,
        }
    }
    /// Max absolute element difference of the stored k/v tensors between two caches
    /// (parity gate for the batched vs sequential prefill). Returns +inf on a shape
    /// mismatch. `(max_k_diff, max_v_diff)`.
    pub fn max_kv_diff(&self, other: &Gemma4Cache) -> (f32, f32) {
        let diff = |a: &[Vec<f32>], b: &[Vec<f32>]| -> f32 {
            if a.len() != b.len() {
                return f32::INFINITY;
            }
            let mut mx = 0.0f32;
            for (la, lb) in a.iter().zip(b.iter()) {
                if la.len() != lb.len() {
                    return f32::INFINITY;
                }
                for (x, y) in la.iter().zip(lb.iter()) {
                    mx = mx.max((x - y).abs());
                }
            }
            mx
        };
        (diff(&self.k, &other.k), diff(&self.v, &other.v))
    }
}

// ============================================================================
// Gemma4Gpu — a fully GPU-resident single-token decoder. Runs the whole forward
// (matmuls + norms + rope + attention + gelu + softcap) in ONE command buffer with
// activations and the KV cache resident on the GPU, so decode is not throttled by
// ~336 per-op CPU<->GPU round trips per token (the reason plain `decode_step` is
// slow). Mirrors qwen35's resident runner; reuses the already-uploaded Weight::Gpu
// matrices. macOS/Metal only, opt-in until proven byte-exact.
// ============================================================================
#[cfg(target_os = "macos")]
pub use resident::Gemma4Gpu;

#[cfg(target_os = "macos")]
pub const GEMMA_MAX_SEQ: usize = 8192;

#[cfg(target_os = "macos")]
mod resident {
    use super::{Gemma4, Gemma4Cache, GEMMA_MAX_SEQ, HEAD_DIM_GLOBAL, INTER};
    use crate::metal_be::{Gpu, GpuMatrix};
    use metal::{
        Buffer, BufferRef, CommandQueue, ComputeCommandEncoderRef, ComputePipelineState, Device,
        MTLResourceOptions, MTLSize, ResourceRef,
    };
    use std::ffi::c_void;

    pub struct Gemma4Gpu {
        p_mv_f16: ComputePipelineState,
        p_mv_q4k: ComputePipelineState,
        p_mv_q5k: ComputePipelineState,
        p_mv_q6k: ComputePipelineState,
        p_mv_q8: ComputePipelineState,
        p_rms: ComputePipelineState,
        p_rms_h: ComputePipelineState,
        p_rope: ComputePipelineState,
        p_attn: ComputePipelineState,
        p_store: ComputePipelineState,
        p_gelu: ComputePipelineState,
        p_scalar: ComputePipelineState,
        p_softcap: ComputePipelineState,
        p_add: ComputePipelineState,
        p_copy: ComputePipelineState,
        // activations (sized for the largest layer)
        x: Buffer,
        a: Buffer,
        pf: Buffer,
        t1: Buffer,
        t2: Buffer,
        q: Buffer,
        k: Buffer,
        v: Buffer,
        att: Buffer,
        gbuf: Buffer,
        ubuf: Buffer,
        logits: Buffer,
        // per-layer params
        input_ln: Vec<Buffer>,
        post_attn_ln: Vec<Buffer>,
        pre_ff_ln: Vec<Buffer>,
        post_ff_ln: Vec<Buffer>,
        q_norm: Vec<Buffer>,
        k_norm: Vec<Buffer>,
        inv_freq: Vec<Buffer>,
        kcache: Vec<Buffer>,
        vcache: Vec<Buffer>,
        final_norm: Buffer,
        ones: Buffer, // all-1 weights so rmsnorm_heads acts as scale-free v-norm
        device: Device,
        queue: CommandQueue,
    }

    impl Gemma4Gpu {
        pub fn new(gpu: &Gpu, m: &Gemma4) -> Gemma4Gpu {
            let device = gpu.device().clone();
            let queue = gpu.queue().clone();
            let lib = gpu.fused_library();
            let pl = |n: &str| {
                device
                    .new_compute_pipeline_state_with_function(&lib.get_function(n, None).unwrap())
                    .unwrap()
            };
            let c = &m.cfg;
            let nb = |n: usize| {
                device.new_buffer((n * 4).max(4) as u64, MTLResourceOptions::StorageModeShared)
            };
            let up = |d: &[f32]| {
                device.new_buffer_with_data(
                    d.as_ptr() as *const c_void,
                    (d.len() * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            };
            let max_q = c.n_heads * HEAD_DIM_GLOBAL; // 16*512
            let mut input_ln = Vec::new();
            let mut post_attn_ln = Vec::new();
            let mut pre_ff_ln = Vec::new();
            let mut post_ff_ln = Vec::new();
            let mut q_norm = Vec::new();
            let mut k_norm = Vec::new();
            let mut inv_freq = Vec::new();
            let mut kcache = Vec::new();
            let mut vcache = Vec::new();
            for lw in &m.layers {
                input_ln.push(up(&lw.input_ln));
                post_attn_ln.push(up(&lw.post_attn_ln));
                pre_ff_ln.push(up(&lw.pre_ff_ln));
                post_ff_ln.push(up(&lw.post_ff_ln));
                q_norm.push(up(&lw.q_norm));
                k_norm.push(up(&lw.k_norm));
                inv_freq.push(up(if lw.global {
                    &m.inv_freq_global
                } else {
                    &m.inv_freq_local
                }));
                let kv_dim = lw.n_kv() * lw.head_dim();
                kcache.push(nb(GEMMA_MAX_SEQ * kv_dim));
                vcache.push(nb(GEMMA_MAX_SEQ * kv_dim));
            }
            eprintln!("[gemma4] GPU-resident decoder ready");
            Gemma4Gpu {
                p_mv_f16: pl("matvec_f16"),
                p_mv_q4k: pl(crate::metal_be::q4k_gemv_name()),
                p_mv_q5k: pl("matvec_q5k_co"),
                p_mv_q6k: pl("matvec_q6k_co"),
                p_mv_q8: pl("matvec_q8_0_co"),
                p_rms: pl("rmsnorm"),
                p_rms_h: pl("rmsnorm_heads"),
                p_rope: pl("rope_table"),
                p_attn: pl("gemma_attn"),
                p_store: pl("store_kv"),
                p_gelu: pl("gelu_mul"),
                p_scalar: pl("scalar_mul"),
                p_softcap: pl("softcap"),
                p_add: pl("add_inplace"),
                p_copy: pl("copy_buf"),
                x: nb(c.hidden),
                a: nb(c.hidden),
                pf: nb(c.hidden),
                t1: nb(c.hidden),
                t2: nb(c.hidden),
                q: nb(max_q),
                k: nb(2048),
                v: nb(2048),
                att: nb(max_q),
                gbuf: nb(INTER),
                ubuf: nb(INTER),
                logits: nb(c.vocab),
                input_ln,
                post_attn_ln,
                pre_ff_ln,
                post_ff_ln,
                q_norm,
                k_norm,
                inv_freq,
                kcache,
                vcache,
                final_norm: up(&m.final_norm),
                ones: up(&vec![1.0f32; HEAD_DIM_GLOBAL]),
                device,
                queue,
            }
        }

        // Order dependent dispatches: one command buffer runs kernels concurrently
        // unless a barrier forces the next to see the previous writes.
        fn bar(enc: &ComputeCommandEncoderRef, outs: &[&BufferRef]) {
            let r: Vec<&ResourceRef> = outs.iter().map(|&b| b as &ResourceRef).collect();
            enc.memory_barrier_with_resources(&r);
        }
        fn set_u(enc: &ComputeCommandEncoderRef, i: u64, v: u32) {
            enc.set_bytes(i, 4, &v as *const u32 as *const c_void);
        }
        fn set_f(enc: &ComputeCommandEncoderRef, i: u64, v: f32) {
            enc.set_bytes(i, 4, &v as *const f32 as *const c_void);
        }
        // y = W x, coalesced K-quant / scalar f16, matching the CPU matvec.
        fn mv(&self, enc: &ComputeCommandEncoderRef, w: &GpuMatrix, x: &BufferRef, y: &BufferRef) {
            use crate::gguf::{GGML_F16, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0};
            let (p, co) = match w.ggml_type {
                GGML_Q4_K => (&self.p_mv_q4k, true),
                GGML_Q5_K => (&self.p_mv_q5k, true),
                GGML_Q6_K => (&self.p_mv_q6k, true),
                GGML_Q8_0 => (&self.p_mv_q8, true),
                GGML_F16 => (&self.p_mv_f16, false),
                t => {
                    eprintln!("[gemma4] no resident matvec for type {t}");
                    std::process::exit(2);
                }
            };
            enc.set_compute_pipeline_state(p);
            enc.set_buffer(0, Some(w.buffer()), 0);
            enc.set_buffer(1, Some(x), 0);
            enc.set_buffer(2, Some(y), 0);
            Self::set_u(enc, 3, w.in_dim as u32);
            if co {
                let ndst: u64 = if w.ggml_type == GGML_Q4_K {
                    crate::metal_be::q4k_ndst()
                } else {
                    2
                };
                Self::set_u(enc, 4, w.n_rows as u32);
                let sg = (w.n_rows as u64).div_ceil(ndst);
                enc.dispatch_threads(MTLSize::new(sg * 32, 1, 1), MTLSize::new(32, 1, 1));
            } else {
                enc.dispatch_threads(
                    MTLSize::new(w.n_rows as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            }
            Self::bar(enc, &[y]);
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
        fn rms_heads(
            &self,
            enc: &ComputeCommandEncoderRef,
            x: &BufferRef,
            w: &BufferRef,
            heads: usize,
            hd: usize,
            eps: f32,
        ) {
            enc.set_compute_pipeline_state(&self.p_rms_h);
            enc.set_buffer(0, Some(x), 0);
            enc.set_buffer(1, Some(w), 0);
            Self::set_u(enc, 2, hd as u32);
            Self::set_f(enc, 3, eps);
            enc.dispatch_thread_groups(MTLSize::new(heads as u64, 1, 1), MTLSize::new(256, 1, 1));
            Self::bar(enc, &[x]);
        }
        fn elt(&self, enc: &ComputeCommandEncoderRef, p: &ComputePipelineState, a: &BufferRef, b: Option<&BufferRef>, n: usize) {
            enc.set_compute_pipeline_state(p);
            enc.set_buffer(0, Some(a), 0);
            if let Some(bb) = b {
                enc.set_buffer(1, Some(bb), 0);
            }
            enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(256, 1, 1));
            Self::bar(enc, &[a]);
        }

        /// One decode step: token at absolute `pos`, returns post-softcap logits.
        pub fn forward(&self, m: &Gemma4, id: u32, pos: usize) -> Vec<f32> {
            let c = &m.cfg;
            let hidden = c.hidden;
            let nh = c.n_heads;
            let eps = c.rms_eps;
            // embed row * embed_scale -> x
            let row = m.embed_row(id);
            let scaled: Vec<f32> = row.iter().map(|&x| x * m.embed_scale).collect();
            unsafe {
                std::ptr::copy_nonoverlapping(scaled.as_ptr(), self.x.contents() as *mut f32, hidden);
            }
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            for (l, lw) in m.layers.iter().enumerate() {
                let hd = lw.head_dim();
                let n_kv = lw.n_kv();
                let kv_dim = n_kv * hd;
                let q_dim = nh * hd;
                let kv_mul = (nh / n_kv) as u32;
                let half = hd / 2;
                // input norm -> a ; projections
                self.rms(enc, &self.x, &self.input_ln[l], &self.a, hidden, eps);
                self.mv(enc, lw.wq.as_gpu(), &self.a, &self.q);
                self.mv(enc, lw.wk.as_gpu(), &self.a, &self.k);
                if lw.global {
                    // k_eq_v: v = k source (before k-norm/rope). copy_buf is dst<-src,
                    // and elt maps (arg0=buffer0=src, arg1=buffer1=dst): src=k, dst=v.
                    self.elt(enc, &self.p_copy, &self.k, Some(&self.v), kv_dim);
                } else {
                    self.mv(enc, lw.wv.as_ref().unwrap().as_gpu(), &self.a, &self.v);
                }
                // per-head q/k norm (before rope), v-norm (scale-free)
                self.rms_heads(enc, &self.q, &self.q_norm[l], nh, hd, eps);
                self.rms_heads(enc, &self.k, &self.k_norm[l], n_kv, hd, eps);
                self.rms_heads(enc, &self.v, &self.ones, n_kv, hd, eps);
                // rope q,k
                for (buf, heads) in [(&self.q, nh), (&self.k, n_kv)] {
                    enc.set_compute_pipeline_state(&self.p_rope);
                    enc.set_buffer(0, Some(buf), 0);
                    enc.set_buffer(1, Some(&self.inv_freq[l]), 0);
                    Self::set_u(enc, 2, half as u32);
                    Self::set_u(enc, 3, pos as u32);
                    enc.dispatch_threads(
                        MTLSize::new((heads * half) as u64, 1, 1),
                        MTLSize::new(64, 1, 1),
                    );
                    Self::bar(enc, &[buf]);
                }
                // store k,v then attend
                enc.set_compute_pipeline_state(&self.p_store);
                enc.set_buffer(0, Some(&self.k), 0);
                enc.set_buffer(1, Some(&self.v), 0);
                enc.set_buffer(2, Some(&self.kcache[l]), 0);
                enc.set_buffer(3, Some(&self.vcache[l]), 0);
                Self::set_u(enc, 4, kv_dim as u32);
                Self::set_u(enc, 5, pos as u32);
                enc.dispatch_threads(MTLSize::new(kv_dim as u64, 1, 1), MTLSize::new(256, 1, 1));
                Self::bar(enc, &[&self.kcache[l], &self.vcache[l]]);
                let lo = if lw.global {
                    0u32
                } else {
                    (pos as u32 + 1).saturating_sub(c.sliding_window as u32)
                };
                enc.set_compute_pipeline_state(&self.p_attn);
                enc.set_buffer(0, Some(&self.q), 0);
                enc.set_buffer(1, Some(&self.kcache[l]), 0);
                enc.set_buffer(2, Some(&self.vcache[l]), 0);
                enc.set_buffer(3, Some(&self.att), 0);
                Self::set_u(enc, 4, hd as u32);
                Self::set_u(enc, 5, kv_dim as u32);
                Self::set_u(enc, 6, kv_mul);
                Self::set_u(enc, 7, pos as u32);
                Self::set_u(enc, 8, lo);
                Self::set_f(enc, 9, 1.0); // Gemma attention scale = 1.0
                enc.dispatch_thread_groups(MTLSize::new(nh as u64, 1, 1), MTLSize::new(hd as u64, 1, 1));
                Self::bar(enc, &[&self.att]);
                let _ = q_dim;
                // o_proj -> t1 ; post_attn_ln -> t2 ; x += t2
                self.mv(enc, lw.wo.as_gpu(), &self.att, &self.t1);
                self.rms(enc, &self.t1, &self.post_attn_ln[l], &self.t2, hidden, eps);
                self.elt(enc, &self.p_add, &self.x, Some(&self.t2), hidden);
                // MLP
                self.rms(enc, &self.x, &self.pre_ff_ln[l], &self.pf, hidden, eps);
                self.mv(enc, lw.gate.as_gpu(), &self.pf, &self.gbuf);
                self.mv(enc, lw.up.as_gpu(), &self.pf, &self.ubuf);
                self.elt(enc, &self.p_gelu, &self.gbuf, Some(&self.ubuf), c.inter);
                self.mv(enc, lw.down.as_gpu(), &self.gbuf, &self.t1);
                self.rms(enc, &self.t1, &self.post_ff_ln[l], &self.t2, hidden, eps);
                self.elt(enc, &self.p_add, &self.x, Some(&self.t2), hidden);
                // per-layer residual scalar
                enc.set_compute_pipeline_state(&self.p_scalar);
                enc.set_buffer(0, Some(&self.x), 0);
                Self::set_f(enc, 1, lw.layer_scalar);
                enc.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256, 1, 1));
                Self::bar(enc, &[&self.x]);
            }
            // final norm + tied lm-head + softcap
            self.rms(enc, &self.x, &self.final_norm, &self.t2, hidden, eps);
            self.mv(enc, m.embed.as_gpu(), &self.t2, &self.logits);
            enc.set_compute_pipeline_state(&self.p_softcap);
            enc.set_buffer(0, Some(&self.logits), 0);
            Self::set_f(enc, 1, c.final_softcap);
            enc.dispatch_threads(MTLSize::new(c.vocab as u64, 1, 1), MTLSize::new(256, 1, 1));
            Self::bar(enc, &[&self.logits]);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            unsafe { std::slice::from_raw_parts(self.logits.contents() as *const f32, c.vocab) }.to_vec()
        }

        /// Seed the GPU KV cache from a CPU-prefilled `Gemma4Cache` (so a fast batched
        /// CPU prefill can hand off to resident decode). Layouts match store_kv.
        pub fn upload_state(&self, m: &Gemma4, cache: &Gemma4Cache) {
            for (l, lw) in m.layers.iter().enumerate() {
                let kv_dim = lw.n_kv() * lw.head_dim();
                let n = cache.len * kv_dim;
                if n == 0 {
                    continue;
                }
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        cache.k[l].as_ptr(),
                        self.kcache[l].contents() as *mut f32,
                        n.min(cache.k[l].len()),
                    );
                    std::ptr::copy_nonoverlapping(
                        cache.v[l].as_ptr(),
                        self.vcache[l].contents() as *mut f32,
                        n.min(cache.v[l].len()),
                    );
                }
            }
        }
    }
}

/// argmax + top-k helper (id, logit), descending.
// ---- portable `.hos` capsule (ingest / load) ----------------------------
// Gemma-4 is its own arch, so it serializes ITS tensors (already-quantized big
// linears + f32 norms + tokenizer + cfg) into a `.hos` with `arch=gemma4`, and
// reconstructs from it without the source safetensors dir. Weights are stored at
// their resident dtype+bytes (no dequant), so a 12B capsule ≈ the .hosw size.

fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

fn weight_to_raw(name: String, role: u8, w: &Weight) -> crate::format::RawTensor {
    use crate::format::{ggml_to_dtype, RawTensor, DTYPE_E8, DTYPE_F32, DTYPE_HQ4};
    match w {
        Weight::Quant {
            bytes,
            ggml_type,
            rows,
            cols,
        } => RawTensor {
            name,
            role,
            shape: vec![*rows, *cols],
            dtype: if *ggml_type == crate::model::HQ4_TYPE {
                DTYPE_HQ4
            } else if *ggml_type == crate::model::E8_TYPE {
                DTYPE_E8
            } else {
                ggml_to_dtype(*ggml_type)
            },
            nfloats: rows * cols,
            bytes: bytes.clone(),
        },
        Weight::Cpu { data, rows, cols } => RawTensor {
            name,
            role,
            shape: vec![*rows, *cols],
            dtype: DTYPE_F32,
            nfloats: data.len(),
            bytes: f32_to_bytes(data),
        },
        Weight::Gpu(_) => panic!("write_capsule: serialize before GPU upload"),
    }
}

fn vec_to_raw(name: String, role: u8, v: &[f32]) -> crate::format::RawTensor {
    crate::format::RawTensor {
        name,
        role,
        shape: vec![v.len()],
        dtype: crate::format::DTYPE_F32,
        nfloats: v.len(),
        bytes: f32_to_bytes(v),
    }
}

fn raw_to_weight(r: crate::format::RawTensor, cols: usize) -> crate::Result<Weight> {
    use crate::format::{decode_raw, dtype_to_ggml, DTYPE_E8, DTYPE_F32, DTYPE_HQ4};
    let rows = if cols == 0 {
        r.nfloats
    } else {
        r.nfloats / cols
    };
    if r.dtype == DTYPE_HQ4 {
        Ok(Weight::Quant {
            bytes: r.bytes,
            ggml_type: crate::model::HQ4_TYPE,
            rows,
            cols,
        })
    } else if r.dtype == DTYPE_E8 {
        Ok(Weight::Quant {
            bytes: r.bytes,
            ggml_type: crate::model::E8_TYPE,
            rows,
            cols,
        })
    } else if let Some(gt) = dtype_to_ggml(r.dtype) {
        Ok(Weight::Quant {
            bytes: r.bytes,
            ggml_type: gt,
            rows,
            cols,
        })
    } else if r.dtype == DTYPE_F32 {
        Ok(Weight::cpu(
            r.bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            cols,
        ))
    } else {
        // q3/hq-only-in-capsule etc.: decode to f32 (not matvec-fusable here).
        Ok(Weight::cpu(
            decode_raw(r.dtype, &r.bytes, r.nfloats).map_err(crate::HosError::from)?,
            cols,
        ))
    }
}

fn take(
    map: &mut std::collections::HashMap<String, crate::format::RawTensor>,
    k: &str,
) -> crate::Result<crate::format::RawTensor> {
    map.remove(k)
        .ok_or_else(|| crate::HosError::Format(format!("gemma4 capsule: missing tensor `{k}`")))
}

impl Gemma4 {
    /// Serialize this decoder to a portable, self-contained `.hos` capsule:
    /// already-quantized big linears (no dequant), the f32 norms the `.hosw`
    /// cache omits, cfg in `card.arch`, and the tokenizer JSON in `card.meta`.
    /// Call on a CPU-resident model (before any `upload_to_gpu`).
    pub fn write_capsule(&self, path: &Path, tokenizer_json: &str) -> crate::Result<()> {
        use crate::format::{self, RawTensor, ROLE_EMBED, ROLE_NORM, ROLE_SCALAR, ROLE_WEIGHT};
        let mut raws: Vec<RawTensor> = Vec::new();
        raws.push(weight_to_raw("embed".to_string(), ROLE_EMBED, &self.embed));
        for (l, lw) in self.layers.iter().enumerate() {
            raws.push(weight_to_raw(format!("l{l}.wq"), ROLE_WEIGHT, &lw.wq));
            raws.push(weight_to_raw(format!("l{l}.wk"), ROLE_WEIGHT, &lw.wk));
            if let Some(wv) = &lw.wv {
                raws.push(weight_to_raw(format!("l{l}.wv"), ROLE_WEIGHT, wv));
            }
            raws.push(weight_to_raw(format!("l{l}.wo"), ROLE_WEIGHT, &lw.wo));
            raws.push(weight_to_raw(format!("l{l}.gate"), ROLE_WEIGHT, &lw.gate));
            raws.push(weight_to_raw(format!("l{l}.up"), ROLE_WEIGHT, &lw.up));
            raws.push(weight_to_raw(format!("l{l}.down"), ROLE_WEIGHT, &lw.down));
            raws.push(vec_to_raw(
                format!("l{l}.input_ln"),
                ROLE_NORM,
                &lw.input_ln,
            ));
            raws.push(vec_to_raw(
                format!("l{l}.post_attn_ln"),
                ROLE_NORM,
                &lw.post_attn_ln,
            ));
            raws.push(vec_to_raw(
                format!("l{l}.pre_ff_ln"),
                ROLE_NORM,
                &lw.pre_ff_ln,
            ));
            raws.push(vec_to_raw(
                format!("l{l}.post_ff_ln"),
                ROLE_NORM,
                &lw.post_ff_ln,
            ));
            raws.push(vec_to_raw(format!("l{l}.q_norm"), ROLE_NORM, &lw.q_norm));
            raws.push(vec_to_raw(format!("l{l}.k_norm"), ROLE_NORM, &lw.k_norm));
            raws.push(vec_to_raw(
                format!("l{l}.layer_scalar"),
                ROLE_SCALAR,
                &[lw.layer_scalar],
            ));
        }
        raws.push(vec_to_raw(
            "final_norm".to_string(),
            ROLE_NORM,
            &self.final_norm,
        ));

        let arch = serde_json::json!({
            "architecture": "gemma4",
            "hidden": self.cfg.hidden,
            "n_layers": self.cfg.n_layers,
            "n_heads": self.cfg.n_heads,
            "inter": self.cfg.inter,
            "vocab": self.cfg.vocab,
            "rms_eps": self.cfg.rms_eps,
            "sliding_window": self.cfg.sliding_window,
            "final_softcap": self.cfg.final_softcap,
        });
        let mut card = format::Card::new("gemma4", arch);
        card.mode = "inference".to_string();
        card.meta = serde_json::json!({ "gemma4.tokenizer_json": tokenizer_json });
        format::save_raw(path, &raws, &card).map_err(crate::HosError::from)
    }

    /// Reconstruct a Gemma-4 decoder + tokenizer from a `.hos` capsule written by
    /// [`write_capsule`] — no safetensors dir needed. Derived state (RoPE tables,
    /// embed scale) is regenerated; the f32 lookup rows are dequantized from the
    /// stored quantized embed.
    pub fn from_capsule(path: &Path) -> crate::Result<(Gemma4, crate::gemma_tok::GemmaTokenizer)> {
        use std::collections::HashMap;
        let (raws, card) = crate::format::load_raw(path).map_err(crate::HosError::from)?;
        let mut map: HashMap<String, crate::format::RawTensor> =
            raws.into_iter().map(|r| (r.name.clone(), r)).collect();
        let cfg = Cfg::default();

        let f32vec = |r: crate::format::RawTensor| -> crate::Result<Vec<f32>> {
            crate::format::decode_raw(r.dtype, &r.bytes, r.nfloats).map_err(crate::HosError::from)
        };

        let embed = raw_to_weight(take(&mut map, "embed")?, cfg.hidden)?;
        // Regenerate the full-precision lookup rows by dequantizing the stored
        // quantized embed (avoids storing a second ~4GB f32 copy in the capsule).
        let embed_cpu = match &embed {
            Weight::Quant { .. } => Some(embed.to_f32()),
            _ => None,
        };
        let final_norm = f32vec(take(&mut map, "final_norm")?)?;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            let g = is_global(l);
            let hd = if g { HEAD_DIM_GLOBAL } else { HEAD_DIM_SLIDING };
            let o_in = cfg.n_heads * hd;
            let wv = if map.contains_key(&format!("l{l}.wv")) {
                Some(raw_to_weight(
                    take(&mut map, &format!("l{l}.wv"))?,
                    cfg.hidden,
                )?)
            } else {
                None
            };
            layers.push(LayerW {
                global: g,
                input_ln: f32vec(take(&mut map, &format!("l{l}.input_ln"))?)?,
                post_attn_ln: f32vec(take(&mut map, &format!("l{l}.post_attn_ln"))?)?,
                pre_ff_ln: f32vec(take(&mut map, &format!("l{l}.pre_ff_ln"))?)?,
                post_ff_ln: f32vec(take(&mut map, &format!("l{l}.post_ff_ln"))?)?,
                q_norm: f32vec(take(&mut map, &format!("l{l}.q_norm"))?)?,
                k_norm: f32vec(take(&mut map, &format!("l{l}.k_norm"))?)?,
                wq: raw_to_weight(take(&mut map, &format!("l{l}.wq"))?, cfg.hidden)?,
                wk: raw_to_weight(take(&mut map, &format!("l{l}.wk"))?, cfg.hidden)?,
                wv,
                wo: raw_to_weight(take(&mut map, &format!("l{l}.wo"))?, o_in)?,
                gate: raw_to_weight(take(&mut map, &format!("l{l}.gate"))?, cfg.hidden)?,
                up: raw_to_weight(take(&mut map, &format!("l{l}.up"))?, cfg.hidden)?,
                down: raw_to_weight(take(&mut map, &format!("l{l}.down"))?, cfg.inter)?,
                layer_scalar: f32vec(take(&mut map, &format!("l{l}.layer_scalar"))?)?[0],
            });
        }

        let inv_freq_local: Vec<f32> = (0..HEAD_DIM_SLIDING / 2)
            .map(|j| (ROPE_THETA_LOCAL.powf(-2.0 * j as f64 / HEAD_DIM_SLIDING as f64)) as f32)
            .collect();
        let inv_freq_global: Vec<f32> = (0..HEAD_DIM_GLOBAL / 2)
            .map(|j| {
                if j < 64 {
                    (ROPE_THETA_GLOBAL.powf(-2.0 * j as f64 / HEAD_DIM_GLOBAL as f64)) as f32
                } else {
                    0.0
                }
            })
            .collect();
        let embed_scale = bf16((cfg.hidden as f32).sqrt());
        let bf16_emul = !std::env::var("HOS_GEMMA4_F32").is_ok_and(|v| v != "0" && !v.is_empty());

        let vocab = cfg.vocab;
        let n_layers = cfg.n_layers;
        let m = Gemma4 {
            cfg,
            embed,
            embed_cpu,
            final_norm,
            layers,
            embed_scale,
            inv_freq_local,
            inv_freq_global,
            bf16_emul,
        };

        let tok_json = card
            .meta
            .get("gemma4.tokenizer_json")
            .and_then(|v| v.as_str())
            .ok_or_else(|| crate::HosError::Format("gemma4 capsule: missing tokenizer".into()))?;
        let jv: serde_json::Value = serde_json::from_str(tok_json)
            .map_err(|e| crate::HosError::Format(format!("gemma4 capsule tokenizer json: {e}")))?;
        let tok = crate::gemma_tok::GemmaTokenizer::from_value(&jv)?;
        eprintln!("[gemma4] loaded {n_layers} layers from .hos capsule, vocab {vocab}");
        Ok((m, tok))
    }
}

pub fn topk(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    idx.into_iter()
        .take(k)
        .map(|i| (i as u32, logits[i]))
        .collect()
}
