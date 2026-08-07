//! AWQ-lite — activation-aware weight quantization.
//!
//! ggml's quants minimise *weight* error. The trick that beats them is to
//! minimise *output* error: weight channels that see large activations matter
//! more, so protect them. We scale each salient input channel **up** before
//! quantizing (so it lands on finer effective levels) and fold the inverse into
//! the preceding RMSNorm weight — which stays f32, so the math is *exact* and
//! the engine's forward/decode never change. Only the encoder sees the salience.
//!
//! `value seen by the linear` = rmsnorm(x) · norm_w. Set `norm_w' = norm_w / s`
//! and `W' = W · diag(s)`; then `W'·(input/s) = W·input` exactly, but `W'` has
//! its salient columns enlarged, so quantizing `W'` protects them.

use crate::format::Named;
use crate::forward::Calib;
use std::path::Path;

/// A compact, general calibration corpus (disjoint from typical eval text) used
/// when a caller doesn't supply its own.
pub const CALIB_CORPUS: &str = "Photosynthesis converts sunlight, water, and carbon dioxide \
into glucose and oxygen inside the chloroplasts of green plants. The mitochondria release \
energy by breaking those sugars back down. Rivers carve valleys over millennia, depositing \
sediment that builds fertile deltas at the coast. Trade routes once carried silk, spices, \
and silver across deserts and seas, binding distant cities into a single economy.";

/// Per-channel salience scale from accumulated |activation|: `s_c = mean|x_c|^α`,
/// normalised to geometric-mean 1 so the fold stays balanced around unity.
fn salience(act_sum: &[f64], tokens: usize, alpha: f32) -> Vec<f32> {
    let tk = tokens.max(1) as f64;
    let mut s: Vec<f32> = act_sum
        .iter()
        .map(|&a| ((a / tk) as f32).max(1e-8).powf(alpha))
        .collect();
    let logmean = s.iter().map(|v| v.ln()).sum::<f32>() / s.len().max(1) as f32;
    let g = logmean.exp();
    if g > 0.0 {
        for v in &mut s {
            *v /= g;
        }
    }
    s
}

fn find<'a>(named: &'a mut [Named], name: &str) -> Option<&'a mut Named> {
    named.iter_mut().find(|t| t.name == name)
}

fn has(named: &[Named], name: &str) -> bool {
    named.iter().any(|t| t.name == name)
}

/// Output dim (rows) of a `[out, in]` tensor, by name.
fn out_dim(named: &[Named], name: &str) -> Option<usize> {
    named
        .iter()
        .find(|t| t.name == name)
        .and_then(|t| t.shape.first().copied())
}

/// Input dim (cols) of a `[out, in]` tensor, by name.
fn in_dim(named: &[Named], name: &str) -> Option<usize> {
    named
        .iter()
        .find(|t| t.name == name)
        .and_then(|t| t.shape.last().copied())
}

/// Divide a 1-D norm weight by `s` (folds the inverse scale in).
fn fold_norm(named: &mut [Named], name: &str, s: &[f32]) {
    if let Some(t) = find(named, name) {
        if t.data.len() == s.len() {
            for (w, &sc) in t.data.iter_mut().zip(s) {
                *w /= sc;
            }
        }
    }
}

/// Multiply each input column `c` of a `[out, in]` weight by `s[c]`.
fn scale_cols(named: &mut [Named], name: &str, s: &[f32]) {
    if let Some(t) = find(named, name) {
        let inn = s.len();
        if t.data.len() % inn == 0 {
            for row in t.data.chunks_mut(inn) {
                for (w, &sc) in row.iter_mut().zip(s) {
                    *w *= sc;
                }
            }
        }
    }
}

/// Multiply each output row `r` of a `[out, in]` weight by `f[r]`.
fn scale_rows(named: &mut [Named], name: &str, f: &[f32]) {
    if let Some(t) = find(named, name) {
        let out = f.len();
        if out > 0 && t.data.len() % out == 0 {
            let inn = t.data.len() / out;
            for (r, row) in t.data.chunks_mut(inn).enumerate() {
                let fr = f[r];
                for w in row {
                    *w *= fr;
                }
            }
        }
    }
}

/// Apply AWQ-lite to GGUF-named f32 tensors in place, using calibrated salience.
/// Targets the attention input (wq/wk/wv via attn_norm) and FFN input (gate/up
/// via ffn_norm) — the bulk of the parameters. `down`/`wo` are left as-is (their
/// inputs have no preceding norm to fold into). Assumes plain RMSNorm
/// (Llama/Qwen/Mistral); other norm conventions are skipped by name mismatch.
/// `wo_ok` enables the attention-output (`wo`) fold, which is exact only when
/// heads == kv_heads (no GQA) — under GQA a shared value channel feeds several
/// output channels with differing saliences, so the fold isn't clean. Pass
/// `false` for GQA models.
pub fn apply_awq(named: &mut [Named], calib: &Calib, alpha: f32, wo_ok: bool) -> usize {
    let mut touched = 0;
    for l in 0..calib.attn_in.len() {
        // Each fold is ATOMIC: we touch the norm only when its consumer
        // projections actually exist by name. Models that store fused tensors
        // (e.g. `attn_qkv`/`ffn_gate_up`) simply skip the fold rather than get
        // a folded norm with un-scaled weights (which would corrupt them).
        if has(named, &format!("blk.{l}.attn_q.weight")) {
            let s_attn = salience(&calib.attn_in[l], calib.tokens, alpha);
            fold_norm(named, &format!("blk.{l}.attn_norm.weight"), &s_attn);
            for w in ["attn_q", "attn_k", "attn_v"] {
                scale_cols(named, &format!("blk.{l}.{w}.weight"), &s_attn);
                touched += 1;
            }
        }
        // wo (attention output): input is the attention context, no preceding
        // norm. V is linear (no RoPE), so dividing wv's output rows by s and
        // scaling wo's input columns by s is exact when heads == kv_heads.
        if wo_ok {
            if let Some(ao) = calib.attn_out.get(l).filter(|v| !v.is_empty()) {
                let (wv, wo) = (
                    format!("blk.{l}.attn_v.weight"),
                    format!("blk.{l}.attn_output.weight"),
                );
                let n = ao.len();
                // exact only if wv's rows and wo's cols both equal the context dim
                if out_dim(named, &wv) == Some(n) && in_dim(named, &wo) == Some(n) {
                    let s_wo = salience(ao, calib.tokens, alpha);
                    let inv: Vec<f32> = s_wo.iter().map(|&s| 1.0 / s).collect();
                    scale_rows(named, &wv, &inv);
                    fold_norm(named, &format!("blk.{l}.attn_v.bias"), &s_wo); // bias ÷ s
                    scale_cols(named, &wo, &s_wo);
                    touched += 1;
                }
            }
        }
        if has(named, &format!("blk.{l}.ffn_gate.weight")) {
            let s_ffn = salience(&calib.ffn_in[l], calib.tokens, alpha);
            fold_norm(named, &format!("blk.{l}.ffn_norm.weight"), &s_ffn);
            for w in ["ffn_gate", "ffn_up"] {
                scale_cols(named, &format!("blk.{l}.{w}.weight"), &s_ffn);
                touched += 1;
            }
        }
        // down_proj: its input is silu(gate)·up, with no preceding norm. up is
        // linear, so dividing up's output row r by s_r divides intermediate[r]
        // by s_r (silu(gate[r]) is untouched); scaling down's column r by s_r
        // cancels it — exact.
        if let Some(inter) = calib.ffn_inter.get(l).filter(|v| !v.is_empty()) {
            let (up, down) = (
                format!("blk.{l}.ffn_up.weight"),
                format!("blk.{l}.ffn_down.weight"),
            );
            let n = inter.len();
            // exact only if up's rows and down's cols both equal the intermediate
            // dim — skips fused gate_up tensors ([2·ffn, dim]) cleanly.
            if out_dim(named, &up) == Some(n) && in_dim(named, &down) == Some(n) {
                let s_down = salience(inter, calib.tokens, alpha);
                let inv: Vec<f32> = s_down.iter().map(|&s| 1.0 / s).collect();
                scale_rows(named, &up, &inv);
                scale_cols(named, &down, &s_down);
                touched += 1;
            }
        }
    }
    touched
}

/// Full AWQ-lite pipeline: calibrate on `calib_text`, activation-scale the
/// weights (folds are exact, so the result is a standard runnable capsule), and
/// write a quantized `.hos` to `out`. `ggml_type` is the block format the
/// conditioned weights are packed into. Returns the number of matrices scaled.
/// Shared by the `hos --quant-awq` command and `flwr quantize --awq`.
pub fn awq_quantize_gguf(
    src: &Path,
    out: &Path,
    ggml_type: u32,
    alpha: f32,
    calib_text: &str,
) -> crate::Result<usize> {
    use crate::format::{self, Card, Named, ROLE_BIAS, ROLE_EMBED, ROLE_NORM, ROLE_WEIGHT};
    use crate::gguf::Gguf;

    // 1. calibrate (CPU forward, so the instrumented path runs).
    let mut eng = crate::Engine::load(src, false)?;
    let g = Gguf::open(src)?;
    let arch_name = g
        .meta_str("general.architecture")
        .unwrap_or("unknown")
        .to_string();
    let mu = |k: &str| g.meta_u64(&format!("{arch_name}.{k}"));
    let n_layers = mu("block_count").unwrap_or(0) as usize;
    let dim = mu("embedding_length").unwrap_or(0) as usize;
    let ffn = mu("feed_forward_length").unwrap_or(0) as usize;
    let nh = mu("attention.head_count").unwrap_or(0);
    let nkv = mu("attention.head_count_kv").unwrap_or(nh);
    let wo_ok = nh != 0 && nh == nkv;
    let ids = eng.tok.encode(calib_text, true);
    crate::forward::calib_start(n_layers, dim, ffn);
    let _ = eng.perplexity(&ids);
    let calib = crate::forward::calib_take().expect("calibration recorded");

    // 2. build f32 tensors from the GGUF source.
    let role = |n: &str| -> u8 {
        if n.contains("token_embd") || n.contains("embed") {
            ROLE_EMBED
        } else if n.ends_with(".bias") {
            ROLE_BIAS
        } else if n.contains("norm") {
            ROLE_NORM
        } else {
            ROLE_WEIGHT
        }
    };
    let mut names: Vec<String> = g.tensors.keys().cloned().collect();
    names.sort();
    let mut named = Vec::with_capacity(names.len());
    for name in names {
        let data = g.dequant(&name)?;
        let shape: Vec<usize> = g.tensors[&name]
            .dims
            .iter()
            .rev()
            .map(|d| *d as usize)
            .collect();
        named.push(Named {
            role: role(&name),
            name,
            shape,
            data,
        });
    }

    // 3. apply AWQ.
    let touched = apply_awq(&mut named, &calib, alpha, wo_ok);

    // 4. runnable card carrying the full GGUF metadata.
    let arch = serde_json::json!({
        "source_format": "gguf", "architecture": arch_name,
        "embedding_length": mu("embedding_length"), "block_count": mu("block_count"),
        "head_count": mu("attention.head_count"), "head_count_kv": mu("attention.head_count_kv"),
        "feed_forward_length": mu("feed_forward_length"), "context_length": mu("context_length"),
    });
    let model_name = out.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
    let mut card = Card::new(model_name, arch);
    card.id = format::model_id(&named);
    card.mode = "inference".into();
    card.provenance.engine = "hos (awq)".into();
    card.provenance.dataset = format!("awq-quantized from {}", src.display());
    let mut meta = serde_json::Map::new();
    for (k, v) in &g.metadata {
        meta.insert(k.clone(), v.to_json());
    }
    card.meta = serde_json::Value::Object(meta);

    // 5. write.
    format::save_quantized_as(out, &named, &card, ggml_type)?;
    Ok(touched)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Input-side fold (attn_norm → wq/wk/wv): scaling a weight's input columns
    /// by `s` and dividing the preceding norm by `s` leaves `norm[c]·W[o,c]`
    /// unchanged. This is why AWQ needs no runtime/decode change — only the
    /// quantizer sees a different (better-conditioned) tensor.
    #[test]
    fn input_fold_exact() {
        let (dim, out) = (4, 3);
        let norm0 = vec![0.7f32, 1.3, 0.4, 2.1];
        let wq0: Vec<f32> = (0..out * dim).map(|i| i as f32 * 0.31 - 1.0).collect();
        let mut named = vec![
            Named {
                name: "blk.0.attn_norm.weight".into(),
                role: 0,
                shape: vec![dim],
                data: norm0.clone(),
            },
            Named {
                name: "blk.0.attn_q.weight".into(),
                role: 0,
                shape: vec![out, dim],
                data: wq0.clone(),
            },
        ];
        let calib = Calib {
            attn_in: vec![vec![3.0, 0.1, 5.0, 0.7]], // non-uniform → real scaling
            ffn_in: vec![vec![1.0; dim]],
            ffn_inter: vec![vec![]],
            attn_out: vec![vec![]],
            tokens: 1,
        };
        apply_awq(&mut named, &calib, 0.5, false);
        let (norm1, wq1) = (&named[0].data, &named[1].data);
        for o in 0..out {
            for c in 0..dim {
                let before = norm0[c] * wq0[o * dim + c];
                let after = norm1[c] * wq1[o * dim + c];
                assert!(
                    (before - after).abs() < 1e-5,
                    "attn fold changed product at ({o},{c})"
                );
            }
        }
        assert!(
            (wq1[0] - wq0[0]).abs() > 1e-6,
            "fold should actually rescale"
        );
    }

    /// Output-side folds (down_proj via up's rows; wo via wv's rows). Input
    /// salience is uniform here so those columns aren't touched, isolating the
    /// output folds. Each preserves an algebraic invariant for any input:
    /// `down[o,c]·up[c,k]` and `wo[o,c]·wv[c,k]`.
    #[test]
    fn output_folds_exact() {
        let (dim, ffn, qd) = (4, 2, 4);
        let up0: Vec<f32> = (0..ffn * dim).map(|i| i as f32 * 0.17 + 0.5).collect();
        let down0: Vec<f32> = (0..dim * ffn).map(|i| i as f32 * -0.23 + 0.9).collect();
        let wv0: Vec<f32> = (0..qd * dim).map(|i| i as f32 * 0.13 - 0.4).collect();
        let wo0: Vec<f32> = (0..dim * qd).map(|i| i as f32 * 0.07 + 0.2).collect();
        let mut named = vec![
            Named {
                name: "blk.0.attn_v.weight".into(),
                role: 0,
                shape: vec![qd, dim],
                data: wv0.clone(),
            },
            Named {
                name: "blk.0.attn_output.weight".into(),
                role: 0,
                shape: vec![dim, qd],
                data: wo0.clone(),
            },
            Named {
                name: "blk.0.ffn_up.weight".into(),
                role: 0,
                shape: vec![ffn, dim],
                data: up0.clone(),
            },
            Named {
                name: "blk.0.ffn_down.weight".into(),
                role: 0,
                shape: vec![dim, ffn],
                data: down0.clone(),
            },
        ];
        let calib = Calib {
            attn_in: vec![vec![1.0; dim]], // uniform → no input-column scaling
            ffn_in: vec![vec![1.0; dim]],
            ffn_inter: vec![vec![5.0, 0.2]],
            attn_out: vec![vec![2.0, 0.3, 4.0, 1.1]],
            tokens: 1,
        };
        apply_awq(&mut named, &calib, 0.5, true);
        let (wv1, wo1, up1, down1) = (
            &named[0].data,
            &named[1].data,
            &named[2].data,
            &named[3].data,
        );
        for c in 0..ffn {
            for o in 0..dim {
                for k in 0..dim {
                    let before = down0[o * ffn + c] * up0[c * dim + k];
                    let after = down1[o * ffn + c] * up1[c * dim + k];
                    assert!(
                        (before - after).abs() < 1e-4,
                        "down fold broke product c{c} o{o} k{k}"
                    );
                }
            }
        }
        for c in 0..qd {
            for o in 0..dim {
                for k in 0..dim {
                    let before = wo0[o * qd + c] * wv0[c * dim + k];
                    let after = wo1[o * qd + c] * wv1[c * dim + k];
                    assert!(
                        (before - after).abs() < 1e-4,
                        "wo fold broke product c{c} o{o} k{k}"
                    );
                }
            }
        }
        assert!(
            (down1[0] - down0[0]).abs() > 1e-6,
            "down fold should rescale"
        );
        assert!((wo1[1] - wo0[1]).abs() > 1e-6, "wo fold should rescale");
    }
}
