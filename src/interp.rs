//! Self-running architecture interpreter — the deep HOS-format moat.
//!
//! A model's `arch` spec (JSON, stored in its `.hos` card) declaratively lists
//! the layers. This interpreter executes that spec against the file's named
//! weights — so a model file *runs itself*, with NO model-specific Rust code.
//! Change the JSON, change the model. That's a self-describing, self-executing
//! artifact — something GGUF/safetensors can't do.
//!
//! Spec shape:
//! ```json
//! { "type": "sequential", "layers": [
//!     {"op": "embedding",    "weight": "tok"},
//!     {"op": "linear",       "weight": "w1", "bias": "b1"},
//!     {"op": "relu"},
//!     {"op": "rmsnorm",      "weight": "norm"},
//!     {"op": "linear",       "weight": "w2"}
//! ]}
//! ```

use std::collections::HashMap;
use std::path::Path;

use serde_json::Value;

use crate::error::{HosError, Result};
use crate::tensor::Tensor;

fn spec(msg: impl Into<String>) -> HosError {
    HosError::Spec(msg.into())
}

/// Run a forward pass from a declarative arch spec + named weight tensors.
/// `ids` feeds an initial `embedding` layer; `input` feeds a tensor-first model.
/// Returns `Err(HosError::Spec)` for a malformed spec or a missing weight,
/// rather than panicking.
pub fn run(
    arch: &Value,
    w: &HashMap<String, Tensor>,
    ids: Option<&[usize]>,
    input: Option<Tensor>,
) -> Result<Tensor> {
    let layers = arch
        .get("layers")
        .and_then(|l| l.as_array())
        .ok_or_else(|| spec("arch.layers must be an array"))?;
    let seq = arch.get("ctx").and_then(|v| v.as_u64()).map(|v| v as usize); // context length for attention
    let mut x = input;
    for layer in layers {
        let op = layer
            .get("op")
            .and_then(|o| o.as_str())
            .ok_or_else(|| spec("each layer needs an \"op\""))?;
        let wt = |key: &str| -> Result<&Tensor> {
            let name = layer
                .get(key)
                .and_then(|v| v.as_str())
                .ok_or_else(|| spec(format!("op '{op}' needs \"{key}\"")))?;
            w.get(name)
                .ok_or_else(|| spec(format!("op '{op}': no weight named '{name}' in the file")))
        };
        let cur = || {
            x.as_ref()
                .ok_or_else(|| spec(format!("op '{op}' has no input tensor yet")))
        };
        let next = match op {
            "embedding" => {
                wt("weight")?.embedding(ids.ok_or_else(|| spec("embedding op needs token ids"))?)
            }
            "linear" => {
                let y = cur()?.matmul(wt("weight")?);
                if layer.get("bias").is_some() {
                    y.add(wt("bias")?)
                } else {
                    y
                }
            }
            "relu" => cur()?.relu(),
            "silu" => cur()?.silu(),
            "rmsnorm" => cur()?.rmsnorm(wt("weight")?),
            "max_pool_rows" => cur()?.max_rows(),
            "softmax_rows" => cur()?.softmax_rows(),
            "scale" => {
                cur()?.scale(layer.get("factor").and_then(|f| f.as_f64()).unwrap_or(1.0) as f32)
            }
            // learned positional embedding added by position (row % seq)
            "add_pos" => {
                let t = seq.ok_or_else(|| spec("add_pos needs arch.ctx"))?;
                let c = cur()?;
                let n = c.shape()[0];
                let pos_ids: Vec<usize> = (0..n).map(|r| r % t).collect();
                c.add(&wt("weight")?.embedding(&pos_ids))
            }
            // x = x + MultiHeadSelfAttention(RMSNorm(x)) — causal, "heads" (default 1)
            "attention_block" => {
                let t = seq.ok_or_else(|| spec("attention_block needs arch.ctx"))?;
                let c = cur()?;
                let (n, d) = (c.shape()[0], c.shape()[1]);
                let bs = n / t;
                let heads = layer.get("heads").and_then(|h| h.as_u64()).unwrap_or(1) as usize;
                let hd = d / heads;
                let inv = 1.0 / (hd as f32).sqrt();
                let xn = c.rmsnorm(wt("norm")?);
                // [N,d] -> [bs,t,heads,hd] -> [bs,heads,t,hd] -> [bs*heads,t,hd]
                let split = |proj: Tensor| {
                    proj.reshape(&[bs, t, heads, hd])
                        .transpose12_4d()
                        .reshape(&[bs * heads, t, hd])
                };
                let q = split(xn.matmul(wt("wq")?));
                let k = split(xn.matmul(wt("wk")?));
                let v = split(xn.matmul(wt("wv")?));
                let scores = q
                    .bmm(&k.transpose_last2())
                    .scale(inv)
                    .reshape(&[bs * heads * t, t]);
                let att = scores
                    .add(&causal_mask(bs * heads, t))
                    .softmax_rows()
                    .reshape(&[bs * heads, t, t]);
                // [bs*heads,t,hd] -> [bs,heads,t,hd] -> [bs,t,heads,hd] -> [N,d]
                let ctxv = att
                    .bmm(&v)
                    .reshape(&[bs, heads, t, hd])
                    .transpose12_4d()
                    .reshape(&[bs * t, d]);
                c.add(&ctxv.matmul(wt("wo")?))
            }
            // x = x + Σ_e router_e · FFN_e(x)  — soft mixture of experts (dense; all experts weighted)
            "moe_ffn" => {
                let c = cur()?;
                let xn = c.rmsnorm(wt("norm")?);
                let experts = layer
                    .get("experts")
                    .and_then(|e| e.as_array())
                    .ok_or_else(|| spec("moe_ffn needs \"experts\" array"))?;
                let router = xn.matmul(wt("gate")?).softmax_rows(); // [N, n_experts]
                let mut acc: Option<Tensor> = None;
                for (e, ex) in experts.iter().enumerate() {
                    let en = |k: &str| -> Result<&Tensor> {
                        let name = ex
                            .get(k)
                            .and_then(|v| v.as_str())
                            .ok_or_else(|| spec(format!("expert needs \"{k}\"")))?;
                        w.get(name)
                            .ok_or_else(|| spec(format!("missing expert weight '{name}'")))
                    };
                    let ff = xn
                        .matmul(en("w1")?)
                        .add(en("b1")?)
                        .relu()
                        .matmul(en("w2")?)
                        .add(en("b2")?);
                    let weighted = ff.mul_broadcast(&router.col(e));
                    acc = Some(match acc {
                        Some(a) => a.add(&weighted),
                        None => weighted,
                    });
                }
                c.add(&acc.ok_or_else(|| spec("moe_ffn needs at least one expert"))?)
            }
            // x = x + FFN(RMSNorm(x))  — relu MLP
            "ffn_block" => {
                let c = cur()?;
                let xn = c.rmsnorm(wt("norm")?);
                let ff = xn
                    .matmul(wt("w1")?)
                    .add(wt("b1")?)
                    .relu()
                    .matmul(wt("w2")?)
                    .add(wt("b2")?);
                c.add(&ff)
            }
            other => return Err(spec(format!("unsupported op '{other}'"))),
        };
        x = Some(next);
    }
    x.ok_or_else(|| spec("arch had no layers"))
}

fn causal_mask(bs: usize, t: usize) -> Tensor {
    let mut m = vec![0.0f32; bs * t * t];
    for r in 0..bs * t {
        let tt = r % t;
        for j in (tt + 1)..t {
            m[r * t + j] = -1e9;
        }
    }
    Tensor::constant(m, &[bs * t, t])
}

/// Load a HOS file and run it purely from its embedded arch spec — the model
/// runs itself, no architecture coded in Rust.
pub fn run_file(path: &Path, ids: Option<&[usize]>, input: Option<Tensor>) -> Result<Tensor> {
    let (named, card) = crate::format::load(path)?;
    let mut w = HashMap::new();
    for n in &named {
        w.insert(n.name.clone(), Tensor::constant(n.data.clone(), &n.shape));
    }
    run(&card.arch, &w, ids, input)
}
// ============================================================================
// M4: spec-driven Llama inference.
//
// The architecture is described entirely by DATA — a JSON spec (dims, rope/norm
// settings) plus the GGUF weight-naming convention — and executed by this one
// generic function. No per-model hand-coded forward: change the spec, run a
// different model. This is the "a model runs itself from its spec" moat made real
// for production transformers (verified bit-close to the hand-coded forward).
// ============================================================================

fn sp_rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let ss = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let s = 1.0 / (ss + eps).sqrt();
    (0..n).map(|i| x[i] * s * w[i]).collect()
}

fn sp_matvec(w: &[f32], x: &[f32], rows: usize) -> Vec<f32> {
    let inn = x.len();
    (0..rows)
        .map(|o| {
            let r = &w[o * inn..o * inn + inn];
            r.iter().zip(x).map(|(a, b)| a * b).sum()
        })
        .collect()
}

fn sp_rope(v: &mut [f32], hd: usize, pos: usize, base: f32, neox: bool) {
    let heads = v.len() / hd;
    let half = hd / 2;
    for h in 0..heads {
        let off = h * hd;
        for i in 0..half {
            let freq = 1.0 / base.powf(2.0 * i as f32 / hd as f32);
            let (s, c) = (pos as f32 * freq).sin_cos();
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

/// Run a Llama-family model purely from its spec + named f32 weights. Returns the
/// last token's logits. Prefills the prompt with a causal KV cache.
pub fn run_llama_from_spec(
    arch: &Value,
    w: &HashMap<String, Vec<f32>>,
    ids: &[usize],
) -> Result<Vec<f32>> {
    let geti = |k: &str| arch.get(k).and_then(|v| v.as_u64()).map(|v| v as usize);
    let need = |k: &str| geti(k).ok_or_else(|| spec(format!("spec missing '{k}'")));
    let dim = need("dim")?;
    let n_layers = need("n_layers")?;
    let n_heads = need("n_heads")?;
    let n_kv = need("n_kv_heads")?;
    let hd = need("head_dim")?;
    let ffn = need("ffn_dim")?;
    let vocab = need("vocab")?;
    let eps = arch.get("rms_eps").and_then(|v| v.as_f64()).unwrap_or(1e-5) as f32;
    let base = arch
        .get("rope_base")
        .and_then(|v| v.as_f64())
        .unwrap_or(10000.0) as f32;
    let neox = arch
        .get("rope_neox")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let kv_dim = n_kv * hd;
    let q_dim = n_heads * hd;
    let kv_mul = n_heads / n_kv;
    let get = |n: &str| {
        w.get(n)
            .ok_or_else(|| spec(format!("missing weight '{n}'")))
    };

    let tok = get("token_embd.weight")?;
    let out_w = w.get("output.weight").unwrap_or(tok);
    let out_norm = get("output_norm.weight")?;
    let seq = ids.len();
    let mut kcache = vec![vec![0f32; seq * kv_dim]; n_layers];
    let mut vcache = vec![vec![0f32; seq * kv_dim]; n_layers];
    let mut last = vec![0f32; vocab];

    for (pos, &tok_id) in ids.iter().enumerate() {
        let mut x = tok[tok_id * dim..tok_id * dim + dim].to_vec();
        for l in 0..n_layers {
            let p = |s: &str| format!("blk.{l}.{s}");
            let xb = sp_rmsnorm(&x, get(&p("attn_norm.weight"))?, eps);
            let mut q = sp_matvec(get(&p("attn_q.weight"))?, &xb, q_dim);
            let mut k = sp_matvec(get(&p("attn_k.weight"))?, &xb, kv_dim);
            let v = sp_matvec(get(&p("attn_v.weight"))?, &xb, kv_dim);
            sp_rope(&mut q, hd, pos, base, neox);
            sp_rope(&mut k, hd, pos, base, neox);
            kcache[l][pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&k);
            vcache[l][pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&v);
            let scale = 1.0 / (hd as f32).sqrt();
            let mut att = vec![0f32; q_dim];
            for h in 0..n_heads {
                let qh = &q[h * hd..(h + 1) * hd];
                let kvh = h / kv_mul;
                let mut sc = vec![0f32; pos + 1];
                for t in 0..=pos {
                    let kh = &kcache[l][t * kv_dim + kvh * hd..t * kv_dim + (kvh + 1) * hd];
                    sc[t] = qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f32>() * scale;
                }
                let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sm = 0.0;
                for s in &mut sc {
                    *s = (*s - mx).exp();
                    sm += *s;
                }
                let oh = &mut att[h * hd..(h + 1) * hd];
                for t in 0..=pos {
                    let vh = &vcache[l][t * kv_dim + kvh * hd..t * kv_dim + (kvh + 1) * hd];
                    let wgt = sc[t] / sm;
                    for i in 0..hd {
                        oh[i] += wgt * vh[i];
                    }
                }
            }
            let ao = sp_matvec(get(&p("attn_output.weight"))?, &att, dim);
            for i in 0..dim {
                x[i] += ao[i];
            }
            let xb2 = sp_rmsnorm(&x, get(&p("ffn_norm.weight"))?, eps);
            let mut g = sp_matvec(get(&p("ffn_gate.weight"))?, &xb2, ffn);
            let u = sp_matvec(get(&p("ffn_up.weight"))?, &xb2, ffn);
            for i in 0..ffn {
                g[i] = g[i] / (1.0 + (-g[i]).exp()) * u[i];
            }
            let fo = sp_matvec(get(&p("ffn_down.weight"))?, &g, dim);
            for i in 0..dim {
                x[i] += fo[i];
            }
        }
        let xf = sp_rmsnorm(&x, out_norm, eps);
        last = sp_matvec(out_w, &xf, vocab);
    }
    Ok(last)
}
