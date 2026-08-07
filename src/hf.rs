//! HuggingFace checkpoint loader: read a `config.json` + `*.safetensors` folder
//! directly — no GGUF, no llama.cpp, no `transformers` in the loop.
//!
//! `HfModel` is an adapter, not a second loader. It implements [`ModelSource`]
//! over an HF checkpoint by doing exactly what `convert_hf_to_gguf.py` does, in
//! memory: translate HF tensor names (`model.layers.N.self_attn.q_proj.weight`)
//! to the GGUF names the engine already understands (`blk.N.attn_q.weight`),
//! synthesize the GGUF metadata keys from `config.json`, fold BF16 → F16 for the
//! matmul weights, and bake Gemma's `(1 + w)` into its norms. Every architecture
//! `Model::load` supports therefore loads from HF unchanged.

use std::collections::HashMap;
use std::path::Path;

use half::f16;

use crate::error::{HosError, Result};
use crate::gguf::{Value, GGML_F16, GGML_F32};
use crate::model::{Arch, ModelSource};
use crate::safetensors::{read_json, SafeTensors};

struct Owned {
    bytes: Vec<u8>,
    ggml_type: u32,
    n: usize,
}

pub struct HfModel {
    meta: HashMap<String, Value>,
    arch: Arch,
    tensors: HashMap<String, Owned>,
}

fn f32_le(v: &[f32]) -> Vec<u8> {
    let mut o = Vec::with_capacity(v.len() * 4);
    for &x in v {
        o.extend_from_slice(&x.to_le_bytes());
    }
    o
}

/// A matmul/embedding weight: stored as F16 (BF16/F32 sources are folded to F16,
/// matching a GGUF F16 conversion — half the memory, and the GPU kernels read it
/// directly).
fn owned_matrix(st: &SafeTensors, name: &str) -> Result<Owned> {
    let n = st.shape(name)?.iter().product();
    Ok(Owned {
        bytes: st.to_f16_bytes(name)?,
        ggml_type: GGML_F16,
        n,
    })
}

fn f32_to_f16_le(v: &[f32]) -> Vec<u8> {
    let mut o = Vec::with_capacity(v.len() * 2);
    for &x in v {
        o.extend_from_slice(&f16::from_f32(x).to_bits().to_le_bytes());
    }
    o
}

/// llama.cpp's Q/K permutation. HF Llama/Mistral store Q/K weights for
/// *rotate-half* RoPE; HOS (like GGUF) applies *interleaved-pair* RoPE to those
/// arches, so the per-head rows must be reordered to match — otherwise RoPE is
/// applied to the wrong pairs (greedy output still reads fine, but the predicted
/// distribution is wrong, which wrecks perplexity). This reshapes
/// `[n_head, 2, hd/2, cols]` → swap axes 1,2 → `[n_head, hd/2, 2, cols]`,
/// exactly as `convert_hf_to_gguf.py` does.
fn permute_rows(data: &[f32], n_head: usize, rows: usize, cols: usize) -> Vec<f32> {
    let hd = rows / n_head;
    let half = hd / 2;
    let mut out = vec![0f32; data.len()];
    for h in 0..n_head {
        for a in 0..2 {
            for b in 0..half {
                let src = h * hd + a * half + b;
                let dst = h * hd + b * 2 + a;
                out[dst * cols..dst * cols + cols]
                    .copy_from_slice(&data[src * cols..src * cols + cols]);
            }
        }
    }
    out
}

/// A Q or K projection for an interleaved-RoPE arch (Llama/Mistral): permuted to
/// the GGUF layout, then stored as F16.
fn owned_qk_permuted(st: &SafeTensors, name: &str, n_head: usize) -> Result<Owned> {
    let shape = st.shape(name)?;
    let (rows, cols) = (shape[0], shape[1]);
    let f = permute_rows(&st.to_f32(name)?, n_head, rows, cols);
    Ok(Owned {
        bytes: f32_to_f16_le(&f),
        ggml_type: GGML_F16,
        n: rows * cols,
    })
}

/// A norm / bias vector: kept in full f32 (they're tiny and precision-sensitive).
/// `add_one` bakes Gemma's `(1 + weight)` so the runtime path stays uniform.
fn owned_f32(st: &SafeTensors, name: &str, add_one: bool) -> Result<Owned> {
    let mut f = st.to_f32(name)?;
    if add_one {
        for v in &mut f {
            *v += 1.0;
        }
    }
    let n = f.len();
    Ok(Owned {
        bytes: f32_le(&f),
        ggml_type: GGML_F32,
        n,
    })
}

/// Stacked per-expert weights (MoE): HF stores each expert as its own tensor;
/// GGUF/HOS wants them concatenated into one blob. Folded to F16.
fn owned_experts(st: &SafeTensors, names: &[String]) -> Result<Owned> {
    let mut bytes = Vec::new();
    let mut n = 0usize;
    for nm in names {
        n += st.shape(nm)?.iter().product::<usize>();
        bytes.extend_from_slice(&st.to_f16_bytes(nm)?);
    }
    Ok(Owned {
        bytes,
        ggml_type: GGML_F16,
        n,
    })
}

impl HfModel {
    pub fn arch(&self) -> Arch {
        self.arch
    }

    /// The synthesized GGUF-style metadata map (for persisting into a `.hos`).
    pub fn meta_map(&self) -> &HashMap<String, Value> {
        &self.meta
    }

    /// All mapped (GGUF-named) tensor names.
    pub fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    pub fn open(dir: &Path) -> Result<HfModel> {
        let cfg = read_json(&dir.join("config.json"))?;
        let st = SafeTensors::open_dir(dir)?;

        let mt = cfg["model_type"].as_str().unwrap_or("llama");
        let (arch, arch_name): (Arch, &str) = match mt {
            "llama" => (Arch::Llama, "llama"),
            "qwen2" => (Arch::Qwen2, "qwen2"),
            "mistral" => (Arch::Mistral, "mistral"),
            "gemma" | "gemma2" => (Arch::Gemma, "gemma2"),
            "phi3" => (Arch::Phi3, "phi3"),
            "olmoe" => (Arch::OlMoe, "olmoe"),
            other => {
                return Err(HosError::UnsupportedArch(format!(
                    "HF model_type '{other}' — HOS loads llama/qwen2/mistral/gemma2/phi3/olmoe checkpoints"
                )))
            }
        };
        let is_gemma = arch == Arch::Gemma;
        let is_phi3 = arch == Arch::Phi3;
        // Llama/Mistral use interleaved-pair RoPE in HOS, so their HF (rotate-half)
        // Q/K weights must be permuted to match — the rest use NEOX rope as-is.
        let permute_qk = matches!(arch, Arch::Llama | Arch::Mistral);

        let cu = |k: &str| cfg[k].as_u64();
        let cf = |k: &str| cfg[k].as_f64();
        let need_u = |k: &str| cu(k).ok_or_else(|| HosError::MissingMeta(k.to_string()));
        let dim = need_u("hidden_size")?;
        let n_layers = need_u("num_hidden_layers")?;
        let n_heads = need_u("num_attention_heads")?;
        let n_kv = cu("num_key_value_heads").unwrap_or(n_heads);
        let ffn = need_u("intermediate_size")?;
        let ctx = cu("max_position_embeddings").unwrap_or(2048);
        let eps = cf("rms_norm_eps").unwrap_or(1e-5) as f32;
        let rope = cf("rope_theta").unwrap_or(10000.0) as f32;
        let n_experts = cu("num_experts")
            .or_else(|| cu("num_local_experts"))
            .unwrap_or(0);
        let n_used = cu("num_experts_per_tok").unwrap_or(0);

        // ---- synthesize the GGUF-style metadata keys Model::load reads ----
        let k = |s: &str| format!("{arch_name}.{s}");
        let mut meta: HashMap<String, Value> = HashMap::new();
        meta.insert("general.architecture".into(), Value::Str(arch_name.into()));
        meta.insert(k("embedding_length"), Value::U64(dim));
        meta.insert(k("block_count"), Value::U64(n_layers));
        meta.insert(k("attention.head_count"), Value::U64(n_heads));
        meta.insert(k("attention.head_count_kv"), Value::U64(n_kv));
        meta.insert(k("feed_forward_length"), Value::U64(ffn));
        meta.insert(k("context_length"), Value::U64(ctx));
        meta.insert(k("attention.layer_norm_rms_epsilon"), Value::F32(eps));
        meta.insert(k("rope.freq_base"), Value::F32(rope));
        if let Some(hd) = cu("head_dim") {
            meta.insert(k("attention.key_length"), Value::U64(hd));
        }
        if n_experts > 0 {
            meta.insert(k("expert_count"), Value::U64(n_experts));
            meta.insert(k("expert_used_count"), Value::U64(n_used));
        }
        if let Some(s) = cf("attn_logit_softcapping") {
            meta.insert(k("attn_logit_softcapping"), Value::F32(s as f32));
        }
        if let Some(s) = cf("final_logit_softcapping") {
            meta.insert(k("final_logit_softcapping"), Value::F32(s as f32));
        }

        let n_layers = n_layers as usize;
        let n_experts = n_experts as usize;

        // ---- map tensors: HF names -> GGUF names ----
        let mut t: HashMap<String, Owned> = HashMap::new();
        t.insert(
            "token_embd.weight".into(),
            owned_matrix(&st, "model.embed_tokens.weight")?,
        );
        t.insert(
            "output_norm.weight".into(),
            owned_f32(&st, "model.norm.weight", is_gemma)?,
        );
        // untied LM head; if absent the loader ties to the embedding table
        if st.contains("lm_head.weight") {
            t.insert("output.weight".into(), owned_matrix(&st, "lm_head.weight")?);
        }

        for i in 0..n_layers {
            let h = |s: &str| format!("model.layers.{i}.{s}");
            let g = |s: &str| format!("blk.{i}.{s}");

            t.insert(
                g("attn_norm.weight"),
                owned_f32(&st, &h("input_layernorm.weight"), is_gemma)?,
            );

            if is_phi3 {
                // Phi-3 fuses Q/K/V; Model::load splits the rows.
                t.insert(
                    g("attn_qkv.weight"),
                    owned_matrix(&st, &h("self_attn.qkv_proj.weight"))?,
                );
            } else {
                if permute_qk {
                    t.insert(
                        g("attn_q.weight"),
                        owned_qk_permuted(&st, &h("self_attn.q_proj.weight"), n_heads as usize)?,
                    );
                    t.insert(
                        g("attn_k.weight"),
                        owned_qk_permuted(&st, &h("self_attn.k_proj.weight"), n_kv as usize)?,
                    );
                } else {
                    t.insert(
                        g("attn_q.weight"),
                        owned_matrix(&st, &h("self_attn.q_proj.weight"))?,
                    );
                    t.insert(
                        g("attn_k.weight"),
                        owned_matrix(&st, &h("self_attn.k_proj.weight"))?,
                    );
                }
                t.insert(
                    g("attn_v.weight"),
                    owned_matrix(&st, &h("self_attn.v_proj.weight"))?,
                );
                // optional q/k/v bias (Qwen2)
                for (gg, pp) in [
                    ("attn_q.bias", "q_proj"),
                    ("attn_k.bias", "k_proj"),
                    ("attn_v.bias", "v_proj"),
                ] {
                    let hn = h(&format!("self_attn.{pp}.bias"));
                    if st.contains(&hn) {
                        t.insert(g(gg), owned_f32(&st, &hn, false)?);
                    }
                }
            }
            t.insert(
                g("attn_output.weight"),
                owned_matrix(&st, &h("self_attn.o_proj.weight"))?,
            );

            // OLMoE QK-norm (RMSNorm on the full q/k projections)
            let qn = h("self_attn.q_norm.weight");
            if st.contains(&qn) {
                t.insert(g("attn_q_norm.weight"), owned_f32(&st, &qn, false)?);
                t.insert(
                    g("attn_k_norm.weight"),
                    owned_f32(&st, &h("self_attn.k_norm.weight"), false)?,
                );
            }

            // FFN norms — Gemma2's names + sandwich norms differ from the rest.
            if is_gemma {
                t.insert(
                    g("ffn_norm.weight"),
                    owned_f32(&st, &h("pre_feedforward_layernorm.weight"), true)?,
                );
                t.insert(
                    g("post_attention_norm.weight"),
                    owned_f32(&st, &h("post_attention_layernorm.weight"), true)?,
                );
                t.insert(
                    g("post_ffw_norm.weight"),
                    owned_f32(&st, &h("post_feedforward_layernorm.weight"), true)?,
                );
            } else {
                t.insert(
                    g("ffn_norm.weight"),
                    owned_f32(&st, &h("post_attention_layernorm.weight"), false)?,
                );
            }

            // FFN weights — dense SwiGLU, Phi-3 fused gate/up, or MoE experts.
            if n_experts > 0 && st.contains(&h("mlp.experts.0.gate_proj.weight")) {
                t.insert(
                    g("ffn_gate_inp.weight"),
                    owned_matrix(&st, &h("mlp.gate.weight"))?,
                );
                let names = |proj: &str| -> Vec<String> {
                    (0..n_experts)
                        .map(|e| h(&format!("mlp.experts.{e}.{proj}.weight")))
                        .collect()
                };
                t.insert(
                    g("ffn_gate_exps.weight"),
                    owned_experts(&st, &names("gate_proj"))?,
                );
                t.insert(
                    g("ffn_up_exps.weight"),
                    owned_experts(&st, &names("up_proj"))?,
                );
                t.insert(
                    g("ffn_down_exps.weight"),
                    owned_experts(&st, &names("down_proj"))?,
                );
            } else if is_phi3 {
                t.insert(
                    g("ffn_up.weight"),
                    owned_matrix(&st, &h("mlp.gate_up_proj.weight"))?,
                );
                t.insert(
                    g("ffn_down.weight"),
                    owned_matrix(&st, &h("mlp.down_proj.weight"))?,
                );
            } else {
                t.insert(
                    g("ffn_gate.weight"),
                    owned_matrix(&st, &h("mlp.gate_proj.weight"))?,
                );
                t.insert(
                    g("ffn_up.weight"),
                    owned_matrix(&st, &h("mlp.up_proj.weight"))?,
                );
                t.insert(
                    g("ffn_down.weight"),
                    owned_matrix(&st, &h("mlp.down_proj.weight"))?,
                );
            }
        }

        eprintln!(
            "[hos] hf checkpoint: {mt} -> {arch:?}, {n_layers} layers, {} tensors mapped",
            t.len()
        );
        Ok(HfModel {
            meta,
            arch,
            tensors: t,
        })
    }
}

impl ModelSource for HfModel {
    fn meta_str(&self, key: &str) -> Option<&str> {
        self.meta.get(key).and_then(|v| v.as_str())
    }
    fn meta_u64(&self, key: &str) -> Option<u64> {
        self.meta.get(key).and_then(|v| v.as_u64())
    }
    fn meta_f32(&self, key: &str) -> Option<f32> {
        self.meta.get(key).and_then(|v| v.as_f32())
    }
    fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
    fn dequant(&self, name: &str) -> Result<Vec<f32>> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| HosError::MissingTensor(name.to_string()))?;
        let mut out = vec![0f32; t.n];
        crate::gguf::Gguf::dequant_into(&t.bytes, t.ggml_type, t.n, &mut out)?;
        Ok(out)
    }
    fn raw(&self, name: &str) -> Result<(&[u8], u32, usize)> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| HosError::MissingTensor(name.to_string()))?;
        Ok((&t.bytes, t.ggml_type, t.n))
    }
}
