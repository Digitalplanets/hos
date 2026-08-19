//! HF safetensors -> `.hos` ingest for the `qwen3_5` hybrid (Qwen3.8-27B family).
//!
//! HOS reads a HuggingFace checkpoint directly, but the generic `HfModel` name
//! map is llama-family; this hybrid has SSM (Gated-DeltaNet) layers, a fused QKV,
//! an MTP draft head, and a `model.language_model.` prefix (it's a VLM — the
//! vision tower is skipped; the language model is what runs as text chat). This
//! module maps those HF tensor names onto HOS's internal (GGUF-convention) names
//! and mints a runnable `.hos` capsule (`arch=qwen35`).
//!
//! It **streams**: read one tensor's bf16 -> f32, quantize it, push the bytes,
//! drop the f32. Peak memory is ~one tensor + the growing quantized output
//! (~14GB for q4_k), not the ~108GB an all-f32 materialization of a 27B would need.

use std::path::Path;

use crate::error::{HosError, Result};
use crate::format::{
    self, Card, RawTensor, DTYPE_F32, ROLE_EMBED, ROLE_NORM, ROLE_WEIGHT,
};
use crate::safetensors::SafeTensors;
use crate::tokenizer::Tokenizer;

/// Map an HF `linear_attn.*` / `self_attn.*` etc. tensor into HOS's internal name,
/// quantize the big matmul weights, keep norms + the small SSM projections in f32,
/// and write a runnable capsule. `quant` is the target ggml type for big weights.
pub fn ingest(dir: &Path, out: &Path, quant: u32) -> Result<()> {
    let cfg: serde_json::Value = crate::safetensors::read_json(&dir.join("config.json"))?;
    let tc = cfg.get("text_config").cloned().unwrap_or(cfg.clone());
    let g = |k: &str| tc.get(k).and_then(|v| v.as_u64()).unwrap_or(0);

    let hidden = g("hidden_size") as usize;
    let n_layers = g("num_hidden_layers") as usize;
    let head_dim = g("head_dim") as usize;
    let inter = g("intermediate_size") as usize;
    let vocab = g("vocab_size") as usize;
    // SSM (Gated-DeltaNet) shape mapping HF -> HOS's Cfg fields.
    let l_key_heads = g("linear_num_key_heads") as usize; // -> ssm.group_count
    let l_key_dim = g("linear_key_head_dim") as usize; // -> ssm.state_size
    let l_val_heads = g("linear_num_value_heads") as usize; // -> ssm.time_step_rank
    let l_val_dim = g("linear_value_head_dim") as usize;
    let inner_size = l_val_heads * l_val_dim; // -> ssm.inner_size
    let conv_k = g("linear_conv_kernel_dim") as usize; // -> ssm.conv_kernel
    let rope_dim = (head_dim as f64
        * tc.get("partial_rotary_factor")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0)) as usize;
    let rope_base = tc
        .get("rope_parameters")
        .and_then(|r| r.get("rope_theta"))
        .and_then(|v| v.as_f64())
        .or_else(|| tc.get("rope_theta").and_then(|v| v.as_f64()))
        .unwrap_or(1e7);
    let full_interval = g("full_attention_interval").max(1) as usize;
    let rms_eps = tc
        .get("rms_norm_eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-6) as f32;
    let layer_types: Vec<String> = tc
        .get("layer_types")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .map(|s| s.as_str().unwrap_or("full_attention").to_string())
                .collect()
        })
        .unwrap_or_else(|| (0..n_layers).map(|i| {
            if (i + 1) % full_interval == 0 { "full_attention".into() } else { "linear_attention".into() }
        }).collect());
    let has_mtp = g("mtp_num_hidden_layers") > 0;

    let st = SafeTensors::open_dir(dir)?;
    let tok = Tokenizer::from_hf(dir)?;

    // ---- engine-readable metadata: the `qwen35.*` keys Cfg::from_gguf reads ----
    let mut meta = serde_json::Map::new();
    let mut mu = |k: &str, v: u64| {
        meta.insert(format!("qwen35.{k}"), serde_json::json!(v));
    };
    mu("embedding_length", hidden as u64);
    // block_count INCLUDES the trailing MTP block; nextn_predict_layers backs it out.
    mu("block_count", (n_layers + if has_mtp { 1 } else { 0 }) as u64);
    mu("nextn_predict_layers", if has_mtp { 1 } else { 0 });
    mu("attention.head_count", g("num_attention_heads"));
    mu("attention.head_count_kv", g("num_key_value_heads"));
    mu("attention.key_length", head_dim as u64);
    mu("feed_forward_length", inter as u64);
    mu("rope.dimension_count", rope_dim as u64);
    mu("full_attention_interval", full_interval as u64);
    mu("ssm.conv_kernel", conv_k as u64);
    mu("ssm.state_size", l_key_dim as u64);
    mu("ssm.group_count", l_key_heads as u64);
    mu("ssm.inner_size", inner_size as u64);
    mu("ssm.time_step_rank", l_val_heads as u64);
    meta.insert(
        "qwen35.attention.layer_norm_rms_epsilon".into(),
        serde_json::json!(rms_eps),
    );
    meta.insert("qwen35.rope.freq_base".into(), serde_json::json!(rope_base));
    meta.insert("hos.tokenizer".into(), tok.to_value());
    let _ = (l_key_dim, vocab);

    // ---- stream tensors: read -> quantize/keep-f32 -> push ----
    let mut raws: Vec<RawTensor> = Vec::new();
    // big matmul weight: quantize to the target ggml type
    let mut w = |raws: &mut Vec<RawTensor>, name: &str, hf: &str, role: u8| -> Result<()> {
        let shape = st.shape(hf)?.to_vec();
        let data = st.to_f32(hf)?;
        let n = data.len();
        let bytes = crate::gguf_write::quantize(&data, quant);
        raws.push(RawTensor {
            name: name.to_string(),
            role,
            shape,
            dtype: format::ggml_to_dtype(quant),
            nfloats: n,
            bytes,
        });
        Ok(())
    };
    // norm / small SSM projection: keep f32 (the loader dequants these to Vec<f32>)
    let mut f = |raws: &mut Vec<RawTensor>, name: &str, hf: &str, map: fn(f32) -> f32| -> Result<()> {
        let shape = st.shape(hf)?.to_vec();
        let data: Vec<f32> = st.to_f32(hf)?.into_iter().map(map).collect();
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for v in &data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        raws.push(RawTensor {
            name: name.to_string(),
            role: ROLE_NORM,
            shape,
            dtype: DTYPE_F32,
            nfloats: data.len(),
            bytes,
        });
        Ok(())
    };
    let id = |x: f32| x;
    // Gated-DeltaNet A = -exp(A_log) (standard Mamba/DeltaNet), matching the GGUF's
    // already-negative `ssm_a`. VALIDATE against the GGUF once downloaded.
    let neg_exp = |x: f32| -(x.exp());

    eprintln!("[qwen35-hf] ingesting {} layers from {} ...", n_layers, dir.display());

    // embeddings + final norm + lm head
    w(&mut raws, "token_embd.weight", "model.language_model.embed_tokens.weight", ROLE_EMBED)?;
    f(&mut raws, "output_norm.weight", "model.language_model.norm.weight", id)?;
    if st.contains("lm_head.weight") {
        w(&mut raws, "output.weight", "lm_head.weight", ROLE_WEIGHT)?;
    }

    for (i, lt) in layer_types.iter().enumerate() {
        let hf = |s: &str| format!("model.language_model.layers.{i}.{s}");
        let blk = |s: &str| format!("blk.{i}.{s}");
        f(&mut raws, &blk("attn_norm.weight"), &hf("input_layernorm.weight"), id)?;
        f(&mut raws, &blk("post_attention_norm.weight"), &hf("post_attention_layernorm.weight"), id)?;
        w(&mut raws, &blk("ffn_gate.weight"), &hf("mlp.gate_proj.weight"), ROLE_WEIGHT)?;
        w(&mut raws, &blk("ffn_up.weight"), &hf("mlp.up_proj.weight"), ROLE_WEIGHT)?;
        w(&mut raws, &blk("ffn_down.weight"), &hf("mlp.down_proj.weight"), ROLE_WEIGHT)?;
        if lt == "full_attention" {
            // q/k stored as-is (no RoPE permutation). VALIDATE against the GGUF.
            w(&mut raws, &blk("attn_q.weight"), &hf("self_attn.q_proj.weight"), ROLE_WEIGHT)?;
            w(&mut raws, &blk("attn_k.weight"), &hf("self_attn.k_proj.weight"), ROLE_WEIGHT)?;
            w(&mut raws, &blk("attn_v.weight"), &hf("self_attn.v_proj.weight"), ROLE_WEIGHT)?;
            w(&mut raws, &blk("attn_output.weight"), &hf("self_attn.o_proj.weight"), ROLE_WEIGHT)?;
            f(&mut raws, &blk("attn_q_norm.weight"), &hf("self_attn.q_norm.weight"), id)?;
            f(&mut raws, &blk("attn_k_norm.weight"), &hf("self_attn.k_norm.weight"), id)?;
        } else {
            // linear_attention (Gated-DeltaNet SSM)
            w(&mut raws, &blk("attn_qkv.weight"), &hf("linear_attn.in_proj_qkv.weight"), ROLE_WEIGHT)?;
            w(&mut raws, &blk("attn_gate.weight"), &hf("linear_attn.in_proj_z.weight"), ROLE_WEIGHT)?;
            w(&mut raws, &blk("ssm_out.weight"), &hf("linear_attn.out_proj.weight"), ROLE_WEIGHT)?;
            // in_proj_a -> ssm_alpha, in_proj_b -> ssm_beta. VALIDATE the a/b order.
            f(&mut raws, &blk("ssm_alpha.weight"), &hf("linear_attn.in_proj_a.weight"), id)?;
            f(&mut raws, &blk("ssm_beta.weight"), &hf("linear_attn.in_proj_b.weight"), id)?;
            f(&mut raws, &blk("ssm_a"), &hf("linear_attn.A_log"), neg_exp)?;
            f(&mut raws, &blk("ssm_conv1d.weight"), &hf("linear_attn.conv1d.weight"), id)?;
            f(&mut raws, &blk("ssm_dt.bias"), &hf("linear_attn.dt_bias"), id)?;
            f(&mut raws, &blk("ssm_norm.weight"), &hf("linear_attn.norm.weight"), id)?;
        }
    }

    // ---- MTP / NextN draft head at block index n_layers ----
    if has_mtp {
        let mi = n_layers;
        let hf = |s: &str| format!("mtp.{s}");
        let blk = |s: &str| format!("blk.{mi}.{s}");
        // its own attention sub-block
        f(&mut raws, &blk("attn_norm.weight"), &hf("layers.0.input_layernorm.weight"), id)?;
        f(&mut raws, &blk("post_attention_norm.weight"), &hf("layers.0.post_attention_layernorm.weight"), id)?;
        w(&mut raws, &blk("attn_q.weight"), &hf("layers.0.self_attn.q_proj.weight"), ROLE_WEIGHT)?;
        w(&mut raws, &blk("attn_k.weight"), &hf("layers.0.self_attn.k_proj.weight"), ROLE_WEIGHT)?;
        w(&mut raws, &blk("attn_v.weight"), &hf("layers.0.self_attn.v_proj.weight"), ROLE_WEIGHT)?;
        w(&mut raws, &blk("attn_output.weight"), &hf("layers.0.self_attn.o_proj.weight"), ROLE_WEIGHT)?;
        f(&mut raws, &blk("attn_q_norm.weight"), &hf("layers.0.self_attn.q_norm.weight"), id)?;
        f(&mut raws, &blk("attn_k_norm.weight"), &hf("layers.0.self_attn.k_norm.weight"), id)?;
        w(&mut raws, &blk("ffn_gate.weight"), &hf("layers.0.mlp.gate_proj.weight"), ROLE_WEIGHT)?;
        w(&mut raws, &blk("ffn_up.weight"), &hf("layers.0.mlp.up_proj.weight"), ROLE_WEIGHT)?;
        w(&mut raws, &blk("ffn_down.weight"), &hf("layers.0.mlp.down_proj.weight"), ROLE_WEIGHT)?;
        // the fusion + norms the MTP head needs
        w(&mut raws, &blk("nextn.eh_proj.weight"), &hf("fc.weight"), ROLE_WEIGHT)?;
        f(&mut raws, &blk("nextn.enorm.weight"), &hf("pre_fc_norm_embedding.weight"), id)?;
        f(&mut raws, &blk("nextn.hnorm.weight"), &hf("pre_fc_norm_hidden.weight"), id)?;
        f(&mut raws, &blk("nextn.shared_head_norm.weight"), &hf("norm.weight"), id)?;
    }

    // ---- write the runnable capsule (arch=qwen35 routes to the qwen35 loader) ----
    let name = out.file_stem().and_then(|s| s.to_str()).unwrap_or("qwen35");
    let arch = serde_json::json!({ "architecture": "qwen35", "source": "hf-ingest" });
    let mut card = Card::new(name, arch);
    card.mode = "inference".into();
    card.meta = serde_json::Value::Object(meta);
    format::save_raw(out, &raws, &card).map_err(HosError::from)?;
    eprintln!(
        "[qwen35-hf] wrote {} ({} tensors)",
        out.display(),
        raws.len()
    );
    Ok(())
}
