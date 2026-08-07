//! `model::gpu_quant_supported` preflight unit tests (safety net, part 2).
//!
//! `gpu_quant_supported` decides whether a model may ride the fused Metal runner.
//! It must return `false` whenever the GPU has no kernel for what would be
//! uploaded — i.e. a MoE model (the fused runner has no expert dispatch) or any
//! linear weight in a quant the GPU can't decode (the native `hq4`/`HQ4_TYPE`, or
//! any non-standard type) — and `true` for the plain GGUF block types that DO
//! have a coalesced Metal matvec (Q8_0/Q4_0/Q5_0/Q4_K/Q5_K/Q6_K and f32/f16).
//!
//! Two layers of coverage:
//!   1. Branch coverage with tiny in-test `ModelSource` mocks (no giant fixtures,
//!      runs anywhere). The hq4 case uses the engine's *real* `encode_hq4` bytes
//!      tagged with the real `HQ4_TYPE`, so it exercises the genuine rejection.
//!   2. Real-model spot checks against files in `models/` when present (skipped
//!      gracefully otherwise), so the preflight is pinned against actual headers.
//!
//! NOTE on real `.hos` HQ4 capsules: when a capsule minted with `--quantize hq4`
//! is loaded through `HosSource`, the hq4 tensors are *materialised to f32* in
//! memory (no GPU hq4 kernel exists, so f32 keeps them runnable). Their `raw()`
//! therefore reports `GGML_F32`, and `gpu_quant_supported` returns `true` for
//! that source — which is correct, f32 has a GPU kernel. The `false` path is the
//! one that matters for a *mixed-quant* source that still reports the native
//! `HQ4_TYPE`; that is what the mock pins. (Verified empirically: a real
//! smol-hq4 capsule's `blk.0.attn_q.weight` reads back as GGML_F32.)

use std::collections::HashMap;
use std::path::PathBuf;

use hos::error::{HosError, Result};
use hos::gguf::{GGML_F32, GGML_Q4_K, GGML_Q8_0};
use hos::model::{gpu_quant_supported, ModelSource, HQ4_TYPE};

/// A minimal in-memory `ModelSource`: just enough tensor headers for the
/// preflight, which only ever calls `has()` and `raw()`.
struct MockSource {
    tensors: HashMap<String, (Vec<u8>, u32, usize)>,
}

impl MockSource {
    /// A standard Llama-family layout with every linear weight in `ty`.
    fn llama(n_layers: usize, ty: u32) -> Self {
        let mut tensors = HashMap::new();
        let names = [
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ];
        for i in 0..n_layers {
            for nm in names {
                tensors.insert(format!("blk.{i}.{nm}.weight"), (vec![0u8; 4], ty, 1));
            }
        }
        MockSource { tensors }
    }
    fn set(&mut self, name: &str, bytes: Vec<u8>, ty: u32, n: usize) {
        self.tensors.insert(name.to_string(), (bytes, ty, n));
    }
}

impl ModelSource for MockSource {
    fn meta_str(&self, _: &str) -> Option<&str> {
        None
    }
    fn meta_u64(&self, _: &str) -> Option<u64> {
        None
    }
    fn meta_f32(&self, _: &str) -> Option<f32> {
        None
    }
    fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
    fn dequant(&self, name: &str) -> Result<Vec<f32>> {
        let (_, _, n) = self
            .tensors
            .get(name)
            .ok_or_else(|| HosError::MissingTensor(name.into()))?;
        Ok(vec![0.0; *n])
    }
    fn raw(&self, name: &str) -> Result<(&[u8], u32, usize)> {
        let (b, t, n) = self
            .tensors
            .get(name)
            .ok_or_else(|| HosError::MissingTensor(name.into()))?;
        Ok((b, *t, *n))
    }
}

#[test]
fn supported_for_standard_block_quants() {
    assert!(
        gpu_quant_supported(&MockSource::llama(4, GGML_Q8_0)),
        "Q8_0 has a Metal kernel"
    );
    assert!(
        gpu_quant_supported(&MockSource::llama(4, GGML_Q4_K)),
        "Q4_K has a Metal kernel"
    );
    assert!(
        gpu_quant_supported(&MockSource::llama(4, GGML_F32)),
        "f32 uploads directly"
    );
}

#[test]
fn unsupported_for_hq4_native_quant() {
    // one offending tensor anywhere in the stack must veto the GPU path.
    let mut m = MockSource::llama(4, GGML_Q8_0);
    // real engine hq4 bytes, tagged with the real native type the GPU can't decode.
    let hq4_bytes = hos::hos_quant::encode_hq4(&[0.1, -0.2, 0.3, 0.0, 0.5, -0.4, 0.25, -0.1]);
    m.set("blk.2.ffn_down.weight", hq4_bytes, HQ4_TYPE, 8);
    assert!(
        !gpu_quant_supported(&m),
        "a tensor in native HQ4_TYPE has no GPU kernel -> preflight must veto the GPU path"
    );
}

#[test]
fn unsupported_for_moe() {
    // MoE is rejected purely on structure (expert tensor present), before any
    // per-tensor type scan — the fused runner has no expert dispatch.
    let mut m = MockSource::llama(4, GGML_Q8_0);
    m.set("blk.0.ffn_gate_exps.weight", vec![0u8; 4], GGML_Q8_0, 1);
    assert!(
        !gpu_quant_supported(&m),
        "MoE (ffn_gate_exps present) -> CPU path"
    );
}

// ---- real-model spot checks (skipped when the file isn't present) ----

fn find_model(name: &str) -> Option<PathBuf> {
    let start = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    start
        .ancestors()
        .map(|a| a.join("models").join(name))
        .find(|p| p.is_file())
}

#[test]
fn real_q8_gguf_is_supported() {
    let Some(p) = find_model("SmolLM2-135M-Instruct-Q8_0.gguf") else {
        eprintln!("[skip] no Q8_0 gguf present");
        return;
    };
    let g = hos::gguf::Gguf::open(&p).expect("open gguf");
    assert!(
        gpu_quant_supported(&g),
        "real Q8_0 GGUF must be GPU-eligible"
    );
}

#[test]
fn real_q4k_capsule_is_supported() {
    let Some(p) = find_model("llama1b.q4k.hos") else {
        eprintln!("[skip] no Q4_K capsule present");
        return;
    };
    let (src, _tok) = hos::hos_capsule::HosSource::open(&p).expect("open capsule");
    assert!(
        gpu_quant_supported(&src),
        "real Q4_K .hos capsule must be GPU-eligible"
    );
}

#[test]
fn real_moe_gguf_is_unsupported() {
    let Some(p) = find_model("OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf") else {
        eprintln!("[skip] no MoE gguf present");
        return;
    };
    let g = hos::gguf::Gguf::open(&p).expect("open gguf");
    assert!(
        g.has("blk.0.ffn_gate_exps.weight"),
        "OLMoE should expose expert tensors"
    );
    assert!(
        !gpu_quant_supported(&g),
        "real MoE GGUF must take the CPU path"
    );
}
