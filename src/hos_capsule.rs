//! Running a `.hos` capsule.
//!
//! A *runnable* capsule carries the engine-readable metadata (the GGUF-style
//! key/value map: hyperparameters + tokenizer) in its card, alongside the
//! tensors. [`HosSource`] presents that to the engine's generic loader exactly
//! like a GGUF or HF checkpoint does — so `.hos` becomes a first-class model
//! format the engine can load and run, not just an archive.

use std::collections::HashMap;
use std::path::Path;

use crate::error::{HosError, Result};
use crate::format;
use crate::gguf::{Gguf, GGML_F32};
use crate::model::ModelSource;
use crate::tokenizer::Tokenizer;

/// One resident tensor: its bytes plus the runtime type they're in. Quantized
/// GGUF block types are kept in their native bytes (so the engine reads ≈4-bit
/// instead of an f32 blow-up, and the GPU uses its fast coalesced quant kernels);
/// `f32` and the native `hq4` are materialised to f32 (`GGML_F32`).
struct Resident {
    bytes: Vec<u8>,
    runtime_type: u32,
    nfloats: usize,
}

/// A `ModelSource` backed by a loaded `.hos` capsule. A quantized capsule stays
/// quantized in memory (the fast path); metadata is the card's `meta` map.
pub struct HosSource {
    meta: serde_json::Value,
    tensors: HashMap<String, Resident>,
}

impl HosSource {
    /// Load a capsule and return the source plus its tokenizer. Errors if the
    /// capsule is archival (no engine metadata) rather than runnable.
    pub fn open(path: &Path) -> Result<(HosSource, Tokenizer)> {
        let (raws, card) = format::load_raw(path).map_err(HosError::from)?;
        if !card.meta.is_object() {
            return Err(HosError::Format(
                "this .hos is an archival capsule (no engine metadata) — not runnable. \
                 Re-mint it with a current `hos --to-hos` or `hos --ingest`."
                    .into(),
            ));
        }
        // HF-origin capsules carry a serialized tokenizer (`hos.tokenizer`);
        // GGUF-origin ones carry the GGUF `tokenizer.ggml.*` keys directly.
        let tok = if card.meta.get("hos.tokenizer").is_some() {
            Tokenizer::from_value(&card.meta["hos.tokenizer"])?
        } else {
            Tokenizer::from_hos(&card.meta)?
        };
        let mut tensors = HashMap::with_capacity(raws.len());
        for r in raws {
            let res = if let Some(gt) = format::dtype_to_ggml(r.dtype) {
                // a GGUF block type — keep the native quantized bytes (fast path)
                Resident {
                    bytes: r.bytes,
                    runtime_type: gt,
                    nfloats: r.nfloats,
                }
            } else if r.dtype == format::DTYPE_F32 {
                Resident {
                    bytes: r.bytes,
                    runtime_type: GGML_F32,
                    nfloats: r.nfloats,
                }
            } else {
                // hq4 (or anything non-GGUF): materialise to f32 — no GPU quant
                // kernel exists for it, so the f32 path keeps it runnable.
                let f = format::decode_raw(r.dtype, &r.bytes, r.nfloats).map_err(HosError::from)?;
                let mut b = Vec::with_capacity(f.len() * 4);
                for v in &f {
                    b.extend_from_slice(&v.to_le_bytes());
                }
                Resident {
                    bytes: b,
                    runtime_type: GGML_F32,
                    nfloats: r.nfloats,
                }
            };
            tensors.insert(r.name, res);
        }
        Ok((
            HosSource {
                meta: card.meta,
                tensors,
            },
            tok,
        ))
    }
}

impl ModelSource for HosSource {
    fn meta_str(&self, key: &str) -> Option<&str> {
        self.meta.get(key).and_then(|v| v.as_str())
    }
    fn meta_u64(&self, key: &str) -> Option<u64> {
        self.meta.get(key).and_then(|v| v.as_u64())
    }
    fn meta_f32(&self, key: &str) -> Option<f32> {
        self.meta
            .get(key)
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
    }
    fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
    fn dequant(&self, name: &str) -> Result<Vec<f32>> {
        let r = self
            .tensors
            .get(name)
            .ok_or_else(|| HosError::MissingTensor(name.to_string()))?;
        if r.runtime_type == GGML_F32 {
            Ok(r.bytes
                .chunks_exact(4)
                .map(|q| f32::from_le_bytes([q[0], q[1], q[2], q[3]]))
                .collect())
        } else {
            let mut d = vec![0f32; r.nfloats];
            Gguf::dequant_into(&r.bytes, r.runtime_type, r.nfloats, &mut d)?;
            Ok(d)
        }
    }
    fn raw(&self, name: &str) -> Result<(&[u8], u32, usize)> {
        let r = self
            .tensors
            .get(name)
            .ok_or_else(|| HosError::MissingTensor(name.to_string()))?;
        Ok((&r.bytes, r.runtime_type, r.nfloats))
    }
}
