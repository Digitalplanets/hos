//! Minimal safetensors reader — single-file and sharded.
//!
//! Format: `u64 header_len (LE) | JSON header | tensor data`. The header maps
//! each tensor name to `{dtype, shape, data_offsets:[start,end]}`, where the
//! offsets are relative to the start of the data section (right after the
//! header). Sharded checkpoints add a `model.safetensors.index.json` whose
//! `weight_map` points each tensor at its shard file.
//!
//! Only the float dtypes LLM checkpoints actually ship in are accepted —
//! `F32` / `F16` / `BF16`. Everything is bounds-checked: a truncated or
//! malformed file returns a clean error rather than panicking.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use half::f16;
use memmap2::Mmap;
use serde_json::Value as J;

use crate::error::{HosError, Result};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Dtype {
    F32,
    F16,
    Bf16,
}

impl Dtype {
    fn parse(s: &str) -> Result<Dtype> {
        Ok(match s {
            "F32" => Dtype::F32,
            "F16" => Dtype::F16,
            "BF16" => Dtype::Bf16,
            other => {
                return Err(HosError::Format(format!(
                    "safetensors dtype '{other}' unsupported (F32/F16/BF16 only)"
                )))
            }
        })
    }
}

struct Entry {
    shard: usize,
    dtype: Dtype,
    shape: Vec<usize>,
    start: usize, // absolute byte offset into the shard's mmap
    end: usize,
}

pub struct SafeTensors {
    maps: Vec<Mmap>,
    index: HashMap<String, Entry>,
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

impl SafeTensors {
    /// Open a checkpoint directory: prefers `model.safetensors.index.json`
    /// (sharded), else a single `model.safetensors`.
    pub fn open_dir(dir: &Path) -> Result<SafeTensors> {
        let index_path = dir.join("model.safetensors.index.json");
        if index_path.exists() {
            let idx: J = read_json(&index_path)?;
            let map = idx
                .get("weight_map")
                .and_then(|v| v.as_object())
                .ok_or_else(|| HosError::Format("index.json has no weight_map".into()))?;
            // unique shard files, in a stable order
            let mut shards: Vec<String> = map
                .values()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            shards.sort();
            shards.dedup();
            let mut st = SafeTensors {
                maps: Vec::new(),
                index: HashMap::new(),
            };
            for (i, fname) in shards.iter().enumerate() {
                st.add_shard(i, &dir.join(fname))?;
            }
            Ok(st)
        } else {
            let single = dir.join("model.safetensors");
            if !single.exists() {
                return Err(HosError::Format(format!(
                    "no safetensors in {} (need model.safetensors or model.safetensors.index.json)",
                    dir.display()
                )));
            }
            let mut st = SafeTensors {
                maps: Vec::new(),
                index: HashMap::new(),
            };
            st.add_shard(0, &single)?;
            Ok(st)
        }
    }

    /// Open a single safetensors file directly (not a directory). Used for
    /// stand-alone checkpoints that aren't named `model.safetensors`.
    pub fn open_file(path: &Path) -> Result<SafeTensors> {
        let mut st = SafeTensors {
            maps: Vec::new(),
            index: HashMap::new(),
        };
        st.add_shard(0, path)?;
        Ok(st)
    }

    fn add_shard(&mut self, shard: usize, path: &Path) -> Result<()> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() < 8 {
            return Err(HosError::Format(format!("{} too small", path.display())));
        }
        let hlen = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let data_base = 8usize
            .checked_add(hlen)
            .filter(|&b| b <= mmap.len())
            .ok_or_else(|| HosError::Format(format!("{}: header out of bounds", path.display())))?;
        let header: J = serde_json::from_slice(&mmap[8..8 + hlen])
            .map_err(|e| HosError::Format(format!("{}: bad header json: {e}", path.display())))?;
        let obj = header
            .as_object()
            .ok_or_else(|| HosError::Format("safetensors header is not an object".into()))?;
        for (name, meta) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = Dtype::parse(meta["dtype"].as_str().unwrap_or(""))?;
            let shape: Vec<usize> = meta["shape"]
                .as_array()
                .ok_or_else(|| HosError::Format(format!("{name}: no shape")))?
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect();
            let off = meta["data_offsets"]
                .as_array()
                .ok_or_else(|| HosError::Format(format!("{name}: no data_offsets")))?;
            let s = data_base + off[0].as_u64().unwrap_or(0) as usize;
            let e = data_base + off[1].as_u64().unwrap_or(0) as usize;
            if e > mmap.len() || s > e {
                return Err(HosError::Format(format!("{name}: data out of bounds")));
            }
            self.index.insert(
                name.clone(),
                Entry {
                    shard,
                    dtype,
                    shape,
                    start: s,
                    end: e,
                },
            );
        }
        self.maps.push(mmap);
        Ok(())
    }

    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// All tensor names, sorted — deterministic order for archival and hashing.
    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.index.keys().cloned().collect();
        v.sort();
        v
    }

    /// Content hash of the source weights in their *native* dtype — the HF
    /// artifact's identity. FNV-1a over each tensor's raw bytes in sorted-name
    /// order, so re-reading the same checkpoint yields the same id (used as the
    /// lineage edge when minting a `.hos` capsule).
    pub fn source_hash(&self) -> String {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for name in self.names() {
            let e = &self.index[&name];
            for &byte in &self.maps[e.shard][e.start..e.end] {
                h ^= byte as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        format!("{h:016x}")
    }

    fn entry(&self, name: &str) -> Result<&Entry> {
        self.index
            .get(name)
            .ok_or_else(|| HosError::MissingTensor(name.to_string()))
    }

    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        Ok(&self.entry(name)?.shape)
    }

    pub fn dtype(&self, name: &str) -> Result<Dtype> {
        Ok(self.entry(name)?.dtype)
    }

    fn bytes(&self, e: &Entry) -> &[u8] {
        &self.maps[e.shard][e.start..e.end]
    }

    /// Dequantize/convert a tensor to a flat f32 vector (row-major).
    pub fn to_f32(&self, name: &str) -> Result<Vec<f32>> {
        let e = self.entry(name)?;
        let b = self.bytes(e);
        let n = numel(&e.shape);
        let mut out = vec![0f32; n];
        match e.dtype {
            Dtype::F32 => {
                for i in 0..n {
                    out[i] =
                        f32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
                }
            }
            Dtype::F16 => {
                for i in 0..n {
                    out[i] = f16::from_bits(u16::from_le_bytes([b[2 * i], b[2 * i + 1]])).to_f32();
                }
            }
            Dtype::Bf16 => {
                for i in 0..n {
                    let bits = u16::from_le_bytes([b[2 * i], b[2 * i + 1]]);
                    out[i] = f32::from_bits((bits as u32) << 16);
                }
            }
        }
        Ok(out)
    }

    /// Raw 16-bit half-precision bytes for a tensor (ggml `F16` layout). `F16`
    /// passes through untouched; `BF16`/`F32` are converted to `F16`. This is the
    /// form the GPU matvec kernels consume.
    pub fn to_f16_bytes(&self, name: &str) -> Result<Vec<u8>> {
        let e = self.entry(name)?;
        let b = self.bytes(e);
        let n = numel(&e.shape);
        Ok(match e.dtype {
            Dtype::F16 => b.to_vec(),
            Dtype::Bf16 => {
                let mut out = Vec::with_capacity(n * 2);
                for i in 0..n {
                    let bits = u16::from_le_bytes([b[2 * i], b[2 * i + 1]]);
                    let f = f32::from_bits((bits as u32) << 16);
                    out.extend_from_slice(&f16::from_f32(f).to_bits().to_le_bytes());
                }
                out
            }
            Dtype::F32 => {
                let mut out = Vec::with_capacity(n * 2);
                for i in 0..n {
                    let f =
                        f32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
                    out.extend_from_slice(&f16::from_f32(f).to_bits().to_le_bytes());
                }
                out
            }
        })
    }
}

pub fn read_json(path: &Path) -> Result<J> {
    let bytes = std::fs::read(path)?;
    serde_json::from_slice(&bytes).map_err(|e| HosError::Format(format!("{}: {e}", path.display())))
}
