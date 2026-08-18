//! Minimal GGUF v2/v3 reader + dequantization.
//!
//! GGUF layout:
//!   magic u32 ("GGUF") | version u32 | tensor_count u64 | metadata_kv_count u64
//!   metadata KVs       (key string, value_type u32, value)
//!   tensor infos       (name string, n_dims u32, dims[u64], ggml_type u32, offset u64)
//!   padding to general.alignment (default 32)
//!   tensor data
//!
//! ref: https://github.com/ggml-org/llama.cpp/blob/master/docs/development/gguf.md

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use half::f16;
use memmap2::Mmap;

use crate::error::{HosError, Result};

// ---- ggml tensor types we support for v0 ----
pub const GGML_F32: u32 = 0;
pub const GGML_F16: u32 = 1;
pub const GGML_Q4_0: u32 = 2;
pub const GGML_Q5_0: u32 = 6;
pub const GGML_Q8_0: u32 = 8;
pub const GGML_Q4_K: u32 = 12;
pub const GGML_Q5_K: u32 = 13;
pub const GGML_Q6_K: u32 = 14;
pub const GGML_BF16: u32 = 30;

const QK: usize = 32; // block size for the simple (non-K) quant formats
const QK_K: usize = 256; // super-block size for K-quants

#[inline]
fn f16(bytes: &[u8], at: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([bytes[at], bytes[at + 1]])).to_f32()
}

/// Unpack a 6-bit scale + 6-bit min from the packed 12-byte `scales` field.
/// (ggml get_scale_min_k4)
#[inline]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Vec<Value>),
}

impl Value {
    /// Convert to a serde_json value (for embedding GGUF metadata in a `.hos` card).
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Value::U8(x) => (*x).into(),
            Value::I8(x) => (*x).into(),
            Value::U16(x) => (*x).into(),
            Value::I16(x) => (*x).into(),
            Value::U32(x) => (*x).into(),
            Value::I32(x) => (*x).into(),
            Value::U64(x) => (*x).into(),
            Value::I64(x) => (*x).into(),
            Value::F32(x) => (*x).into(),
            Value::F64(x) => (*x).into(),
            Value::Bool(x) => (*x).into(),
            Value::Str(s) => s.clone().into(),
            Value::Array(a) => serde_json::Value::Array(a.iter().map(Value::to_json).collect()),
        }
    }
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        Some(match self {
            Value::U8(v) => *v as u64,
            Value::U16(v) => *v as u64,
            Value::U32(v) => *v as u64,
            Value::U64(v) => *v,
            Value::I8(v) => *v as u64,
            Value::I16(v) => *v as u64,
            Value::I32(v) => *v as u64,
            Value::I64(v) => *v as u64,
            _ => return None,
        })
    }
    pub fn as_f32(&self) -> Option<f32> {
        Some(match self {
            Value::F32(v) => *v,
            Value::F64(v) => *v as f32,
            _ => return None,
        })
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64, // relative to the start of the data section
}

impl TensorInfo {
    /// Element count = product of dims. Saturates to `usize::MAX` on overflow so
    /// a corrupt/adversarial dim list can't wrap to a small value (callers reject
    /// implausible counts against the file size).
    pub fn n_elements(&self) -> usize {
        self.dims
            .iter()
            .try_fold(1usize, |a, &d| a.checked_mul(d as usize))
            .unwrap_or(usize::MAX)
    }
}

pub struct Gguf {
    pub mmap: Mmap,
    pub metadata: HashMap<String, Value>,
    pub tensors: HashMap<String, TensorInfo>,
    pub data_offset: usize,
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    /// Ensure `n` more bytes are available before reading — turns truncated /
    /// corrupt files into a clean error instead of an out-of-bounds panic.
    fn need(&self, n: usize) -> Result<()> {
        if self.pos + n > self.buf.len() {
            Err(HosError::Format(format!(
                "unexpected end of file at byte {} (needed {n} more)",
                self.pos
            )))
        } else {
            Ok(())
        }
    }
    fn u8(&mut self) -> Result<u8> {
        self.need(1)?;
        let v = self.buf[self.pos];
        self.pos += 1;
        Ok(v)
    }
    fn read<const N: usize>(&mut self) -> Result<[u8; N]> {
        self.need(N)?;
        let mut a = [0u8; N];
        a.copy_from_slice(&self.buf[self.pos..self.pos + N]);
        self.pos += N;
        Ok(a)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read::<4>()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.read::<8>()?))
    }
    fn string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        self.need(len)?;
        let s = String::from_utf8_lossy(&self.buf[self.pos..self.pos + len]).into_owned();
        self.pos += len;
        Ok(s)
    }
    fn value(&mut self, vtype: u32) -> Result<Value> {
        Ok(match vtype {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.u8()? as i8),
            2 => Value::U16(u16::from_le_bytes(self.read::<2>()?)),
            3 => Value::I16(i16::from_le_bytes(self.read::<2>()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_le_bytes(self.read::<4>()?)),
            7 => Value::Bool(self.u8()? != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let elem_type = self.u32()?;
                let count = self.u64()? as usize;
                let mut out = Vec::with_capacity(count.min(1 << 20));
                for _ in 0..count {
                    out.push(self.value(elem_type)?);
                }
                Value::Array(out)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(i64::from_le_bytes(self.read::<8>()?)),
            12 => Value::F64(f64::from_le_bytes(self.read::<8>()?)),
            other => return Err(HosError::Format(format!("unknown gguf value type {other}"))),
        })
    }
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Gguf> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let (metadata, tensors, data_offset) = {
            let mut r = Reader { buf: &mmap, pos: 0 };
            let magic = r.u32()?;
            if &magic.to_le_bytes() != b"GGUF" {
                return Err(HosError::Format("not a GGUF file (bad magic)".into()));
            }
            let version = r.u32()?;
            if version != 2 && version != 3 {
                return Err(HosError::Format(format!(
                    "unsupported GGUF version {version}"
                )));
            }
            let tensor_count = r.u64()?;
            let kv_count = r.u64()?;

            let mut metadata = HashMap::new();
            for _ in 0..kv_count {
                let key = r.string()?;
                let vtype = r.u32()?;
                let val = r.value(vtype)?;
                metadata.insert(key, val);
            }

            let mut tensors = HashMap::new();
            for _ in 0..tensor_count {
                let name = r.string()?;
                let n_dims = r.u32()? as usize;
                let mut dims = Vec::with_capacity(n_dims);
                for _ in 0..n_dims {
                    dims.push(r.u64()?);
                }
                let ggml_type = r.u32()?;
                let offset = r.u64()?;
                tensors.insert(
                    name,
                    TensorInfo {
                        dims,
                        ggml_type,
                        offset,
                    },
                );
            }

            let alignment = metadata
                .get("general.alignment")
                .and_then(|v| v.as_u64())
                .unwrap_or(32) as usize;
            let data_offset = r.pos.div_ceil(alignment) * alignment;
            (metadata, tensors, data_offset)
        };

        Ok(Gguf {
            mmap,
            metadata,
            tensors,
            data_offset,
        })
    }

    pub fn meta_u64(&self, key: &str) -> Option<u64> {
        self.metadata.get(key).and_then(|v| v.as_u64())
    }
    pub fn meta_f32(&self, key: &str) -> Option<f32> {
        self.metadata.get(key).and_then(|v| v.as_f32())
    }
    pub fn meta_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(|v| v.as_str())
    }

    /// Dequantize a named tensor to a flat f32 vector (row-major over dims).
    pub fn dequant(&self, name: &str) -> Result<Vec<f32>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| HosError::MissingTensor(name.to_string()))?;
        let n = info.n_elements();
        // Reject implausible element counts before allocating (every supported
        // format stores >= ~0.5 byte/element, so n can't exceed 2x the file).
        if n > self.mmap.len().saturating_mul(2) {
            return Err(HosError::Format(format!(
                "tensor '{name}' element count {n} exceeds file size"
            )));
        }
        // `bytes_for` validates the quant type AND tells us exactly how many
        // bytes this tensor occupies, so we can reject a truncated file up front.
        let required = bytes_for(info.ggml_type, n)?;
        let start = self.data_offset.saturating_add(info.offset as usize);
        let end = start.saturating_add(required);
        if end > self.mmap.len() {
            return Err(HosError::Format(format!(
                "tensor '{name}' data out of bounds (need {required} bytes at offset {start})"
            )));
        }
        let mut out = vec![0f32; n];
        Self::dequant_into(&self.mmap[start..end], info.ggml_type, n, &mut out)?;
        Ok(out)
    }

    /// Dequantize `n` elements from a raw quantized byte slice into `out`
    /// (row-major). Single source of truth shared by `dequant` and the
    /// on-the-fly CPU-quant matvec (so big/MoE models needn't expand to f32).
    pub fn dequant_into(bytes: &[u8], ggml_type: u32, n: usize, out: &mut [f32]) -> Result<()> {
        match ggml_type {
            GGML_F32 => {
                for i in 0..n {
                    out[i] = f32::from_le_bytes([
                        bytes[i * 4],
                        bytes[i * 4 + 1],
                        bytes[i * 4 + 2],
                        bytes[i * 4 + 3],
                    ]);
                }
            }
            GGML_F16 => {
                for i in 0..n {
                    let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                    out[i] = f16::from_bits(bits).to_f32();
                }
            }
            GGML_BF16 => {
                for i in 0..n {
                    let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                    out[i] = half::bf16::from_bits(bits).to_f32();
                }
            }
            GGML_Q8_0 => {
                // block: f16 scale (2 bytes) + 32 * i8
                let block_bytes = 2 + QK;
                let n_blocks = n / QK;
                for b in 0..n_blocks {
                    let base = b * block_bytes;
                    let d =
                        f16::from_bits(u16::from_le_bytes([bytes[base], bytes[base + 1]])).to_f32();
                    for j in 0..QK {
                        let q = bytes[base + 2 + j] as i8 as f32;
                        out[b * QK + j] = q * d;
                    }
                }
            }
            GGML_Q4_0 => {
                // block: f16 scale (2 bytes) + 16 bytes (32 packed 4-bit, low nibbles first)
                let block_bytes = 2 + QK / 2;
                let n_blocks = n / QK;
                for b in 0..n_blocks {
                    let base = b * block_bytes;
                    let d =
                        f16::from_bits(u16::from_le_bytes([bytes[base], bytes[base + 1]])).to_f32();
                    for j in 0..QK / 2 {
                        let q = bytes[base + 2 + j];
                        let lo = (q & 0x0F) as i32 - 8;
                        let hi = (q >> 4) as i32 - 8;
                        out[b * QK + j] = lo as f32 * d;
                        out[b * QK + j + QK / 2] = hi as f32 * d;
                    }
                }
            }
            GGML_Q4_K => {
                // super-block: d(f16) dmin(f16) scales[12] qs[128]  -> 256 values
                let block_bytes = 2 + 2 + 12 + 128;
                let n_blocks = n / QK_K;
                for b in 0..n_blocks {
                    let base = b * block_bytes;
                    let d = f16(bytes, base);
                    let dmin = f16(bytes, base + 2);
                    let scales = &bytes[base + 4..base + 16];
                    let qs = &bytes[base + 16..base + 16 + 128];
                    let mut y = b * QK_K;
                    let mut is = 0;
                    for j in (0..QK_K).step_by(64) {
                        let (sc1, m1) = scale_min_k4(is, scales);
                        let (sc2, m2) = scale_min_k4(is + 1, scales);
                        let (d1, mn1) = (d * sc1 as f32, dmin * m1 as f32);
                        let (d2, mn2) = (d * sc2 as f32, dmin * m2 as f32);
                        let q = &qs[j / 2..j / 2 + 32];
                        for l in 0..32 {
                            out[y + l] = d1 * (q[l] & 0x0F) as f32 - mn1;
                        }
                        for l in 0..32 {
                            out[y + 32 + l] = d2 * (q[l] >> 4) as f32 - mn2;
                        }
                        y += 64;
                        is += 2;
                    }
                }
            }
            GGML_Q5_K => {
                // super-block: d(f16) dmin(f16) scales[12] qh[32] qs[128] -> 256 values
                let block_bytes = 2 + 2 + 12 + 32 + 128;
                let n_blocks = n / QK_K;
                for b in 0..n_blocks {
                    let base = b * block_bytes;
                    let d = f16(bytes, base);
                    let dmin = f16(bytes, base + 2);
                    let scales = &bytes[base + 4..base + 16];
                    let qh = &bytes[base + 16..base + 16 + 32];
                    let qs = &bytes[base + 48..base + 48 + 128];
                    let mut y = b * QK_K;
                    let mut is = 0;
                    let (mut u1, mut u2) = (1u8, 2u8);
                    for j in (0..QK_K).step_by(64) {
                        let (sc1, m1) = scale_min_k4(is, scales);
                        let (sc2, m2) = scale_min_k4(is + 1, scales);
                        let (d1, mn1) = (d * sc1 as f32, dmin * m1 as f32);
                        let (d2, mn2) = (d * sc2 as f32, dmin * m2 as f32);
                        let q = &qs[j / 2..j / 2 + 32];
                        for l in 0..32 {
                            let hi = if qh[l] & u1 != 0 { 16 } else { 0 };
                            out[y + l] = d1 * ((q[l] & 0x0F) as f32 + hi as f32) - mn1;
                        }
                        for l in 0..32 {
                            let hi = if qh[l] & u2 != 0 { 16 } else { 0 };
                            out[y + 32 + l] = d2 * ((q[l] >> 4) as f32 + hi as f32) - mn2;
                        }
                        y += 64;
                        is += 2;
                        u1 <<= 2;
                        u2 <<= 2;
                    }
                }
            }
            GGML_Q6_K => {
                // super-block: ql[128] qh[64] scales[16](i8) d(f16) -> 256 values
                let block_bytes = 128 + 64 + 16 + 2;
                let n_blocks = n / QK_K;
                for b in 0..n_blocks {
                    let base = b * block_bytes;
                    let ql = &bytes[base..base + 128];
                    let qh = &bytes[base + 128..base + 128 + 64];
                    let sc = &bytes[base + 192..base + 192 + 16];
                    let d = f16(bytes, base + 208);
                    for n128 in 0..2 {
                        let ql = &ql[n128 * 64..];
                        let qh = &qh[n128 * 32..];
                        let sc = &sc[n128 * 8..];
                        let y0 = b * QK_K + n128 * 128;
                        for l in 0..32 {
                            let is = l / 16;
                            let q1 = ((ql[l] & 0x0F) | ((qh[l] & 3) << 4)) as i32 - 32;
                            let q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                            out[y0 + l] = d * sc[is] as i8 as f32 * q1 as f32;
                            out[y0 + l + 32] = d * sc[is + 2] as i8 as f32 * q2 as f32;
                            out[y0 + l + 64] = d * sc[is + 4] as i8 as f32 * q3 as f32;
                            out[y0 + l + 96] = d * sc[is + 6] as i8 as f32 * q4 as f32;
                        }
                    }
                }
            }
            GGML_Q5_0 => {
                // block: f16 d + qh[4] (5th bits) + qs[16]  -> 32 values
                let block_bytes = 2 + 4 + 16;
                let n_blocks = n / 32;
                for b in 0..n_blocks {
                    let base = b * block_bytes;
                    let d = f16(bytes, base);
                    let qh = u32::from_le_bytes([
                        bytes[base + 2],
                        bytes[base + 3],
                        bytes[base + 4],
                        bytes[base + 5],
                    ]);
                    for j in 0..16 {
                        let xh0 = (((qh >> j) << 4) & 0x10) as i32;
                        let xh1 = ((qh >> (j + 12)) & 0x10) as i32;
                        let q = bytes[base + 6 + j];
                        let x0 = ((q & 0x0F) as i32 | xh0) - 16;
                        let x1 = ((q >> 4) as i32 | xh1) - 16;
                        out[b * 32 + j] = x0 as f32 * d;
                        out[b * 32 + j + 16] = x1 as f32 * d;
                    }
                }
            }
            other => return Err(HosError::UnsupportedQuant(other)),
        }
        Ok(())
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Raw (still-quantized) bytes for a tensor: (bytes, ggml_type, n_elements).
    /// Used to upload weights to the GPU without expanding to f32.
    pub fn raw(&self, name: &str) -> Result<(&[u8], u32, usize)> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| HosError::MissingTensor(name.to_string()))?;
        let n = info.n_elements();
        if n > self.mmap.len().saturating_mul(2) {
            return Err(HosError::Format(format!(
                "tensor '{name}' element count {n} exceeds file size"
            )));
        }
        let nbytes = bytes_for(info.ggml_type, n)?;
        let start = self.data_offset.saturating_add(info.offset as usize);
        let end = start.saturating_add(nbytes);
        if end > self.mmap.len() {
            return Err(HosError::Format(format!(
                "tensor '{name}' data out of bounds"
            )));
        }
        Ok((&self.mmap[start..end], info.ggml_type, n))
    }
}

/// Number of bytes a tensor of `n` elements occupies in the given ggml type.
/// Block size (in elements) of a fusable quant type — the granularity at which a
/// row must be aligned to dequantize it block-by-block. `None` for types we don't
/// fuse (F32/F16/unknown), which fall back to a plain f32 weight.
pub fn block_elems(ggml_type: u32) -> Option<usize> {
    match ggml_type {
        GGML_Q8_0 | GGML_Q4_0 | GGML_Q5_0 => Some(32),
        GGML_Q4_K | GGML_Q5_K | GGML_Q6_K => Some(QK_K),
        _ => None,
    }
}

pub fn bytes_for(ggml_type: u32, n: usize) -> Result<usize> {
    Ok(match ggml_type {
        GGML_F32 => n * 4,
        GGML_F16 => n * 2,
        GGML_BF16 => n * 2,
        GGML_Q8_0 => n / 32 * 34,
        GGML_Q4_0 => n / 32 * 18,
        GGML_Q5_0 => n / 32 * 22,
        GGML_Q4_K => n / QK_K * 144,
        GGML_Q5_K => n / QK_K * 176,
        GGML_Q6_K => n / QK_K * 210,
        other => return Err(HosError::UnsupportedQuant(other)),
    })
}
