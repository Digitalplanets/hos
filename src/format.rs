//! The HOS model format (`.hos` files) — HOS's native, self-describing format.
//!
//! Designed for the whole training lifecycle, not just inference: it stores
//! weights AND optimizer state AND a human-readable "card" (arch / training /
//! provenance), so a single file can be a model, a resumable checkpoint, or an
//! experiment archive. Lean + versioned + feature-flagged so it grows without
//! breaking old files. 32-byte aligned tensor data (mmap-friendly), per-tensor
//! + whole-file checksums.
//!
//! v1 layout (little-endian):
//!   "HOSF" | version u32 | flags u32
//!   card_len u32 | card bytes (UTF-8, human-readable JSON-ish)
//!   n_tensors u32
//!   per tensor: name_len u32, name, role u8, ndim u32, dims[u32..],
//!               n_floats u64, checksum u64 (fnv-1a of the f32 bytes)
//!   [pad to 32] then, in order, each tensor's f32 LE bytes [pad to 32]
//!   file_checksum u64 (fnv-1a of everything before it)
//!
//! v2 layout adds a per-tensor `dtype` byte and a stored-byte length, so a
//! tensor's on-disk bytes can be a compact quantization (Q8_0) instead of raw
//! f32 — a smaller, still-self-describing, still-lineaged capsule. Quantized
//! tensors decode back to f32 on `load`, so everything downstream is unchanged.
//! v1 files still load (the reader branches on version).
//!   per tensor: name_len u32, name, role u8, DTYPE u8, ndim u32, dims[u32..],
//!               n_floats u64, N_BYTES u64, checksum u64 (fnv-1a of stored bytes)
//!   [pad to 32] then each tensor's stored bytes [pad to 32]

use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::gguf::{Gguf, GGML_Q4_0, GGML_Q4_K, GGML_Q5_0, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0};

/// The HOS "card": structured lifecycle metadata. This is the moat — what makes
/// a model a self-describing, reproducible, *lineaged* organism instead of a
/// weights dump. Versioned + flexible (arch is free-form JSON) so it evolves.
#[derive(Serialize, Deserialize, Clone)]
pub struct Card {
    pub format: String, // "hos"
    pub spec: u32,      // metadata schema version
    pub name: String,
    pub id: String,              // content hash of the weights = model identity
    pub mode: String,            // "trainable" | "inference" | "frozen"
    pub arch: serde_json::Value, // self-describing architecture spec
    pub provenance: Provenance,
    pub resume: Resume,         // exact-resume state (deterministic continue)
    pub history: Vec<TrainRun>, // every training run, appended over the model's life
    pub lineage: Vec<String>,   // ancestor ids, oldest..parent (genealogy)
    /// Engine-readable metadata (the GGUF-style key/value map) for a *runnable*
    /// capsule — hyperparameters + the tokenizer. `null` for archival capsules.
    /// `#[serde(default)]` keeps older cards loadable.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub meta: serde_json::Value,
}
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct Provenance {
    pub created: String,
    pub engine: String,
    pub dataset: String,
    pub dataset_hash: String,
}
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct Resume {
    pub seed: u64,
    pub step: u64,
    pub rng_state: u64,
}
#[derive(Serialize, Deserialize, Clone)]
pub struct TrainRun {
    pub steps: u64,
    pub final_loss: f32,
    pub optimizer: String,
    pub lr: f32,
}

impl Card {
    pub fn new(name: &str, arch: serde_json::Value) -> Card {
        Card {
            format: "hos".into(),
            spec: 1,
            name: name.into(),
            id: String::new(),
            mode: "trainable".into(),
            arch,
            provenance: Provenance {
                engine: "hos".into(),
                ..Default::default()
            },
            resume: Resume::default(),
            history: Vec::new(),
            lineage: Vec::new(),
            meta: serde_json::Value::Null,
        }
    }
}

/// A model's identity = content hash of all its tensor data.
pub fn model_id(tensors: &[Named]) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for t in tensors {
        for &f in &t.data {
            for b in f.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
    }
    format!("{h:016x}")
}

pub const MAGIC: &[u8; 4] = b"HOSF";
pub const VERSION: u32 = 2;

// tensor roles
pub const ROLE_WEIGHT: u8 = 0;
pub const ROLE_BIAS: u8 = 1;
pub const ROLE_NORM: u8 = 2;
pub const ROLE_EMBED: u8 = 3;
pub const ROLE_OPT_STATE: u8 = 4;
pub const ROLE_SCALAR: u8 = 5;

// per-tensor on-disk dtypes (v2+)
pub const DTYPE_F32: u8 = 0;
pub const DTYPE_Q8_0: u8 = 1;
pub const DTYPE_Q4_0: u8 = 2;
pub const DTYPE_Q5_0: u8 = 3;
pub const DTYPE_Q4_K: u8 = 4;
pub const DTYPE_Q5_K: u8 = 5;
pub const DTYPE_Q6_K: u8 = 6;
pub const DTYPE_HQ4: u8 = 7; // native HOS non-uniform 4-bit (not a ggml type)
pub const DTYPE_Q3: u8 = 8; // native HOS uniform-symmetric 3-bit (not a ggml type)
pub const DTYPE_E8: u8 = 9; // native HOS E8-lattice vector quant (not a ggml type)

/// What to quantize tensors to when writing a capsule.
pub enum QuantTarget {
    F32,
    Block(u32), // a ggml block type (q8_0/q4_0/q5_0/q4_k/q5_k/q6_k)
    Hq4,        // native HOS non-uniform 4-bit
    Q3,         // native HOS uniform-symmetric 3-bit
}

pub fn dtype_to_ggml(dtype: u8) -> Option<u32> {
    match dtype {
        DTYPE_Q8_0 => Some(GGML_Q8_0),
        DTYPE_Q4_0 => Some(GGML_Q4_0),
        DTYPE_Q5_0 => Some(GGML_Q5_0),
        DTYPE_Q4_K => Some(GGML_Q4_K),
        DTYPE_Q5_K => Some(GGML_Q5_K),
        DTYPE_Q6_K => Some(GGML_Q6_K),
        _ => None,
    }
}

pub fn ggml_to_dtype(ggml_type: u32) -> u8 {
    match ggml_type {
        GGML_Q4_0 => DTYPE_Q4_0,
        GGML_Q5_0 => DTYPE_Q5_0,
        GGML_Q4_K => DTYPE_Q4_K,
        GGML_Q5_K => DTYPE_Q5_K,
        GGML_Q6_K => DTYPE_Q6_K,
        _ => DTYPE_Q8_0,
    }
}

/// Which tensors are worth quantizing: the big 2-D weight/embedding matrices
/// whose element count is a clean multiple of the quant's block size. Norms,
/// biases, scalars, and optimizer state stay full-precision (tiny + precision-
/// sensitive) — the same split llama.cpp uses.
fn eligible(role: u8, nfloats: usize, block: usize) -> bool {
    (role == ROLE_WEIGHT || role == ROLE_EMBED) && nfloats >= block && nfloats % block == 0
}

#[derive(Clone)]
pub struct Named {
    pub name: String,
    pub role: u8,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

fn fnv64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn pad32(buf: &mut Vec<u8>) {
    while buf.len() % 32 != 0 {
        buf.push(0);
    }
}

/// Write a `.hos` capsule with raw f32 tensor data.
pub fn save(path: &Path, tensors: &[Named], card: &Card) -> io::Result<()> {
    write(path, tensors, card, QuantTarget::F32)
}

/// Write a `.hos` capsule, quantizing eligible weight/embedding tensors to Q8_0
/// (everything else stays f32). Same card, same lineage — a smaller twin.
pub fn save_quantized(path: &Path, tensors: &[Named], card: &Card) -> io::Result<()> {
    write(path, tensors, card, QuantTarget::Block(GGML_Q8_0))
}

/// Like `save_quantized`, choosing the block quant (`GGML_Q8_0`/`Q4_0`/`Q5_0`).
pub fn save_quantized_as(
    path: &Path,
    tensors: &[Named],
    card: &Card,
    ggml_type: u32,
) -> io::Result<()> {
    write(path, tensors, card, QuantTarget::Block(ggml_type))
}

/// Write a capsule with the native HOS non-uniform 4-bit quant (`hq4`).
pub fn save_hq4(path: &Path, tensors: &[Named], card: &Card) -> io::Result<()> {
    write(path, tensors, card, QuantTarget::Hq4)
}

/// Write a capsule with the native HOS uniform-symmetric 3-bit quant (`q3`).
pub fn save_q3(path: &Path, tensors: &[Named], card: &Card) -> io::Result<()> {
    write(path, tensors, card, QuantTarget::Q3)
}

/// Write a `.hos` whose tensors are ALREADY in their on-disk dtype + bytes (no
/// f32 round-trip, no re-quantization). The symmetric partner of [`load_raw`]:
/// lets a caller that already holds quantized weights (e.g. the Gemma-4 decoder,
/// whose big linears are resident as K-quant bytes) mint a capsule without
/// dequantizing — essential when the f32 expansion would not fit in memory.
pub fn save_raw(path: &Path, tensors: &[RawTensor], card: &Card) -> io::Result<()> {
    let cs = serde_json::to_string(card).expect("serialize card");
    let mut b: Vec<u8> = Vec::new();
    b.extend_from_slice(MAGIC);
    b.extend_from_slice(&VERSION.to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes()); // flags (reserved)
    let card_b = cs.as_bytes();
    b.extend_from_slice(&(card_b.len() as u32).to_le_bytes());
    b.extend_from_slice(card_b);
    b.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    // header block — identical field order to `write()`, so `load_raw` reads it back.
    for t in tensors {
        b.extend_from_slice(&(t.name.len() as u32).to_le_bytes());
        b.extend_from_slice(t.name.as_bytes());
        b.push(t.role);
        b.push(t.dtype);
        b.extend_from_slice(&(t.shape.len() as u32).to_le_bytes());
        for &d in &t.shape {
            b.extend_from_slice(&(d as u32).to_le_bytes());
        }
        b.extend_from_slice(&(t.nfloats as u64).to_le_bytes());
        b.extend_from_slice(&(t.bytes.len() as u64).to_le_bytes());
        b.extend_from_slice(&fnv64(&t.bytes).to_le_bytes());
    }
    pad32(&mut b);
    for t in tensors {
        b.extend_from_slice(&t.bytes);
        pad32(&mut b);
    }
    let file_ck = fnv64(&b);
    b.extend_from_slice(&file_ck.to_le_bytes());
    fs::write(path, &b)
}

/// Cheaply read only a capsule's `Card` (header + card JSON), without touching
/// the tensor payload — for load-time dispatch on `card.arch` of large files.
pub fn read_card(path: &Path) -> io::Result<Card> {
    use std::io::Read;
    let mut f = fs::File::open(path)?;
    let mut hdr = [0u8; 16];
    f.read_exact(&mut hdr)?;
    if &hdr[0..4] != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "not a .hos capsule"));
    }
    let card_len = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
    let mut cb = vec![0u8; card_len];
    f.read_exact(&mut cb)?;
    serde_json::from_slice(&cb).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn write(path: &Path, tensors: &[Named], card: &Card, target: QuantTarget) -> io::Result<()> {
    let cs = serde_json::to_string(card).expect("serialize card");
    let mut b: Vec<u8> = Vec::new();
    b.extend_from_slice(MAGIC);
    b.extend_from_slice(&VERSION.to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes()); // flags (reserved)
    let card_b = cs.as_bytes();
    b.extend_from_slice(&(card_b.len() as u32).to_le_bytes());
    b.extend_from_slice(card_b);
    b.extend_from_slice(&(tensors.len() as u32).to_le_bytes());

    // Encode each tensor's on-disk bytes once (raw f32, or the block quant if
    // eligible).
    let stored: Vec<(u8, Vec<u8>)> = tensors
        .iter()
        .map(|t| {
            let n = t.data.len();
            match &target {
                QuantTarget::Block(ty)
                    if eligible(t.role, n, crate::gguf_write::block_size(*ty)) =>
                {
                    (
                        ggml_to_dtype(*ty),
                        crate::gguf_write::quantize(&t.data, *ty),
                    )
                }
                QuantTarget::Hq4 if eligible(t.role, n, crate::hos_quant::hq4_block()) => {
                    (DTYPE_HQ4, crate::hos_quant::encode_hq4(&t.data))
                }
                QuantTarget::Q3 if eligible(t.role, n, 32) => {
                    (DTYPE_Q3, crate::hos_quant::encode_q3(&t.data))
                }
                _ => (DTYPE_F32, bytemuck_cast(&t.data).to_vec()),
            }
        })
        .collect();

    for (t, (dtype, bytes)) in tensors.iter().zip(&stored) {
        b.extend_from_slice(&(t.name.len() as u32).to_le_bytes());
        b.extend_from_slice(t.name.as_bytes());
        b.push(t.role);
        b.push(*dtype);
        b.extend_from_slice(&(t.shape.len() as u32).to_le_bytes());
        for &d in &t.shape {
            b.extend_from_slice(&(d as u32).to_le_bytes());
        }
        b.extend_from_slice(&(t.data.len() as u64).to_le_bytes()); // logical f32 count
        b.extend_from_slice(&(bytes.len() as u64).to_le_bytes()); // stored byte count
        b.extend_from_slice(&fnv64(bytes).to_le_bytes());
    }
    pad32(&mut b);
    for (_dtype, bytes) in &stored {
        b.extend_from_slice(bytes);
        pad32(&mut b);
    }
    let file_ck = fnv64(&b);
    b.extend_from_slice(&file_ck.to_le_bytes());
    fs::write(path, &b)
}

/// A tensor as stored on disk: its native dtype + raw (still-quantized) bytes,
/// without the f32 decode. Lets a quantized `.hos` stay quantized in memory.
pub struct RawTensor {
    pub name: String,
    pub role: u8,
    pub shape: Vec<usize>,
    pub dtype: u8,
    pub nfloats: usize,
    pub bytes: Vec<u8>,
}

/// Decode one raw tensor's bytes to f32 (dtype dispatch) — shared by `load` and
/// any caller that needs f32 from a `RawTensor`.
pub fn decode_raw(dtype: u8, bytes: &[u8], nfloats: usize) -> io::Result<Vec<f32>> {
    if dtype == DTYPE_HQ4 {
        return Ok(crate::hos_quant::decode_hq4(bytes, nfloats));
    }
    if dtype == DTYPE_Q3 {
        return Ok(crate::hos_quant::decode_q3(bytes, nfloats));
    }
    if dtype == DTYPE_E8 {
        return Ok(crate::hos_quant::decode_e8(bytes, nfloats));
    }
    match dtype_to_ggml(dtype) {
        Some(ggml_type) => {
            let mut d = vec![0f32; nfloats];
            Gguf::dequant_into(bytes, ggml_type, nfloats, &mut d)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            Ok(d)
        }
        None => Ok(bytes
            .chunks_exact(4)
            .map(|q| f32::from_le_bytes([q[0], q[1], q[2], q[3]]))
            .collect()),
    }
}

/// Load a `.hos`, decoding every tensor to f32 (the historical behaviour).
pub fn load(path: &Path) -> io::Result<(Vec<Named>, Card)> {
    let (raws, card) = load_raw(path)?;
    let mut out = Vec::with_capacity(raws.len());
    for r in raws {
        let data = decode_raw(r.dtype, &r.bytes, r.nfloats)?;
        out.push(Named {
            name: r.name,
            role: r.role,
            shape: r.shape,
            data,
        });
    }
    Ok((out, card))
}

/// Load a `.hos` keeping each tensor's on-disk bytes + dtype (no f32 decode).
pub fn load_raw(path: &Path) -> io::Result<(Vec<RawTensor>, Card)> {
    let b = fs::read(path)?;
    let mut c = Cursor { b: &b, p: 0 };
    let magic = c.take(4);
    if magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "not a HOS file"));
    }
    let ver = c.u32();
    let _flags = c.u32();
    let card_len = c.u32() as usize;
    let card_str = String::from_utf8_lossy(c.take(card_len)).into_owned();
    let card: Card = serde_json::from_str(&card_str)
        .unwrap_or_else(|_| Card::new("unknown", serde_json::json!({})));
    let n = c.u32() as usize;

    struct Meta {
        name: String,
        role: u8,
        dtype: u8,
        shape: Vec<usize>,
        nfloats: usize,
        nbytes: usize,
        checksum: u64,
    }
    let mut metas = Vec::with_capacity(n);
    for _ in 0..n {
        let nl = c.u32() as usize;
        let name = String::from_utf8_lossy(c.take(nl)).into_owned();
        let role = c.take(1)[0];
        // v2 adds a dtype byte; v1 tensors are all raw f32.
        let dtype = if ver >= 2 { c.take(1)[0] } else { DTYPE_F32 };
        let ndim = c.u32() as usize;
        let shape: Vec<usize> = (0..ndim).map(|_| c.u32() as usize).collect();
        let nfloats = c.u64() as usize;
        // v2 records the stored byte count; v1 is always nfloats * 4.
        let nbytes = if ver >= 2 {
            c.u64() as usize
        } else {
            nfloats * 4
        };
        let checksum = c.u64();
        metas.push(Meta {
            name,
            role,
            dtype,
            shape,
            nfloats,
            nbytes,
            checksum,
        });
    }
    c.align32();
    use rayon::prelude::*;
    // Slice each tensor's bytes sequentially (cheap — no copy), then verify the
    // per-tensor checksums across cores (they're independent) instead of serially.
    let mut slices: Vec<(usize, &[u8])> = Vec::with_capacity(n);
    for (i, m) in metas.iter().enumerate() {
        let bytes = c.take(m.nbytes);
        c.align32();
        slices.push((i, bytes));
    }
    if let Some(&(bad, _)) = slices
        .par_iter()
        .find_any(|(i, bytes)| fnv64(bytes) != metas[*i].checksum)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("checksum mismatch in tensor {}", metas[bad].name),
        ));
    }
    let out: Vec<RawTensor> = metas
        .into_iter()
        .zip(slices)
        .map(|(m, (_, bytes))| RawTensor {
            name: m.name,
            role: m.role,
            shape: m.shape,
            dtype: m.dtype,
            nfloats: m.nfloats,
            bytes: bytes.to_vec(),
        })
        .collect();
    Ok((out, card))
}

/// Print a file's card + tensor table (the `--hos-info` inspector).
pub fn inspect(path: &Path) -> io::Result<()> {
    let (tensors, card) = load(path)?;
    let bytes = fs::metadata(path)?.len();
    println!(
        "HOS file: {} ({:.2} MB)",
        path.display(),
        bytes as f64 / 1e6
    );
    println!("  name      : {}", card.name);
    println!("  id        : {}", card.id);
    println!("  mode      : {}", card.mode);
    println!("  arch      : {}", card.arch);
    println!(
        "  provenance: engine={} created={} dataset={} hash={}",
        card.provenance.engine,
        card.provenance.created,
        card.provenance.dataset,
        card.provenance.dataset_hash
    );
    println!(
        "  resume    : step={} seed={} rng_state={}",
        card.resume.step, card.resume.seed, card.resume.rng_state
    );
    if card.lineage.is_empty() {
        println!("  lineage   : (root model — no ancestors)");
    } else {
        println!(
            "  lineage   : {} ancestor(s): {}",
            card.lineage.len(),
            card.lineage.join(" -> ")
        );
    }
    println!("  history   : {} training run(s)", card.history.len());
    for (i, r) in card.history.iter().enumerate() {
        println!(
            "              run {i}: {} steps, {} (lr {}), final_loss {:.4}",
            r.steps, r.optimizer, r.lr, r.final_loss
        );
    }
    println!("  tensors   : {} total", tensors.len());
    let roles = ["weight", "bias", "norm", "embed", "opt_state", "scalar"];
    let mut params = 0usize;
    for t in &tensors {
        params += t.data.len();
        let _ = roles;
    }
    println!("              {} values across all tensors", params);
    Ok(())
}

struct Cursor<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> &'a [u8] {
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        s
    }
    fn u32(&mut self) -> u32 {
        let s = self.take(4);
        u32::from_le_bytes([s[0], s[1], s[2], s[3]])
    }
    fn u64(&mut self) -> u64 {
        let s = self.take(8);
        u64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]])
    }
    fn align32(&mut self) {
        while self.p % 32 != 0 {
            self.p += 1;
        }
    }
}

/// reinterpret &[f32] as &[u8] without a dependency
fn bytemuck_cast(data: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantized_hos_roundtrips_and_shrinks() {
        // A weight tensor (Q8_0-eligible), an embedding, and a norm (kept f32).
        let w: Vec<f32> = (0..64 * 32).map(|i| (i as f32 * 0.001).sin()).collect();
        let e: Vec<f32> = (0..10 * 32)
            .map(|i| (i as f32 * 0.002).cos() * 2.0)
            .collect();
        let norm: Vec<f32> = (0..48).map(|i| 1.0 + i as f32 * 0.01).collect();
        let tensors = vec![
            Named {
                name: "blk.0.attn_q.weight".into(),
                role: ROLE_WEIGHT,
                shape: vec![64, 32],
                data: w.clone(),
            },
            Named {
                name: "token_embd.weight".into(),
                role: ROLE_EMBED,
                shape: vec![10, 32],
                data: e.clone(),
            },
            Named {
                name: "blk.0.attn_norm.weight".into(),
                role: ROLE_NORM,
                shape: vec![48],
                data: norm.clone(),
            },
        ];
        let card = Card::new("test", serde_json::json!({}));
        let dir = std::env::temp_dir();
        let f32_path = dir.join("hos_fmt_test_f32.hos");
        let q8_path = dir.join("hos_fmt_test_q8.hos");
        save(&f32_path, &tensors, &card).unwrap();
        save_quantized(&q8_path, &tensors, &card).unwrap();

        // The quantized capsule must be meaningfully smaller.
        let s_f32 = std::fs::metadata(&f32_path).unwrap().len();
        let s_q8 = std::fs::metadata(&q8_path).unwrap().len();
        assert!(s_q8 < s_f32, "q8 {s_q8} not smaller than f32 {s_f32}");

        // Round-trip: norms exact, weights within Q8_0's per-block bound.
        let (loaded, _) = load(&q8_path).unwrap();
        for (orig, got) in tensors.iter().zip(&loaded) {
            assert_eq!(orig.data.len(), got.data.len());
            if orig.role == ROLE_NORM {
                assert_eq!(orig.data, got.data, "norm must be lossless f32");
            } else {
                for blk in orig.data.chunks(32).zip(got.data.chunks(32)) {
                    let amax = blk.0.iter().fold(0f32, |m, &v| m.max(v.abs()));
                    let tol = amax / 127.0 + 1e-4;
                    for (a, b) in blk.0.iter().zip(blk.1) {
                        assert!((a - b).abs() <= tol, "|{a} - {b}| > {tol}");
                    }
                }
            }
        }
        let _ = std::fs::remove_file(&f32_path);
        let _ = std::fs::remove_file(&q8_path);
    }
}
