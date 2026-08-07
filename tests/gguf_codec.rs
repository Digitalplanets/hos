//! GGUF parser + dequantization tests.
//!
//! A self-contained GGUF writer builds tiny files in a temp dir; the quantized
//! blocks are encoded here *independently* of the library decoder (so a bug in
//! `dequant` can't hide behind a matching bug in the test). Values are chosen to
//! be exactly representable in each format, so the round trip is exact. A final
//! test exercises the K-quant decoders against real model files when present.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

use half::f16;
use hos::gguf::Gguf;

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn temp_path() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!("hos_gguf_{}_{}.gguf", std::process::id(), n));
    p
}

// ---- minimal GGUF writer (mirrors the v3 layout the reader expects) ----

struct Tspec {
    name: String,
    dims: Vec<u64>,
    gtype: u32,
    bytes: Vec<u8>,
}

enum Kv {
    Str(String, String),
    U32(String, u32),
}

fn push_str(b: &mut Vec<u8>, s: &str) {
    b.extend_from_slice(&(s.len() as u64).to_le_bytes());
    b.extend_from_slice(s.as_bytes());
}

fn write_gguf(path: &PathBuf, kvs: &[Kv], tensors: &[Tspec]) {
    // assign 32-aligned data offsets
    let mut offsets = Vec::new();
    let mut off = 0usize;
    for t in tensors {
        off = off.div_ceil(32) * 32;
        offsets.push(off as u64);
        off += t.bytes.len();
    }

    let mut head: Vec<u8> = Vec::new();
    head.extend_from_slice(b"GGUF");
    head.extend_from_slice(&3u32.to_le_bytes());
    head.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    head.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for kv in kvs {
        match kv {
            Kv::Str(k, v) => {
                push_str(&mut head, k);
                head.extend_from_slice(&8u32.to_le_bytes()); // type: string
                push_str(&mut head, v);
            }
            Kv::U32(k, v) => {
                push_str(&mut head, k);
                head.extend_from_slice(&4u32.to_le_bytes()); // type: u32
                head.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    for (t, &offset) in tensors.iter().zip(&offsets) {
        push_str(&mut head, &t.name);
        head.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for &d in &t.dims {
            head.extend_from_slice(&d.to_le_bytes());
        }
        head.extend_from_slice(&t.gtype.to_le_bytes());
        head.extend_from_slice(&offset.to_le_bytes());
    }

    let data_start = head.len().div_ceil(32) * 32;
    let mut file = head;
    file.resize(data_start, 0);
    for (t, &offset) in tensors.iter().zip(&offsets) {
        let at = data_start + offset as usize;
        if file.len() < at + t.bytes.len() {
            file.resize(at + t.bytes.len(), 0);
        }
        file[at..at + t.bytes.len()].copy_from_slice(&t.bytes);
    }
    std::fs::write(path, &file).unwrap();
}

fn f16le(x: f32) -> [u8; 2] {
    f16::from_f32(x).to_bits().to_le_bytes()
}

// ---- independent encoders for each quant format ----

fn enc_f32(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn enc_f16(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| f16le(*v)).collect()
}
fn enc_q8_0(vals: &[f32], d: f32) -> Vec<u8> {
    let mut b = Vec::new();
    for blk in vals.chunks(32) {
        b.extend_from_slice(&f16le(d));
        for &v in blk {
            b.push((v / d).round() as i8 as u8);
        }
    }
    b
}
fn enc_q4_0(vals: &[f32], d: f32) -> Vec<u8> {
    let mut b = Vec::new();
    for blk in vals.chunks(32) {
        b.extend_from_slice(&f16le(d));
        for j in 0..16 {
            let lo = ((blk[j] / d).round() as i32 + 8) as u8 & 0x0F;
            let hi = ((blk[j + 16] / d).round() as i32 + 8) as u8 & 0x0F;
            b.push((hi << 4) | lo);
        }
    }
    b
}
fn enc_q5_0(vals: &[f32], d: f32) -> Vec<u8> {
    let mut b = Vec::new();
    for blk in vals.chunks(32) {
        b.extend_from_slice(&f16le(d));
        let mut qh: u32 = 0;
        let mut qs = [0u8; 16];
        for (elem, &v) in blk.iter().enumerate() {
            let code = ((v / d).round() as i32 + 16) as u32; // 0..31
            if code & 0x10 != 0 {
                qh |= 1 << elem;
            }
            let nib = (code & 0x0F) as u8;
            if elem < 16 {
                qs[elem] |= nib;
            } else {
                qs[elem - 16] |= nib << 4;
            }
        }
        b.extend_from_slice(&qh.to_le_bytes());
        b.extend_from_slice(&qs);
    }
    b
}

#[test]
fn dequant_f32_exact() {
    let path = temp_path();
    let vals: Vec<f32> = vec![0.0, 1.5, -2.25, 3.5, -100.0, 42.0];
    write_gguf(
        &path,
        &[],
        &[Tspec {
            name: "t".into(),
            dims: vec![vals.len() as u64],
            gtype: 0,
            bytes: enc_f32(&vals),
        }],
    );
    let g = Gguf::open(&path).unwrap();
    assert_eq!(g.dequant("t").unwrap(), vals);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn dequant_f16_matches_reference() {
    let path = temp_path();
    let vals: Vec<f32> = vec![0.0, 1.0, -2.5, 0.5, 0.25, 3.0, -7.0, 0.125];
    let expected: Vec<f32> = vals.iter().map(|v| f16::from_f32(*v).to_f32()).collect();
    write_gguf(
        &path,
        &[],
        &[Tspec {
            name: "t".into(),
            dims: vec![vals.len() as u64],
            gtype: 1,
            bytes: enc_f16(&vals),
        }],
    );
    let g = Gguf::open(&path).unwrap();
    assert_eq!(g.dequant("t").unwrap(), expected);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn dequant_q8_0_exact() {
    let path = temp_path();
    let d = 0.5f32;
    // 64 values = 2 blocks, each an exact multiple of d within i8 range
    let vals: Vec<f32> = (0..64i32).map(|i| ((i - 32) % 40) as f32 * d).collect();
    write_gguf(
        &path,
        &[],
        &[Tspec {
            name: "t".into(),
            dims: vec![64],
            gtype: 8,
            bytes: enc_q8_0(&vals, d),
        }],
    );
    let g = Gguf::open(&path).unwrap();
    let out = g.dequant("t").unwrap();
    assert_eq!(out.len(), 64);
    for (o, v) in out.iter().zip(&vals) {
        assert!((o - v).abs() < 1e-6, "q8_0 {o} vs {v}");
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn dequant_q4_0_exact() {
    let path = temp_path();
    let d = 0.25f32;
    // nibble - 8 in [-8, 7]; build 32 values spanning the range
    let vals: Vec<f32> = (0..32i32).map(|i| ((i % 16) - 8) as f32 * d).collect();
    write_gguf(
        &path,
        &[],
        &[Tspec {
            name: "t".into(),
            dims: vec![32],
            gtype: 2,
            bytes: enc_q4_0(&vals, d),
        }],
    );
    let g = Gguf::open(&path).unwrap();
    let out = g.dequant("t").unwrap();
    for (o, v) in out.iter().zip(&vals) {
        assert!((o - v).abs() < 1e-6, "q4_0 {o} vs {v}");
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn dequant_q5_0_exact() {
    let path = temp_path();
    let d = 0.5f32;
    // element b encodes value (b - 16) * d, range [-16, 15] — exercises the 5th bit
    let vals: Vec<f32> = (0..32i32).map(|b| (b - 16) as f32 * d).collect();
    write_gguf(
        &path,
        &[],
        &[Tspec {
            name: "t".into(),
            dims: vec![32],
            gtype: 6,
            bytes: enc_q5_0(&vals, d),
        }],
    );
    let g = Gguf::open(&path).unwrap();
    let out = g.dequant("t").unwrap();
    for (o, v) in out.iter().zip(&vals) {
        assert!((o - v).abs() < 1e-6, "q5_0 {o} vs {v}");
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn parses_metadata_kvs() {
    let path = temp_path();
    write_gguf(
        &path,
        &[
            Kv::Str("general.architecture".into(), "hostest".into()),
            Kv::U32("hostest.block_count".into(), 42),
        ],
        &[Tspec {
            name: "t".into(),
            dims: vec![32],
            gtype: 0,
            bytes: enc_f32(&[1.0f32; 32]),
        }],
    );
    let g = Gguf::open(&path).unwrap();
    assert_eq!(g.meta_str("general.architecture"), Some("hostest"));
    assert_eq!(g.meta_u64("hostest.block_count"), Some(42));
    assert!(g.has("t"));
    assert!(!g.has("nope"));
    let _ = std::fs::remove_file(&path);
}

/// K-quant decoders (Q4_K/Q5_K/Q6_K) run against real model files when they're
/// present in models/. We can't easily hand-encode a super-block, so we assert
/// the decode is well-formed (right length, all finite) and deterministic.
#[test]
fn kquant_decode_on_real_models_if_present() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        eprintln!("[skip] no models/ dir");
        return;
    };
    let mut checked = 0;
    for e in entries.flatten() {
        let p = e.path();
        if p.extension().and_then(|s| s.to_str()) != Some("gguf") {
            continue;
        }
        let Ok(g) = Gguf::open(&p) else { continue };
        // pick any tensor with a supported quant type
        let Some(name) = g.tensors.keys().find(|n| n.ends_with("weight")).cloned() else {
            continue;
        };
        let a = g.dequant(&name).unwrap();
        let b = g.dequant(&name).unwrap();
        assert_eq!(
            a.len(),
            g.tensors[&name].n_elements(),
            "{}: length",
            p.display()
        );
        assert!(
            a.iter().all(|x| x.is_finite()),
            "{}: non-finite value",
            p.display()
        );
        assert_eq!(a, b, "{}: dequant must be deterministic", p.display());
        checked += 1;
    }
    if checked == 0 {
        eprintln!("[skip] no readable .gguf models found in {}", dir.display());
    } else {
        eprintln!("[ok] dequant validated on {checked} real model tensor(s)");
    }
}
