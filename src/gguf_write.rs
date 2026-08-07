//! GGUF writer + Q8_0 quantizer.
//!
//! HOS reads many quant formats but, until now, only ever *decoded* them. This
//! is the encode direction: take a parsed [`Gguf`], requantize its tensors, and
//! write a new, valid GGUF the engine (and llama.cpp) can load.
//!
//! The first, safest target is **GGUF → GGUF Q8_0**. Metadata is copied
//! **verbatim** — including the tokenizer arrays — so nothing needs to be
//! synthesized and the result is a faithful, smaller twin of the source. Q8_0
//! is near-lossless (an 8-bit linear block code), so requantizing preserves
//! quality; we verify that with perplexity. K-quant *encoders* and HF → GGUF
//! quantization (which must synthesize tokenizer metadata) are deliberately not
//! attempted here.

use std::path::Path;

use half::f16;

use crate::error::{HosError, Result};
use crate::gguf::{
    Gguf, Value, GGML_F32, GGML_Q4_0, GGML_Q4_K, GGML_Q5_0, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0,
};

const QK: usize = 32; // block size for the simple (non-K) quants

/// Map a target name to its ggml type.
pub fn target_type(name: &str) -> Option<u32> {
    match name {
        "q8_0" => Some(GGML_Q8_0),
        "q4_0" => Some(GGML_Q4_0),
        "q5_0" => Some(GGML_Q5_0),
        "q4_k" => Some(GGML_Q4_K),
        "q5_k" => Some(GGML_Q5_K),
        "q6_k" => Some(GGML_Q6_K),
        _ => None,
    }
}

/// Quantize a flat f32 tensor to the given ggml block type. `x.len()` must be a
/// multiple of the type's block size (the caller's gate guarantees it).
pub fn quantize(x: &[f32], ggml_type: u32) -> Vec<u8> {
    match ggml_type {
        GGML_Q4_0 => quantize_q4_0(x),
        GGML_Q5_0 => quantize_q5_0(x),
        GGML_Q4_K => quantize_q4_k(x),
        GGML_Q5_K => quantize_q5_k(x),
        GGML_Q6_K => quantize_q6_k(x),
        _ => quantize_q8_0(x),
    }
}

/// What a requantize pass did.
pub struct QuantStats {
    pub tensors: usize,
    pub quantized: usize,
    pub src_bytes: u64,
    pub out_bytes: u64,
}

/// Requantize every quantizable tensor of `src` to `target` and write a new
/// GGUF at `out_path`. Metadata is preserved verbatim. Only `GGML_Q8_0` is
/// supported as a target today.
///
/// A tensor is quantized when it is at least 2-D and its contiguous dimension
/// (`dims[0]`, ggml's `ne[0]`) is a multiple of the block size — exactly ggml's
/// own constraint. Everything else (1-D norms and biases) is stored as F32, as
/// llama.cpp does, so no precision-sensitive vector is block-quantized.
pub fn requantize_gguf(src: &Gguf, out_path: &Path, target: u32) -> Result<QuantStats> {
    if !matches!(
        target,
        GGML_Q8_0 | GGML_Q4_0 | GGML_Q5_0 | GGML_Q4_K | GGML_Q5_K | GGML_Q6_K
    ) {
        return Err(HosError::Format(
            "requantize: target must be q8_0|q4_0|q5_0|q4_k|q5_k|q6_k".into(),
        ));
    }
    let block = block_size(target) as u64;
    let alignment = src
        .metadata
        .get("general.alignment")
        .and_then(|v| v.as_u64())
        .unwrap_or(32)
        .max(1);

    struct OutT {
        name: String,
        dims: Vec<u64>,
        ty: u32,
        data: Vec<u8>,
        offset: u64,
    }

    // Deterministic tensor order (HashMap is unordered; order is functionally
    // irrelevant since readers map by name, but determinism keeps output stable).
    let mut names: Vec<&String> = src.tensors.keys().collect();
    names.sort();

    let mut outs = Vec::with_capacity(names.len());
    let mut quantized = 0usize;
    for name in names {
        let info = &src.tensors[name];
        let f = src.dequant(name)?; // any source quant → f32
        let quantizable = info.dims.len() >= 2 && info.dims[0] % block == 0;
        let (ty, data) = if quantizable {
            quantized += 1;
            (target, quantize(&f, target))
        } else {
            let mut buf = Vec::with_capacity(f.len() * 4);
            for v in &f {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            (GGML_F32, buf)
        };
        outs.push(OutT {
            name: name.clone(),
            dims: info.dims.clone(),
            ty,
            data,
            offset: 0,
        });
    }

    // Assign aligned, data-section-relative offsets.
    let mut cursor = 0u64;
    for t in &mut outs {
        t.offset = cursor;
        cursor += t.data.len() as u64;
        cursor = pad(cursor, alignment);
    }

    // ---- serialize ----
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes()); // version
    buf.extend_from_slice(&(outs.len() as u64).to_le_bytes());
    buf.extend_from_slice(&(src.metadata.len() as u64).to_le_bytes());

    for (k, v) in &src.metadata {
        write_string(&mut buf, k);
        write_value(&mut buf, v);
    }
    for t in &outs {
        write_string(&mut buf, &t.name);
        buf.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for &d in &t.dims {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        buf.extend_from_slice(&t.ty.to_le_bytes());
        buf.extend_from_slice(&t.offset.to_le_bytes());
    }

    // Pad the header+metadata+infos block to alignment; that is the data start.
    while buf.len() as u64 % alignment != 0 {
        buf.push(0);
    }
    let data_start = buf.len() as u64;
    for t in &outs {
        let want = data_start + t.offset;
        while (buf.len() as u64) < want {
            buf.push(0);
        }
        buf.extend_from_slice(&t.data);
    }

    std::fs::write(out_path, &buf)?;
    Ok(QuantStats {
        tensors: outs.len(),
        quantized,
        src_bytes: src.mmap.len() as u64,
        out_bytes: buf.len() as u64,
    })
}

/// Quantize a flat f32 tensor to Q8_0: per 32-element block, one f16 scale
/// `d = max|x| / 127` followed by 32 int8 quants `round(x / d)`. `x.len()` must
/// be a multiple of 32 (guaranteed by the `dims[0] % 32 == 0` gate). Shared with
/// the `.hos` writer (`format.rs`) so both encoders are one implementation.
pub(crate) fn quantize_q8_0(x: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(x.len() / QK * (2 + QK));
    for block in x.chunks(QK) {
        let amax = block.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        for &v in block {
            let q = (v * id).round().clamp(-127.0, 127.0) as i8;
            out.push(q as u8);
        }
        // Pad a short final block (only if the tensor wasn't a clean multiple,
        // which the caller's gate prevents — kept for total safety).
        if block.len() < QK {
            out.resize(out.len() + (QK - block.len()), 0);
        }
    }
    out
}

/// Q4_0: per 32-block, one f16 scale `d = max / -8` then 32 packed 4-bit quants
/// (`(nibble - 8) * d`), low nibble = element j, high nibble = element j+16.
pub(crate) fn quantize_q4_0(x: &[f32]) -> Vec<u8> {
    let half = QK / 2;
    let mut out = Vec::with_capacity(x.len() / QK * (2 + half));
    for block in x.chunks(QK) {
        let (mut max, mut amax) = (0f32, 0f32);
        for &v in block {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        for j in 0..half {
            let lo = block.get(j).copied().unwrap_or(0.0) * id;
            let hi = block.get(j + half).copied().unwrap_or(0.0) * id;
            let xi0 = ((lo + 8.5) as i32).clamp(0, 15) as u8;
            let xi1 = ((hi + 8.5) as i32).clamp(0, 15) as u8;
            out.push(xi0 | (xi1 << 4));
        }
    }
    out
}

/// Q5_0: per 32-block, f16 scale `d = max / -16`, a u32 of the 5th bits, then 32
/// packed low-4-bit quants (`(5bit - 16) * d`).
pub(crate) fn quantize_q5_0(x: &[f32]) -> Vec<u8> {
    let half = QK / 2;
    let mut out = Vec::with_capacity(x.len() / QK * (2 + 4 + half));
    for block in x.chunks(QK) {
        let (mut max, mut amax) = (0f32, 0f32);
        for &v in block {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        let d = max / -16.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        let mut qh: u32 = 0;
        let mut qs = [0u8; 16];
        for j in 0..half {
            let v0 =
                ((block.get(j).copied().unwrap_or(0.0) * id + 16.5) as i32).clamp(0, 31) as u32;
            let v1 = ((block.get(j + half).copied().unwrap_or(0.0) * id + 16.5) as i32).clamp(0, 31)
                as u32;
            qs[j] = ((v0 & 0x0F) | ((v1 & 0x0F) << 4)) as u8;
            qh |= ((v0 >> 4) & 1) << j;
            qh |= ((v1 >> 4) & 1) << (j + half);
        }
        out.extend_from_slice(&qh.to_le_bytes());
        out.extend_from_slice(&qs);
    }
    out
}

const QK_K: usize = 256; // K-quant super-block size

/// The block size (in elements) of a quant type — 256 for K-quants, else 32.
pub fn block_size(ggml_type: u32) -> usize {
    match ggml_type {
        GGML_Q4_K | GGML_Q5_K | GGML_Q6_K => QK_K,
        _ => QK,
    }
}

/// Pack eight 6-bit scales + eight 6-bit mins into the 12-byte field, the exact
/// inverse of `gguf::scale_min_k4`. (ggml's K-quant scale layout.)
fn pack_scales_mins(scales: &[u8; 8], mins: &[u8; 8]) -> [u8; 12] {
    let mut q = [0u8; 12];
    for j in 0..8 {
        let ls = scales[j] & 63;
        let lm = mins[j] & 63;
        if j < 4 {
            q[j] = ls;
            q[j + 4] = lm;
        } else {
            q[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            q[j - 4] |= (ls >> 4) << 6;
            q[j] |= (lm >> 4) << 6;
        }
    }
    q
}

/// Per-sub-block asymmetric scale + (non-negative) min for the K-quants that use
/// `value = scale*q - min` with `q` in `0..=nmax`. Searches a few candidate
/// scales (ggml's `make_qkx2_quants`): for each, assign integer levels, refit
/// (scale, min) by least squares, and score by *actual* reconstruction error —
/// so clipping is penalized. Beats naive min/max without the worst-case blow-up
/// a single-shot refit causes.
fn qkx_scale_min(x: &[f32], nmax: i32) -> (f32, f32) {
    let mut mn = x[0];
    let mut mx = x[0];
    for &v in x {
        mn = mn.min(v);
        mx = mx.max(v);
    }
    if mn > 0.0 {
        mn = 0.0;
    }
    if mx == mn {
        return (0.0, -mn);
    }
    let nf = x.len() as f32;
    let span = mx - mn;
    let nm = nmax as f32;
    let mut best = ((span / nm), -mn);
    let mut best_err = f32::MAX;
    // candidate inverse scales: (nmax + rmin + rdelta*i) / span, rmin=-1, rdelta=0.1
    for i in 0..=18 {
        let iscale = (nm - 1.0 + 0.1 * i as f32) / span;
        if iscale <= 0.0 {
            continue;
        }
        let (mut sl, mut sl2, mut sx, mut slx) = (0f32, 0f32, 0f32, 0f32);
        for &v in x {
            let l = ((iscale * (v - mn)).round() as i32).clamp(0, nmax) as f32;
            sl += l;
            sl2 += l * l;
            sx += v;
            slx += l * v;
        }
        let denom = nf * sl2 - sl * sl;
        let (scale, min) = if denom > 0.0 {
            let s = (nf * slx - sl * sx) / denom;
            let b = (sx - s * sl) / nf; // intercept = -min
            if s > 0.0 {
                (s, (-b).max(0.0))
            } else {
                (span / nm, -mn)
            }
        } else {
            (span / nm, -mn)
        };
        // actual reconstruction error with this (scale, min)
        let inv = 1.0 / scale;
        let mut err = 0f32;
        for &v in x {
            let l = (((v + min) * inv).round() as i32).clamp(0, nmax) as f32;
            let d = scale * l - min - v;
            err += d * d;
        }
        if err < best_err {
            best_err = err;
            best = (scale, min);
        }
    }
    best
}

/// Shared encoder for Q4_K (nmax=15) and Q5_K (nmax=31). Q5_K additionally emits
/// the per-value 5th bit into `qh` (32 bytes); pass `with_high`.
fn quantize_qx_k(x: &[f32], nmax: i32, with_high: bool) -> Vec<u8> {
    let bytes_per = if with_high {
        2 + 2 + 12 + 32 + 128
    } else {
        2 + 2 + 12 + 128
    };
    let mut out = Vec::with_capacity(x.len() / QK_K * bytes_per);
    for sb in x.chunks(QK_K) {
        let mut scales = [0f32; 8];
        let mut mins = [0f32; 8];
        for j in 0..8 {
            let (s, m) = qkx_scale_min(&sb[j * 32..j * 32 + 32], nmax);
            scales[j] = s;
            mins[j] = m;
        }
        let max_scale = scales.iter().copied().fold(0f32, f32::max);
        let max_min = mins.iter().copied().fold(0f32, f32::max);
        let d = max_scale / 63.0;
        let dmin = max_min / 63.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        let idm = if dmin > 0.0 { 1.0 / dmin } else { 0.0 };
        let mut ls = [0u8; 8];
        let mut lm = [0u8; 8];
        for j in 0..8 {
            ls[j] = ((scales[j] * id).round() as i32).clamp(0, 63) as u8;
            lm[j] = ((mins[j] * idm).round() as i32).clamp(0, 63) as u8;
        }
        out.extend_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        out.extend_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
        out.extend_from_slice(&pack_scales_mins(&ls, &lm));

        let mut qs = [0u8; 128];
        let mut qh = [0u8; 32];
        for g in 0..4 {
            for (sub, nib_shift) in [(2 * g, 0u8), (2 * g + 1, 4u8)] {
                let dq = d * ls[sub] as f32;
                let mq = dmin * lm[sub] as f32;
                let off = if nib_shift == 0 { 0 } else { 32 };
                for l in 0..32 {
                    let v = sb[g * 64 + off + l];
                    let q = if dq > 0.0 {
                        (((v + mq) / dq).round() as i32).clamp(0, nmax)
                    } else {
                        0
                    } as u32;
                    qs[g * 32 + l] |= ((q & 0x0F) as u8) << nib_shift;
                    if with_high && (q >> 4) & 1 != 0 {
                        // qh[l] bit (2g) for the low half, (2g+1) for the high half
                        qh[l] |= 1 << (2 * g + if nib_shift == 0 { 0 } else { 1 });
                    }
                }
            }
        }
        out.extend_from_slice(&qh[..if with_high { 32 } else { 0 }]);
        out.extend_from_slice(&qs);
    }
    out
}

pub(crate) fn quantize_q4_k(x: &[f32]) -> Vec<u8> {
    quantize_qx_k(x, 15, false)
}

pub(crate) fn quantize_q5_k(x: &[f32]) -> Vec<u8> {
    // Q5_K stores qh *before* qs in the block; quantize_qx_k already emits in
    // that order (qh then qs) when with_high is set.
    quantize_qx_k(x, 31, true)
}

/// Q6_K: 16 sub-blocks of 16, symmetric 6-bit (`value = d * sc[i] * q`,
/// `q` in `-32..=31`); `sc` are int8 per-sub-block scales, `d` an f16 super scale.
pub(crate) fn quantize_q6_k(x: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(x.len() / QK_K * 210);
    for sb in x.chunks(QK_K) {
        let mut subscale = [0f32; 16];
        for s in 0..16 {
            let blk = &sb[s * 16..s * 16 + 16];
            let mut max = 0f32;
            let mut amax = 0f32;
            for &v in blk {
                if v.abs() > amax {
                    amax = v.abs();
                    max = v;
                }
            }
            subscale[s] = if amax > 0.0 { max / -32.0 } else { 0.0 };
        }
        let mut max_scale = 0f32;
        for &s in &subscale {
            if s.abs() > max_scale.abs() {
                max_scale = s;
            }
        }
        let (d, idd) = if max_scale != 0.0 {
            (max_scale / -128.0, -128.0 / max_scale)
        } else {
            (0.0, 0.0)
        };
        let mut sc = [0i8; 16];
        for s in 0..16 {
            sc[s] = ((subscale[s] * idd).round() as i32).clamp(-128, 127) as i8;
        }
        let mut q6 = [0u8; 256];
        for s in 0..16 {
            let scale = d * sc[s] as f32;
            for i in 0..16 {
                let idx = s * 16 + i;
                let q = if scale != 0.0 {
                    ((sb[idx] / scale).round() as i32).clamp(-32, 31)
                } else {
                    0
                };
                q6[idx] = (q + 32) as u8;
            }
        }
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        for n128 in 0..2 {
            let (y0, qlb, qhb) = (n128 * 128, n128 * 64, n128 * 32);
            for l in 0..32 {
                let b1 = q6[y0 + l] & 63;
                let b2 = q6[y0 + l + 32] & 63;
                let b3 = q6[y0 + l + 64] & 63;
                let b4 = q6[y0 + l + 96] & 63;
                ql[qlb + l] = (b1 & 0xF) | ((b3 & 0xF) << 4);
                ql[qlb + l + 32] = (b2 & 0xF) | ((b4 & 0xF) << 4);
                qh[qhb + l] = ((b1 >> 4) & 3)
                    | (((b2 >> 4) & 3) << 2)
                    | (((b3 >> 4) & 3) << 4)
                    | (((b4 >> 4) & 3) << 6);
            }
        }
        out.extend_from_slice(&ql);
        out.extend_from_slice(&qh);
        for s in 0..16 {
            out.push(sc[s] as u8);
        }
        out.extend_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
    }
    out
}

#[inline]
fn pad(x: u64, a: u64) -> u64 {
    x.div_ceil(a) * a
}

fn write_string(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn type_tag(v: &Value) -> u32 {
    match v {
        Value::U8(_) => 0,
        Value::I8(_) => 1,
        Value::U16(_) => 2,
        Value::I16(_) => 3,
        Value::U32(_) => 4,
        Value::I32(_) => 5,
        Value::F32(_) => 6,
        Value::Bool(_) => 7,
        Value::Str(_) => 8,
        Value::Array(_) => 9,
        Value::U64(_) => 10,
        Value::I64(_) => 11,
        Value::F64(_) => 12,
    }
}

/// Write a metadata value with its leading type tag (matches `Reader::value`).
fn write_value(buf: &mut Vec<u8>, v: &Value) {
    buf.extend_from_slice(&type_tag(v).to_le_bytes());
    write_payload(buf, v);
}

/// Write a value's payload only (no tag) — used for array elements, which the
/// reader decodes against the array's element type.
fn write_payload(buf: &mut Vec<u8>, v: &Value) {
    match v {
        Value::U8(x) => buf.push(*x),
        Value::I8(x) => buf.push(*x as u8),
        Value::U16(x) => buf.extend_from_slice(&x.to_le_bytes()),
        Value::I16(x) => buf.extend_from_slice(&x.to_le_bytes()),
        Value::U32(x) => buf.extend_from_slice(&x.to_le_bytes()),
        Value::I32(x) => buf.extend_from_slice(&x.to_le_bytes()),
        Value::F32(x) => buf.extend_from_slice(&x.to_le_bytes()),
        Value::Bool(x) => buf.push(*x as u8),
        Value::Str(s) => write_string(buf, s),
        Value::U64(x) => buf.extend_from_slice(&x.to_le_bytes()),
        Value::I64(x) => buf.extend_from_slice(&x.to_le_bytes()),
        Value::F64(x) => buf.extend_from_slice(&x.to_le_bytes()),
        Value::Array(a) => {
            let elem_type = a.first().map(type_tag).unwrap_or(4); // u32 for empty
            buf.extend_from_slice(&elem_type.to_le_bytes());
            buf.extend_from_slice(&(a.len() as u64).to_le_bytes());
            for e in a {
                write_payload(buf, e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_roundtrip_within_bound() {
        // Quantize a few blocks, then dequantize with the engine's own decoder
        // and confirm every element is within Q8_0's per-block error bound.
        let n = 256usize;
        let x: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.013).sin() * 3.0 - 1.0)
            .collect();
        let bytes = quantize_q8_0(&x);
        assert_eq!(bytes.len(), n / QK * (2 + QK), "Q8_0 block size");

        let mut out = vec![0f32; n];
        Gguf::dequant_into(&bytes, GGML_Q8_0, n, &mut out).unwrap();

        for block in 0..n / QK {
            let lo = block * QK;
            let amax = x[lo..lo + QK].iter().fold(0f32, |m, &v| m.max(v.abs()));
            let tol = amax / 127.0 + 1e-4; // one quant step + f16-scale slack
            for i in lo..lo + QK {
                assert!(
                    (x[i] - out[i]).abs() <= tol,
                    "elem {i}: |{} - {}| > {tol}",
                    x[i],
                    out[i]
                );
            }
        }
    }

    fn check_roundtrip(ggml_type: u32, levels: f32) {
        let n = 256usize;
        let x: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.017).sin() * 4.0 - 0.7)
            .collect();
        let bytes = quantize(&x, ggml_type);
        let mut out = vec![0f32; n];
        Gguf::dequant_into(&bytes, ggml_type, n, &mut out).unwrap();
        for block in 0..n / QK {
            let lo = block * QK;
            let amax = x[lo..lo + QK].iter().fold(0f32, |m, &v| m.max(v.abs()));
            let tol = amax / levels + 1e-3; // one quant step + f16-scale slack
            for i in lo..lo + QK {
                assert!(
                    (x[i] - out[i]).abs() <= tol,
                    "type {ggml_type}, elem {i}: |{} - {}| > {tol}",
                    x[i],
                    out[i]
                );
            }
        }
    }

    #[test]
    fn q4_0_roundtrip_within_bound() {
        check_roundtrip(GGML_Q4_0, 8.0); // 4-bit: ~16 levels, step ~ amax/8
    }

    #[test]
    fn q5_0_roundtrip_within_bound() {
        check_roundtrip(GGML_Q5_0, 16.0); // 5-bit: ~32 levels, step ~ amax/16
    }

    // K-quants: per-sub-block scales, so error is bounded by the *sub-block*
    // range, not the super-block. Use a 256-multiple length and a looser bound.
    fn check_k_roundtrip(ggml_type: u32, max_rel_err: f32) {
        let n = 512usize;
        let x: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.011).sin() * 5.0 + 0.3)
            .collect();
        let bytes = quantize(&x, ggml_type);
        let mut out = vec![0f32; n];
        Gguf::dequant_into(&bytes, ggml_type, n, &mut out).unwrap();
        let amax = x.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let mut worst = 0f32;
        for i in 0..n {
            worst = worst.max((x[i] - out[i]).abs());
        }
        assert!(
            worst <= max_rel_err * amax,
            "type {ggml_type}: worst err {worst} > {max_rel_err} * {amax}"
        );
    }

    #[test]
    fn q4_k_roundtrip_within_bound() {
        check_k_roundtrip(GGML_Q4_K, 0.08);
    }

    #[test]
    fn q5_k_roundtrip_within_bound() {
        check_k_roundtrip(GGML_Q5_K, 0.04);
    }

    #[test]
    fn q6_k_roundtrip_within_bound() {
        check_k_roundtrip(GGML_Q6_K, 0.02);
    }
}
