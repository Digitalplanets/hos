//! Native HOS quantization — not bound to ggml's formats.
//!
//! ggml's quants place levels *uniformly* and lean on per-block scales. Model
//! weights are bell-shaped, so uniform levels waste codes in the tails. `hq4` is
//! a non-uniform 4-bit format: per 64-block absmax scale, then each weight maps
//! to the nearest of 16 levels placed at the **quantiles of a normal** (the NF4
//! idea). HOS's own format, decoded by the `.hos` loader with no ggml in the path.
//!
//! Block = 64 (bitsandbytes' NF4 default): 4 bits of index per weight plus one
//! f16 scale amortized over 64 weights = 4.25 bits/weight — genuinely below
//! Q4_K's 4.5, so an hq4 capsule is smaller than the Q4_K one at equal quality.
//!
//! Layout per 64-block: f16 absmax scale (2 bytes) + 32 bytes of packed 4-bit
//! indices (element 2k in the low nibble, 2k+1 in the high) = 34 bytes / 64.

use half::f16;

// q3 (below) hard-packs 32-element blocks; hq4 uses its own 64-element block.
const QK: usize = 32;
const HQ4_QK: usize = 64;

/// Block size (elements) of the `hq4` format — the alignment a tensor's row must
/// satisfy to be hq4-encodable.
pub fn hq4_block() -> usize {
    HQ4_QK
}

/// 16 non-uniform levels — quantiles of a unit normal in [-1, 1] (the NF4 set).
const NF4: [f32; 16] = [
    -1.0,
    -0.696_192_8,
    -0.525_073_05,
    -0.394_917_5,
    -0.284_441_38,
    -0.184_773_43,
    -0.091_050_03,
    0.0,
    0.079_580_3,
    0.160_930_2,
    0.246_112_3,
    0.337_915_24,
    0.440_709_83,
    0.562_617,
    0.722_956_84,
    1.0,
];

#[inline]
fn nearest(v: f32) -> u8 {
    // NF4 is sorted ascending; a linear scan over 16 is cheap and exact.
    let mut best = 0usize;
    let mut bd = f32::MAX;
    for (i, &c) in NF4.iter().enumerate() {
        let d = (v - c).abs();
        if d < bd {
            bd = d;
            best = i;
        }
    }
    best as u8
}

/// Encode a flat f32 tensor to `hq4`. `x.len()` must be a multiple of 64.
pub fn encode_hq4(x: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(x.len() / HQ4_QK * (2 + HQ4_QK / 2));
    for blk in x.chunks(HQ4_QK) {
        let amax = blk.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let scale = if amax > 0.0 { amax } else { 1.0 };
        let inv = 1.0 / scale;
        out.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
        for k in 0..HQ4_QK / 2 {
            let lo = nearest(blk.get(2 * k).copied().unwrap_or(0.0) * inv);
            let hi = nearest(blk.get(2 * k + 1).copied().unwrap_or(0.0) * inv);
            out.push(lo | (hi << 4));
        }
    }
    out
}

/// Decode `hq4` bytes back to `n` f32 values.
pub fn decode_hq4(bytes: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    decode_hq4_into(bytes, n, &mut out);
    out
}

/// Decode `n` `hq4` elements into an existing buffer (no allocation) — for the
/// fused quantized matvec row loop.
pub fn decode_hq4_into(bytes: &[u8], n: usize, out: &mut [f32]) {
    let block_bytes = 2 + HQ4_QK / 2;
    for b in 0..n / HQ4_QK {
        let base = b * block_bytes;
        let scale = f16::from_bits(u16::from_le_bytes([bytes[base], bytes[base + 1]])).to_f32();
        for k in 0..HQ4_QK / 2 {
            let byte = bytes[base + 2 + k];
            out[b * HQ4_QK + 2 * k] = NF4[(byte & 0x0F) as usize] * scale;
            out[b * HQ4_QK + 2 * k + 1] = NF4[(byte >> 4) as usize] * scale;
        }
    }
}

/// Stored byte size of an `hq4` tensor of `n` elements.
pub fn hq4_bytes(n: usize) -> usize {
    n / HQ4_QK * (2 + HQ4_QK / 2)
}

// ---- e8: E8-lattice vector quant ---------------------------------------------
//
// Model weights are coded 8 dimensions at a time on the E8 lattice — the densest
// packing in 8-D (normalized 2nd moment G≈0.0717 vs 1/12≈0.0833 for scalar), so
// at equal bit-rate it places ~0.6 dB less quantization noise than a per-axis
// grid. Each 8-block is snapped to its nearest E8 point by the closed-form
// `nn::nearest_e8` (round-to-D8 + coset compare, no codebook search), then the
// point's coordinates — always integer (D8) or half-integer (D8+½) — are stored
// as ×2 integers so both cosets fold to integers with no separate parity bit.
//
// Layout mirrors hq4: a 64-element super-block shares one f16 absmax scale, and
// the eight 8-D sub-blocks each pack their 8 coords as signed 4-bit nibbles
// (coord 2k low, 2k+1 high). 2 + 32 = 34 bytes / 64 = 4.25 bits/weight — below
// Q4_K's 4.5, so an e8 capsule is genuinely smaller than the Q4_K one.

const E8_QK: usize = 64; // scale super-block
const E8_SUB: usize = 8; // E8 lattice block (one lattice point)

// absmax maps to lattice coord ±E8_TARGET; ×2 = ±7 fills the signed-4-bit range
// [-8, 7]. The lattice covering radius can nudge the extreme past 7 — clamped.
const E8_TARGET: f32 = 3.5;

/// Block size (elements) of the `e8` format — the alignment a tensor's row must
/// satisfy to be e8-encodable (the scale super-block).
pub fn e8_block() -> usize {
    E8_QK
}

/// Stored byte size of an `e8` tensor of `n` elements (34 bytes / 64-block).
pub fn e8_bytes(n: usize) -> usize {
    n / E8_QK * (2 + E8_QK / 2)
}

/// Encode a flat f32 tensor to `e8`. `x.len()` must be a multiple of 64.
pub fn encode_e8(x: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(e8_bytes(x.len()));
    for sb in x.chunks(E8_QK) {
        let amax = sb.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let scale = if amax > 0.0 { amax / E8_TARGET } else { 1.0 };
        let inv = 1.0 / scale;
        out.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
        for blk in sb.chunks(E8_SUB) {
            let mut y = [0f32; E8_SUB];
            for (i, &v) in blk.iter().enumerate() {
                y[i] = v * inv;
            }
            let c = crate::nn::nearest_e8(&y);
            for k in 0..E8_SUB / 2 {
                let g0 = ((c[2 * k] * 2.0).round() as i32).clamp(-8, 7);
                let g1 = ((c[2 * k + 1] * 2.0).round() as i32).clamp(-8, 7);
                out.push(((g0 as u8) & 0x0F) | (((g1 as u8) & 0x0F) << 4));
            }
        }
    }
    out
}

/// Decode `e8` bytes back to `n` f32 values.
pub fn decode_e8(bytes: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    decode_e8_into(bytes, n, &mut out);
    out
}

/// Decode `n` `e8` elements into an existing buffer (no allocation) — for the
/// fused quantized matvec row loop.
pub fn decode_e8_into(bytes: &[u8], n: usize, out: &mut [f32]) {
    let block_bytes = 2 + E8_QK / 2;
    for b in 0..n / E8_QK {
        let base = b * block_bytes;
        let scale = f16::from_bits(u16::from_le_bytes([bytes[base], bytes[base + 1]])).to_f32();
        for k in 0..E8_QK / 2 {
            let byte = bytes[base + 2 + k];
            let lo = (byte & 0x0F) as i32;
            let hi = ((byte >> 4) & 0x0F) as i32;
            let g0 = if lo >= 8 { lo - 16 } else { lo };
            let g1 = if hi >= 8 { hi - 16 } else { hi };
            out[b * E8_QK + 2 * k] = g0 as f32 * 0.5 * scale;
            out[b * E8_QK + 2 * k + 1] = g1 as f32 * 0.5 * scale;
        }
    }
}

// ---- q3: uniform-symmetric 3-bit ---------------------------------------------
//
// The "fewer bits = faster decode" lever. Same uniform-symmetric scheme as q4_0
// (which AWQ-conditioned already beats ggml q4_k), but 3-bit: 8 levels, value =
// (code - 3.5) * scale, per-32-block f16 scale. Codes are split into a 2-bit
// plane (8 bytes, 4 codes/byte) + a 1-bit plane (4 bytes, 8 codes/byte) so the
// GPU extracts each with plane-local shifts (no cross-byte spanning). 14 bytes
// / 32 = 3.5 bits/weight — ~22% smaller than q4_0, so ~22% less decode bandwidth.

/// Stored byte size of a `q3` tensor of `n` elements (14 bytes / 32-block).
pub fn q3_bytes(n: usize) -> usize {
    n / QK * (2 + 12)
}

/// Encode a flat f32 tensor to `q3`. `x.len()` must be a multiple of 32.
pub fn encode_q3(x: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(q3_bytes(x.len()));
    for blk in x.chunks(QK) {
        let amax = blk.iter().fold(0f32, |m, &v| m.max(v.abs()));
        // 8 levels (code-4): {-4,-3,-2,-1,0,1,2,3}*scale -- code 4 is a TRUE ZERO,
        // so near-zero weights decode to exactly 0 (no noise floor). scale=amax/3
        // puts +amax at code 7 and -amax at code 1, code 0 spare for -outliers.
        let scale = if amax > 0.0 { amax / 3.0 } else { 1.0 };
        let inv = 1.0 / scale;
        out.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
        let mut codes = [0u8; QK];
        for (i, &v) in blk.iter().enumerate() {
            codes[i] = ((v * inv).round() as i32 + 4).clamp(0, 7) as u8;
        }
        for k in 0..8 {
            out.push(
                (codes[4 * k] & 3)
                    | ((codes[4 * k + 1] & 3) << 2)
                    | ((codes[4 * k + 2] & 3) << 4)
                    | ((codes[4 * k + 3] & 3) << 6),
            );
        }
        for k in 0..4 {
            let mut b = 0u8;
            for j in 0..8 {
                b |= ((codes[8 * k + j] >> 2) & 1) << j;
            }
            out.push(b);
        }
    }
    out
}

/// Decode `q3` bytes back to `n` f32 values.
pub fn decode_q3(bytes: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    decode_q3_into(bytes, n, &mut out);
    out
}

/// Decode `n` `q3` elements into an existing buffer (no allocation).
pub fn decode_q3_into(bytes: &[u8], n: usize, out: &mut [f32]) {
    let bb = 2 + 12;
    for b in 0..n / QK {
        let base = b * bb;
        let scale = f16::from_bits(u16::from_le_bytes([bytes[base], bytes[base + 1]])).to_f32();
        for i in 0..QK {
            let lo = (bytes[base + 2 + i / 4] >> ((i % 4) * 2)) & 3;
            let hi = (bytes[base + 2 + 8 + i / 8] >> (i % 8)) & 1;
            let code = lo | (hi << 2);
            out[b * QK + i] = (code as f32 - 4.0) * scale;
        }
    }
}

#[cfg(test)]
mod q3_tests {
    use super::*;
    #[test]
    fn q3_roundtrip_error() {
        // deterministic pseudo-normal block
        let mut x = vec![0f32; 256];
        let mut s: u32 = 12345;
        for v in x.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let u = (s >> 8) as f32 / (1 << 24) as f32 - 0.5;
            *v = u * 0.2;
        }
        // outlier per block: forces a large scale, so a missing zero-level would
        // blow up the near-zero majority (the bug this guards against).
        for b in 0..x.len() / 32 {
            x[b * 32] = 3.0;
        }
        let enc = encode_q3(&x);
        assert_eq!(enc.len(), q3_bytes(x.len()));
        let dec = decode_q3(&enc, x.len());
        let mut num = 0f32;
        let mut den = 0f32;
        for (a, b) in x.iter().zip(&dec) {
            num += (a - b) * (a - b);
            den += a * a;
        }
        let rel = (num / den).sqrt();
        eprintln!(
            "q3 rel-rmse = {rel:.4}  sample x[0..4]={:?} dec={:?}",
            &x[0..4],
            &dec[0..4]
        );
        assert!(rel < 0.25, "q3 reconstruction too lossy: {rel}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hq4_roundtrip() {
        let n = 256usize;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.03).sin() * 2.0).collect();
        let enc = encode_hq4(&x);
        assert_eq!(enc.len(), hq4_bytes(n));
        let dec = decode_hq4(&enc, n);
        // 4-bit non-uniform: worst error bounded by the largest gap between NF4
        // levels times the block scale.
        let amax = x.iter().fold(0f32, |m, &v| m.max(v.abs()));
        for (a, b) in x.iter().zip(&dec) {
            assert!((a - b).abs() <= 0.18 * amax, "|{a}-{b}| too big");
        }
    }
}
