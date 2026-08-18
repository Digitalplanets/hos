//! Metal GPU backend (v1, step 1): a matrix-vector kernel.
//!
//! Decode is dominated by W·x products (matvec). This uploads weights to a
//! resident GPU buffer once, then dispatches one GPU thread per output row.
//! Validated against the CPU path before it's trusted.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;

use half::f16;
use metal::{
    Buffer, BufferRef, CommandQueue, CompileOptions, ComputeCommandEncoderRef,
    ComputePipelineState, Device, MTLResourceOptions, MTLSize, ResourceRef,
};

use crate::gguf::{
    GGML_F16, GGML_F32, GGML_Q4_0, GGML_Q4_K, GGML_Q5_0, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0,
};
use crate::model::Model;

/// A GPU-resident f32 buffer handle. Aliased so the core `Tensor` type stays
/// platform-agnostic; on non-macOS this is a unit type that is never constructed
/// (the GPU path is gated off). Holding one keeps an activation on the GPU
/// between ops instead of round-tripping through CPU memory.
pub type GpuBuf = metal::Buffer;

/// Read the first `n` f32s of a resident buffer back to CPU. Apple Silicon is
/// unified memory, so this is a direct read of the shared buffer (the GPU write
/// that produced it already completed), not a device copy.
pub fn download_buf(buf: &GpuBuf, n: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, n) }.to_vec()
}

const KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void matvec(
    device const half*  w      [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device float*       y      [[buffer(2)]],
    constant uint&      in_dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint base = gid * in_dim;
    float acc = 0.0f;
    for (uint i = 0; i < in_dim; ++i) {
        acc += float(w[base + i]) * x[i];
    }
    y[gid] = acc;
}
"#;

/// All the kernels for the fully-resident forward pass.
const FUSED_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// read an f16 from arbitrary (possibly unaligned) byte offset
inline float rdh(device const uchar* p, uint off) {
    ushort b = (ushort)p[off] | ((ushort)p[off + 1] << 8);
    return float(as_type<half>(b));
}
// Q4_K/Q5_K packed 6-bit scale+min unpack (ggml get_scale_min_k4)
inline void getsm(device const uchar* q, uint j, thread uchar& d, thread uchar& m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

kernel void matvec_f32(
    device const float* w [[buffer(0)]], device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& in_dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    uint base = gid * in_dim; float acc = 0.0f;
    for (uint i = 0; i < in_dim; ++i) acc += w[base + i] * x[i];
    y[gid] = acc;
}

kernel void matvec_f16(
    device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& in_dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    uint base = gid * in_dim * 2; float acc = 0.0f;
    for (uint i = 0; i < in_dim; ++i) acc += rdh(w, base + i * 2) * x[i];
    y[gid] = acc;
}

kernel void matvec_q8_0(
    device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& in_dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    uint nb = in_dim / 32; uint base = gid * nb * 34; float acc = 0.0f;
    for (uint b = 0; b < nb; ++b) {
        uint bb = base + b * 34; float d = rdh(w, bb); uint xo = b * 32;
        for (uint j = 0; j < 32; ++j) acc += d * float((char)w[bb + 2 + j]) * x[xo + j];
    }
    y[gid] = acc;
}

kernel void matvec_q5_0(
    device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& in_dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    uint nb = in_dim / 32; uint base = gid * nb * 22; float acc = 0.0f;
    for (uint b = 0; b < nb; ++b) {
        uint bb = base + b * 22; float d = rdh(w, bb);
        uint qh = (uint)w[bb+2] | ((uint)w[bb+3]<<8) | ((uint)w[bb+4]<<16) | ((uint)w[bb+5]<<24);
        uint xo = b * 32;
        for (uint j = 0; j < 16; ++j) {
            int xh0 = (int)(((qh >> j) << 4) & 0x10);
            int xh1 = (int)((qh >> (j + 12)) & 0x10);
            uchar q = w[bb + 6 + j];
            acc += d * (float)(((int)(q & 0xF) | xh0) - 16) * x[xo + j];
            acc += d * (float)(((int)(q >> 4) | xh1) - 16) * x[xo + j + 16];
        }
    }
    y[gid] = acc;
}

kernel void matvec_q4k(
    device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& in_dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    uint nsb = in_dim / 256; uint base = gid * nsb * 144; float acc = 0.0f;
    for (uint sb = 0; sb < nsb; ++sb) {
        uint p = base + sb * 144; float d = rdh(w, p); float dmin = rdh(w, p + 2);
        device const uchar* scales = w + p + 4; device const uchar* qs = w + p + 16;
        uint xo = sb * 256; uint is = 0;
        for (uint j = 0; j < 256; j += 64) {
            uchar sc1, m1, sc2, m2;
            getsm(scales, is, sc1, m1); getsm(scales, is + 1, sc2, m2);
            float d1 = d * sc1, mn1 = dmin * m1, d2 = d * sc2, mn2 = dmin * m2;
            uint qo = j / 2;
            for (uint l = 0; l < 32; ++l) acc += (d1 * float(qs[qo + l] & 0xF) - mn1) * x[xo + j + l];
            for (uint l = 0; l < 32; ++l) acc += (d2 * float(qs[qo + l] >> 4) - mn2) * x[xo + j + 32 + l];
            is += 2;
        }
    }
    y[gid] = acc;
}

kernel void matvec_q5k(
    device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& in_dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    uint nsb = in_dim / 256; uint base = gid * nsb * 176; float acc = 0.0f;
    for (uint sb = 0; sb < nsb; ++sb) {
        uint p = base + sb * 176; float d = rdh(w, p); float dmin = rdh(w, p + 2);
        device const uchar* scales = w + p + 4; device const uchar* qh = w + p + 16;
        device const uchar* qs = w + p + 48;
        uint xo = sb * 256; uint is = 0; uchar u1 = 1, u2 = 2;
        for (uint j = 0; j < 256; j += 64) {
            uchar sc1, m1, sc2, m2;
            getsm(scales, is, sc1, m1); getsm(scales, is + 1, sc2, m2);
            float d1 = d * sc1, mn1 = dmin * m1, d2 = d * sc2, mn2 = dmin * m2;
            uint qo = j / 2;
            for (uint l = 0; l < 32; ++l) {
                float hi = (qh[l] & u1) ? 16.0f : 0.0f;
                acc += (d1 * (float(qs[qo + l] & 0xF) + hi) - mn1) * x[xo + j + l];
            }
            for (uint l = 0; l < 32; ++l) {
                float hi = (qh[l] & u2) ? 16.0f : 0.0f;
                acc += (d2 * (float(qs[qo + l] >> 4) + hi) - mn2) * x[xo + j + 32 + l];
            }
            is += 2; u1 <<= 2; u2 <<= 2;
        }
    }
    y[gid] = acc;
}

kernel void matvec_q6k(
    device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& in_dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    uint nsb = in_dim / 256; uint base = gid * nsb * 210; float acc = 0.0f;
    for (uint sb = 0; sb < nsb; ++sb) {
        uint p = base + sb * 210;
        device const uchar* ql = w + p; device const uchar* qh = w + p + 128;
        device const char* sc = (device const char*)(w + p + 192);
        float d = rdh(w, p + 208); uint xo = sb * 256;
        for (uint n = 0; n < 2; ++n) {
            device const uchar* ql2 = ql + n * 64; device const uchar* qh2 = qh + n * 32;
            device const char* sc2 = sc + n * 8; uint yo = xo + n * 128;
            for (uint l = 0; l < 32; ++l) {
                uint is = l / 16;
                int q1 = (int)((ql2[l] & 0xF) | (((qh2[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql2[l + 32] & 0xF) | (((qh2[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql2[l] >> 4) | (((qh2[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql2[l + 32] >> 4) | (((qh2[l] >> 6) & 3) << 4)) - 32;
                acc += d * float(sc2[is + 0]) * float(q1) * x[yo + l];
                acc += d * float(sc2[is + 2]) * float(q2) * x[yo + l + 32];
                acc += d * float(sc2[is + 4]) * float(q3) * x[yo + l + 64];
                acc += d * float(sc2[is + 6]) * float(q4) * x[yo + l + 96];
            }
        }
    }
    y[gid] = acc;
}

// ---- coalesced matvec: one 32-lane simdgroup per output row ----
// Lanes read consecutive bytes (coalesced) and reduce with simd_sum.

// ---- N rows per simdgroup ----
// Each lane loads its x value once and reuses it across NDST independent weight
// rows: NDST× the in-flight weight loads (hides memory latency) and 1/NDST the
// activation traffic. Per-row accumulation order is unchanged, so results are
// bit-identical to the one-row-per-simdgroup version. `n_rows` guards the tail.
#define NDST 2

kernel void matvec_q8_0_co(
    device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& in_dim [[buffer(3)]],
    constant uint& n_rows [[buffer(4)]],
    uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = (gid / 32) * NDST; uint nb = in_dim / 32; uint rb = nb * 34;
    float acc[NDST]; for (uint r = 0; r < NDST; ++r) acc[r] = 0.0f;
    for (uint b = 0; b < nb; ++b) {
        float xv = x[b * 32 + lane];
        for (uint r = 0; r < NDST; ++r) {
            uint rr = min(row0 + r, n_rows - 1);
            uint bb = rr * rb + b * 34;
            acc[r] += rdh(w, bb) * (float)((char)w[bb + 2 + lane]) * xv;
        }
    }
    for (uint r = 0; r < NDST; ++r) { float t = simd_sum(acc[r]); if (lane == 0 && row0 + r < n_rows) y[row0 + r] = t; }
}

// q4_0: 18 bytes/block (f16 scale + 32 symmetric 4-bit). Dequant is just
// (nibble - 8) * scale — no scale unpacking — so it's bandwidth-bound, not
// ALU-bound like q4_k. Weight `lane`: low nibble of qs[lane] for lane<16, high
// nibble of qs[lane-16] otherwise (ggml q4_0 layout).
kernel void matvec_q4_0_co(
    device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& in_dim [[buffer(3)]],
    constant uint& n_rows [[buffer(4)]],
    uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    // 2 blocks per pass: lanes 0-15 take block bp, lanes 16-31 take block bp+1, so
    // every lane loads a UNIQUE byte (no redundancy) and handles its two nibbles
    // (weights byteidx and byteidx+16). Full 32-lane utilization -> ~2x bandwidth.
    uint row0 = (gid / 32) * NDST; uint nb = in_dim / 32; uint rb = nb * 18;
    uint blkoff = lane >> 4;     // 0 for lanes 0-15, 1 for lanes 16-31
    uint bidx = lane & 15u;      // byte within the block
    float acc[NDST]; for (uint r = 0; r < NDST; ++r) acc[r] = 0.0f;
    for (uint bp = 0; bp < nb; bp += 2) {
        uint blk = bp + blkoff;
        if (blk >= nb) continue;
        float xlo = x[blk * 32 + bidx];
        float xhi = x[blk * 32 + bidx + 16];
        for (uint r = 0; r < NDST; ++r) {
            uint rr = min(row0 + r, n_rows - 1);
            uint bb = rr * rb + blk * 18;
            float d = rdh(w, bb);
            uchar by = w[bb + 2 + bidx];
            acc[r] += d * ((float)((int)(by & 0xF) - 8) * xlo + (float)((int)(by >> 4) - 8) * xhi);
        }
    }
    for (uint r = 0; r < NDST; ++r) { float t = simd_sum(acc[r]); if (lane == 0 && row0 + r < n_rows) y[row0 + r] = t; }
}

// hq4 (HOS native NF4): 34 bytes / 64-block (f16 absmax scale + 32 packed nibble
// bytes), the CPU decode_hq4_into layout. Two things differ from q4_0:
//   1. dequant is non-uniform: value = NF4[nibble] * scale, where NF4 is the
//      16-entry normal-quantile LUT (hos_quant.rs:18-35), not q4_0's (nibble-8).
//   2. nibble packing: byte k holds weight 2k in the LOW nibble and 2k+1 in the
//      HIGH nibble (hos_quant decode_hq4_into), whereas q4_0 byte j holds weights
//      j and j+16. So the x-index for byte `bidx` is 2*bidx / 2*bidx+1, not
//      bidx / bidx+16.
// One 64-block per pass, one byte per lane (32 lanes cover the block's 32 nibble
// bytes = 64 weights). Per-row accumulation order matches the CPU decode_hq4_into
// + dot reference (simd_sum reorder only), within the parity tolerance.
constant float HQ4_NF4[16] = {
    -1.0f, -0.6961928f, -0.52507305f, -0.3949175f,
    -0.28444138f, -0.18477343f, -0.09105003f, 0.0f,
    0.0795803f, 0.1609302f, 0.2461123f, 0.33791524f,
    0.44070983f, 0.562617f, 0.72295684f, 1.0f
};

kernel void matvec_hq4_co(
    device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& in_dim [[buffer(3)]],
    constant uint& n_rows [[buffer(4)]],
    uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = (gid / 32) * NDST; uint nb = in_dim / 64; uint rb = nb * 34;
    uint bidx = lane;            // byte within the 64-block (holds weights 2*bidx, 2*bidx+1)
    float acc[NDST]; for (uint r = 0; r < NDST; ++r) acc[r] = 0.0f;
    for (uint blk = 0; blk < nb; ++blk) {
        float xlo = x[blk * 64 + 2 * bidx];
        float xhi = x[blk * 64 + 2 * bidx + 1];
        for (uint r = 0; r < NDST; ++r) {
            uint rr = min(row0 + r, n_rows - 1);
            uint bb = rr * rb + blk * 34;
            float d = rdh(w, bb);
            uchar by = w[bb + 2 + bidx];
            acc[r] += d * (HQ4_NF4[by & 0xF] * xlo + HQ4_NF4[by >> 4] * xhi);
        }
    }
    for (uint r = 0; r < NDST; ++r) { float t = simd_sum(acc[r]); if (lane == 0 && row0 + r < n_rows) y[row0 + r] = t; }
}

kernel void matvec_q4k_co(
    device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& in_dim [[buffer(3)]],
    constant uint& n_rows [[buffer(4)]],
    uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = (gid / 32) * NDST; uint nsb = in_dim / 256; uint sbb = nsb * 144;
    float acc[NDST]; for (uint r = 0; r < NDST; ++r) acc[r] = 0.0f;
    for (uint sbk = 0; sbk < nsb; ++sbk) {
        float d[NDST], dmin[NDST]; device const uchar* scales[NDST]; device const uchar* qs[NDST];
        for (uint r = 0; r < NDST; ++r) {
            uint p = min(row0 + r, n_rows - 1) * sbb + sbk * 144;
            d[r] = rdh(w, p); dmin[r] = rdh(w, p + 2); scales[r] = w + p + 4; qs[r] = w + p + 16;
        }
        uint xo = sbk * 256;
        for (uint sb = 0; sb < 8; ++sb) {
            uint qi = (sb / 2) * 32 + lane; float xv = x[xo + sb * 32 + lane];
            for (uint r = 0; r < NDST; ++r) {
                uchar sc, m; getsm(scales[r], sb, sc, m);
                uint nib = (sb & 1) ? (qs[r][qi] >> 4) : (qs[r][qi] & 0xF);
                acc[r] += (d[r] * sc * (float)nib - dmin[r] * m) * xv;
            }
        }
    }
    for (uint r = 0; r < NDST; ++r) { float t = simd_sum(acc[r]); if (lane == 0 && row0 + r < n_rows) y[row0 + r] = t; }
}

// 2-token coalesced q4_k matmul (MTP verify): same coalesced NDST-rows-per-simd
// structure as matvec_q4k_co (so it's ~as fast per weight read), but each decoded
// weight is multiplied into BOTH tokens — weight read ONCE for 2 tokens.
// X = [2, in_dim], Y = [2, n_rows].
kernel void matmul_q4k_co2(
    device const uchar* w [[buffer(0)]], device const float* X [[buffer(1)]],
    device float* Y [[buffer(2)]], constant uint& in_dim [[buffer(3)]],
    constant uint& n_rows [[buffer(4)]],
    uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = (gid / 32) * NDST; uint nsb = in_dim / 256; uint sbb = nsb * 144;
    float a0[NDST], a1[NDST];
    for (uint r = 0; r < NDST; ++r) { a0[r] = 0.0f; a1[r] = 0.0f; }
    for (uint sbk = 0; sbk < nsb; ++sbk) {
        float d[NDST], dmin[NDST]; device const uchar* scales[NDST]; device const uchar* qs[NDST];
        for (uint r = 0; r < NDST; ++r) {
            uint p = min(row0 + r, n_rows - 1) * sbb + sbk * 144;
            d[r] = rdh(w, p); dmin[r] = rdh(w, p + 2); scales[r] = w + p + 4; qs[r] = w + p + 16;
        }
        uint xo = sbk * 256;
        for (uint sb = 0; sb < 8; ++sb) {
            uint qi = (sb / 2) * 32 + lane; uint xc = xo + sb * 32 + lane;
            float xv0 = X[xc]; float xv1 = X[in_dim + xc];
            for (uint r = 0; r < NDST; ++r) {
                uchar sc, m; getsm(scales[r], sb, sc, m);
                uint nib = (sb & 1) ? (qs[r][qi] >> 4) : (qs[r][qi] & 0xF);
                float wv = d[r] * sc * (float)nib - dmin[r] * m;
                a0[r] += wv * xv0; a1[r] += wv * xv1;
            }
        }
    }
    for (uint r = 0; r < NDST; ++r) {
        float t0 = simd_sum(a0[r]); float t1 = simd_sum(a1[r]);
        if (lane == 0 && row0 + r < n_rows) { Y[row0 + r] = t0; Y[n_rows + row0 + r] = t1; }
    }
}

// ---- BATCHED q4_k matmul (prefill): Y[ntok, n_rows] = X[ntok, in_dim] @ W^T ----
// One 32-lane simdgroup per output row. Each dequantized weight element is
// computed ONCE and multiplied into all `ntok` token-vectors — so the row's
// weight is read once per token-tile instead of once per token (the prefill win).
// ntok <= NTOK_B. Per-token accumulation order matches matvec, so a 1-token tile
// is bit-identical to matvec_q4k_co.
#define NTOK_B 8
kernel void matmul_q4k_batch(
    device const uchar* w [[buffer(0)]], device const float* X [[buffer(1)]],
    device float* Y [[buffer(2)]], constant uint& in_dim [[buffer(3)]],
    constant uint& n_rows [[buffer(4)]], constant uint& ntok [[buffer(5)]],
    uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint row = gid / 32;
    if (row >= n_rows) return;
    uint nsb = in_dim / 256; uint sbb = nsb * 144;
    float acc[NTOK_B]; for (uint t = 0; t < NTOK_B; ++t) acc[t] = 0.0f;
    for (uint sbk = 0; sbk < nsb; ++sbk) {
        uint p = row * sbb + sbk * 144;
        float d = rdh(w, p); float dmin = rdh(w, p + 2);
        device const uchar* scales = w + p + 4; device const uchar* qs = w + p + 16;
        uint xo = sbk * 256;
        for (uint sb = 0; sb < 8; ++sb) {
            uint qi = (sb / 2) * 32 + lane;
            uchar sc, m; getsm(scales, sb, sc, m);
            uint nib = (sb & 1) ? (qs[qi] >> 4) : (qs[qi] & 0xF);
            float wv = d * sc * (float)nib - dmin * m;
            uint xcol = xo + sb * 32 + lane;
            for (uint t = 0; t < ntok; ++t) acc[t] += wv * X[t * in_dim + xcol];
        }
    }
    for (uint t = 0; t < ntok; ++t) { float s = simd_sum(acc[t]); if (lane == 0) Y[t * n_rows + row] = s; }
}

// ---- prefill q4_k matmul: dequant each weight ROW once into shared memory, then
// reuse it across ALL `ntok` tokens. Dequant (the compute bottleneck) is amortized
// over the whole prompt instead of per token. One threadgroup (8 simdgroups) per
// output row; the 8 simdgroups compute 8 tokens' dot products in parallel, looping
// token-groups. in_dim must be <= PF_MAXIN. ---------------------------------------
#define PF_MAXIN 4096
kernel void matmul_q4k_prefill(
    device const uchar* w [[buffer(0)]], device const float* X [[buffer(1)]],
    device float* Y [[buffer(2)]], constant uint& in_dim [[buffer(3)]],
    constant uint& n_rows [[buffer(4)]], constant uint& ntok [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint tcount [[threads_per_threadgroup]]) {
    threadgroup float wrow[PF_MAXIN];
    uint sbb = (in_dim / 256) * 144;
    uint rb = row * sbb;
    for (uint i = tid; i < in_dim; i += tcount) {
        uint sbk = i / 256, within = i % 256, sb = within / 32, ln = within % 32;
        uint qi = (sb / 2) * 32 + ln;
        uint p = rb + sbk * 144;
        float d = rdh(w, p); float dmin = rdh(w, p + 2);
        device const uchar* scales = w + p + 4; device const uchar* qs = w + p + 16;
        uchar sc, m; getsm(scales, sb, sc, m);
        uint nib = (sb & 1) ? (qs[qi] >> 4) : (qs[qi] & 0xF);
        wrow[i] = d * sc * (float)nib - dmin * m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint lane = tid & 31u, sgid = tid >> 5, nsg = tcount >> 5;
    // Each simdgroup processes a tile of TT tokens: wrow[i] is read once and FMA'd
    // into all TT tokens (4x fewer shared reads, 4x FMA density, 1/TT the reductions).
    // Token buffers are padded by >= TT rows, so the over-reads are in-bounds garbage
    // and only valid tokens are written. TT must divide the simdgroup tiling cleanly.
    const uint TT = 4;
    for (uint tb = sgid * TT; tb < ntok; tb += nsg * TT) {
        float partial[TT];
        #pragma unroll
        for (uint tt = 0; tt < TT; ++tt) partial[tt] = 0.0f;
        for (uint i = lane; i < in_dim; i += 32) {
            float wv = wrow[i];
            #pragma unroll
            for (uint tt = 0; tt < TT; ++tt) partial[tt] += wv * X[(tb + tt) * in_dim + i];
        }
        #pragma unroll
        for (uint tt = 0; tt < TT; ++tt) {
            float s = simd_sum(partial[tt]);
            if (lane == 0 && tb + tt < ntok) Y[(tb + tt) * n_rows + row] = s;
        }
    }
}

// Dequant a full q4_k weight [n_rows, in_dim] -> f32 scratch, one thread/element.
// Done ONCE per prefill matmul, then a clean tiled GEMM reads the f32 scratch --
// the dequant cost amortizes over ALL prompt tokens (vs re-dequanting per token),
// and the GEMM gets small-tile shared memory => high occupancy => compute-bound.
kernel void dequant_q4k_to_f32(
    device const uchar* w [[buffer(0)]], device float* out [[buffer(1)]],
    constant uint& in_dim [[buffer(2)]], constant uint& n_rows [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= n_rows * in_dim) return;
    uint row = gid / in_dim, col = gid % in_dim;
    uint sbb = (in_dim / 256) * 144;
    uint sbk = col / 256, within = col % 256, sb = within / 32, ln = within % 32;
    uint qi = (sb / 2) * 32 + ln;
    uint p = row * sbb + sbk * 144;
    float d = rdh(w, p); float dmin = rdh(w, p + 2);
    device const uchar* scales = w + p + 4; device const uchar* qs = w + p + 16;
    uchar sc, m; getsm(scales, sb, sc, m);
    uint nib = (sb & 1) ? (qs[qi] >> 4) : (qs[qi] & 0xF);
    out[gid] = d * sc * (float)nib - dmin * m;
}

// Per-element q6_k -> f32 (mirror of the matvec_q6k block decode). One thread per
// output element; feeds the shared batched-prefill GEMM for q6_k weights.
kernel void dequant_q6k_to_f32(
    device const uchar* w [[buffer(0)]], device float* out [[buffer(1)]],
    constant uint& in_dim [[buffer(2)]], constant uint& n_rows [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= n_rows * in_dim) return;
    uint row = gid / in_dim, col = gid % in_dim;
    uint nsb = in_dim / 256; uint sbb = nsb * 210;
    uint sbk = col / 256, within = col % 256;
    uint p = row * sbb + sbk * 210;
    device const uchar* ql = w + p; device const uchar* qh = w + p + 128;
    device const char* sc = (device const char*)(w + p + 192);
    float d = rdh(w, p + 208);
    uint n = within / 128, r = within % 128;
    uint group = r / 32, l = r % 32, is = l / 16;
    device const uchar* ql2 = ql + n * 64; device const uchar* qh2 = qh + n * 32;
    device const char* sc2 = sc + n * 8;
    int q;
    if (group == 0)      q = (int)((ql2[l]      & 0xF) | (((qh2[l] >> 0) & 3) << 4)) - 32;
    else if (group == 1) q = (int)((ql2[l + 32] & 0xF) | (((qh2[l] >> 2) & 3) << 4)) - 32;
    else if (group == 2) q = (int)((ql2[l]      >> 4)  | (((qh2[l] >> 4) & 3) << 4)) - 32;
    else                 q = (int)((ql2[l + 32] >> 4)  | (((qh2[l] >> 6) & 3) << 4)) - 32;
    out[gid] = d * float(sc2[is + group * 2]) * float(q);
}

// Small-batch q6_k matmul: one thread per output ROW reads that row's quantized
// weight ONCE and multiplies each decoded value into all `ntok` token vectors
// (no f32 scratch, no double-read). The q6_k twin of matmul_q4k_batch, for the
// MTP 2-token verify. ntok <= 8.
kernel void matmul_q6k_batch(
    device const uchar* w [[buffer(0)]], device const float* X [[buffer(1)]],
    device float* Y [[buffer(2)]], constant uint& in_dim [[buffer(3)]],
    constant uint& n_rows [[buffer(4)]], constant uint& ntok [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    uint row = gid;
    if (row >= n_rows) return;
    uint nsb = in_dim / 256; uint base = row * nsb * 210;
    float acc[8]; for (uint t = 0; t < 8; ++t) acc[t] = 0.0f;
    for (uint sb = 0; sb < nsb; ++sb) {
        uint p = base + sb * 210;
        device const uchar* ql = w + p; device const uchar* qh = w + p + 128;
        device const char* sc = (device const char*)(w + p + 192);
        float d = rdh(w, p + 208); uint xo = sb * 256;
        for (uint n = 0; n < 2; ++n) {
            device const uchar* ql2 = ql + n * 64; device const uchar* qh2 = qh + n * 32;
            device const char* sc2 = sc + n * 8; uint yo = xo + n * 128;
            for (uint l = 0; l < 32; ++l) {
                uint is = l / 16;
                int q1 = (int)((ql2[l]      & 0xF) | (((qh2[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql2[l + 32] & 0xF) | (((qh2[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql2[l]      >> 4)  | (((qh2[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql2[l + 32] >> 4)  | (((qh2[l] >> 6) & 3) << 4)) - 32;
                float w1 = d * float(sc2[is + 0]) * float(q1);
                float w2 = d * float(sc2[is + 2]) * float(q2);
                float w3 = d * float(sc2[is + 4]) * float(q3);
                float w4 = d * float(sc2[is + 6]) * float(q4);
                for (uint t = 0; t < ntok; ++t) {
                    device const float* xt = X + t * in_dim;
                    acc[t] += w1 * xt[yo + l] + w2 * xt[yo + l + 32]
                            + w3 * xt[yo + l + 64] + w4 * xt[yo + l + 96];
                }
            }
        }
    }
    for (uint t = 0; t < ntok; ++t) Y[t * n_rows + row] = acc[t];
}

kernel void matvec_q5k_co(
    device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& in_dim [[buffer(3)]],
    constant uint& n_rows [[buffer(4)]],
    uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = (gid / 32) * NDST; uint nsb = in_dim / 256; uint sbb = nsb * 176;
    float acc[NDST]; for (uint r = 0; r < NDST; ++r) acc[r] = 0.0f;
    for (uint sbk = 0; sbk < nsb; ++sbk) {
        float d[NDST], dmin[NDST]; device const uchar* scales[NDST]; device const uchar* qh[NDST]; device const uchar* qs[NDST];
        for (uint r = 0; r < NDST; ++r) {
            uint p = min(row0 + r, n_rows - 1) * sbb + sbk * 176;
            d[r] = rdh(w, p); dmin[r] = rdh(w, p + 2); scales[r] = w + p + 4; qh[r] = w + p + 16; qs[r] = w + p + 48;
        }
        uint xo = sbk * 256;
        for (uint sb = 0; sb < 8; ++sb) {
            uint qi = (sb / 2) * 32 + lane; float xv = x[xo + sb * 32 + lane];
            for (uint r = 0; r < NDST; ++r) {
                uchar sc, m; getsm(scales[r], sb, sc, m);
                uint nib = (sb & 1) ? (qs[r][qi] >> 4) : (qs[r][qi] & 0xF);
                float hi = ((qh[r][lane] >> sb) & 1) ? 16.0f : 0.0f;
                acc[r] += (d[r] * sc * ((float)nib + hi) - dmin[r] * m) * xv;
            }
        }
    }
    for (uint r = 0; r < NDST; ++r) { float t = simd_sum(acc[r]); if (lane == 0 && row0 + r < n_rows) y[row0 + r] = t; }
}

kernel void matvec_q6k_co(
    device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint& in_dim [[buffer(3)]],
    constant uint& n_rows [[buffer(4)]],
    uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = (gid / 32) * NDST; uint nsb = in_dim / 256; uint sbb = nsb * 210;
    float acc[NDST]; for (uint r = 0; r < NDST; ++r) acc[r] = 0.0f;
    for (uint sbk = 0; sbk < nsb; ++sbk) {
        device const uchar* ql[NDST]; device const uchar* qh[NDST]; device const char* sc[NDST]; float d[NDST];
        for (uint r = 0; r < NDST; ++r) {
            uint p = min(row0 + r, n_rows - 1) * sbb + sbk * 210;
            ql[r] = w + p; qh[r] = w + p + 128; sc[r] = (device const char*)(w + p + 192); d[r] = rdh(w, p + 208);
        }
        uint xo = sbk * 256;
        for (uint n = 0; n < 2; ++n) {
            uint yo = xo + n * 128; uint is = lane / 16;
            float x0 = x[yo + lane], x1 = x[yo + lane + 32], x2 = x[yo + lane + 64], x3 = x[yo + lane + 96];
            for (uint r = 0; r < NDST; ++r) {
                device const uchar* ql2 = ql[r] + n * 64; device const uchar* qh2 = qh[r] + n * 32; device const char* sc2 = sc[r] + n * 8;
                int q1 = (int)((ql2[lane] & 0xF) | (((qh2[lane] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql2[lane + 32] & 0xF) | (((qh2[lane] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql2[lane] >> 4) | (((qh2[lane] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql2[lane + 32] >> 4) | (((qh2[lane] >> 6) & 3) << 4)) - 32;
                acc[r] += d[r] * (float)sc2[is + 0] * (float)q1 * x0;
                acc[r] += d[r] * (float)sc2[is + 2] * (float)q2 * x1;
                acc[r] += d[r] * (float)sc2[is + 4] * (float)q3 * x2;
                acc[r] += d[r] * (float)sc2[is + 6] * (float)q4 * x3;
            }
        }
    }
    for (uint r = 0; r < NDST; ++r) { float t = simd_sum(acc[r]); if (lane == 0 && row0 + r < n_rows) y[row0 + r] = t; }
}

// 2-token coalesced q6_k matmul (MTP verify): matvec_q6k_co extended to both
// tokens — weight read once, X=[2,in_dim], Y=[2,n_rows].
kernel void matmul_q6k_co2(
    device const uchar* w [[buffer(0)]], device const float* X [[buffer(1)]],
    device float* Y [[buffer(2)]], constant uint& in_dim [[buffer(3)]],
    constant uint& n_rows [[buffer(4)]],
    uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = (gid / 32) * NDST; uint nsb = in_dim / 256; uint sbb = nsb * 210;
    float a0[NDST], a1[NDST];
    for (uint r = 0; r < NDST; ++r) { a0[r] = 0.0f; a1[r] = 0.0f; }
    for (uint sbk = 0; sbk < nsb; ++sbk) {
        device const uchar* ql[NDST]; device const uchar* qh[NDST]; device const char* sc[NDST]; float d[NDST];
        for (uint r = 0; r < NDST; ++r) {
            uint p = min(row0 + r, n_rows - 1) * sbb + sbk * 210;
            ql[r] = w + p; qh[r] = w + p + 128; sc[r] = (device const char*)(w + p + 192); d[r] = rdh(w, p + 208);
        }
        uint xo = sbk * 256;
        for (uint n = 0; n < 2; ++n) {
            uint yo = xo + n * 128; uint is = lane / 16;
            float x0 = X[yo + lane], x1 = X[yo + lane + 32], x2 = X[yo + lane + 64], x3 = X[yo + lane + 96];
            float z0 = X[in_dim + yo + lane], z1 = X[in_dim + yo + lane + 32], z2 = X[in_dim + yo + lane + 64], z3 = X[in_dim + yo + lane + 96];
            for (uint r = 0; r < NDST; ++r) {
                device const uchar* ql2 = ql[r] + n * 64; device const uchar* qh2 = qh[r] + n * 32; device const char* sc2 = sc[r] + n * 8;
                int q1 = (int)((ql2[lane] & 0xF) | (((qh2[lane] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql2[lane + 32] & 0xF) | (((qh2[lane] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql2[lane] >> 4) | (((qh2[lane] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql2[lane + 32] >> 4) | (((qh2[lane] >> 6) & 3) << 4)) - 32;
                float w1 = d[r] * (float)sc2[is + 0] * (float)q1;
                float w2 = d[r] * (float)sc2[is + 2] * (float)q2;
                float w3 = d[r] * (float)sc2[is + 4] * (float)q3;
                float w4 = d[r] * (float)sc2[is + 6] * (float)q4;
                a0[r] += w1 * x0 + w2 * x1 + w3 * x2 + w4 * x3;
                a1[r] += w1 * z0 + w2 * z1 + w3 * z2 + w4 * z3;
            }
        }
    }
    for (uint r = 0; r < NDST; ++r) {
        float t0 = simd_sum(a0[r]); float t1 = simd_sum(a1[r]);
        if (lane == 0 && row0 + r < n_rows) { Y[row0 + r] = t0; Y[n_rows + row0 + r] = t1; }
    }
}

// one threadgroup of 256 threads; out = x/sqrt(mean(x^2)+eps) * w
kernel void rmsnorm(
    device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
    device float* out [[buffer(2)]], constant uint& n [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]], uint tcount [[threads_per_threadgroup]]) {
    threadgroup float partial[256];
    float s = 0.0f;
    for (uint i = tid; i < n; i += tcount) { float v = x[i]; s += v * v; }
    partial[tid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tcount / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = 1.0f / sqrt(partial[0] / float(n) + eps);
    for (uint i = tid; i < n; i += tcount) out[i] = x[i] * scale * w[i];
}

// interleaved (GPT-J) rope, in place; gid over n_heads*(hd/2) pairs
kernel void rope(
    device float* v [[buffer(0)]], constant uint& hd [[buffer(1)]],
    constant uint& pos [[buffer(2)]], constant float& base [[buffer(3)]],
    constant uint& neox [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    uint hh = hd / 2; uint pair = gid % hh; uint head = gid / hh;
    float freq = 1.0f / pow(base, (2.0f * float(pair)) / float(hd));
    float angle = float(pos) * freq; float c = cos(angle), s = sin(angle);
    uint a, b;
    if (neox) { a = head * hd + pair; b = head * hd + pair + hh; }
    else      { a = head * hd + 2 * pair; b = a + 1; }
    float x0 = v[a], x1 = v[b];
    v[a] = x0 * c - x1 * s; v[b] = x0 * s + x1 * c;
}

kernel void store_kv(
    device const float* k [[buffer(0)]], device const float* val [[buffer(1)]],
    device float* kcache [[buffer(2)]], device float* vcache [[buffer(3)]],
    constant uint& kv_dim [[buffer(4)]], constant uint& pos [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    kcache[pos * kv_dim + gid] = k[gid];
    vcache[pos * kv_dim + gid] = val[gid];
}

// One threadgroup per query head, `hd` threads (hd is a multiple of 32, <= 256).
// Thread `tid` owns output dim tid: a single scalar accumulator (no spills). The
// per-key q·k dot is reduced with `simd_sum` within each 32-lane simdgroup, then
// combined across the (<=8) simdgroups through a tiny shared array — ~2 barriers
// per key instead of a log2(hd)-step tree, and full occupancy (hd threads/head).
// Online (flash) softmax keeps it single-pass with no score storage.
kernel void attention(
    device const float* q [[buffer(0)]], device const float* kcache [[buffer(1)]],
    device const float* vcache [[buffer(2)]], device float* att [[buffer(3)]],
    constant uint& hd [[buffer(4)]], constant uint& kv_dim [[buffer(5)]],
    constant uint& kv_mul [[buffer(6)]], constant uint& pos [[buffer(7)]],
    uint h [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float sg[8];
    uint kvh = h / kv_mul; float scale = 1.0f / sqrt(float(hd));
    uint lane = tid & 31u, sgid = tid >> 5, nsg = hd >> 5;
    float qreg = q[h * hd + tid], acc = 0.0f, m = -3.0e38f, l = 0.0f;
    for (uint t = 0; t <= pos; ++t) {
        uint koff = t * kv_dim + kvh * hd;
        float ps = simd_sum(qreg * kcache[koff + tid]);
        if (lane == 0) sg[sgid] = ps;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float dot = 0.0f;
        for (uint g = 0; g < nsg; ++g) dot += sg[g];
        dot *= scale;
        float m_new = max(m, dot); float corr = exp(m - m_new); float p = exp(dot - m_new);
        l = l * corr + p;
        acc = acc * corr + p * vcache[koff + tid];
        m = m_new;
        threadgroup_barrier(mem_flags::mem_threadgroup); // before next iter rewrites sg
    }
    att[h * hd + tid] = acc / l;
}

// Batched attention: one threadgroup per (token, head), all in one dispatch — so
// prefill does ONE attention dispatch per layer instead of one per token. Token t
// attends causally to positions 0..(pos0+t). Same math as `attention`.
kernel void attention_batch(
    device const float* q [[buffer(0)]], device const float* kcache [[buffer(1)]],
    device const float* vcache [[buffer(2)]], device float* att [[buffer(3)]],
    constant uint& hd [[buffer(4)]], constant uint& kv_dim [[buffer(5)]],
    constant uint& kv_mul [[buffer(6)]], constant uint& pos0 [[buffer(7)]],
    constant uint& n_heads [[buffer(8)]],
    uint gid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float sg[8];
    uint token = gid / n_heads; uint h = gid % n_heads;
    uint q_dim = n_heads * hd; uint pos = pos0 + token;
    uint kvh = h / kv_mul; float scale = 1.0f / sqrt(float(hd));
    uint lane = tid & 31u, sgid = tid >> 5, nsg = hd >> 5;
    uint qbase = token * q_dim + h * hd;
    float qreg = q[qbase + tid], acc = 0.0f, m = -3.0e38f, l = 0.0f;
    for (uint t = 0; t <= pos; ++t) {
        uint koff = t * kv_dim + kvh * hd;
        float ps = simd_sum(qreg * kcache[koff + tid]);
        if (lane == 0) sg[sgid] = ps;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float dot = 0.0f;
        for (uint g = 0; g < nsg; ++g) dot += sg[g];
        dot *= scale;
        float m_new = max(m, dot); float corr = exp(m - m_new); float p = exp(dot - m_new);
        l = l * corr + p;
        acc = acc * corr + p * vcache[koff + tid];
        m = m_new;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    att[qbase + tid] = acc / l;
}

// Barrier-free batched attention: ONE 32-lane simdgroup per (token, head); each
// lane owns hd/32 head dims, so the q·k reduction is a single `simd_sum` with no
// threadgroup barriers (the per-key sync the other kernels pay). Online softmax.
// Requires hd a multiple of 32. EPL = hd/32 <= 8.
kernel void attention_batch_sg(
    device const float* q [[buffer(0)]], device const float* kcache [[buffer(1)]],
    device const float* vcache [[buffer(2)]], device float* att [[buffer(3)]],
    constant uint& hd [[buffer(4)]], constant uint& kv_dim [[buffer(5)]],
    constant uint& kv_mul [[buffer(6)]], constant uint& pos0 [[buffer(7)]],
    constant uint& n_heads [[buffer(8)]],
    uint gid [[threadgroup_position_in_grid]], uint lane [[thread_position_in_threadgroup]]) {
    uint token = gid / n_heads; uint h = gid % n_heads;
    uint q_dim = n_heads * hd; uint pos = pos0 + token;
    uint kvh = h / kv_mul; float scale = 1.0f / sqrt(float(hd));
    uint epl = hd / 32; uint qbase = token * q_dim + h * hd;
    float qreg[8], acc[8];
    for (uint e = 0; e < epl; ++e) { qreg[e] = q[qbase + e * 32 + lane]; acc[e] = 0.0f; }
    float m = -3.0e38f, l = 0.0f;
    for (uint t = 0; t <= pos; ++t) {
        uint koff = t * kv_dim + kvh * hd;
        float dot = 0.0f;
        for (uint e = 0; e < epl; ++e) dot += qreg[e] * kcache[koff + e * 32 + lane];
        dot = simd_sum(dot) * scale;
        float m_new = max(m, dot); float corr = exp(m - m_new); float p = exp(dot - m_new);
        l = l * corr + p;
        for (uint e = 0; e < epl; ++e) acc[e] = acc[e] * corr + p * vcache[koff + e * 32 + lane];
        m = m_new;
    }
    for (uint e = 0; e < epl; ++e) att[qbase + e * 32 + lane] = acc[e] / l;
}

kernel void swiglu(
    device float* gate [[buffer(0)]], device const float* up [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    float g = gate[gid]; gate[gid] = (g / (1.0f + exp(-g))) * up[gid];
}

kernel void add_inplace(
    device float* x [[buffer(0)]], device const float* y [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) { x[gid] += y[gid]; }

kernel void copy_buf(
    device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) { dst[gid] = src[gid]; }

// ---- batched (prefill) variants: one row = one token ----
// rmsnorm over each of N rows; one threadgroup per row.
kernel void rmsnorm_batch(
    device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
    device float* out [[buffer(2)]], constant uint& n [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint tcount [[threads_per_threadgroup]]) {
    threadgroup float partial[256];
    uint b = row * n;
    float s = 0.0f;
    for (uint i = tid; i < n; i += tcount) { float v = x[b + i]; s += v * v; }
    partial[tid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tcount / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = 1.0f / sqrt(partial[0] / float(n) + eps);
    for (uint i = tid; i < n; i += tcount) out[b + i] = x[b + i] * scale * w[i];
}

// rope each of N tokens by its own position (pos0 + row). gid over N*n_heads*(hd/2).
kernel void rope_batch(
    device float* v [[buffer(0)]], constant uint& hd [[buffer(1)]],
    constant uint& pos0 [[buffer(2)]], constant float& base [[buffer(3)]],
    constant uint& neox [[buffer(4)]], constant uint& n_heads [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    uint hh = hd / 2; uint per_row = n_heads * hh; uint row_stride = n_heads * hd;
    uint row = gid / per_row; uint r = gid % per_row;
    uint pair = r % hh; uint head = r / hh;
    uint pos = pos0 + row;
    float freq = 1.0f / pow(base, (2.0f * float(pair)) / float(hd));
    float angle = float(pos) * freq; float c = cos(angle), s = sin(angle);
    uint rb = row * row_stride, a, b2;
    if (neox) { a = rb + head * hd + pair; b2 = rb + head * hd + pair + hh; }
    else      { a = rb + head * hd + 2 * pair; b2 = a + 1; }
    float x0 = v[a], x1 = v[b2];
    v[a] = x0 * c - x1 * s; v[b2] = x0 * s + x1 * c;
}

// store N tokens' k,v into the cache at positions pos0..pos0+N. gid over N*kv_dim.
kernel void store_kv_batch(
    device const float* k [[buffer(0)]], device const float* val [[buffer(1)]],
    device float* kcache [[buffer(2)]], device float* vcache [[buffer(3)]],
    constant uint& kv_dim [[buffer(4)]], constant uint& pos0 [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    uint row = gid / kv_dim; uint e = gid % kv_dim;
    uint cpos = pos0 + row;
    kcache[cpos * kv_dim + e] = k[row * kv_dim + e];
    vcache[cpos * kv_dim + e] = val[row * kv_dim + e];
}

// general f32 matmul C[m,n] = A[m,k] @ B[k,n] (for training; one thread per output)
kernel void matmul_f32(
    device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]], constant uint& m [[buffer(3)]],
    constant uint& k [[buffer(4)]], constant uint& n [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    uint i = gid / n;
    uint j = gid % n;
    if (i >= m) return;
    float acc = 0.0f;
    for (uint p = 0; p < k; ++p) acc += A[i * k + p] * B[p * n + j];
    C[i * n + j] = acc;
}

// C[m,k] = A[m,n] @ B[k,n]^T  — B is the SAME [k,n] layout as the forward weight,
// so the cached forward buffer is reused here with no transpose (matmul backward dA).
kernel void matmul_abt(
    device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]], constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]], constant uint& k [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    uint i = gid / k;
    uint p = gid % k;
    if (i >= m) return;
    float acc = 0.0f;
    for (uint j = 0; j < n; ++j) acc += A[i * n + j] * B[p * n + j];
    C[i * k + p] = acc;
}

// C[k,n] = A[m,k]^T @ B[m,n]  — matmul/bmm backward dB, no CPU transpose.
kernel void matmul_atb(
    device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]], constant uint& k [[buffer(3)]],
    constant uint& m [[buffer(4)]], constant uint& n [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    uint p = gid / n;
    uint j = gid % n;
    if (p >= k) return;
    float acc = 0.0f;
    for (uint i = 0; i < m; ++i) acc += A[i * k + p] * B[i * n + j];
    C[p * n + j] = acc;
}

// Tiled C[M,N] = A[M,K] @ B[K,N] with threadgroup shared memory. Summation order
// is identical to the naive kernel (global k ascending), so results are
// bit-for-bit the same — purely faster. 16x16 tiles.
#define MM_TILE 16
kernel void matmul_f32_tiled(
    device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]], constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 bid [[threadgroup_position_in_grid]]) {
    threadgroup float As[MM_TILE][MM_TILE];
    threadgroup float Bs[MM_TILE][MM_TILE];
    uint row = bid.y * MM_TILE + tid.y;
    uint col = bid.x * MM_TILE + tid.x;
    float acc = 0.0f;
    uint nt = (K + MM_TILE - 1) / MM_TILE;
    for (uint t = 0; t < nt; ++t) {
        uint ac = t * MM_TILE + tid.x;
        uint br = t * MM_TILE + tid.y;
        As[tid.y][tid.x] = (row < M && ac < K) ? A[row * K + ac] : 0.0f;
        Bs[tid.y][tid.x] = (br < K && col < N) ? B[br * N + col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint p = 0; p < MM_TILE; ++p) acc += As[tid.y][p] * Bs[p][tid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// Tiled C[M,K] = A[M,Nc] @ B[K,Nc]^T  (matmul/bmm backward dA), contraction over Nc.
kernel void matmul_abt_tiled(
    device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]], constant uint& M [[buffer(3)]],
    constant uint& Nc [[buffer(4)]], constant uint& K [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 bid [[threadgroup_position_in_grid]]) {
    threadgroup float As[MM_TILE][MM_TILE];
    threadgroup float Bs[MM_TILE][MM_TILE];
    uint row = bid.y * MM_TILE + tid.y; // 0..M
    uint col = bid.x * MM_TILE + tid.x; // 0..K
    float acc = 0.0f;
    uint nt = (Nc + MM_TILE - 1) / MM_TILE;
    for (uint t = 0; t < nt; ++t) {
        uint aj = t * MM_TILE + tid.x;
        uint bj = t * MM_TILE + tid.y;
        As[tid.y][tid.x] = (row < M && aj < Nc) ? A[row * Nc + aj] : 0.0f;
        Bs[tid.y][tid.x] = (col < K && bj < Nc) ? B[col * Nc + bj] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint p = 0; p < MM_TILE; ++p) acc += As[tid.y][p] * Bs[p][tid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < K) C[row * K + col] = acc;
}

// Register-tiled C[M,K] = A[M,Nc] @ B[K,Nc]^T. Each thread computes an RTM x RTN
// micro-tile of outputs out of shared A/B block tiles -> RTM*RTN FMAs per RTM+RTN
// shared loads (high arithmetic intensity), vs 1 output/thread in matmul_abt_tiled.
// Standard fast-GEMM structure; shared arrays padded by 1 to soften bank conflicts.
#define RBM 64
#define RBN 64
#define RBK 8
#define RTM 4
#define RTN 4
kernel void matmul_abt_reg(
    device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]], constant uint& M [[buffer(3)]],
    constant uint& Nc [[buffer(4)]], constant uint& K [[buffer(5)]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 bid [[threadgroup_position_in_grid]]) {
    threadgroup float As[RBM][RBK + 1];
    threadgroup float Bs[RBN][RBK + 1];
    const uint TX = RBN / RTN;          // threads spanning K (output cols)
    const uint NTHREAD = (RBM / RTM) * (RBN / RTN);
    uint tid = tid2.x;                  // threadgroup is (256,1,1)
    uint tx = tid % TX, ty = tid / TX;
    uint block_m = bid.y * RBM, block_k = bid.x * RBN;
    float acc[RTM][RTN];
    for (uint i = 0; i < RTM; ++i) for (uint j = 0; j < RTN; ++j) acc[i][j] = 0.0f;
    uint nt = (Nc + RBK - 1) / RBK;
    for (uint t = 0; t < nt; ++t) {
        uint nc0 = t * RBK;
        for (uint idx = tid; idx < RBM * RBK; idx += NTHREAD) {
            uint r = idx / RBK, c = idx % RBK;
            uint gm = block_m + r, gn = nc0 + c;
            As[r][c] = (gm < M && gn < Nc) ? A[gm * Nc + gn] : 0.0f;
        }
        for (uint idx = tid; idx < RBN * RBK; idx += NTHREAD) {
            uint r = idx / RBK, c = idx % RBK;
            uint gk = block_k + r, gn = nc0 + c;
            Bs[r][c] = (gk < K && gn < Nc) ? B[gk * Nc + gn] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < RBK; ++kk) {
            float a[RTM], b[RTN];
            for (uint i = 0; i < RTM; ++i) a[i] = As[ty * RTM + i][kk];
            for (uint j = 0; j < RTN; ++j) b[j] = Bs[tx * RTN + j][kk];
            for (uint i = 0; i < RTM; ++i)
                for (uint j = 0; j < RTN; ++j) acc[i][j] += a[i] * b[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < RTM; ++i) {
        uint gm = block_m + ty * RTM + i;
        if (gm >= M) continue;
        for (uint j = 0; j < RTN; ++j) {
            uint gk = block_k + tx * RTN + j;
            if (gk < K) C[gm * K + gk] = acc[i][j];
        }
    }
}

kernel void transpose_f32(
    device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
    constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    uint total = rows * cols;
    if (gid >= total) return;
    uint r = gid / cols, c = gid % cols;
    dst[c * rows + r] = src[gid];
}

// Tiled C[K,N] = A[M,K]^T @ B[M,N]  (matmul/bmm backward dB), contraction over M.
kernel void matmul_atb_tiled(
    device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]], constant uint& K [[buffer(3)]],
    constant uint& M [[buffer(4)]], constant uint& N [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 bid [[threadgroup_position_in_grid]]) {
    threadgroup float As[MM_TILE][MM_TILE];
    threadgroup float Bs[MM_TILE][MM_TILE];
    uint row = bid.y * MM_TILE + tid.y; // 0..K
    uint col = bid.x * MM_TILE + tid.x; // 0..N
    float acc = 0.0f;
    uint nt = (M + MM_TILE - 1) / MM_TILE;
    for (uint t = 0; t < nt; ++t) {
        uint ai = t * MM_TILE + tid.x;
        uint bi = t * MM_TILE + tid.y;
        As[tid.y][tid.x] = (row < K && ai < M) ? A[ai * K + row] : 0.0f;
        Bs[tid.y][tid.x] = (col < N && bi < M) ? B[bi * N + col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint p = 0; p < MM_TILE; ++p) acc += As[tid.y][p] * Bs[p][tid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < K && col < N) C[row * N + col] = acc;
}

// Frozen NDHWC Conv3D + inference BatchNorm affine + ReLU. Weights use HOS's
// unfold-compatible layout [Kd*Kh*Kw*Cin, Cout]. One thread computes one output
// scalar; adjacent threads span output channels, giving coalesced weight reads.
kernel void conv3d_bn_relu(
    device const float* x     [[buffer(0)]],
    device const float* wgt   [[buffer(1)]],
    device const float* scale [[buffer(2)]],
    device const float* shift [[buffer(3)]],
    device float* out         [[buffer(4)]],
    constant uint* p          [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    // p: n,d,h,w,cin,kd,kh,kw,cout,od,oh,ow,sd,sh,sw,pd,ph,pw,total
    uint total = p[18];
    if (gid >= total) return;
    uint co = gid % p[8];
    uint q = gid / p[8];
    uint ox = q % p[11]; q /= p[11];
    uint oy = q % p[10]; q /= p[10];
    uint oz = q % p[9];
    uint ni = q / p[9];
    float acc = 0.0f;
    for (uint kz = 0; kz < p[5]; ++kz) {
        int iz = int(oz * p[12] + kz) - int(p[15]);
        if (iz < 0 || iz >= int(p[1])) continue;
        for (uint ky = 0; ky < p[6]; ++ky) {
            int iy = int(oy * p[13] + ky) - int(p[16]);
            if (iy < 0 || iy >= int(p[2])) continue;
            for (uint kx = 0; kx < p[7]; ++kx) {
                int ix = int(ox * p[14] + kx) - int(p[17]);
                if (ix < 0 || ix >= int(p[3])) continue;
                uint xb = ((((ni * p[1] + uint(iz)) * p[2] + uint(iy)) * p[3] + uint(ix)) * p[4]);
                uint wb = (((kz * p[6] + ky) * p[7] + kx) * p[4]) * p[8] + co;
                for (uint ci = 0; ci < p[4]; ++ci) acc += x[xb + ci] * wgt[wb + ci * p[8]];
            }
        }
    }
    out[gid] = max(acc * scale[co] + shift[co], 0.0f);
}

// Backward input for the direct Conv3D + frozen BN affine + ReLU operation.
// One thread owns one input scalar, so no atomics are required.
kernel void conv3d_bn_relu_dx(
    device const float* wgt   [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const float* out   [[buffer(2)]],
    device const float* gout  [[buffer(3)]],
    device float* dx          [[buffer(4)]],
    constant uint* p          [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    // p: n,d,h,w,cin,kd,kh,kw,cout,od,oh,ow,sd,sh,sw,pd,ph,pw,total_out,total_x,total_w
    if (gid >= p[19]) return;
    uint ci = gid % p[4];
    uint q = gid / p[4];
    uint ix = q % p[3]; q /= p[3];
    uint iy = q % p[2]; q /= p[2];
    uint iz = q % p[1];
    uint ni = q / p[1];
    float acc = 0.0f;
    for (uint kz = 0; kz < p[5]; ++kz) {
        int nz = int(iz + p[15]) - int(kz);
        if (nz < 0 || (uint(nz) % p[12]) != 0) continue;
        uint oz = uint(nz) / p[12];
        if (oz >= p[9]) continue;
        for (uint ky = 0; ky < p[6]; ++ky) {
            int ny = int(iy + p[16]) - int(ky);
            if (ny < 0 || (uint(ny) % p[13]) != 0) continue;
            uint oy = uint(ny) / p[13];
            if (oy >= p[10]) continue;
            for (uint kx = 0; kx < p[7]; ++kx) {
                int nx = int(ix + p[17]) - int(kx);
                if (nx < 0 || (uint(nx) % p[14]) != 0) continue;
                uint ox = uint(nx) / p[14];
                if (ox >= p[11]) continue;
                uint ob = ((((ni * p[9] + oz) * p[10] + oy) * p[11] + ox) * p[8]);
                uint wb = (((kz * p[6] + ky) * p[7] + kx) * p[4] + ci) * p[8];
                for (uint co = 0; co < p[8]; ++co) {
                    float go = (out[ob + co] > 0.0f) ? gout[ob + co] * scale[co] : 0.0f;
                    acc += go * wgt[wb + co];
                }
            }
        }
    }
    dx[gid] = acc;
}

// Backward weight. One thread owns one [kz,ky,kx,ci,co] scalar and reduces over
// batch/output positions, again avoiding atomics and nondeterministic reductions.
kernel void conv3d_bn_relu_dw(
    device const float* x     [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const float* out   [[buffer(2)]],
    device const float* gout  [[buffer(3)]],
    device float* dw          [[buffer(4)]],
    constant uint* p          [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p[20]) return;
    uint co = gid % p[8];
    uint q = gid / p[8];
    uint ci = q % p[4]; q /= p[4];
    uint kx = q % p[7]; q /= p[7];
    uint ky = q % p[6];
    uint kz = q / p[6];
    float acc = 0.0f;
    for (uint ni = 0; ni < p[0]; ++ni) {
        for (uint oz = 0; oz < p[9]; ++oz) {
            int iz = int(oz * p[12] + kz) - int(p[15]);
            if (iz < 0 || iz >= int(p[1])) continue;
            for (uint oy = 0; oy < p[10]; ++oy) {
                int iy = int(oy * p[13] + ky) - int(p[16]);
                if (iy < 0 || iy >= int(p[2])) continue;
                for (uint ox = 0; ox < p[11]; ++ox) {
                    int ix = int(ox * p[14] + kx) - int(p[17]);
                    if (ix < 0 || ix >= int(p[3])) continue;
                    uint oi = ((((ni * p[9] + oz) * p[10] + oy) * p[11] + ox) * p[8] + co);
                    if (out[oi] <= 0.0f) continue;
                    uint xi = ((((ni * p[1] + uint(iz)) * p[2] + uint(iy)) * p[3] + uint(ix)) * p[4] + ci);
                    acc += x[xi] * gout[oi] * scale[co];
                }
            }
        }
    }
    dw[gid] = acc;
}

// Materialize NDHWC Conv3D patches entirely in unified Metal memory. Used only
// transiently during dWeight so the existing tiled A^T*B GEMM can replace the
// slow per-weight direct reduction without retaining im2col in the autograd graph.
kernel void unfold3d_gpu(
    device const float* x [[buffer(0)]],
    device float* cols    [[buffer(1)]],
    constant uint* p      [[buffer(2)]],
    constant uint& total_cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= total_cols) return;
    uint patch = p[5] * p[6] * p[7] * p[4];
    uint pc = gid % patch;
    uint row = gid / patch;
    uint ci = pc % p[4]; pc /= p[4];
    uint kx = pc % p[7]; pc /= p[7];
    uint ky = pc % p[6];
    uint kz = pc / p[6];
    uint ox = row % p[11]; row /= p[11];
    uint oy = row % p[10]; row /= p[10];
    uint oz = row % p[9];
    uint ni = row / p[9];
    int iz = int(oz * p[12] + kz) - int(p[15]);
    int iy = int(oy * p[13] + ky) - int(p[16]);
    int ix = int(ox * p[14] + kx) - int(p[17]);
    if (iz < 0 || iz >= int(p[1]) || iy < 0 || iy >= int(p[2]) || ix < 0 || ix >= int(p[3])) {
        cols[gid] = 0.0f;
    } else {
        uint xi = ((((ni * p[1] + uint(iz)) * p[2] + uint(iy)) * p[3] + uint(ix)) * p[4] + ci);
        cols[gid] = x[xi];
    }
}

kernel void conv3d_grad_raw(
    device const float* scale [[buffer(0)]],
    device const float* out   [[buffer(1)]],
    device const float* gout  [[buffer(2)]],
    device float* grad_raw    [[buffer(3)]],
    constant uint* p          [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p[18]) return;
    uint co = gid % p[8];
    grad_raw[gid] = (out[gid] > 0.0f) ? gout[gid] * scale[co] : 0.0f;
}

// Fold frozen inference-BN and ReLU after an unfold @ weight GEMM. Keeping this
// separate lets the large spatial kernels use the shared-memory tiled GEMM while
// cheap 1x1x1 convolutions stay on the direct fused path.
kernel void conv3d_affine_relu(
    device const float* raw   [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const float* shift [[buffer(2)]],
    device float* out         [[buffer(3)]],
    constant uint* p          [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p[18]) return;
    uint co = gid % p[8];
    out[gid] = max(raw[gid] * scale[co] + shift[co], 0.0f);
}

// Inverse of unfold for dInput after dCols = grad_raw @ weight^T. One thread
// owns one input scalar, so overlapping patches are reduced deterministically
// without float atomics.
kernel void fold3d_dx(
    device const float* dcols [[buffer(0)]],
    device float* dx          [[buffer(1)]],
    constant uint* p          [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p[19]) return;
    uint q = gid;
    uint ci = q % p[4]; q /= p[4];
    uint ix = q % p[3]; q /= p[3];
    uint iy = q % p[2]; q /= p[2];
    uint iz = q % p[1];
    uint ni = q / p[1];
    uint patch = p[5] * p[6] * p[7] * p[4];
    float acc = 0.0f;
    for (uint kz = 0; kz < p[5]; ++kz) {
        int oz0 = int(iz) + int(p[15]) - int(kz);
        if (oz0 < 0 || (uint(oz0) % p[12]) != 0) continue;
        uint oz = uint(oz0) / p[12];
        if (oz >= p[9]) continue;
        for (uint ky = 0; ky < p[6]; ++ky) {
            int oy0 = int(iy) + int(p[16]) - int(ky);
            if (oy0 < 0 || (uint(oy0) % p[13]) != 0) continue;
            uint oy = uint(oy0) / p[13];
            if (oy >= p[10]) continue;
            for (uint kx = 0; kx < p[7]; ++kx) {
                int ox0 = int(ix) + int(p[17]) - int(kx);
                if (ox0 < 0 || (uint(ox0) % p[14]) != 0) continue;
                uint ox = uint(ox0) / p[14];
                if (ox >= p[11]) continue;
                uint row = (((ni * p[9] + oz) * p[10] + oy) * p[11] + ox);
                uint pc = (((kz * p[6] + ky) * p[7] + kx) * p[4] + ci);
                acc += dcols[row * patch + pc];
            }
        }
    }
    dx[gid] = acc;
}

// Raw NDHWC depthwise Conv3D. Weight layout is [Cin, Kd*Kh*Kw], matching
// HOS grouped-conv depthwise storage. Used for grouped/depthwise 3D feature
// convolutions.
kernel void depthwise_conv3d(
    device const float* x   [[buffer(0)]],
    device const float* wgt [[buffer(1)]],
    device float* out       [[buffer(2)]],
    constant uint* p        [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    // p: n,d,h,w,c,kd,kh,kw,od,oh,ow,sd,sh,sw,pd,ph,pw,total_out,total_x,total_w
    if (gid >= p[17]) return;
    uint c = gid % p[4];
    uint q = gid / p[4];
    uint ox = q % p[10]; q /= p[10];
    uint oy = q % p[9]; q /= p[9];
    uint oz = q % p[8];
    uint ni = q / p[8];
    float acc = 0.0f;
    uint patch = p[5] * p[6] * p[7];
    for (uint kz = 0; kz < p[5]; ++kz) {
        int iz = int(oz * p[11] + kz) - int(p[14]);
        if (iz < 0 || iz >= int(p[1])) continue;
        for (uint ky = 0; ky < p[6]; ++ky) {
            int iy = int(oy * p[12] + ky) - int(p[15]);
            if (iy < 0 || iy >= int(p[2])) continue;
            for (uint kx = 0; kx < p[7]; ++kx) {
                int ix = int(ox * p[13] + kx) - int(p[16]);
                if (ix < 0 || ix >= int(p[3])) continue;
                uint xi = ((((ni * p[1] + uint(iz)) * p[2] + uint(iy)) * p[3] + uint(ix)) * p[4] + c);
                uint wi = c * patch + ((kz * p[6] + ky) * p[7] + kx);
                acc += x[xi] * wgt[wi];
            }
        }
    }
    out[gid] = acc;
}

kernel void depthwise_conv3d_dx(
    device const float* wgt [[buffer(0)]],
    device const float* gout [[buffer(1)]],
    device float* dx [[buffer(2)]],
    constant uint* p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p[18]) return;
    uint c = gid % p[4];
    uint q = gid / p[4];
    uint ix = q % p[3]; q /= p[3];
    uint iy = q % p[2]; q /= p[2];
    uint iz = q % p[1];
    uint ni = q / p[1];
    uint patch = p[5] * p[6] * p[7];
    float acc = 0.0f;
    for (uint kz = 0; kz < p[5]; ++kz) {
        int oz0 = int(iz + p[14]) - int(kz);
        if (oz0 < 0 || (uint(oz0) % p[11]) != 0) continue;
        uint oz = uint(oz0) / p[11];
        if (oz >= p[8]) continue;
        for (uint ky = 0; ky < p[6]; ++ky) {
            int oy0 = int(iy + p[15]) - int(ky);
            if (oy0 < 0 || (uint(oy0) % p[12]) != 0) continue;
            uint oy = uint(oy0) / p[12];
            if (oy >= p[9]) continue;
            for (uint kx = 0; kx < p[7]; ++kx) {
                int ox0 = int(ix + p[16]) - int(kx);
                if (ox0 < 0 || (uint(ox0) % p[13]) != 0) continue;
                uint ox = uint(ox0) / p[13];
                if (ox >= p[10]) continue;
                uint oi = ((((ni * p[8] + oz) * p[9] + oy) * p[10] + ox) * p[4] + c);
                uint wi = c * patch + ((kz * p[6] + ky) * p[7] + kx);
                acc += gout[oi] * wgt[wi];
            }
        }
    }
    dx[gid] = acc;
}

kernel void depthwise_conv3d_dw(
    device const float* x [[buffer(0)]],
    device const float* gout [[buffer(1)]],
    device float* dw [[buffer(2)]],
    constant uint* p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p[19]) return;
    uint patch = p[5] * p[6] * p[7];
    uint c = gid / patch;
    uint pc = gid % patch;
    uint kx = pc % p[7]; pc /= p[7];
    uint ky = pc % p[6];
    uint kz = pc / p[6];
    float acc = 0.0f;
    for (uint ni = 0; ni < p[0]; ++ni) {
        for (uint oz = 0; oz < p[8]; ++oz) {
            int iz = int(oz * p[11] + kz) - int(p[14]);
            if (iz < 0 || iz >= int(p[1])) continue;
            for (uint oy = 0; oy < p[9]; ++oy) {
                int iy = int(oy * p[12] + ky) - int(p[15]);
                if (iy < 0 || iy >= int(p[2])) continue;
                for (uint ox = 0; ox < p[10]; ++ox) {
                    int ix = int(ox * p[13] + kx) - int(p[16]);
                    if (ix < 0 || ix >= int(p[3])) continue;
                    uint xi = ((((ni * p[1] + uint(iz)) * p[2] + uint(iy)) * p[3] + uint(ix)) * p[4] + c);
                    uint oi = ((((ni * p[8] + oz) * p[9] + oy) * p[10] + ox) * p[4] + c);
                    acc += x[xi] * gout[oi];
                }
            }
        }
    }
    dw[gid] = acc;
}

// Gated delta-net per-head recurrence. One threadgroup of `n` threads (n=head dim);
// thread j owns row/column j of the head's nxn state S. g = exp(decay), beta scalar.
//   S *= g;  sk[j]=ΣS[j][i]k[i];  d[j]=β(v[j]-sk[j]);  o[j]=ΣS[j][i]q[i]+d[j](k·q);  S[i][j]+=d[i]k[j]
// q is expected pre-scaled by 1/sqrt(n).
kernel void deltanet(
    device float* S       [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device const float* k [[buffer(2)]],
    device const float* v [[buffer(3)]],
    device float* o       [[buffer(4)]],
    constant float& g     [[buffer(5)]],
    constant float& beta  [[buffer(6)]],
    constant uint& n      [[buffer(7)]],
    uint j [[thread_position_in_threadgroup]]) {
    threadgroup float dsh[256];
    // decay row j and accumulate sk, sq over the decayed row
    float kq = 0.0f;
    for (uint i = 0; i < n; ++i) kq += k[i] * q[i];
    float sk = 0.0f, sq = 0.0f;
    for (uint i = 0; i < n; ++i) {
        float s = S[j * n + i] * g;
        S[j * n + i] = s;
        sk += s * k[i];
        sq += s * q[i];
    }
    float dj = beta * (v[j] - sk);
    o[j] = sq + dj * kq;
    dsh[j] = dj;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // rank-1 update of column j: S[i][j] += d[i] * k[j]
    float kj = k[j];
    for (uint i = 0; i < n; ++i) S[i * n + j] += dsh[i] * kj;
}

// ---- qwen35-specific kernels ----

// de-interleave wq output [q(hd), gate(hd)] per head -> contiguous q and gate
kernel void extract_qgate(
    device const float* qfull [[buffer(0)]], device float* q [[buffer(1)]],
    device float* gate [[buffer(2)]], constant uint& hd [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    uint head = gid / hd; uint i = gid % hd;
    q[gid] = qfull[head * hd * 2 + i];
    gate[gid] = qfull[head * hd * 2 + hd + i];
}

// per-head RMSNorm in place (one threadgroup per head, hd threads)
kernel void rmsnorm_heads(
    device float* x [[buffer(0)]], device const float* w [[buffer(1)]],
    constant uint& hd [[buffer(2)]], constant float& eps [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tcount [[threads_per_threadgroup]]) {
    threadgroup float part[256];
    uint base = tg * hd; float s = 0.0f;
    for (uint i = tid; i < hd; i += tcount) { float v = x[base + i]; s += v * v; }
    part[tid] = s; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st = tcount / 2; st > 0; st >>= 1) { if (tid < st) part[tid] += part[tid + st]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float scale = 1.0f / sqrt(part[0] / float(hd) + eps);
    for (uint i = tid; i < hd; i += tcount) x[base + i] = x[base + i] * scale * w[i];
}

// per-head L2 normalize in place (x / sqrt(sumsq + eps))
kernel void l2norm_heads(
    device float* x [[buffer(0)]], constant uint& hd [[buffer(1)]], constant float& eps [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tcount [[threads_per_threadgroup]]) {
    threadgroup float part[256];
    uint base = tg * hd; float s = 0.0f;
    for (uint i = tid; i < hd; i += tcount) { float v = x[base + i]; s += v * v; }
    part[tid] = s; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st = tcount / 2; st > 0; st >>= 1) { if (tid < st) part[tid] += part[tid + st]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float scale = 1.0f / sqrt(part[0] + eps);
    for (uint i = tid; i < hd; i += tcount) x[base + i] *= scale;
}

kernel void sigmoid_mul(
    device float* a [[buffer(0)]], device const float* b [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) { a[gid] *= 1.0f / (1.0f + exp(-b[gid])); }

kernel void sigmoid_inplace(
    device float* a [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    a[gid] = 1.0f / (1.0f + exp(-a[gid]));
}

// decay[h] = a[h] * softplus(alpha[h] + dt[h]), written in place over alpha
kernel void ssm_decay(
    device float* alpha [[buffer(0)]], device const float* a [[buffer(1)]],
    device const float* dt [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    float z = alpha[gid] + dt[gid];
    float sp = z > 20.0f ? z : log(1.0f + exp(z));
    alpha[gid] = a[gid] * sp;
}

// causal depthwise conv1d + silu, updates conv state. one thread per channel.
kernel void conv1d(
    device const float* qkv [[buffer(0)]], device float* cstate [[buffer(1)]],
    device const float* cw [[buffer(2)]], device float* out [[buffer(3)]],
    constant uint& kk [[buffer(4)]], uint ch [[thread_position_in_grid]]) {
    float acc = 0.0f;
    for (uint t = 0; t < kk - 1; ++t) acc += cstate[ch * (kk - 1) + t] * cw[ch * kk + t];
    acc += qkv[ch] * cw[ch * kk + (kk - 1)];
    out[ch] = acc / (1.0f + exp(-acc));
    for (uint t = 0; t + 2 < kk; ++t) cstate[ch * (kk - 1) + t] = cstate[ch * (kk - 1) + t + 1];
    cstate[ch * (kk - 1) + (kk - 2)] = qkv[ch];
}

// multi-head gated delta-net: threadgroup = v-head h, thread j = state row/col.
// q,k have nk heads (tiled to nv: k-head = h % nk); v has nv heads. q pre-L2'd, scaled here.
kernel void deltanet_multi(
    device float* S [[buffer(0)]], device const float* q [[buffer(1)]],
    device const float* k [[buffer(2)]], device const float* v [[buffer(3)]],
    device const float* decay [[buffer(4)]], device const float* beta [[buffer(5)]],
    device float* o [[buffer(6)]],
    constant uint& hv [[buffer(7)]], constant uint& hk [[buffer(8)]],
    constant uint& nk [[buffer(9)]], constant float& qscale [[buffer(10)]],
    uint h [[threadgroup_position_in_grid]], uint j [[thread_position_in_threadgroup]]) {
    threadgroup float dsh[256];
    uint kh = h % nk;
    device float* Sh = S + h * hv * hv;
    device const float* qh = q + kh * hk;
    device const float* kh2 = k + kh * hk;
    device const float* vh = v + h * hv;
    float g = exp(decay[h]); float b = beta[h];
    float kq = 0.0f;
    for (uint i = 0; i < hk; ++i) kq += kh2[i] * (qh[i] * qscale);
    float sk = 0.0f, sq = 0.0f;
    for (uint i = 0; i < hk; ++i) { float s = Sh[j * hv + i] * g; Sh[j * hv + i] = s; sk += s * kh2[i]; sq += s * (qh[i] * qscale); }
    float dj = b * (vh[j] - sk);
    o[h * hv + j] = sq + dj * kq;
    dsh[j] = dj; threadgroup_barrier(mem_flags::mem_threadgroup);
    float kj = kh2[j];
    for (uint i = 0; i < hv; ++i) Sh[i * hv + j] += dsh[i] * kj;
}

// per-head gated RMSNorm: o = rmsnorm(o, w) * silu(z), in place over o
kernel void gated_norm(
    device float* o [[buffer(0)]], device const float* w [[buffer(1)]],
    device const float* z [[buffer(2)]], constant uint& hv [[buffer(3)]], constant float& eps [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tcount [[threads_per_threadgroup]]) {
    threadgroup float part[256];
    uint base = tg * hv; float s = 0.0f;
    for (uint i = tid; i < hv; i += tcount) { float v = o[base + i]; s += v * v; }
    part[tid] = s; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st = tcount / 2; st > 0; st >>= 1) { if (tid < st) part[tid] += part[tid + st]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float scale = 1.0f / sqrt(part[0] / float(hv) + eps);
    for (uint i = tid; i < hv; i += tcount) {
        float norm = o[base + i] * scale * w[i];
        float zz = z[base + i];
        o[base + i] = norm * (zz / (1.0f + exp(-zz)));
    }
}

// partial NEOX rope: rotate first n_rot dims of each hd-wide head
kernel void rope_partial(
    device float* v [[buffer(0)]], constant uint& hd [[buffer(1)]], constant uint& n_rot [[buffer(2)]],
    constant uint& pos [[buffer(3)]], constant float& base [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    uint hh = n_rot / 2; uint pair = gid % hh; uint head = gid / hh;
    uint off = head * hd;
    float freq = 1.0f / pow(base, (2.0f * float(pair)) / float(n_rot));
    float ang = float(pos) * freq; float c = cos(ang), s = sin(ang);
    uint a = off + pair, bb = off + pair + hh;
    float x0 = v[a], x1 = v[bb];
    v[a] = x0 * c - x1 * s; v[bb] = x0 * s + x1 * c;
}

// ---- Gemma-4 specific kernels (resident decoder) ------------------------------
// NEOX rope from a precomputed inv_freq table (Gemma uses per-layer tables, and the
// global layers are partial — freqs past a cutoff are 0). gid over n_heads*half.
kernel void rope_table(
    device float* v [[buffer(0)]], device const float* inv_freq [[buffer(1)]],
    constant uint& hh [[buffer(2)]], constant uint& pos [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    uint head = gid / hh; uint j = gid % hh; uint base = head * (hh * 2);
    float ang = float(pos) * inv_freq[j]; float c = cos(ang), s = sin(ang);
    float a = v[base + j], b = v[base + j + hh];
    v[base + j] = a * c - b * s;
    v[base + j + hh] = b * c + a * s;
}

// Attention over cached keys [lo, pos] with an explicit scale (Gemma uses 1.0).
// One threadgroup per query head, `hd` threads (hd multiple of 32, <= 512). Online
// (flash) softmax. `hd` up to 512 => up to 16 simdgroups, so the combine array is 16.
kernel void gemma_attn(
    device const float* q [[buffer(0)]], device const float* kcache [[buffer(1)]],
    device const float* vcache [[buffer(2)]], device float* att [[buffer(3)]],
    constant uint& hd [[buffer(4)]], constant uint& kv_dim [[buffer(5)]],
    constant uint& kv_mul [[buffer(6)]], constant uint& pos [[buffer(7)]],
    constant uint& lo [[buffer(8)]], constant float& scale [[buffer(9)]],
    uint h [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float sg[16];
    uint kvh = h / kv_mul;
    uint lane = tid & 31u, sgid = tid >> 5, nsg = hd >> 5;
    float qreg = q[h * hd + tid], acc = 0.0f, m = -3.0e38f, l = 0.0f;
    for (uint t = lo; t <= pos; ++t) {
        uint koff = t * kv_dim + kvh * hd;
        float ps = simd_sum(qreg * kcache[koff + tid]);
        if (lane == 0) sg[sgid] = ps;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float dot = 0.0f;
        for (uint g = 0; g < nsg; ++g) dot += sg[g];
        dot *= scale;
        float m_new = max(m, dot); float corr = exp(m - m_new); float p = exp(dot - m_new);
        l = l * corr + p;
        acc = acc * corr + p * vcache[koff + tid];
        m = m_new;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    att[h * hd + tid] = acc / l;
}

// gate = gelu_tanh(gate) * up
kernel void gelu_mul(
    device float* gate [[buffer(0)]], device const float* up [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    float x = gate[gid];
    float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    gate[gid] = (0.5f * x * (1.0f + precise::tanh(inner))) * up[gid];
}

// x *= s  (Gemma's per-layer residual scalar)
kernel void scalar_mul(
    device float* x [[buffer(0)]], constant float& s [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) { x[gid] *= s; }

// x = cap * tanh(x / cap)  (Gemma final logit softcap)
kernel void softcap(
    device float* x [[buffer(0)]], constant float& cap [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) { x[gid] = cap * tanh(x[gid] / cap); }
"#;

pub struct Gpu {
    device: Device,
    queue: metal::CommandQueue,
    matvec: ComputePipelineState,
    /// Coalesced K-quant matvec kernels keyed by ggml type (Q4_K/Q5_K/Q6_K/Q8_0/
    /// Q4_0). Lets `matvec_into` run a natively-quantized `GpuMatrix` (used by the
    /// Gemma4 `--q4k --gpu` path) instead of only f16.
    quant_mv: HashMap<u32, ComputePipelineState>,
    deltanet: ComputePipelineState,
    conv3d_bn_relu: ComputePipelineState,
    conv3d_bn_relu_dx: ComputePipelineState,
    unfold3d_gpu: ComputePipelineState,
    conv3d_grad_raw: ComputePipelineState,
    conv3d_affine_relu: ComputePipelineState,
    fold3d_dx: ComputePipelineState,
    depthwise_conv3d: ComputePipelineState,
    depthwise_conv3d_dx: ComputePipelineState,
    depthwise_conv3d_dw: ComputePipelineState,
    matmul_f32_tiled: ComputePipelineState, // tiled (shared-mem) forward GEMM
    transpose_f32: ComputePipelineState,
    matmul_abt_tiled: ComputePipelineState, // tiled C = A @ B^T (backward dA, reuses cached B)
    matmul_atb_tiled: ComputePipelineState, // tiled C = A^T @ B (backward dB)
    // Standalone batched q4_k prefill (used by the Gemma4 custom-arch batched
    // prefill): dequant one weight to f32 scratch, then a register-tiled GEMM
    // Y[ntok,n_rows] = X[ntok,in_dim] @ W^T. Weight is read/dequanted ONCE and
    // reused across all `ntok` tokens (vs once-per-token in `matvec_into`).
    dequant_q4k: ComputePipelineState,
    dequant_q6k: ComputePipelineState,
    // register-tiled small-batch matmuls: read the quantized weight ONCE and do up
    // to 8 tokens in registers (no f32 scratch) — the weights-read-once primitive
    // for MTP's 2-token speculative verify. One per quant type present in Q4_K_M.
    mm_q4k_small: ComputePipelineState,
    mm_q6k_small: ComputePipelineState,
    mm_q4k_2tok: ComputePipelineState, // efficient coalesced 2-token q4_k (verify)
    mm_q6k_2tok: ComputePipelineState, // efficient coalesced 2-token q6_k (verify)
    mm_abt_reg: ComputePipelineState,
    wdq: RefCell<Option<metal::Buffer>>, // f32 scratch for one dequantized weight
    px_buf: RefCell<Option<metal::Buffer>>, // persistent batched activation input scratch
    py_buf: RefCell<Option<metal::Buffer>>, // persistent batched output scratch
    // persistent scratch buffers, reused across matvecs to avoid per-call allocation
    x_buf: metal::Buffer,
    y_buf: metal::Buffer,
    // GPU buffers for FROZEN operands (keyed by stable tensor identity). A frozen
    // base weight is reused every matmul of every step; caching it uploads the
    // tensor to the GPU once instead of re-copying it on every call.
    wcache: RefCell<HashMap<u64, metal::Buffer>>,
}

const X_CAP: usize = 64 * 1024; // max input width (>= largest ffn_dim)
const Y_CAP: usize = 512 * 1024; // max output rows (>= largest vocab)

/// A weight matrix resident on the GPU as [n_rows, in_dim] row-major, stored in
/// its native ggml type (f16/quantized); dequantized inside the matvec kernel.
pub struct GpuMatrix {
    buf: metal::Buffer,
    pub n_rows: usize,
    pub in_dim: usize,
    pub ggml_type: u32,
}

impl GpuMatrix {
    pub fn buffer(&self) -> &metal::BufferRef {
        &self.buf
    }
}

impl Gpu {
    pub fn new() -> Gpu {
        let device = Device::system_default().expect("no Metal device");
        let queue = device.new_command_queue();
        let lib = device
            .new_library_with_source(KERNEL_SRC, &CompileOptions::new())
            .expect("compile MSL");
        let func = lib.get_function("matvec", None).expect("get matvec");
        let matvec = device
            .new_compute_pipeline_state_with_function(&func)
            .expect("pipeline");
        let lib2 = device
            .new_library_with_source(FUSED_SRC, &CompileOptions::new())
            .expect("compile fused MSL");
        // Coalesced K-quant matvec pipelines (for a natively-quantized GpuMatrix).
        let mut quant_mv = HashMap::new();
        quant_mv.insert(GGML_Q4_K, pipeline(&device, &lib2, "matvec_q4k_co"));
        quant_mv.insert(GGML_Q5_K, pipeline(&device, &lib2, "matvec_q5k_co"));
        quant_mv.insert(GGML_Q6_K, pipeline(&device, &lib2, "matvec_q6k_co"));
        quant_mv.insert(GGML_Q8_0, pipeline(&device, &lib2, "matvec_q8_0_co"));
        quant_mv.insert(GGML_Q4_0, pipeline(&device, &lib2, "matvec_q4_0_co"));
        // hq4 (HOS-native NF4): the coalesced kernel is parity-checked against the
        // CPU decode_hq4_into path (tests/parity_hq4.rs), so a resident HQ4 GpuMatrix
        // (e.g. the gemma4 --gpu path) matvecs on-device instead of panicking.
        quant_mv.insert(
            crate::model::HQ4_TYPE,
            pipeline(&device, &lib2, "matvec_hq4_co"),
        );
        let deltanet = device
            .new_compute_pipeline_state_with_function(
                &lib2.get_function("deltanet", None).expect("deltanet"),
            )
            .expect("deltanet pipeline");
        let conv3d_bn_relu = pipeline(&device, &lib2, "conv3d_bn_relu");
        let conv3d_bn_relu_dx = pipeline(&device, &lib2, "conv3d_bn_relu_dx");
        let unfold3d_gpu = pipeline(&device, &lib2, "unfold3d_gpu");
        let conv3d_grad_raw = pipeline(&device, &lib2, "conv3d_grad_raw");
        let conv3d_affine_relu = pipeline(&device, &lib2, "conv3d_affine_relu");
        let fold3d_dx = pipeline(&device, &lib2, "fold3d_dx");
        let depthwise_conv3d = pipeline(&device, &lib2, "depthwise_conv3d");
        let depthwise_conv3d_dx = pipeline(&device, &lib2, "depthwise_conv3d_dx");
        let depthwise_conv3d_dw = pipeline(&device, &lib2, "depthwise_conv3d_dw");
        let matmul_f32_tiled = device
            .new_compute_pipeline_state_with_function(
                &lib2
                    .get_function("matmul_f32_tiled", None)
                    .expect("matmul_f32_tiled"),
            )
            .expect("matmul_f32_tiled pipeline");
        let transpose_f32 = pipeline(&device, &lib2, "transpose_f32");
        let matmul_abt_tiled = device
            .new_compute_pipeline_state_with_function(
                &lib2
                    .get_function("matmul_abt_tiled", None)
                    .expect("matmul_abt_tiled"),
            )
            .expect("matmul_abt_tiled pipeline");
        let matmul_atb_tiled = device
            .new_compute_pipeline_state_with_function(
                &lib2
                    .get_function("matmul_atb_tiled", None)
                    .expect("matmul_atb_tiled"),
            )
            .expect("matmul_atb_tiled pipeline");
        let dequant_q4k = pipeline(&device, &lib2, "dequant_q4k_to_f32");
        let dequant_q6k = pipeline(&device, &lib2, "dequant_q6k_to_f32");
        let mm_q4k_small = pipeline(&device, &lib2, "matmul_q4k_batch");
        let mm_q6k_small = pipeline(&device, &lib2, "matmul_q6k_batch");
        let mm_q4k_2tok = pipeline(&device, &lib2, "matmul_q4k_co2");
        let mm_q6k_2tok = pipeline(&device, &lib2, "matmul_q6k_co2");
        let mm_abt_reg = pipeline(&device, &lib2, "matmul_abt_reg");
        eprintln!("[hos] Metal device: {}", device.name());
        let x_buf = device.new_buffer((X_CAP * 4) as u64, MTLResourceOptions::StorageModeShared);
        let y_buf = device.new_buffer((Y_CAP * 4) as u64, MTLResourceOptions::StorageModeShared);
        Gpu {
            device,
            queue,
            matvec,
            quant_mv,
            deltanet,
            conv3d_bn_relu,
            conv3d_bn_relu_dx,
            unfold3d_gpu,
            conv3d_grad_raw,
            conv3d_affine_relu,
            fold3d_dx,
            depthwise_conv3d,
            depthwise_conv3d_dx,
            depthwise_conv3d_dw,
            matmul_f32_tiled,
            transpose_f32,
            matmul_abt_tiled,
            matmul_atb_tiled,
            dequant_q4k,
            dequant_q6k,
            mm_q4k_small,
            mm_q6k_small,
            mm_q4k_2tok,
            mm_q6k_2tok,
            mm_abt_reg,
            wdq: RefCell::new(None),
            px_buf: RefCell::new(None),
            py_buf: RefCell::new(None),
            x_buf,
            y_buf,
            wcache: RefCell::new(HashMap::new()),
        }
    }

    /// Batched q4_k prefill projection: `out`[ntok, n_rows] = `x`[ntok, in_dim] @ W^T,
    /// row-major (out[t*n_rows + row]). Dequants the whole q4_k weight to an f32
    /// scratch buffer ONCE (amortized over all `ntok` tokens), then runs the
    /// register-tiled GEMM. Replaces `ntok` sequential `matvec_into` calls (each of
    /// which re-reads+re-dequants the weight) with a single weight read — the
    /// prefill speedup. `w` MUST be a resident q4_k GpuMatrix.
    pub fn matmul_q4k_prefill_into(&self, w: &GpuMatrix, x: &[f32], ntok: usize, out: &mut [f32]) {
        assert_eq!(w.ggml_type, GGML_Q4_K, "matmul_q4k_prefill_into: weight not q4_k");
        self.matmul_prefill_into(w, x, ntok, out, &self.dequant_q4k);
    }

    /// Small-batch q4_k matmul `Y[ntok,n_rows] = X[ntok,in_dim] @ Wᵀ` (ntok ≤ 8)
    /// via the register-tiled kernel: the quantized weight is read ONCE and
    /// multiplied into all `ntok` token vectors in registers (no f32 scratch). This
    /// is the primitive that makes the 2-token MTP verify read weights once — the
    /// f32-scratch prefill path only pays off past ~16 tokens.
    pub fn matmul_q4k_small_into(&self, w: &GpuMatrix, x: &[f32], ntok: usize, out: &mut [f32]) {
        assert_eq!(w.ggml_type, GGML_Q4_K, "matmul_q4k_small_into: weight not q4_k");
        assert!((1..=8).contains(&ntok), "matmul_q4k_small_into: ntok must be 1..=8");
        assert_eq!(x.len(), ntok * w.in_dim, "x shape");
        assert_eq!(out.len(), ntok * w.n_rows, "out shape");
        // Persistent grow-on-demand scratch (no per-call Metal buffer alloc — that
        // churn was ~1ms/call, i.e. the whole cost, across the verify's ~250 calls).
        let ensure = |slot: &RefCell<Option<metal::Buffer>>, floats: usize| -> metal::Buffer {
            let need = (floats * 4) as u64;
            let mut s = slot.borrow_mut();
            if !s.as_ref().is_some_and(|b| b.length() >= need) {
                *s = Some(
                    self.device
                        .new_buffer(need, MTLResourceOptions::StorageModeShared),
                );
            }
            s.as_ref().unwrap().clone()
        };
        let xbuf = ensure(&self.px_buf, x.len());
        let ybuf = ensure(&self.py_buf, out.len());
        unsafe {
            std::ptr::copy_nonoverlapping(x.as_ptr(), xbuf.contents() as *mut f32, x.len());
        }
        let (in32, nr32, nt32) = (w.in_dim as u32, w.n_rows as u32, ntok as u32);
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.mm_q4k_small);
        enc.set_buffer(0, Some(&w.buf), 0);
        enc.set_buffer(1, Some(&xbuf), 0);
        enc.set_buffer(2, Some(&ybuf), 0);
        enc.set_bytes(3, 4, &in32 as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &nr32 as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &nt32 as *const u32 as *const c_void);
        enc.dispatch_threads(
            MTLSize::new((w.n_rows * 32) as u64, 1, 1),
            MTLSize::new(32, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let ptr = ybuf.contents() as *const f32;
        out.copy_from_slice(unsafe { std::slice::from_raw_parts(ptr, out.len()) });
    }

    /// Efficient 2-token q4_k matmul (MTP verify): coalesced NDST-rows-per-simdgroup
    /// like the single-token decode kernel, so it runs at ~decode speed but produces
    /// BOTH tokens from one weight read.
    pub fn matmul_q4k_2tok_into(&self, w: &GpuMatrix, x: &[f32], out: &mut [f32]) {
        assert_eq!(w.ggml_type, GGML_Q4_K, "matmul_q4k_2tok_into: weight not q4_k");
        assert_eq!(x.len(), 2 * w.in_dim, "x shape");
        assert_eq!(out.len(), 2 * w.n_rows, "out shape");
        let ensure = |slot: &RefCell<Option<metal::Buffer>>, floats: usize| -> metal::Buffer {
            let need = (floats * 4) as u64;
            let mut s = slot.borrow_mut();
            if !s.as_ref().is_some_and(|b| b.length() >= need) {
                *s = Some(
                    self.device
                        .new_buffer(need, MTLResourceOptions::StorageModeShared),
                );
            }
            s.as_ref().unwrap().clone()
        };
        let xbuf = ensure(&self.px_buf, x.len());
        let ybuf = ensure(&self.py_buf, out.len());
        unsafe {
            std::ptr::copy_nonoverlapping(x.as_ptr(), xbuf.contents() as *mut f32, x.len());
        }
        let (in32, nr32) = (w.in_dim as u32, w.n_rows as u32);
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.mm_q4k_2tok);
        enc.set_buffer(0, Some(&w.buf), 0);
        enc.set_buffer(1, Some(&xbuf), 0);
        enc.set_buffer(2, Some(&ybuf), 0);
        enc.set_bytes(3, 4, &in32 as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &nr32 as *const u32 as *const c_void);
        let simdgroups = (w.n_rows as u64).div_ceil(2);
        enc.dispatch_threads(MTLSize::new(simdgroups * 32, 1, 1), MTLSize::new(32, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let ptr = ybuf.contents() as *const f32;
        out.copy_from_slice(unsafe { std::slice::from_raw_parts(ptr, out.len()) });
    }

    /// Efficient 2-token q6_k matmul (MTP verify): coalesced, weight read once.
    pub fn matmul_q6k_2tok_into(&self, w: &GpuMatrix, x: &[f32], out: &mut [f32]) {
        assert_eq!(w.ggml_type, GGML_Q6_K, "matmul_q6k_2tok_into: weight not q6_k");
        assert_eq!(x.len(), 2 * w.in_dim, "x shape");
        assert_eq!(out.len(), 2 * w.n_rows, "out shape");
        let ensure = |slot: &RefCell<Option<metal::Buffer>>, floats: usize| -> metal::Buffer {
            let need = (floats * 4) as u64;
            let mut s = slot.borrow_mut();
            if !s.as_ref().is_some_and(|b| b.length() >= need) {
                *s = Some(
                    self.device
                        .new_buffer(need, MTLResourceOptions::StorageModeShared),
                );
            }
            s.as_ref().unwrap().clone()
        };
        let xbuf = ensure(&self.px_buf, x.len());
        let ybuf = ensure(&self.py_buf, out.len());
        unsafe {
            std::ptr::copy_nonoverlapping(x.as_ptr(), xbuf.contents() as *mut f32, x.len());
        }
        let (in32, nr32) = (w.in_dim as u32, w.n_rows as u32);
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.mm_q6k_2tok);
        enc.set_buffer(0, Some(&w.buf), 0);
        enc.set_buffer(1, Some(&xbuf), 0);
        enc.set_buffer(2, Some(&ybuf), 0);
        enc.set_bytes(3, 4, &in32 as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &nr32 as *const u32 as *const c_void);
        let simdgroups = (w.n_rows as u64).div_ceil(2);
        enc.dispatch_threads(MTLSize::new(simdgroups * 32, 1, 1), MTLSize::new(32, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let ptr = ybuf.contents() as *const f32;
        out.copy_from_slice(unsafe { std::slice::from_raw_parts(ptr, out.len()) });
    }

    /// q6_k twin of `matmul_q4k_small_into` (one thread per output row, weight read
    /// once for all `ntok` tokens). Lets the MTP verify batch the q6_k weights
    /// (ffn_down, attn_v) instead of reading them twice per 2-token step.
    pub fn matmul_q6k_small_into(&self, w: &GpuMatrix, x: &[f32], ntok: usize, out: &mut [f32]) {
        assert_eq!(w.ggml_type, GGML_Q6_K, "matmul_q6k_small_into: weight not q6_k");
        assert!((1..=8).contains(&ntok), "matmul_q6k_small_into: ntok must be 1..=8");
        assert_eq!(x.len(), ntok * w.in_dim, "x shape");
        assert_eq!(out.len(), ntok * w.n_rows, "out shape");
        let ensure = |slot: &RefCell<Option<metal::Buffer>>, floats: usize| -> metal::Buffer {
            let need = (floats * 4) as u64;
            let mut s = slot.borrow_mut();
            if !s.as_ref().is_some_and(|b| b.length() >= need) {
                *s = Some(
                    self.device
                        .new_buffer(need, MTLResourceOptions::StorageModeShared),
                );
            }
            s.as_ref().unwrap().clone()
        };
        let xbuf = ensure(&self.px_buf, x.len());
        let ybuf = ensure(&self.py_buf, out.len());
        unsafe {
            std::ptr::copy_nonoverlapping(x.as_ptr(), xbuf.contents() as *mut f32, x.len());
        }
        let (in32, nr32, nt32) = (w.in_dim as u32, w.n_rows as u32, ntok as u32);
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.mm_q6k_small);
        enc.set_buffer(0, Some(&w.buf), 0);
        enc.set_buffer(1, Some(&xbuf), 0);
        enc.set_buffer(2, Some(&ybuf), 0);
        enc.set_bytes(3, 4, &in32 as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &nr32 as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &nt32 as *const u32 as *const c_void);
        enc.dispatch_threads(
            MTLSize::new(w.n_rows as u64, 1, 1),
            MTLSize::new(64, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let ptr = ybuf.contents() as *const f32;
        out.copy_from_slice(unsafe { std::slice::from_raw_parts(ptr, out.len()) });
    }

    /// Batched prefill for a q6_k weight — mirror of the q4_k path with the q6_k
    /// dequant kernel. Lets qwen35 Q4_K_M batch its q6_k weights (ffn_down/attn_v)
    /// instead of falling back to per-token. Additive: q4_k path unchanged.
    pub fn matmul_q6k_prefill_into(&self, w: &GpuMatrix, x: &[f32], ntok: usize, out: &mut [f32]) {
        assert_eq!(w.ggml_type, GGML_Q6_K, "matmul_q6k_prefill_into: weight not q6_k");
        self.matmul_prefill_into(w, x, ntok, out, &self.dequant_q6k);
    }

    /// Shared batched prefill: dequant the whole weight to f32 scratch ONCE via
    /// `dequant`, then a register-tiled GEMM over all `ntok` tokens (weight read
    /// once, not once-per-token). The GEMM is quant-agnostic — only the dequant
    /// kernel differs per type.
    fn matmul_prefill_into(
        &self,
        w: &GpuMatrix,
        x: &[f32],
        ntok: usize,
        out: &mut [f32],
        dequant: &ComputePipelineState,
    ) {
        assert_eq!(x.len(), ntok * w.in_dim, "x shape");
        assert_eq!(out.len(), ntok * w.n_rows, "out shape");
        let (in_dim, n_rows) = (w.in_dim, w.n_rows);

        // Persistent, grow-on-demand scratch (avoids a fresh alloc/copy per proj).
        let ensure = |slot: &RefCell<Option<metal::Buffer>>, floats: usize| -> metal::Buffer {
            let need = (floats * 4) as u64;
            let mut s = slot.borrow_mut();
            if !s.as_ref().is_some_and(|b| b.length() >= need) {
                *s = Some(
                    self.device
                        .new_buffer(need, MTLResourceOptions::StorageModeShared),
                );
            }
            s.as_ref().unwrap().clone()
        };
        let xbuf = ensure(&self.px_buf, x.len());
        let ybuf = ensure(&self.py_buf, out.len());
        let wdq = ensure(&self.wdq, n_rows * in_dim);
        unsafe {
            std::ptr::copy_nonoverlapping(x.as_ptr(), xbuf.contents() as *mut f32, x.len());
        }
        let (in32, nr32, m32) = (in_dim as u32, n_rows as u32, ntok as u32);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        // Dequant the whole weight to f32 scratch ONCE (amortized over all tokens),
        // then a register-tiled GEMM: C[M,K]=A[M,Nc]@B[K,Nc]^T, A=x, B=wdq
        // (M=ntok, Nc=in_dim, K=n_rows). This beat the shared-memory prefill kernel
        // on M4 Max for Gemma4's dims. Metal hazard tracking serializes the two.
        enc.set_compute_pipeline_state(dequant);
        enc.set_buffer(0, Some(&w.buf), 0);
        enc.set_buffer(1, Some(&wdq), 0);
        enc.set_bytes(2, 4, &in32 as *const u32 as *const c_void);
        enc.set_bytes(3, 4, &nr32 as *const u32 as *const c_void);
        enc.dispatch_threads(
            MTLSize::new((n_rows * in_dim) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.set_compute_pipeline_state(&self.mm_abt_reg);
        enc.set_buffer(0, Some(&xbuf), 0);
        enc.set_buffer(1, Some(&wdq), 0);
        enc.set_buffer(2, Some(&ybuf), 0);
        enc.set_bytes(3, 4, &m32 as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &in32 as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &nr32 as *const u32 as *const c_void);
        let gx = (n_rows as u64).div_ceil(64);
        let gy = (ntok as u64).div_ceil(64);
        enc.dispatch_thread_groups(MTLSize::new(gx, gy, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let ptr = ybuf.contents() as *const f32;
        out.copy_from_slice(unsafe { std::slice::from_raw_parts(ptr, out.len()) });
    }

    /// A shared (CPU/GPU unified) Metal buffer holding `d`.
    fn mk_buf(&self, d: &[f32]) -> metal::Buffer {
        self.device.new_buffer_with_data(
            d.as_ptr() as *const c_void,
            (d.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Buffer for `b`, cached under `key` (frozen weight) or freshly uploaded.
    fn buf_keyed(&self, b: &[f32], key: Option<u64>) -> metal::Buffer {
        match key {
            Some(k) => {
                let hit = self.wcache.borrow().get(&k).cloned();
                hit.unwrap_or_else(|| {
                    let buf = self.mk_buf(b);
                    self.wcache.borrow_mut().insert(k, buf.clone());
                    buf
                })
            }
            None => self.mk_buf(b),
        }
    }

    /// C[m,k] = A[m,n] @ B[k,n]^T — matmul backward dA, reusing the cached forward
    /// weight buffer (`b_key` = the same id as the forward matmul). No transpose.
    pub fn matmul_abt_keyed(
        &self,
        a: &[f32],
        m: usize,
        n: usize,
        b: &[f32],
        k: usize,
        b_key: Option<u64>,
    ) -> Vec<f32> {
        let ab = self.mk_buf(a);
        let bb = self.buf_keyed(b, b_key);
        self.run_tiled(&self.matmul_abt_tiled, &ab, &bb, m, n, k, m, k)
    }

    /// C[k,n] = A[m,k]^T @ B[m,n] — matmul/bmm backward dB. `a_key` caches A when
    /// it's frozen (uncommon for dB, but symmetric).
    pub fn matmul_atb_keyed(
        &self,
        a: &[f32],
        k: usize,
        m: usize,
        b: &[f32],
        n: usize,
        a_key: Option<u64>,
    ) -> Vec<f32> {
        let ab = self.buf_keyed(a, a_key);
        let bb = self.mk_buf(b);
        self.run_tiled(&self.matmul_atb_tiled, &ab, &bb, k, m, n, k, n)
    }

    /// general f32 matmul: C[m,n] = A[m,k] @ B[k,n]
    pub fn matmul_f32(&self, a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
        self.matmul_f32_keyed(a, m, k, b, n, None)
    }

    /// Tiled GEMM dispatch (2D threadgroups). `d3,d4,d5` are the kernel's three
    /// uint params; output is `[out_rows, out_cols]`. Bit-identical to the naive
    /// kernels (same summation order), faster via shared-memory reuse.
    fn run_tiled(
        &self,
        pipe: &ComputePipelineState,
        ab: &metal::Buffer,
        bb: &metal::Buffer,
        d3: usize,
        d4: usize,
        d5: usize,
        out_rows: usize,
        out_cols: usize,
    ) -> Vec<f32> {
        let cb = self.device.new_buffer(
            (out_rows * out_cols * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let (a3, a4, a5) = (d3 as u32, d4 as u32, d5 as u32);
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(ab), 0);
        enc.set_buffer(1, Some(bb), 0);
        enc.set_buffer(2, Some(&cb), 0);
        enc.set_bytes(3, 4, &a3 as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &a4 as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &a5 as *const u32 as *const c_void);
        let tile = 16u64;
        let groups = MTLSize::new(
            (out_cols as u64 + tile - 1) / tile,
            (out_rows as u64 + tile - 1) / tile,
            1,
        );
        enc.dispatch_thread_groups(groups, MTLSize::new(tile, tile, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        unsafe { std::slice::from_raw_parts(cb.contents() as *const f32, out_rows * out_cols) }
            .to_vec()
    }

    /// Drop a cached weight buffer (call if a cached tensor's data is mutated).
    pub fn evict(&self, key: u64) {
        self.wcache.borrow_mut().remove(&key);
    }

    /// Empty the whole keyed-buffer cache. The cache assumes keyed operands are
    /// PERSISTENT (frozen weights) — true for inference, but a training loop mints
    /// a fresh tensor id every step, so transient non-grad activations (e.g. the
    /// stem's im2col matrix, whose input is a constant) would accumulate forever.
    /// Call this once per training step; nothing persistent is cached during
    /// training (all real weights require grad and are never keyed).
    pub fn clear_cache(&self) {
        self.wcache.borrow_mut().clear();
    }

    /// Like `matmul_f32`, but if `b_key` is set the B buffer is cached under that
    /// key and reused on later calls — for FROZEN weights, which are identical
    /// every step. The caller must only pass a key for tensors whose data does not
    /// change (a base constant); A is always uploaded fresh.
    pub fn matmul_f32_keyed(
        &self,
        a: &[f32],
        m: usize,
        k: usize,
        b: &[f32],
        n: usize,
        b_key: Option<u64>,
    ) -> Vec<f32> {
        let ab = self.mk_buf(a);
        let bb = self.buf_keyed(b, b_key);
        self.run_tiled(&self.matmul_f32_tiled, &ab, &bb, m, k, n, m, n)
    }

    /// Raw NDHWC depthwise Conv3D for grouped/depthwise 3D feature convolutions.
    /// Weight layout is `[Cin, Kd*Kh*Kw]`; output channels equal input channels.
    #[allow(clippy::too_many_arguments)]
    pub fn depthwise_conv3d_forward(
        &self,
        x: &[f32],
        w: &[f32],
        shape: [usize; 5],
        k: [usize; 3],
        stride: [usize; 3],
        pad: [usize; 3],
        w_key: u64,
    ) -> Vec<f32> {
        let started = std::time::Instant::now();
        let [n, d, h, wi, c] = shape;
        let od = (d + 2 * pad[0] - k[0]) / stride[0] + 1;
        let oh = (h + 2 * pad[1] - k[1]) / stride[1] + 1;
        let ow = (wi + 2 * pad[2] - k[2]) / stride[2] + 1;
        let total_out = n * od * oh * ow * c;
        let total_x = x.len();
        let total_w = w.len();
        assert_eq!(total_x, n * d * h * wi * c);
        assert_eq!(total_w, c * k[0] * k[1] * k[2]);
        let xb = self.mk_buf(x);
        let wb = self.buf_keyed(w, Some(w_key));
        let outb = self.device.new_buffer(
            (total_out * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let p: [u32; 20] = [
            n, d, h, wi, c, k[0], k[1], k[2], od, oh, ow, stride[0], stride[1], stride[2], pad[0],
            pad[1], pad[2], total_out, total_x, total_w,
        ]
        .map(|v| v as u32);
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.depthwise_conv3d);
        enc.set_buffer(0, Some(&xb), 0);
        enc.set_buffer(1, Some(&wb), 0);
        enc.set_buffer(2, Some(&outb), 0);
        enc.set_bytes(
            3,
            std::mem::size_of_val(&p) as u64,
            p.as_ptr() as *const c_void,
        );
        let tg = self
            .depthwise_conv3d
            .max_total_threads_per_threadgroup()
            .min(256);
        enc.dispatch_threads(MTLSize::new(total_out as u64, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        if std::env::var_os("HOS_CONV_PROFILE").is_some() {
            eprintln!(
                "[dw3d-fwd ] {:6.2}ms {:6} rows k={:?} c={}",
                started.elapsed().as_secs_f64() * 1e3,
                n * od * oh * ow,
                k,
                c
            );
        }
        unsafe { std::slice::from_raw_parts(outb.contents() as *const f32, total_out) }.to_vec()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn depthwise_conv3d_backward(
        &self,
        x: &[f32],
        w: &[f32],
        gout: &[f32],
        shape: [usize; 5],
        k: [usize; 3],
        stride: [usize; 3],
        pad: [usize; 3],
        w_key: u64,
        need_dx: bool,
        need_dw: bool,
    ) -> (Vec<f32>, Vec<f32>) {
        let started = std::time::Instant::now();
        let [n, d, h, wi, c] = shape;
        let od = (d + 2 * pad[0] - k[0]) / stride[0] + 1;
        let oh = (h + 2 * pad[1] - k[1]) / stride[1] + 1;
        let ow = (wi + 2 * pad[2] - k[2]) / stride[2] + 1;
        let total_out = n * od * oh * ow * c;
        let total_x = x.len();
        let total_w = w.len();
        assert_eq!(gout.len(), total_out);
        let xb = self.mk_buf(x);
        let wb = self.buf_keyed(w, Some(w_key));
        let gb = self.mk_buf(gout);
        let dxb = need_dx.then(|| {
            self.device
                .new_buffer((total_x * 4) as u64, MTLResourceOptions::StorageModeShared)
        });
        let dwb = need_dw.then(|| {
            self.device
                .new_buffer((total_w * 4) as u64, MTLResourceOptions::StorageModeShared)
        });
        let p: [u32; 20] = [
            n, d, h, wi, c, k[0], k[1], k[2], od, oh, ow, stride[0], stride[1], stride[2], pad[0],
            pad[1], pad[2], total_out, total_x, total_w,
        ]
        .map(|v| v as u32);
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        if let Some(dx) = &dxb {
            enc.set_compute_pipeline_state(&self.depthwise_conv3d_dx);
            enc.set_buffer(0, Some(&wb), 0);
            enc.set_buffer(1, Some(&gb), 0);
            enc.set_buffer(2, Some(dx), 0);
            enc.set_bytes(
                3,
                std::mem::size_of_val(&p) as u64,
                p.as_ptr() as *const c_void,
            );
            let tg = self
                .depthwise_conv3d_dx
                .max_total_threads_per_threadgroup()
                .min(256);
            enc.dispatch_threads(MTLSize::new(total_x as u64, 1, 1), MTLSize::new(tg, 1, 1));
        }
        if let Some(dw) = &dwb {
            enc.set_compute_pipeline_state(&self.depthwise_conv3d_dw);
            enc.set_buffer(0, Some(&xb), 0);
            enc.set_buffer(1, Some(&gb), 0);
            enc.set_buffer(2, Some(dw), 0);
            enc.set_bytes(
                3,
                std::mem::size_of_val(&p) as u64,
                p.as_ptr() as *const c_void,
            );
            let tg = self
                .depthwise_conv3d_dw
                .max_total_threads_per_threadgroup()
                .min(256);
            enc.dispatch_threads(MTLSize::new(total_w as u64, 1, 1), MTLSize::new(tg, 1, 1));
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let dx = dxb
            .as_ref()
            .map(|b| {
                unsafe { std::slice::from_raw_parts(b.contents() as *const f32, total_x) }.to_vec()
            })
            .unwrap_or_default();
        let dw = dwb
            .as_ref()
            .map(|b| {
                unsafe { std::slice::from_raw_parts(b.contents() as *const f32, total_w) }.to_vec()
            })
            .unwrap_or_default();
        if std::env::var_os("HOS_CONV_PROFILE").is_some() {
            eprintln!(
                "[dw3d-bwd ] {:6.2}ms {:6} rows k={:?} c={} dx={} dw={}",
                started.elapsed().as_secs_f64() * 1e3,
                n * od * oh * ow,
                k,
                c,
                need_dx,
                need_dw
            );
        }
        (dx, dw)
    }

    /// Direct frozen Conv3D + inference-BN affine + ReLU for NDHWC tensors.
    /// Avoids materializing the enormous CPU im2col matrix used by trainable
    /// convolutions. `w` is [Kd*Kh*Kw*Cin,Cout] and is cached by `w_key`.
    #[allow(clippy::too_many_arguments)]
    pub fn conv3d_bn_relu_infer(
        &self,
        x: &[f32],
        w: &[f32],
        scale: &[f32],
        shift: &[f32],
        shape: [usize; 5],
        k: [usize; 3],
        stride: [usize; 3],
        pad: [usize; 3],
        cout: usize,
        w_key: u64,
    ) -> Vec<f32> {
        let [n, d, h, wi, cin] = shape;
        let od = (d + 2 * pad[0] - k[0]) / stride[0] + 1;
        let oh = (h + 2 * pad[1] - k[1]) / stride[1] + 1;
        let ow = (wi + 2 * pad[2] - k[2]) / stride[2] + 1;
        let total = n * od * oh * ow * cout;
        assert_eq!(x.len(), n * d * h * wi * cin);
        assert_eq!(w.len(), k[0] * k[1] * k[2] * cin * cout);
        assert_eq!(scale.len(), cout);
        assert_eq!(shift.len(), cout);
        let xb = self.mk_buf(x);
        let wb = self.buf_keyed(w, Some(w_key));
        let sb = self.mk_buf(scale);
        let bb = self.mk_buf(shift);
        let outb = self
            .device
            .new_buffer((total * 4) as u64, MTLResourceOptions::StorageModeShared);
        let p: [u32; 19] = [
            n, d, h, wi, cin, k[0], k[1], k[2], cout, od, oh, ow, stride[0], stride[1], stride[2],
            pad[0], pad[1], pad[2], total,
        ]
        .map(|v| v as u32);
        let rows = n * od * oh * ow;
        let patch = k[0] * k[1] * k[2] * cin;
        // Measured on M4 Max: register-GEMM wins for the 7x7 stem and spatial
        // kernels with enough rows/channels to amortize unfold + transpose.
        let conv_mode = std::env::var("HOS_METAL_CONV3D").unwrap_or_else(|_| "auto".into());
        let force_gemm = std::env::var_os("HOS_CONV3D_GEMM").is_some() || conv_mode == "gemm";
        let force_direct = conv_mode == "direct";
        let gemm = !force_direct
            && k != [1, 1, 1]
            && (force_gemm || k[0] >= 7 || (rows >= 6_272 && cin >= 64));
        let started = std::time::Instant::now();
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        // Keep these alive through command completion when the GEMM path owns
        // its large transient buffers.
        let mut cols_buf = None;
        let mut raw_buf = None;
        let mut wt_buf = None;
        if gemm {
            let cols = self.device.new_buffer(
                (rows * patch * 4) as u64,
                MTLResourceOptions::StorageModePrivate,
            );
            let raw = self
                .device
                .new_buffer((total * 4) as u64, MTLResourceOptions::StorageModePrivate);
            let wt = self.device.new_buffer(
                (patch * cout * 4) as u64,
                MTLResourceOptions::StorageModePrivate,
            );

            enc.set_compute_pipeline_state(&self.unfold3d_gpu);
            enc.set_buffer(0, Some(&xb), 0);
            enc.set_buffer(1, Some(&cols), 0);
            enc.set_bytes(
                2,
                std::mem::size_of_val(&p) as u64,
                p.as_ptr() as *const c_void,
            );
            let total_cols = (rows * patch) as u32;
            enc.set_bytes(3, 4, &total_cols as *const u32 as *const c_void);
            let tg = self
                .unfold3d_gpu
                .max_total_threads_per_threadgroup()
                .min(256);
            enc.dispatch_threads(
                MTLSize::new(total_cols as u64, 1, 1),
                MTLSize::new(tg, 1, 1),
            );

            enc.set_compute_pipeline_state(&self.transpose_f32);
            enc.set_buffer(0, Some(&wb), 0);
            enc.set_buffer(1, Some(&wt), 0);
            let (pr, pc) = (patch as u32, cout as u32);
            enc.set_bytes(2, 4, &pr as *const u32 as *const c_void);
            enc.set_bytes(3, 4, &pc as *const u32 as *const c_void);
            let tg = self
                .transpose_f32
                .max_total_threads_per_threadgroup()
                .min(256);
            enc.dispatch_threads(
                MTLSize::new((patch * cout) as u64, 1, 1),
                MTLSize::new(tg, 1, 1),
            );

            enc.set_compute_pipeline_state(&self.mm_abt_reg);
            enc.set_buffer(0, Some(&cols), 0);
            enc.set_buffer(1, Some(&wt), 0);
            enc.set_buffer(2, Some(&raw), 0);
            let (pm, pk, pn) = (rows as u32, patch as u32, cout as u32);
            enc.set_bytes(3, 4, &pm as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &pk as *const u32 as *const c_void);
            enc.set_bytes(5, 4, &pn as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new((cout as u64).div_ceil(64), (rows as u64).div_ceil(64), 1),
                MTLSize::new(256, 1, 1),
            );

            enc.set_compute_pipeline_state(&self.conv3d_affine_relu);
            enc.set_buffer(0, Some(&raw), 0);
            enc.set_buffer(1, Some(&sb), 0);
            enc.set_buffer(2, Some(&bb), 0);
            enc.set_buffer(3, Some(&outb), 0);
            enc.set_bytes(
                4,
                std::mem::size_of_val(&p) as u64,
                p.as_ptr() as *const c_void,
            );
            let tg = self
                .conv3d_affine_relu
                .max_total_threads_per_threadgroup()
                .min(256);
            enc.dispatch_threads(MTLSize::new(total as u64, 1, 1), MTLSize::new(tg, 1, 1));
            cols_buf = Some(cols);
            raw_buf = Some(raw);
            wt_buf = Some(wt);
        } else {
            enc.set_compute_pipeline_state(&self.conv3d_bn_relu);
            enc.set_buffer(0, Some(&xb), 0);
            enc.set_buffer(1, Some(&wb), 0);
            enc.set_buffer(2, Some(&sb), 0);
            enc.set_buffer(3, Some(&bb), 0);
            enc.set_buffer(4, Some(&outb), 0);
            enc.set_bytes(
                5,
                std::mem::size_of_val(&p) as u64,
                p.as_ptr() as *const c_void,
            );
            let tg = self
                .conv3d_bn_relu
                .max_total_threads_per_threadgroup()
                .min(256);
            enc.dispatch_threads(MTLSize::new(total as u64, 1, 1), MTLSize::new(tg, 1, 1));
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        drop(cols_buf);
        drop(raw_buf);
        drop(wt_buf);
        if std::env::var_os("HOS_CONV_PROFILE").is_some() {
            eprintln!(
                "[conv3d-fwd] {:6.2}ms {:6} rows k={:?} {:4}->{:4} {}",
                started.elapsed().as_secs_f64() * 1e3,
                rows,
                k,
                cin,
                cout,
                if gemm { "gemm" } else { "direct" }
            );
        }
        unsafe { std::slice::from_raw_parts(outb.contents() as *const f32, total) }.to_vec()
    }

    /// Direct backward for `conv3d_bn_relu_infer`. Returns `(d_input,d_weight)`;
    /// either vector is empty when its corresponding `need_*` flag is false.
    #[allow(clippy::too_many_arguments)]
    pub fn conv3d_bn_relu_backward(
        &self,
        x: &[f32],
        w: &[f32],
        scale: &[f32],
        out: &[f32],
        gout: &[f32],
        shape: [usize; 5],
        k: [usize; 3],
        stride: [usize; 3],
        pad: [usize; 3],
        cout: usize,
        w_key: u64,
        need_dx: bool,
        need_dw: bool,
    ) -> (Vec<f32>, Vec<f32>) {
        let started = std::time::Instant::now();
        let [n, d, h, wi, cin] = shape;
        let od = (d + 2 * pad[0] - k[0]) / stride[0] + 1;
        let oh = (h + 2 * pad[1] - k[1]) / stride[1] + 1;
        let ow = (wi + 2 * pad[2] - k[2]) / stride[2] + 1;
        let total_out = n * od * oh * ow * cout;
        let total_x = x.len();
        let total_w = w.len();
        assert_eq!(out.len(), total_out);
        assert_eq!(gout.len(), total_out);
        let xb = self.mk_buf(x);
        let wb = self.buf_keyed(w, Some(w_key));
        let sb = self.mk_buf(scale);
        let ob = self.mk_buf(out);
        let gb = self.mk_buf(gout);
        let dxb = need_dx.then(|| {
            self.device
                .new_buffer((total_x * 4) as u64, MTLResourceOptions::StorageModeShared)
        });
        let dwb = need_dw.then(|| {
            self.device
                .new_buffer((total_w * 4) as u64, MTLResourceOptions::StorageModeShared)
        });
        let rows = n * od * oh * ow;
        let patch = k[0] * k[1] * k[2] * cin;
        // GEMM + deterministic fold wins for all measured spatial 3D-conv layers.
        // A 1x1x1 convolution stays direct to avoid pointless patch scratch.
        let conv_mode = std::env::var("HOS_METAL_CONV3D").unwrap_or_else(|_| "auto".into());
        let force_direct = conv_mode == "direct";
        let gemm_dx = need_dx && k != [1, 1, 1] && !force_direct;
        let cols = need_dw.then(|| {
            self.device.new_buffer(
                (rows * patch * 4) as u64,
                MTLResourceOptions::StorageModePrivate,
            )
        });
        let grad_raw = (need_dw || gemm_dx).then(|| {
            self.device.new_buffer(
                (total_out * 4) as u64,
                MTLResourceOptions::StorageModePrivate,
            )
        });
        let dcols = gemm_dx.then(|| {
            self.device.new_buffer(
                (rows * patch * 4) as u64,
                MTLResourceOptions::StorageModePrivate,
            )
        });
        let p: [u32; 21] = [
            n, d, h, wi, cin, k[0], k[1], k[2], cout, od, oh, ow, stride[0], stride[1], stride[2],
            pad[0], pad[1], pad[2], total_out, total_x, total_w,
        ]
        .map(|v| v as u32);
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        if let Some(grad_raw) = &grad_raw {
            enc.set_compute_pipeline_state(&self.conv3d_grad_raw);
            enc.set_buffer(0, Some(&sb), 0);
            enc.set_buffer(1, Some(&ob), 0);
            enc.set_buffer(2, Some(&gb), 0);
            enc.set_buffer(3, Some(grad_raw), 0);
            enc.set_bytes(
                4,
                std::mem::size_of_val(&p) as u64,
                p.as_ptr() as *const c_void,
            );
            let tg = self
                .conv3d_grad_raw
                .max_total_threads_per_threadgroup()
                .min(256);
            enc.dispatch_threads(MTLSize::new(total_out as u64, 1, 1), MTLSize::new(tg, 1, 1));
        }
        if let Some(dx) = dxb.as_ref().filter(|_| !gemm_dx) {
            enc.set_compute_pipeline_state(&self.conv3d_bn_relu_dx);
            enc.set_buffer(0, Some(&wb), 0);
            enc.set_buffer(1, Some(&sb), 0);
            enc.set_buffer(2, Some(&ob), 0);
            enc.set_buffer(3, Some(&gb), 0);
            enc.set_buffer(4, Some(dx), 0);
            enc.set_bytes(
                5,
                std::mem::size_of_val(&p) as u64,
                p.as_ptr() as *const c_void,
            );
            let tg = self
                .conv3d_bn_relu_dx
                .max_total_threads_per_threadgroup()
                .min(256);
            enc.dispatch_threads(MTLSize::new(total_x as u64, 1, 1), MTLSize::new(tg, 1, 1));
        }
        if let (Some(dw), Some(cols), Some(grad_raw)) = (&dwb, &cols, &grad_raw) {
            enc.set_compute_pipeline_state(&self.unfold3d_gpu);
            enc.set_buffer(0, Some(&xb), 0);
            enc.set_buffer(1, Some(cols), 0);
            enc.set_bytes(
                2,
                std::mem::size_of_val(&p) as u64,
                p.as_ptr() as *const c_void,
            );
            let total_cols = (rows * patch) as u32;
            enc.set_bytes(3, 4, &total_cols as *const u32 as *const c_void);
            let tg = self
                .unfold3d_gpu
                .max_total_threads_per_threadgroup()
                .min(256);
            enc.dispatch_threads(
                MTLSize::new(total_cols as u64, 1, 1),
                MTLSize::new(tg, 1, 1),
            );

            enc.set_compute_pipeline_state(&self.matmul_atb_tiled);
            enc.set_buffer(0, Some(cols), 0);
            enc.set_buffer(1, Some(grad_raw), 0);
            enc.set_buffer(2, Some(dw), 0);
            let (pk, pm, pn) = (patch as u32, rows as u32, cout as u32);
            enc.set_bytes(3, 4, &pk as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &pm as *const u32 as *const c_void);
            enc.set_bytes(5, 4, &pn as *const u32 as *const c_void);
            let tile = 16u64;
            enc.dispatch_thread_groups(
                MTLSize::new(
                    (cout as u64).div_ceil(tile),
                    (patch as u64).div_ceil(tile),
                    1,
                ),
                MTLSize::new(tile, tile, 1),
            );
        }
        if let (Some(dx), Some(dcols), Some(grad_raw)) = (&dxb, &dcols, &grad_raw) {
            // dcols[rows,patch] = grad_raw[rows,cout] @ w[patch,cout]^T.
            enc.set_compute_pipeline_state(&self.matmul_abt_tiled);
            enc.set_buffer(0, Some(grad_raw), 0);
            enc.set_buffer(1, Some(&wb), 0);
            enc.set_buffer(2, Some(dcols), 0);
            let (pm, pnc, pk) = (rows as u32, cout as u32, patch as u32);
            enc.set_bytes(3, 4, &pm as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &pnc as *const u32 as *const c_void);
            enc.set_bytes(5, 4, &pk as *const u32 as *const c_void);
            let tile = 16u64;
            enc.dispatch_thread_groups(
                MTLSize::new(
                    (patch as u64).div_ceil(tile),
                    (rows as u64).div_ceil(tile),
                    1,
                ),
                MTLSize::new(tile, tile, 1),
            );

            enc.set_compute_pipeline_state(&self.fold3d_dx);
            enc.set_buffer(0, Some(dcols), 0);
            enc.set_buffer(1, Some(dx), 0);
            enc.set_bytes(
                2,
                std::mem::size_of_val(&p) as u64,
                p.as_ptr() as *const c_void,
            );
            let tg = self.fold3d_dx.max_total_threads_per_threadgroup().min(256);
            enc.dispatch_threads(MTLSize::new(total_x as u64, 1, 1), MTLSize::new(tg, 1, 1));
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let dx = dxb
            .as_ref()
            .map(|b| {
                unsafe { std::slice::from_raw_parts(b.contents() as *const f32, total_x) }.to_vec()
            })
            .unwrap_or_default();
        let dw = dwb
            .as_ref()
            .map(|b| {
                unsafe { std::slice::from_raw_parts(b.contents() as *const f32, total_w) }.to_vec()
            })
            .unwrap_or_default();
        if std::env::var_os("HOS_CONV_PROFILE").is_some() {
            eprintln!(
                "[conv3d-bwd] {:6.2}ms {:6} rows k={:?} {:4}->{:4} dx={} dw={} {}",
                started.elapsed().as_secs_f64() * 1e3,
                rows,
                k,
                cin,
                cout,
                need_dx,
                need_dw,
                if gemm_dx { "gemm-dx" } else { "direct-dx" }
            );
        }
        (dx, dw)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }
    /// A freshly compiled library of all fused kernels (for building runners).
    pub fn fused_library(&self) -> metal::Library {
        self.device
            .new_library_with_source(FUSED_SRC, &CompileOptions::new())
            .expect("compile fused MSL")
    }

    /// Run one gated-delta-net step on the GPU for a single head. `q` must be
    /// pre-scaled by 1/sqrt(n); `g` = exp(decay). Returns (output, updated state).
    pub fn deltanet_step(
        &self,
        s: &[f32],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        g: f32,
        beta: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let n = q.len();
        let mk = |d: &[f32]| {
            self.device.new_buffer_with_data(
                d.as_ptr() as *const c_void,
                (d.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        let sb = mk(s);
        let qb = mk(q);
        let kb = mk(k);
        let vb = mk(v);
        let ob = self
            .device
            .new_buffer((n * 4) as u64, MTLResourceOptions::StorageModeShared);
        let n32 = n as u32;
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.deltanet);
        enc.set_buffer(0, Some(&sb), 0);
        enc.set_buffer(1, Some(&qb), 0);
        enc.set_buffer(2, Some(&kb), 0);
        enc.set_buffer(3, Some(&vb), 0);
        enc.set_buffer(4, Some(&ob), 0);
        enc.set_bytes(5, 4, &g as *const f32 as *const c_void);
        enc.set_bytes(6, 4, &beta as *const f32 as *const c_void);
        enc.set_bytes(7, 4, &n32 as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(n as u64, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let o = unsafe { std::slice::from_raw_parts(ob.contents() as *const f32, n) }.to_vec();
        let ns = unsafe { std::slice::from_raw_parts(sb.contents() as *const f32, n * n) }.to_vec();
        (o, ns)
    }

    pub fn upload_matrix(&self, w: &[f32], n_rows: usize, in_dim: usize) -> GpuMatrix {
        assert_eq!(w.len(), n_rows * in_dim);
        // store weights as f16 to halve memory bandwidth (decode is bandwidth-bound)
        let h: Vec<f16> = w.iter().map(|&v| f16::from_f32(v)).collect();
        let buf = self.device.new_buffer_with_data(
            h.as_ptr() as *const c_void,
            (h.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        GpuMatrix {
            buf,
            n_rows,
            in_dim,
            ggml_type: GGML_F16,
        }
    }

    /// Upload raw (still-quantized) tensor bytes to the GPU as-is.
    pub fn upload_quant(
        &self,
        bytes: &[u8],
        ggml_type: u32,
        n_rows: usize,
        in_dim: usize,
    ) -> GpuMatrix {
        let buf = self.device.new_buffer_with_data(
            bytes.as_ptr() as *const c_void,
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        GpuMatrix {
            buf,
            n_rows,
            in_dim,
            ggml_type,
        }
    }

    /// y = W · x, computed on the GPU, writing into `out` (reuses scratch buffers).
    pub fn matvec_into(&self, w: &GpuMatrix, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), w.in_dim);
        assert_eq!(out.len(), w.n_rows);
        assert!(
            x.len() <= X_CAP && w.n_rows <= Y_CAP,
            "matvec exceeds scratch capacity"
        );

        // copy x into the persistent input buffer
        unsafe {
            std::ptr::copy_nonoverlapping(x.as_ptr(), self.x_buf.contents() as *mut f32, x.len());
        }
        let in_dim = w.in_dim as u32;

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        if w.ggml_type == GGML_F16 {
            enc.set_compute_pipeline_state(&self.matvec);
            enc.set_buffer(0, Some(&w.buf), 0);
            enc.set_buffer(1, Some(&self.x_buf), 0);
            enc.set_buffer(2, Some(&self.y_buf), 0);
            enc.set_bytes(3, 4, &in_dim as *const u32 as *const c_void);
            let tg = self.matvec.max_total_threads_per_threadgroup().min(256);
            enc.dispatch_threads(MTLSize::new(w.n_rows as u64, 1, 1), MTLSize::new(tg, 1, 1));
        } else {
            // Natively-quantized weight: dispatch the coalesced K-quant kernel
            // (32-lane simdgroup per row, NDST rows per group — same tiling as the
            // fused GpuRunner's `enc_matvec`).
            let p = self.quant_mv.get(&w.ggml_type).unwrap_or_else(|| {
                panic!("[hos] no GPU matvec kernel for ggml type {}", w.ggml_type)
            });
            let n_rows = w.n_rows as u32;
            enc.set_compute_pipeline_state(p);
            enc.set_buffer(0, Some(&w.buf), 0);
            enc.set_buffer(1, Some(&self.x_buf), 0);
            enc.set_buffer(2, Some(&self.y_buf), 0);
            enc.set_bytes(3, 4, &in_dim as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &n_rows as *const u32 as *const c_void);
            const NDST: u64 = 2;
            let simdgroups = (w.n_rows as u64).div_ceil(NDST);
            enc.dispatch_threads(MTLSize::new(simdgroups * 32, 1, 1), MTLSize::new(32, 1, 1));
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let ptr = self.y_buf.contents() as *const f32;
        out.copy_from_slice(unsafe { std::slice::from_raw_parts(ptr, w.n_rows) });
    }

    /// Convenience wrapper returning a fresh Vec (used by the benchmark).
    pub fn matvec(&self, w: &GpuMatrix, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; w.n_rows];
        self.matvec_into(w, x, &mut out);
        out
    }

    /// Diagnostic/parity helper: run ONE coalesced quantized matvec (y = W·x) for a
    /// resident GpuMatrix using the SAME FUSED_SRC kernels the runner dispatches
    /// (matvec_hq4_co / matvec_q4_0_co), returning y[n_rows]. Compiles the kernel on
    /// demand — not for the hot path; it exists so a test can pin the in-kernel
    /// dequant against the CPU `decode_*_into` reference. `in_dim`/`n_rows` come from
    /// the matrix; the NDST=2 simdgroup tiling matches `enc_matvec`.
    pub fn matvec_co_for_test(&self, w: &GpuMatrix, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), w.in_dim, "x width must equal in_dim");
        let kname = match w.ggml_type {
            crate::model::HQ4_TYPE => "matvec_hq4_co",
            GGML_Q4_0 => "matvec_q4_0_co",
            t => panic!("matvec_co_for_test: no coalesced kernel for ggml type {t}"),
        };
        let lib = self
            .device
            .new_library_with_source(FUSED_SRC, &CompileOptions::new())
            .expect("compile fused MSL");
        let p = pipeline(&self.device, &lib, kname);
        let xbuf = self.device.new_buffer_with_data(
            x.as_ptr() as *const c_void,
            (x.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let ybuf = self
            .device
            .new_buffer((w.n_rows * 4) as u64, MTLResourceOptions::StorageModeShared);
        let in_dim = w.in_dim as u32;
        let n_rows = w.n_rows as u32;
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&p);
        enc.set_buffer(0, Some(&w.buf), 0);
        enc.set_buffer(1, Some(&xbuf), 0);
        enc.set_buffer(2, Some(&ybuf), 0);
        enc.set_bytes(3, 4, &in_dim as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &n_rows as *const u32 as *const c_void);
        const NDST: u64 = 2;
        let simdgroups = (w.n_rows as u64).div_ceil(NDST);
        enc.dispatch_threads(MTLSize::new(simdgroups * 32, 1, 1), MTLSize::new(32, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        download_buf(&ybuf, w.n_rows)
    }
}

// ============================================================================
// GpuRunner: the fully-resident forward pass. Activations, KV cache, and norms
// all live on the GPU; one command buffer + one sync per token.
// ============================================================================

pub struct GpuRunner {
    #[allow(dead_code)] // held to keep the Metal device alive for the runner's lifetime
    device: Device,
    queue: CommandQueue,
    p_mv_f32: ComputePipelineState,
    p_mv_f16: ComputePipelineState,
    p_mv_q8: ComputePipelineState,
    p_mv_q4_0: ComputePipelineState,
    p_mv_hq4: ComputePipelineState,
    p_mv_q5_0: ComputePipelineState,
    p_mv_q4k: ComputePipelineState,
    p_mv_q5k: ComputePipelineState,
    p_mv_q6k: ComputePipelineState,
    p_rmsnorm: ComputePipelineState,
    p_rope: ComputePipelineState,
    p_store_kv: ComputePipelineState,
    p_attention: ComputePipelineState,
    p_swiglu: ComputePipelineState,
    p_add: ComputePipelineState,
    // batched (prefill) pipelines
    p_mm_q4k_batch: ComputePipelineState,
    p_mm_q4k_prefill: ComputePipelineState,
    p_dequant_q4k: ComputePipelineState,
    p_mm_abt_tiled: ComputePipelineState,
    p_mm_abt_reg: ComputePipelineState,
    p_rmsnorm_batch: ComputePipelineState,
    p_rope_batch: ComputePipelineState,
    p_store_kv_batch: ComputePipelineState,
    p_attention_batch: ComputePipelineState,
    p_attention_batch_sg: ComputePipelineState,

    // activation buffers
    x: Buffer,
    xb: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    att: Buffer,
    gate: Buffer,
    up: Buffer,
    logits: Buffer,

    // batched (prefill) activation buffers: up to MAX_BATCH tokens at once
    x_batch: Buffer,
    xb_batch: Buffer,
    q_batch: Buffer,
    k_batch: Buffer,
    v_batch: Buffer,
    att_batch: Buffer,
    gate_batch: Buffer,
    up_batch: Buffer,
    wdq: Buffer, // f32 scratch: one prefill weight dequantized for the tiled GEMM
    max_batch: usize,

    // norms (CPU-uploaded)
    attn_norm: Vec<Buffer>,
    ffn_norm: Vec<Buffer>,
    output_norm: Buffer,

    // kv cache per layer
    kcache: Vec<Buffer>,
    vcache: Vec<Buffer>,

    // optional per-layer attention biases (Qwen2-family)
    bq: Vec<Option<Buffer>>,
    bk: Vec<Option<Buffer>>,
    bv: Vec<Option<Buffer>>,

    pub max_seq: usize,
}

fn pipeline(device: &Device, lib: &metal::Library, name: &str) -> ComputePipelineState {
    let f = lib
        .get_function(name, None)
        .unwrap_or_else(|e| panic!("fn {name}: {e}"));
    device
        .new_compute_pipeline_state_with_function(&f)
        .unwrap_or_else(|e| panic!("pipeline {name}: {e}"))
}

impl GpuRunner {
    pub fn new(gpu: Gpu, model: &Model) -> GpuRunner {
        let device = gpu.device.clone();
        let queue = gpu.queue.clone();
        let lib = device
            .new_library_with_source(FUSED_SRC, &CompileOptions::new())
            .expect("compile fused MSL");

        let cfg = &model.cfg;
        let max_seq = cfg.ctx_len.min(8192);
        let max_batch = 1024usize; // prompts up to this batch prefill in one pass
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let q_dim = cfg.n_heads * cfg.head_dim;

        let nb =
            |n: usize| device.new_buffer((n * 4) as u64, MTLResourceOptions::StorageModeShared);
        let upload = |data: &[f32]| {
            device.new_buffer_with_data(
                data.as_ptr() as *const c_void,
                (data.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };

        let attn_norm = model.layers.iter().map(|l| upload(&l.attn_norm)).collect();
        let ffn_norm = model.layers.iter().map(|l| upload(&l.ffn_norm)).collect();
        let upload_opt = |o: &Option<Vec<f32>>| o.as_ref().map(|d| upload(d));
        let bq = model.layers.iter().map(|l| upload_opt(&l.bq)).collect();
        let bk = model.layers.iter().map(|l| upload_opt(&l.bk)).collect();
        let bv = model.layers.iter().map(|l| upload_opt(&l.bv)).collect();
        let kcache = (0..cfg.n_layers).map(|_| nb(max_seq * kv_dim)).collect();
        let vcache = (0..cfg.n_layers).map(|_| nb(max_seq * kv_dim)).collect();

        eprintln!("[hos] GPU runner ready (resident activations, max_seq={max_seq})");

        GpuRunner {
            p_mv_f32: pipeline(&device, &lib, "matvec_f32"),
            p_mv_f16: pipeline(&device, &lib, "matvec_f16"),
            p_mv_q8: pipeline(&device, &lib, "matvec_q8_0_co"),
            p_mv_q4_0: pipeline(&device, &lib, "matvec_q4_0_co"),
            p_mv_hq4: pipeline(&device, &lib, "matvec_hq4_co"),
            p_mv_q5_0: pipeline(&device, &lib, "matvec_q5_0"),
            p_mv_q4k: pipeline(&device, &lib, "matvec_q4k_co"),
            p_mv_q5k: pipeline(&device, &lib, "matvec_q5k_co"),
            p_mv_q6k: pipeline(&device, &lib, "matvec_q6k_co"),
            p_rmsnorm: pipeline(&device, &lib, "rmsnorm"),
            p_rope: pipeline(&device, &lib, "rope"),
            p_store_kv: pipeline(&device, &lib, "store_kv"),
            p_attention: pipeline(&device, &lib, "attention"),
            p_swiglu: pipeline(&device, &lib, "swiglu"),
            p_add: pipeline(&device, &lib, "add_inplace"),
            p_mm_q4k_batch: pipeline(&device, &lib, "matmul_q4k_batch"),
            p_mm_q4k_prefill: pipeline(&device, &lib, "matmul_q4k_prefill"),
            p_dequant_q4k: pipeline(&device, &lib, "dequant_q4k_to_f32"),
            p_mm_abt_tiled: pipeline(&device, &lib, "matmul_abt_tiled"),
            p_mm_abt_reg: pipeline(&device, &lib, "matmul_abt_reg"),
            p_rmsnorm_batch: pipeline(&device, &lib, "rmsnorm_batch"),
            p_rope_batch: pipeline(&device, &lib, "rope_batch"),
            p_store_kv_batch: pipeline(&device, &lib, "store_kv_batch"),
            p_attention_batch: pipeline(&device, &lib, "attention_batch"),
            p_attention_batch_sg: pipeline(&device, &lib, "attention_batch_sg"),
            x: nb(cfg.dim),
            xb: nb(cfg.dim),
            q: nb(q_dim),
            k: nb(kv_dim),
            v: nb(kv_dim),
            att: nb(q_dim),
            gate: nb(cfg.ffn_dim),
            up: nb(cfg.ffn_dim),
            logits: nb(cfg.vocab_size),
            // +8 padding rows: the register-tiled matmul reads up to TT-1 tokens
            // past `ntok`; padding keeps those over-reads in-bounds (values unused).
            x_batch: nb((max_batch + 8) * cfg.dim),
            xb_batch: nb((max_batch + 8) * cfg.dim),
            q_batch: nb((max_batch + 8) * q_dim),
            k_batch: nb((max_batch + 8) * kv_dim),
            v_batch: nb((max_batch + 8) * kv_dim),
            att_batch: nb((max_batch + 8) * q_dim),
            gate_batch: nb((max_batch + 8) * cfg.ffn_dim),
            up_batch: nb((max_batch + 8) * cfg.ffn_dim),
            // scratch for one dequantized prefill weight: the largest linear is
            // ffn (gate/up/down), dim*ffn_dim either way.
            wdq: nb((cfg.ffn_dim * cfg.dim)
                .max(q_dim * cfg.dim)
                .max(kv_dim * cfg.dim)),
            max_batch,
            attn_norm,
            ffn_norm,
            output_norm: upload(&model.output_norm),
            kcache,
            vcache,
            bq,
            bk,
            bv,
            max_seq,
            device,
            queue,
        }
    }

    /// Ordering barrier between dependent dispatches sharing one encoder.
    fn barrier(enc: &ComputeCommandEncoderRef, outs: &[&BufferRef]) {
        let res: Vec<&ResourceRef> = outs
            .iter()
            .map(|&b| {
                let r: &ResourceRef = b;
                r
            })
            .collect();
        enc.memory_barrier_with_resources(&res);
    }

    fn enc_matvec(
        &self,
        enc: &ComputeCommandEncoderRef,
        w: &GpuMatrix,
        x: &BufferRef,
        y: &BufferRef,
    ) {
        let in_dim = w.in_dim as u32;
        // coalesced kernels use a 32-lane simdgroup per row; scalar kernels use 1 thread/row
        let (p, coalesced) = match w.ggml_type {
            GGML_F32 => (&self.p_mv_f32, false),
            GGML_F16 => (&self.p_mv_f16, false),
            GGML_Q5_0 => (&self.p_mv_q5_0, false),
            GGML_Q4_0 => (&self.p_mv_q4_0, true),
            crate::model::HQ4_TYPE => (&self.p_mv_hq4, true),
            GGML_Q8_0 => (&self.p_mv_q8, true),
            GGML_Q4_K => (&self.p_mv_q4k, true),
            GGML_Q5_K => (&self.p_mv_q5k, true),
            GGML_Q6_K => (&self.p_mv_q6k, true),
            t => {
                // Unreachable: `model::gpu_quant_supported` preflights every linear weight
                // type and routes HQ4/unsupported quants to the CPU path before a GPU runner
                // is built. A library must never kill the process; panic if the guard was bypassed.
                panic!(
                    "[hos] no GPU matvec kernel for ggml type {t} (supported: \
                     F32/F16/Q8_0/Q4_0/Q5_0/Q4_K/Q5_K/Q6_K/HQ4) — gpu_quant_supported \
                     preflight should have selected the CPU path"
                );
            }
        };
        enc.set_compute_pipeline_state(p);
        enc.set_buffer(0, Some(&w.buf), 0);
        enc.set_buffer(1, Some(x), 0);
        enc.set_buffer(2, Some(y), 0);
        enc.set_bytes(3, 4, &in_dim as *const u32 as *const c_void);
        if coalesced {
            // coalesced kernels process NDST rows per 32-lane simdgroup.
            const NDST: u64 = 2;
            let n_rows = w.n_rows as u32;
            enc.set_bytes(4, 4, &n_rows as *const u32 as *const c_void);
            let simdgroups = (w.n_rows as u64).div_ceil(NDST);
            enc.dispatch_threads(MTLSize::new(simdgroups * 32, 1, 1), MTLSize::new(32, 1, 1));
        } else {
            let tg = p.max_total_threads_per_threadgroup().min(256);
            enc.dispatch_threads(MTLSize::new(w.n_rows as u64, 1, 1), MTLSize::new(tg, 1, 1));
        }
        // NB: caller inserts barriers only at true data dependencies, so
        // independent dispatches (q/k/v, gate/up) can overlap on the GPU.
    }

    fn enc_rmsnorm(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &BufferRef,
        w: &BufferRef,
        out: &BufferRef,
        n: usize,
        eps: f32,
    ) {
        let n32 = n as u32;
        enc.set_compute_pipeline_state(&self.p_rmsnorm);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(w), 0);
        enc.set_buffer(2, Some(out), 0);
        enc.set_bytes(3, 4, &n32 as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &eps as *const f32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
    }

    fn enc_rope(
        &self,
        enc: &ComputeCommandEncoderRef,
        buf: &BufferRef,
        hd: usize,
        pos: usize,
        base: f32,
        n_heads: usize,
        neox: bool,
    ) {
        let (hd32, pos32, neox32) = (hd as u32, pos as u32, neox as u32);
        enc.set_compute_pipeline_state(&self.p_rope);
        enc.set_buffer(0, Some(buf), 0);
        enc.set_bytes(1, 4, &hd32 as *const u32 as *const c_void);
        enc.set_bytes(2, 4, &pos32 as *const u32 as *const c_void);
        enc.set_bytes(3, 4, &base as *const f32 as *const c_void);
        enc.set_bytes(4, 4, &neox32 as *const u32 as *const c_void);
        let n = n_heads * hd / 2;
        let tg = self.p_rope.max_total_threads_per_threadgroup().min(256);
        enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(tg, 1, 1));
    }

    fn enc_store_kv(&self, enc: &ComputeCommandEncoderRef, l: usize, kv_dim: usize, pos: usize) {
        let (kv32, pos32) = (kv_dim as u32, pos as u32);
        enc.set_compute_pipeline_state(&self.p_store_kv);
        enc.set_buffer(0, Some(&self.k), 0);
        enc.set_buffer(1, Some(&self.v), 0);
        enc.set_buffer(2, Some(&self.kcache[l]), 0);
        enc.set_buffer(3, Some(&self.vcache[l]), 0);
        enc.set_bytes(4, 4, &kv32 as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &pos32 as *const u32 as *const c_void);
        let tg = self.p_store_kv.max_total_threads_per_threadgroup().min(256);
        enc.dispatch_threads(MTLSize::new(kv_dim as u64, 1, 1), MTLSize::new(tg, 1, 1));
    }

    fn enc_attention(
        &self,
        enc: &ComputeCommandEncoderRef,
        l: usize,
        hd: usize,
        kv_dim: usize,
        kv_mul: usize,
        pos: usize,
        n_heads: usize,
    ) {
        let (hd32, kv32, mul32, pos32) = (hd as u32, kv_dim as u32, kv_mul as u32, pos as u32);
        enc.set_compute_pipeline_state(&self.p_attention);
        enc.set_buffer(0, Some(&self.q), 0);
        enc.set_buffer(1, Some(&self.kcache[l]), 0);
        enc.set_buffer(2, Some(&self.vcache[l]), 0);
        enc.set_buffer(3, Some(&self.att), 0);
        enc.set_bytes(4, 4, &hd32 as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &kv32 as *const u32 as *const c_void);
        enc.set_bytes(6, 4, &mul32 as *const u32 as *const c_void);
        enc.set_bytes(7, 4, &pos32 as *const u32 as *const c_void);
        // one threadgroup per head, `hd` threads (= head dim) cooperating
        enc.dispatch_thread_groups(
            MTLSize::new(n_heads as u64, 1, 1),
            MTLSize::new(hd as u64, 1, 1),
        );
    }

    fn enc_elementwise(
        &self,
        enc: &ComputeCommandEncoderRef,
        p: &ComputePipelineState,
        a: &BufferRef,
        b: &BufferRef,
        n: usize,
    ) {
        enc.set_compute_pipeline_state(p);
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(b), 0);
        let tg = p.max_total_threads_per_threadgroup().min(256);
        enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(tg, 1, 1));
    }

    /// One token through the model on the GPU. With `head`, runs the final norm +
    /// lm-head and returns logits; without it, stops after the last block (KV cache
    /// is still advanced) and returns empty — used during prefill for every prompt
    /// token except the last, skipping the biggest matvec (vocab×dim) T−1 times.
    /// flwr terminal snap: quantize the post-output-norm hidden sitting in the
    /// shared `xb` buffer onto the E8 lattice, in place, using the SAME routine
    /// the CPU forward uses (`forward::flwr_e8_quant`) so the two paths cannot
    /// drift. The caller MUST have already committed+waited the rmsnorm that
    /// filled `xb`, since this reads the GPU result back on the CPU.
    fn snap_terminal_hidden(&self, dim: usize) {
        let hidden = unsafe { std::slice::from_raw_parts_mut(self.xb.contents() as *mut f32, dim) };
        crate::forward::flwr_e8_quant(hidden);
    }

    fn forward_inner(&self, model: &Model, token: u32, pos: usize, head: bool) -> Vec<f32> {
        let cfg = &model.cfg;
        let dim = cfg.dim;
        let hd = cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * hd;
        let q_dim = cfg.n_heads * hd;
        let kv_mul = cfg.n_heads / cfg.n_kv_heads;

        // embedding lookup -> x buffer (CPU copy into shared buffer)
        let row = &model.tok_embd[token as usize * dim..(token as usize + 1) * dim];
        unsafe {
            std::ptr::copy_nonoverlapping(row.as_ptr(), self.x.contents() as *mut f32, dim);
        }

        let t_enc = std::time::Instant::now();
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        for (l, layer) in model.layers.iter().enumerate() {
            self.enc_rmsnorm(enc, &self.x, &self.attn_norm[l], &self.xb, dim, cfg.rms_eps);
            Self::barrier(enc, &[&self.xb]);
            // q/k/v read the same xb and write distinct buffers — no barriers
            // between them, so the three matvecs can overlap on the GPU.
            self.enc_matvec(enc, layer.wq.as_gpu(), &self.xb, &self.q);
            self.enc_matvec(enc, layer.wk.as_gpu(), &self.xb, &self.k);
            self.enc_matvec(enc, layer.wv.as_gpu(), &self.xb, &self.v);
            Self::barrier(enc, &[&self.q, &self.k, &self.v]);
            let mut biased = false;
            if let Some(b) = &self.bq[l] {
                self.enc_elementwise(enc, &self.p_add, &self.q, b, q_dim);
                biased = true;
            }
            if let Some(b) = &self.bk[l] {
                self.enc_elementwise(enc, &self.p_add, &self.k, b, kv_dim);
                biased = true;
            }
            if let Some(b) = &self.bv[l] {
                self.enc_elementwise(enc, &self.p_add, &self.v, b, kv_dim);
                biased = true;
            }
            if biased {
                Self::barrier(enc, &[&self.q, &self.k]);
            }
            self.enc_rope(
                enc,
                &self.q,
                hd,
                pos,
                cfg.rope_base,
                cfg.n_heads,
                cfg.rope_neox,
            );
            self.enc_rope(
                enc,
                &self.k,
                hd,
                pos,
                cfg.rope_base,
                cfg.n_kv_heads,
                cfg.rope_neox,
            );
            Self::barrier(enc, &[&self.q, &self.k]);
            self.enc_store_kv(enc, l, kv_dim, pos);
            Self::barrier(enc, &[&self.kcache[l], &self.vcache[l]]);
            self.enc_attention(enc, l, hd, kv_dim, kv_mul, pos, cfg.n_heads);
            Self::barrier(enc, &[&self.att]);
            self.enc_matvec(enc, layer.wo.as_gpu(), &self.att, &self.xb);
            Self::barrier(enc, &[&self.xb]);
            self.enc_elementwise(enc, &self.p_add, &self.x, &self.xb, dim);
            Self::barrier(enc, &[&self.x]);

            self.enc_rmsnorm(enc, &self.x, &self.ffn_norm[l], &self.xb, dim, cfg.rms_eps);
            Self::barrier(enc, &[&self.xb]);
            // gate/up are independent — overlap them.
            self.enc_matvec(enc, layer.w_gate.as_gpu(), &self.xb, &self.gate);
            self.enc_matvec(enc, layer.w_up.as_gpu(), &self.xb, &self.up);
            Self::barrier(enc, &[&self.gate, &self.up]);
            self.enc_elementwise(enc, &self.p_swiglu, &self.gate, &self.up, cfg.ffn_dim);
            Self::barrier(enc, &[&self.gate]);
            self.enc_matvec(enc, layer.w_down.as_gpu(), &self.gate, &self.xb);
            Self::barrier(enc, &[&self.xb]);
            self.enc_elementwise(enc, &self.p_add, &self.x, &self.xb, dim);
            Self::barrier(enc, &[&self.x]);
        }

        if head {
            self.enc_rmsnorm(enc, &self.x, &self.output_norm, &self.xb, dim, cfg.rms_eps);
            Self::barrier(enc, &[&self.xb]);
            if cfg.arch.needs_terminal_hidden_snap() {
                // flwr: the final post-output-norm hidden is E8-quantized on the
                // CPU before the lm-head. Flush the rmsnorm so its result lands in
                // the shared `xb` buffer, snap it bit-for-bit with the CPU path,
                // then run the lm-head matvec in a fresh command buffer.
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                self.snap_terminal_hidden(dim);
                let cmd2 = self.queue.new_command_buffer();
                let enc2 = cmd2.new_compute_command_encoder();
                self.enc_matvec(enc2, model.output.as_gpu(), &self.xb, &self.logits);
                enc2.end_encoding();
                cmd2.commit();
                cmd2.wait_until_completed();
                let ptr = self.logits.contents() as *const f32;
                return unsafe { std::slice::from_raw_parts(ptr, cfg.vocab_size).to_vec() };
            }
            // Fail loud if a future arch needs a terminal hidden snap but isn't
            // routed through the branch above — otherwise its GPU tokens would
            // silently diverge from the CPU path.
            // `assert!` (not debug_assert) so this fails loud in RELEASE too — a future
            // arch with a terminal snap that isn't wired here would otherwise silently
            // emit divergent tokens on GPU. Never fires for current arches (Flwr returns
            // from the branch above; everything else needs no snap).
            assert!(
                !cfg.arch.needs_terminal_hidden_snap(),
                "GPU head reached the lm-head without the terminal hidden snap required by {:?}",
                cfg.arch
            );
            self.enc_matvec(enc, model.output.as_gpu(), &self.xb, &self.logits);
        }

        enc.end_encoding();
        let prof = std::env::var("HOS_PROF").is_ok();
        if prof {
            eprintln!("[prof] cpu encode {:?}", t_enc.elapsed());
        }
        let t_gpu = std::time::Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        if prof {
            eprintln!("[prof] gpu execute {:?}", t_gpu.elapsed());
        }

        if !head {
            return Vec::new();
        }
        let ptr = self.logits.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, cfg.vocab_size).to_vec() }
    }

    /// One token, returning logits (final norm + lm-head included).
    pub fn forward(&self, model: &Model, token: u32, pos: usize) -> Vec<f32> {
        self.forward_inner(model, token, pos, true)
    }

    /// Ingest one prompt token WITHOUT computing logits — advances the KV cache
    /// only, skipping the vocab-sized lm-head matvec. For prefill.
    pub fn prefill_step(&self, model: &Model, token: u32, pos: usize) {
        self.forward_inner(model, token, pos, false);
    }

    // ---- batched prefill: process the whole prompt in one pass ----

    /// True if this model can use the batched-prefill path: all projection weights
    /// are q4_k (the one batched matmul kernel) and there are no attention biases.
    fn can_batch(&self, model: &Model) -> bool {
        self.bq.iter().all(Option::is_none)
            && self.bk.iter().all(Option::is_none)
            && self.bv.iter().all(Option::is_none)
            && model.layers.iter().all(|l| {
                [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down]
                    .iter()
                    .all(|w| w.as_gpu().ggml_type == GGML_Q4_K)
            })
    }

    /// Batched q4_k projection Y[ntok,n_rows] = X[ntok,in_dim] @ W^T, in tiles of
    /// NTOK_B tokens (the weight row is read once per tile, not once per token).
    fn enc_mm_q4k_batch(
        &self,
        enc: &ComputeCommandEncoderRef,
        w: &GpuMatrix,
        x: &BufferRef,
        y: &BufferRef,
        ntok: usize,
    ) {
        let (in_dim, n_rows, nt) = (w.in_dim as u32, w.n_rows as u32, ntok as u32);
        // Tiny prompts: the old dequant-into-shared kernel wins (the tiled GEMM's
        // fixed dequant-to-scratch cost isn't yet amortized). Crossover ~pp48 on
        // 1B; above it the tiled GEMM scales 2-3x better. HOS_PREFILL=old forces it.
        if ntok < 48 || std::env::var("HOS_PREFILL").as_deref() == Ok("old") {
            if w.in_dim <= 4096 {
                enc.set_compute_pipeline_state(&self.p_mm_q4k_prefill);
                enc.set_buffer(0, Some(&w.buf), 0);
                enc.set_buffer(1, Some(x), 0);
                enc.set_buffer(2, Some(y), 0);
                enc.set_bytes(3, 4, &in_dim as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &n_rows as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &nt as *const u32 as *const c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(w.n_rows as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
                return;
            }
            const NTOK_B: usize = 8;
            let mut t0 = 0usize;
            while t0 < ntok {
                let ntc = (ntok - t0).min(NTOK_B) as u32;
                enc.set_compute_pipeline_state(&self.p_mm_q4k_batch);
                enc.set_buffer(0, Some(&w.buf), 0);
                enc.set_buffer(1, Some(x), (t0 * w.in_dim * 4) as u64);
                enc.set_buffer(2, Some(y), (t0 * w.n_rows * 4) as u64);
                enc.set_bytes(3, 4, &in_dim as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &n_rows as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &ntc as *const u32 as *const c_void);
                enc.dispatch_threads(
                    MTLSize::new((w.n_rows * 32) as u64, 1, 1),
                    MTLSize::new(32, 1, 1),
                );
                t0 += NTOK_B;
            }
            return;
        }
        // 1. dequant the whole weight to f32 scratch ONCE (amortized over all tokens).
        enc.set_compute_pipeline_state(&self.p_dequant_q4k);
        enc.set_buffer(0, Some(&w.buf), 0);
        enc.set_buffer(1, Some(&self.wdq), 0);
        enc.set_bytes(2, 4, &in_dim as *const u32 as *const c_void);
        enc.set_bytes(3, 4, &n_rows as *const u32 as *const c_void);
        enc.dispatch_threads(
            MTLSize::new((w.n_rows * w.in_dim) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        // 2. tiled GEMM: Y[ntok,n_rows] = X[ntok,in_dim] @ wdq[n_rows,in_dim]^T.
        // (A[M,Nc] @ B[K,Nc]^T -> C[M,K]; M=ntok, Nc=in_dim, K=n_rows.)
        // Metal hazard tracking serializes the dequant write before this read.
        let basic = std::env::var("HOS_PREFILL").as_deref() == Ok("tiled");
        enc.set_compute_pipeline_state(if basic {
            &self.p_mm_abt_tiled
        } else {
            &self.p_mm_abt_reg
        });
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(&self.wdq), 0);
        enc.set_buffer(2, Some(y), 0);
        enc.set_bytes(3, 4, &nt as *const u32 as *const c_void); // M
        enc.set_bytes(4, 4, &in_dim as *const u32 as *const c_void); // Nc
        enc.set_bytes(5, 4, &n_rows as *const u32 as *const c_void); // K
        if basic {
            let gx = ((n_rows + 15) / 16) as u64;
            let gy = ((nt + 15) / 16) as u64;
            enc.dispatch_thread_groups(MTLSize::new(gx, gy, 1), MTLSize::new(16, 16, 1));
        } else {
            // register-tiled: 64x64 output block per threadgroup, 256 threads
            let gx = ((n_rows + 63) / 64) as u64; // blocks over K = n_rows
            let gy = ((nt + 63) / 64) as u64; // blocks over M = ntok
            enc.dispatch_thread_groups(MTLSize::new(gx, gy, 1), MTLSize::new(256, 1, 1));
        }
    }

    fn enc_rmsnorm_batch(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &BufferRef,
        w: &BufferRef,
        out: &BufferRef,
        n: usize,
        eps: f32,
        rows: usize,
    ) {
        let n32 = n as u32;
        enc.set_compute_pipeline_state(&self.p_rmsnorm_batch);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(w), 0);
        enc.set_buffer(2, Some(out), 0);
        enc.set_bytes(3, 4, &n32 as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &eps as *const f32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    #[allow(clippy::too_many_arguments)]
    fn enc_rope_batch(
        &self,
        enc: &ComputeCommandEncoderRef,
        buf: &BufferRef,
        hd: usize,
        pos0: usize,
        base: f32,
        n_heads: usize,
        neox: bool,
        rows: usize,
    ) {
        let (hd32, pos32, neox32, nh32) = (hd as u32, pos0 as u32, neox as u32, n_heads as u32);
        enc.set_compute_pipeline_state(&self.p_rope_batch);
        enc.set_buffer(0, Some(buf), 0);
        enc.set_bytes(1, 4, &hd32 as *const u32 as *const c_void);
        enc.set_bytes(2, 4, &pos32 as *const u32 as *const c_void);
        enc.set_bytes(3, 4, &base as *const f32 as *const c_void);
        enc.set_bytes(4, 4, &neox32 as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &nh32 as *const u32 as *const c_void);
        let total = rows * n_heads * (hd / 2);
        let tg = self
            .p_rope_batch
            .max_total_threads_per_threadgroup()
            .min(256);
        enc.dispatch_threads(MTLSize::new(total as u64, 1, 1), MTLSize::new(tg, 1, 1));
    }

    fn enc_store_kv_batch(
        &self,
        enc: &ComputeCommandEncoderRef,
        l: usize,
        kv_dim: usize,
        pos0: usize,
        rows: usize,
    ) {
        let (kv32, pos32) = (kv_dim as u32, pos0 as u32);
        enc.set_compute_pipeline_state(&self.p_store_kv_batch);
        enc.set_buffer(0, Some(&self.k_batch), 0);
        enc.set_buffer(1, Some(&self.v_batch), 0);
        enc.set_buffer(2, Some(&self.kcache[l]), 0);
        enc.set_buffer(3, Some(&self.vcache[l]), 0);
        enc.set_bytes(4, 4, &kv32 as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &pos32 as *const u32 as *const c_void);
        let total = rows * kv_dim;
        let tg = self
            .p_store_kv_batch
            .max_total_threads_per_threadgroup()
            .min(256);
        enc.dispatch_threads(MTLSize::new(total as u64, 1, 1), MTLSize::new(tg, 1, 1));
    }

    /// All-tokens attention in ONE dispatch (prefill): N*n_heads threadgroups.
    #[allow(clippy::too_many_arguments)]
    fn enc_attention_batch(
        &self,
        enc: &ComputeCommandEncoderRef,
        l: usize,
        hd: usize,
        kv_dim: usize,
        kv_mul: usize,
        pos0: usize,
        n_heads: usize,
        ntok: usize,
    ) {
        let (hd32, kv32, mul32, pos32, nh32) = (
            hd as u32,
            kv_dim as u32,
            kv_mul as u32,
            pos0 as u32,
            n_heads as u32,
        );
        // barrier-free one-simdgroup-per-head kernel when hd is a multiple of 32
        let (pipe, threads) = if hd % 32 == 0 {
            (&self.p_attention_batch_sg, 32u64)
        } else {
            (&self.p_attention_batch, hd as u64)
        };
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&self.q_batch), 0);
        enc.set_buffer(1, Some(&self.kcache[l]), 0);
        enc.set_buffer(2, Some(&self.vcache[l]), 0);
        enc.set_buffer(3, Some(&self.att_batch), 0);
        enc.set_bytes(4, 4, &hd32 as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &kv32 as *const u32 as *const c_void);
        enc.set_bytes(6, 4, &mul32 as *const u32 as *const c_void);
        enc.set_bytes(7, 4, &pos32 as *const u32 as *const c_void);
        enc.set_bytes(8, 4, &nh32 as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new((ntok * n_heads) as u64, 1, 1),
            MTLSize::new(threads, 1, 1),
        );
    }

    #[allow(clippy::too_many_arguments, dead_code)]
    fn enc_attention_off(
        &self,
        enc: &ComputeCommandEncoderRef,
        l: usize,
        hd: usize,
        kv_dim: usize,
        kv_mul: usize,
        pos: usize,
        n_heads: usize,
        q_buf: &BufferRef,
        q_off: u64,
        att_buf: &BufferRef,
        att_off: u64,
    ) {
        let (hd32, kv32, mul32, pos32) = (hd as u32, kv_dim as u32, kv_mul as u32, pos as u32);
        enc.set_compute_pipeline_state(&self.p_attention);
        enc.set_buffer(0, Some(q_buf), q_off);
        enc.set_buffer(1, Some(&self.kcache[l]), 0);
        enc.set_buffer(2, Some(&self.vcache[l]), 0);
        enc.set_buffer(3, Some(att_buf), att_off);
        enc.set_bytes(4, 4, &hd32 as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &kv32 as *const u32 as *const c_void);
        enc.set_bytes(6, 4, &mul32 as *const u32 as *const c_void);
        enc.set_bytes(7, 4, &pos32 as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(n_heads as u64, 1, 1),
            MTLSize::new(hd as u64, 1, 1),
        );
    }

    /// Batched prefill of `tokens` starting at `pos0`. Fills the KV cache for all
    /// positions and returns logits for the LAST token (to start generation).
    /// `None` if the model isn't eligible (caller falls back to token-by-token).
    pub fn forward_prefill_gpu(
        &self,
        model: &Model,
        tokens: &[u32],
        pos0: usize,
    ) -> Option<Vec<f32>> {
        let cfg = &model.cfg;
        let n = tokens.len();
        if n == 0 || n > self.max_batch || !self.can_batch(model) {
            if std::env::var("HOS_PROF").is_ok() {
                eprintln!(
                    "[prof] batched prefill INELIGIBLE (n={n}, can_batch={})",
                    self.can_batch(model)
                );
            }
            return None;
        }
        let t_pf = std::time::Instant::now();
        let dim = cfg.dim;
        let hd = cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * hd;
        let kv_mul = cfg.n_heads / cfg.n_kv_heads;

        // prompt embeddings -> x_batch (one row per token)
        let xb = self.x_batch.contents() as *mut f32;
        for (i, &tok) in tokens.iter().enumerate() {
            let row = &model.tok_embd[tok as usize * dim..(tok as usize + 1) * dim];
            unsafe { std::ptr::copy_nonoverlapping(row.as_ptr(), xb.add(i * dim), dim) };
        }

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        for (l, layer) in model.layers.iter().enumerate() {
            self.enc_rmsnorm_batch(
                enc,
                &self.x_batch,
                &self.attn_norm[l],
                &self.xb_batch,
                dim,
                cfg.rms_eps,
                n,
            );
            Self::barrier(enc, &[&self.xb_batch]);
            self.enc_mm_q4k_batch(enc, layer.wq.as_gpu(), &self.xb_batch, &self.q_batch, n);
            self.enc_mm_q4k_batch(enc, layer.wk.as_gpu(), &self.xb_batch, &self.k_batch, n);
            self.enc_mm_q4k_batch(enc, layer.wv.as_gpu(), &self.xb_batch, &self.v_batch, n);
            Self::barrier(enc, &[&self.q_batch, &self.k_batch, &self.v_batch]);
            self.enc_rope_batch(
                enc,
                &self.q_batch,
                hd,
                pos0,
                cfg.rope_base,
                cfg.n_heads,
                cfg.rope_neox,
                n,
            );
            self.enc_rope_batch(
                enc,
                &self.k_batch,
                hd,
                pos0,
                cfg.rope_base,
                cfg.n_kv_heads,
                cfg.rope_neox,
                n,
            );
            Self::barrier(enc, &[&self.q_batch, &self.k_batch]);
            self.enc_store_kv_batch(enc, l, kv_dim, pos0, n);
            Self::barrier(enc, &[&self.kcache[l], &self.vcache[l]]);
            self.enc_attention_batch(enc, l, hd, kv_dim, kv_mul, pos0, cfg.n_heads, n);
            Self::barrier(enc, &[&self.att_batch]);
            self.enc_mm_q4k_batch(enc, layer.wo.as_gpu(), &self.att_batch, &self.xb_batch, n);
            Self::barrier(enc, &[&self.xb_batch]);
            self.enc_elementwise(enc, &self.p_add, &self.x_batch, &self.xb_batch, n * dim);
            Self::barrier(enc, &[&self.x_batch]);

            self.enc_rmsnorm_batch(
                enc,
                &self.x_batch,
                &self.ffn_norm[l],
                &self.xb_batch,
                dim,
                cfg.rms_eps,
                n,
            );
            Self::barrier(enc, &[&self.xb_batch]);
            self.enc_mm_q4k_batch(
                enc,
                layer.w_gate.as_gpu(),
                &self.xb_batch,
                &self.gate_batch,
                n,
            );
            self.enc_mm_q4k_batch(enc, layer.w_up.as_gpu(), &self.xb_batch, &self.up_batch, n);
            Self::barrier(enc, &[&self.gate_batch, &self.up_batch]);
            self.enc_elementwise(
                enc,
                &self.p_swiglu,
                &self.gate_batch,
                &self.up_batch,
                n * cfg.ffn_dim,
            );
            Self::barrier(enc, &[&self.gate_batch]);
            self.enc_mm_q4k_batch(
                enc,
                layer.w_down.as_gpu(),
                &self.gate_batch,
                &self.xb_batch,
                n,
            );
            Self::barrier(enc, &[&self.xb_batch]);
            self.enc_elementwise(enc, &self.p_add, &self.x_batch, &self.xb_batch, n * dim);
            Self::barrier(enc, &[&self.x_batch]);
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // logits for the LAST token: its hidden -> final norm -> lm-head
        unsafe {
            let src = (self.x_batch.contents() as *const f32).add((n - 1) * dim);
            std::ptr::copy_nonoverlapping(src, self.x.contents() as *mut f32, dim);
        }
        let cmd2 = self.queue.new_command_buffer();
        let enc2 = cmd2.new_compute_command_encoder();
        self.enc_rmsnorm(enc2, &self.x, &self.output_norm, &self.xb, dim, cfg.rms_eps);
        Self::barrier(enc2, &[&self.xb]);
        if cfg.arch.needs_terminal_hidden_snap() {
            // flwr: E8-snap the post-output-norm hidden on the CPU before the
            // lm-head. Flush the rmsnorm so `xb` is populated, snap it bit-for-bit
            // with the CPU path, then run the lm-head matvec in a fresh buffer.
            enc2.end_encoding();
            cmd2.commit();
            cmd2.wait_until_completed();
            self.snap_terminal_hidden(dim);
            let cmd3 = self.queue.new_command_buffer();
            let enc3 = cmd3.new_compute_command_encoder();
            self.enc_matvec(enc3, model.output.as_gpu(), &self.xb, &self.logits);
            enc3.end_encoding();
            cmd3.commit();
            cmd3.wait_until_completed();
        } else {
            // Fail loud if a future arch needs the terminal snap but isn't routed
            // through the branch above — its prefill logits would silently diverge.
            // assert! (not debug_assert) — fail loud in release; see the decode-head note.
            assert!(
                !cfg.arch.needs_terminal_hidden_snap(),
                "batched-prefill head skipped the terminal hidden snap required by {:?}",
                cfg.arch
            );
            self.enc_matvec(enc2, model.output.as_gpu(), &self.xb, &self.logits);
            enc2.end_encoding();
            cmd2.commit();
            cmd2.wait_until_completed();
        }
        if std::env::var("HOS_PROF").is_ok() {
            eprintln!("[prof] batched prefill {n} tok in {:?}", t_pf.elapsed());
        }
        let lp = self.logits.contents() as *const f32;
        Some(unsafe { std::slice::from_raw_parts(lp, cfg.vocab_size) }.to_vec())
    }
}

/// Drain an Objective-C autorelease pool around `f`. Metal's `new_command_buffer`
/// (and other calls) return autoreleased objects; in a long training loop with no
/// pool draining they accumulate until the OS OOM-kills the process. Wrap each
/// training batch / eval chunk in this so transient Metal objects are freed.
pub fn autorelease<R>(f: impl FnOnce() -> R) -> R {
    objc::rc::autoreleasepool(f)
}
