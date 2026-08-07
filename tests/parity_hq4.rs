//! HQ4 GPU-kernel parity (feature `wf/hq4-gpu`).
//!
//! `matvec_hq4_co` (metal_be.rs) is a fork of `matvec_q4_0_co`: same coalesced
//! 32-lane / NDST=2 skeleton, but the dequant is the non-uniform NF4 quantile LUT
//! (value = NF4[nibble]*scale) and the nibble packing is hq4's (byte k -> weights
//! 2k, 2k+1) rather than q4_0's (byte j -> weights j, j+16). This test pins the GPU
//! kernel against the CPU `decode_hq4_into` reference: both apply the SAME f16
//! scale and SAME NF4 levels, so the only legitimate difference is the f32
//! summation reorder a simd_sum performs — bounded tightly here. A wrong LUT,
//! wrong packing, or a dropped lane shifts the result far past the bound.
//!
//! macOS-only (the Metal backend); a no-op elsewhere.

#![cfg(target_os = "macos")]

use hos::hos_quant::{decode_hq4, encode_hq4};
use hos::metal_be::Gpu;
use hos::model::HQ4_TYPE;

// deterministic xorshift in [-1, 1] — no rng dep, reproducible across runs.
fn rng_stream(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// CPU reference: decode the hq4 row bytes (exactly as the engine's quant_matvec
/// does) and dot each row with x in high-precision f64.
fn cpu_ref(bytes: &[u8], n_rows: usize, in_dim: usize, x: &[f32]) -> Vec<f32> {
    let dec = decode_hq4(bytes, n_rows * in_dim);
    (0..n_rows)
        .map(|r| {
            let row = &dec[r * in_dim..(r + 1) * in_dim];
            row.iter()
                .zip(x)
                .map(|(&w, &xi)| w as f64 * xi as f64)
                .sum::<f64>() as f32
        })
        .collect()
}

#[test]
fn matvec_hq4_co_matches_cpu_decode() {
    let gpu = Gpu::new();
    let mut rng = rng_stream(0x51A4_7E2D_1122_3344);

    // shapes: in_dim multiple of 64 (hq4 block), various n_rows incl. odd tails
    // (the NDST=2 row tiling must guard the last simdgroup) and odd block counts
    // (e.g. in_dim=192 -> nb=3).
    for &(n_rows, in_dim) in &[
        (1usize, 64usize),
        (3, 64),
        (4, 128),
        (5, 192), // nb=3 (odd block count) + odd row count
        (7, 128),
        (16, 256),
        (33, 512),
        (64, 2048),
    ] {
        let w: Vec<f32> = (0..n_rows * in_dim).map(|_| rng()).collect();
        let x: Vec<f32> = (0..in_dim).map(|_| rng()).collect();
        let bytes = encode_hq4(&w);

        let gm = gpu.upload_quant(&bytes, HQ4_TYPE, n_rows, in_dim);
        let got = gpu.matvec_co_for_test(&gm, &x);
        let want = cpu_ref(&bytes, n_rows, in_dim, &x);

        assert_eq!(got.len(), n_rows);
        for r in 0..n_rows {
            // only the f32 reorder of a simd_sum separates GPU from the f64 ref;
            // it grows ~sqrt(in_dim)*eps*|sum|, far below this bound. A real kernel
            // bug (wrong nibble/LUT/lane) is O(value) and trips it.
            let tol = 2e-3 + 1e-3 * want[r].abs();
            let d = (got[r] - want[r]).abs();
            assert!(
                d <= tol,
                "hq4 matvec mismatch [shape {n_rows}x{in_dim}] row {r}: \
                 got={} want={} diff={d} tol={tol}",
                got[r],
                want[r]
            );
        }
    }
}
