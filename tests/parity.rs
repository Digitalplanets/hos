//! Numeric parity net (parity rule, part 1): guards the hot CPU dot kernel.
//!
//! `dot_f32` has a hand-written aarch64 NEON path and (incoming) an x86 AVX2+FMA
//! path; both reorder the f32 summation vs a naive scalar loop. This test pins
//! every such path to a high-precision f64 reference within a tight bound, so a
//! SIMD reorder is allowed but a real bug (mis-handled tail, wrong stride,
//! dropped lane) is caught. Runs against whatever arch the test is built for.
//!
//! flwr is the floating-point canary at the model level (its terminal E8 snap
//! turns sub-ULP drift into a visible token flip); the golden token-stream test
//! that covers that lives separately and needs a model fixture.

use hos::model::dot_f32;

fn ref_dot_f64(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| *x as f64 * *y as f64)
        .sum::<f64>() as f32
}

#[test]
fn dot_f32_matches_high_precision_reference() {
    // deterministic xorshift in [-1, 1] — no rng dep, reproducible across arches.
    let mut s: u64 = 0x1234_5678_9abc_def1;
    let mut rng = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    // include non-multiple-of-8 lengths to exercise SIMD remainder/tail handling.
    for &len in &[
        1usize, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 127, 128, 1000, 2048, 4096,
    ] {
        for _ in 0..32 {
            let a: Vec<f32> = (0..len).map(|_| rng()).collect();
            let b: Vec<f32> = (0..len).map(|_| rng()).collect();
            let got = dot_f32(&a, &b);
            let want = ref_dot_f64(&a, &b);
            // reorder noise grows ~sqrt(len)*eps; a real kernel bug is O(element) >> this.
            let tol = 1e-3 + 1e-4 * want.abs();
            assert!(
                (got - want).abs() <= tol,
                "dot_f32 mismatch at len={len}: got={got} want={want} diff={} tol={tol}",
                (got - want).abs()
            );
        }
    }
}

#[test]
fn dot_f32_edge_cases() {
    assert_eq!(dot_f32(&[], &[]), 0.0);
    assert_eq!(dot_f32(&[2.0], &[3.0]), 6.0);
    // exact for small integer-valued inputs (no rounding)
    let a: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..16).map(|_| 1.0).collect();
    assert_eq!(dot_f32(&a, &b), (0..16).sum::<i32>() as f32);
}
