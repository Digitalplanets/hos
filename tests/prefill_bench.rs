//! Lightweight microbenchmark for the CPU batched-prefill change: it isolates the
//! ONE primitive that changed (per-token `Weight::matvec` loop vs the batched
//! `cpu_matmat`, which dequantizes each weight row once and dots all tokens) on a
//! single synthetic weight. No model load — a few MB and a few ms, so it is safe to
//! run anywhere. Also asserts the two paths are bit-identical.
//!
//! Run: cargo test --release --test prefill_bench -- --nocapture

use hos::model::{cpu_matmat, Weight};
use std::time::Instant;

fn lcg(state: &mut u64) -> f32 {
    // tiny deterministic PRNG in [-1, 1)
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

#[test]
fn prefill_batched_vs_per_token() {
    // One representative projection weight (Gemma-12B hidden = 3840; cols must be a
    // multiple of the K-quant block size 256 — 3840 = 15*256).
    let rows = 3840usize;
    let cols = 3840usize;
    let p = 32usize; // prompt tokens processed together

    let mut s = 0x1234_5678_9abc_def0u64;
    let wf: Vec<f32> = (0..rows * cols).map(|_| lcg(&mut s) * 0.05).collect();
    let x: Vec<f32> = (0..p * cols).map(|_| lcg(&mut s)).collect();

    // Quantized weight (q4_k) — the real prefill path. dequant is fused into matvec.
    let bytes = hos::gguf_write::quantize(&wf, hos::gguf::GGML_Q4_K);
    let wq = Weight::Quant { bytes, ggml_type: hos::gguf::GGML_Q4_K, rows, cols };

    let mut out_pt = vec![0.0f32; p * rows]; // per-token result [P, rows]
    let mut out_ba = vec![0.0f32; p * rows]; // batched result

    // ---- per-token loop (the OLD prefill: re-reads/re-dequants the whole weight per token) ----
    let per_token = |out: &mut [f32]| {
        for t in 0..p {
            wq.matvec(None, &x[t * cols..(t + 1) * cols], &mut out[t * rows..(t + 1) * rows]);
        }
    };
    // ---- batched (the NEW prefill: dequant each row once, dot all P tokens) ----
    let batched = |out: &mut [f32]| cpu_matmat(out, &wq, &x, p);

    // correctness: bit-identical
    per_token(&mut out_pt);
    batched(&mut out_ba);
    assert_eq!(out_pt, out_ba, "batched cpu_matmat must be bit-identical to per-token matvec");

    // warmup
    per_token(&mut out_pt);
    batched(&mut out_ba);

    let iters = 5;
    let t0 = Instant::now();
    for _ in 0..iters {
        per_token(&mut out_pt);
    }
    let pt = t0.elapsed().as_secs_f64() / iters as f64;

    let t1 = Instant::now();
    for _ in 0..iters {
        batched(&mut out_ba);
    }
    let ba = t1.elapsed().as_secs_f64() / iters as f64;

    println!("\n[prefill-bench] q4_k weight {rows}x{cols}, {p} tokens (one projection):");
    println!("  per-token matvec : {:.2} ms", pt * 1e3);
    println!("  batched cpu_matmat: {:.2} ms   ({:.2}x faster, bit-identical)", ba * 1e3, pt / ba);
    println!("  (a real layer has 7 such projections; a full model has ~48 layers)\n");
}
