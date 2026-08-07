# Brief 02 — x86 AVX2 dot + making hos build for Windows

**Goal.** Consumer Windows/Intel/AMD performance, no new deps. Since x86 GPU would need a
dependency, the only lever there is an optimized CPU kernel.

**What was there.** `dot_f32` (the one hot kernel behind every matvec/FFN/lm_head) had a
hand-written aarch64 NEON dual-accumulator FMA path but a **plain scalar fallback on x86** —
which LLVM will not autovectorize (f32 add isn't associative). So x86 ran the matmul inner
loop unvectorized.

**Change 1 — x86 AVX2+FMA path (not a ggml copy).** Ported *our own* NEON two-accumulator
design to x86: two 256-bit YMM accumulators, `_mm256_fmadd_ps`, 16 floats/iter, an 8-wide
block, then a scalar tail. Runtime-gated by `is_x86_feature_detected!("avx2","fma")` cached in
a `OnceLock`, with the scalar path as fallback so pre-Haswell CPUs stay correct (we deliberately
do **not** set `target-cpu=native`, which would SIGILL old CPUs). The `target_feature` fn is
only ever reached through the cached probe (calling it without the features is UB).

**Parity.** Both SIMD paths reorder the summation, so their target is the **high-precision
reference**, not the bit-exact scalar loop — see `tests/parity.rs` (`dot_f32` within a tight
bound of an f64 accumulation across many lengths incl. non-multiple-of-16 tails). The test runs
against whichever arch it's built for, so it validates the AVX2 path on x86 CI.

**Change 2 — hos now builds for Windows at all.** Cross-compiling `x86_64-pc-windows-gnu`
revealed the non-macOS Metal **stub was incomplete**: `metal_be_stub.rs` was missing
`matmul_f32_keyed`, `matmul_abt_keyed`, `matmul_atb_keyed`, `evict` (Gpu) and
`forward_prefill_gpu` (GpuRunner), which `tensor.rs`/`lib.rs` reference unconditionally. Added
the stubs (all `unreachable!` — the GPU path is gated off on non-macOS via `cfg!(target_os =
"macos")`, so they're never called, only needed to type-check). **`cargo build --target
x86_64-pc-windows-gnu` now succeeds.**

**Files:** `src/model.rs` (`dot_f32`, `dot_f32_avx2`, `dot_x86_avx2_fma`, `dot_f32_scalar`),
`src/metal_be_stub.rs`, `tests/parity.rs`.

**Verification.** Native aarch64 `cargo test --release`: all green (0 failed), parity test
passes on the NEON path. Windows target compiles. **Open:** runtime perf/correctness on a real
x86 box (run `cargo test --test parity` on x86 CI to validate the AVX2 numerics live).

**Next (the real high-perf path, separate):** a blocked **SIMD-fused dequant+dot** for HQ4/Q4_K
modeled on our Metal coalesced kernels (VPSHUFB nibble unpack + NF4 LUT + FMA into YMM) — one
pass, no scratch, makes the quant-resident path competitive on x86.
