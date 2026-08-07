# Brief 01 — GPU preflight: stop HQ4 / MoE hard-crashes

**Problem (two reachable hard crashes on supported models).**
1. **HQ4 on GPU → `exit(2)`.** A capsule carrying our NF4 `HQ4` weights uploads every linear
   to the GPU; `enc_matvec` has no HQ4 kernel, so it hit a catch-all that called
   `std::process::exit(2)` — the process died before the first token (a library must never
   kill the host process).
2. **qwen2moe on GPU → panic on token 1.** A MoE model passes `arch.gpu_supported()` (it's
   detected as Qwen2) but its dense-FFN slots are placeholders; the fused runner has no expert
   branch, so it panicked at `as_gpu` on the dummy weight.

**Fix (one preflight covers both).** `model::gpu_quant_supported(src)` scans *every* layer's
linear weight types and returns `false` if any lacks a Metal matvec kernel (only
F32/F16/Q4_0/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K are supported) **or** if the model is MoE
(`blk.0.ffn_gate_exps.weight` present). The GPU decision in `lib.rs` and `main.rs` now ANDs
this in, so HQ4 / MoE models take the **correct, already-working CPU forward** instead of
crashing. The dead `exit(2)` arm became a `panic!` documented as unreachable-after-preflight.

**Why this is the right stopgap.** It is pure predicate logic — no numerics change, so it
cannot perturb any model's output (the golden/parity net is unaffected). Cost: HQ4/MoE lose
GPU acceleration on Apple Silicon, which is strictly better than `exit(2)`.

**Files:** `src/model.rs` (`gpu_quant_supported`), `src/lib.rs` (use_gpu), `src/main.rs`
(use_gpu), `src/metal_be.rs` (exit→panic).

**Verification.** `cargo test --release` green (23 + 22 + … all pass). Full hardening (next):
a focused test that synthesizes a tiny HQ4 capsule and a tiny MoE capsule and asserts the
predicate returns `false` for both and `true` for a Q8_0/Q4_K capsule.

**The real (sovereign + fast) follow-ups, tracked separately:**
- `matvec_hq4_co` — fork `matvec_q4_0_co`, swap the dequant for a 16-entry NF4 LUT → HQ4 finally
  gets a GPU kernel (~8–15×). Needs a CPU-vs-GPU parity test before merge.
- GPU MoE — CPU routing + per-selected-expert `enc_matvec` dispatch, reusing existing kernels.
