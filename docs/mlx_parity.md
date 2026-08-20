# Closing the MLX speed gap (qwen35 / GPU decode)

Deep inspection of MLX to reach parity **without an MLX dependency or their model** —
by understanding the mechanism and building our own, not copying code.

## The measured gap (Qwen3.8-27B, q4_k, M4 Max)

| stack | decode tok/s | ms/token | bandwidth vs 546 GB/s peak |
|---|---|---|---|
| MLX (mlx_vlm) | **29.3** | 34 | ~80% (near-optimal) |
| HOS single-token | 11.2 | 89 | ~26% |
| HOS resident-MTP | 9.5 | 105 | (MTP is net-negative here — verify overhead > acceptance) |

Bandwidth floor = 14GB (q4_k weights) / 546 GB/s ≈ **26 ms/token ≈ 38 tok/s**. MLX (34ms)
runs essentially at that floor. HOS (89ms) spends **~62 ms/token = overhead, not bandwidth.**
So this is a *dispatch/latency* gap, not a kernel-math gap. Our isolated q4_k matvec already
hits 52% of peak (after the cooperative-unpack fix); the problem is the forward runs those
matmuls in a dependency chain that drains the GPU between ~640 small dispatches/token.

## Why MLX has almost no overhead (from its headers: device.h / resident.h)

1. **Concurrent command encoding + fences.** `CommandEncoder::ConcurrentContext` +
   per-output `fences`: independent ops (q/k/v, gate/up, and across the graph) run
   concurrently; dependencies are enforced by *fences* on specific resources, not by a
   serial encoder that drains after every op. HOS uses a **serial** encoder with
   `memory_barrier_with_resources` — every dispatch waits for the previous.
2. **Rare commits.** `needs_commit()`/`commit()` batch many dispatches per command buffer.
   HOS already does one command buffer/token (good), but see #1.
3. **Residency sets.** `ResidencySets` (Metal-3 `MTLResidencySet`) with a standing
   `requestResidency()` attached to the queue keeps weights pinned resident — no per-access
   residency/paging cost. **HOS uses none.**
4. **Graph fusion (lazy eval / `mx.compile`).** MLX builds an op graph then fuses elementwise
   chains (rmsnorm → matmul → add → norm, gate·silu·up) into far fewer kernels before
   dispatch. HOS dispatches each elementwise op separately (~10-15 ops/layer × 64 layers).

The affine-4bit format itself is *not* the reason (same ~4.3 bits, same ~14GB, same bandwidth).

## Integration roadmap (what's implementable in metal-rs 0.33 — all present: `use_resource`, `use_heap`, `update_fence`/`wait_for_fence`)

Ordered by impact/effort. Each is HOS's own implementation of the *technique*, not MLX code.

1. **Fuse the elementwise chain into the matmuls (biggest lever).** Fold RMSNorm into the
   consuming matvec (kernel normalizes its input then dots), fold the residual-add into the
   matvec output, keep SwiGLU as one kernel. Cuts dependency hops per layer ~2-3× → the GPU
   stops draining between the small ops. Target the gemma + qwen35 resident runners. Effort:
   high (new fused kernels), impact: high — this is the bulk of the 62ms.
2. **Concurrent encoder + fences.** Replace the serial encoder with a concurrent one; emit a
   fence after each producer and wait on it only at real consumers (q/k/v overlap; gate/up
   overlap). A plain concurrent encoder without fences did NOT help gemma earlier — the fences
   are the missing piece. Effort: medium-high, impact: medium (overlaps only the independent ops).
3. **Residency: allocate weights from a MTLHeap + `use_heap` (or `use_resource` the weight
   set once per encoder).** Pins the ~14GB resident so no per-token residency cost. Effort:
   medium (route GpuMatrix allocations through a heap), impact: unknown-but-cheap-to-try —
   measure first.
4. **Wider speculative verify (3-4 tokens) OR drop MTP here.** MTP is currently net-negative
   on prose (9.5 < 11.2 single-token). Either raise acceptance / widen the verify batch so the
   weight read amortizes over more tokens, or gate MTP off by default for this model.

## Validation method (already built)

- `HOS_QWEN35_TIMING=1 flwr run <model>` → in-process decode tok/s (single-token & resident-MTP).
- `flwr __cmphf/__cmpcap/__cmphfv` → cosine-validate any ingest/kernel change stays byte-faithful.
- Re-bench MLX any time: `mlx_vlm.generate` in a venv → `res.generation_tps`.

## Bottom line

The gap is real (2.6×) and it's **dispatch overhead, not our kernels**. Parity is reachable by
adopting MLX's *techniques* — fuse elementwise into matmuls (#1), fence-based concurrency (#2),
pinned residency (#3) — each built natively in HOS. Start with #1 (largest) + #3 (cheapest to
try), measure against the 34ms MLX floor.
