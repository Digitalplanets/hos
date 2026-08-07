# The HOS Model Format (`.hos`) — Specification v2

`.hos` is HOS's native model format: a single, self-describing capsule holding
**weights + optimizer state + a lifecycle card** (architecture, identity, lineage,
training history, exact-resume state) with integrity checksums. Unlike GGUF or
safetensors — passive weight containers that need external code to run — a `.hos`
file carries its own architecture spec, which the HOS interpreter can execute, and
its own genealogy.

This document specifies the on-disk format precisely so other tools can read and
write it.

## 1. Conventions

- All integers are **little-endian**.
- Tensor data is **32-byte aligned** (mmap-friendly).
- Checksums are **FNV-1a (64-bit)**: `h = 0xcbf29ce484222325`, then for each byte
  `h = (h XOR byte) * 0x100000001b3` (wrapping).
- Strings are UTF-8, length-prefixed where noted.

## 2. File layout

```
┌────────────────────────────────────────────────────────────┐
│ magic        : "HOSF"            (4 bytes)                   │
│ version      : u32               (= 2; v1 also readable)    │
│ flags        : u32               (reserved, 0)              │
│ card_len     : u32                                          │
│ card         : card_len bytes    (UTF-8 JSON — see §3)      │
│ n_tensors    : u32                                          │
│ tensor descriptors × n_tensors   (see §4)                   │
│ ── pad to 32-byte boundary ──                               │
│ tensor data blocks × n_tensors   (stored bytes, then pad 32)│
│ file_checksum: u64               (FNV-1a of all prior bytes)│
└────────────────────────────────────────────────────────────┘
```

**Versioning.** v2 adds a per-tensor `dtype` byte and a stored-byte length to the
descriptor (§4), so a tensor's on-disk bytes may be a compact quantization (Q8_0)
rather than raw f32. v1 files have neither field and are all f32; the reader
branches on `version`, so old files still load. Quantized tensors **decode to
f32 on load**, so the in-memory representation and every consumer are unchanged.

## 3. The card (JSON)

A human-readable JSON object. Required fields:

| Field | Type | Meaning |
|-------|------|---------|
| `format` | string | `"hos"` |
| `spec` | u32 | metadata schema version (1) |
| `name` | string | model name |
| `id` | string | content hash of all tensor data (FNV-1a, hex) = identity |
| `mode` | string | `"trainable"` \| `"inference"` \| `"frozen"` |
| `arch` | object | **self-describing architecture spec** (free-form; the interpreter executes a `{"type":"sequential","layers":[…]}` form) |
| `provenance` | object | `{created, engine, dataset, dataset_hash}` |
| `resume` | object | `{seed, step, rng_state}` — deterministic continuation |
| `history` | array | training runs: `{steps, final_loss, optimizer, lr}` |
| `lineage` | array | ancestor ids, oldest→parent (genealogy) |

The `arch` object is the heart of the format. For a runnable model it is a layer
list the interpreter walks (`embedding`, `linear`, `rmsnorm`, `attention_block`,
`moe_ffn`, …); for a converted model it is a descriptive spec of the source
architecture.

## 4. Tensor descriptor

Per tensor, in order:

```
name_len   : u32
name       : name_len bytes (UTF-8)
role       : u8     (0=weight 1=bias 2=norm 3=embed 4=opt_state 5=scalar)
dtype      : u8     (v2 only: 0=f32 1=q8_0)
ndim       : u32
dims       : u32 × ndim   (row-major, [out, …, in])
n_floats   : u64    (logical element count, always f32-equivalent)
n_bytes    : u64    (v2 only: stored byte count = n_floats*4 for f32)
checksum   : u64    (FNV-1a of this tensor's stored bytes)
```

(v1 descriptors omit `dtype` and `n_bytes`; all v1 tensors are f32 with
`n_bytes = n_floats*4`.) Tensor data (after the descriptor block + 32-byte pad)
is written in descriptor order: each tensor's `n_bytes` stored bytes, each
followed by padding to the next 32-byte boundary. On load, each tensor's bytes
are checksummed against its descriptor `checksum` (a mismatch is a hard error),
then decoded to f32.

**Q8_0 storage.** A `q8_0` tensor stores 32-value blocks of `[f16 scale ‖ 32×i8]`
(34 bytes per 32 weights, ≈3.76× smaller than f32). Only large weight/embedding
tensors (`n_floats % 32 == 0`) are quantized; norms, biases, scalars, and
optimizer state stay f32. The encoder is shared with the GGUF writer.

## 5. Integrity

- **Per-tensor checksum** detects corruption of any single tensor's data.
- **Whole-file checksum** (trailing u64) detects any change anywhere before it.
- **Content identity** (`card.id`) is the FNV-1a of all tensor f32 bytes, so two
  files with identical weights share an id regardless of metadata — a stable,
  reproducible model fingerprint.

## 6. Why this format

| Capability | GGUF | safetensors | **`.hos`** |
|------------|:----:|:-----------:|:----------:|
| Weights + dtype/shape | ✓ | ✓ | ✓ |
| Carries full architecture (runnable) | ✗ | ✗ | ✓ |
| Lineage / genealogy | ✗ | ✗ | ✓ |
| Training history + exact resume | ✗ | ✗ | ✓ |
| Per-tensor + whole-file checksums | ✗ | ✗ | ✓ |
| Content-hash identity | ✗ | ✗ | ✓ |
| Native quantized storage (Q8_0) | ✓ | ✗ | ✓ (v2) |

## 7. Tooling

- `hos --hos-info <file.hos>` — print the card + tensor summary.
- `hos --to-hos <model.gguf> [-o out.hos] [--quantize q8_0]` — convert GGUF → `.hos` (`--quantize q8_0` for a compact capsule).
- `hos --ingest <hf-dir> [-o out.hos] [--quantize q8_0]` — mint from a HuggingFace
  checkpoint; `--quantize q8_0` produces a compact capsule (no GGUF in the loop).
- `hos --hos-viz <file.hos> [-o out.html]` — render a visual gene/lineage report.
- Library: `hos::format::{save, save_quantized, load, Card, Named}`;
  `hos::interp::run_file` runs a `.hos` from its embedded arch spec.

## 8. Limitations / roadmap

- **Q8_0** is the only on-disk quant today; lower-bit **K-quant** storage (Q4_K,
  etc.) is future work.
- Quantized tensors **decode to f32 in memory** on load, so v2 shrinks the *file*
  but not yet the runtime footprint; a quant-resident in-memory path is roadmap.
- `flags` remains reserved for future feature bits (e.g., whole-file compression).
- **Runnable.** Both `hos --to-hos <gguf>` and `hos --ingest <hf-dir>` write a
  capsule whose card carries the full engine metadata (`card.meta`:
  hyperparameters + tokenizer), and `hos -m model.hos` / `flwr run model.hos`
  load and run it directly (CPU; GPU for `.hos` is roadmap). Verified
  bit-identical to the source — GGUF: Qwen2.5-0.5B both 22.533; HF: SmolLM2-135M
  checkpoint and `.hos` both 42.288.
- **Quantization targets:** `--quantize q8_0 | q4_0 | q5_0 | q6_k | q5_k | q4_k`
  (weight/embedding tensors; norms/biases stay f32). The K-quants use 256-element
  super-blocks with per-sub-block scales + a least-squares scale fit, so they
  beat the simple quants at equal bit-width (Llama-3.2-1B: q4_k 18.4 < q4_0 19.3,
  q5_k 16.8 < q5_0 17.5, q6_k 16.3). A tensor is K-quantized only if its row
  dimension is a multiple of 256; otherwise it falls back to f32.
