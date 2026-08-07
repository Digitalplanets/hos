# HOS — Architecture & API Reference

HOS is a from-scratch local LLM inference engine in Rust. It loads GGUF models
and runs them on CPU (multithreaded) or Apple-Silicon GPU (Metal). It is both a
**library** (`hos::…`) and a **CLI** (`hos`).

---

## Crate layout

```
src/
  lib.rs        public API: Engine (load/generate/perplexity/bench), sample(), exports
  main.rs       CLI: arg parsing, generate loop, run_qwen35, bench/perplexity, demos
  error.rs      HosError + Result — fallible loading/parsing (no panics on bad input)
  gguf.rs       GGUF parser (bounds-checked) + dequantization
  gguf_write.rs GGUF writer + Q8_0 quantizer (the encode direction)
  safetensors.rs  HuggingFace safetensors reader (single-file + sharded)
  hf.rs         HF checkpoint adapter: config.json + safetensors -> ModelSource
  tokenizer.rs  byte-exact GPT-2-family BPE tokenizer (gpt-2 / qwen2 / llama-bpe)
  model.rs      Arch detection, Config, Weight (CPU/GPU), Model loader, cpu_matmul
  forward.rs    CPU transformer forward pass + KV-cache State
  metal_be.rs   Metal backend: Gpu, GpuMatrix, GpuRunner, all compute kernels
  qwen35.rs     Qwen3.5 hybrid (gated delta-net): Cfg, Qwen35, State, Qwen35Gpu
  tensor.rs     hos-tensor: from-scratch tensor + reverse-mode autograd + AdamW
  interp.rs     self-running interpreter — forward pass from a JSON arch spec
  format.rs     the HOS native format (.hos): weights + optimizer state + card
  train.rs      training helpers: minibatch grad-accumulation, shuffling
  finetune.rs   real-model finetuning: a Llama-family forward in autograd ops (FtModel)
  peft.rs       parameter-efficient finetuning on a frozen base: LoRA + RGA (genome)
  chat.rs       chat templating: per-family templates + special-token-aware encoding
  bin/flwr/     flwr — a separate app over the hos library (not the hos engine bin):
                  main.rs   run / serve / pull / list / show / cp / rm + arg parsing
                  serve.rs  hand-rolled OpenAI HTTP daemon; concurrent accept, serial gen
                  store.rs  provenance-native model store + `flwr pull` (via system curl)
examples/
  chat.py       reference chat client (calls the `hos` CLI)
tests/          autograd grad-checks, gguf codec, hos format, fuzz, harness
```

Two inference families:
- **Standard transformers** (Llama / Mistral / Qwen2): `model.rs` + `forward.rs`
  (CPU) and `metal_be::GpuRunner` (GPU).
- **Qwen3.5 hybrid** (`qwen35`): `qwen35.rs` — `Qwen35` (CPU) and `Qwen35Gpu`
  (GPU-resident). Experimental.

…plus a **training + format stack** (`tensor.rs` autograd, `interp.rs` spec
interpreter, `format.rs` `.hos` files, `train.rs`) that is independent of the
inference path — see the sections below.

---

## Public library API (`lib.rs`)

```rust
let mut eng = hos::Engine::load(path: &Path, gpu: bool)?;   // -> hos::Result<Engine>
eng.generate(prompt: &str, max_tokens, temp, top_k, top_p,
             rep_penalty, repeat_last_n, seed, |piece: &str| { ... }) -> usize;
let (scored, mean_nll, ppl) = eng.perplexity(&ids);          // correctness metric
let b = eng.bench(prompt, decode_tokens);                    // prefill/decode tok/s
```

- `Engine::load` — opens the GGUF, builds the tokenizer, loads the model (GPU if
  `gpu=true`), allocates state. Returns `Err(HosError)` (never panics) on a
  missing / malformed / unsupported model.
- `Engine::generate` — encodes the prompt, prefills, then streams tokens to the
  `on_token` callback; returns the number of tokens generated.
- `Engine::perplexity` — mean per-token NLL + perplexity over a token sequence
  (exact log-softmax; resets the KV cache).
- `Engine::bench` — separated prefill / greedy-decode timings (`Bench`).

All loading/parsing returns `hos::Result<T>` (`= Result<T, hos::HosError>`); the
CLI prints `[hos] error: …` and exits non-zero rather than panicking.

> Note: `Engine` currently drives the standard-transformer path. The qwen35
> hybrid is driven via the CLI (`run_qwen35` in `main.rs`); folding it behind
> `Engine` is a small future cleanup.

Free functions:
- `sample(logits, temp, top_k, top_p, rep_penalty, recent, rng) -> u32` —
  temperature softmax → top-k → nucleus top-p → repetition penalty; greedy when
  `temp <= 0`.
- `next_rand(&mut u64) -> f32` — xorshift64 RNG.

---

## `gguf.rs` — model file parsing

```rust
Gguf::open(path) -> io::Result<Gguf>
g.meta_u64/meta_f32/meta_str(key)     // metadata access
g.dequant(name) -> Vec<f32>           // tensor → f32 (any supported quant)
g.raw(name) -> (&[u8], u32, usize)    // raw quantized bytes, type, n_elements
g.has(name) -> bool
bytes_for(ggml_type, n) -> usize      // byte size of a tensor
```

Supported quant types (`GGML_*` consts): `F32, F16, Q8_0, Q4_0, Q5_0, Q4_K,
Q5_K, Q6_K`. Unsupported types exit with a clear message rather than panicking.

---

## `tokenizer.rs` — byte-level BPE

```rust
Tokenizer::from_gguf(&g) -> Tokenizer  // reads vocab + merges from metadata
tok.encode(text, add_bos) -> Vec<u32>
tok.decode_into(id, &mut Vec<u8>)      // streaming decode
tok.decode(&[u32]) -> String
tok.bos / tok.eos                      // Option<u32>
```

Encoding is **byte-exact**: a hand-rolled GPT-2-family pre-tokenizer with the
right variant (`gpt-2` / `qwen2` / `llama-bpe`) chosen from `tokenizer.ggml.pre`,
then rank-ordered BPE merges. SentencePiece models (`tokenizer.ggml.model ==
"llama"`) take a separate score-merge path with ▁-space normalization and
`<0xNN>` byte-fallback. Decoding is exact for both. `bos`/`eos` are read from metadata.

---

## `model.rs` — config, weights, loading

```rust
enum Arch { Llama, Qwen2, Mistral, Gemma, Phi3, OlMoe, Qwen35Hybrid, Other }
Arch::detect(&g)        // by general.architecture + tensor presence (ssm_a)
Arch::is_transformer()  // all but Qwen35Hybrid/Other run today
Arch::gpu_supported()   // GPU-fused: Llama | Qwen2 | Mistral (others = CPU)
Arch::rope_neox()       // NEOX: Qwen2/Gemma/Phi3/OlMoe; interleaved: Llama/Mistral

enum Weight { Cpu { data, rows, cols }, Gpu(GpuMatrix) }
Weight::matvec(gpu, x, &mut out)   // y = W·x on whichever backend holds W
cpu_matmul(&mut y, w, x)           // rayon-parallel f32 matvec

struct Config { dim, n_layers, n_heads, n_kv_heads, head_dim, ffn_dim,
                vocab_size, ctx_len, rms_eps, rope_base, arch, attn_bias, rope_neox,
                // per-arch quirks:
                embed_scale,      // Gemma scales embeddings by sqrt(dim)
                norm_add_one,     // Gemma (1+w) RMSNorm — false: converter bakes it in
                geglu,            // Gemma FFN uses a GELU gate, not SiLU
                attn_softcap, final_softcap,   // Gemma-2 logit soft-caps (0 = off)
                n_experts, n_experts_used }    // MoE: total experts / top-k routed (0 = dense)

// Layer also carries optionals: post_attn_norm/post_ffn_norm (Gemma sandwich),
// q_norm/k_norm (OLMoE QK-norm), and moe: Option<MoeLayer> (router + quantized experts).
Model::load(&g, gpu: Option<&Gpu>) -> Result<Model>
```

Per-arch behavior is selected automatically: RoPE style (NEOX vs interleaved),
optional q/k/v attention bias (Qwen2), tied embeddings, and the Gemma/Phi-3/MoE
quirks above. Linear weights load to the GPU in native quantized form when `gpu`
is set and the arch is GPU-supported; norms/embeddings — and **all** weights for
the CPU-only arches — stay on CPU. Phi-3's fused `attn_qkv` / `ffn_up` tensors are
row-split on load; MoE experts are stored quantized and dequantized per routed token.

---

## `forward.rs` — CPU transformer pass

```rust
forward::State::new(&cfg)                               // KV cache
forward::forward(&model, &mut state, token, pos, gpu)   // -> logits
```

Per layer: RMSNorm → Q/K/V projections (+bias) → RoPE → GQA attention (KV cache)
→ output proj → residual → RMSNorm → SwiGLU FFN → residual. With `gpu = Some`,
matmuls dispatch to the GPU per call; otherwise multithreaded CPU.

Arch-conditional branches in the same pass:
- **Gemma-2**: embedding scaled by √dim; GeGLU (GELU gate) instead of SwiGLU;
  attention- and final-logit soft-caps (`c·tanh(x/c)`); "sandwich" `post_attn_norm`
  / `post_ffn_norm` applied before each residual add.
- **OLMoE / MoE**: the FFN is replaced by a router → softmax → top-k expert
  selection; each routed expert is dequantized on demand (`dequant_expert`),
  run as SwiGLU, and accumulated by its raw router weight (no renormalization).
- **OLMoE QK-norm**: `q_norm` / `k_norm` RMSNorm applied to the full q/k vectors
  before RoPE.

Soft-cap and the MoE expert math are **CPU-only** — there is no GPU kernel for them.

---

## `metal_be.rs` — Metal backend

```rust
Gpu::new() -> Gpu
gpu.upload_matrix(&[f32], rows, cols) -> GpuMatrix   // stored as f16
gpu.upload_quant(bytes, ggml_type, rows, cols) -> GpuMatrix  // native quant
gpu.matvec_into(&GpuMatrix, x, &mut out)             // f16 scalar matvec
gpu.deltanet_step(s, q, k, v, g, beta) -> (o, new_s) // single-head delta-net
gpu.device() / gpu.queue() / gpu.fused_library()

GpuRunner::new(gpu, &model) ; runner.forward(&model, token, pos) -> logits
```

- **`GpuRunner`** — the transformer GPU path: weights resident (native quant),
  whole token in one command buffer, **coalesced** matvec kernels.
- **Kernels** (MSL, in `FUSED_SRC`): `matvec_{f32,f16,q8_0,q5_0,q4k,q5k,q6k}` and
  coalesced `*_co` variants; `rmsnorm`, `rope`, `store_kv`, `attention`
  (flash-style online softmax), `swiglu`, `add_inplace`; and the qwen35 set:
  `deltanet(_multi)`, `conv1d`, `l2norm_heads`, `rmsnorm_heads`, `gated_norm`,
  `rope_partial`, `sigmoid_mul/inplace`, `ssm_decay`, `extract_qgate`.

Matvec kernels: **coalesced** = one 32-lane SIMD group per output row (reads
consecutive bytes, `simd_sum` reduction); used for all K-quants.

---

## `qwen35.rs` — Qwen3.5 hybrid (gated delta-net)

32 blocks = 8 full-attention (every 4th) + 24 gated-delta-net (linear-attention)
blocks, each followed by a SwiGLU FFN.

```rust
Cfg::from_gguf(&g)              // hybrid config (ssm dims, rope sections, etc.)
block_kinds(&g, n_layers)      // which layers are attention vs SSM
validate(&g)                   // structure check (hos --qwen35-check)
Qwen35::load(&g, gpu) -> Qwen35
Qwen35::forward(&mut State, token, pos, gpu) -> logits   // CPU (+GPU matmuls)
Qwen35Gpu::new(gpu, &model) ; .forward(&model, token, pos) -> logits  // resident
```

- **Attention block:** gated attention (Q projection emits query+gate), per-head
  QK-RMSNorm, partial NEOX rope, GQA, sigmoid gate, output proj.
- **Linear block:** input proj → causal conv1d (+SiLU) → split Q/K/V → L2-norm
  Q/K → gated delta-net recurrence → gated RMSNorm → output proj.
- **`Qwen35Gpu`** keeps activations + KV/conv/SSM state resident on the GPU and
  runs a whole token in one command buffer.

Gated delta-net per-head recurrence (state `S` is `head_dim × head_dim`):
```
g = exp(decay);  S *= g
sk[j] = Σ S[j][i]·k[i]
d[j]  = β·(v[j] − sk[j])
o[j]  = Σ S[j][i]·q[i] + d[j]·(k·q)
S[i][j] += d[i]·k[j]
```

---

## `safetensors.rs` + `hf.rs` — native HuggingFace loading

HOS loads a raw HF checkpoint folder (`config.json` + `*.safetensors` +
`tokenizer.json`) with no GGUF / llama.cpp / transformers in the loop.

```rust
safetensors::SafeTensors::open_dir(dir)   // single-file or sharded; F32/F16/BF16
hf::HfModel::open(dir) -> HfModel          // implements model::ModelSource
tokenizer::Tokenizer::from_hf(dir)         // byte-level BPE from tokenizer.json
```

`HfModel` is an **adapter**, not a second loader: it implements the
`model::ModelSource` trait (`meta_*`, `has`, `dequant`, `raw`) — the same surface
`Gguf` exposes — so `Model::load` runs all six architectures unchanged. In
`HfModel::open` it does, in memory, what `convert_hf_to_gguf.py` does on disk:

- **renames tensors** HF → GGUF (`model.layers.N.self_attn.q_proj.weight` →
  `blk.N.attn_q.weight`), arch-aware (Gemma's sandwich/pre-FFN norm names, Phi-3's
  fused `qkv_proj`/`gate_up_proj`, OLMoE's per-expert stacking);
- **synthesizes the GGUF metadata keys** from `config.json`
  (`hidden_size` → `…embedding_length`, etc.);
- **permutes Q/K** for Llama/Mistral — HF stores them for rotate-half RoPE, but
  HOS (like GGUF) uses interleaved-pair RoPE for those arches, so the per-head
  rows are reordered. *Skipping this gives coherent greedy output but wrong
  probabilities — it roughly doubled perplexity until fixed;*
- **bakes Gemma's `(1 + w)`** into the norm weights, and folds BF16 → F16 for the
  matmul weights (lossless: F16 has more mantissa bits than BF16).

`Engine::load` (and the `hos` CLI) auto-route: a directory with a `config.json`
goes to the HF path, anything else to GGUF. `Tokenizer::from_hf` routes by family
— byte-level BPE (`Ġ`) from `tokenizer.json`, or SentencePiece (`▁`) from the
`tokenizer.model` protobuf (parsed by a small hand-rolled reader — no protobuf
dependency). Both reuse the existing encode/decode; the SP path was verified
token-for-token identical to the GGUF tokenizer on Phi-3.

---

## `tensor.rs` — autograd core

A from-scratch reverse-mode autograd, independent of the inference path. A
`Tensor` is an `Rc<RefCell<Inner>>` node holding `data`, `shape`, `grad`,
`parents`, and a `backward` closure (the op's vector–Jacobian product).

```rust
Tensor::param(data, shape) / ::constant(...) / ::randn(shape, &mut seed)
t.matmul(&b) / .add / .sub / .mul / .relu / .silu / .sigmoid / .softmax_rows
 .rmsnorm(&w) / .rmsnorm_eps(&w, eps) / .embedding(ids) / .cross_entropy(&targets)
 .reshape / .transpose / .bmm / .transpose12_4d / .col / .mul_broadcast / ...
 .rope(n_heads, head_dim, base, neox, &pos)  // configurable RoPE (interleaved or NEOX)
 .repeat_interleave_dim0(g)                   // grouped-query attention K/V expansion
t.backward()        // topo-sort the graph, seed scalar grad = 1.0, run VJPs in reverse
t.zero_grad() / t.grad() / t.set_data(&d)

AdamW::new(&params, lr, wd).step(&params, &decay)   // β1=0.9 β2=0.95, decoupled decay
```

Backward does a DFS to build topological order, seeds the output grad to 1.0
(scalar-loss assumption), then walks nodes in reverse calling each `backward`
closure; grads accumulate via `add_grad` so shared subgraphs sum correctly.
`matmul` can dispatch to the GPU, and its backward **skips the gradient of a
frozen (`constant`) weight** — roughly halving cost when a base model is frozen
for PEFT. Every op is finite-difference gradient-checked in `tests/autograd.rs`.
`train.rs` adds `shuffle` and `train_minibatch` (per-example grad accumulation).

---

## `finetune.rs` + `peft.rs` — training & adaptation of real models

The autograd core trains toy nets; these two modules use it to **finetune real
pretrained transformers**. Both build a real Llama-family forward (RMSNorm,
configurable RoPE, grouped-query attention, SwiGLU) out of `tensor.rs` ops, so
gradients flow to the weights — and both are gated on **forward-parity** with the
inference path (`forward.rs`): the autograd-built forward is verified
**byte-identical** to the hand-coded one before any training is trusted.

```rust
// full-parameter finetuning (finetune.rs)
finetune::FtModel::from_model(&model)         // wrap a loaded model's weights as trainable Tensors
ft.forward(ids) -> logits ; finetune::check_parity(&model, ids)  // == forward.rs (0.000e0)

// parameter-efficient finetuning on a FROZEN base (peft.rs)
peft::PeftModel::build_multi(&model, "lora"|"rga", &cfg, n_genomes, seed)
p.loss_g(ids, targets, lambda, genome_idx)    // base = constants; only adapters train
p.params_bank() / p.params_genome(gi)         // freeze the bank, train one genome
```

- **`FtModel`** — full-parameter finetuning. Loads any model's weights (GGUF *or*
  HF) as trainable tensors and trains them. CLI: `hos --finetune`,
  `hos --finetune-check`. Llama / Mistral / Qwen2 today.
- **`PeftModel`** — two adapter methods on one frozen-base substrate (base weights
  are autograd `constant`s, so only the adapter trains and the matmul backward
  skips the base gradient):
  - **LoRA** — the low-rank baseline (`W·x + (α/r)·B(A·x)` on Q/V).
  - **RGA** (Regulatory Genome Adapters) — a shared bank of tiny gene-modules,
    gated by a compact per-domain **genome** (`gate = sigmoid(h·W_r + γ·W_g)`).
    This is the M3 genome-regulated-mixture mechanism applied as PEFT.
  CLI: `hos --peft` plus the experiment harness `--peft-compare` /
  `--peft-interference` / `--peft-heldout` / `--peft-recombine`. Adapters save as
  a `.hos` capsule whose lineage points to the base model.

---

## `chat.rs` + `flwr` — the chat layer

`chat.rs` (library) turns a conversation into the exact token stream a model
expects. It detects the chat dialect from the **special tokens present in the
vocabulary** — ChatML, Llama-3 headers, Gemma, Phi-3, Mistral — and renders the
turns by splicing special-token ids directly between BPE'd text (the byte-exact
`encode` path deliberately does not split on specials; `Tokenizer::special_id`
looks them up). `Engine::chat` resets the KV cache, renders the full history,
and generates until a turn-terminator (`stop_ids`) or `eos`. No Jinja engine —
canned templates keep the dependency-free stance.

`flwr` (`src/bin/flwr/`) is a **separate binary** built on the `hos` library —
the usable app the way `ollama` wraps `llama.cpp`. It is deliberately *not* part
of the `hos` engine command. `flwr run` is an interactive REPL; `flwr serve` is
a hand-rolled (`std::net`, no web framework) HTTP daemon speaking the
OpenAI-compatible `/v1/chat/completions` API (SSE streaming or one-shot JSON)
plus `/v1/models`. The engine stays an engine; the app adds the product surface.

**The store (`store.rs`)** is the provenance-native answer to Ollama's blob
registry. `flwr pull <hf-repo|gguf-url>` downloads into `~/.hos/store/<name>/`
(via the system `curl` — downloading is the one interop seam, so it stays out of
the crate's dependency set) and writes a `manifest.json` that records, like a
`.hos` lineage card, the `source` + `revision`, an FNV content hash per file,
and a combined `identity` hash. `flwr list` / `flwr show` read it back; name
resolution (`run`/`serve`) checks the store first. `flwr cp <src> <dst>`
either duplicates a store entry (copying the files, keeping the same content
`identity`, adding a `copied_from` edge — lineage, not a blank clone) or imports
a local path — an HF checkpoint directory or a `.gguf` file — under a new name
with `source: local:<abspath>`; `flwr rm <name>` deletes one. A stored model
carries its ancestry instead of being an anonymous blob.

**Quantizing (`gguf_write.rs` + `flwr quantize`).** HOS reads many quant formats
but, until recently, only *decoded* them. `gguf_write::requantize_gguf` is the
encode direction: it reads each tensor of a source GGUF as f32, re-encodes the
quantizable ones (≥2-D with `dims[0] % 32 == 0`) to **Q8_0** while keeping 1-D
norms/biases as F32, and writes a new valid GGUF — **copying the source metadata
(including the tokenizer arrays) verbatim**, so nothing needs to be synthesized
and the engine loads the result unchanged. `flwr quantize <src> <dst>` drives it
and records the derived model's lineage (`quant` + `copied_from`). Verified to
preserve perplexity (Qwen2.5-0.5B Q4_K_M → Q8_0: 22.53 → 22.72). K-quant
*encoders* and HF→GGUF quantization (which must synthesize tokenizer metadata)
are future work.

**Concurrency in `serve`.** Connections are accepted and parsed on per-connection
threads, so a slow client never stalls the accept loop and `GET /` / `/v1/models`
answer immediately. Chat jobs (all `Send`: `TcpStream` + parsed messages) funnel
over an `mpsc` channel to the single thread that owns the `Engine`, which runs
them one at a time — the engine never crosses a thread boundary, and the single
KV cache is never shared. Parallel decode would require parallel engines
(parallel memory), which a local single-model runner deliberately avoids.

---

## `interp.rs` — self-running interpreter

Executes a model whose architecture is **data**, not code: the `arch` JSON in a
`.hos` card is a list of ops the interpreter walks.

```rust
interp::run(arch, &weights, ids, input) -> Result<Tensor>   // generic net (autograd Tensors)
interp::run_file(path, ids, input) -> Result<Tensor>        // load .hos, then run
interp::run_llama_from_spec(arch, &weights, ids) -> Result<Vec<f32>>  // Llama-family prefill
```

`run` supports `embedding`, `linear`, `relu`, `silu`, `rmsnorm`, `softmax_rows`,
`max_pool_rows`, `scale`, `add_pos`, `attention_block`, `ffn_block`, `moe_ffn` —
each returns a `Result` (malformed specs yield `HosError::Spec`, never panic).
`run_llama_from_spec` is the M4 path: a generic Llama/Qwen/Mistral prefill driven
purely by spec params + GGUF tensor names, verified byte-identical to the
hand-coded `forward.rs` (`hos --interp-check`).

---

## `format.rs` — the `.hos` native format

A self-describing capsule: magic `HOSF`, a JSON **card**, tensor descriptors, then
32-byte-aligned f32 tensor blocks, with FNV-1a checksums per tensor and over the
whole file. The card carries `arch` (the interpreter spec above), content-hash
`id`, `mode`, `provenance`, deterministic `resume` state, append-only training
`history`, and `lineage` (ancestor ids).

```rust
format::save(path, &[Named], &Card) -> io::Result<()>
format::save_quantized(path, &[Named], &Card) -> io::Result<()>  // Q8_0 weights
format::load(path) -> io::Result<(Vec<Named>, Card)>   // verifies checksums
```

**Quantized capsules (v2).** The format is versioned; v2 adds a per-tensor
`dtype` byte and a stored-byte length, so a tensor's on-disk bytes can be Q8_0
instead of raw f32. `save_quantized` quantizes the big weight/embedding matrices
(`nfloats % 32 == 0`) and keeps norms/biases/scalars/optimizer-state in f32 — the
llama.cpp split. Quantized tensors **decode back to f32 on `load`**, so every
caller (interpreter, training, inspection) is unchanged; only the file is
smaller. v1 files still load (the reader branches on version). This is what
`hos --ingest --quantize q8_0` uses to produce **HF → quantized `.hos`** with no
GGUF in the loop (SmolLM2-135M: 538 MB → 143 MB; round-trip verified within the
Q8_0 error bound). The Q8_0 encoder is shared with `gguf_write.rs`.

See [HOS_FORMAT.md](HOS_FORMAT.md) for the byte-level spec and a comparison to
GGUF / safetensors. Tooling: `hos --to-hos` (convert a GGUF), `hos --ingest`
(mint from HF, `--quantize q8_0` for a compact capsule), `hos --hos-info`
(inspect a card), `hos --hos-viz` (HTML "genetic code" report).

---

## Python bindings (`py/`)

PyO3 (abi3) wrapper built with maturin:

```python
import hos
eng = hos.Engine("model.gguf", gpu=True)
eng.generate("Hello", max_tokens=128, temp=0.7)   # -> str
eng.perplexity("held-out text")                    # -> (scored, mean_nll, ppl)
eng.tokenize("hi"); eng.vocab_size()
hos.inspect_hos("model.hos")                       # -> (name, id, mode, arch_json, n_tensors, n_values)
```

---

## CLI reference

```
hos -m <model.gguf> -p "<prompt>" [options]

  -m, --model PATH|NAME   model (NAME resolves via $HOS_MODELS_DIR,
                          ~/Documents/hos/models, ~/.hos/models, or $HOS_MODEL)
  -p, --prompt TEXT
  -n, --n-predict N       max tokens (default 64)
      --temp F            0 = greedy (default 0)
      --top-k N           default 40
      --top-p F           default 0.95
      --repeat-penalty F  default 1.1
      --repeat-last-n N   default 64
      --seed N            default 42
      --gpu               use the Metal backend
      --no-echo           don't print the prompt (for programmatic use)
      --info              inspect arch / config / tensors / quant types
      --gpu-test          validate + benchmark the GPU matvec kernel
      --qwen35-check      validate a qwen35 model's structure
      --deltanet-test     validate the delta-net GPU kernel vs CPU
```

Backends are selected automatically by architecture; `--gpu` enables Metal.
Output is verified bit-identical between CPU and GPU on supported models.
