# HOS Manual

The complete operator's manual for HOS: a from-scratch local LLM inference and
training engine written in Rust, with zero ML runtime dependencies. This document
is the single reference for both shipped binaries and the workflows they support.
For internals see [ARCHITECTURE.md](ARCHITECTURE.md); for the byte-level capsule
spec see [HOS_FORMAT.md](HOS_FORMAT.md); for the embeddable API see
[LIBRARY.md](LIBRARY.md).

---

## 1. What ships

The build produces two binaries plus one library:

| Component | What it is | Use it when |
|---|---|---|
| `hos` | The engine CLI. Raw completion, conversion, quantization, fine-tuning, adapters, diagnostics. | You want direct control over the engine or a training/eval workflow. |
| `flwr` | The app on top of the engine. Chat, an OpenAI-compatible server, and a model store. | You just want to talk to a model or serve an API. |
| `hos` (lib crate) | The engine as a Rust library (`use hos::...`). | You are building your own app on the engine. |

Both binaries link the same core. `hos` is the low-level surface; `flwr` is the
friendly front door. Neither pulls in `llama.cpp`, a Python runtime, or a
web framework.

---

## 2. Requirements and install

You need a Rust toolchain (install via `rustup`). Nothing else.

```sh
# from the repo root
cargo install --path . --bin hos --bin flwr   # installs both to ~/.cargo/bin
```

Or build without installing:

```sh
cargo build --release       # binaries land in target/release/
```

Platform notes:

- macOS (Apple Silicon): full Metal GPU acceleration, on by default (`--cpu` to opt out).
- Windows and Linux (x86-64): CPU inference, accelerated with AVX2 where present.
- GPU acceleration is Metal-only by design. On non-Apple hardware everything runs
  on the CPU path.

---

## 3. Quick start

```sh
# 1. get a model (any GGUF, or a .hos capsule)
flwr pull HuggingFaceTB/SmolLM2-135M-Instruct        # fetch into the local store
flwr list                                             # see what you have

# 2. talk to it
flwr run SmolLM2-135M-Instruct                        # interactive chat (Metal on Apple Silicon)
flwr run SmolLM2-135M-Instruct -p "Explain photosynthesis." -n 200   # one-shot

# 3. serve an OpenAI-compatible API
flwr serve SmolLM2-135M-Instruct --port 11434
# then POST to http://127.0.0.1:11434/v1/chat/completions
```

If you already have a model file, skip the pull and point at the path directly:
`flwr run ./mymodel.gguf` or `hos -m ./mymodel.gguf -p "..."`.

---

## 4. `flwr`: the app

`flwr` wraps the engine with chat templating, a provenance-bearing model store,
and an HTTP server. Run `flwr` with no arguments for the built-in usage summary.

### Commands

```
flwr run <model> [--cpu] [-p "prompt"] [-n N] [--temp T] [--seed S] [--think] [--effort low|medium|xhigh]
flwr serve <model> [--cpu] [--host 127.0.0.1] [--port 11434]
flwr pull <hf-repo | gguf-url> [--revision main] [--name X]
flwr list
flwr show <name>
flwr cp <src> <dst>
flwr quantize <src> <dst> [--type q4_0] [--awq]
flwr rm <name>
```

| Command | What it does |
|---|---|
| `run` | Load a model and chat with it. Interactive REPL by default; add `-p` for a one-shot completion. The chat dialect is auto-detected and the turn ends cleanly on the model's stop token. |
| `serve` | Start an OpenAI-compatible HTTP daemon (`/v1/chat/completions` streaming or one-shot, plus `/v1/models`). Connections are handled concurrently; generation is serialized through the one resident engine. |
| `pull` | Download a HuggingFace repo or a direct `.gguf` URL into the local store and record its provenance. |
| `list` / `ls` | Show the models in the store. |
| `show` | Print a model's provenance card (source, revision, content hashes, lineage). |
| `cp` | Duplicate a store entry (keeping lineage) or import a local path (HF dir or `.gguf`) under a new name. |
| `quantize` | Derive a smaller quantized model from a source, optionally AWQ-scaled (`--awq`). |
| `rm` | Delete a model from the store. |

### Model resolution

`<model>` can be any of:

- a path to a `.hos`, `.flwr`, or `.gguf` file,
- a name previously added with `flwr pull` / `flwr cp`,
- a bare name resolved from the search path below.

The search path, in order:

1. the current directory,
2. `$HOS_MODELS_DIR` (recommended: set this to your models folder),
3. `~/Documents/hos/models`,
4. `~/.hos/models`.

### Common flags

| Flag | Default | Meaning |
|---|---|---|
| `--cpu` | — | Force the CPU path. On Apple Silicon the Metal GPU backend is the default; elsewhere CPU is the only path. |
| `--think` / `--effort` | off / low | Reasoning models (qwen3.5): show the reasoning trace; `--effort low\|medium\|xhigh` sets its depth. Off by default. |
| `-p, --prompt` | (none) | One-shot prompt instead of the interactive REPL. |
| `-n, --n-predict` | 512 | Max tokens to generate. |
| `--temp` | 0.7 | Sampling temperature. `0.0` = greedy. |
| `--top-k` | 40 | Top-k sampling cutoff. |
| `--top-p` | 0.95 | Nucleus sampling cutoff. |
| `--seed` | 42 | RNG seed for reproducible sampling. |
| `--host` / `--port` | 127.0.0.1 / 11434 | Serve address. 11434 is the de-facto local-LLM port, so existing clients work unchanged. |

---

## 5. `hos`: the engine CLI

`hos` drives the raw completion engine and every offline workflow. The core
invocation is:

```sh
hos -m <model> -p "<prompt>" [-n 64] [--temp 0.0] [--seed 42] [--gpu]
```

Sampling flags mirror `flwr`: `--temp`, `--top-k`, `--top-p`, `--seed`,
`--repeat-penalty`, `--repeat-last-n`. `--no-echo` suppresses the prompt in the
output. Below, commands are grouped by task. Run `hos --help` for the short form.

### 5.1 Generate

```sh
hos -m model.gguf -p "The capital of France is" -n 64            # CPU greedy
hos --gpu -m model.gguf -p "Write a haiku about the sea" --temp 0.7
```

Works on `.gguf` and `.hos` inputs alike.

### 5.2 Inspect a model

| Command | Purpose |
|---|---|
| `hos --info -m model.gguf` | Print architecture, tensor list, and quant types. |
| `hos --hos-info file.hos` | Inspect a `.hos` capsule's card: arch spec, provenance, lineage, training history. |
| `hos --hos-viz file.hos [-o out.html]` | Render a `.hos` capsule as a standalone HTML visualization. |
| `hos --ids -m model.gguf -p "text"` | Show the tokenizer's token IDs for a string. |

### 5.3 Convert to the `.hos` format

The native `.hos` capsule bundles weights, optional optimizer state, and a
self-describing card. It is runnable directly.

| Command | Purpose |
|---|---|
| `hos --to-hos model.gguf [-o out.hos] [--quantize q8_0\|q4_0\|q5_0]` | Convert a GGUF model into a self-describing `.hos` capsule. Verified bit-identical to the source. |
| `hos --ingest <hf_dir> [-o out.hos]` | Load a raw HuggingFace checkpoint folder and mint a `.hos` capsule. |
| `hos --gemma4-ingest <hf_dir> [-o out.hos]` | Mint a portable `.hos` capsule from a Gemma 3/4 HuggingFace checkpoint. |

### 5.4 Quantize

HOS has its own from-scratch quantizer and GGUF writer (metadata copied verbatim,
verified to preserve perplexity).

| Command | Purpose |
|---|---|
| `hos --quantize <q8_0\|q4_0\|q5_0>` | Quantize during a `--to-hos` conversion. |
| `hos --quant-awq model.gguf [-o out.hos] [--quantize q4_0] [--awq-alpha 0.5]` | Activation-aware quantization. `--awq-alpha` sets the scaling strength; calibration uses a built-in disjoint corpus. |
| `hos --quant-bench -m model.gguf` | Head-to-head reconstruction error across every quant type on a sample. |
| `hos --q4k` / `--q5k` / `--q6k` | K-quant selector flags for the Gemma loader (see 5.8). |

### 5.5 Benchmark and evaluate

| Command | Purpose |
|---|---|
| `hos --bench -m model.gguf [--gpu]` | Time prefill and greedy decode separately on a fixed workload. Numbers are comparable run-to-run. |
| `hos --perplexity -m model.gguf [file.txt]` | Score held-out text. Uses a built-in fixed passage, or the file you pass. |
| `hos --matmul-bench -m model.gguf` | Benchmark the model's largest matmul (the output projection). |
| `hos --batch-bench` / `--batch-attn-test` | Batched-throughput and batched-attention microbenchmarks. |
| `hos --gpu-test -m model.gguf` | Validate and benchmark the GPU matmul kernel against the CPU (bit-identical check). |

### 5.6 Fine-tune and train

The engine has its own reverse-mode autograd, so training runs without any
external framework. See [FINETUNING.md](FINETUNING.md) for the full workflow.

| Command | Purpose |
|---|---|
| `hos --finetune -m model [--corpus f] [--opt adamw\|sgd] [-o out.hos]` | Fine-tune a loaded model on a corpus and emit a `.hos` capsule. |
| `hos --finetune-check -m model` | Prove the autograd-built Llama forward matches the inference forward (gradient correctness gate). |
| `hos --train-lm` | Train a small char-level LM from scratch through the full engine (attention, RMSNorm, FFN, cross-entropy, AdamW). |
| `hos --gen-hos file.hos` | Load a char-LM trained by `--train-lm` and generate. Seed with `-p`, sample `-n` chars. |
| `hos --train-spec` | Training-path spec/self-check. |
| `hos --train-gpu-test` | Confirm the GPU matmul forward and backward produce the same training result as the CPU. |

Relevant environment variables: `FT_LR`, `FT_GPU`, `FT_GROW_FFN`, `FT_FLWR`,
`FT_VIZ`, `LM_STEPS`, `LM_GPU` (see the table in section 8).

### 5.7 RGA adapters (genomes)

RGA is HOS's parameter-efficient adaptation scheme: small, ownable "genome"
adapters over a frozen base, an alternative to LoRA. These commands are the
experiment and demonstration surface.

| Command | Purpose |
|---|---|
| `hos --peft --method lora\|rga -m model [--corpus f] [-o out.hos]` | Train an adapter (LoRA or RGA) on a corpus and save it. |
| `hos --peft-demo -m model` | The plainest demonstration: train one RGA genome and show it working. |
| `hos --peft-compare -m model` | Matched-budget multi-domain comparison of RGA vs LoRA. |
| `hos --peft-heldout -m model` | Train a shared RGA bank and test best-of-both generalization. |
| `hos --peft-interference -m model` | Continual-learning interference test. |
| `hos --peft-compose -m model` | Whether bank diversity helps composition. |
| `hos --peft-clonal -m model` | Clonal selection: can RGA invent a held-out capability. |
| `hos --peft-grow -m model` | The full loop in one run. |
| `hos --peft-replay -m model` | Consolidation: distill a newly-learned domain back into the base. |
| `hos --peft-recombine -m model` | Train two genomes on two domains and recombine them. |
| `hos --peft-fuse -m base.hos` | Two-parent fusion: build two specialist parents over one frozen base, fuse them, and mint a runnable fused `.hos` body. Pick a genome at run time with `--genome <name>`. |

Relevant environment variables: `PEFT_LR`, `PEFT_T`, `PEFT_LAMBDA`, `PEFT_GPU`,
`FUSE_OUT`.

### 5.8 Gemma 3/4 multimodal

HOS runs the Google Gemma text decoder natively, including a native image path.

| Command | Purpose |
|---|---|
| `hos --gemma4 -m <dir> [-p "text"] [--ids csv] [-n k] [--q4k\|--q5k\|--q6k] [--gpu]` | Run a Gemma 3/4 checkpoint. |
| `hos --gemma4 --image <path> -p "<question>"` | Native image-plus-text prompting. |
| `hos --gemma4-bench [-m dir] [--q4k\|--q5k\|--q6k] [--gpu] [-n k]` | Load-time and throughput benchmark. |
| `hos --gemma4-kv-check` | Correctness gate for the KV cache. |
| `hos --gemma4-prefill-check` | Parity and speed gate for the batched GPU prefill. |
| `hos --gemma-tok-selftest` | Validate the native Gemma tokenizer against reference vectors. |

K-quant selection for the Gemma loader is via `--q4k` / `--q5k` / `--q6k`, which
map to the `HOS_GEMMA4_QUANT` variable.

### 5.9 Diagnostics and verification

| Command | Purpose |
|---|---|
| `hos --verify-against <hf_dir> -m model.gguf` | Audit a quantized GGUF against its original HuggingFace checkpoint. |
| `hos --interp-check` | Interpretability / internal-state check. |
| `hos --qwen35-check` | Detect and inspect a Qwen3.5 hybrid (Gated-DeltaNet + attention) model's structure. |
| `hos --deltanet-test` | Gated delta-net (Qwen3.5 hybrid) unit path. |
| `hos --hos-selfrun` / `--selfrun-tf` | Load-and-run self-tests for the `.hos` path. |

### 5.10 Engine capability demos

These prove specific engine capabilities end to end. They take no model file.

| Command | Shows |
|---|---|
| `hos --autograd-demo` | The reverse-mode autograd training a 2-layer MLP on XOR. |
| `hos --nn-demo` | A small MLP trained via the `nn::Module` API. |
| `hos --rnn-demo` | An LSTM and a GRU on a sequential task. |
| `hos --vq-demo` | Classification through a discrete E8-quantized bottleneck (QAT primitive). |
| `hos --op-demo` | Ownable, knowledge-retaining operators: add a domain with the base frozen. |
| `hos --mem-demo` | Local, no-bleed editing in a discrete addressable memory. |
| `hos --fixed-scale` | Fixed-address slot memory: learned values and pointer. |
| `hos --contradict-scale` | Whether the addressing-collision tail grows with scale. |
| `hos --hos-demo` / `--lm-demo` | End-to-end lifecycle and tiny-LM demonstrations. |

---

## 6. Model formats

| Format | What it is |
|---|---|
| `.gguf` | Standard GGUF. Loaded and parsed natively; weights stay in their quantized form and dequantize inside the kernels. |
| `.hos` | The native capsule: weights plus optimizer state plus a self-describing card (arch spec, provenance, lineage, training history). Runnable directly. |
| `.flwr` | Identical bytes to `.hos`; the app's label for the same capsule. |

Convert with `hos --to-hos model.gguf -o model.hos [--quantize q8_0]` and inspect
with `hos --hos-info model.hos`. The byte-level spec is in
[HOS_FORMAT.md](HOS_FORMAT.md).

---

## 7. Supported architectures

| Family | Backend | Notes |
|---|---|---|
| Llama / Mistral / Qwen2 / Qwen2.5 / SmolLM2 | CPU + Metal GPU (fused) | Bit-identical CPU and GPU output. |
| Gemma 2 / Gemma 3 / Gemma 4 | CPU (+ Gemma multimodal path) | Embed scale, GeGLU, attention and final soft-cap, sandwich norms, native image path. |
| Phi-3 | CPU | Fused QKV and gate-up tensors split on load. |
| OLMoE | CPU | Mixture-of-experts: top-k routing, experts kept quantized. |
| Qwen3.5 (hybrid Gated-DeltaNet + attention) | CPU, GPU | Native runner with MTP speculative decoding and vision; GGUF or minted `.hos`. |

The engine is architecture-aware: it auto-selects RoPE style, attention bias,
tied embeddings, and per-arch norm and activation quirks from the model's
metadata. The tokenizer is byte-exact and dispatches the right BPE variant
(`gpt-2` / `qwen2` / `llama-bpe`) from that metadata. Loading is fully fallible:
a missing, malformed, or unsupported model returns a clean error, not a panic.

---

## 8. Environment variables

| Variable | Applies to | Effect |
|---|---|---|
| `HOS_MODELS_DIR` | both | Extra directory to resolve bare model names from. |
| `HOS_THREADS` | engine | CPU worker-thread count. |
| `HOS_PREFILL` | engine | Prefill batching control. |
| `HOS_QRESIDENT` | engine | Keep weights resident at their quantized size. |
| `HOS_PROF` / `HOS_CONV_PROFILE` | engine | Profiling and convolution-profile toggles. |
| `HOS_METAL_CONV` / `HOS_METAL_DEPTHWISE` | Metal | Metal convolution path selectors. |
| `HOS_GEMMA4_QUANT` | Gemma | K-quant selection (also set by `--q4k` / `--q5k` / `--q6k`). |
| `HOS_MODEL` | engine | Default model path. |
| `FLWR_STORE` / `HOS_STORE` | flwr | Model store location. |
| `FLWR_CHATS` | flwr | Chat-transcript directory. |
| `FLWR_CONTEXT_TOKENS` | flwr | Approximate prompt budget before compacting long conversations. |
| `FLWR_RECENT_TURNS` | flwr | Recent messages kept verbatim after compaction. |
| `FLWR_MEMORY_BATCH_MESSAGES` | flwr | Messages per stable summary receipt. |
| `FT_LR` / `FT_GPU` / `FT_GROW_FFN` / `FT_FLWR` / `FT_VIZ` | fine-tune | Fine-tuning knobs. |
| `LM_STEPS` / `LM_GPU` | train-lm | Char-LM training knobs. |
| `PEFT_LR` / `PEFT_T` / `PEFT_LAMBDA` / `PEFT_GPU` | RGA | Adapter-training knobs. |
| `FUSE_OUT` | peft-fuse | Output path for the fused capsule. |

---

## 9. Troubleshooting

- Unknown-argument error mentioning brackets: the `[ ]` in this manual mean
  "optional." Do not type the brackets.
- `--gpu` seems ignored: GPU is Metal-only. On Windows and Linux it runs on CPU
  by design.
- A bare model name is not found: set `HOS_MODELS_DIR`, or pass a full path, or
  `flwr pull` the model into the store first. `flwr list` shows what resolves.
- A model fails to load: run `hos --info -m <model>` to see whether the
  architecture and quant types are supported (section 7).

---

## 10. Desktop apps and platforms

The same engine runs everywhere; only a thin native shell differs per platform.
`flwr serve` hosts the chat UI on `http://127.0.0.1:<port>`, and each shell opens
that UI in a chromeless window using the OS's own browser engine. There is no
bundled runtime.

| Platform | How to run the app | Acceleration |
|---|---|---|
| macOS | `bash desktop/macos/build.sh` builds `~/Applications/Flwr.app` (WKWebView). | Metal GPU with `--gpu`. |
| Windows | Double-click `desktop/windows/flwr.cmd`, or `powershell -File desktop/windows/flwr.ps1` (Edge / WebView2). | CPU, AVX2 on x86. |
| Any browser / ChromeOS | `flwr serve <model>` then open `http://127.0.0.1:11434`. Chrome or Edge can install it as an app window. | Whatever the host OS provides. |

Prerequisite for all three: `cargo install --path . --bin flwr --bin hos`. Per-shell
configuration (`FLWR_MODEL`, `FLWR_PORT`, `FLWR_BIN`) and details are in
[../desktop/README.md](../desktop/README.md). Double-click file associations for
`.flwr` / `.hos` models are in [../packaging/README.md](../packaging/README.md).

## 11. Library

Embed the engine in your own Rust program: `use hos::...`. The API surface
(loading, generation, chat templating, the `.hos` codec) is documented in
[LIBRARY.md](LIBRARY.md).

---

## 12. Deep-dive documents

| Document | Covers |
|---|---|
| [FLWR.md](FLWR.md) | The `flwr` app: full command and API reference, the model store, troubleshooting. |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Engine internals: layout, kernels, backends. |
| [HOS_FORMAT.md](HOS_FORMAT.md) | Byte-level `.hos` capsule specification. |
| [LIBRARY.md](LIBRARY.md) | Embeddable library API. |
| [FINETUNING.md](FINETUNING.md) | Fine-tuning and training workflow. |
| [EXPLAINER.md](EXPLAINER.md) / [DIAGRAM.md](DIAGRAM.md) | Conceptual overview and diagrams. |
</content>
</invoke>
