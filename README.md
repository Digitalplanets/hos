# HOS

A from-scratch local LLM inference engine. Loads GGUF or other models and
runs them on CPU (multithreaded) or Apple Silicon GPU (Metal), with weights kept
in their native quantized form and dequantized inside the compute kernels.

No `llama.cpp`, no Python runtime. One static binary with a library you can build on.

> 📖 **New here? Read [The HOS Book](docs/HOS_BOOK.md).** It opens with how HOS
> compares to `llama.cpp`, llm.c, transformers, and PyTorch, then walks from a
> five-minute quick start through the engine internals and the research the whole
> thing is built on.

## Status

Runs six transformer families today:

| Current Model Families | Backend | Notes |
|---|---|---|
| Llama / Mistral / Qwen2 / Qwen2.5 / SmolLM2 | CPU + Metal GPU (fused) | bit-identical CPU↔GPU |
| Gemma-2 | CPU | embed scale, GeGLU, attn/final soft-cap, sandwich norms |
| Phi-3 | CPU | fused QKV / gate-up tensors split on load |
| OLMoE | CPU | mixture-of-experts: top-k routing, experts kept quantized |

The engine is architecture-aware — it auto-selects per model: RoPE style
(Llama interleaved vs NEOX), attention bias (Qwen2 has it; Llama doesn't),
tied embeddings, and the per-arch norm/activation quirks above. Output is
verified bit-identical between the CPU and GPU backends on the GPU families.

The tokenizer is **byte-exact**: it reproduces the GPT-2-family pre-tokenization
and dispatches the right variant (`gpt-2` / `qwen2` / `llama-bpe`) from the
model's metadata. Loading and parsing are "fully fallible", meaning a missing,
malformed, or unsupported model returns a clean error instead of panicking.

Hybrid SSM/Mamba models (Qwen3.5 `qwen35`) are detected and inspectable
(`hos --qwen35-check`); an experimental CPU+GPU runner exists but the standard
transformer path above is the supported one.

On an Apple M4 Max, the GPU backend runs Llama-3.2-1B (Q4_K_M) at ~6× the CPU speed,
with weights resident at their compressed size. Reproduce throughput and quality
on your own machine with `hos --bench` and `hos --perplexity` (below).

## Build & install

```sh
cargo install --path .     # installs `hos` to ~/.cargo/bin
```

## Usage

```sh
# generate (CPU)
hos -m model.gguf -p "The capital of France is" -n 64

# generate (Apple Silicon GPU)
hos --gpu -m model.gguf -p "The capital of France is" -n 64 --temp 0.7

# inspect a model's architecture / tensors / quant types
hos --info -m model.gguf

# throughput benchmark (fixed prompt): prefill + greedy decode tok/s
hos --bench -m model.gguf [--gpu]

# perplexity over held-out text (built-in passage, or pass a file)
hos --perplexity -m model.gguf
hos --perplexity corpus.txt -m model.gguf

# validate + benchmark the GPU matmul kernel against CPU
hos --gpu-test -m model.gguf
```

## `flwr` — the chat app on top of the engine

The `hos` commands above drive the raw completion engine. `flwr` is a **separate
app** built on the `hos` *library* that turns a model into something you talk to
— what `ollama` is to `llama.cpp`. It is its own binary; it does not change how
`hos` works. The chat template is applied from the engine (`Engine::chat`), and
the HTTP server is hand-rolled over `std::net`, so HOS gains no web-framework
dependency.

```sh
# pull a model into the provenance store (HF repo or a direct .gguf URL)
flwr pull HuggingFaceTB/SmolLM2-135M-Instruct       # downloads via system curl
flwr list                                            # what's in the store
flwr show SmolLM2-135M-Instruct                      # the provenance card
flwr cp SmolLM2-135M-Instruct my-variant             # duplicate (keeps lineage)
flwr cp ./my-finetune-dir my-finetune                # import a local HF dir / .gguf
flwr quantize big.gguf big-q8 --type q8_0            # derive a smaller Q8_0 (keeps lineage)
flwr rm my-variant                                   # delete from the store

# interactive chat REPL — by store name, bare model name, or path
flwr run SmolLM2-135M-Instruct                       # resolved from the store
flwr run Llama-3.2-1B-Instruct-Q4_K_M.gguf --gpu
flwr run Qwen2.5-0.5B-Instruct-Q4_K_M.gguf -p "Name three primary colors."  # one-shot

# OpenAI-compatible HTTP daemon (default 127.0.0.1:11434)
flwr serve Qwen2.5-0.5B-Instruct-Q4_K_M.gguf --port 11434
# then open http://127.0.0.1:11434/ in a browser for a built-in chat UI, or:
curl http://127.0.0.1:11434/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"stream":true}'
```

**The store is provenance-native.** `flwr pull` downloads into `~/.hos/store/`
(downloading is an interop seam, so it shells out to the system `curl` — no
HTTP+TLS crate dependency) and writes a `manifest.json` carrying, in the spirit
of the `.hos` lineage card: the source repo + revision, a content hash per file,
and a combined `identity` hash for the whole artifact. A pulled model is never
an anonymous blob — `flwr show` prints its ancestry. `flwr cp` either duplicates
a store entry (preserving the content `identity` and recording a `copied_from`
lineage edge) or **imports a local path** — an HF checkpoint dir or a `.gguf`
file — under a new name with `source: local:<path>`. (The `.hos` converter takes
the same flag: `hos --to-hos model.gguf --quantize q8_0` for a compact capsule.)
`flwr quantize` derives a
smaller **Q8_0** GGUF from a GGUF source (HOS's own from-scratch quantizer +
GGUF writer — metadata copied verbatim, verified to preserve perplexity), and
the derived model records a lineage edge back to its parent. `flwr rm` deletes
one. `flwr serve` handles
connections concurrently (a slow client never blocks others; `GET /` and
`/v1/models` answer instantly); generation itself is serialized through the one
resident engine, since there is a single KV cache.

The engine ships **canned per-family chat templates** (`src/chat.rs`), detected
from the special tokens actually present in the vocabulary — ChatML (Qwen /
SmolLM2 / OLMoE), Llama-3 headers, Gemma `<start_of_turn>`, Phi-3 `<|user|>`,
Mistral `[INST]` — rather than parsing a Jinja `chat_template` string, keeping
the dependency-free stance. Turn-terminators (`<|im_end|>`, `<|eot_id|>`,
`<end_of_turn>`, `<|end|>`) are detected as stop tokens, so replies end cleanly.
`flwr serve` speaks the OpenAI `/v1/chat/completions` API (streaming via SSE or
one-shot JSON) plus `/v1/models`, so existing chat clients and IDE plugins point
at it unchanged. Build installs both binaries: `hos` (engine) and `flwr` (app).
The served page is a built-in **brutalist chat UI** with an in-app theme switcher
and **auto-saved, provenance-bearing chat transcripts** (`~/.hos/chats/*.json`,
each recording which model produced it);
`bash desktop/macos/build.sh` wraps it in a **native macOS app** (`WKWebView`,
Apple frameworks only — no packages). **Full command + API reference and
troubleshooting:** [docs/FLWR.md](docs/FLWR.md).

HOS also has a **native model format** (`.hos`) — weights + optimizer state + a
self-describing "card" (arch spec, provenance, lineage, training history). It is
**runnable**: `hos --to-hos model.gguf -o model.hos [--quantize q8_0]` mints a
capsule and `hos -m model.hos` / `flwr run model.hos` load and run it directly
(verified bit-identical to the source GGUF). Inspect one with
`hos --hos-info file.hos`. See [docs/HOS_FORMAT.md](docs/HOS_FORMAT.md) for the
byte-level spec and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for internals.

Models are resolved by bare name from `$HOS_MODELS_DIR`, `~/Documents/hos/models`,
or `~/.hos/models`, or pass a full path with `-m`. Set `HOS_MODEL` for a default.

Flags: `-m/--model`, `-p/--prompt`, `-n/--n-predict`, `--temp` (0 = greedy),
`--seed`, `--gpu`, `--info`, `--gpu-test`.

## Library

```rust
use std::path::Path;

// load returns hos::Result<Engine> — never panics on a bad/unsupported model
let mut eng = hos::Engine::load(Path::new("model.gguf"), true /* gpu */)?;

// prompt, max_tokens, temp, top_k, top_p, rep_penalty, repeat_last_n, seed, on_token
eng.generate("Once upon a time", 128, 0.8, 40, 0.95, 1.1, 64, 42,
             |piece| print!("{piece}"));

let ids = eng.tok.encode("held-out text", true);
let (scored, mean_nll, ppl) = eng.perplexity(&ids);
```

**Full integration reference for embedding the engine:**
[docs/LIBRARY.md](docs/LIBRARY.md) — the public crate API, error handling,
threading/perf, and examples.

A PyO3 binding lives in `py/` — `maturin develop` there gives you
`hos.Engine(path, gpu=True).generate(...)`, `.perplexity(...)`, `.tokenize(...)`,
and `hos.inspect_hos(path)` for `.hos` capsules.

## Loading raw HuggingFace checkpoints

HOS reads a HuggingFace checkpoint folder directly — `config.json` +
`*.safetensors` (single-file or sharded) + `tokenizer.json` — with **no GGUF, no
`llama.cpp`, no `transformers`** anywhere in the pipeline. Point any command at
the directory:

```sh
hos -m ./Qwen2.5-0.5B-Instruct -p "List three colors:" -n 20 --gpu
hos --perplexity -m ./SmolLM2-135M-Instruct
```

It loads BF16/F16/F32 weights (folded to F16 for the matmul path), replicates the
converter's work in memory — GGUF-style tensor renaming, the Llama/Mistral Q/K
RoPE permutation, Gemma's `(1 + w)` norm fold — and runs every architecture the
GGUF path supports. Verified against the equivalent GGUFs: SmolLM2-135M (Llama)
matches its GGUF perplexity; Qwen2.5-0.5B (Qwen2) runs *lower* perplexity at F16
than its Q4 GGUF, as expected. Both tokenizer families are read natively:
byte-level BPE from `tokenizer.json` (Llama-3 / Qwen2 / SmolLM2 / OLMoE) and
SentencePiece from the `tokenizer.model` protobuf (Gemma / Llama-2 / Mistral /
Phi-3) — the latter parsed by a small hand-rolled reader, token-for-token
identical to the GGUF tokenizer (verified on Phi-3).

### Provenance: ingest & audit

Loading is interop; what HOS does *with* an ingested model is the part that's
unique. Because the `.hos` format carries content-hash identity + lineage, HOS
can give an external model an ancestry, and — being the one engine that loads
both the HF source and a GGUF — it can *audit a conversion*:

```sh
# mint a .hos capsule from an HF checkpoint: weights + provenance + a content-hash
# lineage edge back to the source artifact
hos --ingest ./SmolLM2-135M-Instruct -o smollm2.hos               # runnable capsule
hos --ingest ./SmolLM2-135M-Instruct -o smollm2-q8.hos --quantize q8_0   # q8_0|q4_0|q5_0
hos -m smollm2-q8.hos -p "..."  # run it — verified bit-identical to the HF source
hos --hos-info smollm2-q8.hos   # id, lineage, the arch card

# audit a quantized GGUF against its original HF checkpoint:
# same architecture? how much did quantization actually cost?
hos --verify-against ./Phi-3-mini-4k-instruct -m Phi-3-mini-4k-instruct-q4.gguf
#   -> architecture: all fields match; quantization cost: +78.5%; verdict CONSISTENT
```

The audit makes quant sensitivity measurable per model — e.g. Phi-3-mini Q4 costs
**+78.5%** perplexity vs its F16 source, while Qwen2.5 Q4 costs only **+3.2%**.

The `--quantize q8_0` flag stores the capsule's weight/embedding tensors as Q8_0
(norms/biases stay f32), so a `.hos` can be a **smaller, still-self-describing,
still-lineaged** artifact — HF → quantized `.hos` with no GGUF in the loop
(SmolLM2-135M: 538 MB → 143 MB). Quantized tensors decode to f32 on load, so the
capsule behaves identically to its f32 twin.

## Training & finetuning

HOS doesn't just run models — it trains them, on the same from-scratch autograd
core (`tensor.rs`), with no ML dependencies. The forward used for training is
verified **byte-identical** to the inference forward before any gradients are
trusted.

```sh
# full-parameter finetune a real pretrained model on your own text (GGUF or HF)
hos --finetune -m model.gguf --corpus mytext.txt -o model-ft.hos
hos --finetune-check -m model.gguf        # prove the trainable forward == inference forward

# parameter-efficient finetuning on a frozen base (only the adapter trains)
hos --peft --method lora -m model.gguf    # low-rank adaptation (the baseline)
hos --peft --method rga  -m model.gguf    # Regulatory Genome Adapters (genome-gated)
```

Two PEFT methods share one frozen-base substrate: **LoRA** (low-rank edits) and
**RGA** — a shared bank of tiny modules gated by a compact per-task *genome*
(the M3 genome mechanism as PEFT). A research harness compares them head-to-head:
`--peft-compare` (multi-domain), `--peft-interference` (continual learning),
`--peft-recombine` (genome crossover), `--peft-heldout` (new-domain adaptation).
Adapters and finetunes save as `.hos` capsules whose lineage points to the base.

## Supported quantization

`F32`, `F16`, `Q8_0`, `Q4_0`, `Q5_0`, and the K-quants `Q4_K`, `Q5_K`, `Q6_K`
(dequantized in-kernel on GPU). Unsupported types fail with a clear message.

## Testing

```sh
cargo test       # autograd gradient-checks, GGUF codec + fuzz, .hos round-trip, perplexity/bench
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check
```

CI runs all three on every push (`.github/workflows/ci.yml`). Tests that need a
GPU device or a model file self-skip when unavailable.

## Roadmap

- Hybrid SSM/Mamba architectures (e.g. Qwen3.5) — finish the state-space path
- GPU-resident path for Gemma-2 / Phi-3 / MoE (CPU-only today)
- Faster GPU kernels (better attention parallelism, fewer dispatches; vectorized matvec)

Done since the first cut: MoE routing (OLMoE), Gemma-2 + Phi-3, keeping MoE
experts quantized in memory (dequant only the routed experts per token), a
self-describing `.hos` format whose architecture spec the interpreter can run
directly (`interp::run_llama_from_spec`, verified byte-equal to the hand-coded
pass), **native HuggingFace `safetensors` loading** (above) — HOS no longer
needs `llama.cpp`/`transformers` to ingest a model — **real-model
finetuning + PEFT** (full finetune, LoRA, and the RGA genome adapter), all on the
hand-written autograd core, and a **chat layer**: per-family templating
(`Engine::chat`) plus **`flwr`**, a separate app over the `hos` library that
gives you `flwr run` (REPL) and `flwr serve` (OpenAI-compatible HTTP) — the
usable-tool layer on top of the engine, with no web framework dependency — and a
**GGUF writer + Q8_0 quantizer** (`gguf_write.rs`), so HOS now encodes as well as
decodes: `flwr quantize` derives a smaller GGUF and preserves perplexity.

## License

NA'AT Keystone License (Open Source)
