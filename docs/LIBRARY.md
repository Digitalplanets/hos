# HOS Library Reference

`hos` is a from-scratch local LLM engine exposed as a Rust library. This document
is the integration reference for developers embedding `hos` in their own
applications: the public API, its semantics, error handling, threading and
performance characteristics, and stability expectations.

It complements two neighbours: **[ARCHITECTURE.md](ARCHITECTURE.md)** (internals
and design) and **[FLWR.md](FLWR.md)** (the end-user `flwr` app built on this
library).

- **Crate name:** `hos`
- **Edition:** 2021
- **Dependencies:** `memmap2`, `half`, `rayon`, `metal`, `serde`, `serde_json`
  — none of them a machine-learning framework.
- **Platforms:** macOS (Apple Silicon GPU via Metal) and any platform for the
  CPU path. The GPU backend is Metal-only; CPU is portable.

---

## Contents

1. [Adding the dependency](#1-adding-the-dependency)
2. [Quick start](#2-quick-start)
3. [Core concepts](#3-core-concepts)
4. [`Engine` — the public API](#4-engine--the-public-api)
5. [Sampling](#5-sampling)
6. [Tokenizer](#6-tokenizer)
7. [Chat templating](#7-chat-templating)
8. [Model sources: GGUF & HuggingFace](#8-model-sources-gguf--huggingface)
9. [Error handling](#9-error-handling)
10. [Concurrency & threading](#10-concurrency--threading)
11. [Performance & memory model](#11-performance--memory-model)
12. [Advanced: autograd, finetuning, PEFT, formats](#12-advanced)
13. [Python bindings](#13-python-bindings)
14. [Stability & versioning](#14-stability--versioning)

---

## 1. Adding the dependency

`hos` is not published to crates.io. Depend on it by path or git:

```toml
[dependencies]
hos = { path = "../hos" }            # local checkout
# or
hos = { git = "https://your.host/hos", rev = "<commit>" }
```

The crate builds a library target plus two binaries (`hos`, `flwr`); depending on
the crate links only the library. Building the Metal backend requires the Apple
toolchain; on non-macOS targets the GPU path is unavailable and `gpu: false`
should be used.

---

## 2. Quick start

```rust
use std::path::Path;

fn main() -> hos::Result<()> {
    // load() never panics on a bad/unsupported model — it returns Err.
    // Accepts a .gguf file OR a HuggingFace checkpoint directory.
    let mut eng = hos::Engine::load(Path::new("model.gguf"), /* gpu */ true)?;

    // Raw completion. The closure receives each decoded piece as it streams.
    // (prompt, max_tokens, temp, top_k, top_p, rep_penalty, repeat_last_n, seed, on_token)
    eng.generate("Once upon a time", 128, 0.8, 40, 0.95, 1.1, 64, 42, |piece| {
        print!("{piece}");
    });

    // Chat (applies the model's detected chat template).
    let convo = [
        hos::chat::Message::new("system", "You are concise."),
        hos::chat::Message::new("user", "Name three primary colors."),
    ];
    eng.chat(&convo, 128, 0.7, 40, 0.95, 1.1, 64, 42, |piece| print!("{piece}"));

    // Quality metric over held-out text.
    let ids = eng.tok.encode("the quick brown fox", true);
    let (scored, mean_nll, ppl) = eng.perplexity(&ids);
    println!("scored={scored} nll={mean_nll:.3} ppl={ppl:.2}");
    Ok(())
}
```

---

## 3. Core concepts

| Concept | Type | Role |
|---|---|---|
| Engine | `hos::Engine` | Owns a model, tokenizer, optional GPU runner, and KV-cache state. The one handle most callers need. |
| Model source | `hos::model::ModelSource` | Trait abstracting where weights come from. Implemented by `gguf::Gguf` and `hf::HfModel`. |
| Tokenizer | `hos::tokenizer::Tokenizer` | Byte-exact BPE / SentencePiece encode & decode. |
| Config | `hos::model::Config` | Resolved architecture + hyperparameters. |
| Arch | `hos::model::Arch` | The detected architecture family. |

`Engine::load` is the front door: it inspects the path, builds the right
`ModelSource` and `Tokenizer`, selects CPU or GPU, and returns a ready `Engine`.

---

## 4. `Engine` — the public API

```rust
pub struct Engine {
    pub model: model::Model,        // weights + Config (see model.rs)
    pub tok:   tokenizer::Tokenizer,// the model's tokenizer
    /* private: GPU runner, KV-cache State, position cursor */
}
```

### `Engine::load`

```rust
pub fn load(path: &Path, gpu: bool) -> hos::Result<Engine>
```

Loads a model. If `path` is a directory containing `config.json`, it is read as a
HuggingFace checkpoint (`*.safetensors` + tokenizer); a `.hos` capsule (magic
`HOSF`) carrying engine metadata is loaded via its own source (CPU); otherwise it
is parsed as a GGUF file. `gpu` requests the Metal backend, which is used **only** for the
GPU-supported arches (Llama / Mistral / Qwen2 families); other arches and
non-macOS builds silently fall back to CPU. Returns `Err` on any malformed,
missing, or unsupported model — never panics.

### `Engine::generate`

```rust
pub fn generate(
    &mut self,
    prompt: &str,
    max_tokens: usize,
    temp: f32,            // 0.0 = greedy/argmax
    top_k: usize,
    top_p: f32,
    rep_penalty: f32,     // 1.0 = off
    repeat_last_n: usize, // window the penalty applies over
    seed: u64,
    on_token: impl FnMut(&str),
) -> usize                // number of tokens generated
```

Raw text completion. Encodes `prompt` (with BOS), prefills, then samples up to
`max_tokens`, calling `on_token` with each decoded UTF-8 piece. Stops at EOS or
the model's context length. **Continues from the engine's current position** —
call [`reset`](#enginereset) first for an independent generation.

### `Engine::chat`

```rust
pub fn chat(
    &mut self,
    msgs: &[chat::Message],
    max_tokens: usize,
    temp: f32, top_k: usize, top_p: f32,
    rep_penalty: f32, repeat_last_n: usize, seed: u64,
    on_token: impl FnMut(&str),
) -> usize
```

Runs one assistant turn over a conversation. Detects the model's chat dialect,
renders the full message history through the matching template, **resets the KV
cache**, and generates until a turn-terminator or EOS. For multi-turn, keep a
`Vec<Message>`, push the model's reply as an `"assistant"` message, and call
again — the full history is re-rendered each call.

### `Engine::generate_from_ids`

```rust
pub fn generate_from_ids(
    &mut self,
    ids: &[u32],
    max_tokens: usize,
    temp: f32, top_k: usize, top_p: f32,
    rep_penalty: f32, repeat_last_n: usize, seed: u64,
    stops: &[u32],
    on_token: impl FnMut(&str),
) -> usize
```

Lower-level generator: prefills caller-supplied token `ids` (no BOS added) and
stops at `max_tokens`, the context limit, or any id in `stops`. Continues from
the current position (does not reset). Use this when you control tokenization or
templating yourself.

### `Engine::reset`

```rust
pub fn reset(&mut self)
```

Clears the KV cache and resets the position cursor so the next call starts fresh.

### `Engine::chat_family`

```rust
pub fn chat_family(&self) -> chat::ChatFamily
```

The chat dialect HOS detected for the loaded model (see [§7](#7-chat-templating)).

### `Engine::perplexity`

```rust
pub fn perplexity(&mut self, ids: &[u32]) -> (usize, f64, f64)
// -> (scored_tokens, mean_nll_nats, perplexity)
```

Resets the cache, feeds `ids` left-to-right, and scores each next-token
prediction with an exact log-softmax. The standard correctness/quality metric.

### `Engine::bench`

```rust
pub fn bench(&mut self, prompt: &str, decode_tokens: usize) -> Bench
```

Measures prefill and decode throughput.

```rust
pub struct Bench {
    pub prefill_tokens: usize,
    pub prefill_secs: f64,
    pub decode_tokens: usize,
    pub decode_secs: f64,
}
impl Bench {
    pub fn prefill_tps(&self) -> f64; // tokens/sec
    pub fn decode_tps(&self) -> f64;
}
```

---

## 5. Sampling

`generate`/`chat` sample internally, but the sampler is also public for custom
decode loops:

```rust
pub fn hos::sample(
    logits: &[f32],
    temp: f32,            // 0.0 => argmax; ignores top_k/top_p
    top_k: usize,
    top_p: f32,
    rep_penalty: f32,
    recent: &[u32],       // recent tokens for the repetition penalty
    rng: &mut u64,        // xorshift state; advance with next_rand
) -> u32

pub fn hos::next_rand(state: &mut u64) -> f32  // uniform [0,1)
```

Order of operations: repetition penalty → temperature → top-k → top-p →
sample. With `temp == 0.0` the result is deterministic argmax.

---

## 6. Tokenizer

```rust
pub struct Tokenizer {
    pub id_to_token: Vec<String>,
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    /* private maps */
}
```

| Method | Signature | Notes |
|---|---|---|
| `encode` | `fn encode(&self, text: &str, add_bos: bool) -> Vec<u32>` | Byte-exact. Does **not** split on special tokens (see `special_id`). |
| `decode` | `fn decode(&self, ids: &[u32]) -> String` | Lossy UTF-8 of the concatenated bytes. |
| `decode_into` | `fn decode_into(&self, id: u32, out: &mut Vec<u8>)` | Appends one token's raw bytes; for streaming. |
| `special_id` | `fn special_id(&self, s: &str) -> Option<u32>` | Look up a literal special/added token id (e.g. `"<\|im_start\|>"`). |
| `from_gguf` | `fn from_gguf(g: &Gguf) -> Result<Tokenizer>` | Build from GGUF metadata. |
| `from_hf` | `fn from_hf(dir: &Path) -> Result<Tokenizer>` | Build from `tokenizer.json` / `tokenizer.model`. |

The tokenizer reproduces GPT-2-family byte-level BPE (`gpt-2` / `qwen2` /
`llama-bpe` variants) and SentencePiece, dispatched from model metadata. It is
verified byte-for-byte against the reference tokenizers.

---

## 7. Chat templating

```rust
pub mod chat {
    pub struct Message { pub role: String, pub content: String }
    impl Message { pub fn new(role: &str, content: &str) -> Message; }

    pub enum ChatFamily { ChatMl, Llama3, Gemma, Phi3, Mistral, Plain }
    impl ChatFamily {
        pub fn detect(tok: &Tokenizer, arch: Arch) -> ChatFamily;
        pub fn label(self) -> &'static str;
    }

    pub fn build_ids(tok: &Tokenizer, fam: ChatFamily,
                     msgs: &[Message], add_generation_prompt: bool) -> Vec<u32>;
    pub fn stop_ids(tok: &Tokenizer, fam: ChatFamily) -> Vec<u32>;
}
```

`ChatFamily::detect` chooses the dialect from the special tokens present in the
vocabulary (falling back to the architecture). `build_ids` renders a conversation
to token ids by splicing special-token ids directly between BPE'd text — there is
no Jinja engine. `stop_ids` returns the dialect's turn-terminators plus EOS.
`Engine::chat` wires all three together; call them directly only for custom
prompt assembly.

---

## 8. Model sources: GGUF & HuggingFace

Both loaders implement one trait, so the engine is source-agnostic:

```rust
pub trait ModelSource {
    fn meta_str(&self, key: &str) -> Option<&str>;
    fn meta_u64(&self, key: &str) -> Option<u64>;
    fn meta_f32(&self, key: &str) -> Option<f32>;
    fn has(&self, name: &str) -> bool;
    fn dequant(&self, name: &str) -> Result<Vec<f32>>;          // tensor -> flat f32
    fn raw(&self, name: &str) -> Result<(&[u8], u32, usize)>;   // raw bytes + ggml type + n
}
```

**GGUF** (`hos::gguf`):

```rust
let g = hos::gguf::Gguf::open(Path::new("model.gguf"))?;
let weights: Vec<f32> = g.dequant("blk.0.attn_q.weight")?;
let arch = g.meta_str("general.architecture");
```

`Gguf` is a bounds-checked, mmap-backed parser (GGUF v2/v3). Supported quant
types: `F32`, `F16`, `Q8_0`, `Q4_0`, `Q5_0`, `Q4_K`, `Q5_K`, `Q6_K`. Unsupported
types return `HosError::UnsupportedQuant`. `gguf::bytes_for(ggml_type, n)`
computes the byte size of a tensor; `Gguf::dequant_into(bytes, ty, n, out)`
decodes into a caller buffer.

**Writing GGUF** (`hos::gguf_write`): the encode direction.

```rust
let src = hos::gguf::Gguf::open(Path::new("model-f16.gguf"))?;
let stats = hos::gguf_write::requantize_gguf(&src, Path::new("model-q8.gguf"),
                                             hos::gguf::GGML_Q8_0)?;
// stats.tensors / stats.quantized / stats.src_bytes / stats.out_bytes
```

`requantize_gguf` reads every tensor as f32, re-encodes the quantizable ones
(≥2-D, `dims[0] % 32 == 0`) to **Q8_0** while keeping 1-D tensors as F32, and
writes a valid GGUF with the source metadata copied **verbatim** (tokenizer
arrays included). Q8_0 is the only target today.

**HuggingFace** (`hos::hf`, `hos::safetensors`):

```rust
let hfm = hos::hf::HfModel::open(Path::new("./Qwen2.5-0.5B-Instruct"))?;
// hfm implements ModelSource; Engine::load uses this path automatically for dirs.
```

`HfModel` reads `config.json` + `*.safetensors` (single-file or sharded),
synthesizes GGUF-style metadata, maps HF tensor names to the engine's names,
applies the Llama/Mistral Q/K RoPE permutation and Gemma's `(1+w)` norm fold, and
folds BF16→F16. No `llama.cpp`/`transformers` involved.

To build an engine from a source you already hold, prefer `Engine::load`; the
generic `from_source` constructor is internal.

---

## 9. Error handling

All fallible APIs return `hos::Result<T>` = `std::result::Result<T, HosError>`.
Loading and parsing are fully fallible; HOS does not panic on bad input.

```rust
pub enum HosError {
    Io(std::io::Error),       // file open/read/write failed
    Format(String),           // bad magic / truncation / bad version
    MissingTensor(String),    // a required tensor was absent
    UnsupportedQuant(u32),    // quant type HOS does not decode
    UnsupportedArch(String),  // architecture HOS cannot run yet
    MissingMeta(String),      // required metadata key absent
    Spec(String),             // invalid interpreter arch spec
}
```

`HosError` implements `Display`/`Error` and `From<std::io::Error>`. Match on
variants for programmatic handling, or surface `Display` to users.

---

## 10. Concurrency & threading

- **`Engine` is single-threaded and stateful.** It owns a KV cache and a position
  cursor; `generate`/`chat`/`perplexity` take `&mut self`. Do **not** share one
  `Engine` across threads.
- **The GPU runner is not `Send`.** An `Engine` built with `gpu: true` must stay
  on the thread that created it. Move work, not the engine.
- **Recommended server pattern** (as used by `flwr serve`): keep the `Engine` on
  one owner thread and funnel requests to it over a channel; accept and parse
  connections on other threads. Tokens, ids, and parsed requests are all `Send`,
  so they cross thread boundaries while the engine does not.
- **For parallel generation, run independent `Engine`s** (each its own model
  load, its own memory). There is no shared-weights multi-session mode.

---

## 11. Performance & memory model

- **Weights stay quantized in memory** and are dequantized inside the compute
  kernels. An N-bit GGUF costs roughly its on-disk size in RAM, not the f32
  blow-up. MoE experts are dequantized only for the routed experts per token.
- **GGUF is mmap-backed** (`memmap2`); load is fast and pages in lazily.
- **CPU path** uses `rayon` for matmul parallelism. **GPU path** (Apple Metal)
  is verified bit-identical to CPU on the supported arches and runs materially
  faster (≈6× on Llama-3.2-1B Q4_K_M on an M4 Max).
- **Activations** on the inference fast-path are kept in `f32`/`f16` buffers; the
  KV cache grows with sequence length up to the model's context limit.
- Reproduce numbers on your hardware with `Engine::bench` and `Engine::perplexity`
  (or the `hos --bench` / `hos --perplexity` CLI).

---

## 12. Advanced

These modules are public but lower-level; see [ARCHITECTURE.md](ARCHITECTURE.md)
for design detail.

- **`hos::tensor`** — a from-scratch tensor type with reverse-mode autograd
  (tape of VJP closures) and an `AdamW` optimizer. The basis for all training.
  Ops include matmul, RMSNorm (`rmsnorm_eps`), configurable RoPE
  (interleaved/NEOX), `repeat_interleave_dim0` (GQA), `sigmoid`, softmax, etc.
- **`hos::finetune`** — `FtModel`: full-parameter finetuning of a real
  pretrained model, with a forward pass verified byte-identical to inference.
- **`hos::peft`** — `PeftModel`: parameter-efficient finetuning on a frozen base
  — LoRA and **RGA** (Regulatory Genome Adapters), with a multi-genome variant
  over a shared bank.
- **`hos::format`** — the self-describing `.hos` capsule: weights + optimizer
  state + a card (arch spec, provenance, content-hash lineage, training history).
  `save` writes raw f32; `save_quantized` stores the big weight/embedding tensors
  as Q8_0 (norms/biases stay f32) for a ~3.8× smaller capsule that still decodes
  to f32 on `load`. v2 format; v1 files still load.
- **`hos::interp`** — an interpreter that runs a forward pass directly from a
  `.hos` arch spec (`run_llama_from_spec`), verified byte-equal to the hand-coded
  pass.
- **`hos::model`** — `Config` (all hyperparameters), `Arch`, `Model`, `QExperts`,
  and `cpu_matmul`.

---

## 13. Python bindings

A PyO3 binding lives in `py/`. After `maturin develop`:

```python
import hos
eng = hos.Engine("model.gguf", gpu=True)
eng.generate("Once upon a time", ...)
eng.perplexity(...)
eng.tokenize(...)
hos.inspect_hos("capsule.hos")
```

The binding mirrors the core `Engine` surface; the Rust API above is the source
of truth.

---

## 14. Stability & versioning

- Crate version is `0.0.x`: pre-1.0, and the API may change between revisions.
  Pin to a specific commit/rev for reproducible builds.
- The **`Engine` surface** (`load`, `generate`, `chat`, `generate_from_ids`,
  `perplexity`, `bench`, `reset`) is the most stable and the recommended
  integration point.
- Lower-level modules (`tensor`, `peft`, `interp`, `format`) are evolving and may
  change shape; depend on them with that in mind.
- HOS does not panic on malformed input by contract; a panic from a load/parse
  path is a bug — please report it.

---

*See also: [ARCHITECTURE.md](ARCHITECTURE.md) for internals, [FLWR.md](FLWR.md)
for the end-user app, and the crate `README.md` for the CLI.*
