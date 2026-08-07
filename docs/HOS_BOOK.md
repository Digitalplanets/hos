# The HOS Book

**The engine you can open up.**

HOS is a from-scratch local LLM inference engine written in Rust, and **flwr** is
the app that sits on top of it. Together they load a model, run it on your CPU or
Apple Silicon GPU, and let you pull, serve, and chat with models you own all the
way down. No `llama.cpp`, no PyTorch, no Python in the hot path. One static
binary, plus a library you can build on.

This book is the full tour. If you just want to run a model, jump to
[How to use it](#1-how-to-use-it-five-minutes). If you want to understand why the
whole thing is built from scratch, start at the top.

---

## 0. One binary that does what a whole stack does

Most of the local-AI world is assembled from separate pieces: a training
framework, an inference runtime, a quantizer, a tokenizer library, a model format,
and a server, each from a different project, each a dependency you don't control.
HOS is the wager that one person can hold the whole thing in their head, in one
readable Rust workspace, and lose nothing that matters.

Here is where HOS sits next to the tools you already know:

| | what it is | deps | reads GGUF | own format + provenance | learnable (train / edit / fuse) | one static binary |
|---|---|---|---|---|---|---|
| **HOS + flwr** | from-scratch engine + app | none | ✅ | ✅ `.hos` capsule | ✅ | ✅ |
| **llama.cpp** | C/C++ inference runtime | ggml | ✅ | ✗ (anonymous blob) | ✗ run-only | ✅ |
| **llm.c** (Karpathy) | single-file C/CUDA GPT-2 trainer | cuDNN/CUDA | ✗ | ✗ | ✅ train-only, one arch | ✅ |
| **transformers** (HF) | Python model library | torch + many | via convert | ✗ | ✅ | ✗ (Python) |
| **PyTorch** | the framework everyone depends on | CUDA/BLAS/… | ✗ | ✗ | ✅ | ✗ |
| **GGUF** | a weights-plus-metadata file format | (a format) | is GGUF | ✗ no lineage | ✗ | (a format) |

Read the table honestly. Each of those projects wins on an axis: `llama.cpp` has
years of kernel tuning, PyTorch has infinite flexibility, transformers has the
ecosystem, llm.c is the most beautiful teaching artifact in the field. **HOS is
the only row that is all of those things at once** and adds the two nobody else
has: a model format that remembers where it came from, and an engine you can
actually open up and change.

The thesis in one line: **AI should stay a craft you can touch.** Not a stack of
someone else's black boxes glued together, but a single sovereign tool you can
read top to bottom, run offline, and grow.

**What "from scratch" actually means here.** The tensor type, the autograd graph,
the attention kernels, the RoPE variants, the GGUF reader and writer, the
byte-exact tokenizer, the quantizers, and the Metal GPU kernels are all in-tree
under `src/`. There is no PyTorch, no candle, no ggml, no BLAS, no CUDA runtime,
no `tokenizers` crate, and no Python at inference time. If `cargo` is on your
`PATH`, you have everything you need to build the whole thing.

---

## 1. How to use it (five minutes)

**Install.** One command installs both the `hos` engine and the `flwr` app.

```sh
# macOS / Linux
curl -fsSL https://www.flwr.systems/install.sh | sh

# Windows (PowerShell)
irm https://www.flwr.systems/install.ps1 | iex

# or from source, any OS (needs a Rust toolchain from rustup.rs)
cargo install --git https://github.com/Digitalplanets/hos
```

**Pull a model and run it.** A bare name resolves from the flwr model hub. No
account, no config, nothing leaves your machine except the download.

```sh
flwr pull flwr-bloom          # a 1.5B model, as a provenance-bearing .hos capsule
flwr run flwr-bloom           # chat in your terminal
flwr run flwr-bloom -p "Name three primary colors."   # one-shot
```

**Serve it.** An OpenAI-compatible HTTP API plus a built-in chat UI.

```sh
flwr serve flwr-bloom         # http://127.0.0.1:11434
```

Then open the address in any browser, or point an existing OpenAI client at it
unchanged:

```sh
curl http://127.0.0.1:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"stream":true}'
```

**The two binaries.** `flwr` is the friendly front door (pull / run / serve, a
provenance-keeping store, chat). `hos` is the engine underneath, exposed as its
own CLI for the raw and offline work: completion, format conversion,
quantization, benchmarking, and inspection. Same core, two faces.

```sh
hos -m model.gguf -p "The capital of France is" -n 64     # raw completion
hos --gpu -m model.gguf -p "..." --temp 0.7                # on the GPU
hos --info -m model.gguf                                    # arch / tensors / quant
hos --bench -m model.gguf [--gpu]                           # prefill + decode tok/s
```

That is the whole surface for a first session. The rest of the book is *why* each
of those works the way it does.

---

## 2. The engine

### 2.1 The stack, and what is not in it

HOS is one Rust workspace. The pieces you would normally import are written in
tree:

- **Tensors and autograd** (`src/tensor.rs`): the n-dimensional array type and the
  reverse-mode graph, with backward passes for the ops a transformer needs.
- **The model runner** (`src/model.rs`, `src/forward.rs`): attention, the MLP,
  norms, RoPE, and the fused quantized matmul that is the hot loop.
- **Formats** (`src/gguf.rs`, `src/gguf_write.rs`, `src/safetensors.rs`,
  `src/format.rs`): read GGUF, write GGUF, read HuggingFace safetensors, and the
  native `.hos` capsule.
- **Tokenizer** (`src/tokenizer.rs`): a byte-exact GPT-2-family BPE.
- **Quantizers** (`src/hos_quant.rs`, `src/hos_awq.rs`): K-quants, an AWQ-lite
  calibration path, and the perplexity harness that proves the result.
- **GPU** (`src/metal_be.rs`): hand-written Metal compute kernels, macOS-gated so
  the rest builds anywhere on CPU.

Removing the usual dependencies is not asceticism for its own sake. It is what
makes the engine comprehensible and portable: no version pins to chase, no
opaque kernel you cannot step into, and a single binary you can drop on a machine
with nothing installed.

### 2.2 The tokenizer is byte-exact

Getting tokenization *almost* right is the quiet way to get subtly wrong outputs
forever. HOS reproduces the GPT-2-family pre-tokenization exactly and dispatches
the right variant (`gpt-2`, `qwen2`, or `llama-bpe`) from the model's own
metadata. The regex pre-tokenizer, the byte-level mapping, and the merge ranks
match the reference, so a prompt tokenizes to the same ids the model was trained
on.

### 2.3 It reads the formats everyone else uses

HOS loads **GGUF** (the file format `llama.cpp` and Ollama use) natively. It
dequantizes every common type on load, so a model you already have just works.
It also reads a raw **HuggingFace safetensors** checkpoint plus its `config.json`
directly, with no `transformers`, no `convert_hf_to_gguf.py`, and no Python. The
architecture is detected and the per-family quirks are applied automatically:
RoPE style (interleaved vs NEOX), attention bias, tied embeddings, and the
norm/activation details each family needs.

Loading is **fully fallible**. A missing, malformed, or unsupported model returns
a clean error instead of panicking.

Families that run today: Llama / Mistral / Qwen2 / Qwen2.5 / SmolLM2 (CPU and
fused Metal GPU, bit-identical between the two), plus Gemma-2, Phi-3, and the
OLMoE mixture-of-experts on CPU. Gemma-4 multimodal image input is supported.

### 2.4 The `.hos` capsule: a model that remembers where it came from

A GGUF file is weights plus a metadata bag. It is an anonymous blob: nothing in
it tells you what it descended from or how it was made. The native **`.hos`**
format fixes that. A capsule carries the weights, optional optimizer state, and a
self-describing **card**: the architecture spec, provenance, a lineage chain, and
training history. It is not an archive, it is **runnable**:

```sh
hos --to-hos model.gguf -o model.hos --quantize q8_0   # mint a capsule
hos -m model.hos                                        # run it directly
hos --hos-info model.hos                                # read its card + lineage
```

Minting is bit-identical to the source, and `--hos-name` / `--source-note` let
you brand a capsule while keeping honest attribution in its card. This is the
technical backbone of the flwr model hub: a model pulled from flwr is never an
anonymous blob, it arrives with a card you can inspect and a lineage you can
trace back.

### 2.5 Quantization, with a proof

HOS quantizes weights and can *show its work*. The formats span the usual
size/accuracy trade: `q8_0`, `q6_k`, `q5_k`, `q4_k`, plus an `hq4` and a `q3`
lever for the low-bit end. Precision-sensitive tensors (norms, embeddings) stay
in higher precision, the same split the mature engines use, so nothing fragile
gets block-quantized.

Two things set it apart from "trust me, it is smaller":

- **AWQ-lite calibration** (`src/hos_awq.rs`): an activation-aware scaling pass
  that beats plain round-to-nearest at the *same* bit budget, and the engine
  proves it by measuring perplexity before and after on held-out text.
- **A GGUF-to-GGUF requantizer** (`hos --to-hos` and the flwr `quantize`
  command): derive a smaller GGUF that another engine can still load, with the
  metadata copied verbatim and perplexity verified to be preserved.

Reproduce any of it on your own machine with `hos --perplexity` and `hos --bench`.

### 2.6 CPU and GPU, bit-identical

The CPU path is not a fallback, it is a first-class, SIMD-accelerated backend.
The hot quantized matmul reads compressed weight bytes, dequantizes into a
per-thread scratch buffer, and dots against the activations, parallelized across
cores. It dispatches at **runtime** to AVX2 + FMA on x86 or NEON on Apple
Silicon, so one portable binary is fast on whatever CPU it lands on, no build
flags required.

On Apple Silicon, the Metal backend runs the same math on the GPU and is verified
**bit-identical** to the CPU result on the GPU families. On an M4 Max, the GPU
backend runs Llama-3.2-1B (Q4_K_M) at roughly six times CPU speed with weights
resident at their compressed size. HOS is not trying to win a tok/s sprint
against a decade of `llama.cpp` kernel tuning. It is trying to be the engine that
is competitive *and* readable *and* yours.

---

## 3. flwr, the app

`flwr` is to HOS what Ollama is to `llama.cpp`, with one difference: the model
stays open. Same friendly commands, a different deal.

- **`flwr pull <name>`** resolves a bare name from the flwr hub (or a HuggingFace
  repo, or a direct URL), downloads it, verifies its `sha256`, and stores it with
  provenance. Set `FLWR_REGISTRY` to point at your own hub instead.
- **`flwr run` / `flwr serve`** load and chat, with per-family chat templates
  detected from the special tokens actually present in the vocabulary, so replies
  end cleanly on the right stop token. `serve` speaks the OpenAI
  `/v1/chat/completions` API (streaming or one-shot) plus `/v1/models`.
- **The store is provenance-native.** `flwr list`, `flwr show`, `flwr cp`,
  `flwr quantize`, `flwr rm`. Every entry records where it came from, a content
  hash, and a lineage edge to its parent. A pulled model is never anonymous.

**Run your own hub.** Because pull is just "resolve a name to a URL plus a hash,
download, verify," you can host `.hos` capsules anywhere (a static bucket, your
own site) and point `flwr` at it. The bytes never route through the metadata
host, so distribution is cheap. That is exactly how the flwr model family is
served: capsules on object storage, a tiny registry that hands out URLs and
hashes, and `flwr pull` verifying every download.

---

## 4. The learnable turn: why "open up" matters

Everything above makes HOS a good local runtime. This chapter is why it is more
than that. Because the whole training loop and autograd graph are in-tree, a
model is not a frozen artifact you can only run. It is something you can shape.

### 4.1 Genes, not edits (shipped)

The parameter-efficient tuning in HOS (`src/peft.rs`) is built on a different
metaphor from LoRA. Instead of learning a low-rank *edit* that is added to a
weight, HOS learns a compact **regulatory** signal that gates and modulates what
is already there, closer to how a genome expresses a cell than how a patch
rewrites a file. The consequences that made this worth shipping: adapters compose
and recombine without stomping each other, and the behavior is more robust to
damage than an additive edit, because you are regulating capacity rather than
overwriting it. This is the "Genes, not Edits" line, and it is real code you can
run.

### 4.2 Research that proved fruitful

Not all of the research made it into the shipped binary, but the results are the
reason the engine is shaped the way it is. Presented as findings, not recipes.

- **Fusing two specialists into one model.** Given two frozen models good at
  different things, HOS can align their internal representations, bridge them, and
  distill the pair into a single new capsule that inherits both. The interesting
  result was not that it runs, but that alignment plus a learned bridge beats
  naive concatenation, and the fused model keeps a lineage edge back to both
  parents in its card.
- **A shared codebook across different models.** With a learned ridge alignment
  and a shared code space, the *same fact* lands on the *same code* even across
  models from different families and sizes. On a Llama-1B versus Qwen-0.5B pair,
  cross-model agreement on a fact-probe went from about 2% to 65%. That is the
  seed of models that can talk about the same thing in the same internal
  language.
- **Editing a fact at the output, on a real 1B.** A surgically addressed,
  output-level fact edit that changes what a real model says about one fact
  without retraining and without collateral drift on unrelated facts. Proven
  locally on a shipped-size model, not a toy.
- **Typing relations instead of memorizing strings.** A measured result that pure
  memorization does not transfer, but relational typing plus slot-memory
  collision detection is the lever that does. This is the direction that turns a
  next-token predictor into something that reasons over structure.

None of these are marketing. They are experiments that ran, produced a number, and
changed how the engine is built. The shipped engine gives you the substrate they
were built on: a real autograd graph, a real training loop, and a format that can
carry the lineage of whatever you grow.

---

## 5. Reference

**`hos` (engine).**

```
hos -m <model> -p <prompt> [-n N] [--temp T] [--seed S]   # generate
hos --gpu ...                                              # Metal backend
hos --info -m <model>                                      # arch / tensors / quant
hos --bench -m <model> [--gpu]                             # throughput
hos --perplexity [corpus.txt] -m <model>                   # quality
hos --to-hos <in.gguf> -o <out.hos> [--quantize q4_k]      # mint a .hos capsule
        [--hos-name NAME] [--source-note "..."]
hos --hos-info <file.hos>                                  # read the card + lineage
```

Models resolve by bare name from `$HOS_MODELS_DIR`, `~/.hos/models`, or a full
path with `-m`. Set `HOS_MODEL` for a default.

**`flwr` (app).**

```
flwr pull <name | hf-repo | url>     # into the store, verified, with provenance
flwr list | show <name>              # what you have, and its card
flwr run <name> [--gpu] [-p "..."]   # chat / one-shot
flwr serve <name> [--port 11434]     # OpenAI-compatible API + chat UI
flwr cp <src> <dst>                  # duplicate (keeps lineage)
flwr quantize <src> <dst> --type q8_0# derive a smaller model (keeps lineage)
flwr rm <name>                       # delete from the store
```

**Deeper docs in this repo:** [`ARCHITECTURE.md`](ARCHITECTURE.md) (internals),
[`HOS_FORMAT.md`](HOS_FORMAT.md) (the byte-level `.hos` spec),
[`LIBRARY.md`](LIBRARY.md) (embedding the engine as a Rust crate),
[`FINETUNING.md`](FINETUNING.md), and [`FLWR.md`](FLWR.md) (the full app + API
reference).

---

## License

HOS and flwr are released under the **Na'at Keystone License**: use it, study it,
change it, train with it, build from it, self-host it, teach with it, ship with
it, and sell what you make. The only covenant is fairness: give credit, preserve
provenance, do not pretend shared work was born closed, and return improvements
to the shared core when you offer the core to the world as a service. See
[`../LICENSE`](../LICENSE).

Intelligence is not a priesthood. It is a trade, a craft, a public instrument.
This engine exists so you can learn the machine, shape the machine, and build with
the machine without asking permission from a permanent ruling class.

Build freely. Attribute honestly. Keep the craft alive.
