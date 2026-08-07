# flwr — User Manual

**flwr** is the app you talk to. It runs local language models on the
[HOS engine](LIBRARY.md), gives you an interactive chat REPL and an
OpenAI-compatible server, and manages a provenance-tracking model store — what
`ollama` is to `llama.cpp`, built as a separate binary over the `hos` library.

flwr does not change the `hos` engine command; it is its own program. Installing
the project gives you both: `hos` (the engine/CLI) and `flwr` (this app).

---

## Contents

1. [Install](#1-install)
2. [Quick start](#2-quick-start)
3. [Commands](#3-commands)
   - [`flwr run`](#flwr-run--chat) · [`flwr serve`](#flwr-serve--http-server) ·
     [`flwr pull`](#flwr-pull--download) · [`flwr list`](#flwr-list) ·
     [`flwr show`](#flwr-show) · [`flwr cp`](#flwr-cp--copy--import) ·
     [`flwr quantize`](#flwr-quantize) · [`flwr rm`](#flwr-rm)
4. [How models are found](#4-how-models-are-found)
5. [The model store](#5-the-model-store)
6. [The OpenAI-compatible API](#6-the-openai-compatible-api)
7. [Configuration & environment](#7-configuration--environment)
8. [Sampling parameters](#8-sampling-parameters)
9. [Troubleshooting](#9-troubleshooting)
10. [FAQ](#10-faq)

---

## 1. Install

Build the project; both binaries land in `~/.cargo/bin`:

```sh
cargo install --path .      # installs `hos` and `flwr`
# or, without installing:
cargo build --release       # binaries in ./target/release/{hos,flwr}
```

Requirements: a Rust toolchain. The GPU backend is Apple-Silicon only (Metal);
elsewhere flwr runs on CPU. `flwr pull` shells out to the system **`curl`**.

---

## 2. Quick start

```sh
# fetch a small instruct model into the store (provenance recorded)
flwr pull HuggingFaceTB/SmolLM2-135M-Instruct

# chat with it
flwr run SmolLM2-135M-Instruct

# or serve it on an OpenAI-compatible endpoint
flwr serve SmolLM2-135M-Instruct --port 11434
```

In the REPL, type to chat. `/bye` ends the session, `/reset` clears the
conversation.

---

## 3. Commands

General form:

```
flwr <command> [model] [options]
```

`[model]` is a store name, a bare model name, or a filesystem path (see
[§4](#4-how-models-are-found)).

### `flwr run` — chat

Interactive chat REPL, or a one-shot turn with `-p`.

```sh
flwr run SmolLM2-135M-Instruct                 # interactive
flwr run Llama-3.2-1B-Instruct-Q4_K_M.gguf --gpu
flwr run my-model -p "Name three primary colors."   # one-shot, prints and exits
```

| Option | Default | Meaning |
|---|---|---|
| `-p, --prompt <text>` | — | One-shot mode: single user turn, print reply, exit. |
| `-n, --n-predict <N>` | 512 | Max tokens to generate per turn. |
| `--temp <T>` | 0.7 | Sampling temperature (`0` = greedy). |
| `--top-k <K>` | 40 | Top-k cutoff. |
| `--top-p <P>` | 0.95 | Nucleus cutoff. |
| `--seed <S>` | 42 | RNG seed. |
| `--gpu` | off | Use the Metal GPU backend (Apple Silicon, supported arches). |

**REPL commands:** `/bye`, `/exit`, `/quit` end the session; `/reset` clears the
in-memory conversation. Multi-turn history is kept in process and re-sent each
turn. `Ctrl-D` also exits.

### `flwr serve` — HTTP server

Starts an OpenAI-compatible HTTP daemon (see [§6](#6-the-openai-compatible-api)).

```sh
flwr serve SmolLM2-135M-Instruct --port 11434
flwr serve Qwen2.5-0.5B-Instruct-Q4_K_M.gguf --host 0.0.0.0 --port 8080 --gpu
```

| Option | Default | Meaning |
|---|---|---|
| `--host <H>` | `127.0.0.1` | Bind address. Use `0.0.0.0` to expose on the network. |
| `--port <P>` | `11434` | Listen port (the de-facto local-LLM port). |
| `--gpu` | off | Metal GPU backend. |

Runs until interrupted (`Ctrl-C`). Connections are handled concurrently;
generation is serialized through the single resident model.

> **Security:** the server has no authentication. Bind to `127.0.0.1` (the
> default) unless you intend to expose it, and put a reverse proxy / auth in
> front if you do.

### `flwr pull` — download

Download a model into the store.

```sh
flwr pull HuggingFaceTB/SmolLM2-135M-Instruct          # a HuggingFace repo id
flwr pull Qwen/Qwen2.5-0.5B-Instruct --revision main   # pin a revision
flwr pull https://example.com/model.gguf --name my-gguf  # a direct .gguf URL
```

| Option | Default | Meaning |
|---|---|---|
| `--revision <rev>` | `main` | HuggingFace revision/branch/tag. |
| `--name <name>` | repo/file basename | Store name to save under. |

For a HuggingFace repo, flwr queries the model's file list and downloads exactly
what HOS loads: `*.safetensors` (and the shard index), `config.json`,
`generation_config.json`, and the tokenizer files. Training logs, `onnx/`
subfolders, and `.bin` duplicates are skipped. HOS loads **safetensors**, not
PyTorch `.bin` — a repo with only `.bin` weights is rejected.

### `flwr list`

List stored models with size, architecture, and source.

```sh
flwr list      # (alias: flwr ls)
```

### `flwr show`

Print a model's full provenance card: source, revision, architecture, identity
hash, on-disk path, and a per-file size + content hash table.

```sh
flwr show SmolLM2-135M-Instruct
```

### `flwr cp` — copy / import

Two modes:

```sh
# 1) duplicate an existing store entry under a new name (keeps lineage)
flwr cp SmolLM2-135M-Instruct my-variant

# 2) import a local path into the store (HF checkpoint dir or .gguf file)
flwr cp ./my-finetuned-checkpoint my-finetune
flwr cp /models/some-model.gguf   my-gguf
```

A duplicate keeps the source's content `identity` and records a `copied_from`
lineage edge. An import records `source: local:<absolute-path>`. The destination
name must not already exist.

### `flwr quantize`

Derive a smaller GGUF from a GGUF source, recording a lineage edge to the parent.

```sh
flwr quantize big-f16.gguf big-q8 --type q8_0   # from a path
flwr quantize my-model      my-model-q8         # from a store entry
```

| Option | Default | Meaning |
|---|---|---|
| `-t, --type <quant>` | `q8_0` | `q8_0` (≈lossless), `q5_0`, `q4_0`, and the K-quants `q6_k` / `q5_k` / `q4_k` (better quality per bit; need a row dim that's a multiple of 256). |

HOS reads each tensor of the source as f32 and re-encodes the weight matrices to
Q8_0 (≈8 bits/weight, near-lossless), keeping 1-D norm/bias tensors in full
precision. The source's metadata — including the tokenizer — is copied verbatim,
so the result loads exactly like the original, just smaller. The derived model
records `quant: q8_0` and a `copied_from` edge in its manifest.

**Scope:** the source must be a **GGUF** (a store entry or a `.gguf` path);
GGUF→GGUF only. Quantizing a HuggingFace checkpoint directly, and K-quant
targets (`Q4_K` etc.), are not yet supported. Verify quality with
`hos --perplexity -m <output>.gguf` — Q8_0 should track the source closely.

### `flwr rm`

Delete a model from the store.

```sh
flwr rm my-variant
```

---

## 4. How models are found

For `run` / `serve`, the `[model]` argument resolves in this order:

1. An **exact filesystem path** that exists (a `.gguf` file or an HF directory).
2. A **store name** (something you pulled or imported — see [§5](#5-the-model-store)).
3. A **bare name** searched in: `$HOS_MODELS_DIR`, `~/Documents/hos/models`,
   `~/.hos/models`.

If nothing matches, flwr prints where it looked and exits. You can also set
`HOS_MODEL` as a default model name.

---

## 5. The model store

flwr keeps pulled and imported models under a store root:

```
$FLWR_STORE  ||  $HOS_STORE  ||  ~/.hos/store
└── <name>/
    ├── <model files...>        # .safetensors + config + tokenizer, or a .gguf
    └── manifest.json           # the provenance card
```

Each `manifest.json` records, in the spirit of HOS's `.hos` lineage card:

| Field | Meaning |
|---|---|
| `name` | Store name. |
| `kind` | `hf` or `gguf`. |
| `source` | `hf:<repo>`, a URL, or `local:<path>`. |
| `revision` | HuggingFace revision (or `-`). |
| `arch` | Architecture (read from `config.json` when available). |
| `files` | Each file's size and **FNV content hash**. |
| `identity` | Combined content hash for the whole artifact. |
| `copied_from` | If made by `flwr cp`/`flwr quantize`, the store entry it descends from. |
| `quant` | If made by `flwr quantize`, the target quant (e.g. `q8_0`). |
| `pulled_unix` | When it entered the store. |

This is the **provenance-native** difference from an opaque blob cache: every
stored model carries where it came from and a content identity you can audit with
`flwr show`. Two copies of identical weights share an `identity`; a copy also
keeps a lineage edge to its parent.

---

## 6. The OpenAI-compatible API

`flwr serve` speaks a subset of the OpenAI REST API, so existing chat clients,
SDKs, and IDE plugins work by pointing their base URL at flwr.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | **Built-in browser chat UI** (a single self-contained page). |
| `GET` | `/health` | Liveness line. |
| `GET` | `/v1/models` | The one resident model. |
| `POST` | `/v1/chat/completions` | Chat. `"stream": true` for SSE token streaming. |
| `POST` | `/chats` | Save a transcript (`{id, messages}`); server stamps provenance. |
| `GET` | `/chats` | List saved transcripts (newest first). |
| `GET` | `/chats/<id>` | One full saved transcript. |
| `DELETE` | `/chats/<id>` | Delete a saved transcript. |
| `GET` | `/models/available` | Models you can switch to (store + model dirs). |
| `POST` | `/model` | Switch the resident model live (`{name}`). |

**Model selection in the UI.** The header has a **MODEL** dropdown listing every
model `flwr` can find: store entries, plus `.gguf` / `.hos` files and HF
checkpoints in the model dirs (`$HOS_MODELS_DIR`, `~/Documents/hos/models`,
`~/.hos/models`). A converted `.hos` shows up if you save it into one of those
dirs or import it with `flwr cp model.hos <name>` — then **reload the page**
(the list is fetched on load). Pick one and the server **loads it live** — no
restart — and keeps
serving on the same port. Switching takes as long as loading that model; the UI
shows "switching… (loading)" until it's ready.

**Browser UI.** Open `http://127.0.0.1:11434/` in any browser for a built-in chat
interface — a single hand-written HTML page (no framework, no CDN, no build
step), brutalist styling with an in-app **theme switcher** (BONE / CONCRETE /
ACID / INK, remembered across sessions), that streams from this same server.

### Desktop app (macOS)

A native macOS app — a real window (`WKWebView`), **Apple system frameworks
only, no packages or crates** — wraps the same UI. Build it once:

```sh
bash desktop/macos/build.sh          # → ~/Applications/Flwr.app  (needs Xcode CLT / swiftc)
open ~/Applications/Flwr.app          # launches the server + native window
```

It auto-starts `flwr serve` and loads the UI. Configure via environment:
`FLWR_MODEL` (default `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`), `FLWR_PORT`
(default `11599`), `FLWR_BIN` (default `~/.cargo/bin/flwr`).

Because all the real work lives in `flwr serve`, a desktop app on any OS is just
a thin native window over `localhost`. Windows (WebView2) and Linux (WebKitGTK)
shells follow the same `desktop/<os>/` pattern and can be added per device.

### Saved chats (provenance-bearing)

Conversations in the UI are **auto-saved** to `~/.hos/chats/<id>.json` (override
with `FLWR_CHATS`). The sidebar lists them; click to reopen, **+ NEW** starts a
fresh one, **×** deletes. Each transcript is not a convenience cache — it records
*which model produced it* (name + content-hash `id` + `source` + `quant`, looked
up from the store) and the sampling params, so a saved chat is a **reproducible,
lineaged record** in the same spirit as a `.hos` capsule:

```jsonc
{ "id": "...", "created": …, "title": "What is HOS?",
  "model": { "name": "SmolLM2-135M-Instruct", "id": "5ca01265ececa865",
             "source": "hf:HuggingFaceTB/SmolLM2-135M-Instruct", "quant": null },
  "params": { "temperature": 0.7, "seed": 42, … },
  "messages": [ … ] }
```

They're plain JSON files you can open, grep, version, or share.

### Request fields

`messages` (required, array of `{role, content}`), plus optional `max_tokens`,
`temperature`, `top_p`, `seed`, and `stream`.

### Examples

Non-streaming:

```sh
curl http://127.0.0.1:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello in one word."}],"max_tokens":20}'
```

Streaming (Server-Sent Events; one `delta` per token, terminated by `[DONE]`):

```sh
curl -N http://127.0.0.1:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Count to three."}],"stream":true}'
```

With the official OpenAI Python client (point `base_url` at flwr):

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="not-needed")
r = client.chat.completions.create(
    model="any",  # flwr serves one resident model
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in r:
    print(chunk.choices[0].delta.content or "", end="")
```

The chat template is applied automatically from the model's detected dialect:
**ChatML** (Qwen / SmolLM2 / OLMoE), **Llama-3** headers, **Gemma**
`<start_of_turn>`, **Phi-3** `<|user|>`, **Mistral** `[INST]`. Turn-terminators
are detected as stop tokens, so replies end cleanly.

### Concurrency

Connections are accepted and parsed on independent threads, so a slow client
never blocks others and `GET /` / `/v1/models` answer immediately. **Generation
is serialized** through the one resident model (a single KV cache): concurrent
chat requests queue and run one at a time. For parallel decode, run multiple
`flwr serve` processes on different ports.

---

## 7. Configuration & environment

| Variable | Used by | Effect |
|---|---|---|
| `FLWR_STORE` | flwr | Store root (highest priority). |
| `HOS_STORE` | flwr | Store root (fallback). Default `~/.hos/store`. |
| `HOS_MODELS_DIR` | resolution | Extra directory searched for bare model names. |
| `HOS_MODEL` | resolution | Default model name when none is given. |

---

## 8. Sampling parameters

| Parameter | Default | Notes |
|---|---|---|
| temperature | 0.7 (`run`/`serve`) | `0.0` = deterministic greedy decoding. |
| top-k | 40 | 0/large = effectively off. |
| top-p | 0.95 | Nucleus sampling cutoff. |
| repetition penalty | 1.1 | Applied over the recent-token window. |
| seed | 42 | Fixed seed → reproducible output at a given temperature. |
| max tokens | 512 | Per turn (`-n` for `run`; `max_tokens` in the API). |

For fully reproducible output, use `--temp 0`.

---

## 9. Troubleshooting

**`flwr pull` fails immediately / "curl not available".**
flwr uses the system `curl` to download. Ensure `curl` is on your `PATH`.

**"no .safetensors in that repo — HOS loads safetensors, not .bin".**
The HuggingFace repo ships only PyTorch `.bin` weights. HOS reads safetensors;
pick a repo that publishes `.safetensors` (most modern ones do).

**"model not found".**
flwr printed the directories it searched. Pass a full path, `flwr pull` it first,
or set `HOS_MODELS_DIR` / `HOS_MODEL`.

**"could not bind … address already in use".**
Another process holds the port. Choose a different `--port`, or stop the other
server.

**Replies don't stop / look mis-formatted.**
The model's chat dialect may be unrecognized (served as `plain`). Check
`flwr run` — the banner prints the detected `dialect`. Base (non-instruct) models
have no chat template and are best used via `hos` raw completion.

**The model over-refuses, contradicts itself, or claims it "can't see history."**
That's a too-small model, not a bug — memory and context *are* sent every turn.
Very small models (e.g. SmolLM2-135M) emit canned refusals and hallucinated
disclaimers. Switch to a larger one (Llama-3.2-1B, Qwen2.5-7B) via the MODEL
dropdown and the behavior improves sharply on the same engine.

**GPU flag seems ignored.**
The Metal backend covers the Llama / Mistral / Qwen2 families; other arches
(Gemma-2, Phi-3, MoE) and non-macOS builds run on CPU regardless of `--gpu`.

---

## 10. FAQ

**Is flwr the same as the `hos` command?**
No. `hos` is the engine and its low-level CLI (raw completion, perplexity, bench,
ingest/convert, training). `flwr` is the chat/app layer on top. They ship
together but are separate binaries.

**Does flwr send anything to the cloud?**
Only `flwr pull` makes network requests (to HuggingFace or a URL you give it),
via the system `curl`. `run`/`serve` are fully local.

**Can it serve more than one model at once?**
One model per `flwr serve` process. Run several on different ports for several
models, or for parallelism on one model.

**Where are my models stored?**
`~/.hos/store` by default; override with `FLWR_STORE` or `HOS_STORE`. See
[§5](#5-the-model-store).

**Which model formats are supported?**
GGUF (`F32/F16/Q8_0/Q4_0/Q5_0/Q4_K/Q5_K/Q6_K`) and HuggingFace safetensors
checkpoints (single-file or sharded), with byte-level BPE or SentencePiece
tokenizers.

---

*See also: [LIBRARY.md](LIBRARY.md) to build on the engine, and the project
`README.md` for the `hos` CLI.*
