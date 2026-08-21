# flwr — quickstart

**flwr** is the app you run; **HOS** (Hueman OS) is the engine under it. Zero ML
dependencies. Runs real models on Mac (Metal GPU) and Windows/Linux (CPU).

## Install
```bash
git clone <your-repo> hos && cd hos
cargo install --path . --bin flwr --bin hos     # installs `flwr` and `hos` to ~/.cargo/bin
```
Requires a Rust toolchain (`rustup`). No other dependencies.

## Get a model
A model is a `.hos` capsule (or a `.flwr` — same format, flwr's label). Point flwr at
any `.hos`/`.flwr`/`.gguf` file, or pull one:
```bash
flwr pull <hf-repo-or-gguf-url>          # fetch + convert into the local store
flwr list                                # see what you have
```
Or just use a path you already have.

## Run it
```bash
flwr run ~/Documents/hos/models/mymodel.hos --gpu        # Mac: Metal GPU
flwr run mymodel.flwr -p "Explain how photosynthesis works." -n 200   # one-shot
flwr run mymodel.hos                                       # Windows/Linux: CPU
```
- `--gpu` uses Metal on macOS. On Windows/Linux it runs on CPU (AVX2-accelerated on x86).
- The model's chat dialect is detected automatically; the turn ends cleanly (stop-aware).

## Serve an OpenAI-compatible API
```bash
flwr serve mymodel.hos --gpu --port 11434
# then POST to http://127.0.0.1:11434/v1/chat/completions
```

## Notes
- **Mac:** full Metal acceleration on Apple Silicon.
- **Windows:** CPU inference (AVX2). GPU is Metal-only by design.
- **`.flwr` vs `.hos`:** identical format. `.hos` is the engine capsule; `.flwr` is the
  same bytes wearing flwr's label. Either loads anywhere — the engine reads by content,
  not extension.
- **Quantize for speed:** `flwr quantize <src> <dst> --type q4_k` (Q4_K ≈ 7× smaller +
  ~5× faster than f32, small quality cost).
