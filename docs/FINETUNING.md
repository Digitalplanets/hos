# Fine-Tuning Your Own Model (the practical path)

Goal: turn a strong open base model into **your** model (your voice, your domain,
your tasks) and run it on HOS. We do **not** build a training framework for this —
we use a mature trainer for the hard part and let HOS do what it's good at:
inference.

```
   download base (HF)  ──▶  fine-tune (LoRA, MLX-LM)  ──▶  fuse adapter
        │                                                      │
        └──────────────── your dataset (JSONL) ────────────────┘
                                                               ▼
                              convert to GGUF  ──▶  run in HOS  (hos --gpu)
```

> Tooling note: commands below are the standard **MLX-LM** LoRA flow for Apple
> Silicon. Verify exact flags against the current `mlx-lm` docs before running —
> the API moves. (Ask and I'll check the latest.)

---

## 0. Why this split

- **Training ≠ inference.** Training needs full-precision gradients, an autograd
  engine, an optimizer, lots of memory for activations. Mature tools (MLX-LM,
  Unsloth, PyTorch) do this well and are tuned for it.
- **HOS already does inference.** It runs GGUF models on your GPU. So: train with
  the right tool, *serve* with your own engine.
- **Your 128GB M4 can LoRA-fine-tune a 7–9B locally** (hours), no cloud needed.

---

## 1. Pick the base

- **Qwen2.5-7B-Instruct** — HOS already runs it; strong; supported arch.
- (or Llama-3.1-8B-Instruct.)

For training you need the **original Hugging Face weights** (safetensors), *not*
the GGUF. MLX-LM converts HF → MLX format automatically on first use.

---

## 2. Build the dataset (this is 80% of the result)

Quality and consistency beat quantity. A few hundred to a few thousand clean
examples in chat format. JSONL, one example per line:

```json
{"messages": [
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

Rules of thumb:
- **Consistency** — same tone/format you want out. The model copies your data.
- **Cover the task** — variety within the target task, not random internet text.
- Split ~90/10 into `train.jsonl` / `valid.jsonl`.

This is the real work and where your project's edge lives (e.g. a focused
vernacular/linguistic dataset, your AUDIA persona, your domain Q&A).

---

## 3. LoRA fine-tune (MLX-LM, on your Mac)

LoRA trains small "adapter" matrices and freezes the base — ~90%+ of full
fine-tune quality, fits in memory, runs in hours.

```sh
pip install mlx-lm

python -m mlx_lm.lora \
  --model Qwen/Qwen2.5-7B-Instruct \
  --train \
  --data ./data \           # dir with train.jsonl / valid.jsonl
  --iters 600 \
  --batch-size 2 \
  --lora-layers 16
```

Watch the validation loss; stop when it stops improving (avoid overfitting your
small dataset).

---

## 4. Fuse the adapter into the base

```sh
python -m mlx_lm.fuse \
  --model Qwen/Qwen2.5-7B-Instruct \
  --adapter-path ./adapters \
  --save-path ./my-model-merged
```

This produces a standalone model (safetensors) with your tuning baked in.

---

## 5. Convert to GGUF + quantize (so HOS can run it)

Using the llama.cpp tools you already have at `~/llama.cpp`:

```sh
python ~/llama.cpp/convert_hf_to_gguf.py ./my-model-merged \
  --outfile my-model-f16.gguf --outtype f16

~/llama.cpp/build/bin/llama-quantize my-model-f16.gguf my-model-Q4_K_M.gguf Q4_K_M
```

---

## 6. Run YOUR model on HOS

```sh
hos --gpu -m my-model-Q4_K_M.gguf -p "..." -n 100
# or point the chat client at it:
HOS_CHAT_MODEL=$PWD/my-model-Q4_K_M.gguf python3 examples/chat.py
```

That's the loop: **your data → LoRA fine-tune → GGUF → served on your own
engine.** Iterate on the dataset; re-tune; re-serve.

---

## What this gives you vs. doesn't

- ✅ Your voice/persona, your domain language, your task formats — genuinely your
  model, running locally and privately on HOS.
- ✅ Fast iteration (hours per tuning run on your Mac).
- ❌ Not new fundamental reasoning beyond the 7–9B base (that's the base's
  ceiling; use RAG to inject knowledge).
- ❌ Not "trained from scratch" — that needs datacenter compute. This is
  adaptation of a strong base, which is the right tool 99% of the time.

---

## The longer game (optional, parallel)

Building HOS's own **tensor + autograd library** (so HOS itself can train, not
just serve) is the full-ownership project — feasible but months of work. The
practical path above gets you a real fine-tune now; the library is the
"own-the-entire-stack" goal to pursue separately if/when you want it. See the
roadmap in chat / `ARCHITECTURE.md`.
