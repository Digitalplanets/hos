# HOS in Plain English — What's Actually Going On

This is the no-jargon tour of what HOS is and how it works. If `ARCHITECTURE.md`
is the engineer's manual, this is the "explain it to a smart friend" version.

---

## What is HOS?

HOS is a program that **runs AI language models on your own Mac** — no cloud, no
internet, no accounts. You give it a model file and a prompt, and it generates
text, one word-piece at a time. It's written from scratch (it doesn't use
llama.cpp or anyone else's engine under the hood — just our own code).

Think of it as a **engine in a car**: the "model" is the fuel (the trained
knowledge), and HOS is the engine that burns it to actually move (produce words).

---

## The pieces, and what each one does

**1. The model file (`.gguf`)** — A few gigabytes of numbers ("weights") that are
the model's learned knowledge. We never change these; we just read them.

**2. Quantization — why the file is "compressed."**
A raw model stores each number as 32 bits. That's huge. *Quantization* squeezes
each number down to ~4–6 bits with barely any quality loss — like saving a photo
as a smaller JPEG. A 9-billion-parameter model goes from ~36 GB down to ~5 GB.
HOS reads these compressed numbers and unpacks them on the fly. (`gguf.rs`)

**3. The tokenizer** — The model doesn't read letters; it reads "tokens" (chunks
of words). The tokenizer turns your text into token-numbers going in, and
token-numbers back into text coming out. (`tokenizer.rs`)

**4. The forward pass — the actual "thinking."**
To produce the next word-piece, the model runs your text through dozens of
**layers**. Each layer does two big jobs:
  - **Attention:** "which earlier words matter for what comes next?"
  - **Feed-forward:** a big calculation that mixes information around.
Stacked 30–60 times, this is what turns "The capital of France is" into "Paris".
Mathematically, almost all of it is **matrix multiplication** — multiplying big
grids of numbers. That's where ~95% of the time goes. (`forward.rs`)

**5. The backends — CPU vs GPU.**
- **CPU:** your Mac's general-purpose cores. Flexible, but slow for this — it
  does the math more or less one chunk at a time across ~16 cores.
- **GPU (Metal):** your Mac's graphics chip. It has *thousands* of tiny cores,
  perfect for doing the same multiplication on huge grids at once. We write tiny
  programs called **kernels** that run on the GPU. HOS does this through Apple's
  **Metal** framework. (`metal_be.rs`)

On an M-series Mac, the CPU and GPU **share the same memory** (all 128 GB), which
is a big advantage — we can keep a whole model on the GPU without copying.

---

## The speed story (and the honest lessons)

Getting fast wasn't one trick — it was a sequence, and a couple of our guesses
were wrong (we measured instead of assuming):

1. **Run the math on the GPU instead of the CPU.** First obvious win.
2. **Keep the weights compressed in GPU memory** (quantized), instead of
   unpacking them to full size. Less data to move = faster, and the model fits.
3. **"Coalesced" memory reads.** The GPU is fastest when neighboring cores read
   neighboring bits of memory. Our first kernel had each core reading scattered
   locations (slow). Rewriting so a team of 32 cores reads a row *together* was a
   real speedup.
4. **Stop bouncing between CPU and GPU.** Early on, every single multiplication
   shipped data to the GPU and waited for it to come back — hundreds of round
   trips per word. We rebuilt it so the whole calculation stays on the GPU and
   only the final answer comes back. *Surprise:* this helped less than expected —
   because the real bottleneck was the math itself, not the trips. Good thing we
   measured.

Net result on the 9B model: from **1.5 words/sec → 11+ words/sec**, and the
answer is **identical** to the slow-but-trusted CPU version (we check this).

---

## The two kinds of models HOS runs

**Normal transformers** (Llama, Mistral, Qwen2, etc.) — the standard design.
HOS runs these on CPU or GPU. It also auto-detects each one's quirks (different
models rotate their numbers differently, some add an extra bias term, etc.) and
configures itself correctly. (`model.rs`)

**The Qwen3.5 hybrid** — your bigger models are a *newer, weirder* design. Most
of their layers don't use classic attention; they use a **"gated delta-net"** —
a running memory that updates a little bit with each new word (like keeping a
rolling summary in your head instead of re-reading the whole conversation every
time). This is cutting-edge and much harder to implement.

We got the exact recipe by reading the reference implementation, ported it, and
the hardest part (the rolling-memory update) we **verified on the GPU to match
the CPU math to 8 decimal places.** That's why your 9B — a model most home
engines can't run at all — runs on HOS. (`qwen35.rs`)

---

## What's honest about where it stands

- It **works and is correct** — outputs match the trusted CPU path bit-for-bit.
- It's **fast enough to use** (~11 words/sec on the 9B), but **not as fast as
  llama.cpp or Apple's MLX.** Those have had years of tuning and run *every*
  step on the GPU as one fused program. HOS still does a few small steps on the
  CPU. Closing that gap is more kernel work with diminishing returns.
- The big advantage you have isn't speed — it's that this is **yours**: a real,
  from-scratch engine you understand and control, running models locally and
  privately on your own hardware.

---

## How to use it

```sh
# install
cargo install --path .

# run a model
hos -m model.gguf -p "Write a haiku about rust" -n 60 --temp 0.7

# use your GPU
hos --gpu -m model.gguf -p "..." -n 100

# chat (reference client)
python3 examples/chat.py
```

That's the whole thing: read a compressed model, turn text into numbers, do a
mountain of multiplication (as fast as your Mac's GPU allows), turn the answer
back into text — locally, privately, on an engine that's entirely yours.
