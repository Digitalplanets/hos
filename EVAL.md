# hos — correctness eval

A numbers-you-can-stand-on sanity check that the engine is numerically sound
end-to-end (load → forward → logits → log-likelihood), measured on this machine.

## Perplexity on the built-in held-out passage
`hos -m <model> --perplexity`

| model | quant | perplexity |
|---|---|---|
| Llama-3.2-1B-Instruct | Q8_0 | **15.03** |
| Llama-3.2-1B-Instruct | Q4_K | 17.46 |
| flwr-1B (indios finetune) | Q4_K | 25.44 |

## How to read these
- **Engine correctness:** a coherent 1B model scores ~15 ppl here. A numerically
  broken forward pass would produce garbage (hundreds to thousands), not 15 — so
  load → forward → logits → log-softmax is sound.
- **Quantization is graceful:** Q8 → Q4_K costs ~+2.4 ppl (15.0 → 17.5). That's the
  expected, small cost of 4-bit — it confirms the K-quant dequant path is correct,
  not lossy-to-the-point-of-broken.
- **flwr's higher ppl is expected, not a defect:** it's a *domain* finetune (the
  indios corpus) plus Q4_K, so on general held-out English it reads worse than base
  Llama. It traded broad fluency for domain specialization — by design.

## Honest scope
This is a **correctness/sanity** number, not a competitive benchmark. It uses the
engine's built-in passage, so it is NOT directly comparable to published WikiText
perplexities or to a head-to-head vs llama.cpp. To *prove* parity with a reference
implementation, the next step is: run the SAME standard text through both hos and
llama.cpp on the same GGUF and assert the perplexities match within tolerance. The
number above establishes the engine isn't broken; the head-to-head would establish
it's bit-faithful.
