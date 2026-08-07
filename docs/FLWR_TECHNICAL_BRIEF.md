# flwr — Technical Brief

*HOS engine · drafted 2026-06-26*

---

## Abstract

**flwr** is a derived LLM architecture (`Arch::Flwr`, `src/model.rs:23,41`) and the pipeline that produces it: take a stock **SmolLM2‑135M** seed (Hugging Face, 2024), **GROW** its FFN losslessly via Net2Net function-preserving widening, **WEAVE** an E8-lattice vector-quantized bottleneck into the forward pass on the final hidden state before the lm-head, and mint the result as a **SOVEREIGN** capsule whose lineage is rooted at its own `flwr-genesis` id rather than at the donor model. The defining architectural divergence from Llama is the woven E8‑VQ bottleneck; the defining *product* claim is a model that "owns its identity." What was achieved is an end-to-end, reproducible build — grow parity ≈ 0, straight-through training that converges (probe cross-entropy 3.x → **0.3379** over 400 steps), and a capsule that mints and remints to runnable — plus, in the sibling `edit_lab` workstream, a measured demonstration that an **E8 discrete code can serve as an editable *address space*** in which edit-locality is a structural property and ripple is observable and tunable. This brief separates, rigorously, what is **measured** from what is **proposed**, and what is genuinely **owned** (combinations and framings) from what is **assembled from known primitives** (VQ, STE, E8, Net2Net, model editing — none invented here).

---

## 1. The Architecture Edge

The flwr pipeline is three stages. Each stage's *mechanism* is prior art; the *composition* and *framing* are the contribution.

### GROW — function-preserving FFN widening (`finetune.rs::from_model_grown`, line 116; driven by `FT_GROW_FFN`, `main.rs:2675-2683`)

To widen the FFN from `ffn` to `new_ffn`, each new intermediate unit replicates a randomly chosen existing unit `j` (`rep[k] = rng()%ffn`). Gate/up columns are copied verbatim for originals and copied-with-tiny-noise (`*(1.0+0.001·noise)`) for replicas to break symmetry; the down-projection rows of any unit duplicated `count[j]` times are divided by `count[j]` (`nd[u*dim+jd] = down[j*dim+jd]/c`). The layer's summed output is therefore **identical at init** — capability is inherited losslessly, then copies diverge under training.

**Why it matters:** the grown model *starts* as exactly the seed's function (book-reported init `max|Δ| ~1e-6`), so no capability is paid for at widening time. This is textbook **Net2WiderNet (Chen et al. 2016)**, applied as-is.

### WEAVE — E8-VQ bottleneck on the final hidden (`finetune.rs::forward_vq`, line 327; `nn.rs::VectorQuantizer::quantize`, line 608; driven by `FT_FLWR`, `main.rs:2738-2746`)

The final hidden `h=[T,dim]` is split into block=8 chunks, each snapped to its nearest E8 lattice point at scale=1.0 via `nearest_e8` (`nn.rs:570`) — the exact Conway–Sloane decoder (`E8 = D8 ∪ (D8+½)`; `nearest_d8` rounds, and on odd coordinate-sum flips the single coord with the largest rounding residual). Training uses straight-through estimation `output = h + strength·(hard − h)` (value ≈ lattice point, gradient to encoder = identity) plus a commitment loss `mean‖h − sg[hard]‖²·0.25` added to cross-entropy (`main.rs:2786-2790`). `strength` anneals 0→1 over the first half of training (`main.rs:2781`) so discretization doesn't shock early steps.

**This is the actual divergence from Llama** — the modification that makes flwr "the flwr architecture" rather than a fine-tune.

**Training/inference twin.** The hand-coded inference forward `flwr_e8_quant` (`forward.rs:107-115`) snaps the final hidden to E8 (block=8, scale=1.0) before the lm-head at both prefill (`forward.rs:391`) and decode (`forward.rs:567`) — the deterministic mirror of `forward_vq` at strength=1. `check_parity` (`finetune.rs:336`) gates that the autograd forward reproduces the inference forward (≈ 0 max-abs logit diff). **The forward is yours**: the same discrete operation runs in training and in the engine, verified equal.

### SOVEREIGN — self-rooted lineage capsule (`main.rs::save_flwr`, line 2844; `cmd_remint_ft`, line 475)

`save_flwr` writes a `trainable` capsule with `architecture:"flwr"`, `bottleneck:"e8-vq"`, and `lineage = [flwr_root]`, where `flwr_root` is a fixed content hash of a synthetic `"flwr-genesis"` tensor. Every flwr model shares this root, so **the lineage starts at flwr, not at the donor**. `cmd_remint_ft` remaps FtModel-layout tensors back to GGUF/engine names (+transpose), grafts the base tokenizer/hyperparameters, detects the grown FFN width and corrects metadata (`main.rs:572-577`), and for a flwr source copies hyperparams to the `flwr.*` prefix and sets `general.architecture="flwr"` so the loader re-applies the E8 bottleneck (`main.rs:582-598`).

### What was measured (artifacts in `models/`, via `hos --hos-info`)

- **`flwr.hos`** (863.7 MB, trainable): arch `flwr`, 30 layers, dim 576, **FFN grown 1536 → 2560**, bottleneck `e8-vq`; one run of **400 steps, AdamW lr 2e‑5, final_loss 0.3379**; lineage rooted at `flwr-genesis` (`4d25767f9dce13f5`); 273 tensors / 215.9M values.
- **`flwr_run.hos`** (864.8 MB, inference): the remint of `flwr.hos`, runnable, lineage root preserved.
- **`--vq-demo`** (`nn.rs:658`): a 4-class classifier trained through the same E8 STE bottleneck, reporting accuracy and distinct-E8-codes-used — proves the VQ machinery works, separate from the LM.

### Honesty caveats on this stage

1. **The 0.3379 is a probe loss, not an eval.** It is training cross-entropy on a fixed probe window over a small/repeated corpus (`main.rs:2701-2705, 2766-2771`) — memorization on a tiny window, **not** held-out perplexity or any downstream benchmark. That the E8 bottleneck *improves or even preserves* LM quality is **proposed, not measured.**
2. **`flwr_run.hos` does not yet evidence inference-time E8.** Its card shows `architecture:"llama"` / `feed_forward_length:1536` — it was minted before/without the `is_flwr` branch carrying flwr arch + grown width into the card JSON. The code path that re-applies E8 at inference (`main.rs:582-598`) is present but **not evidenced in this saved file**. "It runs as Flwr" is proven as a code path and as the `flwr.hos` card; it is **not yet proven end-to-end in this particular remint artifact.**

---

## 2. Innovations (full ledger)

Every entry is tagged. For [KNOWN-APPLIED] and the pieces inside [NOVEL-COMBO], prior art is named. **No primitive here is invented.**

### Architecture / pipeline (Part 1)

| Innovation | Tag | Prior art / notes |
|---|---|---|
| Net2Net function-preserving FFN widening (`finetune.rs:116`, `peft.rs:238`) | **[KNOWN-APPLIED]** | Net2Net / Net2WiderNet, Chen et al. 2016. Replica-count down-projection split is the standard construction. |
| STE + commitment-loss training through a discrete bottleneck (`nn.rs:608-637`) | **[KNOWN-APPLIED]** | VQ-VAE, van den Oord et al. 2017; straight-through estimator, Bengio 2013. |
| E8 lattice nearest-point decoder (`nn.rs:553-577`) | **[KNOWN-APPLIED]** | Conway & Sloane, *Sphere Packings, Lattices and Groups* (`E8 = D8 ∪ (D8+½)`). |
| Annealed quantization strength 0→1 (`main.rs:2781`) | **[KNOWN-APPLIED]** | Standard QAT temperature/annealing practice. |
| SmolLM2-135M as the seed | **[KNOWN-APPLIED]** | SmolLM2, Hugging Face 2024. |
| Training/inference twin with a parity gate (`flwr_e8_quant` mirroring `forward_vq` @ strength=1; `check_parity`) | **[KNOWN-APPLIED]** | Forward-parity / gradient-check testing; standard engineering. |
| **E8-lattice VQ bottleneck on the FINAL hidden of an autoregressive transformer, before the lm-head, as the defining arch mod** | **[NOVEL-COMBO]** *(proposed beneficial)* | VQ-VAE STE + Conway–Sloane E8 + standard LM head. Discrete LM bottlenecks exist (discrete-latent VAEs, VQ representations), so the exact placement is **plausibly-novel, not strictly new**; unproven as beneficial. |
| **grow→weave→sovereign as one pipeline ending in a self-rooted lineage capsule** (`save_flwr` flwr-genesis root + `cmd_remint_ft` archival→runnable) | **[NOVEL-COMBO]** *(framing)* | Built on ordinary content-hash provenance/lineage. The workflow + "the model owns its identity / lineage starts at flwr" framing is the defensible, ownable part. |

### Memory / editing substrate (Parts 2 & 4)

| Innovation | Tag | Prior art / notes |
|---|---|---|
| E8 lattice as a discrete **address space** for editable memory (vs. as a compressor) | **[NOVEL-COMBO]**, mostly **proposed** | Lattice VQ (known) + content-based memory addressing (known). Framing "discreteness = built-in locality = isolable slots" is defensible; proven only as a quantizer so far. |
| **Fact-as-fingerprint addressing**: concatenated B=h/8 nearest-E8 codes as a content-address, then keying an edit to that address so it can only reach the address-sharing cohort (`edit_lab/src/main.rs:382-534`) | **[NOVEL-COMBO]** *(framing; mechanism, not a measured editing win)* | Lattice VQ + associative/content-addressable memory + knowledge-editing locality testing: ROME (Meng et al. 2022), MEMIT, MEND (Mitchell et al. 2022), ripple-effect eval (Cohen et al. 2023). The *lattice code as the edit key* is the defensible piece. |
| **Decoupling addressability (distinct/collision) from value capacity (recall), with H = number of E8 blocks as an explicit tunable collision lever** | **[NOVEL-COMBO]** *(framing; proven as a measured trend)* | Each ingredient standard (hash-collision rate vs table size; capacity vs params). The clean decomposition with a geometric lever is the contribution. |
| Hard top-1 STE slot addressing + load-balancing to make single-slot edits local (`mem_lm_demo`, `hard1`, `nn.rs:1551/1671/1748`) | **[NOVEL-COMBO]**, **proven (toy)** | Key-value/memory networks (Weston/Sukhbaatar; Lample product-key memory 2019), MoE load-balancing (Shazeer 2017; Fedus Switch 2021), hard-attention STE. Finding that the load-balance term is *what makes edits local* is the contribution. |

### Editing protocol (Part 3)

| Innovation | Tag | Prior art / notes |
|---|---|---|
| Optimizer-subset "freezing" (zero-all-grads + step-a-subset; `train`, lines 104-114) | **[KNOWN-APPLIED]** | Standard selective/layer-freezing fine-tuning. |
| Phase-1 value edit confined to the readout head while the shared token vector is frozen | **[NOVEL-COMBO]** | Selective freezing (known) + editing-locality goal of ROME/MEMIT (Meng 2022), MEND (Mitchell 2021). The *ordering* as a two-phase value/re-anchor protocol is the contribution; pieces are prior art. |
| Contrastive re-anchor of the shared embedding using old AND new value tokens' *other* relations (`reanchor`, lines 127-135; Phase 2) | **[NOVEL-COMBO]**, *uncertain — possibly constrained-FT re-expression* | ROME/MEMIT locality/specificity + KL preservation anchor; contrastive representation learning. Explicitly re-anchoring a perturbed token against its surviving relations as a dedicated post-edit phase is, to our knowledge, unnamed — **but we are not certain it is novel.** Treat as a framing claim, not an algorithmic one. |
| Arm C preservation anchors (sampled untouched facts batched into the value edit) | **[KNOWN-APPLIED]** | Standard locality-preserving edit / MEMIT preservation term / rehearsal-replay anti-forgetting. |
| Shared-embedding LM deliberately constructed as a ripple stress-test (entangled recurring city tokens + watch set + old-value-token survival metric) | **[NOVEL-COMBO]** *(experimental framing)* | Mirrors CounterFact/zsRE locality splits, ROME/MEMIT evals. Deliberate weight-shared embeddings to *guarantee* ripple + per-old-token survival metric is a clean framing built from known patterns. |
| A/B/C ablation as evidence that ripple localizes to "the head moving for other contexts" | **[NOVEL-COMBO]** *(diagnostic finding, proven only at toy scale)* | Measured observation on this testbed, not a general result. |

### PEFT / fusion / growth (Part 4)

| Innovation | Tag | Prior art / notes |
|---|---|---|
| **RGA**: compact per-domain "genome" gating a shared frozen gene bank as a PEFT adapter (`peft.rs` Method::Rga) | **[NOVEL-COMBO]**, **proven at 135M** | Adapters (Houlsby 2019), LoRA (Hu 2021, the matched baseline), conditional/gated computation + MoE gating, hypernetwork conditioning (Ha 2016). Defensible: genome-as-separable-code giving zero forgetting + recombinable parents. |
| Clonal selection: proliferate selected genes into a private mutable bank (`proliferate`/`set_clones`) | **[NOVEL-COMBO]** *(framing)* | ≈ copy-on-write adapter expansion / progressive networks (Rusu 2016). Immune-system framing + "seen domains literally unaffected" guarantee are the contribution. |
| Net2Net FFN widening with duplicated-neuron warm start (`grow_ffn`) | **[KNOWN-APPLIED]** | Net2Net (Chen et al. 2016); zero-init-output / duplicate-neuron is exactly their transform. |
| **Fusion by bank concatenation in a shared frozen-base coordinate frame (span-union)** (`fuse`) | **[NOVEL-COMBO]**, **proven** | Model merging / task arithmetic (Wortsman model soups; Ilharco 2022; TIES-merging), shared-representation alignment (Platonic Representation Hypothesis). Defensible twist: alignment is *identity* because both specialists regulate the same base, so the merge is exact concatenation, not lossy averaging. |
| Knowledge-retaining operator = grow + RGA gate + E8 mix, zero-init output (identity at insertion) (`GatedTrunkOp`, `nn.rs:3113`) | **[NOVEL-COMBO]**, **proven (toy)** | Composition of the three pieces above; +0.00 insertion guarantee follows from zero-init. |
| flwr zero-dep ANSI growth-organism TUI (`viz.rs`) | **[KNOWN-APPLIED]** | Standard truecolor terminal rendering; product polish, not a research claim. |

**Bottom line for the frontier claim:** none of the primitives (VQ, STE, E8, Net2Net, lineage hashing, model editing) are invented here, and that is not claimed. Defensible ownership is in the *combinations and framings*.

---

## 3. The Frontier You Own

This is the honest, defensible core. Three pieces, stated with what is genuinely the contribution vs assembled-from-known, and proven vs proposed.

### (a) E8-addressed editable memory: edit-locality as a *structural* guarantee, ripple as *observable & predictable*

**The idea (yours, [NOVEL-COMBO] framing):** snap a fact's hidden vector to E8 (block=8), concatenate the integer codes into a per-fact **fingerprint**, and treat that fingerprint as the fact's *address*. An edit is a hidden-space patch keyed to a fingerprint, so it can only reach the cohort that shares the address. Ripple stops being an invisible diffuse side-effect and becomes a *visible collision* in a discrete code-space.

**Measured (from `edit_lab/RIPPLE_STATUS.md`, deterministic):**
- **Addressing holds at fixed H=64 up to ~20k:** distinct 100% (N≤5000), 99.7% (N=10000, collision 0.5%), 93.8% (N=20000, collision 7.9%).
- **The LEVER — collision is controlled by H (number of E8 blocks):** at fixed N=2000, collision 39.3% (H=8) → 1.4% (H=16) → **0% (H=32)**. Reproduced at scale: fixed N=50000, collision 90.9% (H=32) → **0% (H=256, distinct 100%)**.
- **Address-space saturation onset:** N=100000 at H=256 → distinct 57.3%, collision 47.7% (the ceiling for H/8=32 E8 blocks).
- **"Promise met" config:** N=10000, H=256 → recall 90.6%, distinct 100%, collision 0%.
- **Recall is a SEPARATE, decaying capacity limit** (decoupled from addressing): at H=256, recall 90.6% (10k) → 40.8% (20k) → 13.7% (50k), while distinct stays ~100%.

**What is genuinely measured vs definitional — stated plainly:**
- **Measured & falsifiable:** `distinct`/`collision` (addressing capacity) and `recall` (value capacity). The clean finding is: **E8-block fingerprints stay distinct up to N≈50k at H=256, collision is a monotone H-tunable knob, and value recall is a decoupled axis.**
- **Definitional, NOT empirical:** because the edit patch is *physically* added only to cohort facts (`if fps[i]==fp0`), every rippled fact is necessarily in the cohort, so **`foreseen` = 100% by construction** and `locality` ≈ 100% whenever the cohort is small. These are **design properties, not discoveries.** The "edits stay local / ripple foreseen" half of the promise must be framed as a *structural guarantee of the construction*, not as evidence.
- **Scope:** synthetic random 4-char→char facts, single-hidden-layer MLP, no real knowledge, no ROME/MEMIT baseline. This is a **proof-of-mechanism on a toy.**
- **Do not claim:** "continuous nets can't see this" (`flwr_ripple`, `main.rs:370-372`) is rhetorical and unproven — continuous nets can also compute neighbor/collision structure.

### (b) Staged-freeze + contrastive-reanchor editing (measured ripple reduction)

**The idea (yours, [NOVEL-COMBO]):** on a *deliberately entangled* shared-embedding LM (so a naive overwrite *will* ripple), apply a two-phase edit: **Phase 1** writes the new value into the readout head with the shared token embedding frozen; **Phase 2** re-anchors the embedding against the old and new value tokens' *other, surviving* relations. Arm C adds preservation anchors to Phase 1.

**Measured (deterministic, seed `0xABCD`, 17 facts, `edit_lab/src/main.rs`):**

| Arm | edit applied | untouched kept | paris's own facts kept |
|---|---|---|---|
| A · naive overwrite | true | **0.0%** | 0/2 |
| B · staged + reanchor | true | **56.2%** | 2/2 |
| C · localized + reanchor | true | **68.8%** | 2/2 |

Naive editing applies the fact but wrecks every other fact (0% survival); the staged protocol fully saves the old value token (2/2) and lifts survival to 56.2%; preservation anchors raise it to 68.8%.

**Honest scope:** toy (17 facts, 25-word vocab, 2-layer MLP-LM, no attention). The numbers are **proven/measured**; even the best arm still ripples ~31%. Any claim it scales to real LMs is **proposed**. The Phase-2 contrastive re-anchor is **possibly a re-expression of constrained fine-tuning** — treat as a framing claim, not a new algorithm. Prior art it stands on: ROME/MEMIT (Meng 2022), MEND (Mitchell 2021), standard layer-freezing, contrastive representation learning.

### (c) The discrete-observable-ripple framing → a learned repair controller

The unifying, defensible thesis: **putting the representation on a discrete lattice makes ripple a first-class, addressable, *predictable* object** — a collision in code-space rather than a diffuse weight perturbation. (a) shows the address structure is real and H-tunable; (b) shows that, once ripple is localized to an identifiable cause ("the head moving for other contexts"), a staged protocol measurably repairs it. Together they motivate — but do **not** yet demonstrate — a **learned meta-controller** that reads the discrete collision structure and emits the repair edit. That controller is **proposed**, not built.

**What you own here, in one sentence:** the *framing and combination* — E8 code as an edit address with an H-lever on collisions, plus a value-edit-then-contrastive-reanchor protocol with a clean measured ablation — not the invention of VQ, lattices, or knowledge editing.

---

## 4. Proven vs Proposed

| Claim | Status | Evidence / caveat |
|---|---|---|
| Net2Net grow is function-preserving at init | **Proven** | init `max|Δ| ~1e-6`; `check_parity` ≈ 0 logit diff |
| STE training through the E8 bottleneck converges | **Proven** | probe loss 3.x → 0.3379 over 400 steps |
| flwr capsule mints with flwr-rooted lineage + remints to runnable | **Proven** | `flwr.hos`, `flwr_run.hos` artifacts |
| VQ machinery (gradients through discrete E8) works | **Proven** | `--vq-demo` 4-class STE classifier |
| 0.3379 = held-out quality / perplexity | **Proposed (NOT this)** | it is memorization on a tiny fixed probe window |
| E8 bottleneck preserves/improves LM quality | **Proposed** | no held-out eval, no end-to-end generation-quality result |
| `flwr_run.hos` re-applies E8 at inference end-to-end | **Proposed / not evidenced** | card shows `llama`/1536; code path present (`main.rs:582-598`) but not in this file |
| E8 fingerprints distinct to N≈50k (H=256), collision = monotone H-lever, recall decoupled | **Proven (toy)** | `RIPPLE_STATUS.md` tables |
| Edit-locality / ripple-foreseen in the addressed memory | **Definitional, not measured** | `foreseen`=100% by construction (patch added only to cohort) |
| Staged-freeze + reanchor reduces ripple (0% → 56.2% → 68.8%) | **Proven (toy)** | 17 facts, seed `0xABCD`; ~31% residual ripple remains |
| Contrastive re-anchor is a novel algorithm | **Uncertain** | possibly constrained-FT re-expression; claim as framing only |
| RGA matches LoRA quality with zero forgetting; fusion span-union exact | **Proven at 135M** | 2.39 vs 2.42 ppl; 1.0× vs 14.4× forgetting; 1184× span-union, 0.0000 drift |
| Memory/contradiction/collision-at-scale, free-run generation quality | **Proposed / open** | toy/char-level; author-flagged open bets |
| Method scales to real (≥1B) LMs | **Proposed** | untested |
| Learned meta-controller that repairs ripple from collision structure | **Proposed** | not built |

---

## 5. Open / Next

1. **Bridge to the real flwr model.** Fix the remint card so `flwr_run.hos` carries `architecture:"flwr"` + grown FFN width (`main.rs:582-598` path), then prove **end-to-end** that the engine re-applies E8 at inference and generates text — closing the one honest gap in §1.
2. **A held-out eval.** Replace the fixed-probe 0.3379 with held-out perplexity and a downstream task, with a no-bottleneck control, to convert "E8 preserves quality" from **proposed** to **measured** (or to falsify it).
3. **Lift editing off the toy.** Move (a) and (b) from synthetic char-level facts to a real transformer with a ROME/MEMIT/MEND baseline; measure *real* locality (not the by-construction `foreseen`) and compare ripple to published editing methods.
4. **Collision-at-scale.** Push the address-space past the H/8=32-block ceiling seen at N=100k — more blocks, learned (vs fixed) lattice scale, or hierarchical addressing — and characterize the recall capacity axis independently.
5. **The meta-controller.** Build the learned repair controller that reads discrete collision structure and emits the edit — the payoff of the discrete-observable-ripple framing, currently proposed.
6. **1B scale.** Re-seed the grow→weave→sovereign pipeline and the RGA/fusion stack from a ≥1B base to test whether the framings (zero-forgetting genome, identity-frame fusion, E8 addressing) survive scale.

---

*Files: `src/finetune.rs`, `src/forward.rs`, `src/nn.rs`, `src/model.rs`, `src/main.rs`, `src/peft.rs`, `src/viz.rs`; artifacts `models/flwr.hos`, `models/flwr_run.hos`.*
