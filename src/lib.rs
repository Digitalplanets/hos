//! HOS — a from-scratch local LLM inference engine.
//!
//! Library entry point. The modules are public so you can use the low-level
//! pieces directly; [`Engine`] is the high-level "load a model, generate text"
//! API most callers want.
//!
//! ```no_run
//! use std::path::Path;
//! let mut eng = hos::Engine::load(Path::new("model.gguf"), true /* gpu */).unwrap();
//! // prompt, max_tokens, temp, top_k, top_p, rep_penalty, repeat_last_n, seed, on_token
//! eng.generate("The capital of France is", 64, 0.0, 0, 1.0, 1.0, 64, 42, |piece| print!("{piece}"));
//! ```

// Intentional patterns for numeric / ML code: index-based loops mirror the math
// and read clearer than iterators in kernels; forward/load take many tunables;
// the autograd backward closure is a deliberately rich type.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::manual_is_multiple_of,
    clippy::new_without_default,
    clippy::unnecessary_map_or
)]

pub mod chat;
pub mod constraint;
pub mod error;
pub mod finetune;
pub mod format;
pub mod forward;
pub mod gemma4;
pub mod gemma4_chat;
pub mod gemma4_vision;
pub mod gemma_tok;
pub mod gguf;
pub mod gguf_write;
pub mod hf;
pub mod hos_awq;
pub mod hos_capsule;
pub mod hos_quant;
pub mod interp;
#[cfg(target_os = "macos")]
pub mod metal_be;
#[cfg(not(target_os = "macos"))]
#[path = "metal_be_stub.rs"]
pub mod metal_be;
pub mod model;
pub mod nn;
pub mod peft;
pub mod qwen35;
pub mod safetensors;
pub mod tensor;
pub mod tokenizer;
pub mod train;
pub mod viz;

use std::path::Path;

pub use error::{HosError, Result};

/// xorshift64 — deterministic, dependency-free RNG for sampling.
/// True if `path` is a `.hos` capsule (magic "HOSF"), checked by content.
pub fn is_hos_file(path: &Path) -> bool {
    use std::io::Read;
    if !path.is_file() {
        return false;
    }
    let mut buf = [0u8; 4];
    std::fs::File::open(path)
        .and_then(|mut f| f.read_exact(&mut buf))
        .map(|_| &buf == format::MAGIC)
        .unwrap_or(false)
}

pub fn next_rand(state: &mut u64) -> f32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    (x >> 40) as f32 / (1u64 << 24) as f32
}

/// Sample a token id. `temp <= 0` is greedy (argmax); otherwise temperature
/// softmax filtered by top-k then nucleus top-p. `top_k = 0` disables top-k;
/// `top_p >= 1.0` disables nucleus filtering. `rep_penalty > 1.0` penalizes
/// tokens in `recent` (the just-generated tokens) to curb loops/repetition.
pub fn sample(
    logits: &[f32],
    temp: f32,
    top_k: usize,
    top_p: f32,
    rep_penalty: f32,
    recent: &[u32],
    rng: &mut u64,
) -> u32 {
    // repetition penalty (operate on a local copy only when active)
    let owned;
    let logits: &[f32] = if rep_penalty != 1.0 && !recent.is_empty() {
        let mut l = logits.to_vec();
        for &t in recent {
            let t = t as usize;
            if t < l.len() {
                l[t] = if l[t] > 0.0 {
                    l[t] / rep_penalty
                } else {
                    l[t] * rep_penalty
                };
            }
        }
        owned = l;
        &owned
    } else {
        logits
    };

    if temp <= 0.0 {
        let mut best = 0usize;
        for i in 1..logits.len() {
            if logits[i] > logits[best] {
                best = i;
            }
        }
        return best as u32;
    }
    // temperature softmax
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&l| ((l - max) / temp).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }
    // rank by probability, keep top-k then nucleus top-p
    let mut idx: Vec<usize> = (0..probs.len()).collect();
    idx.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
    if top_k > 0 {
        idx.truncate(top_k.min(idx.len()));
    }
    if top_p < 1.0 {
        let mut cum = 0.0;
        let mut cut = idx.len();
        for (i, &id) in idx.iter().enumerate() {
            cum += probs[id];
            if cum >= top_p {
                cut = i + 1;
                break;
            }
        }
        idx.truncate(cut);
    }
    // sample from the kept set
    let total: f32 = idx.iter().map(|&i| probs[i]).sum();
    let r = next_rand(rng) * total;
    let mut cdf = 0.0;
    for &i in &idx {
        cdf += probs[i];
        if r < cdf {
            return i as u32;
        }
    }
    *idx.last().unwrap() as u32
}

/// High-level engine: a loaded model + tokenizer + backend, ready to generate.
pub struct Engine {
    pub model: model::Model,
    pub tok: tokenizer::Tokenizer,
    runner: Option<metal_be::GpuRunner>,
    state: forward::State,
    pos: usize,
    /// When true, chat generation is JSON-constrained (R2): the sampler can only
    /// pick tokens that keep the output valid JSON. Default false = unchanged.
    json_mode: bool,
}

/// Size the rayon thread pool once, politely. HOS ships to other people's
/// machines, so by default it leaves a couple of cores free for the system
/// rather than pegging every core (the "don't monopolise the box" default).
/// Override with `HOS_THREADS=<n>` (0 or unset = the polite default).
fn configure_threads() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let n = std::env::var("HOS_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| {
                let avail = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4);
                // leave 2 cores for the OS/UI on a roomy machine; use all on small ones
                if avail > 4 {
                    avail - 2
                } else {
                    avail
                }
            });
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global();
    });
}

impl Engine {
    /// Load a GGUF model. `gpu = true` uses the Metal backend (weights resident,
    /// dequantized in-kernel); `false` runs multithreaded on CPU. Returns an
    /// error (not a panic) on a missing/malformed/unsupported model file.
    pub fn load(path: &Path, gpu: bool) -> Result<Engine> {
        configure_threads();
        // A HuggingFace checkpoint folder (config.json + *.safetensors) loads
        // natively — no GGUF, no llama.cpp, no transformers.
        if path.is_dir() && path.join("config.json").exists() {
            return Self::from_source(
                &hf::HfModel::open(path)?,
                tokenizer::Tokenizer::from_hf(path)?,
                gpu,
            );
        }
        // A runnable `.hos` capsule (magic "HOSF") loads via its own ModelSource.
        // Like the HF path, it can ride the Metal backend (the capsule's f32
        // tensors upload exactly as HF/GGUF weights do) for Llama-family arches.
        if is_hos_file(path) {
            let (src, tok) = hos_capsule::HosSource::open(path)?;
            return Self::from_source(&src, tok, gpu);
        }
        let g = gguf::Gguf::open(path)?;
        let tok = tokenizer::Tokenizer::from_gguf(&g)?;
        Self::from_source(&g, tok, gpu)
    }

    /// Build the engine from any weight source (GGUF or HF) plus its tokenizer.
    fn from_source<S: model::ModelSource>(
        src: &S,
        tok: tokenizer::Tokenizer,
        gpu: bool,
    ) -> Result<Engine> {
        // the fused GPU runner supports only the Llama-family arches; Gemma2/Phi3
        // run on the CPU forward path.
        let use_gpu = gpu
            && cfg!(target_os = "macos")
            && model::Arch::detect(src).gpu_supported()
            && model::gpu_quant_supported(src); // HQ4/MoE have no GPU kernel -> CPU path
        let gpu_ctx = if use_gpu {
            Some(metal_be::Gpu::new())
        } else {
            None
        };
        let model = model::Model::load(src, gpu_ctx.as_ref())?;
        let runner = gpu_ctx.map(|gp| metal_be::GpuRunner::new(gp, &model));
        let state = forward::State::new(&model.cfg);
        Ok(Engine {
            model,
            tok,
            runner,
            state,
            pos: 0,
            json_mode: false,
        })
    }

    /// Enable/disable JSON-constrained decoding for subsequent `chat` calls (R2).
    /// Opt-in; when off, generation is byte-for-byte unchanged.
    pub fn set_json_mode(&mut self, on: bool) {
        self.json_mode = on;
    }

    fn forward(&mut self, token: u32, pos: usize) -> Vec<f32> {
        match &self.runner {
            Some(r) => r.forward(&self.model, token, pos),
            None => forward::forward(&self.model, &mut self.state, token, pos, None),
        }
    }

    /// Final hidden state (post-output-norm, pre-lm-head) for the LAST token of
    /// `ids`. Runs the CPU forward path so the representation tap fires, on a fresh
    /// KV cache, leaving the engine's own generation state untouched. This is the
    /// per-prompt anchor used by representation alignment / fusion (`--stitch`).
    pub fn last_hidden(&mut self, ids: &[u32]) -> Vec<f32> {
        let mut st = forward::State::new(&self.model.cfg);
        forward::htap_arm();
        for (pos, &t) in ids.iter().enumerate() {
            let _ = forward::forward(&self.model, &mut st, t, pos, None);
        }
        forward::htap_take().unwrap_or_default()
    }

    /// Mean-pooled final hidden state (post-output-norm, pre-lm-head) over ALL
    /// tokens of `ids`. Same fresh-state CPU tap as `last_hidden`, re-armed per
    /// position so every token's final hidden contributes. This is the pooling
    /// that makes an untrained decoder usable as a text embedder (RAG/zettel).
    pub fn mean_hidden(&mut self, ids: &[u32]) -> Vec<f32> {
        let mut st = forward::State::new(&self.model.cfg);
        let mut acc: Vec<f32> = Vec::new();
        let mut n = 0usize;
        for (pos, &t) in ids.iter().enumerate() {
            forward::htap_arm();
            let _ = forward::forward(&self.model, &mut st, t, pos, None);
            if let Some(h) = forward::htap_take() {
                if acc.is_empty() {
                    acc = vec![0.0; h.len()];
                }
                for (a, v) in acc.iter_mut().zip(&h) {
                    *a += v;
                }
                n += 1;
            }
        }
        if n > 0 {
            let inv = 1.0 / n as f32;
            for a in acc.iter_mut() {
                *a *= inv;
            }
        }
        acc
    }

    /// Per-layer residual-stream representation (last token of `ids`) at every
    /// depth: `out[l]` is the hidden state after layer `l`. One CPU forward pass
    /// captures the whole stack, used to sweep alignment across depth (`--stitch`).
    pub fn all_hidden(&mut self, ids: &[u32]) -> Vec<Vec<f32>> {
        let mut st = forward::State::new(&self.model.cfg);
        forward::htap_arm();
        for (pos, &t) in ids.iter().enumerate() {
            let _ = forward::forward(&self.model, &mut st, t, pos, None);
        }
        forward::htap_take_layers().unwrap_or_default()
    }

    /// Ingest the prompt and return the last token's logits. On the CPU path this
    /// batches the prompt (one weight read reused across all tokens); the GPU
    /// runner and MoE fall back to the token-by-token loop. Advances `pos`.
    fn prefill(&mut self, ids: &[u32]) -> Vec<f32> {
        // GPU: skip the lm-head for every prompt token but the last (KV cache is
        // still filled), saving the vocab×dim matvec T−1 times.
        if !ids.is_empty() {
            if let Some(r) = self.runner.as_ref() {
                // batched prefill: ingest the whole prompt in one pass, reusing each
                // weight read across all tokens. Falls back if the model isn't eligible.
                if let Some(logits) = r.forward_prefill_gpu(&self.model, ids, self.pos) {
                    self.pos += ids.len();
                    return logits;
                }
                let mut pos = self.pos;
                let mut logits = Vec::new();
                for (i, &t) in ids.iter().enumerate() {
                    if i + 1 == ids.len() {
                        logits = r.forward(&self.model, t, pos);
                    } else {
                        r.prefill_step(&self.model, t, pos);
                    }
                    pos += 1;
                }
                self.pos = pos;
                return logits;
            }
        }
        // CPU: batched prefill (weight reused across all prompt tokens).
        if self.runner.is_none() {
            if let Some(logits) =
                forward::forward_prefill(&self.model, &mut self.state, ids, self.pos, None)
            {
                self.pos += ids.len();
                return logits;
            }
        }
        let mut logits = Vec::new();
        for &t in ids {
            logits = self.forward(t, self.pos);
            self.pos += 1;
        }
        logits
    }

    /// Generate up to `max_tokens` from `prompt`, invoking `on_token` with each
    /// decoded piece as it streams. Returns the number of tokens generated.
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temp: f32,
        top_k: usize,
        top_p: f32,
        rep_penalty: f32,
        repeat_last_n: usize,
        seed: u64,
        mut on_token: impl FnMut(&str),
    ) -> usize {
        let ids = self.tok.encode(prompt, true);
        let mut logits = self.prefill(&ids);
        let mut rng = seed;
        let mut generated = 0usize;
        let mut recent: Vec<u32> = Vec::new();
        let mut buf = Vec::new();
        for _ in 0..max_tokens {
            let from = recent.len().saturating_sub(repeat_last_n);
            let next = sample(
                &logits,
                temp,
                top_k,
                top_p,
                rep_penalty,
                &recent[from..],
                &mut rng,
            );
            recent.push(next);
            if Some(next) == self.tok.eos {
                break;
            }
            buf.clear();
            self.tok.decode_into(next, &mut buf);
            on_token(&String::from_utf8_lossy(&buf));
            if self.pos >= self.model.cfg.ctx_len {
                break;
            }
            logits = self.forward(next, self.pos);
            self.pos += 1;
            generated += 1;
        }
        generated
    }

    /// Generate from `prompt` and return the raw generated token-id stream
    /// (not decoded text). Same decode loop as [`generate`] — prefill, then
    /// repeatedly sample + feed back — but it records the token ids the engine
    /// actually emits. With `temp <= 0` this is deterministic greedy decoding,
    /// which the golden regression test pins to a committed fixture. The flwr
    /// arch's terminal E8 lattice snap turns sub-ULP drift into a token flip, so
    /// this id stream is the engine's floating-point canary. Stops at `eos`, the
    /// context limit, or `max_tokens`; the eos token is not included.
    pub fn generate_ids(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temp: f32,
        seed: u64,
    ) -> Vec<u32> {
        let ids = self.tok.encode(prompt, true);
        let mut logits = self.prefill(&ids);
        let mut rng = seed;
        let mut out: Vec<u32> = Vec::new();
        let mut recent: Vec<u32> = Vec::new();
        for _ in 0..max_tokens {
            // greedy/argmax when temp<=0; top_k=0, top_p=1, rep_penalty=1 keep it
            // a pure next-token pick so the fixture is a function of the logits only.
            let next = sample(&logits, temp, 0, 1.0, 1.0, &recent, &mut rng);
            recent.push(next);
            if Some(next) == self.tok.eos {
                break;
            }
            out.push(next);
            if self.pos >= self.model.cfg.ctx_len {
                break;
            }
            logits = self.forward(next, self.pos);
            self.pos += 1;
        }
        out
    }

    /// Reset the KV cache and position so the next generation starts fresh.
    pub fn reset(&mut self) {
        self.state = forward::State::new(&self.model.cfg);
        self.pos = 0;
    }

    /// The chat dialect HOS detected for the loaded model.
    pub fn chat_family(&self) -> chat::ChatFamily {
        chat::ChatFamily::detect(&self.tok, self.model.cfg.arch)
    }

    /// Stream a completion from pre-tokenized `ids`, stopping at `max_tokens`,
    /// the context limit, or any id in `stops`. Continues from the current
    /// position (does not reset) and does not prepend bos — the caller owns the
    /// prompt. This is the chat-templating path's generator.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_from_ids(
        &mut self,
        ids: &[u32],
        max_tokens: usize,
        temp: f32,
        top_k: usize,
        top_p: f32,
        rep_penalty: f32,
        repeat_last_n: usize,
        seed: u64,
        stops: &[u32],
        stop_strs: &[String],
        mut on_token: impl FnMut(&str),
    ) -> usize {
        use crate::constraint::Constraint;
        let mut logits = self.prefill(ids);
        let mut rng = seed;
        let mut generated = 0usize;
        let mut recent: Vec<u32> = Vec::new();
        let mut buf = Vec::new();
        let mut pending = String::new();
        // R2 — JSON-constrained decoding. Built fresh per generation so the
        // grammar state never carries across turns. `None` (the default) leaves
        // sampling untouched. End tokens (stops + EOS) may terminate only once
        // the JSON is a complete document.
        let mut constraint = if self.json_mode {
            let mut ends: Vec<u32> = stops.to_vec();
            if let Some(eos) = self.tok.eos {
                if !ends.contains(&eos) {
                    ends.push(eos);
                }
            }
            Some(crate::constraint::JsonConstraint::new(&self.tok, ends))
        } else {
            None
        };
        for _ in 0..max_tokens {
            let from = recent.len().saturating_sub(repeat_last_n);
            if let Some(c) = &constraint {
                c.mask(&mut logits);
            }
            let next = sample(
                &logits,
                temp,
                top_k,
                top_p,
                rep_penalty,
                &recent[from..],
                &mut rng,
            );
            recent.push(next);
            if let Some(c) = &mut constraint {
                c.accept(next);
            }
            if stops.contains(&next) {
                break;
            }
            buf.clear();
            self.tok.decode_into(next, &mut buf);
            let piece = String::from_utf8_lossy(&buf);
            // String-level stop with BUFFERED streaming: models fine-tuned on flat
            // transcripts never emit an EOS token, so we watch for a new role header
            // (e.g. "\nUser:", "\nSystem:"). Any trailing text that could be the START
            // of a stop marker is HELD back, so a partial header never reaches the user,
            // and the turn ends cleanly the instant one completes. Stops are ASCII, so
            // every slice below lands on a char boundary.
            if stop_strs.is_empty() {
                on_token(&piece);
            } else {
                pending.push_str(&piece);
                if let Some(pos) = stop_strs
                    .iter()
                    .filter_map(|s| pending.find(s.as_str()))
                    .min()
                {
                    on_token(&pending[..pos]); // emit up to the marker, drop the rest
                    pending.clear();
                    break;
                }
                // longest suffix of `pending` that is a prefix of some stop → hold it back
                let hold = stop_strs
                    .iter()
                    .map(|s| {
                        let mut k = s.len().min(pending.len());
                        while k > 0 && !pending.as_bytes().ends_with(s[..k].as_bytes()) {
                            k -= 1;
                        }
                        k
                    })
                    .max()
                    .unwrap_or(0);
                let emit_to = pending.len() - hold;
                on_token(&pending[..emit_to]);
                pending.drain(..emit_to);
            }
            if self.pos >= self.model.cfg.ctx_len {
                break;
            }
            logits = self.forward(next, self.pos);
            self.pos += 1;
            generated += 1;
        }
        // Flush any held tail that turned out NOT to be a stop marker (e.g. we hit
        // max_tokens / ctx_len mid-hold). A real stop already cleared `pending`.
        if !pending.is_empty() {
            on_token(&pending);
        }
        generated
    }

    /// Run one assistant turn over a conversation, applying the model's detected
    /// chat template. Resets the KV cache and re-renders the full history each
    /// call, so multi-turn callers just append messages and call again.
    #[allow(clippy::too_many_arguments)]
    pub fn chat(
        &mut self,
        msgs: &[chat::Message],
        max_tokens: usize,
        temp: f32,
        top_k: usize,
        top_p: f32,
        rep_penalty: f32,
        repeat_last_n: usize,
        seed: u64,
        on_token: impl FnMut(&str),
    ) -> usize {
        let fam = chat::ChatFamily::detect(&self.tok, self.model.cfg.arch);
        let ids = chat::build_ids(&self.tok, fam, msgs, true);
        let stops = chat::stop_ids(&self.tok, fam);
        let stop_strs = chat::stop_strs(fam);
        self.reset();
        self.generate_from_ids(
            &ids,
            max_tokens,
            temp,
            top_k,
            top_p,
            rep_penalty,
            repeat_last_n,
            seed,
            &stops,
            &stop_strs,
            on_token,
        )
    }

    /// Mean per-token negative log-likelihood (nats) and perplexity over a
    /// token sequence. Feeds tokens left-to-right and scores each next-token
    /// prediction with an exact log-softmax — the standard correctness metric
    /// for an inference engine. Returns `(scored_tokens, mean_nll, perplexity)`.
    /// Resets the KV cache first, so it is independent of any prior generation.
    pub fn perplexity(&mut self, ids: &[u32]) -> (usize, f64, f64) {
        self.state = forward::State::new(&self.model.cfg);
        self.pos = 0;
        let mut nll = 0.0f64;
        let mut count = 0usize;
        let limit = self.model.cfg.ctx_len;
        for i in 0..ids.len().saturating_sub(1) {
            if i >= limit {
                break;
            }
            let logits = self.forward(ids[i], i);
            // log p(target) = logit[target] - logsumexp(logits)
            let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sumexp = 0.0f64;
            for &l in &logits {
                sumexp += ((l - max) as f64).exp();
            }
            let target = ids[i + 1] as usize;
            let logp = (logits[target] - max) as f64 - sumexp.ln();
            nll += -logp;
            count += 1;
        }
        let mean = if count > 0 { nll / count as f64 } else { 0.0 };
        (count, mean, mean.exp())
    }

    /// Time prefill (encode + ingest the prompt) and greedy decode separately —
    /// a reproducible throughput benchmark. Resets the KV cache first.
    pub fn bench(&mut self, prompt: &str, decode_tokens: usize) -> Bench {
        use std::time::Instant;
        self.state = forward::State::new(&self.model.cfg);
        self.pos = 0;

        let ids = self.tok.encode(prompt, true);
        // warm up the GPU pipelines so the timed prefill isn't charged for first-use
        // shader compilation / cold dispatch.
        let _ = self.prefill(&ids);
        self.state = forward::State::new(&self.model.cfg);
        self.pos = 0;
        let t0 = Instant::now();
        let mut logits = self.prefill(&ids);
        let prefill_secs = t0.elapsed().as_secs_f64();

        let argmax = |v: &[f32]| -> u32 {
            let mut best = 0usize;
            for i in 1..v.len() {
                if v[i] > v[best] {
                    best = i;
                }
            }
            best as u32
        };

        let t1 = Instant::now();
        let mut produced = 0usize;
        for _ in 0..decode_tokens {
            let next = argmax(&logits);
            if Some(next) == self.tok.eos || self.pos >= self.model.cfg.ctx_len {
                break;
            }
            logits = self.forward(next, self.pos);
            self.pos += 1;
            produced += 1;
        }
        let decode_secs = t1.elapsed().as_secs_f64();

        Bench {
            prefill_tokens: ids.len(),
            prefill_secs,
            decode_tokens: produced,
            decode_secs,
        }
    }
}

/// Result of [`Engine::bench`]: separated prefill and decode timings.
pub struct Bench {
    pub prefill_tokens: usize,
    pub prefill_secs: f64,
    pub decode_tokens: usize,
    pub decode_secs: f64,
}

impl Bench {
    pub fn prefill_tps(&self) -> f64 {
        self.prefill_tokens as f64 / self.prefill_secs.max(1e-9)
    }
    pub fn decode_tps(&self) -> f64 {
        self.decode_tokens as f64 / self.decode_secs.max(1e-9)
    }
}
