//! Canonical Gemma-4 chat capability — lives in the HOS **library** so every
//! front-end (the `flwr` runner, `flwr serve`, and the `flwr_agent` capability
//! layer) drives the native Gemma-4 decoder through one implementation instead
//! of reimplementing the chat template / tokenizer wiring in three places.
//!
//! Gemma-4 is HOS's own additive arch (`crate::gemma4`), separate from the
//! generic `Engine` Llama-family path. This module owns: the chat special
//! tokens, a loaded `Session` (decoder + tokenizer, optional GPU), history →
//! reply generation, and the checkpoint-discovery helpers. Front-ends keep only
//! their own UI.

use std::path::{Path, PathBuf};

use crate::gemma4::{Gemma4, Gemma4Cache};
use crate::gemma_tok::GemmaTokenizer;
use crate::metal_be::Gpu;
use crate::Result;

// Gemma chat special tokens (verified in the classifier prefix/suffix wiring).
const BOS: u32 = 2; // <bos>
const SOT: u32 = 105; // <start_of_turn>
const EOT: u32 = 106; // <end_of_turn>
const NL: u32 = 107; // "\n"
const USER: u32 = 2364; // "user"
const MODEL: u32 = 4368; // "model"
const EOS: u32 = 1; // <eos>

/// Prefill lengths at/above this engage the batched GPU prefill.
const BATCH_THRESHOLD: usize = 16;

/// Sampling / generation knobs (mirrors flwr's `Opts`/`Params`).
pub struct GenParams {
    pub n_predict: usize,
    pub temp: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub rep_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
}

impl Default for GenParams {
    fn default() -> GenParams {
        GenParams {
            n_predict: 512,
            temp: 0.7,
            top_k: 40,
            top_p: 0.95,
            rep_penalty: 1.1,
            repeat_last_n: 64,
            seed: 42,
        }
    }
}

/// A loaded Gemma-4 chat session: decoder (optionally GPU-resident) + tokenizer.
pub struct Session {
    m: Gemma4,
    tok: GemmaTokenizer,
    gpu: Option<Gpu>,
}

impl Session {
    /// Load a Gemma-4 checkpoint for chat: either a portable `.hos` capsule or a
    /// HuggingFace checkpoint dir. Defaults to the fast q4_k weights.
    pub fn load(path: &Path, gpu_on: bool) -> Result<Session> {
        let (mut m, tok) = if is_gemma4_capsule(path) {
            // Self-contained capsule — no safetensors dir, tokenizer embedded.
            Gemma4::from_capsule(path)?
        } else {
            if std::env::var("HOS_GEMMA4_QUANT").is_err() {
                std::env::set_var("HOS_GEMMA4_QUANT", "q4k");
            }
            let m = Gemma4::load(path)?;
            let tok = GemmaTokenizer::load_from_model_dir(path)?;
            (m, tok)
        };
        let gpu = if gpu_on {
            let g = Gpu::new();
            m.upload_to_gpu(&g);
            Some(g)
        } else {
            None
        };
        Ok(Session { m, tok, gpu })
    }

    /// Generate the model's reply to `history` (a list of (role, content) turns),
    /// streaming decoded text to `on_piece`. Returns the number of tokens emitted.
    pub fn generate<F: FnMut(&str)>(
        &self,
        history: &[(String, String)],
        gp: &GenParams,
        rng: &mut u64,
        mut on_piece: F,
    ) -> usize {
        let m = &self.m;
        let tok = &self.tok;
        let gref = self.gpu.as_ref();

        // <bos> { <start_of_turn> role \n content <end_of_turn> \n } … <start_of_turn> model \n
        let mut ids: Vec<u32> = vec![BOS];
        for (role, content) in history {
            ids.push(SOT);
            ids.push(if role == "user" { USER } else { MODEL });
            ids.push(NL);
            ids.extend(tok.encode(content, false));
            ids.push(EOT);
            ids.push(NL);
        }
        ids.push(SOT);
        ids.push(MODEL);
        ids.push(NL);

        let mut cache = Gemma4Cache::new(m);
        let mut logits = match gref {
            Some(g) if ids.len() >= BATCH_THRESHOLD && m.can_batch_prefill() => {
                m.prefill_batched(&mut cache, &ids, g)
            }
            g => m.prefill(&mut cache, &ids, g),
        };

        let mut out_ids: Vec<u32> = Vec::new();
        let mut prev = String::new();
        // Gemma-4 prepends a bare `thought` line before its answer. Buffer the first
        // line and drop it if that's all it is, so the transcript starts at the reply.
        let mut head_buf = String::new();
        let mut head_done = false;
        for _ in 0..gp.n_predict.max(1) {
            let start = out_ids.len().saturating_sub(gp.repeat_last_n);
            let recent = &out_ids[start..];
            let next = crate::sample(
                &logits,
                gp.temp,
                gp.top_k,
                gp.top_p,
                gp.rep_penalty,
                recent,
                rng,
            );
            if next == EOT || next == EOS {
                break;
            }
            out_ids.push(next);
            // UTF-8-safe streaming: only slice when the new decode extends the prior.
            let now = tok.decode(&out_ids, true);
            if now.len() > prev.len() && now.starts_with(&prev) {
                let piece = &now[prev.len()..];
                if head_done {
                    on_piece(piece);
                } else {
                    head_buf.push_str(piece);
                    if let Some(nl) = head_buf.find('\n') {
                        if head_buf[..nl].trim().eq_ignore_ascii_case("thought") {
                            let rest = head_buf[nl + 1..].to_string();
                            if !rest.is_empty() {
                                on_piece(&rest);
                            }
                        } else {
                            let all = head_buf.clone();
                            on_piece(&all);
                        }
                        head_done = true;
                        head_buf.clear();
                    } else if head_buf.len() > 40 {
                        let all = head_buf.clone();
                        on_piece(&all);
                        head_done = true;
                        head_buf.clear();
                    }
                }
            }
            prev = now;
            logits = m.decode_step(&mut cache, next, gref);
        }
        // Flush a short answer that never contained a newline (so it wasn't a header).
        if !head_done && !head_buf.is_empty() {
            on_piece(&head_buf);
        }
        out_ids.len()
    }
}

/// True if `path` is a Gemma-4 HuggingFace checkpoint dir (config.json model_type
/// contains `gemma4`). Cheap string check — no full parse.
pub fn is_gemma4(path: &Path) -> bool {
    if !path.is_dir() {
        return false;
    }
    std::fs::read_to_string(path.join("config.json"))
        .map(|s| s.contains("gemma4"))
        .unwrap_or(false)
}

/// True if `path` is a `.hos` capsule whose card records `arch=gemma4` — so the
/// loader routes it to the native decoder instead of the generic `Engine`.
pub fn is_gemma4_capsule(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    crate::format::read_card(path)
        .ok()
        .and_then(|c| {
            c.arch
                .get("architecture")
                .and_then(|v| v.as_str())
                .map(|s| s == "gemma4")
        })
        .unwrap_or(false)
}

/// True if a bare model name looks like a request for Gemma-4 (so we can resolve
/// it to a local checkpoint without the user typing a 200-char HF snapshot path).
pub fn is_gemma4_name(name: &str) -> bool {
    let n = name.to_lowercase();
    n == "gemma4" || n == "gemma-4" || n.contains("gemma-4") || n.contains("gemma4")
}

/// Search the usual model roots (and the HF hub cache) for a local Gemma-4
/// checkpoint and return the newest snapshot dir whose config.json is gemma4.
pub fn find_checkpoint() -> Option<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(d) = std::env::var("HOS_MODELS_DIR") {
        roots.push(PathBuf::from(d));
    }
    if let Some(home) = std::env::var_os("HOME") {
        let h = Path::new(&home);
        roots.push(h.join("Documents/hos/models"));
        roots.push(h.join("Documents/HOS/models"));
        roots.push(h.join(".hos/models"));
        roots.push(h.join(".cache/huggingface/hub"));
    }
    for root in roots {
        // Look directly under the root and under a nested `huggingface/hub`.
        for hub in [root.clone(), root.join("huggingface/hub")] {
            let Ok(rd) = std::fs::read_dir(&hub) else {
                continue;
            };
            for e in rd.flatten() {
                let p = e.path();
                let nm = e.file_name().to_string_lossy().to_lowercase();
                if !p.is_dir() || !nm.contains("gemma") || !nm.contains('4') {
                    continue;
                }
                // HF layout: <repo>/snapshots/<hash>/config.json ; else the dir itself.
                let snaps = p.join("snapshots");
                let search = if snaps.is_dir() { snaps } else { p.clone() };
                if is_gemma4(&search) {
                    return Some(search);
                }
                if let Ok(sd) = std::fs::read_dir(&search) {
                    for s in sd.flatten() {
                        let sp = s.path();
                        if is_gemma4(&sp) {
                            return Some(sp);
                        }
                    }
                }
            }
        }
    }
    None
}
