//! Native Gemma (SentencePiece-BPE) tokenizer for HOS.
//!
//! STRICTLY ADDITIVE. This module implements Google Gemma's tokenizer directly
//! from `tokenizer.json` so that `--gemma4 -p "<text>"` can tokenize raw text.
//! It does NOT touch HOS's existing byte-BPE `src/tokenizer.rs`.
//!
//! The Gemma tokenizer.json is a HuggingFace `tokenizers` BPE model with:
//!   - `normalizer` = Replace(" " -> "\u{2581}")  (metaspace)
//!   - `pre_tokenizer` = Split(" ", MergedWithPrevious)  (a no-op after
//!     normalization, since there are no ASCII spaces left)
//!   - `model` = BPE { byte_fallback, ignore_merges, fuse_unk, ... }
//!   - `decoder` = Sequence[ Replace("\u{2581}" -> " "), ByteFallback, Fuse ]
//!   - 24 `added_tokens` (specials: <pad> <eos> <bos> <unk> <mask> ...).
//!
//! Byte-fallback tokens are the `<0x00>`..`<0xFF>` entries in the vocab.

use crate::error::{HosError, Result};
use std::collections::HashMap;
use std::path::Path;

const METASPACE: char = '\u{2581}'; // U+2581 LOWER ONE EIGHTH BLOCK ("▁")

/// A loaded Gemma SentencePiece-BPE tokenizer.
pub struct GemmaTokenizer {
    /// token string -> id
    vocab: HashMap<String, u32>,
    /// id -> token string
    inv: HashMap<u32, String>,
    /// (left, right) -> merge rank (lower = higher priority)
    merges: HashMap<(String, String), u32>,
    /// added/special tokens: content -> id (sorted-longest-first for matching)
    added: Vec<(String, u32)>,
    /// set of ids that are "special" (from added_tokens with special=true)
    special_ids: std::collections::HashSet<u32>,
    /// byte value (0..=255) -> id of the `<0xXX>` fallback token
    byte_fallback: [u32; 256],
    /// id -> byte value, for the 256 `<0xXX>` fallback tokens (decode)
    byte_id_to_val: HashMap<u32, u8>,
    /// BPE `ignore_merges` flag
    ignore_merges: bool,
    /// id of `<bos>`
    pub bos_id: u32,
    /// id of `<eos>`
    pub eos_id: u32,
}

impl GemmaTokenizer {
    /// Load from a `tokenizer.json` path.
    pub fn load(tokenizer_json_path: &Path) -> Result<Self> {
        let raw = std::fs::read(tokenizer_json_path).map_err(|e| {
            HosError::Format(format!(
                "gemma_tok: read {}: {e}",
                tokenizer_json_path.display()
            ))
        })?;
        let j: serde_json::Value = serde_json::from_slice(&raw)
            .map_err(|e| HosError::Format(format!("gemma_tok: parse json: {e}")))?;
        Self::from_value(&j)
    }

    /// Build from an already-parsed `tokenizer.json` value — lets a `.hos` capsule
    /// embed the tokenizer JSON and reconstruct the tokenizer without the source dir.
    pub fn from_value(j: &serde_json::Value) -> Result<Self> {
        let model = &j["model"];

        // ---- vocab ----
        let vobj = model["vocab"]
            .as_object()
            .ok_or_else(|| HosError::Format("gemma_tok: model.vocab not an object".into()))?;
        let mut vocab: HashMap<String, u32> = HashMap::with_capacity(vobj.len());
        let mut inv: HashMap<u32, String> = HashMap::with_capacity(vobj.len());
        for (k, v) in vobj {
            let id = v
                .as_u64()
                .ok_or_else(|| HosError::Format("gemma_tok: vocab id not int".into()))?
                as u32;
            vocab.insert(k.clone(), id);
            inv.insert(id, k.clone());
        }

        // ---- merges ----
        // Each entry may be a JSON array ["A","B"] or a single string "A B".
        let marr = model["merges"]
            .as_array()
            .ok_or_else(|| HosError::Format("gemma_tok: model.merges not an array".into()))?;
        let mut merges: HashMap<(String, String), u32> = HashMap::with_capacity(marr.len());
        for (rank, m) in marr.iter().enumerate() {
            let (a, b) = if let Some(pair) = m.as_array() {
                let a = pair
                    .first()
                    .and_then(|x| x.as_str())
                    .ok_or_else(|| HosError::Format("gemma_tok: merge pair[0]".into()))?;
                let b = pair
                    .get(1)
                    .and_then(|x| x.as_str())
                    .ok_or_else(|| HosError::Format("gemma_tok: merge pair[1]".into()))?;
                (a.to_string(), b.to_string())
            } else if let Some(s) = m.as_str() {
                let mut it = s.splitn(2, ' ');
                let a = it.next().unwrap_or("").to_string();
                let b = it.next().unwrap_or("").to_string();
                (a, b)
            } else {
                return Err(HosError::Format("gemma_tok: bad merge entry".into()));
            };
            merges.entry((a, b)).or_insert(rank as u32);
        }

        // ---- added / special tokens ----
        let mut added: Vec<(String, u32)> = Vec::new();
        let mut special_ids = std::collections::HashSet::new();
        if let Some(at) = j["added_tokens"].as_array() {
            for t in at {
                let content = t["content"].as_str().unwrap_or("").to_string();
                let id = t["id"].as_u64().unwrap_or(0) as u32;
                if content.is_empty() {
                    continue;
                }
                added.push((content.clone(), id));
                // ensure inv/vocab know it too
                inv.entry(id).or_insert_with(|| content.clone());
                vocab.entry(content).or_insert(id);
                if t["special"].as_bool().unwrap_or(false) {
                    special_ids.insert(id);
                }
            }
        }
        // longest content first, so the scanner prefers the longest match.
        added.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        // ---- byte-fallback table ----
        let mut byte_fallback = [u32::MAX; 256];
        let mut byte_id_to_val = HashMap::with_capacity(256);
        for b in 0u16..=255 {
            let tok = format!("<0x{:02X}>", b);
            if let Some(&id) = vocab.get(&tok) {
                byte_fallback[b as usize] = id;
                byte_id_to_val.insert(id, b as u8);
            }
        }

        let ignore_merges = model["ignore_merges"].as_bool().unwrap_or(false);
        let bos_id = vocab.get("<bos>").copied().unwrap_or(2);
        let eos_id = vocab.get("<eos>").copied().unwrap_or(1);

        Ok(GemmaTokenizer {
            vocab,
            inv,
            merges,
            added,
            special_ids,
            byte_fallback,
            byte_id_to_val,
            ignore_merges,
            bos_id,
            eos_id,
        })
    }

    /// Convenience: locate `tokenizer.json` inside a model directory and load it.
    pub fn load_from_model_dir(dir: &Path) -> Result<Self> {
        let p = dir.join("tokenizer.json");
        Self::load(&p)
    }

    /// Encode raw text into token ids. Prepends `<bos>` if `add_bos`.
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if add_bos {
            out.push(self.bos_id);
        }
        self.encode_normal_and_specials(text, &mut out);
        out
    }

    /// Split `text` around any added/special token occurrences (literal match,
    /// longest-first), emitting specials as their id and BPE'ing the rest.
    fn encode_normal_and_specials(&self, text: &str, out: &mut Vec<u32>) {
        let mut normal_start = 0usize;
        let mut i = 0usize;
        while i < text.len() {
            // longest added-token content matching at byte position i
            let rest = &text[i..];
            let mut best: Option<(usize, u32)> = None;
            for (content, id) in &self.added {
                if content.len() <= rest.len() && rest.as_bytes().starts_with(content.as_bytes()) {
                    // `added` is sorted longest-first, so the first hit is longest.
                    best = Some((content.len(), *id));
                    break;
                }
            }
            if let Some((len, id)) = best {
                if normal_start < i {
                    self.bpe_span(&text[normal_start..i], out);
                }
                out.push(id);
                i += len;
                normal_start = i;
            } else {
                // advance one full char
                let ch_len = rest.chars().next().map(|c| c.len_utf8()).unwrap_or(1);
                i += ch_len;
            }
        }
        if normal_start < text.len() {
            self.bpe_span(&text[normal_start..], out);
        }
    }

    /// Normalize + BPE a span of "normal" (non-special) text into ids.
    fn bpe_span(&self, span: &str, out: &mut Vec<u32>) {
        if span.is_empty() {
            return;
        }
        // Normalizer: Replace(' ' -> '▁').  (No NFKC.)
        let norm: String = span
            .chars()
            .map(|c| if c == ' ' { METASPACE } else { c })
            .collect();
        if norm.is_empty() {
            return;
        }

        // ignore_merges: if the entire piece is directly in the vocab, emit it.
        if self.ignore_merges {
            if let Some(&id) = self.vocab.get(&norm) {
                out.push(id);
                return;
            }
        }

        // Symbols start as individual unicode scalars (chars).
        let mut symbols: Vec<String> = norm.chars().map(|c| c.to_string()).collect();

        // Rank-based BPE: repeatedly merge the adjacent pair with the lowest
        // merge rank (leftmost on ties), matching HF `tokenizers` heap order.
        loop {
            let mut best_rank = u32::MAX;
            let mut best_k: Option<usize> = None;
            for k in 0..symbols.len().saturating_sub(1) {
                let key = (symbols[k].clone(), symbols[k + 1].clone());
                if let Some(&r) = self.merges.get(&key) {
                    if r < best_rank {
                        best_rank = r;
                        best_k = Some(k);
                    }
                }
            }
            let Some(k) = best_k else { break };
            let merged = format!("{}{}", symbols[k], symbols[k + 1]);
            symbols[k] = merged;
            symbols.remove(k + 1);
        }

        // Map symbols to ids, with byte_fallback for anything not in vocab.
        for s in &symbols {
            if let Some(&id) = self.vocab.get(s) {
                out.push(id);
            } else {
                for byte in s.bytes() {
                    let id = self.byte_fallback[byte as usize];
                    if id != u32::MAX {
                        out.push(id);
                    }
                }
            }
        }
    }

    /// Decode ids back to text using the decoder sequence:
    /// Replace('▁' -> ' '), then ByteFallback (runs of `<0xXX>` -> raw bytes),
    /// then Fuse. If `skip_special`, special-token ids are dropped.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        let mut out = String::new();
        let mut byte_buf: Vec<u8> = Vec::new();
        for &id in ids {
            if let Some(&b) = self.byte_id_to_val.get(&id) {
                byte_buf.push(b);
                continue;
            }
            if !byte_buf.is_empty() {
                out.push_str(&String::from_utf8_lossy(&byte_buf));
                byte_buf.clear();
            }
            if skip_special && self.special_ids.contains(&id) {
                continue;
            }
            if let Some(s) = self.inv.get(&id) {
                out.push_str(&s.replace(METASPACE, " "));
            }
        }
        if !byte_buf.is_empty() {
            out.push_str(&String::from_utf8_lossy(&byte_buf));
        }
        out
    }

    /// Number of vocab entries (for diagnostics).
    pub fn vocab_len(&self) -> usize {
        self.vocab.len()
    }
}
