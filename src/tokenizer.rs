//! Byte-level BPE tokenizer, reading vocab + merges from GGUF metadata.
//!
//! Encoding is byte-exact: it reproduces the GPT-2-family pre-tokenization
//! (the regex split that decides word boundaries) before applying BPE merges,
//! and dispatches the exact variant — `gpt-2`, `qwen2`, or `llama-bpe` — from
//! the model's `tokenizer.ggml.pre` field. Decoding has always been exact.
//!
//! The pre-tokenizer is implemented by hand (no `regex` crate): the GPT-2
//! alternation is evaluated in order at each position, with greedy quantifiers
//! and the `\s+(?!\S)` look-ahead emulated directly. Unicode letter/number
//! classes use Rust std's `char::is_alphabetic`/`is_numeric`, which match
//! `\p{L}`/`\p{N}` for the scripts these models actually tokenize.

use std::collections::HashMap;
use std::path::Path;

use serde_json::Value as J;

use crate::error::{HosError, Result};
use crate::gguf::Gguf;

pub struct Tokenizer {
    pub id_to_token: Vec<String>,
    token_to_id: HashMap<String, u32>,
    merge_rank: HashMap<(String, String), u32>,
    byte_decoder: HashMap<char, u8>,
    byte_encoder: [char; 256],
    pre: PreCfg,
    /// SentencePiece (llama) mode vs GPT-2 byte-level BPE.
    spm: bool,
    /// SPM token scores (aligned to id_to_token); empty in byte-level mode.
    scores: Vec<f32>,
    pub bos: Option<u32>,
    pub eos: Option<u32>,
}

/// SPM marker: the U+2581 "lower one eighth block" used for spaces.
const SPM_SPACE: char = '\u{2581}';

/// Knobs that distinguish the GPT-2-family pre-tokenizer regexes. Selected from
/// the GGUF `tokenizer.ggml.pre` tag so each model splits exactly as it was
/// trained.
#[derive(Clone, Copy)]
struct PreCfg {
    /// contractions match either case ('S as well as 's) — true for qwen2/llama3
    contraction_ci: bool,
    /// letters may take any single non-letter/non-digit/non-newline char as a
    /// prefix (`[^\r\n\p{L}\p{N}]?`); false = only an optional space (` ?`).
    letter_any_prefix: bool,
    /// max digits per token: 0 = greedy `\p{N}+` (gpt-2), else `\p{N}{1,N}`.
    digit_group: usize,
    /// symbol runs may take an optional leading space (` ?[^\s\p{L}\p{N}]+`).
    /// gpt-2 has the space; so do qwen2/llama3.
    symbol_space_prefix: bool,
    /// symbol runs absorb trailing newlines (`...[\r\n]*`) — qwen2/llama3.
    symbol_newline_tail: bool,
    /// the `\s*[\r\n]+` alternative is present — qwen2/llama3.
    newline_rule: bool,
    /// digits may take an optional leading space (` ?\p{N}+`) — gpt-2 only.
    digit_space_prefix: bool,
}

impl PreCfg {
    fn from_pre(pre: &str) -> PreCfg {
        match pre {
            "qwen2" => PreCfg {
                contraction_ci: true,
                letter_any_prefix: true,
                digit_group: 1,
                symbol_space_prefix: true,
                symbol_newline_tail: true,
                newline_rule: true,
                digit_space_prefix: false,
            },
            "llama-bpe" | "llama3" | "llama-v3" | "llama-bpe-v3" => PreCfg {
                contraction_ci: true,
                letter_any_prefix: true,
                digit_group: 3,
                symbol_space_prefix: true,
                symbol_newline_tail: true,
                newline_rule: true,
                digit_space_prefix: false,
            },
            // "gpt-2", "smollm", "olmo", "default", and unknown tags
            _ => PreCfg {
                contraction_ci: false,
                letter_any_prefix: false,
                digit_group: 0,
                symbol_space_prefix: true,
                symbol_newline_tail: false,
                newline_rule: false,
                digit_space_prefix: true,
            },
        }
    }
}

/// Derive the pre-tokenizer variant from a HuggingFace `tokenizer.json`. The
/// regex-driven families (Llama-3's `\p{N}{1,3}`, Qwen2's `(?i:...)`) are matched
/// on the serialized pre_tokenizer; otherwise the GPT-2 base is used, honoring an
/// explicit individual-digits split (SmolLM2) when present.
fn precfg_from_tokenizer(tj: &J) -> PreCfg {
    let pt = tj
        .get("pre_tokenizer")
        .map(|v| v.to_string())
        .unwrap_or_default();
    if pt.contains("{1,3}") {
        return PreCfg::from_pre("llama-bpe");
    }
    if pt.contains("?i:") {
        return PreCfg::from_pre("qwen2");
    }
    let mut c = PreCfg::from_pre("gpt-2");
    if pt.contains("\"Digits\"") && pt.contains("\"individual_digits\":true") {
        c.digit_group = 1;
        c.digit_space_prefix = false;
    }
    c
}

/// Read a protobuf varint at `i`, returning (value, next_index).
fn read_varint(b: &[u8], mut i: usize) -> Result<(u64, usize)> {
    let mut val = 0u64;
    let mut shift = 0u32;
    loop {
        let byte = *b
            .get(i)
            .ok_or_else(|| HosError::Format("truncated varint".into()))?;
        i += 1;
        val |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok((val, i));
        }
        shift += 7;
        if shift >= 64 {
            return Err(HosError::Format("varint too long".into()));
        }
    }
}

/// Parse a SentencePiece `ModelProto` (`tokenizer.model`) just enough to recover
/// the vocabulary: each `pieces` entry (field 1) is a `SentencePiece` message with
/// a `piece` string (field 1) and `score` float32 (field 2). Other top-level
/// fields (trainer/normalizer specs) are skipped. Hand-rolled — no protobuf dep.
fn parse_sentencepiece(b: &[u8]) -> Result<(Vec<String>, Vec<f32>)> {
    fn parse_piece(b: &[u8]) -> Result<(String, f32)> {
        let (mut piece, mut score) = (String::new(), 0.0f32);
        let mut i = 0;
        while i < b.len() {
            let (tag, ni) = read_varint(b, i)?;
            i = ni;
            let (field, wire) = (tag >> 3, tag & 7);
            match (field, wire) {
                (1, 2) => {
                    let (len, ni) = read_varint(b, i)?;
                    let end = ni + len as usize;
                    if end > b.len() {
                        return Err(HosError::Format("piece string out of bounds".into()));
                    }
                    piece = String::from_utf8_lossy(&b[ni..end]).into_owned();
                    i = end;
                }
                (2, 5) => {
                    if i + 4 > b.len() {
                        return Err(HosError::Format("score out of bounds".into()));
                    }
                    score = f32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
                    i += 4;
                }
                (_, 0) => i = read_varint(b, i)?.1,
                (_, 1) => i += 8,
                (_, 5) => i += 4,
                (_, 2) => {
                    let (len, ni) = read_varint(b, i)?;
                    i = ni + len as usize;
                }
                _ => return Err(HosError::Format(format!("bad wire type {wire}"))),
            }
        }
        Ok((piece, score))
    }

    let (mut pieces, mut scores) = (Vec::new(), Vec::new());
    let mut i = 0;
    while i < b.len() {
        let (tag, ni) = read_varint(b, i)?;
        i = ni;
        let (field, wire) = (tag >> 3, tag & 7);
        match wire {
            0 => i = read_varint(b, i)?.1,
            1 => i += 8,
            5 => i += 4,
            2 => {
                let (len, ni) = read_varint(b, i)?;
                let end = ni + len as usize;
                if end > b.len() {
                    return Err(HosError::Format("protobuf length out of bounds".into()));
                }
                if field == 1 {
                    let (p, s) = parse_piece(&b[ni..end])?;
                    pieces.push(p);
                    scores.push(s);
                }
                i = end;
            }
            _ => return Err(HosError::Format(format!("bad wire type {wire}"))),
        }
    }
    if pieces.is_empty() {
        return Err(HosError::Format(
            "tokenizer.model has no SentencePiece pieces".into(),
        ));
    }
    Ok((pieces, scores))
}

/// GPT-2 reversible byte<->unicode mapping, so every byte is a printable char.
fn bytes_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut mapped = vec![false; 256];
    // printable ranges that map to themselves
    for b in b'!'..=b'~' {
        table[b as usize] = b as char;
        mapped[b as usize] = true;
    }
    for b in 0xA1u8..=0xAC {
        table[b as usize] = b as char;
        mapped[b as usize] = true;
    }
    for b in 0xAEu8..=0xFF {
        table[b as usize] = b as char;
        mapped[b as usize] = true;
    }
    // everything else gets shifted into the 256+ private range
    let mut n = 0u32;
    for b in 0..256usize {
        if !mapped[b] {
            table[b] = char::from_u32(256 + n).unwrap();
            n += 1;
        }
    }
    table
}

#[inline]
fn is_l(c: char) -> bool {
    c.is_alphabetic()
}
#[inline]
fn is_n(c: char) -> bool {
    c.is_numeric()
}
#[inline]
fn is_ws(c: char) -> bool {
    c.is_whitespace()
}
#[inline]
fn is_nl(c: char) -> bool {
    c == '\r' || c == '\n'
}

/// Pre-tokenize `text` into the GPT-2-family pieces (byte-exact word boundaries)
/// per `cfg`. Returns byte ranges into `text`. This is the step that GGUF /
/// safetensors loaders get wrong when they skip the regex; doing it right is
/// what makes our encode match the reference tokenizer.
fn pretokenize(text: &str, cfg: PreCfg) -> Vec<(usize, usize)> {
    let cv: Vec<(usize, char)> = text.char_indices().collect();
    let n = cv.len();
    let blen = text.len();
    // byte offset of char index k (end-exclusive aware)
    let at = |k: usize| if k < n { cv[k].0 } else { blen };

    let mut out = Vec::new();
    let mut i = 0usize;
    while i < n {
        let c = cv[i].1;

        // 1. contractions: 's 't 're 've 'm 'll 'd  (case per cfg)
        if c == '\'' && i + 1 < n {
            let lc = |k: usize| {
                let ch = cv[k].1;
                if cfg.contraction_ci {
                    ch.to_ascii_lowercase()
                } else {
                    ch
                }
            };
            let c1 = lc(i + 1);
            let two = if i + 2 < n { Some(lc(i + 2)) } else { None };
            let consumed = if matches!(c1, 's' | 't' | 'm' | 'd') {
                2
            } else if matches!(
                (c1, two),
                ('r', Some('e')) | ('v', Some('e')) | ('l', Some('l'))
            ) {
                3
            } else {
                0
            };
            if consumed > 0 {
                out.push((at(i), at(i + consumed)));
                i += consumed;
                continue;
            }
        }

        // 2. letters, with optional prefix
        //    qwen2/llama: [^\r\n\p{L}\p{N}]?\p{L}+     gpt-2: ` ?\p{L}+`
        {
            let mut j = i;
            if cfg.letter_any_prefix {
                let p = cv[j].1;
                if !is_nl(p) && !is_l(p) && !is_n(p) && j + 1 < n && is_l(cv[j + 1].1) {
                    j += 1;
                }
            } else if cv[j].1 == ' ' && j + 1 < n && is_l(cv[j + 1].1) {
                j += 1;
            }
            if is_l(cv[j].1) {
                while j < n && is_l(cv[j].1) {
                    j += 1;
                }
                out.push((at(i), at(j)));
                i = j;
                continue;
            }
        }

        // 3. digits.  gpt-2: ` ?\p{N}+`   qwen2: \p{N}   llama3: \p{N}{1,3}
        {
            let mut j = i;
            if cfg.digit_space_prefix && cv[j].1 == ' ' && j + 1 < n && is_n(cv[j + 1].1) {
                j += 1;
            }
            if is_n(cv[j].1) {
                let start = j;
                while j < n && is_n(cv[j].1) {
                    j += 1;
                    if cfg.digit_group != 0 && j - start >= cfg.digit_group {
                        break;
                    }
                }
                out.push((at(i), at(j)));
                i = j;
                continue;
            }
        }

        // 4. symbols: ` ?[^\s\p{L}\p{N}]+` (+ optional [\r\n]* tail for qwen/llama)
        {
            let mut j = i;
            if cfg.symbol_space_prefix && cv[j].1 == ' ' && j + 1 < n {
                let nc = cv[j + 1].1;
                if !is_ws(nc) && !is_l(nc) && !is_n(nc) {
                    j += 1;
                }
            }
            let sc = cv[j].1;
            if !is_ws(sc) && !is_l(sc) && !is_n(sc) {
                while j < n {
                    let ch = cv[j].1;
                    if is_ws(ch) || is_l(ch) || is_n(ch) {
                        break;
                    }
                    j += 1;
                }
                if cfg.symbol_newline_tail {
                    while j < n && is_nl(cv[j].1) {
                        j += 1;
                    }
                }
                out.push((at(i), at(j)));
                i = j;
                continue;
            }
        }

        // 5/6/7. whitespace: \s*[\r\n]+ | \s+(?!\S) | \s+   (in that order)
        {
            // maximal whitespace run [i, e)
            let mut e = i;
            while e < n && is_ws(cv[e].1) {
                e += 1;
            }
            debug_assert!(e > i, "whitespace branch reached on a non-space char");

            // \s*[\r\n]+ : if the run contains a newline, match up to the last one
            if cfg.newline_rule {
                let mut last_nl = None;
                for k in i..e {
                    if is_nl(cv[k].1) {
                        last_nl = Some(k);
                    }
                }
                if let Some(k) = last_nl {
                    out.push((at(i), at(k + 1)));
                    i = k + 1;
                    continue;
                }
            }

            // \s+(?!\S): if the run is followed by a non-space, hold back the
            // last whitespace char (it becomes the next token's prefix); if the
            // run ends the string, take all of it.
            if e == n {
                out.push((at(i), at(e)));
                i = e;
            } else if e - 1 > i {
                out.push((at(i), at(e - 1)));
                i = e - 1;
            } else {
                // single whitespace before a non-space: bare \s+
                out.push((at(i), at(e)));
                i = e;
            }
        }
    }
    out
}

impl Tokenizer {
    /// Serialize the tokenizer's reconstructable state to JSON — source-agnostic
    /// (works for any built tokenizer, GGUF- or HF-origin). Stored in a runnable
    /// `.hos` capsule's card and rebuilt by [`from_value`](Self::from_value).
    pub fn to_value(&self) -> serde_json::Value {
        let mut merges: Vec<(&(String, String), &u32)> = self.merge_rank.iter().collect();
        merges.sort_by_key(|(_, r)| **r);
        let merges: Vec<String> = merges
            .iter()
            .map(|((a, b), _)| format!("{a} {b}"))
            .collect();
        serde_json::json!({
            "tokens": self.id_to_token,
            "merges": merges,
            "scores": self.scores,
            "bos": self.bos,
            "eos": self.eos,
            "spm": self.spm,
            "pre": {
                "contraction_ci": self.pre.contraction_ci,
                "letter_any_prefix": self.pre.letter_any_prefix,
                "digit_group": self.pre.digit_group,
                "symbol_space_prefix": self.pre.symbol_space_prefix,
                "symbol_newline_tail": self.pre.symbol_newline_tail,
                "newline_rule": self.pre.newline_rule,
                "digit_space_prefix": self.pre.digit_space_prefix,
            }
        })
    }

    /// Rebuild a tokenizer from [`to_value`](Self::to_value) output.
    pub fn from_value(v: &serde_json::Value) -> Result<Tokenizer> {
        let id_to_token: Vec<String> = v["tokens"]
            .as_array()
            .ok_or_else(|| HosError::MissingMeta("tokenizer tokens".into()))?
            .iter()
            .map(|x| x.as_str().unwrap_or("").to_string())
            .collect();
        let mut token_to_id = HashMap::with_capacity(id_to_token.len());
        for (i, t) in id_to_token.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }
        let mut merge_rank = HashMap::new();
        if let Some(m) = v["merges"].as_array() {
            for (rank, s) in m.iter().enumerate() {
                if let Some(s) = s.as_str() {
                    if let Some((a, b)) = s.split_once(' ') {
                        merge_rank.insert((a.to_string(), b.to_string()), rank as u32);
                    }
                }
            }
        }
        let scores: Vec<f32> = v["scores"]
            .as_array()
            .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect())
            .unwrap_or_default();
        let byte_encoder = bytes_to_unicode();
        let mut byte_decoder = HashMap::new();
        for (b, &c) in byte_encoder.iter().enumerate() {
            byte_decoder.insert(c, b as u8);
        }
        let p = &v["pre"];
        let pb = |k: &str, d: bool| p[k].as_bool().unwrap_or(d);
        let pre = PreCfg {
            contraction_ci: pb("contraction_ci", false),
            letter_any_prefix: pb("letter_any_prefix", false),
            digit_group: p["digit_group"].as_u64().unwrap_or(0) as usize,
            symbol_space_prefix: pb("symbol_space_prefix", true),
            symbol_newline_tail: pb("symbol_newline_tail", false),
            newline_rule: pb("newline_rule", false),
            digit_space_prefix: pb("digit_space_prefix", true),
        };
        Ok(Tokenizer {
            id_to_token,
            token_to_id,
            merge_rank,
            byte_decoder,
            byte_encoder,
            pre,
            spm: v["spm"].as_bool().unwrap_or(false),
            scores,
            bos: v["bos"].as_u64().map(|x| x as u32),
            eos: v["eos"].as_u64().map(|x| x as u32),
        })
    }

    /// Build a tokenizer from a runnable `.hos` capsule's metadata — the same
    /// `tokenizer.ggml.*` keys GGUF uses, stored as JSON. Mirrors `from_gguf`.
    pub fn from_hos(meta: &serde_json::Value) -> Result<Tokenizer> {
        let arr = |k: &str| meta.get(k).and_then(|v| v.as_array());
        let tokens = arr("tokenizer.ggml.tokens")
            .ok_or_else(|| HosError::MissingMeta("tokenizer.ggml.tokens".into()))?;
        let id_to_token: Vec<String> = tokens
            .iter()
            .map(|v| v.as_str().unwrap_or("").to_string())
            .collect();
        let mut token_to_id = HashMap::with_capacity(id_to_token.len());
        for (i, t) in id_to_token.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }

        let mut merge_rank = HashMap::new();
        if let Some(merges) = arr("tokenizer.ggml.merges") {
            for (rank, m) in merges.iter().enumerate() {
                if let Some(s) = m.as_str() {
                    if let Some((a, b)) = s.split_once(' ') {
                        merge_rank.insert((a.to_string(), b.to_string()), rank as u32);
                    }
                }
            }
        }

        let byte_encoder = bytes_to_unicode();
        let mut byte_decoder = HashMap::new();
        for (b, &c) in byte_encoder.iter().enumerate() {
            byte_decoder.insert(c, b as u8);
        }

        let bos = meta
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        let eos = meta
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        let pre = PreCfg::from_pre(
            meta.get("tokenizer.ggml.pre")
                .and_then(|v| v.as_str())
                .unwrap_or("default"),
        );
        let spm = meta.get("tokenizer.ggml.model").and_then(|v| v.as_str()) == Some("llama");
        let scores: Vec<f32> = arr("tokenizer.ggml.scores")
            .map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect())
            .unwrap_or_default();

        Ok(Tokenizer {
            id_to_token,
            token_to_id,
            merge_rank,
            byte_decoder,
            byte_encoder,
            pre,
            spm,
            scores,
            bos,
            eos,
        })
    }

    pub fn from_gguf(g: &Gguf) -> Result<Tokenizer> {
        let tokens = g
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .ok_or_else(|| HosError::MissingMeta("tokenizer.ggml.tokens".into()))?;
        let id_to_token: Vec<String> = tokens
            .iter()
            .map(|v| v.as_str().unwrap_or("").to_string())
            .collect();
        let mut token_to_id = HashMap::with_capacity(id_to_token.len());
        for (i, t) in id_to_token.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }

        let mut merge_rank = HashMap::new();
        if let Some(merges) = g
            .metadata
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.as_array())
        {
            for (rank, m) in merges.iter().enumerate() {
                if let Some(s) = m.as_str() {
                    if let Some((a, b)) = s.split_once(' ') {
                        merge_rank.insert((a.to_string(), b.to_string()), rank as u32);
                    }
                }
            }
        }

        let byte_encoder = bytes_to_unicode();
        let mut byte_decoder = HashMap::new();
        for (b, &c) in byte_encoder.iter().enumerate() {
            byte_decoder.insert(c, b as u8);
        }

        let bos = g.meta_u64("tokenizer.ggml.bos_token_id").map(|v| v as u32);
        let eos = g.meta_u64("tokenizer.ggml.eos_token_id").map(|v| v as u32);
        let pre = PreCfg::from_pre(g.meta_str("tokenizer.ggml.pre").unwrap_or("default"));

        // SentencePiece ("llama") models tokenize by merge-score, use ▁ for
        // spaces, and fall back to <0xNN> byte tokens — distinct from GPT-2 BPE.
        let spm = g.meta_str("tokenizer.ggml.model") == Some("llama");
        let scores: Vec<f32> = g
            .metadata
            .get("tokenizer.ggml.scores")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().map(|v| v.as_f32().unwrap_or(0.0)).collect())
            .unwrap_or_default();

        Ok(Tokenizer {
            id_to_token,
            token_to_id,
            merge_rank,
            byte_decoder,
            byte_encoder,
            pre,
            spm,
            scores,
            bos,
            eos,
        })
    }

    /// Build a tokenizer from a HuggingFace checkpoint folder. Routes by family:
    /// byte-level BPE (Ġ spaces — Llama-3 / Qwen2 / SmolLM2 / OLMoE) reads
    /// `tokenizer.json`; SentencePiece (▁ spaces — Gemma / Llama-2 / Mistral /
    /// Phi-3) reads the `tokenizer.model` protobuf. The two are told apart by the
    /// presence of a `ByteLevel` stage in `tokenizer.json`.
    pub fn from_hf(dir: &Path) -> Result<Tokenizer> {
        let tj = dir.join("tokenizer.json");
        let tm = dir.join("tokenizer.model");
        let byte_level = tj.exists() && std::fs::read_to_string(&tj)?.contains("ByteLevel");
        if byte_level {
            return Self::from_hf_bpe(dir);
        }
        if tm.exists() {
            return Self::from_hf_spm(dir);
        }
        if tj.exists() {
            // BPE without an explicit ByteLevel marker — try the byte-level path.
            return Self::from_hf_bpe(dir);
        }
        Err(HosError::Format(
            "checkpoint has no tokenizer.json or tokenizer.model".into(),
        ))
    }

    /// SentencePiece path (▁ spaces, `<0xNN>` byte-fallback) — parses the
    /// `tokenizer.model` protobuf for pieces + scores, then merges any HF
    /// `added_tokens` (special tokens that live past the SP vocab).
    fn from_hf_spm(dir: &Path) -> Result<Tokenizer> {
        let cfg = crate::safetensors::read_json(&dir.join("config.json"))?;
        let model_bytes = std::fs::read(dir.join("tokenizer.model"))?;
        let (pieces, mut scores) = parse_sentencepiece(&model_bytes)?;

        let size =
            (cfg["vocab_size"].as_u64().unwrap_or(pieces.len() as u64) as usize).max(pieces.len());
        let mut id_to_token = vec![String::new(); size];
        for (i, p) in pieces.iter().enumerate() {
            id_to_token[i] = p.clone();
        }
        scores.resize(size, 0.0);

        // special/added tokens (e.g. Phi-3's <|end|>) — from added_tokens.json
        // and/or tokenizer.json's added_tokens array
        if let Ok(added) = crate::safetensors::read_json(&dir.join("added_tokens.json")) {
            if let Some(obj) = added.as_object() {
                for (tok, id) in obj {
                    if let Some(i) = id.as_u64() {
                        if (i as usize) < size {
                            id_to_token[i as usize] = tok.clone();
                        }
                    }
                }
            }
        }
        if let Ok(tj) = crate::safetensors::read_json(&dir.join("tokenizer.json")) {
            if let Some(arr) = tj.get("added_tokens").and_then(|v| v.as_array()) {
                for a in arr {
                    if let (Some(i), Some(c)) = (a["id"].as_u64(), a["content"].as_str()) {
                        if (i as usize) < size {
                            id_to_token[i as usize] = c.to_string();
                        }
                    }
                }
            }
        }

        let mut token_to_id = HashMap::with_capacity(size);
        for (i, t) in id_to_token.iter().enumerate() {
            if !t.is_empty() {
                token_to_id.insert(t.clone(), i as u32);
            }
        }
        let byte_encoder = bytes_to_unicode();
        let mut byte_decoder = HashMap::new();
        for (b, &c) in byte_encoder.iter().enumerate() {
            byte_decoder.insert(c, b as u8);
        }
        let bos = cfg["bos_token_id"].as_u64().map(|v| v as u32);
        let eos = cfg["eos_token_id"]
            .as_u64()
            .or_else(|| {
                cfg["eos_token_id"]
                    .as_array()
                    .and_then(|a| a.first())
                    .and_then(|v| v.as_u64())
            })
            .map(|v| v as u32);

        Ok(Tokenizer {
            id_to_token,
            token_to_id,
            merge_rank: HashMap::new(),
            byte_decoder,
            byte_encoder,
            pre: PreCfg::from_pre("default"),
            spm: true,
            scores,
            bos,
            eos,
        })
    }

    /// Byte-level BPE path (Ġ spaces). Reads `tokenizer.json` `model.{vocab,merges}`.
    fn from_hf_bpe(dir: &Path) -> Result<Tokenizer> {
        let cfg = crate::safetensors::read_json(&dir.join("config.json"))?;
        let tj = crate::safetensors::read_json(&dir.join("tokenizer.json"))?;
        let model = &tj["model"];
        if model.get("type").and_then(|v| v.as_str()) != Some("BPE") {
            return Err(HosError::Format(
                "expected a byte-level BPE tokenizer.json".into(),
            ));
        }
        let vocab = model["vocab"]
            .as_object()
            .ok_or_else(|| HosError::Format("tokenizer.json: model.vocab missing".into()))?;
        let added = tj.get("added_tokens").and_then(|v| v.as_array());

        // size the id table to cover the embedding rows and any out-of-range
        // special token id
        let mut size = cfg["vocab_size"].as_u64().unwrap_or(0) as usize;
        for id in vocab.values() {
            if let Some(i) = id.as_u64() {
                size = size.max(i as usize + 1);
            }
        }
        if let Some(arr) = added {
            for a in arr {
                if let Some(i) = a["id"].as_u64() {
                    size = size.max(i as usize + 1);
                }
            }
        }
        let mut id_to_token = vec![String::new(); size];
        for (tok, id) in vocab {
            if let Some(i) = id.as_u64() {
                let i = i as usize;
                if i < id_to_token.len() {
                    id_to_token[i] = tok.clone();
                }
            }
        }
        // added/special tokens override (their content is the literal string)
        if let Some(arr) = added {
            for a in arr {
                if let (Some(i), Some(c)) = (a["id"].as_u64(), a["content"].as_str()) {
                    let i = i as usize;
                    if i < id_to_token.len() {
                        id_to_token[i] = c.to_string();
                    }
                }
            }
        }
        let mut token_to_id = HashMap::with_capacity(id_to_token.len());
        for (i, t) in id_to_token.iter().enumerate() {
            if !t.is_empty() {
                token_to_id.insert(t.clone(), i as u32);
            }
        }

        // merges: ["a b", ...] (older) or [["a","b"], ...] (newer)
        let mut merge_rank = HashMap::new();
        if let Some(merges) = model["merges"].as_array() {
            for (rank, m) in merges.iter().enumerate() {
                let pair = if let Some(s) = m.as_str() {
                    s.split_once(' ')
                        .map(|(a, b)| (a.to_string(), b.to_string()))
                } else if let Some(arr) = m.as_array() {
                    match (
                        arr.first().and_then(|v| v.as_str()),
                        arr.get(1).and_then(|v| v.as_str()),
                    ) {
                        (Some(a), Some(b)) => Some((a.to_string(), b.to_string())),
                        _ => None,
                    }
                } else {
                    None
                };
                if let Some(p) = pair {
                    merge_rank.insert(p, rank as u32);
                }
            }
        }

        let byte_encoder = bytes_to_unicode();
        let mut byte_decoder = HashMap::new();
        for (b, &c) in byte_encoder.iter().enumerate() {
            byte_decoder.insert(c, b as u8);
        }

        let pre = precfg_from_tokenizer(&tj);
        let bos = cfg["bos_token_id"].as_u64().map(|v| v as u32);
        let eos = cfg["eos_token_id"]
            .as_u64()
            .or_else(|| {
                cfg["eos_token_id"]
                    .as_array()
                    .and_then(|a| a.first())
                    .and_then(|v| v.as_u64())
            })
            .map(|v| v as u32);

        Ok(Tokenizer {
            id_to_token,
            token_to_id,
            merge_rank,
            byte_decoder,
            byte_encoder,
            pre,
            spm: false,
            scores: Vec::new(),
            bos,
            eos,
        })
    }

    /// SentencePiece encode: normalize spaces to ▁, then greedily merge the
    /// adjacent symbol pair with the highest score that is a vocab token, with
    /// `<0xNN>` byte-fallback for symbols absent from the vocabulary.
    fn spm_encode(&self, text: &str, ids: &mut Vec<u32>) {
        // SPM normalization: leading space + spaces -> ▁
        let norm: String = format!(" {text}").replace(' ', &SPM_SPACE.to_string());
        let mut syms: Vec<String> = norm.chars().map(|c| c.to_string()).collect();
        loop {
            let mut best: Option<(usize, f32)> = None;
            for i in 0..syms.len().saturating_sub(1) {
                let merged = format!("{}{}", syms[i], syms[i + 1]);
                if let Some(&id) = self.token_to_id.get(&merged) {
                    let sc = self.scores.get(id as usize).copied().unwrap_or(f32::MIN);
                    if best.map_or(true, |(_, b)| sc > b) {
                        best = Some((i, sc));
                    }
                }
            }
            let Some((i, _)) = best else { break };
            let merged = format!("{}{}", syms[i], syms[i + 1]);
            syms.splice(i..i + 2, [merged]);
        }
        for s in syms {
            if let Some(&id) = self.token_to_id.get(&s) {
                ids.push(id);
            } else {
                // byte-fallback: emit <0xNN> for each UTF-8 byte
                for b in s.bytes() {
                    if let Some(&id) = self.token_to_id.get(&format!("<0x{b:02X}>")) {
                        ids.push(id);
                    }
                }
            }
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Apply BPE merges to a sequence of single-char symbols.
    fn bpe(&self, mut symbols: Vec<String>) -> Vec<String> {
        loop {
            let mut best: Option<(usize, u32)> = None;
            for i in 0..symbols.len().saturating_sub(1) {
                let pair = (symbols[i].clone(), symbols[i + 1].clone());
                if let Some(&rank) = self.merge_rank.get(&pair) {
                    if best.map_or(true, |(_, r)| rank < r) {
                        best = Some((i, rank));
                    }
                }
            }
            let Some((i, _)) = best else { break };
            let merged = format!("{}{}", symbols[i], symbols[i + 1]);
            symbols.splice(i..i + 2, [merged]);
        }
        symbols
    }

    /// Look up the id of a literal token (e.g. a special/added token like
    /// `<|im_start|>`). The byte-exact `encode` path deliberately does not split
    /// on special tokens; chat templating uses this to splice them in directly.
    pub fn special_id(&self, s: &str) -> Option<u32> {
        self.token_to_id.get(s).copied()
    }

    /// Encode `text` to token ids. Pre-tokenizes into GPT-2-family pieces, maps
    /// each piece's bytes through the reversible byte<->unicode table, then BPEs
    /// each piece independently — the byte-exact path.
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut ids = Vec::new();
        if add_bos {
            if let Some(b) = self.bos {
                ids.push(b);
            }
        }
        if self.spm {
            self.spm_encode(text, &mut ids);
            return ids;
        }
        for (s, e) in pretokenize(text, self.pre) {
            // raw bytes of this piece -> byte-level chars
            let symbols: Vec<String> = text[s..e]
                .bytes()
                .map(|b| self.byte_encoder[b as usize].to_string())
                .collect();
            for sym in self.bpe(symbols) {
                if let Some(&id) = self.token_to_id.get(&sym) {
                    ids.push(id);
                } else {
                    // fall back to per-char ids if a merge result isn't in vocab
                    for c in sym.chars() {
                        if let Some(&id) = self.token_to_id.get(&c.to_string()) {
                            ids.push(id);
                        }
                    }
                }
            }
        }
        ids
    }

    /// Decode a single token id to its raw bytes appended onto `out`.
    pub fn decode_into(&self, id: u32, out: &mut Vec<u8>) {
        let Some(tok) = self.id_to_token.get(id as usize) else {
            return;
        };
        if self.spm {
            // <0xNN> byte-fallback token -> raw byte; ▁ -> space; else UTF-8 bytes
            if tok.len() == 6 && tok.starts_with("<0x") && tok.ends_with('>') {
                if let Ok(b) = u8::from_str_radix(&tok[3..5], 16) {
                    out.push(b);
                }
                return;
            }
            for c in tok.chars() {
                if c == SPM_SPACE {
                    out.push(b' ');
                } else {
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                }
            }
            return;
        }
        for c in tok.chars() {
            if let Some(&b) = self.byte_decoder.get(&c) {
                out.push(b);
            }
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            self.decode_into(id, &mut bytes);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces(text: &str, cfg: PreCfg) -> Vec<String> {
        pretokenize(text, cfg)
            .into_iter()
            .map(|(s, e)| text[s..e].to_string())
            .collect()
    }

    #[test]
    fn gpt2_leading_space_attaches_to_word() {
        let cfg = PreCfg::from_pre("gpt-2");
        assert_eq!(pieces("hello world", cfg), vec!["hello", " world"]);
    }

    #[test]
    fn gpt2_runs_of_spaces_hold_back_last() {
        // "a   b" -> "a", "  ", " b"  (\s+(?!\S) takes all but the last space,
        // which becomes the prefix of " b")
        let cfg = PreCfg::from_pre("gpt-2");
        assert_eq!(pieces("a   b", cfg), vec!["a", "  ", " b"]);
    }

    #[test]
    fn gpt2_contractions_lowercase_only() {
        let cfg = PreCfg::from_pre("gpt-2");
        assert_eq!(pieces("I'm", cfg), vec!["I", "'m"]);
        // uppercase contraction is NOT special-cased in gpt-2
        assert_eq!(pieces("I'M", cfg), vec!["I", "'", "M"]);
    }

    #[test]
    fn qwen2_contractions_case_insensitive() {
        let cfg = PreCfg::from_pre("qwen2");
        assert_eq!(pieces("I'M", cfg), vec!["I", "'M"]);
        assert_eq!(pieces("don't", cfg), vec!["don", "'t"]);
    }

    #[test]
    fn qwen2_digits_split_individually() {
        let cfg = PreCfg::from_pre("qwen2");
        assert_eq!(pieces("2024", cfg), vec!["2", "0", "2", "4"]);
    }

    #[test]
    fn llama3_digits_group_by_three() {
        let cfg = PreCfg::from_pre("llama-bpe");
        assert_eq!(pieces("2024", cfg), vec!["202", "4"]);
        assert_eq!(pieces("1234567", cfg), vec!["123", "456", "7"]);
    }

    #[test]
    fn gpt2_digits_greedy() {
        let cfg = PreCfg::from_pre("gpt-2");
        assert_eq!(pieces("2024", cfg), vec!["2024"]);
    }

    #[test]
    fn symbols_take_leading_space() {
        let cfg = PreCfg::from_pre("qwen2");
        assert_eq!(pieces("a = b", cfg), vec!["a", " =", " b"]);
    }

    #[test]
    fn trailing_whitespace_at_eof_kept_whole() {
        let cfg = PreCfg::from_pre("gpt-2");
        assert_eq!(pieces("hi   ", cfg), vec!["hi", "   "]);
    }

    #[test]
    fn newline_rule_groups_trailing_newline() {
        let cfg = PreCfg::from_pre("qwen2");
        // "a  \n b": "a", then \s*[\r\n]+ = "  \n", then " b"
        assert_eq!(pieces("a  \n b", cfg), vec!["a", "  \n", " b"]);
    }

    #[test]
    fn reassembly_is_lossless() {
        // every pre-tokenizer must partition the input with no gaps/overlaps
        for pre in ["gpt-2", "qwen2", "llama-bpe"] {
            let cfg = PreCfg::from_pre(pre);
            for text in [
                "Hello, world! 123",
                "  spaced\tout\n\n",
                "café ÷ 99 — x'y",
                "",
            ] {
                let joined: String = pieces(text, cfg).concat();
                assert_eq!(joined, *text, "pre={pre} text={text:?}");
            }
        }
    }

    #[test]
    fn unicode_letters_classed_correctly() {
        let cfg = PreCfg::from_pre("qwen2");
        // accented letters stay together as a word (café is one piece)...
        assert_eq!(pieces("café stop", cfg), vec!["café", " stop"]);
        // ...and the em-dash classifies as a symbol, not a letter: spaced out it
        // forms its own " —" piece.
        assert_eq!(pieces("a — b", cfg), vec!["a", " —", " b"]);
        // the qwen2 letter rule [^\r\n\p{L}\p{N}]?\p{L}+ glues a single leading
        // punctuation char onto the following word.
        assert_eq!(pieces("café—x", cfg), vec!["café", "—x"]);
    }
}
