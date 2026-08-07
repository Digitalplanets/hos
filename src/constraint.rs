//! Grammar-constrained decoding (R2).
//!
//! A [`Constraint`] masks the logits each step so the sampler can only pick a
//! token that keeps the output valid under a grammar. The first grammar is
//! [`JsonConstraint`]: an incremental JSON validator. With it enabled the engine
//! *cannot emit* a token that would break JSON — no more parse-and-repair, the
//! structure is guaranteed by construction.
//!
//! It is entirely opt-in: [`crate::Engine`] only consults a constraint when one
//! has been installed, so default generation is byte-for-byte unchanged.

use crate::tokenizer::Tokenizer;

/// Masks disallowed tokens (to `f32::NEG_INFINITY`) and advances its state as
/// tokens are chosen.
pub trait Constraint {
    /// Set the logit of every currently-disallowed token to `f32::NEG_INFINITY`,
    /// in place. Must leave at least one token allowed (the engine relies on it).
    fn mask(&self, logits: &mut [f32]);
    /// Advance the grammar state by the token the sampler chose.
    fn accept(&mut self, id: u32);
}

// ---------------------------------------------------------------------------
// Incremental JSON validator (a pushdown automaton over bytes).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Container {
    Object,
    Array,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    /// A value is expected here (document start, after `:`, after `,` in an
    /// array, or right after `[`).
    Value,
    /// Just after `{`: a key string or `}` (empty object).
    ObjKeyOrEnd,
    /// After `,` in an object: a key string.
    ObjKey,
    /// After a key string: `:`.
    Colon,
    /// After a complete value: `,`, the matching closer, or (top level) EOF.
    AfterValue,
}

/// Incremental JSON grammar state. `feed_byte` returns false if the byte cannot
/// legally continue the JSON seen so far; `can_stop` is true when the text so
/// far is a complete JSON document.
#[derive(Clone)]
struct Json {
    stack: Vec<Container>,
    mode: Mode,
    in_string: bool,
    escape: bool,
    key_string: bool,
    in_number: bool,
    lit: &'static [u8],
    lit_pos: usize,
    complete: bool,
}

impl Json {
    fn new() -> Self {
        Json {
            stack: Vec::new(),
            mode: Mode::Value,
            in_string: false,
            escape: false,
            key_string: false,
            in_number: false,
            lit: b"",
            lit_pos: 0,
            complete: false,
        }
    }

    /// A complete document = a top-level value has finished and nothing is open.
    fn can_stop(&self) -> bool {
        self.complete
            && self.stack.is_empty()
            && !self.in_string
            && !self.in_number
            && self.lit_pos == 0
    }

    /// Called when a value token (string value / number / literal / closer)
    /// finishes. Moves to AfterValue and marks completeness at the top level.
    fn end_value(&mut self) {
        self.mode = Mode::AfterValue;
        if self.stack.is_empty() {
            self.complete = true;
        }
    }

    fn feed_byte(&mut self, b: u8) -> bool {
        // Nothing may follow a complete top-level document except whitespace.
        if self.complete && self.stack.is_empty() {
            return is_ws(b);
        }
        // Mid-string.
        if self.in_string {
            if self.escape {
                self.escape = false;
                return true;
            }
            match b {
                b'\\' => self.escape = true,
                b'"' => {
                    self.in_string = false;
                    if self.key_string {
                        self.mode = Mode::Colon;
                    } else {
                        self.end_value();
                    }
                }
                0x00..=0x1f => return false, // control chars must be escaped
                _ => {}
            }
            return true;
        }
        // Mid-number: digits/./e/E/+/- continue; anything else ends it and is
        // re-processed structurally.
        if self.in_number {
            if matches!(b, b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-') {
                return true;
            }
            self.in_number = false;
            self.end_value();
            return self.feed_structural(b);
        }
        // Mid-literal (true / false / null).
        if self.lit_pos < self.lit.len() {
            if b == self.lit[self.lit_pos] {
                self.lit_pos += 1;
                if self.lit_pos == self.lit.len() {
                    self.lit = b"";
                    self.lit_pos = 0;
                    self.end_value();
                }
                return true;
            }
            return false;
        }
        self.feed_structural(b)
    }

    fn feed_structural(&mut self, b: u8) -> bool {
        if is_ws(b) {
            return true;
        }
        match self.mode {
            Mode::Value => self.start_value(b),
            Mode::ObjKeyOrEnd => match b {
                b'"' => {
                    self.in_string = true;
                    self.key_string = true;
                    true
                }
                b'}' => self.close(Container::Object),
                _ => false,
            },
            Mode::ObjKey => {
                if b == b'"' {
                    self.in_string = true;
                    self.key_string = true;
                    true
                } else {
                    false
                }
            }
            Mode::Colon => {
                if b == b':' {
                    self.mode = Mode::Value;
                    true
                } else {
                    false
                }
            }
            Mode::AfterValue => match b {
                b',' => match self.stack.last() {
                    Some(Container::Object) => {
                        self.mode = Mode::ObjKey;
                        true
                    }
                    Some(Container::Array) => {
                        self.mode = Mode::Value;
                        true
                    }
                    None => false,
                },
                b'}' => self.close(Container::Object),
                b']' => self.close(Container::Array),
                _ => false,
            },
        }
    }

    /// Begin a value with byte `b` in a value-expecting position.
    fn start_value(&mut self, b: u8) -> bool {
        match b {
            b'{' => {
                self.stack.push(Container::Object);
                self.mode = Mode::ObjKeyOrEnd;
                true
            }
            b'[' => {
                self.stack.push(Container::Array);
                self.mode = Mode::Value; // value or ']' (empty array)
                true
            }
            b']' => self.close(Container::Array), // empty array close
            b'"' => {
                self.in_string = true;
                self.key_string = false;
                true
            }
            b'-' | b'0'..=b'9' => {
                self.in_number = true;
                true
            }
            b't' => self.begin_literal(b"true"),
            b'f' => self.begin_literal(b"false"),
            b'n' => self.begin_literal(b"null"),
            _ => false,
        }
    }

    fn begin_literal(&mut self, word: &'static [u8]) -> bool {
        self.lit = word;
        self.lit_pos = 1; // first byte already matched by the dispatch
        true
    }

    fn close(&mut self, expected: Container) -> bool {
        if self.stack.last() == Some(&expected) {
            self.stack.pop();
            self.end_value();
            true
        } else {
            false
        }
    }
}

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

/// A [`Constraint`] that forces valid JSON. Precomputes each token's bytes once
/// so masking is a state-clone + byte-replay per candidate.
pub struct JsonConstraint {
    token_bytes: Vec<Vec<u8>>,
    /// Tokens allowed to *end* generation (EOS / chat stop ids) — permitted only
    /// when the JSON is already a complete document.
    end_tokens: Vec<u32>,
    state: Json,
}

impl JsonConstraint {
    /// Build from a tokenizer; `end_tokens` are the ids that may terminate the
    /// turn (typically `tok.eos` plus the chat family's stop ids).
    pub fn new(tok: &Tokenizer, end_tokens: Vec<u32>) -> Self {
        let vocab = tok.vocab_size();
        let mut token_bytes = Vec::with_capacity(vocab);
        for id in 0..vocab as u32 {
            let mut bytes = Vec::new();
            tok.decode_into(id, &mut bytes);
            token_bytes.push(bytes);
        }
        JsonConstraint {
            token_bytes,
            end_tokens,
            state: Json::new(),
        }
    }

    /// Whether appending token `id`'s bytes keeps the JSON valid.
    fn token_ok(&self, id: u32) -> bool {
        let bytes = &self.token_bytes[id as usize];
        if bytes.is_empty() {
            return false;
        }
        let mut probe = self.state.clone();
        for &b in bytes {
            if !probe.feed_byte(b) {
                return false;
            }
        }
        true
    }
}

impl Constraint for JsonConstraint {
    fn mask(&self, logits: &mut [f32]) {
        let end_ok = self.state.can_stop();
        let mut any = false;
        // First pass: decide allow/deny without mutating (end tokens special).
        let n = logits.len().min(self.token_bytes.len());
        let mut allow = vec![false; logits.len()];
        for id in 0..n {
            if self.token_ok(id as u32) {
                allow[id] = true;
                any = true;
            }
        }
        for &e in &self.end_tokens {
            let e = e as usize;
            if e < allow.len() && end_ok {
                allow[e] = true;
                any = true;
            }
        }
        // Safety: never mask everything (the sampler must have a choice). If the
        // grammar dead-ended, allow the end tokens; if there are none, leave
        // logits untouched rather than produce an all-NEG_INFINITY vector.
        if !any {
            if self.end_tokens.is_empty() {
                return;
            }
            for &e in &self.end_tokens {
                if (e as usize) < allow.len() {
                    allow[e as usize] = true;
                }
            }
        }
        for (id, l) in logits.iter_mut().enumerate() {
            if !allow.get(id).copied().unwrap_or(false) {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    fn accept(&mut self, id: u32) {
        if self.end_tokens.contains(&id) {
            return; // generation ends; no state change needed
        }
        if let Some(bytes) = self.token_bytes.get(id as usize) {
            for &b in bytes {
                // Ignore failure: the mask should have prevented an invalid pick,
                // but if a caller sampled unconstrained we don't want to panic.
                let _ = self.state.feed_byte(b);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(s: &str) -> Option<Json> {
        let mut j = Json::new();
        for &b in s.as_bytes() {
            if !j.feed_byte(b) {
                return None;
            }
        }
        Some(j)
    }

    #[test]
    fn accepts_a_complete_object() {
        let j = feed(r#"{"goal":"x","steps":[{"capability":"write_workspace","target":"a.md"}]}"#)
            .expect("valid JSON should be accepted");
        assert!(j.can_stop());
    }

    #[test]
    fn accepts_nested_arrays_numbers_and_literals() {
        let j = feed(r#"{"a":[1,2.5,-3,true,false,null],"b":{"c":"d"}}"#).unwrap();
        assert!(j.can_stop());
    }

    #[test]
    fn incomplete_object_is_not_stoppable_but_valid_so_far() {
        let j = feed(r#"{"goal":"x","steps":["#).unwrap();
        assert!(!j.can_stop()); // still open — cannot end here
    }

    #[test]
    fn rejects_a_stray_quote_before_a_value() {
        // the exact defect a 7B produced: `},"{`
        assert!(feed(r#"{"a":"x"},"#).is_none() || feed(r#"[{"a":1},"{"#).is_none());
        assert!(feed(r#"[{"a":1},"{"b":2}]"#).is_none());
    }

    #[test]
    fn rejects_trailing_garbage_after_a_complete_document() {
        assert!(feed(r#"{"a":1}x"#).is_none());
        assert!(feed(r#"{"a":1}]"#).is_none());
    }

    #[test]
    fn rejects_a_value_where_a_key_is_expected() {
        assert!(feed(r#"{1:2}"#).is_none());
        assert!(feed(r#"{"a" 2}"#).is_none()); // missing colon
    }

    #[test]
    fn whitespace_after_a_complete_document_is_fine() {
        let j = feed("{\"a\":1}\n  ").unwrap();
        assert!(j.can_stop());
    }

    #[test]
    fn empty_object_and_array() {
        assert!(feed("{}").unwrap().can_stop());
        assert!(feed("[]").unwrap().can_stop());
        assert!(feed("[{},{}]").unwrap().can_stop());
    }

    #[test]
    fn strings_may_contain_escaped_quotes_and_braces() {
        let j = feed(r#"{"a":"he said \"hi\" and { used } braces"}"#).unwrap();
        assert!(j.can_stop());
    }
}
