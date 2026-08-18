//! Reasoning receipts — a thinking model's `<think>` trace captured as a
//! content-addressed, persistent artifact instead of being dimmed and thrown
//! away. A receipt is first-class knowledge: agents (flwr or external) can
//! **cite** it by id, **audit** the trace before trusting the answer, **chain**
//! one agent's reasoning into another's input, or **branch** it to re-run from a
//! step. Same provenance spirit as `.hos` capsules, saved chats, and memory
//! receipts — applied to the model's own reasoning.
//!
//! Produced in flwr core (during generation), consumed by agents downstream.
//! Non-reasoning models simply emit no receipt (an empty trace is not saved), so
//! this is graceful for the whole model zoo. Stored as JSON in `~/.hos/reasoning/`
//! keyed by a content hash, and referenced by id from the chat transcript.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// One captured reasoning turn. The `trace`/`answer` split is exact — it comes
/// from the atomic `</think>` boundary in the token stream, not string scraping.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ReasoningReceipt {
    /// Content-addressed id: `rr-<fnv1a(query‖trace‖answer)>`. Stable + dedupable.
    pub id: String,
    pub chat_id: String,
    /// Index of the user turn this reasoning answered.
    pub turn: usize,
    pub query: String,
    /// Reasoning depth used: `xhigh` | `medium` | `low`.
    pub effort: String,
    /// The full `<think>` reasoning trace.
    pub trace: String,
    /// The committed, user-facing answer.
    pub answer: String,
    pub tokens: usize,
    pub created: u64,
    /// Model provenance (name + id + quant), same shape chats stamp.
    pub model: Value,
    /// Reserved for the extractive enrichment pass (steps/assumptions). Empty in
    /// v1 — adding these later does not change the stored contract.
    pub steps: Vec<String>,
    pub embedding_ref: Option<String>,
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// `$FLWR_REASONING`, else `~/.hos/reasoning`.
pub fn dir() -> PathBuf {
    if let Ok(d) = std::env::var("FLWR_REASONING") {
        return PathBuf::from(d);
    }
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".hos/reasoning")
}

/// FNV-1a over the reasoning's semantic content — the receipt is addressed by
/// what it *is*, so identical reasoning dedupes and any agent can re-derive the id.
pub fn content_id(query: &str, trace: &str, answer: &str) -> String {
    let mut h = 0xcbf29ce484222325u64;
    for part in [query, "\u{0}", trace, "\u{0}", answer] {
        for b in part.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    format!("rr-{h:016x}")
}

impl ReasoningReceipt {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        chat_id: &str,
        turn: usize,
        query: &str,
        effort: &str,
        trace: &str,
        answer: &str,
        tokens: usize,
        model: Value,
    ) -> ReasoningReceipt {
        ReasoningReceipt {
            id: content_id(query, trace, answer),
            chat_id: chat_id.to_string(),
            turn,
            query: query.to_string(),
            effort: effort.to_string(),
            trace: trace.trim().to_string(),
            answer: answer.trim().to_string(),
            tokens,
            created: now(),
            model,
            steps: Vec::new(),
            embedding_ref: None,
        }
    }

    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

fn path_for(id: &str) -> Option<PathBuf> {
    // id is our own `rr-<hex>`; guard anyway since it becomes a filename.
    if id.is_empty() || id.len() > 128 || !id.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-')
    {
        return None;
    }
    Some(dir().join(format!("{id}.json")))
}

/// Persist a receipt. A receipt with an empty trace is a non-reasoning turn and
/// is skipped (returns `Ok(None)`); otherwise returns the stored id.
pub fn save(r: &ReasoningReceipt) -> std::io::Result<Option<String>> {
    if r.trace.is_empty() {
        return Ok(None);
    }
    let Some(p) = path_for(&r.id) else {
        return Ok(None);
    };
    std::fs::create_dir_all(dir())?;
    std::fs::write(&p, serde_json::to_string_pretty(&r.to_json())?)?;
    Ok(Some(r.id.clone()))
}

/// Fetch a receipt by id — the agent-facing read path.
pub fn get(id: &str) -> Option<ReasoningReceipt> {
    let p = path_for(id)?;
    let raw = std::fs::read(&p).ok()?;
    serde_json::from_slice(&raw).ok()
}

/// Every receipt for a conversation, newest first — lets an agent walk a chat's
/// reasoning history.
pub fn list_for_chat(chat_id: &str) -> Vec<ReasoningReceipt> {
    let mut out: Vec<ReasoningReceipt> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir()) {
        for e in rd.flatten() {
            if let Ok(raw) = std::fs::read(e.path()) {
                if let Ok(r) = serde_json::from_slice::<ReasoningReceipt>(&raw) {
                    if r.chat_id == chat_id {
                        out.push(r);
                    }
                }
            }
        }
    }
    out.sort_by(|a, b| b.created.cmp(&a.created).then(b.turn.cmp(&a.turn)));
    out
}

/// JSON list of a chat's receipts (for the `GET /v1/reasoning?chat=<id>` route).
pub fn list_json(chat_id: &str) -> Value {
    json!(list_for_chat(chat_id)
        .iter()
        .map(|r| r.to_json())
        .collect::<Vec<_>>())
}
