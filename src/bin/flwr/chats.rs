//! flwr chat transcripts — conversations saved as provenance-bearing JSON files
//! in `~/.hos/chats/`. Each record carries which model produced it (name +
//! content-hash id + source + quant) and the sampling params, so a saved chat is
//! a lineaged, reproducible record — not a convenience cache. Same spirit as the
//! `.hos` capsule and the store manifest, applied to a conversation.

use serde_json::{json, Value};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// `$FLWR_CHATS`, else `~/.hos/chats`.
pub fn dir() -> PathBuf {
    if let Ok(d) = std::env::var("FLWR_CHATS") {
        return PathBuf::from(d);
    }
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".hos/chats")
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// A fresh id when the client doesn't supply one.
pub fn new_id() -> String {
    format!("c{}", now())
}

/// Keep only filename-safe characters; `None` if nothing valid remains. This is
/// the path-traversal guard — the id becomes a filename, so `..`/`/` are stripped.
pub fn safe_id(raw: &str) -> Option<String> {
    let s: String = raw
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .collect();
    if s.is_empty() || s.len() > 128 {
        None
    } else {
        Some(s)
    }
}

fn path_for(id: &str) -> PathBuf {
    dir().join(format!("{id}.json"))
}

/// Upsert a transcript. `messages` come from the client; `model` provenance and
/// `params` are stamped by the caller (server-authoritative). The original
/// `created` time is preserved across updates.
pub fn save(id: &str, messages: &Value, model: Value, params: Value) -> std::io::Result<Value> {
    std::fs::create_dir_all(dir())?;
    let p = path_for(id);
    let created = std::fs::read(&p)
        .ok()
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .and_then(|v| v["created"].as_u64())
        .unwrap_or_else(now);
    let title = messages
        .as_array()
        .and_then(|a| a.iter().find(|m| m["role"] == "user"))
        .and_then(|m| m["content"].as_str())
        .map(|s| s.chars().take(60).collect::<String>())
        .unwrap_or_else(|| "untitled".into());
    let rec = json!({
        "id": id, "created": created, "updated": now(),
        "title": title, "model": model, "params": params,
        "messages": messages,
    });
    std::fs::write(&p, serde_json::to_string_pretty(&rec)?)?;
    Ok(json!({ "id": id, "title": title, "created": created }))
}

/// Summaries of every saved chat, newest first.
pub fn list() -> Value {
    let mut items: Vec<Value> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir()) {
        for e in rd.flatten() {
            if e.path().extension().and_then(|x| x.to_str()) != Some("json") {
                continue;
            }
            let Some(v) = std::fs::read(e.path())
                .ok()
                .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
            else {
                continue;
            };
            let n = v["messages"].as_array().map(|a| a.len()).unwrap_or(0);
            items.push(json!({
                "id": v["id"], "title": v["title"], "created": v["created"],
                "updated": v["updated"], "model": v["model"]["name"], "messages": n
            }));
        }
    }
    items.sort_by(|a, b| {
        b["updated"]
            .as_u64()
            .unwrap_or(0)
            .cmp(&a["updated"].as_u64().unwrap_or(0))
    });
    json!({ "data": items })
}

/// Raw JSON text of one saved chat (for `GET /chats/<id>`).
pub fn load(id: &str) -> Option<String> {
    std::fs::read_to_string(path_for(id)).ok()
}

pub fn delete(id: &str) -> bool {
    std::fs::remove_file(path_for(id)).is_ok()
}
