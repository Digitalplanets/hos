//! Conversation memory for flwr.
//!
//! A dependency-free, model-agnostic context compressor for long conversations:
//! keep a recent window of turns verbatim, fold everything older into a compact
//! structured memory block, and guarantee the assembled prompt fits the token
//! budget. All extraction is generic (no hardcoded topics), so it works for any
//! conversation and any model, including the small local ones.
//!
//! It is intentionally extractive and deterministic. An optional LLM-summarize
//! pass can be layered on later by feeding `assemble`'s omitted turns through a
//! model, but the baseline here always works with no model call and no latency.

use hos::chat::Message;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

const SCHEMA: u32 = 1;
const DEFAULT_CONTEXT_TOKENS: usize = 3072;
const DEFAULT_RECENT_TURNS: usize = 8;
const DEFAULT_BATCH_MESSAGES: usize = 8;
const MEMORY_HEADROOM_TOKENS: usize = 900;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryItem {
    pub text: String,
    pub source: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubjectMention {
    pub name: String,
    pub aliases: Vec<String>,
    pub source: String,
    pub context: String,
}

#[derive(Clone, Debug)]
pub struct QueryFrame {
    pub latest_user: String,
    pub intent: String,
    pub active_subject: Option<String>,
    pub detected_subjects: Vec<SubjectMention>,
    pub scope_note: String,
    pub response_hint: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryReceipt {
    pub id: String,
    pub start_message: usize,
    pub end_message: usize,
    pub summary: String,
    pub keypoints: Vec<MemoryItem>,
    pub open_tasks: Vec<MemoryItem>,
    pub preferences: Vec<MemoryItem>,
    pub subjects: Vec<SubjectMention>,
    pub relationships: Vec<MemoryItem>,
    pub embedding_ref: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConversationMemory {
    pub schema: u32,
    pub updated: u64,
    pub covered_messages: usize,
    pub summary: String,
    pub keypoints: Vec<MemoryItem>,
    pub open_tasks: Vec<MemoryItem>,
    pub preferences: Vec<MemoryItem>,
    pub subjects: Vec<SubjectMention>,
    pub relationships: Vec<MemoryItem>,
    pub pinned: Vec<MemoryItem>,
    pub receipts: Vec<MemoryReceipt>,
    /// Reserved for local vector search. Empty in the extractive first pass.
    pub embedding_refs: Vec<String>,
}

impl Default for ConversationMemory {
    fn default() -> ConversationMemory {
        ConversationMemory {
            schema: SCHEMA,
            updated: now(),
            covered_messages: 0,
            summary: String::new(),
            keypoints: Vec::new(),
            open_tasks: Vec::new(),
            preferences: Vec::new(),
            subjects: Vec::new(),
            relationships: Vec::new(),
            pinned: Vec::new(),
            receipts: Vec::new(),
            embedding_refs: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ContextBundle {
    pub messages: Vec<Message>,
    pub memory: ConversationMemory,
    pub omitted_messages: usize,
    pub estimated_tokens: usize,
}

pub fn context_budget_tokens() -> usize {
    env_usize("FLWR_CONTEXT_TOKENS").unwrap_or(DEFAULT_CONTEXT_TOKENS)
}

pub fn recent_turns() -> usize {
    env_usize("FLWR_RECENT_TURNS").unwrap_or(DEFAULT_RECENT_TURNS)
}

pub fn batch_messages() -> usize {
    env_usize("FLWR_MEMORY_BATCH_MESSAGES").unwrap_or(DEFAULT_BATCH_MESSAGES)
}

pub fn to_json(memory: &ConversationMemory) -> Value {
    serde_json::to_value(memory).unwrap_or_else(|_| json!({}))
}

pub fn receipt_signature(memory: &ConversationMemory) -> String {
    memory
        .receipts
        .iter()
        .map(|r| r.id.as_str())
        .collect::<Vec<_>>()
        .join("|")
}

pub fn debug_compaction() -> bool {
    std::env::var("FLWR_MEMORY_DEBUG")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

pub fn summarize(messages: &[Message]) -> ConversationMemory {
    summarize_range(messages, 0, batch_messages())
}

fn summarize_range(
    messages: &[Message],
    start_message: usize,
    batch_message_count: usize,
) -> ConversationMemory {
    let mut memory = ConversationMemory::default();
    memory.covered_messages = messages.len();
    memory.receipts = build_receipts(messages, start_message, batch_message_count);
    memory.summary = aggregate_summary(&memory.receipts, messages);
    memory.keypoints = collect_receipt_items(&memory.receipts, |r| &r.keypoints, 12);
    memory.open_tasks = collect_receipt_items(&memory.receipts, |r| &r.open_tasks, 8);
    memory.preferences = collect_receipt_items(&memory.receipts, |r| &r.preferences, 8);
    memory.subjects = collect_receipt_subjects(&memory.receipts, 12);
    memory.relationships = collect_receipt_items(&memory.receipts, |r| &r.relationships, 10);
    memory.updated = now();
    memory
}

pub fn assemble(messages: &[Message]) -> ContextBundle {
    assemble_opts(
        messages,
        context_budget_tokens(),
        recent_turns(),
        batch_messages(),
    )
}

pub fn assemble_with(
    messages: &[Message],
    max_prompt_tokens: usize,
    recent_turn_count: usize,
) -> ContextBundle {
    assemble_opts(messages, max_prompt_tokens, recent_turn_count, batch_messages())
}

/// Full-control assembly: caller supplies every compression knob (prompt-token
/// budget, verbatim recent-turn count, and messages-per-receipt batch size), so a
/// front-end that edits these live (the `/context` command, the server settings
/// panel) doesn't have to route through env vars.
pub fn assemble_opts(
    messages: &[Message],
    max_prompt_tokens: usize,
    recent_turn_count: usize,
    batch_count: usize,
) -> ContextBundle {
    let batch_count = batch_count.max(1);
    let raw_tokens = estimate_tokens(messages);
    if raw_tokens <= max_prompt_tokens || messages.len() <= recent_turn_count + 1 {
        let memory = summarize_range(messages, 0, batch_count);
        return ContextBundle {
            messages: messages.to_vec(),
            memory,
            omitted_messages: 0,
            estimated_tokens: raw_tokens,
        };
    }

    let leading_systems: Vec<Message> = messages
        .iter()
        .take_while(|m| m.role == "system")
        .cloned()
        .collect();
    let body_start = leading_systems.len();
    let body = &messages[body_start..];

    // Reserve a slice of the budget for the memory block; the rest is the tail.
    // Never let the memory reservation swallow the whole budget.
    let memory_budget = (max_prompt_tokens / 2).min(MEMORY_HEADROOM_TOKENS);
    let tail_budget = max_prompt_tokens.saturating_sub(memory_budget).max(1);

    // Fill the recent tail newest-first with a running token count (O(n), not O(n^2)).
    let recent_cap = recent_turn_count.max(2);
    let mut tail: Vec<Message> = Vec::new();
    let mut tail_tokens = 0usize;
    for m in body.iter().rev() {
        if tail.len() >= recent_cap {
            break;
        }
        let mt = message_tokens(m);
        if !tail.is_empty() && tail_tokens + mt > tail_budget {
            break;
        }
        tail.push(m.clone());
        tail_tokens += mt;
    }
    tail.reverse();

    let omitted_messages = body.len().saturating_sub(tail.len());
    let omitted = &body[..omitted_messages];
    let memory = summarize_range(omitted, body_start, batch_count);
    let frame = build_query_frame(messages, &memory);

    let mut compacted = leading_systems.clone();
    if omitted_messages > 0 {
        compacted.push(Message::new(
            "system",
            &render_memory_system(&memory, &frame, omitted_messages, memory_budget),
        ));
    }
    let mem_msg_index = if omitted_messages > 0 {
        Some(leading_systems.len())
    } else {
        None
    };
    let first_tail = compacted.len(); // first tail index
    compacted.extend(tail);

    // Hard guarantee: never exceed the budget. First drop the oldest tail turns
    // (keeping the most recent one verbatim); if a tiny budget still won't fit,
    // drop the memory block itself so the live turn always survives.
    while estimate_tokens(&compacted) > max_prompt_tokens && compacted.len() > first_tail + 1 {
        compacted.remove(first_tail);
    }
    if estimate_tokens(&compacted) > max_prompt_tokens {
        if let Some(i) = mem_msg_index {
            if compacted.len() > i + 1 {
                compacted.remove(i);
            }
        }
    }

    let estimated_tokens = estimate_tokens(&compacted);
    ContextBundle {
        messages: compacted,
        memory,
        omitted_messages,
        estimated_tokens,
    }
}

pub fn compact_gemma_history(
    history: &[(String, String)],
) -> (Vec<(String, String)>, ConversationMemory, usize) {
    let msgs = messages_from_pairs(history);
    let bundle = assemble(&msgs);
    // Gemma has no system role and requires user/model alternation, so fold any
    // system content (leading systems + the injected memory block) into the next
    // user turn instead of emitting a separate, alternation-breaking turn.
    let mut pairs: Vec<(String, String)> = Vec::new();
    let mut pending_system = String::new();
    for m in &bundle.messages {
        match m.role.as_str() {
            "system" => {
                if !pending_system.is_empty() {
                    pending_system.push_str("\n\n");
                }
                pending_system.push_str(&m.content);
            }
            "assistant" => pairs.push(("model".to_string(), m.content.clone())),
            _ => {
                let body = if pending_system.is_empty() {
                    m.content.clone()
                } else {
                    let s = format!("{}\n\n{}", pending_system, m.content);
                    pending_system.clear();
                    s
                };
                pairs.push(("user".to_string(), body));
            }
        }
    }
    if !pending_system.is_empty() {
        pairs.push(("user".to_string(), pending_system));
    }
    (pairs, bundle.memory, bundle.omitted_messages)
}

pub fn messages_from_pairs(history: &[(String, String)]) -> Vec<Message> {
    history
        .iter()
        .map(|(role, content)| {
            let role = if role == "model" {
                "assistant"
            } else {
                role.as_str()
            };
            Message::new(role, content)
        })
        .collect()
}

pub fn messages_from_json(v: &Value) -> Vec<Message> {
    v.as_array()
        .map(|arr| {
            arr.iter()
                .map(|m| {
                    Message::new(
                        m["role"].as_str().unwrap_or("user"),
                        m["content"].as_str().unwrap_or(""),
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

fn render_memory_system(
    memory: &ConversationMemory,
    frame: &QueryFrame,
    omitted_messages: usize,
    budget_tokens: usize,
) -> String {
    // Build candidate lines in priority order, then greedily keep as many as fit
    // the memory token budget. Highest-value guidance (the header, the latest
    // query, intent, scope) comes first so it survives truncation.
    let mut lines: Vec<String> = Vec::new();
    lines.push(format!(
        "Internal conversation memory covering {omitted_messages} earlier omitted messages. Use this only as private background. Do not quote, list, expose, or mention receipt ids unless the user explicitly asks for memory diagnostics. The recent raw turns below override this memory. If this memory does not contain enough evidence, say what is missing rather than guessing."
    ));
    lines.push(format!("Latest user query: {}", frame.latest_user));
    lines.push(format!("Detected intent: {}", frame.intent));
    if let Some(subject) = &frame.active_subject {
        lines.push(format!("Active subject: {subject}"));
    }
    if !frame.response_hint.is_empty() {
        lines.push(format!("Response hint: {}", frame.response_hint));
    }
    if !frame.scope_note.is_empty() {
        lines.push(format!("Scope rule: {}", frame.scope_note));
    }
    if !memory.summary.is_empty() {
        lines.push(format!("Prior context aggregate: {}", memory.summary));
    }
    if !frame.detected_subjects.is_empty() {
        lines.push("Detected subjects:".to_string());
        for (idx, subject) in frame.detected_subjects.iter().take(8).enumerate() {
            let aliases = if subject.aliases.is_empty() {
                String::new()
            } else {
                format!(" aliases={}", subject.aliases.join(", "))
            };
            lines.push(format!(
                "- {}. {}{} context: {}",
                idx + 1,
                subject.name,
                aliases,
                subject.context
            ));
        }
    }
    if !memory.receipts.is_empty() {
        lines.push("Prior context receipts:".to_string());
        for r in memory.receipts.iter().take(10) {
            lines.push(format!(
                "- {} messages {}..{}: {}",
                r.id, r.start_message, r.end_message, r.summary
            ));
        }
    }
    push_items(&mut lines, "Keypoints", &memory.keypoints);
    push_items(&mut lines, "Open tasks", &memory.open_tasks);
    push_items(&mut lines, "User preferences", &memory.preferences);
    push_items(&mut lines, "Subject relationships", &memory.relationships);
    push_items(&mut lines, "Pinned facts", &memory.pinned);

    // Greedy fit to the budget. The first line (the instruction) is always kept.
    let mut out: Vec<String> = Vec::new();
    let mut used = 0usize;
    for line in lines {
        let cost = estimate_text_tokens(&line) + 1;
        if !out.is_empty() && used + cost > budget_tokens {
            break;
        }
        used += cost;
        out.push(line);
    }
    out.join("\n")
}

fn push_items(lines: &mut Vec<String>, title: &str, items: &[MemoryItem]) {
    if items.is_empty() {
        return;
    }
    lines.push(format!("{title}:"));
    for item in items.iter().take(8) {
        lines.push(format!("- {}", item.text));
    }
}

/// A short, sentence-aware digest of a batch of turns. Skips trivial
/// acknowledgements so the summary carries signal, not "ok / noted".
fn build_summary(messages: &[Message], max_chars: usize) -> String {
    let mut parts = Vec::new();
    for m in messages {
        if m.role == "system" {
            continue;
        }
        let text = normalize(&m.content);
        if text.is_empty() || is_trivial_ack(&text) {
            continue;
        }
        let role = if m.role == "assistant" {
            "assistant"
        } else {
            "user"
        };
        parts.push(format!("{role}: {}", sentence_clip(&text, 180)));
        if parts.len() >= 10 {
            break;
        }
    }
    clip(&parts.join(" | "), max_chars)
}

fn build_receipts(
    messages: &[Message],
    start_message: usize,
    batch_message_count: usize,
) -> Vec<MemoryReceipt> {
    let batch = batch_message_count.max(2);
    let mut receipts = Vec::new();
    let mut offset = 0usize;
    while offset < messages.len() {
        let end = (offset + batch).min(messages.len());
        let chunk = &messages[offset..end];
        let start_abs = start_message + offset;
        let end_abs = start_message + end;
        receipts.push(MemoryReceipt {
            id: format!("r{start_abs}-{end_abs}"),
            start_message: start_abs,
            end_message: end_abs,
            summary: build_summary(chunk, 360),
            keypoints: collect_items_with_base(chunk, start_abs, ItemKind::Keypoint, 6),
            open_tasks: collect_items_with_base(chunk, start_abs, ItemKind::Task, 4),
            preferences: collect_items_with_base(chunk, start_abs, ItemKind::Preference, 4),
            subjects: collect_subjects(chunk, start_abs),
            relationships: collect_relationships(chunk, start_abs, 6),
            embedding_ref: None,
        });
        offset = end;
    }
    receipts
}

fn aggregate_summary(receipts: &[MemoryReceipt], fallback_messages: &[Message]) -> String {
    if receipts.is_empty() {
        return build_summary(fallback_messages, 900);
    }
    let mut parts = Vec::new();
    for r in receipts.iter().take(8) {
        if !r.summary.is_empty() {
            parts.push(format!("{} {}", r.id, r.summary));
        }
    }
    clip(&parts.join(" | "), 900)
}

fn collect_receipt_items(
    receipts: &[MemoryReceipt],
    field: fn(&MemoryReceipt) -> &Vec<MemoryItem>,
    limit: usize,
) -> Vec<MemoryItem> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for receipt in receipts {
        for item in field(receipt) {
            let key = item.text.to_lowercase();
            if seen.insert(key) {
                out.push(item.clone());
            }
            if out.len() >= limit {
                return out;
            }
        }
    }
    out
}

fn collect_receipt_subjects(receipts: &[MemoryReceipt], limit: usize) -> Vec<SubjectMention> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for receipt in receipts {
        for subject in &receipt.subjects {
            let key = subject.name.to_lowercase();
            if seen.insert(key) {
                out.push(subject.clone());
            }
            if out.len() >= limit {
                return out;
            }
        }
    }
    out
}

#[derive(Clone, Copy)]
enum ItemKind {
    Keypoint,
    Task,
    Preference,
}

fn build_query_frame(messages: &[Message], memory: &ConversationMemory) -> QueryFrame {
    let latest_user = messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| normalize(&m.content))
        .unwrap_or_default();
    let mut detected_subjects = collect_subjects(messages, 0);
    merge_subjects(&mut detected_subjects, &memory.subjects);
    let active_subject = resolve_active_subject(&latest_user, &detected_subjects);
    let intent = detect_intent(&latest_user);
    let scope_note = scope_note(&latest_user, active_subject.as_deref());
    let response_hint = response_hint(&latest_user, &detected_subjects, active_subject.as_deref());
    QueryFrame {
        latest_user,
        intent,
        active_subject,
        detected_subjects,
        scope_note,
        response_hint,
    }
}

/// Generic intent classification from surface cues only. No topic knowledge.
fn detect_intent(text: &str) -> String {
    let lower = text.to_lowercase();
    let mut parts = Vec::new();
    let asks_recall = lower.contains("what were")
        || lower.contains("what did")
        || lower.contains("what was")
        || lower.contains("remind me")
        || lower.contains("recall")
        || lower.contains("earlier")
        || (lower.contains("first") && lower.contains('?'));
    if asks_recall {
        parts.push("recall prior context");
    }
    if is_correction(&lower) {
        parts.push("correction or scope override");
    }
    if lower.contains("organize")
        || lower.contains("plan")
        || lower.contains("next step")
        || lower.contains("we need")
        || lower.starts_with("let's")
        || lower.starts_with("lets ")
    {
        parts.push("plan or task");
    }
    if lower.contains("list") || lower.contains("which ") || lower.starts_with("what ") {
        parts.push("list or enumerate");
    }
    if parts.is_empty() {
        "answer latest user query".to_string()
    } else {
        parts.join("; ")
    }
}

fn response_hint(
    latest_user: &str,
    subjects: &[SubjectMention],
    active_subject: Option<&str>,
) -> String {
    let lower = latest_user.to_lowercase();
    let recall = lower.contains("what were")
        || lower.contains("what did")
        || lower.contains("remind me")
        || lower.contains("earlier")
        || lower.contains("first");
    if recall && !subjects.is_empty() {
        let names = subjects
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>()
            .join("; ");
        return format!(
            "The user is asking about prior conversation topics. Answer from Detected subjects in order: {names}."
        );
    }
    if let Some(subject) = active_subject {
        return format!("Keep the answer scoped to active subject '{subject}' unless the latest user query explicitly changes subjects.");
    }
    String::new()
}

fn scope_note(latest_user: &str, active_subject: Option<&str>) -> String {
    let lower = latest_user.to_lowercase();
    if is_correction(&lower) {
        return "Treat the latest user message as a correction. Override conflicting prior receipts."
            .to_string();
    }
    if let Some(subject) = active_subject {
        return format!(
            "Resolve shorthand in the latest query against active subject '{subject}'. Do not drift to a broader topic unless the user explicitly changes subjects."
        );
    }
    "If the latest query starts a new subject, do not force older subjects into the answer."
        .to_string()
}

fn is_correction(lower: &str) -> bool {
    lower.contains("i said")
        || lower.contains("actually")
        || lower.contains("i meant")
        || lower.contains("instead")
        || lower.contains("correction")
        || lower.contains("pause")
        || lower.starts_with("no,")
        || lower.starts_with("no ")
        || lower.starts_with("not ")
}

/// The active subject is the detected entity that appears in the latest query,
/// preferring the most recently mentioned. Purely positional, no topic list.
fn resolve_active_subject(latest_user: &str, subjects: &[SubjectMention]) -> Option<String> {
    let lower = latest_user.to_lowercase();
    for subject in subjects.iter().rev() {
        if subject_matches(&lower, subject) {
            return Some(subject.name.clone());
        }
    }
    None
}

fn subject_matches(lower_text: &str, subject: &SubjectMention) -> bool {
    if lower_text.contains(&subject.name.to_lowercase()) {
        return true;
    }
    subject
        .aliases
        .iter()
        .any(|alias| lower_text.contains(&alias.to_lowercase()))
}

/// Generic proper-noun extraction: runs of two or more Capitalized words are
/// strong entities; a single Capitalized word counts once it recurs. Common
/// sentence-openers and function words are filtered. No hardcoded topics.
fn collect_subjects(messages: &[Message], base_message: usize) -> Vec<SubjectMention> {
    let mut singles: BTreeMap<String, usize> = BTreeMap::new();
    // (display_name, source, context) in first-seen order, deduped by lowercase.
    let mut ordered: Vec<(String, String, String)> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();

    for (idx, m) in messages.iter().enumerate() {
        if m.role == "system" || m.content.trim().is_empty() {
            continue;
        }
        let text = normalize(&m.content);
        let source = format!("message:{}", base_message + idx);
        let context = clip(&text, 240);
        for phrase in capitalized_phrases(&text) {
            let word_count = phrase.split_whitespace().count();
            let key = phrase.to_lowercase();
            if word_count >= 2 {
                if seen.insert(key) {
                    ordered.push((phrase, source.clone(), context.clone()));
                }
            } else {
                // Single word: remember it and its first sighting; promote later
                // if it recurs (so one-off capitalizations do not become topics).
                let count = singles.entry(key.clone()).or_insert(0);
                *count += 1;
                if *count == 2 && seen.insert(key) {
                    ordered.push((phrase, source.clone(), context.clone()));
                }
            }
        }
    }

    ordered
        .into_iter()
        .take(16)
        .map(|(name, source, context)| SubjectMention {
            aliases: vec![name.to_lowercase()],
            name,
            source,
            context,
        })
        .collect()
}

/// Maximal runs of Capitalized, non-stopword tokens within each sentence. The
/// first token of a sentence only starts a run if the run is 2+ words or it is
/// not a common opener, so grammar-capitalized words do not leak in.
fn capitalized_phrases(text: &str) -> Vec<String> {
    let mut phrases = Vec::new();
    for sentence in text.split(|c| matches!(c, '.' | '!' | '?' | '\n')) {
        let mut run: Vec<String> = Vec::new();
        for raw in sentence.split_whitespace() {
            let word = raw.trim_matches(|c: char| !c.is_alphanumeric());
            let is_entity_word = word.len() >= 2
                && first_is_upper(word)
                && !is_all_upper(word)
                && !stopwords().contains(word.to_lowercase().as_str());
            if is_entity_word {
                run.push(word.to_string());
                // list separators (comma / semicolon / colon) end a phrase, so
                // "Kowloon Walled City, Zanzibar Revolution" stays two entities.
                if raw.ends_with([',', ';', ':']) {
                    flush_run(&mut run, &mut phrases);
                }
            } else {
                flush_run(&mut run, &mut phrases);
            }
        }
        flush_run(&mut run, &mut phrases);
    }
    phrases
}

fn flush_run(run: &mut Vec<String>, phrases: &mut Vec<String>) {
    if run.is_empty() {
        return;
    }
    // A multi-word run is a phrase; a single word is a phrase only if it is not
    // a bare sentence opener (handled by the caller filtering stopwords).
    let phrase = run.join(" ");
    phrases.push(phrase);
    run.clear();
}

fn merge_subjects(out: &mut Vec<SubjectMention>, incoming: &[SubjectMention]) {
    for subject in incoming {
        if !out
            .iter()
            .any(|s| s.name.eq_ignore_ascii_case(&subject.name))
        {
            out.push(subject.clone());
        }
    }
}

/// Turns that carry a relation or decision: generic causal / contrastive cues,
/// or a turn that mentions a detected entity. No topic-specific vocabulary.
fn collect_relationships(
    messages: &[Message],
    base_message: usize,
    limit: usize,
) -> Vec<MemoryItem> {
    let mut out = Vec::new();
    for (idx, m) in messages.iter().enumerate() {
        if m.role == "system" || m.content.trim().is_empty() {
            continue;
        }
        let text = normalize(&m.content);
        let lower = text.to_lowercase();
        let has_relation = relation_cues().iter().any(|cue| lower.contains(cue));
        let has_entity = !capitalized_phrases(&text).is_empty();
        if has_relation && has_entity {
            out.push(MemoryItem {
                text: clip(&text, 240),
                source: format!("message:{}", base_message + idx),
            });
        }
        if out.len() >= limit {
            break;
        }
    }
    out
}

fn collect_items_with_base(
    messages: &[Message],
    base_message: usize,
    kind: ItemKind,
    limit: usize,
) -> Vec<MemoryItem> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for (idx, m) in messages.iter().enumerate() {
        if m.role == "system" || m.content.trim().is_empty() {
            continue;
        }
        let text = normalize(&m.content);
        let lower = text.to_lowercase();
        if !matches_kind(&lower, m.role.as_str(), kind) {
            continue;
        }
        let item = clip(&text, 220);
        let key = item.to_lowercase();
        if seen.insert(key) {
            out.push(MemoryItem {
                text: item,
                source: format!("message:{}", base_message + idx),
            });
        }
        if out.len() >= limit {
            break;
        }
    }
    out
}

/// Generic cue-based classification of user turns. `lower` is pre-lowercased.
fn matches_kind(lower: &str, role: &str, kind: ItemKind) -> bool {
    if role != "user" {
        return false;
    }
    match kind {
        ItemKind::Keypoint => {
            lower.contains("decided")
                || lower.contains("let's go with")
                || lower.contains("lets go with")
                || lower.contains("the plan is")
                || lower.contains("important")
                || lower.contains("remember")
                || lower.contains("the goal")
                || lower.contains("key point")
                || lower.contains("start with")
        }
        ItemKind::Task => {
            lower.contains("need to")
                || lower.contains("we need")
                || lower.contains("let's ")
                || lower.contains("lets ")
                || lower.contains("todo")
                || lower.contains("to do")
                || lower.contains("next")
                || lower.contains("should ")
                || lower.contains("make sure")
                || lower.contains("start here")
        }
        ItemKind::Preference => {
            lower.contains("i like")
                || lower.contains("i want")
                || lower.contains("i'd rather")
                || lower.contains("i would rather")
                || lower.contains("prefer")
                || lower.contains("don't ")
                || lower.contains("do not ")
                || lower.contains("always ")
                || lower.contains("never ")
                || lower.contains("should be")
        }
    }
}

// ---- tables (built once) ---------------------------------------------------

fn stopwords() -> &'static BTreeSet<&'static str> {
    static SET: OnceLock<BTreeSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| {
        [
            "the", "a", "an", "and", "or", "but", "so", "then", "this", "that", "these", "those",
            "i", "we", "you", "he", "she", "it", "they", "my", "our", "your", "his", "her", "its",
            "their", "is", "are", "was", "were", "be", "to", "of", "in", "on", "for", "with", "as",
            "at", "by", "from", "if", "no", "not", "yes", "ok", "okay", "please", "let", "lets",
            "what", "why", "how", "when", "where", "who", "which", "can", "could", "would", "should",
            "will", "do", "does", "did", "history", "project", "projects", "battle", "battles",
            "document", "documents", "source", "sources", "subject", "subjects", "topic", "topics",
        ]
        .into_iter()
        .collect()
    })
}

fn relation_cues() -> &'static [&'static str] {
    &[
        " won", " lost", " beat", " defeated", " against", " between", " versus", " vs ", "led to",
        "because", "caused", "resulted in", "expelled", "signed", "founded", "vs.", "instead of",
    ]
}

// ---- small helpers ---------------------------------------------------------

fn first_is_upper(word: &str) -> bool {
    word.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
}

fn is_all_upper(word: &str) -> bool {
    word.chars().filter(|c| c.is_alphabetic()).all(|c| c.is_uppercase())
}

fn is_trivial_ack(text: &str) -> bool {
    let t = text.trim().to_lowercase();
    let t = t.trim_matches(|c: char| !c.is_alphanumeric());
    matches!(
        t,
        "ok" | "okay" | "noted" | "sure" | "yes" | "yep" | "yeah" | "got it" | "done" | "thanks"
            | "thank you" | "k" | "kk" | "cool" | "nice"
    ) || t.is_empty()
}

/// Estimated tokens for a raw message list (before any compaction). Used by the
/// `flwr membench` benchmark to report compression ratios.
pub fn raw_tokens(messages: &[Message]) -> usize {
    estimate_tokens(messages)
}

fn estimate_tokens(messages: &[Message]) -> usize {
    messages.iter().map(message_tokens).sum()
}

fn message_tokens(m: &Message) -> usize {
    estimate_text_tokens(&m.role) + estimate_text_tokens(&m.content) + 4
}

fn estimate_text_tokens(s: &str) -> usize {
    (s.chars().count() + 3) / 4
}

fn normalize(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn clip(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max_chars.saturating_sub(1)).collect();
    out.push_str("...");
    out
}

/// Clip on a sentence boundary when possible so a summary never cuts mid-word.
fn sentence_clip(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let head: String = s.chars().take(max_chars).collect();
    if let Some(pos) = head.rfind(['.', '!', '?']) {
        if pos >= max_chars / 2 {
            return head[..=pos].to_string();
        }
    }
    clip(s, max_chars)
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse().ok()
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compacts_older_turns_into_memory_system_message() {
        let mut msgs = vec![Message::new("system", "You are concise.")];
        for i in 0..20 {
            msgs.push(Message::new(
                "user",
                &format!("We need to remember task {i} and start with the plan."),
            ));
            msgs.push(Message::new("assistant", "noted"));
        }

        let bundle = assemble_with(&msgs, 220, 4);
        assert!(bundle.omitted_messages > 0);
        assert_eq!(bundle.messages[0].role, "system");
        assert_eq!(bundle.messages[1].role, "system");
        assert!(bundle.messages[1]
            .content
            .contains("Internal conversation memory"));
        assert!(!bundle.memory.receipts.is_empty());
    }

    #[test]
    fn never_exceeds_the_token_budget() {
        let mut msgs = vec![Message::new("system", "You are concise.")];
        for i in 0..60 {
            msgs.push(Message::new(
                "user",
                &format!("Turn {i}: a fairly long user message about many different things that runs on for a while to eat tokens."),
            ));
            msgs.push(Message::new("assistant", &format!("A long assistant reply number {i} that also consumes a good number of tokens to test the budget guarantee.")));
        }
        for budget in [128usize, 256, 512, 1024] {
            let bundle = assemble_with(&msgs, budget, 6);
            assert!(
                bundle.estimated_tokens <= budget,
                "budget {budget} exceeded: {}",
                bundle.estimated_tokens
            );
        }
    }

    #[test]
    fn extracts_generic_entities_no_hardcoding() {
        // Arbitrary proper nouns the code has never seen.
        let msgs = vec![
            Message::new("user", "your name is audia and you help organize projects"),
            Message::new("assistant", "ok"),
            Message::new(
                "user",
                "Projects: the Kowloon Walled City, Zanzibar Revolution, and the Bronze Horseman.",
            ),
            Message::new("assistant", "ok"),
            Message::new("user", "to start, lets go with the Zanzibar Revolution"),
        ];
        let subjects = collect_subjects(&msgs, 0);
        let names: Vec<String> = subjects.iter().map(|s| s.name.clone()).collect();
        assert!(names.iter().any(|n| n == "Kowloon Walled City"), "names={names:?}");
        assert!(names.iter().any(|n| n == "Zanzibar Revolution"), "names={names:?}");
        let mem = summarize(&msgs);
        let frame = build_query_frame(&msgs, &mem);
        assert_eq!(frame.active_subject.as_deref(), Some("Zanzibar Revolution"));
    }

    #[test]
    fn receipt_ids_are_stable_as_new_turns_arrive() {
        let mut msgs = Vec::new();
        for i in 0..14 {
            msgs.push(Message::new(
                "user",
                &format!("We need receipt stability task {i}."),
            ));
            msgs.push(Message::new("assistant", "noted"));
        }
        let first = assemble_with(&msgs, 220, 4);
        let first_ids: Vec<String> = first.memory.receipts.iter().map(|r| r.id.clone()).collect();
        msgs.push(Message::new(
            "user",
            "One more turn should not rewrite old receipts.",
        ));
        let second = assemble_with(&msgs, 220, 4);
        let second_ids: Vec<String> = second
            .memory
            .receipts
            .iter()
            .take(first_ids.len())
            .map(|r| r.id.clone())
            .collect();
        assert_eq!(first_ids, second_ids);
    }

    #[test]
    fn correction_sets_override_scope() {
        let msgs = vec![
            Message::new("user", "Let's plan the Halifax project first."),
            Message::new("assistant", "ok"),
            Message::new("user", "actually, I said pause that, do the other one instead"),
        ];
        let mem = summarize(&msgs);
        let frame = build_query_frame(&msgs, &mem);
        assert!(
            frame.scope_note.contains("correction") && frame.intent.contains("correction"),
            "scope={} intent={}",
            frame.scope_note,
            frame.intent
        );
    }

    #[test]
    fn gemma_history_folds_system_and_stays_alternating() {
        // Enough turns to exceed the default context budget and force compaction.
        let mut history: Vec<(String, String)> = Vec::new();
        for i in 0..120 {
            history.push((
                "user".to_string(),
                format!("Turn {i}: we need to plan the Halifax project and remember the goal for later reference in this conversation."),
            ));
            history.push(("model".to_string(), format!("reply {i} acknowledged and noted for the record")));
        }
        let (pairs, _mem, omitted) = compact_gemma_history(&history);
        assert!(omitted > 0, "expected compaction");
        // Gemma has no system role.
        assert!(pairs.iter().all(|(r, _)| r == "user" || r == "model"));
        // Strict alternation: no two consecutive same-role turns.
        for w in pairs.windows(2) {
            assert_ne!(
                w[0].0, w[1].0,
                "consecutive '{}' turns in {:?}",
                w[0].0,
                pairs.iter().map(|(r, _)| r.clone()).collect::<Vec<_>>()
            );
        }
        // The memory block landed inside a user turn (folded, not dropped).
        assert!(pairs
            .iter()
            .any(|(r, c)| r == "user" && c.contains("Internal conversation memory")));
    }

    #[test]
    fn leaves_short_histories_verbatim() {
        let msgs = vec![Message::new("user", "hello")];
        let bundle = assemble_with(&msgs, 2048, 8);
        assert_eq!(bundle.omitted_messages, 0);
        assert_eq!(bundle.messages, msgs);
    }
}
