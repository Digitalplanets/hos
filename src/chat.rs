//! Chat templating — turns a conversation into the exact token stream a model
//! expects, and tells the generator where a turn ends.
//!
//! HOS ships canned per-family templates, *detected from the special tokens
//! actually present in the vocabulary* rather than by parsing a Jinja
//! `chat_template` string. That keeps the dependency-free stance (no template
//! engine) and is the same pragmatic choice the popular runners make. The
//! tokenizer that runs inference is the one that speaks every model's chat
//! dialect — one tissue, not a bolted-on formatter.

use crate::model::Arch;
use crate::tokenizer::Tokenizer;

/// A single conversation turn.
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn new(role: &str, content: &str) -> Message {
        Message {
            role: role.to_string(),
            content: content.to_string(),
        }
    }
}

/// The chat dialect a model was trained on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatFamily {
    /// `<|im_start|>role\n…<|im_end|>` — Qwen2/2.5, SmolLM2, OLMoE.
    ChatMl,
    /// `<|start_header_id|>role<|end_header_id|>\n\n…<|eot_id|>` — Llama-3.
    Llama3,
    /// `<start_of_turn>role\n…<end_of_turn>` — Gemma.
    Gemma,
    /// `<|user|>\n…<|end|>` — Phi-3.
    Phi3,
    /// `[INST] … [/INST]` — Mistral.
    Mistral,
    /// No known turn markers: concatenate content (base / completion model).
    Plain,
}

impl ChatFamily {
    /// Detect the dialect from the special tokens present in the vocab, falling
    /// back to the architecture. Presence-based so a model that *has* the
    /// markers is templated even if its arch name is unusual.
    pub fn detect(tok: &Tokenizer, arch: Arch) -> ChatFamily {
        if tok.special_id("<|im_start|>").is_some() {
            ChatFamily::ChatMl
        } else if tok.special_id("<|start_header_id|>").is_some() {
            ChatFamily::Llama3
        } else if tok.special_id("<start_of_turn>").is_some() {
            ChatFamily::Gemma
        } else if tok.special_id("<|assistant|>").is_some() && tok.special_id("<|user|>").is_some()
        {
            ChatFamily::Phi3
        } else if matches!(arch, Arch::Mistral) {
            ChatFamily::Mistral
        } else {
            ChatFamily::Plain
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ChatFamily::ChatMl => "chatml",
            ChatFamily::Llama3 => "llama-3",
            ChatFamily::Gemma => "gemma",
            ChatFamily::Phi3 => "phi-3",
            ChatFamily::Mistral => "mistral",
            ChatFamily::Plain => "plain",
        }
    }
}

fn sp(ids: &mut Vec<u32>, tok: &Tokenizer, s: &str) {
    if let Some(id) = tok.special_id(s) {
        ids.push(id);
    }
}

fn tx(ids: &mut Vec<u32>, tok: &Tokenizer, s: &str) {
    ids.extend(tok.encode(s, false));
}

/// Render a conversation to the token ids the model expects. When
/// `add_generation_prompt` is set, the open of the assistant turn is appended so
/// the model continues as the assistant.
pub fn build_ids(
    tok: &Tokenizer,
    fam: ChatFamily,
    msgs: &[Message],
    add_generation_prompt: bool,
) -> Vec<u32> {
    let mut ids = Vec::new();
    match fam {
        ChatFamily::ChatMl => {
            for m in msgs {
                sp(&mut ids, tok, "<|im_start|>");
                tx(&mut ids, tok, &format!("{}\n", m.role));
                tx(&mut ids, tok, &m.content);
                sp(&mut ids, tok, "<|im_end|>");
                tx(&mut ids, tok, "\n");
            }
            if add_generation_prompt {
                sp(&mut ids, tok, "<|im_start|>");
                tx(&mut ids, tok, "assistant\n");
            }
        }
        ChatFamily::Llama3 => {
            sp(&mut ids, tok, "<|begin_of_text|>");
            for m in msgs {
                sp(&mut ids, tok, "<|start_header_id|>");
                tx(&mut ids, tok, &m.role);
                sp(&mut ids, tok, "<|end_header_id|>");
                tx(&mut ids, tok, &format!("\n\n{}", m.content));
                sp(&mut ids, tok, "<|eot_id|>");
            }
            if add_generation_prompt {
                sp(&mut ids, tok, "<|start_header_id|>");
                tx(&mut ids, tok, "assistant");
                sp(&mut ids, tok, "<|end_header_id|>");
                tx(&mut ids, tok, "\n\n");
            }
        }
        ChatFamily::Gemma => {
            sp(&mut ids, tok, "<bos>");
            for m in msgs {
                // Gemma has no system role and calls the assistant "model".
                let role = match m.role.as_str() {
                    "assistant" => "model",
                    "system" => "user",
                    r => r,
                };
                sp(&mut ids, tok, "<start_of_turn>");
                tx(&mut ids, tok, &format!("{}\n{}", role, m.content));
                sp(&mut ids, tok, "<end_of_turn>");
                tx(&mut ids, tok, "\n");
            }
            if add_generation_prompt {
                sp(&mut ids, tok, "<start_of_turn>");
                tx(&mut ids, tok, "model\n");
            }
        }
        ChatFamily::Phi3 => {
            for m in msgs {
                let marker = match m.role.as_str() {
                    "assistant" => "<|assistant|>",
                    "system" => "<|system|>",
                    _ => "<|user|>",
                };
                sp(&mut ids, tok, marker);
                tx(&mut ids, tok, &format!("\n{}", m.content));
                sp(&mut ids, tok, "<|end|>");
                tx(&mut ids, tok, "\n");
            }
            if add_generation_prompt {
                sp(&mut ids, tok, "<|assistant|>");
                tx(&mut ids, tok, "\n");
            }
        }
        ChatFamily::Mistral => {
            // [INST]/[/INST] are plain text for Mistral; bos opens the sequence.
            if let Some(b) = tok.bos {
                ids.push(b);
            }
            // Fold any leading system message into the first instruction.
            let mut sys = String::new();
            for m in msgs {
                match m.role.as_str() {
                    "system" => sys = m.content.clone(),
                    "user" => {
                        let body = if sys.is_empty() {
                            m.content.clone()
                        } else {
                            let s = format!("{}\n\n{}", sys, m.content);
                            sys.clear();
                            s
                        };
                        tx(&mut ids, tok, &format!("[INST] {} [/INST]", body));
                    }
                    _ => tx(&mut ids, tok, &format!(" {}", m.content)),
                }
            }
        }
        ChatFamily::Plain => {
            if let Some(b) = tok.bos {
                ids.push(b);
            }
            for m in msgs {
                tx(&mut ids, tok, &m.content);
                tx(&mut ids, tok, "\n");
            }
        }
    }
    ids
}

/// The token ids that end an assistant turn for this dialect. The model's own
/// `eos` is always a stop; the family adds its turn-terminator on top.
pub fn stop_ids(tok: &Tokenizer, fam: ChatFamily) -> Vec<u32> {
    let mut stops = Vec::new();
    if let Some(e) = tok.eos {
        stops.push(e);
    }
    let extra: &[&str] = match fam {
        ChatFamily::ChatMl => &["<|im_end|>"],
        ChatFamily::Llama3 => &["<|eot_id|>"],
        ChatFamily::Gemma => &["<end_of_turn>"],
        ChatFamily::Phi3 => &["<|end|>"],
        ChatFamily::Mistral | ChatFamily::Plain => &[],
    };
    for s in extra {
        if let Some(id) = tok.special_id(s) {
            if !stops.contains(&id) {
                stops.push(id);
            }
        }
    }
    stops
}

/// String-level stop sequences — a backstop for models that never learned to emit
/// a stop token (e.g. a base fine-tuned on flat `User:/Assistant:/System:` or
/// `Instruction:/Response:` transcripts). The generator ends the turn when the
/// decoded tail starts a new role header. Newline-prefixed so they don't trip on
/// the same words appearing mid-answer. Applies to every dialect (harmless when
/// the model already stops on its token).
pub fn stop_strs(_fam: ChatFamily) -> Vec<String> {
    [
        "\nUser:",
        "\nSystem:",
        "\nAssistant:",
        "\nInstruction:",
        "\nResponse:",
        "\nuser:",
        "\nassistant:",
        "\nsystem:",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}
