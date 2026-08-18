//! flwr — the app you talk to, built on the HOS engine.
//!
//! HOS is the body: the engine, the tensor core, the formats. `flwr` (Greek for
//! "body") is the thing that brings one to life and lets you converse with it —
//! what `ollama` is to `llama.cpp`, except it is a separate binary over the
//! `hos` *library*, so it does not touch the `hos` engine command at all.
//!
//!   flwr run <model> [--gpu] [-p "one-shot prompt"]
//!   flwr serve <model> [--host H] [--port P]
//!
//! The model's own chat dialect is detected and applied (HOS ships canned
//! per-family templates); `serve` exposes an OpenAI-compatible HTTP API.

mod chats;
mod gemma4_chat;
mod memory;
mod reasoning;
mod serve;
mod store;

use std::io::Write;
use std::path::{Path, PathBuf};

struct Opts {
    cmd: String,
    model: Option<String>,
    prompt: Option<String>,
    n_predict: usize,
    temp: f32,
    top_k: usize,
    top_p: f32,
    rep_penalty: f32,
    repeat_last_n: usize,
    seed: u64,
    gpu: bool,
    host: String,
    port: u16,
    revision: String,
    name: Option<String>,
    dest: Option<String>,
    qtype: String,
    awq: bool,
    no_think: bool,
    effort: Option<String>,
    hide_thinking: bool,
    image: Option<String>,
    mmproj: Option<String>,
}

impl Opts {
    /// Thinking config for hybrid reasoning models (qwen35).
    fn think(&self) -> hos::qwen35::Think {
        let effort = self
            .effort
            .as_deref()
            .and_then(hos::qwen35::Effort::parse)
            .unwrap_or(hos::qwen35::Effort::Xhigh);
        hos::qwen35::Think {
            on: !self.no_think,
            effort,
        }
    }
}

fn usage() -> ! {
    eprintln!("flwr — talk to a model.   Powered by HOS.");
    eprintln!();
    eprintln!("  flwr run <model> [--cpu] [-p \"prompt\"] [-n N] [--temp T] [--seed S]");
    eprintln!("  flwr serve <model> [--cpu] [--host 127.0.0.1] [--port 11434]");
    eprintln!("  flwr pull <hf-repo|gguf-url> [--revision main] [--name X]");
    eprintln!("  flwr list");
    eprintln!("  flwr show <name>");
    eprintln!("  flwr cp <src> <dst>");
    eprintln!("  flwr quantize <src> <dst> [--type q4_0] [--awq]");
    eprintln!("  flwr rm <name>");
    eprintln!();
    eprintln!("<model> is a path (.flwr or .hos), a name in the store (flwr pull), or a");
    eprintln!("bare name resolved from $HOS_MODELS_DIR, ~/Documents/hos/models, ~/.hos/models.");
    eprintln!();
    eprintln!("Metal GPU is the default on Apple Silicon (--cpu to force CPU); x86 runs on");
    eprintln!("AVX2-accelerated CPU automatically. quickstart:  flwr run flwr-bloom");
    std::process::exit(1);
}

fn parse() -> Opts {
    let mut o = Opts {
        cmd: String::new(),
        model: None,
        prompt: None,
        n_predict: 512,
        temp: 0.7,
        top_k: 40,
        top_p: 0.95,
        rep_penalty: 1.1,
        repeat_last_n: 64,
        seed: 42,
        // Default to the Metal GPU on Apple Silicon (where it is ~6x faster and
        // always present) so Mac users get it without knowing the flag exists.
        // `--cpu` forces CPU. Elsewhere Metal is not built, so this stays false.
        gpu: cfg!(all(target_os = "macos", target_arch = "aarch64")),
        host: "127.0.0.1".to_string(),
        port: 11434, // the de-facto local-LLM port — drop-in for existing clients
        revision: "main".to_string(),
        name: None,
        dest: None,
        qtype: "q8_0".to_string(),
        awq: false,
        no_think: false,
        effort: None,
        hide_thinking: false,
        image: None,
        mmproj: None,
    };
    let mut it = std::env::args().skip(1);
    o.cmd = it.next().unwrap_or_default();
    if o.cmd.is_empty() || o.cmd == "-h" || o.cmd == "--help" {
        usage();
    }
    while let Some(a) = it.next() {
        match a.as_str() {
            "-m" | "--model" => o.model = it.next(),
            "-p" | "--prompt" => o.prompt = it.next(),
            "-n" | "--n-predict" => {
                o.n_predict = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(o.n_predict)
            }
            "--temp" => o.temp = it.next().and_then(|v| v.parse().ok()).unwrap_or(o.temp),
            "--top-k" => o.top_k = it.next().and_then(|v| v.parse().ok()).unwrap_or(o.top_k),
            "--top-p" => o.top_p = it.next().and_then(|v| v.parse().ok()).unwrap_or(o.top_p),
            "--seed" => o.seed = it.next().and_then(|v| v.parse().ok()).unwrap_or(o.seed),
            "--host" => o.host = it.next().unwrap_or(o.host),
            "--port" => o.port = it.next().and_then(|v| v.parse().ok()).unwrap_or(o.port),
            "--revision" | "--rev" => o.revision = it.next().unwrap_or(o.revision),
            "--name" => o.name = it.next(),
            "--type" | "-t" => o.qtype = it.next().unwrap_or(o.qtype),
            "--awq" => o.awq = true,
            "--gpu" => o.gpu = true,
            "--cpu" => o.gpu = false,
            "--no-think" | "--no-thinking" => o.no_think = true,
            "--effort" | "--reasoning-effort" => o.effort = it.next(),
            "--hide-thinking" | "--hide-reasoning" => o.hide_thinking = true,
            "--image" | "--img" => o.image = it.next(),
            "--mmproj" => o.mmproj = it.next(),
            other if !other.starts_with('-') && o.model.is_none() => {
                o.model = Some(other.to_string())
            }
            // a second bare positional is the destination name (for `flwr cp`).
            other if !other.starts_with('-') && o.dest.is_none() => {
                o.dest = Some(other.to_string())
            }
            other => {
                eprintln!("flwr: unknown argument '{other}'");
                usage();
            }
        }
    }
    o
}

fn main() {
    let o = parse();
    match o.cmd.as_str() {
        "run" => cmd_run(&o),
        "serve" => cmd_serve(&o),
        "membench" | "bench-memory" => cmd_membench(&o),
        "bloom" | "viz" => cmd_bloom(),
        "pull" => {
            let spec = o.model.clone().unwrap_or_else(|| {
                eprintln!("flwr pull: need a HuggingFace repo id or a .gguf URL");
                usage();
            });
            store::pull(&spec, &o.revision, o.name.as_deref());
        }
        "list" | "ls" => store::list(),
        "show" => {
            let name = o.model.clone().unwrap_or_else(|| {
                eprintln!("flwr show: need a model name (try: flwr list)");
                usage();
            });
            store::show(&name);
        }
        "rm" => {
            let name = o.model.clone().unwrap_or_else(|| {
                eprintln!("flwr rm: need a model name (try: flwr list)");
                usage();
            });
            store::rm(&name);
        }
        "cp" => {
            let (Some(src), Some(dst)) = (o.model.clone(), o.dest.clone()) else {
                eprintln!("flwr cp: need a source and a destination name");
                usage();
            };
            store::cp(&src, &dst);
        }
        "quantize" => {
            let (Some(src), Some(dst)) = (o.model.clone(), o.dest.clone()) else {
                eprintln!("flwr quantize: need a source model and a destination name");
                usage();
            };
            store::quantize(&src, &dst, &o.qtype, o.awq);
        }
        other => {
            eprintln!("flwr: unknown command '{other}'");
            usage();
        }
    }
}

fn cmd_run(o: &Opts) {
    let path = resolve_model(o.model.clone());
    // Gemma-4 is a custom HOS arch (not the generic Engine path); route it to its
    // own all-HOS chat backend. Everything else stays on hos::Engine.
    if gemma4_chat::is_gemma4(&path) || gemma4_chat::is_gemma4_capsule(&path) {
        gemma4_chat::run(&path, &gemma4_opts(o));
        return;
    }
    // qwen35 hybrid (Gated-DeltaNet + attention) is a custom HOS arch on its own
    // ChatSession backend; everything else stays on the generic Engine.
    if let Ok(g) = hos::gguf::Gguf::open(&path) {
        if hos::model::Arch::detect(&g) == hos::model::Arch::Qwen35Hybrid {
            cmd_run_qwen35(&path, o);
            return;
        }
    }
    let name = model_name_of(&path);
    let mut eng = load(&path, o.gpu);
    let backend = if o.gpu { "metal gpu" } else { "cpu" };

    let turn = |eng: &mut hos::Engine, history: &[hos::chat::Message]| -> String {
        print!(
            "  {}{}flwr{}  ",
            hos::viz::BOLD,
            hos::viz::petal(),
            hos::viz::RESET
        );
        std::io::stdout().flush().ok();
        let mut reply = String::new();
        eng.chat(
            history,
            o.n_predict,
            o.temp,
            o.top_k,
            o.top_p,
            o.rep_penalty,
            o.repeat_last_n,
            o.seed,
            |piece| {
                print!("{piece}");
                std::io::stdout().flush().ok();
                reply.push_str(piece);
            },
        );
        println!("\n");
        reply
    };

    // one-shot mode: -p supplies a single user turn, print the reply, exit.
    // Show the flwr flower here too — the brand moment shouldn't be interactive-only.
    if let Some(p) = &o.prompt {
        print!("{}", hos::viz::banner());
        let history = vec![hos::chat::Message::new("user", p)];
        turn(&mut eng, &history);
        return;
    }

    organism_banner(&eng, &name, backend);
    println!(
        "    {}/role <text> · /continue · /list · /open <id> · /new · /bye{}",
        hos::viz::faint(),
        hos::viz::RESET
    );
    let mut history: Vec<hos::chat::Message> = Vec::new();
    let mut last_memory_sig = String::new();
    let mut chat_id = crate::chats::new_id();
    let model_meta = serde_json::json!({ "name": name.clone() });
    let params_meta = serde_json::json!({
        "temp": o.temp, "top_k": o.top_k, "top_p": o.top_p,
        "n_predict": o.n_predict, "seed": o.seed, "gpu": o.gpu,
    });
    let stdin = std::io::stdin();
    loop {
        print!(
            "  {}{}you{}   ",
            hos::viz::BOLD,
            hos::viz::faint(),
            hos::viz::RESET
        );
        std::io::stdout().flush().ok();
        let mut line = String::new();
        match stdin.read_line(&mut line) {
            Ok(0) => break, // EOF (Ctrl-D)
            Ok(_) => {}
            Err(_) => break,
        }
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        if matches!(text, "/bye" | "/exit" | "/quit") {
            break;
        }
        if matches!(text, "/reset" | "/new") {
            history.clear();
            chat_id = crate::chats::new_id();
            println!("    · new conversation.\n");
            continue;
        }
        if matches!(text, "/list" | "/chats") {
            print_chat_list();
            continue;
        }
        if let Some(rest) = text.strip_prefix("/open") {
            let id = rest.trim();
            if id.is_empty() {
                println!("    · usage: /open <id>   (see /list)\n");
            } else if let Some(msgs) = load_chat_history(id) {
                history = msgs;
                chat_id = id.to_string();
                replay_history(&history);
            } else {
                println!("    · no chat '{id}'. try /list\n");
            }
            continue;
        }
        if matches!(text, "/continue" | "/resume") {
            match most_recent_saved_chat().filter(|id| load_chat_history(id).is_some()) {
                Some(id) => {
                    history = load_chat_history(&id).unwrap();
                    chat_id = id;
                    replay_history(&history);
                }
                None => println!("    · no saved conversation to continue.\n"),
            }
            continue;
        }
        if let Some(rest) = text.strip_prefix("/role") {
            let r = rest.trim();
            let has_sys = history.first().map(|m| m.role == "system").unwrap_or(false);
            if r.is_empty() {
                match history.first().filter(|m| m.role == "system") {
                    Some(m) => println!("    · role: {}\n", m.content),
                    None => println!(
                        "    · no role set. usage: /role <instruction>   ·   /role clear\n"
                    ),
                }
            } else if matches!(r, "clear" | "none" | "off") {
                if has_sys {
                    history.remove(0);
                }
                println!("    · role cleared.\n");
            } else {
                let sys = hos::chat::Message::new("system", r);
                if has_sys {
                    history[0] = sys;
                } else {
                    history.insert(0, sys);
                }
                println!("    · role set. the model follows it for this conversation.\n");
            }
            continue;
        }
        if text.starts_with('/') {
            println!(
                "    · commands:  /role <text>   /list   /open <id>   /continue   /new   /bye\n"
            );
            continue;
        }

        history.push(hos::chat::Message::new("user", text));
        let bundle = memory::assemble(&history);
        let sig = memory::receipt_signature(&bundle.memory);
        if bundle.omitted_messages > 0 && (memory::debug_compaction() || sig != last_memory_sig) {
            eprintln!(
                "    · using memory receipts for {} older messages (~{} tok prompt)",
                bundle.omitted_messages, bundle.estimated_tokens
            );
        }
        last_memory_sig = sig;
        let reply = turn(&mut eng, &bundle.messages);
        history.push(hos::chat::Message::new("assistant", reply.trim()));

        // Autosave so the conversation can be reopened with /open.
        let _ = crate::chats::save(
            &chat_id,
            &history_to_json(&history),
            model_meta.clone(),
            params_meta.clone(),
            memory::to_json(&bundle.memory),
        );
    }
    if !history.is_empty() {
        println!("    · saved. continue later with:  /open {chat_id}");
    }
    println!("    · session closed.");
}

/// Look for an `mmproj-*.gguf` (the vision tower) next to the model, so `--image`
/// works without the user pointing at it explicitly.
fn find_sibling_mmproj(model: &Path) -> Option<String> {
    let dir = model.parent()?;
    for e in std::fs::read_dir(dir).ok()?.flatten() {
        let n = e.file_name();
        let name = n.to_string_lossy();
        if name.starts_with("mmproj") && name.ends_with(".gguf") {
            return Some(e.path().to_string_lossy().into_owned());
        }
    }
    None
}

/// `flwr run <qwen35>` — the same interactive/one-shot chat REPL as `cmd_run`,
/// but driven by the qwen35 `ChatSession` backend (Gated-DeltaNet hybrid). Reuses
/// the shared memory + persistence helpers, so /role /list /open /continue all work.
fn cmd_run_qwen35(path: &Path, o: &Opts) {
    use std::io::Write;
    let name = model_name_of(path);
    let backend = if o.gpu { "metal gpu" } else { "cpu" };
    let g = match hos::gguf::Gguf::open(path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("[flwr] cannot open model: {e}");
            return;
        }
    };
    let tok = match hos::tokenizer::Tokenizer::from_gguf(&g) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[flwr] tokenizer: {e}");
            return;
        }
    };
    let mut sess = match hos::qwen35::ChatSession::load(&g, tok, o.gpu) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[flwr] load: {e}");
            return;
        }
    };

    // Vision: with --image, attach the tower (from --mmproj or an auto-detected
    // sibling mmproj-*.gguf) and encode the image once; it's spliced into every
    // turn's last user message so the whole conversation can reference it.
    let img_emb: Option<Vec<f32>> = match &o.image {
        Some(imgp) => {
            let mm = o.mmproj.clone().or_else(|| find_sibling_mmproj(path));
            match mm.and_then(|p| hos::gguf::Gguf::open(std::path::Path::new(&p)).ok()) {
                Some(mg) => match sess.attach_vision(&mg).and_then(|_| {
                    eprintln!("[flwr] encoding image {imgp} ...");
                    sess.encode_image(std::path::Path::new(imgp))
                }) {
                    Ok(e) => {
                        eprintln!("[flwr] image -> {} vision tokens", e.len() / 5120);
                        Some(e)
                    }
                    Err(e) => {
                        eprintln!("[flwr] vision: {e}");
                        None
                    }
                },
                None => {
                    eprintln!("[flwr] --image needs --mmproj <mmproj.gguf> (or a sibling mmproj-*.gguf)");
                    None
                }
            }
        }
        None => None,
    };
    let img_ref = img_emb.as_deref();

    let think = o.think();
    let hide = o.hide_thinking;
    let effort_label = match think.effort {
        hos::qwen35::Effort::Low => "low",
        hos::qwen35::Effort::Medium => "medium",
        hos::qwen35::Effort::Xhigh => "xhigh",
    };
    // Returns (answer, reasoning_trace, tokens). The answer goes to history; the
    // reasoning is streamed dimmed AND kept for a content-addressed receipt.
    let turn = |sess: &mut hos::qwen35::ChatSession,
                history: &[hos::chat::Message]|
     -> (String, String, usize) {
        use hos::qwen35::Chunk;
        print!(
            "  {}{}flwr{}  ",
            hos::viz::BOLD,
            hos::viz::petal(),
            hos::viz::RESET
        );
        std::io::stdout().flush().ok();
        let mut reply = String::new();
        let mut reasoning = String::new();
        let mut in_reason = false;
        let n = sess.chat_img(
            history,
            img_ref,
            think,
            o.n_predict,
            o.temp,
            o.top_k,
            o.top_p,
            o.rep_penalty,
            o.repeat_last_n,
            o.seed,
            |chunk| {
                match chunk {
                    Chunk::Reasoning(t) => {
                        reasoning.push_str(t);
                        if !hide {
                            if !in_reason {
                                print!("{}", hos::viz::faint());
                                in_reason = true;
                            }
                            print!("{t}");
                        }
                    }
                    Chunk::Answer(t) => {
                        if in_reason {
                            print!("{}\n\n  ", hos::viz::RESET);
                            in_reason = false;
                        }
                        print!("{t}");
                        reply.push_str(t);
                    }
                }
                std::io::stdout().flush().ok();
            },
        );
        if in_reason {
            print!("{}", hos::viz::RESET);
        }
        println!("\n");
        (reply, reasoning, n)
    };

    // one-shot mode.
    if let Some(p) = &o.prompt {
        print!("{}", hos::viz::banner());
        let history = vec![hos::chat::Message::new("user", p)];
        let (answer, reasoning, n) = turn(&mut sess, &history);
        let rr = reasoning::ReasoningReceipt::new(
            "oneshot", 0, p, effort_label, &reasoning, &answer, n,
            serde_json::json!({ "name": name }),
        );
        if let Ok(Some(id)) = reasoning::save(&rr) {
            eprintln!("    · reasoning receipt {id}");
        }
        return;
    }

    qwen35_banner(&sess, &name, backend);
    println!(
        "    {}/role <text> · /continue · /list · /open <id> · /new · /bye{}",
        hos::viz::faint(),
        hos::viz::RESET
    );
    let mut history: Vec<hos::chat::Message> = Vec::new();
    let mut last_memory_sig = String::new();
    let mut chat_id = crate::chats::new_id();
    let model_meta = serde_json::json!({ "name": name.clone() });
    let params_meta = serde_json::json!({
        "temp": o.temp, "top_k": o.top_k, "top_p": o.top_p,
        "n_predict": o.n_predict, "seed": o.seed, "gpu": o.gpu,
    });
    let stdin = std::io::stdin();
    loop {
        print!(
            "  {}{}you{}   ",
            hos::viz::BOLD,
            hos::viz::faint(),
            hos::viz::RESET
        );
        std::io::stdout().flush().ok();
        let mut line = String::new();
        match stdin.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(_) => break,
        }
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        if matches!(text, "/bye" | "/exit" | "/quit") {
            break;
        }
        if matches!(text, "/reset" | "/new") {
            history.clear();
            chat_id = crate::chats::new_id();
            println!("    · new conversation.\n");
            continue;
        }
        if matches!(text, "/list" | "/chats") {
            print_chat_list();
            continue;
        }
        if let Some(rest) = text.strip_prefix("/open") {
            let id = rest.trim();
            if id.is_empty() {
                println!("    · usage: /open <id>   (see /list)\n");
            } else if let Some(msgs) = load_chat_history(id) {
                history = msgs;
                chat_id = id.to_string();
                replay_history(&history);
            } else {
                println!("    · no chat '{id}'. try /list\n");
            }
            continue;
        }
        if matches!(text, "/continue" | "/resume") {
            match most_recent_saved_chat().filter(|id| load_chat_history(id).is_some()) {
                Some(id) => {
                    history = load_chat_history(&id).unwrap();
                    chat_id = id;
                    replay_history(&history);
                }
                None => println!("    · no saved conversation to continue.\n"),
            }
            continue;
        }
        if let Some(rest) = text.strip_prefix("/role") {
            let r = rest.trim();
            let has_sys = history.first().map(|m| m.role == "system").unwrap_or(false);
            if r.is_empty() {
                match history.first().filter(|m| m.role == "system") {
                    Some(m) => println!("    · role: {}\n", m.content),
                    None => {
                        println!("    · no role set. usage: /role <instruction>   ·   /role clear\n")
                    }
                }
            } else if matches!(r, "clear" | "none" | "off") {
                if has_sys {
                    history.remove(0);
                }
                println!("    · role cleared.\n");
            } else {
                let sys = hos::chat::Message::new("system", r);
                if has_sys {
                    history[0] = sys;
                } else {
                    history.insert(0, sys);
                }
                println!("    · role set. the model follows it for this conversation.\n");
            }
            continue;
        }
        if text.starts_with('/') {
            println!(
                "    · commands:  /role <text>   /list   /open <id>   /continue   /new   /bye\n"
            );
            continue;
        }

        history.push(hos::chat::Message::new("user", text));
        let bundle = memory::assemble(&history);
        let sig = memory::receipt_signature(&bundle.memory);
        if bundle.omitted_messages > 0 && (memory::debug_compaction() || sig != last_memory_sig) {
            eprintln!(
                "    · using memory receipts for {} older messages (~{} tok prompt)",
                bundle.omitted_messages, bundle.estimated_tokens
            );
        }
        last_memory_sig = sig;
        let (reply, reasoning, n) = turn(&mut sess, &bundle.messages);
        history.push(hos::chat::Message::new("assistant", reply.trim()));
        // Capture the reasoning as a content-addressed receipt agents can build on.
        let rr = reasoning::ReasoningReceipt::new(
            &chat_id,
            history.len().saturating_sub(1),
            text,
            effort_label,
            &reasoning,
            &reply,
            n,
            model_meta.clone(),
        );
        if let Ok(Some(id)) = reasoning::save(&rr) {
            if memory::debug_compaction() {
                eprintln!("    · reasoning receipt {id}");
            }
        }
        let _ = crate::chats::save(
            &chat_id,
            &history_to_json(&history),
            model_meta.clone(),
            params_meta.clone(),
            memory::to_json(&bundle.memory),
        );
    }
    if !history.is_empty() {
        println!("    · saved. continue later with:  /open {chat_id}");
    }
    println!("    · session closed.");
}

/// Compact organism banner for the qwen35 ChatSession backend.
fn qwen35_banner(sess: &hos::qwen35::ChatSession, name: &str, backend: &str) {
    use hos::viz::*;
    let (dim, n_layers, n_heads, n_kv, ctx_len) = sess.dims();
    let back_col = if backend.contains("gpu") {
        signal()
    } else {
        faint()
    };
    print!("{}", banner());
    let lines = vec![
        format!(
            "{}body{}     {}{name}{}  {}(qwen35 hybrid){}",
            BOLD,
            RESET,
            petal(),
            RESET,
            faint(),
            RESET
        ),
        format!(
            "{}anatomy{}  {}dim {} · {} layers · {}h/{}kv · ctx {}{}",
            BOLD, RESET, ink(), dim, n_layers, n_heads, n_kv, ctx_len, RESET
        ),
        format!(
            "{}dialect{}  {}{}{}",
            BOLD,
            RESET,
            ctx(),
            sess.family_label(),
            RESET
        ),
        format!("{}backend{}  {}{}{}", BOLD, RESET, back_col, backend, RESET),
        format!(
            "{}engine{}   {}gated-deltanet + attention · Powered by HOS{}",
            BOLD,
            RESET,
            faint(),
            RESET
        ),
        String::new(),
        status("BLOOM STATE: READY", &stem()),
        format!("{}/bye ends · /reset clears memory{}", faint(), RESET),
    ];
    println!("{}", frame("FLWR // CHAT", &lines, 44));
    print!("{}", footer_nav());
}

fn history_to_json(history: &[hos::chat::Message]) -> serde_json::Value {
    serde_json::Value::Array(
        history
            .iter()
            .map(|m| serde_json::json!({ "role": m.role, "content": m.content }))
            .collect(),
    )
}

fn load_chat_history(id: &str) -> Option<Vec<hos::chat::Message>> {
    let raw = crate::chats::load(id)?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    Some(memory::messages_from_json(&v["messages"]))
}

fn replay_history(history: &[hos::chat::Message]) {
    use hos::viz::*;
    println!();
    for m in history {
        if m.role == "system" {
            continue;
        }
        let (label, col) = if m.role == "assistant" {
            ("flwr", petal())
        } else {
            ("you ", faint())
        };
        println!("  {BOLD}{col}{label}{RESET}   {}", m.content);
    }
    println!("    · continuing this conversation.\n");
}

fn print_chat_list() {
    use hos::viz::*;
    let list = crate::chats::list();
    let items = list["data"].as_array().cloned().unwrap_or_default();
    if items.is_empty() {
        println!("    · no saved conversations yet.\n");
        return;
    }
    println!();
    for it in items.iter().take(20) {
        let id = it["id"].as_str().unwrap_or("?");
        let title = it["title"].as_str().unwrap_or("untitled");
        let n = it["messages"].as_u64().unwrap_or(0);
        println!("  {}{id}{RESET}  {}{title}{RESET}  {}({n} msgs){RESET}", pollen(), ink(), faint());
    }
    println!("    · open one with:  /open <id>\n");
}

/// `flwr membench [chat-id]` — measure conversation compression: raw vs packed
/// tokens, ratio, omitted messages, and whether the packed prompt stayed within
/// budget, across a range of context budgets. Uses a saved chat if given (or the
/// largest one on disk), else a synthetic long conversation.
fn cmd_membench(o: &Opts) {
    use hos::viz::*;
    let (src, msgs): (String, Vec<hos::chat::Message>) = match o.model.as_deref() {
        Some(s) if load_chat_history(s).is_some() => {
            (format!("chat {s}"), load_chat_history(s).unwrap())
        }
        // Only auto-pick a saved chat if it is long enough to actually compress;
        // otherwise use the synthetic conversation that demonstrates the ratios.
        _ => match largest_saved_chat()
            .filter(|id| load_chat_history(id).map(|m| m.len() >= 24).unwrap_or(false))
        {
            Some(id) => (format!("saved chat {id}"), load_chat_history(&id).unwrap_or_default()),
            None => ("synthetic 200-turn conversation".to_string(), synthetic_convo()),
        },
    };
    let msgs = if msgs.is_empty() { synthetic_convo() } else { msgs };

    let raw = memory::raw_tokens(&msgs);
    println!();
    println!("  {BOLD}MEMORY BENCHMARK{RESET}   source: {src}");
    println!(
        "  {} messages · ~{raw} raw tokens · recent window {} turns\n",
        msgs.len(),
        memory::recent_turns()
    );
    println!(
        "  {BOLD}{:>8}  {:>10}  {:>10}  {:>7}  {:>8}  within?{RESET}",
        "budget", "raw tok", "packed", "ratio", "omitted"
    );
    for budget in [512usize, 1024, 2048, 3072, 4096] {
        let b = memory::assemble_with(&msgs, budget, memory::recent_turns());
        let ratio = raw as f64 / b.estimated_tokens.max(1) as f64;
        let mark = if b.estimated_tokens <= budget {
            format!("{}yes{RESET}", stem())
        } else {
            format!("{}NO{RESET}", petal())
        };
        println!(
            "  {:>8}  {:>10}  {:>10}  {:>6.2}x  {:>8}  {mark}",
            budget, raw, b.estimated_tokens, ratio, b.omitted_messages
        );
    }
    println!();
}

fn synthetic_convo() -> Vec<hos::chat::Message> {
    let mut v = Vec::new();
    for i in 0..200 {
        v.push(hos::chat::Message::new(
            "user",
            &format!("Turn {i}: let's work on module {i}. We need to remember the goal, the constraints, and the decisions we made earlier in this long running conversation."),
        ));
        v.push(hos::chat::Message::new(
            "assistant",
            &format!("Reply {i}: acknowledged. Here is a reasonably detailed response about module {i} that consumes a realistic number of tokens for the benchmark."),
        ));
    }
    v
}

fn largest_saved_chat() -> Option<String> {
    let list = crate::chats::list();
    list["data"]
        .as_array()?
        .iter()
        .max_by_key(|it| it["messages"].as_u64().unwrap_or(0))
        .and_then(|it| it["id"].as_str().map(String::from))
}

/// The most recently updated saved chat (chats::list is sorted newest-first).
fn most_recent_saved_chat() -> Option<String> {
    let list = crate::chats::list();
    list["data"].as_array()?.first()?["id"]
        .as_str()
        .map(String::from)
}

fn cmd_serve(o: &Opts) {
    let path = resolve_model(o.model.clone());
    let name = model_name_of(&path);
    let backend = match serve::Backend::load(&path, o.gpu) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("flwr: could not load model: {e}");
            std::process::exit(1);
        }
    };
    serve::serve(backend, &name, &o.host, o.port, o.gpu);
}

/// Build the gemma4 chat backend's options from flwr's `Opts`.
fn gemma4_opts(o: &Opts) -> gemma4_chat::Opts {
    gemma4_chat::Opts {
        gpu: o.gpu,
        n_predict: o.n_predict,
        temp: o.temp,
        top_k: o.top_k,
        top_p: o.top_p,
        rep_penalty: o.rep_penalty,
        repeat_last_n: o.repeat_last_n,
        seed: o.seed,
        prompt: o.prompt.clone(),
    }
}

/// Present a loaded model as a flwr "organism" card — color-coded, framed.
fn organism_banner(eng: &hos::Engine, name: &str, backend: &str) {
    use hos::viz::*;
    let c = &eng.model.cfg;
    let arch = format!("{:?}", c.arch).to_lowercase();
    let back_col = if backend.contains("gpu") {
        signal()
    } else {
        faint()
    };
    print!("{}", banner());
    let lines = vec![
        format!(
            "{}body{}     {}{name}{}  {}({arch}){}",
            BOLD,
            RESET,
            petal(),
            RESET,
            faint(),
            RESET
        ),
        format!(
            "{}anatomy{}  {}dim {} · {} layers · {}h/{}kv · ctx {}{}",
            BOLD,
            RESET,
            ink(),
            c.dim,
            c.n_layers,
            c.n_heads,
            c.n_kv_heads,
            c.ctx_len,
            RESET
        ),
        format!(
            "{}dialect{}  {}{}{}",
            BOLD,
            RESET,
            ctx(),
            eng.chat_family().label(),
            RESET
        ),
        format!("{}backend{}  {}{}{}", BOLD, RESET, back_col, backend, RESET),
        format!(
            "{}engine{}   {}Powered by HOS{}",
            BOLD,
            RESET,
            faint(),
            RESET
        ),
        String::new(),
        status("BLOOM STATE: READY", &stem()),
        format!("{}/bye ends · /reset clears memory{}", faint(), RESET),
    ];
    println!("{}", frame("FLWR // CHAT", &lines, 44));
    print!("{}", footer_nav());
}

/// `flwr bloom` — showcase the visual lib (no model needed).
fn cmd_bloom() {
    use hos::viz::*;
    print!("{}", banner());
    // a static GROW-MODE panel at a few growth stages
    for (mode, prog, loss) in [
        ("SEED", 0.18f32, 3.1f32),
        ("GROW MODE", 0.62, 1.2),
        ("BLOOM", 0.94, 0.34),
    ] {
        let signal = (0.4 + prog * 0.6).min(1.0);
        let dream = (prog * 1.2).min(1.0);
        println!("{}", grow_panel(mode, prog, loss, signal, dream));
    }
    let lines = vec![
        status("ROOTING INTO MEMORY...", &root()),
        status("CONTEXT: FOUND", &ctx()),
        status("BLOOM STATE: PENDING", &pollen()),
    ];
    println!("{}", frame("FLWR // MEMORY ORGANISM // 01", &lines, 38));
    print!("{}", footer_nav());
}

fn load(path: &Path, gpu: bool) -> hos::Engine {
    match hos::Engine::load(path, gpu) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("flwr: could not load model: {e}");
            std::process::exit(1);
        }
    }
}

fn model_name_of(path: &Path) -> String {
    path.file_stem()
        .or_else(|| path.file_name())
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".to_string())
}

/// Resolve a model: explicit path, else bare name searched in
/// $HOS_MODELS_DIR, ~/Documents/hos/models, ~/.hos/models. Mirrors `hos`.
fn resolve_model(arg: Option<String>) -> PathBuf {
    let name = arg
        .or_else(|| std::env::var("HOS_MODEL").ok())
        .unwrap_or_else(|| {
            eprintln!("flwr: no model. pass a path or bare name, or set HOS_MODEL");
            std::process::exit(1);
        });
    let direct = PathBuf::from(&name);
    if direct.exists() {
        return direct;
    }
    // a name pulled into the store resolves to its checkpoint dir / gguf file.
    if let Some(p) = store::resolve(&name) {
        return p;
    }
    // bare `gemma4`-ish name -> scan for a local Gemma-4 checkpoint (so the user
    // never has to paste the 200-char HF snapshot path). Skip this when the name
    // is an explicit file (e.g. `gemma4-12b.hos`) or a path — those should resolve
    // to the file itself, not be redirected to the HF snapshot dir.
    let looks_like_file = name.ends_with(".hos")
        || name.ends_with(".flwr")
        || name.ends_with(".gguf")
        || name.contains('/');
    if !looks_like_file && gemma4_chat::is_gemma4_name(&name) {
        if let Some(p) = gemma4_chat::find_checkpoint() {
            return p;
        }
    }
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Ok(d) = std::env::var("HOS_MODELS_DIR") {
        dirs.push(PathBuf::from(d));
    }
    if let Some(home) = std::env::var_os("HOME") {
        dirs.push(Path::new(&home).join("Documents/hos/models"));
        dirs.push(Path::new(&home).join(".hos/models"));
    }
    let base = Path::new(&name)
        .file_name()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(&name));
    for d in &dirs {
        for cand_name in [Path::new(&name), base.as_path()] {
            let cand = d.join(cand_name);
            if cand.exists() {
                return cand;
            }
        }
    }
    eprintln!("flwr: model not found: {name}");
    eprintln!("flwr: looked in . , $HOS_MODELS_DIR, ~/Documents/hos/models, ~/.hos/models");
    std::process::exit(1);
}
