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

/// A fresh per-run seed (zero-dep): scramble the wall-clock nanos + pid with
/// splitmix64 so even close-in-time launches diverge. Used when `--seed` is omitted
/// so chat feels fresh each run; pass `--seed`/`/param seed` to make it reproducible.
fn random_seed() -> u64 {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E37_79B9_7F4A_7C15);
    let mut x = t ^ (std::process::id() as u64).rotate_left(32) ^ 0x9E37_79B9_7F4A_7C15;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    let s = x ^ (x >> 31);
    if s == 0 {
        0x1234_5678_9ABC_DEF1
    } else {
        s
    }
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
    let mut seed_given = false;
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
            "--seed" => {
                o.seed = it.next().and_then(|v| v.parse().ok()).unwrap_or(o.seed);
                seed_given = true;
            }
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
    // No explicit --seed: pick a fresh one per run so chat isn't deterministic.
    // (It's shown at startup and saved with each chat, so a good reply is reproducible.)
    if !seed_given {
        o.seed = random_seed();
    }
    o
}

fn main() {
    let o = parse();
    match o.cmd.as_str() {
        "__cmphf" => cmd_cmphf(&o),
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
        "ingest" => {
            let (Some(src), Some(dst)) = (o.model.clone(), o.dest.clone()) else {
                eprintln!("flwr ingest: need an HF checkpoint dir and a destination name");
                usage();
            };
            store::ingest_hf(&src, &dst, &o.qtype);
        }
        other => {
            eprintln!("flwr: unknown command '{other}'");
            usage();
        }
    }
}

/// `__cmphf <gguf> <hf-dir>`: diagnostic — compare the qwen35 GGUF's tensors
/// (ground truth) against the HF tensors under our ingest mapping/conversions.
/// Cosine ~1.0 = the mapping is right (q4_k vs bf16 differ only by quant error);
/// cosine ~0 = wrong tensor/conversion. Pinpoints which conversion to fix.
fn cmd_cmphf(o: &Opts) {
    use hos::model::ModelSource;
    let gguf = o.model.clone().expect("need gguf");
    let hf = o.dest.clone().expect("need hf dir");
    let g = hos::gguf::Gguf::open(std::path::Path::new(&gguf)).expect("open gguf");
    let st = hos::safetensors::SafeTensors::open_dir(std::path::Path::new(&hf)).expect("open hf");
    let cos = |a: &[f32], b: &[f32]| -> (f64, f64) {
        let n = a.len().min(b.len());
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for i in 0..n {
            dot += a[i] as f64 * b[i] as f64;
            na += (a[i] as f64).powi(2);
            nb += (b[i] as f64).powi(2);
        }
        (dot / (na.sqrt() * nb.sqrt() + 1e-12), (na / n as f64).sqrt())
    };
    let mut cmp = |label: &str, gname: &str, hname: &str, conv: fn(f32) -> f32| {
        let a = match g.dequant(gname) {
            Ok(v) => v,
            Err(e) => {
                println!("  {label:22} GGUF miss {gname}: {e}");
                return;
            }
        };
        let b: Vec<f32> = match st.to_f32(hname) {
            Ok(v) => v.into_iter().map(conv).collect(),
            Err(e) => {
                println!("  {label:22} HF miss {hname}: {e}");
                return;
            }
        };
        let (c, rms) = cos(&a, &b);
        println!(
            "  {label:22} cos={c:+.4}  (gguf n={} hf n={} rms={rms:.4}) {}",
            a.len(),
            b.len(),
            if c > 0.9 { "OK" } else { "?? MISMATCH" }
        );
    };
    let id = |x: f32| x;
    let negexp = |x: f32| -(x.exp());
    let neg = |x: f32| -x;
    let l0 = "model.language_model.layers.0.linear_attn";
    let l3 = "model.language_model.layers.3.self_attn";
    println!("== embeddings / head ==");
    cmp("token_embd", "token_embd.weight", "model.language_model.embed_tokens.weight", id);
    cmp("output(lm_head)", "output.weight", "lm_head.weight", id);
    cmp("output_norm", "output_norm.weight", "model.language_model.norm.weight", id);
    println!("== full-attn layer 3 (q/k permute test) ==");
    cmp("attn_q l3", "blk.3.attn_q.weight", &format!("{l3}.q_proj.weight"), id);
    cmp("attn_k l3", "blk.3.attn_k.weight", &format!("{l3}.k_proj.weight"), id);
    cmp("attn_v l3", "blk.3.attn_v.weight", &format!("{l3}.v_proj.weight"), id);
    cmp("attn_out l3", "blk.3.attn_output.weight", &format!("{l3}.o_proj.weight"), id);
    println!("== ssm layer 0 (a/b order + A_log conversion tests) ==");
    cmp("attn_qkv l0", "blk.0.attn_qkv.weight", &format!("{l0}.in_proj_qkv.weight"), id);
    cmp("attn_gate l0", "blk.0.attn_gate.weight", &format!("{l0}.in_proj_z.weight"), id);
    cmp("ssm_out l0", "blk.0.ssm_out.weight", &format!("{l0}.out_proj.weight"), id);
    cmp("ssm_alpha=a", "blk.0.ssm_alpha.weight", &format!("{l0}.in_proj_a.weight"), id);
    cmp("ssm_alpha=b?", "blk.0.ssm_alpha.weight", &format!("{l0}.in_proj_b.weight"), id);
    cmp("ssm_beta=b", "blk.0.ssm_beta.weight", &format!("{l0}.in_proj_b.weight"), id);
    cmp("ssm_conv1d", "blk.0.ssm_conv1d.weight", &format!("{l0}.conv1d.weight"), id);
    cmp("ssm_dt", "blk.0.ssm_dt.bias", &format!("{l0}.dt_bias"), id);
    cmp("ssm_norm", "blk.0.ssm_norm.weight", &format!("{l0}.norm.weight"), id);
    println!("== A_log -> ssm_a: try id / -x / -exp(x) ==");
    cmp("ssm_a = A_log", "blk.0.ssm_a", &format!("{l0}.A_log"), id);
    cmp("ssm_a = -A_log", "blk.0.ssm_a", &format!("{l0}.A_log"), neg);
    cmp("ssm_a = -exp(A_log)", "blk.0.ssm_a", &format!("{l0}.A_log"), negexp);
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
    // ChatSession backend; everything else stays on the generic Engine. Route both
    // a qwen35 GGUF and a minted qwen35 `.hos` capsule (from HF ingest) to it.
    if is_qwen35_capsule(&path) {
        cmd_run_qwen35(&path, o);
        return;
    }
    if let Ok(g) = hos::gguf::Gguf::open(&path) {
        if hos::model::Arch::detect(&g) == hos::model::Arch::Qwen35Hybrid {
            cmd_run_qwen35(&path, o);
            return;
        }
    }
    let mut name = model_name_of(&path);
    let mut eng = load(&path, o.gpu);
    let backend = if o.gpu { "metal gpu" } else { "cpu" };
    // Live-editable sampling knobs (via /param), seeded from the CLI flags.
    let mut smp = Sampling {
        temp: o.temp,
        top_k: o.top_k,
        top_p: o.top_p,
        rep_penalty: o.rep_penalty,
        repeat_last_n: o.repeat_last_n,
        n_predict: o.n_predict,
        seed: o.seed,
    };

    let turn = |eng: &mut hos::Engine, history: &[hos::chat::Message], s: &Sampling| -> String {
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
            s.n_predict,
            s.temp,
            s.top_k,
            s.top_p,
            s.rep_penalty,
            s.repeat_last_n,
            s.seed,
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
        turn(&mut eng, &history, &smp);
        return;
    }

    organism_banner(&eng, &name, backend);
    println!("{}", cmd_hint(true, smp.seed));
    let mut history: Vec<hos::chat::Message> = Vec::new();
    let mut last_memory_sig = String::new();
    let mut chat_id = crate::chats::new_id();
    let mut model_meta = serde_json::json!({ "name": name.clone() });
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
            let list = crate::chats::list();
            let items = list["data"].as_array().cloned().unwrap_or_default();
            if items.is_empty() {
                println!("    · no saved conversations yet.\n");
                continue;
            }
            let labels: Vec<String> = items
                .iter()
                .map(|it| {
                    let title = it["title"].as_str().unwrap_or("untitled");
                    let n = it["messages"].as_u64().unwrap_or(0);
                    format!("{title}   ({n} msgs)")
                })
                .collect();
            match select_menu("open conversation", &labels, "↑/↓ move · enter open · q cancel") {
                Some(i) => {
                    if let Some(id) = items[i]["id"].as_str() {
                        if let Some(msgs) = load_chat_history(id) {
                            history = msgs;
                            chat_id = id.to_string();
                            replay_history(&history);
                        } else {
                            println!("    · could not open that conversation.\n");
                        }
                    }
                }
                None => print_chat_list(), // not a TTY (piped) or cancelled -> plain list
            }
            continue;
        }
        if matches!(text, "/models" | "/model") {
            let models = switchable_models();
            if models.is_empty() {
                println!("    · no models found in the model directories.\n");
                continue;
            }
            let labels: Vec<String> = models
                .iter()
                .map(|m| if m == &name { format!("{m}   (current)") } else { m.clone() })
                .collect();
            let Some(i) = select_menu("switch model", &labels, "↑/↓ move · enter load · q cancel")
            else {
                continue;
            };
            let pick = models[i].clone();
            if pick == name {
                println!("    · already on {pick}.\n");
                continue;
            }
            let Some(newpath) = resolve_model_opt(&pick) else {
                println!("    · could not find '{pick}'.\n");
                continue;
            };
            // Gemma-4 and qwen35 hybrid run on their own REPL backends; a live
            // in-place swap would need the Engine to become them. Instead hand off
            // to a fresh REPL with the right backend by re-exec'ing flwr — a clean
            // switch to any family. Reset the terminal first: exec replaces this
            // process, so the Meter's Drop won't run.
            let is_gemma = gemma4_chat::is_gemma4(&newpath) || gemma4_chat::is_gemma4_capsule(&newpath);
            let is_qwen35 = !is_gemma
                && hos::gguf::Gguf::open(&newpath)
                    .map(|g| hos::model::Arch::detect(&g) == hos::model::Arch::Qwen35Hybrid)
                    .unwrap_or(false);
            if is_gemma || is_qwen35 {
                print!("\x1b[r\x1b[2J\x1b[H\x1b[?25h");
                std::io::stdout().flush().ok();
                let exe =
                    std::env::current_exe().unwrap_or_else(|_| std::path::PathBuf::from("flwr"));
                let mut cmd = std::process::Command::new(exe);
                cmd.arg("run").arg(&pick);
                if !o.gpu {
                    cmd.arg("--cpu");
                }
                #[cfg(unix)]
                {
                    use std::os::unix::process::CommandExt;
                    let err = cmd.exec(); // replaces this process on success
                    eprintln!("    · could not switch to {pick}: {err}");
                }
                #[cfg(not(unix))]
                {
                    let _ = cmd.status();
                    std::process::exit(0);
                }
                continue;
            }
            print!("    · loading {}{pick}{} …", hos::viz::petal(), hos::viz::RESET);
            std::io::stdout().flush().ok();
            match hos::Engine::load(&newpath, o.gpu) {
                Ok(e) => {
                    eng = e;
                    name = model_name_of(&newpath);
                    model_meta = serde_json::json!({ "name": name.clone() });
                    history.clear();
                    chat_id = crate::chats::new_id();
                    println!(
                        "\r    · now running {}{}{}. fresh conversation.        \n",
                        hos::viz::petal(),
                        name,
                        hos::viz::RESET
                    );
                }
                Err(e) => println!("\r    · could not load {pick}: {e}          \n"),
            }
            continue;
        }
        if let Some(rest) = text.strip_prefix("/open") {
            let id = rest.trim();
            if id.is_empty() {
                println!("    · usage: /open <id>   (or just /list to pick)\n");
            } else if let Some(msgs) = load_chat_history(id) {
                history = msgs;
                chat_id = id.to_string();
                replay_history(&history);
            } else {
                println!("    · no chat '{id}'. try /list\n");
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
            if handle_theme(text) || handle_context(text) || handle_param(text, &mut smp) {
                continue;
            }
            print_help(true);
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
        let reply = turn(&mut eng, &bundle.messages, &smp);
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
/// True if `path` is a `.hos` capsule whose card records `arch=qwen35` — so a
/// minted (HF-ingested or GGUF-derived) qwen35 capsule routes to the hybrid loader.
fn is_qwen35_capsule(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    hos::format::read_card(path)
        .ok()
        .and_then(|c| {
            c.arch
                .get("architecture")
                .and_then(|v| v.as_str())
                .map(|s| s == "qwen35")
        })
        .unwrap_or(false)
}

fn cmd_run_qwen35(path: &Path, o: &Opts) {
    use std::io::Write;
    let name = model_name_of(path);
    let backend = if o.gpu { "metal gpu" } else { "cpu" };
    // A minted `.hos` capsule loads through HosSource (a generic ModelSource); a
    // GGUF loads through Gguf. ChatSession::load is generic over both.
    let mut sess = if is_qwen35_capsule(path) {
        match hos::hos_capsule::HosSource::open(path) {
            Ok((src, tok)) => match hos::qwen35::ChatSession::load(&src, tok, o.gpu) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("[flwr] load: {e}");
                    return;
                }
            },
            Err(e) => {
                eprintln!("[flwr] cannot open capsule: {e}");
                return;
            }
        }
    } else {
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
        match hos::qwen35::ChatSession::load(&g, tok, o.gpu) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[flwr] load: {e}");
                return;
            }
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
    let mut smp = Sampling {
        temp: o.temp,
        top_k: o.top_k,
        top_p: o.top_p,
        rep_penalty: o.rep_penalty,
        repeat_last_n: o.repeat_last_n,
        n_predict: o.n_predict,
        seed: o.seed,
    };
    let turn = |sess: &mut hos::qwen35::ChatSession,
                history: &[hos::chat::Message],
                s: &Sampling|
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
            s.n_predict,
            s.temp,
            s.top_k,
            s.top_p,
            s.rep_penalty,
            s.repeat_last_n,
            s.seed,
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
        let (answer, reasoning, n) = turn(&mut sess, &history, &smp);
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
    println!("{}", cmd_hint(true, smp.seed));
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
            crate::pick_and_open(&mut history, &mut chat_id);
            continue;
        }
        if matches!(text, "/models" | "/model") {
            pick_model_reexec(o.gpu, &name);
            continue;
        }
        if let Some(rest) = text.strip_prefix("/open") {
            let id = rest.trim();
            if id.is_empty() {
                println!("    · usage: /open <id>   (or just /list to pick)\n");
            } else if let Some(msgs) = load_chat_history(id) {
                history = msgs;
                chat_id = id.to_string();
                replay_history(&history);
            } else {
                println!("    · no chat '{id}'. try /list\n");
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
            if handle_theme(text) || handle_context(text) || handle_param(text, &mut smp) {
                continue;
            }
            print_help(true);
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
        let (reply, reasoning, n) = turn(&mut sess, &bundle.messages, &smp);
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

/// Run `stty ARGS` reading the REAL controlling terminal (inherited stdin) and
/// capture stdout. `Command::output()` nulls stdin — which makes `stty -g`/`stty
/// size` fail with "not a terminal" — so we must inherit stdin explicitly. Returns
/// the trimmed output, or None when there's no TTY (so callers degrade gracefully).
fn tty_stty(args: &[&str]) -> Option<String> {
    use std::process::{Command, Stdio};
    let out = Command::new("stty")
        .args(args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?
        .wait_with_output()
        .ok()?;
    if out.status.success() {
        String::from_utf8(out.stdout).ok().map(|s| s.trim().to_string())
    } else {
        None
    }
}

/// Interactive arrow-key menu, dependency-free. Puts the terminal in raw mode via
/// `stty` (restored on drop — even on panic — and the cursor re-shown), draws the
/// list with the selected row highlighted, and reads keys: ↑/↓ (or k/j) move, Enter
/// selects, q/Esc/Ctrl-C cancel. Returns the chosen index, or None on cancel / when
/// stdin isn't a TTY (so callers can fall back to a plain print). Scrolls when the
/// list is taller than the window.
fn select_menu(title: &str, items: &[String], footer: &str) -> Option<usize> {
    use hos::viz::*;
    use std::io::{Read, Write};
    if items.is_empty() {
        return None;
    }
    // Capture current termios; a failure means we're not on a TTY -> bail.
    let saved = match tty_stty(&["-g"]) {
        Some(s) => s,
        None => return None,
    };
    struct Guard(String);
    impl Drop for Guard {
        fn drop(&mut self) {
            let _ = std::process::Command::new("stty").arg(&self.0).status();
            print!("\x1b[?25h"); // show cursor
            let _ = std::io::stdout().flush();
        }
    }
    let _guard = Guard(saved);
    let _ = std::process::Command::new("stty").args(["raw", "-echo"]).status();
    print!("\x1b[?25l"); // hide cursor

    let win = items.len().min(12);
    let mut sel = 0usize;
    let mut top = 0usize;
    let paint = |sel: usize, top: usize| -> String {
        let mut s = String::new();
        s.push_str(&format!("  {}{}{title}{}\r\n", BOLD, petal(), RESET));
        for i in top..(top + win).min(items.len()) {
            if i == sel {
                s.push_str(&format!("  {}▸ {}{}\r\n", pollen(), items[i], RESET));
            } else {
                s.push_str(&format!("    {}{}{}\r\n", ink(), items[i], RESET));
            }
        }
        s.push_str(&format!("  {}{footer}{}\r\n", faint(), RESET));
        s
    };
    let block = win + 2; // title + rows + footer
    print!("\r\n{}", paint(sel, top));
    std::io::stdout().flush().ok();

    let mut stdin = std::io::stdin();
    let mut b = [0u8; 1];
    let chosen = loop {
        if stdin.read(&mut b).unwrap_or(0) == 0 {
            break None;
        }
        match b[0] {
            b'\r' | b'\n' => break Some(sel),
            b'q' | 3 => break None, // q or Ctrl-C
            27 => {
                // ESC: an arrow sequence (ESC [ A/B), or a lone Esc (cancel). Read
                // one byte at a time — `read` may return fewer than requested.
                let mut c = [0u8; 1];
                if stdin.read(&mut c).unwrap_or(0) == 1 && c[0] == b'[' {
                    if stdin.read(&mut c).unwrap_or(0) == 1 {
                        match c[0] {
                            b'A' if sel > 0 => sel -= 1,
                            b'B' if sel + 1 < items.len() => sel += 1,
                            _ => {}
                        }
                    }
                } else {
                    break None;
                }
            }
            b'k' if sel > 0 => sel -= 1,
            b'j' if sel + 1 < items.len() => sel += 1,
            _ => {}
        }
        if sel < top {
            top = sel;
        }
        if sel >= top + win {
            top = sel + 1 - win;
        }
        print!("\x1b[{block}A\x1b[0J{}", paint(sel, top));
        std::io::stdout().flush().ok();
    };
    // Erase the whole menu block on exit so it leaves no residue (one blank line of
    // padding above remains). Cursor ends where the menu started.
    print!("\x1b[{block}A\r\x1b[0J");
    std::io::stdout().flush().ok();
    chosen
}

/// The model dirs `flwr`/`hos` search, in priority order.
fn model_search_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(d) = std::env::var("HOS_MODELS_DIR") {
        dirs.push(PathBuf::from(d));
    }
    if let Some(h) = std::env::var_os("HOME") {
        dirs.push(Path::new(&h).join("Documents/hos/models"));
        dirs.push(Path::new(&h).join(".hos/models"));
    }
    dirs
}

/// Names a `/models` pick can switch to: store entries + `.gguf`/`.hos`/`.flwr`
/// files and HF checkpoint dirs in the model directories, de-duped and sorted.
fn switchable_models() -> Vec<String> {
    use std::collections::BTreeSet;
    let mut names: BTreeSet<String> = BTreeSet::new();
    for m in store::all() {
        names.insert(m.name);
    }
    for d in model_search_dirs() {
        if let Ok(rd) = std::fs::read_dir(d) {
            for e in rd.flatten() {
                let p = e.path();
                let nm = e.file_name().to_string_lossy().into_owned();
                let ext = p.extension().and_then(|x| x.to_str());
                let is_model =
                    ext == Some("gguf") || ext == Some("hos") || ext == Some("flwr");
                let is_hf = p.is_dir() && p.join("config.json").exists();
                if is_model || is_hf {
                    names.insert(nm);
                }
            }
        }
    }
    names.into_iter().collect()
}

/// Resolve a model name to a path WITHOUT exiting the process (unlike
/// `resolve_model`) — for the live `/models` switch, where a bad pick should be a
/// message, not a crash.
fn resolve_model_opt(name: &str) -> Option<PathBuf> {
    let direct = PathBuf::from(name);
    if direct.exists() {
        return Some(direct);
    }
    if let Some(p) = store::resolve(name) {
        return Some(p);
    }
    for d in model_search_dirs() {
        let c = d.join(name);
        if c.exists() {
            return Some(c);
        }
    }
    None
}

/// One-line status bar, printed under the banner and after each reply: a live
/// activity readout (model · speed · memory · turns) plus the essential command
/// shortcuts, so both the meter and the commands stay visible as the conversation
/// scrolls — no scroll-region tricks, just a line that flows with the content.
/// Shared by every REPL (llama / gemma4 / qwen35) so the look is consistent.
/// The compact command line shown once under the banner (kept short; `/help` lists
/// everything). `full` is the llama/qwen35 set; the gemma REPL is more minimal.
pub fn cmd_hint(full: bool, seed: u64) -> String {
    use hos::viz::*;
    let cmds = if full {
        "/models · /list · /role · /new · /param · /context · /theme · /help · /bye"
    } else {
        "/models · /reset · /param · /context · /theme · /help · /bye"
    };
    // seed is shown so a good reply can be reproduced (--seed / /param seed to fix it)
    format!("    {}{cmds}     seed {seed}{}", faint(), RESET)
}

/// `/help` — list every command (including the ones not in the compact hint).
pub fn print_help(full: bool) {
    use hos::viz::*;
    let row = |c: &str, d: &str| println!("    {}{:<14}{}{}", petal(), c, RESET, d);
    println!("\n  {}commands{}", BOLD, RESET);
    row("/models", "switch the resident model");
    if full {
        row("/list", "open a saved conversation (arrow-pick)");
        row("/open <id>", "open a conversation by its id");
        row("/role <text>", "set a system instruction (persists this chat)");
        row("/new", "start a fresh conversation");
    } else {
        row("/reset", "clear the conversation");
    }
    row("/param", "view or set  temp / max / top_k / top_p / seed / penalty");
    row("/context", "history/compression budget  (/context 3072 · recent · batch)");
    row("/theme", "palette:  light | dark | auto  (for white terminals)");
    row("/help", "this list");
    row("/bye", "quit");
    println!();
}

/// Live-editable sampling knobs (via `/param`), seeded from the CLI flags.
#[derive(Clone)]
pub struct Sampling {
    pub temp: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub rep_penalty: f32,
    pub repeat_last_n: usize,
    pub n_predict: usize,
    pub seed: u64,
}
impl Sampling {
    pub fn show(&self) {
        use hos::viz::*;
        let r = |k: &str, v: String| println!("    {}{:<9}{}{}", faint(), k, RESET, v);
        println!("\n  {}params{}", BOLD, RESET);
        r("temp", self.temp.to_string());
        r("top_k", self.top_k.to_string());
        r("top_p", self.top_p.to_string());
        r("max", format!("{}   (max new tokens per reply)", self.n_predict));
        r("seed", self.seed.to_string());
        r("penalty", self.rep_penalty.to_string());
        println!("    {}· set with:  /param <key> <value>{}\n", faint(), RESET);
    }
    /// Apply `/param <key> <value>`; returns a human-readable result message.
    pub fn set(&mut self, key: &str, val: &str) -> String {
        macro_rules! num {
            ($f:expr, $label:expr) => {
                match val.parse() {
                    Ok(v) => {
                        $f = v;
                        format!("{} = {v}", $label)
                    }
                    Err(_) => format!("'{val}' is not a valid {}", $label),
                }
            };
        }
        match key {
            "temp" => num!(self.temp, "temp"),
            "top_k" => num!(self.top_k, "top_k"),
            "top_p" => num!(self.top_p, "top_p"),
            "max" | "n_predict" | "tokens" => num!(self.n_predict, "max"),
            "seed" => num!(self.seed, "seed"),
            "penalty" | "rep_penalty" => num!(self.rep_penalty, "penalty"),
            _ => format!("unknown param '{key}'  (temp / top_k / top_p / max / seed / penalty)"),
        }
    }
}

/// Handle a `/param [key value]` command line. Returns true if it was a /param.
pub fn handle_param(text: &str, smp: &mut Sampling) -> bool {
    let Some(rest) = text.strip_prefix("/param") else {
        return false;
    };
    let args: Vec<&str> = rest.split_whitespace().collect();
    match args.as_slice() {
        [] => smp.show(),
        [k, v] => println!("    · {}\n", smp.set(k, v)),
        _ => println!("    · usage:  /param            show all\n    ·         /param temp 0.8   set one\n"),
    }
    true
}

/// Handle `/theme [light|dark|auto]`. The palette normally auto-detects the
/// terminal background, but terminals that don't answer the background query get
/// stuck on the dark palette — this forces it. Returns true if it was a /theme.
pub fn handle_theme(text: &str) -> bool {
    use hos::viz::*;
    let Some(rest) = text.strip_prefix("/theme") else {
        return false;
    };
    let arg = rest.trim();
    if arg.is_empty() {
        println!("\n  {}theme{}", BOLD, RESET);
        println!("    {}palette{}  {}", faint(), RESET, theme_name());
        println!(
            "    {}· set with:  /theme light | dark | auto{}\n",
            faint(),
            RESET
        );
    } else {
        let set = set_theme(arg);
        // Show a swatch in the freshly-applied palette so the change is visible.
        println!(
            "\n    {}theme = {set}{}   {}flwr{} {}petal{} {}sage{} {}ink text{}\n",
            petal(),
            RESET,
            petal(),
            RESET,
            deep(),
            RESET,
            stem(),
            RESET,
            ink(),
            RESET
        );
    }
    true
}

/// Handle `/context [key value]` (alias `/ctx`) — the conversation-compression
/// knobs. Older turns fold into bullet receipts; the recent window stays verbatim;
/// the whole prompt is capped at `tokens`. Values live in the env the memory module
/// reads each turn, so edits take effect immediately. Returns true if it was /context.
pub fn handle_context(text: &str) -> bool {
    use hos::viz::*;
    let Some(rest) = text
        .strip_prefix("/context")
        .or_else(|| text.strip_prefix("/ctx"))
    else {
        return false;
    };
    let show = || {
        let r = |k: &str, v: String, note: &str| {
            println!("    {}{:<8}{}{:<6} {}{}{}", faint(), k, RESET, v, faint(), note, RESET)
        };
        println!("\n  {}context compression{}", BOLD, RESET);
        r("tokens", crate::memory::context_budget_tokens().to_string(), "total prompt budget");
        r("recent", crate::memory::recent_turns().to_string(), "recent turns kept verbatim");
        r("batch", crate::memory::batch_messages().to_string(), "older messages per receipt");
        println!(
            "    {}· set the budget:  /context 3072    ·   /context recent 4  ·  /context batch 8{}\n",
            faint(),
            RESET
        );
    };
    // Set one knob into the env the memory module reads each turn.
    let set = |env: &str, key: &str, v: &str| match v.parse::<usize>() {
        Ok(n) if n > 0 => {
            std::env::set_var(env, v);
            println!("    · {key} = {n}\n");
        }
        _ => println!("    · '{v}' must be a positive whole number\n"),
    };
    let args: Vec<&str> = rest.split_whitespace().collect();
    match args.as_slice() {
        [] => show(),
        // Simple form: a bare number sets the token budget (the common case).
        [only] => {
            if only.chars().all(|c| c.is_ascii_digit()) {
                set("FLWR_CONTEXT_TOKENS", "tokens", only);
            } else {
                println!(
                    "    · usage:  /context 3072          set the token budget\n    ·         /context recent 4     ·   /context batch 8\n"
                );
            }
        }
        [k, v] => {
            let (env, key) = match *k {
                "tokens" | "ctx" | "context" | "budget" => ("FLWR_CONTEXT_TOKENS", "tokens"),
                "recent" | "turns" => ("FLWR_RECENT_TURNS", "recent"),
                "batch" | "messages" | "msgs" => ("FLWR_MEMORY_BATCH_MESSAGES", "batch"),
                _ => {
                    println!("    · unknown key '{k}'  (tokens / recent / batch)\n");
                    return true;
                }
            };
            set(env, key, v);
        }
        _ => println!(
            "    · usage:  /context 3072          set the token budget\n    ·         /context recent 4     ·   /context batch 8\n"
        ),
    }
    true
}

/// Resident-set size of this process in MB (via `ps`), for the activity monitor.
fn current_rss_mb() -> u64 {
    let pid = std::process::id().to_string();
    std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|kb| kb / 1024)
        .unwrap_or(0)
}

/// Shared `/list` picker: choose a saved conversation with the arrow menu and load
/// it into `history`/`chat_id`. Falls back to a plain printed list off a TTY. Used
/// by every REPL so conversation-opening looks the same everywhere.
pub fn pick_and_open(history: &mut Vec<hos::chat::Message>, chat_id: &mut String) {
    let list = crate::chats::list();
    let items = list["data"].as_array().cloned().unwrap_or_default();
    if items.is_empty() {
        println!("    · no saved conversations yet.\n");
        return;
    }
    let labels: Vec<String> = items
        .iter()
        .map(|it| {
            let title = it["title"].as_str().unwrap_or("untitled");
            let n = it["messages"].as_u64().unwrap_or(0);
            format!("{title}   ({n} msgs)")
        })
        .collect();
    match select_menu("open conversation", &labels, "↑/↓ move · enter open · q cancel") {
        Some(i) => {
            if let Some(id) = items[i]["id"].as_str() {
                if let Some(msgs) = load_chat_history(id) {
                    *history = msgs;
                    *chat_id = id.to_string();
                    replay_history(history);
                } else {
                    println!("    · could not open that conversation.\n");
                }
            }
        }
        None => print_chat_list(),
    }
}

/// Shared `/models` picker for the gemma4 / qwen35 REPLs: choose a model and hand
/// off to a fresh REPL of the right family by re-exec'ing flwr (those backends can't
/// swap the resident engine in place; the generic Engine REPL does an in-place swap
/// for same-family models instead).
pub fn pick_model_reexec(gpu: bool, current: &str) {
    let models = switchable_models();
    if models.is_empty() {
        println!("    · no models found in the model directories.\n");
        return;
    }
    let labels: Vec<String> = models
        .iter()
        .map(|m| if m == current { format!("{m}   (current)") } else { m.clone() })
        .collect();
    let Some(i) = select_menu("switch model", &labels, "↑/↓ move · enter load · q cancel") else {
        return;
    };
    let pick = models[i].clone();
    if pick == current {
        println!("    · already on {pick}.\n");
        return;
    }
    if resolve_model_opt(&pick).is_none() {
        println!("    · could not find '{pick}'.\n");
        return;
    }
    print!("\x1b[?25h\x1b[2J\x1b[H");
    std::io::stdout().flush().ok();
    let exe = std::env::current_exe().unwrap_or_else(|_| std::path::PathBuf::from("flwr"));
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("run").arg(&pick);
    if !gpu {
        cmd.arg("--cpu");
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let err = cmd.exec(); // replaces this process on success
        eprintln!("    · could not switch to {pick}: {err}");
    }
    #[cfg(not(unix))]
    {
        let _ = cmd.status();
        std::process::exit(0);
    }
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
