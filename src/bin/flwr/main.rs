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
}

fn usage() -> ! {
    eprintln!("flwr — talk to a model.   Powered by HOS.");
    eprintln!();
    eprintln!("  flwr run <model> [--gpu] [-p \"prompt\"] [-n N] [--temp T] [--seed S]");
    eprintln!("  flwr serve <model> [--gpu] [--host 127.0.0.1] [--port 11434]");
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
    eprintln!("quickstart:  flwr run ~/Documents/hos/models/<model>.hos --gpu");
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
        gpu: false,
        host: "127.0.0.1".to_string(),
        port: 11434, // the de-facto local-LLM port — drop-in for existing clients
        revision: "main".to_string(),
        name: None,
        dest: None,
        qtype: "q8_0".to_string(),
        awq: false,
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
    let mut history: Vec<hos::chat::Message> = Vec::new();
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
        match text {
            "" => continue,
            "/bye" | "/exit" | "/quit" => break,
            "/reset" => {
                history.clear();
                println!("    · memory cleared.\n");
                continue;
            }
            _ => {}
        }
        history.push(hos::chat::Message::new("user", text));
        let reply = turn(&mut eng, &history);
        history.push(hos::chat::Message::new("assistant", reply.trim()));
    }
    println!("    · session closed.");
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
    let looks_like_file =
        name.ends_with(".hos") || name.ends_with(".flwr") || name.ends_with(".gguf") || name.contains('/');
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
