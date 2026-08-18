//! flwr's Gemma-4 front-end — the interactive/one-shot chat UI only.
//!
//! The model capability (loading, tokenizer, chat template, generation, and
//! checkpoint discovery) lives in the HOS **library** at `hos::gemma4_chat`, so
//! `flwr` and `flwr_agent` share one implementation. This file keeps only the
//! terminal UI: the REPL loop, the framed "organism" banner, and flwr's `Opts`.

use std::io::Write;
use std::path::Path;

// The canonical Gemma-4 chat capability now lives in the hos library; re-export
// the pieces `flwr`'s `main.rs` reaches for so call sites stay `gemma4_chat::*`.
pub use hos::gemma4_chat::{
    find_checkpoint, is_gemma4, is_gemma4_capsule, is_gemma4_name, GenParams, Session,
};

/// CLI knobs for the interactive `run` entry.
pub struct Opts {
    pub gpu: bool,
    pub n_predict: usize,
    pub temp: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub rep_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
    pub prompt: Option<String>,
}

/// `flwr run <gemma4>` — interactive (or one-shot `-p`) chat loop.
pub fn run(path: &Path, o: &Opts) {
    eprintln!("[flwr] loading gemma4 decoder from {} ...", path.display());
    let sess = match Session::load(path, o.gpu) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("flwr: gemma4 load failed: {e}");
            std::process::exit(1);
        }
    };
    let mut smp = crate::Sampling {
        temp: o.temp,
        top_k: o.top_k,
        top_p: o.top_p,
        rep_penalty: o.rep_penalty,
        repeat_last_n: o.repeat_last_n,
        n_predict: o.n_predict,
        seed: o.seed,
    };
    let mut rng: u64 = if o.seed == 0 {
        0xDEAD_BEEF_CAFE_F00D
    } else {
        o.seed
    };

    let say = |sess: &Session,
               hist: &[(String, String)],
               rng: &mut u64,
               s: &crate::Sampling|
     -> (String, usize) {
        let gp = GenParams {
            n_predict: s.n_predict,
            temp: s.temp,
            top_k: s.top_k,
            top_p: s.top_p,
            rep_penalty: s.rep_penalty,
            repeat_last_n: s.repeat_last_n,
            seed: s.seed,
        };
        print!(
            "  {}{}flwr{}  ",
            hos::viz::BOLD,
            hos::viz::petal(),
            hos::viz::RESET
        );
        std::io::stdout().flush().ok();
        let mut reply = String::new();
        let mut ntok = 0usize;
        sess.generate(hist, &gp, rng, |piece| {
            print!("{piece}");
            std::io::stdout().flush().ok();
            reply.push_str(piece);
            ntok += 1;
        });
        println!("\n");
        (reply, ntok)
    };

    // One-shot mode.
    if let Some(p) = &o.prompt {
        print!("{}", hos::viz::banner());
        let history = vec![("user".to_string(), p.clone())];
        say(&sess, &history, &mut rng, &smp);
        return;
    }

    // Interactive REPL.
    banner(path, o.gpu);
    let name = friendly_name(path);
    println!("{}", crate::cmd_hint(false));
    let mut history: Vec<(String, String)> = Vec::new();
    let mut last_memory_sig = String::new();
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
            Ok(0) => break, // Ctrl-D
            Ok(_) => {}
            Err(_) => break,
        }
        let text = line.trim();
        if crate::handle_param(text, &mut smp) {
            continue;
        }
        match text {
            "" => continue,
            "/bye" | "/exit" | "/quit" => break,
            "/reset" | "/new" => {
                history.clear();
                println!("    · memory cleared.\n");
                continue;
            }
            "/models" | "/model" => {
                crate::pick_model_reexec(o.gpu, &name);
                continue;
            }
            "/help" | "/?" | "/commands" => {
                crate::print_help(false);
                continue;
            }
            t if t.starts_with('/') => {
                crate::print_help(false);
                continue;
            }
            _ => {}
        }
        history.push(("user".to_string(), text.to_string()));
        let (compact, memory, omitted) = crate::memory::compact_gemma_history(&history);
        let sig = crate::memory::receipt_signature(&memory);
        if omitted > 0 && (crate::memory::debug_compaction() || sig != last_memory_sig) {
            eprintln!("    · using memory receipts for {omitted} older messages");
        }
        last_memory_sig = sig;
        let (reply, _ntok) = say(&sess, &compact, &mut rng, &smp);
        history.push(("model".to_string(), reply.trim().to_string()));
    }
    println!("    · session closed.");
}

/// Same framed "organism" card the generic `flwr run` shows (FLWR // CHAT box +
/// footer nav), rendered for the gemma4 backend so the UI is identical.
fn banner(path: &Path, gpu: bool) {
    use hos::viz::*;
    let name = friendly_name(path);
    let backend = if gpu { "metal gpu · q4k" } else { "cpu" };
    let back_col = if gpu { signal() } else { faint() };
    print!("{}", banner());
    let lines = vec![
        format!(
            "{}body{}     {}{name}{}  {}(gemma4){}",
            BOLD,
            RESET,
            petal(),
            RESET,
            faint(),
            RESET
        ),
        format!(
            "{}anatomy{}  {}dim 3840 · 48 layers · 16h/8kv · softcap 30{}",
            BOLD,
            RESET,
            ink(),
            RESET
        ),
        format!("{}dialect{}  {}gemma{}", BOLD, RESET, ctx(), RESET),
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

/// A readable model label from a HF snapshot path: prefer a path component that
/// names the repo (e.g. `models--google--gemma-4-12B-it` -> `gemma-4-12B-it`),
/// else the final dir name.
fn friendly_name(path: &Path) -> String {
    for comp in path.components().rev() {
        let s = comp.as_os_str().to_string_lossy();
        if let Some(idx) = s.to_lowercase().find("gemma") {
            let tail = &s[idx..];
            return tail.trim_end_matches('/').to_string();
        }
    }
    path.file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("gemma4")
        .to_string()
}
