//! `flwr serve` — a tiny HTTP daemon that exposes the loaded HOS model over an
//! OpenAI-compatible API, so existing chat clients, IDE plugins, and scripts
//! talk to it without changes.
//!
//! The server is hand-rolled over `std::net` — HOS keeps no web framework as a
//! dependency, the same way it keeps no ML framework.
//!
//! **Concurrency.** Connections are accepted and parsed on their own threads, so
//! a slow client never blocks the accept loop or other clients, and `GET /` /
//! `GET /v1/models` answer instantly. *Generation*, though, is serialized: there
//! is one resident model with one KV cache, so chat jobs funnel over a channel
//! to the single thread that owns the engine and run one at a time. That is the
//! honest model for a single local body — parallel decode would need parallel
//! engines (parallel memory), which a local runner deliberately does not do.
//! Endpoints:
//!   GET  /                      health line
//!   GET  /v1/models             the one resident model
//!   POST /v1/chat/completions   chat (set `"stream": true` for SSE token stream)

use hos::chat::Message;
use hos::Engine;
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread;

/// The resident generation backend: the generic `hos::Engine` (Llama-family) or
/// the HOS-native Gemma-4 chat session. Both are single-threaded and stateful and
/// live only on the engine thread.
pub enum Backend {
    Engine(Engine),
    Gemma4 {
        sess: crate::gemma4_chat::Session,
        rng: u64,
    },
}

impl Backend {
    fn family_label(&self) -> &'static str {
        match self {
            Backend::Engine(e) => e.chat_family().label(),
            Backend::Gemma4 { .. } => "gemma",
        }
    }

    /// Generate a reply to `msgs`, streaming text to `on`. Same shape as
    /// `Engine::chat`; returns the emitted token count.
    #[allow(clippy::too_many_arguments)]
    fn chat(
        &mut self,
        msgs: &[Message],
        max_tokens: usize,
        temp: f32,
        top_k: usize,
        top_p: f32,
        rep_penalty: f32,
        repeat_last_n: usize,
        seed: u64,
        on: impl FnMut(&str),
    ) -> usize {
        match self {
            Backend::Engine(e) => e.chat(
                msgs,
                max_tokens,
                temp,
                top_k,
                top_p,
                rep_penalty,
                repeat_last_n,
                seed,
                on,
            ),
            Backend::Gemma4 { sess, rng } => {
                if seed != 0 {
                    *rng = seed;
                }
                let hist: Vec<(String, String)> = msgs
                    .iter()
                    .map(|m| (m.role.clone(), m.content.clone()))
                    .collect();
                let gp = crate::gemma4_chat::GenParams {
                    n_predict: max_tokens,
                    temp,
                    top_k,
                    top_p,
                    rep_penalty,
                    repeat_last_n,
                    seed,
                };
                sess.generate(&hist, &gp, rng, on)
            }
        }
    }

    /// Load a backend for `path`, picking the gemma4 session for gemma4 checkpoints.
    pub fn load(path: &Path, gpu: bool) -> Result<Backend, String> {
        if crate::gemma4_chat::is_gemma4(path) {
            crate::gemma4_chat::Session::load(path, gpu)
                .map(|sess| Backend::Gemma4 { sess, rng: 42 })
                .map_err(|e| e.to_string())
        } else {
            Engine::load(path, gpu)
                .map(Backend::Engine)
                .map_err(|e| e.to_string())
        }
    }
}

/// Sampling defaults mirror the CLI; requests may override per call.
struct Params {
    max_tokens: usize,
    temp: f32,
    top_k: usize,
    top_p: f32,
    rep_penalty: f32,
    repeat_last_n: usize,
    seed: u64,
}

impl Default for Params {
    fn default() -> Params {
        Params {
            max_tokens: 512,
            temp: 0.7,
            top_k: 40,
            top_p: 0.95,
            rep_penalty: 1.1,
            repeat_last_n: 64,
            seed: 42,
        }
    }
}

/// A parsed chat request waiting for the engine. Everything in it is `Send`, so
/// it crosses from a connection thread to the engine thread; the `Engine` never
/// does.
struct ChatJob {
    stream: TcpStream,
    msgs: Vec<Message>,
    params: Params,
    want_stream: bool,
}

/// Work for the single engine-owning thread. `Switch` swaps the resident model
/// at runtime (the model is loaded on this thread; the `Engine` never crosses a
/// thread boundary). Both variants are `Send`.
enum Job {
    Chat(ChatJob),
    Switch {
        name: String,
        reply: Sender<Result<String, String>>,
    },
}

/// Block serving requests until the process is killed. `gpu` is reused when
/// switching models at runtime.
pub fn serve(mut backend: Backend, model_name: &str, host: &str, port: u16, gpu: bool) {
    let addr = format!("{host}:{port}");
    let listener = match TcpListener::bind(&addr) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("  ✗ could not bind {addr}: {e}");
            std::process::exit(1);
        }
    };
    let family = backend.family_label();
    print_banner(model_name, family, &addr);

    // The currently-resident model name — shared so connection threads can read
    // it (for /v1/models and provenance) while the engine thread updates it.
    let model = Arc::new(Mutex::new(model_name.to_string()));
    let (tx, rx) = mpsc::channel::<Job>();
    let accept_model = model.clone();
    // Accept + parse off the engine thread: one reader thread per connection.
    thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(stream) = conn else { continue };
            let tx = tx.clone();
            let m = accept_model.clone();
            thread::spawn(move || {
                if let Err(e) = dispatch(stream, m, tx) {
                    eprintln!("  ! connection error: {e}");
                }
            });
        }
    });

    // The engine is single-threaded and stateful; generation (and model swaps)
    // run here, one job at a time, while connections arrive in parallel.
    for job in rx {
        match job {
            Job::Chat(c) => {
                let name = model.lock().map(|g| g.clone()).unwrap_or_default();
                if let Err(e) = run_chat(&mut backend, &name, c) {
                    eprintln!("  ! generation write error: {e}");
                }
            }
            Job::Switch { name, reply } => match resolve(&name) {
                Some(path) => match Backend::load(&path, gpu) {
                    Ok(nb) => {
                        backend = nb;
                        if let Ok(mut g) = model.lock() {
                            *g = name.clone();
                        }
                        eprintln!("  · switched model -> {name}");
                        let _ = reply.send(Ok(name));
                    }
                    Err(e) => {
                        let _ = reply.send(Err(e));
                    }
                },
                None => {
                    let _ = reply.send(Err(format!("model not found: {name}")));
                }
            },
        }
    }
}

/// Resolve a model name to a path the engine can load: an explicit path, a store
/// entry, or a bare name in the usual model dirs. Non-fatal (returns None).
fn resolve(name: &str) -> Option<PathBuf> {
    let p = PathBuf::from(name);
    if p.exists() {
        return Some(p);
    }
    if let Some(p) = crate::store::resolve(name) {
        return Some(p);
    }
    for d in model_dirs() {
        let c = d.join(name);
        if c.exists() {
            return Some(c);
        }
    }
    if crate::gemma4_chat::is_gemma4_name(name) {
        if let Some(p) = crate::gemma4_chat::find_checkpoint() {
            return Some(p);
        }
    }
    None
}

fn model_dirs() -> Vec<PathBuf> {
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

/// Names a client can switch to: store entries + `.gguf` files and HF checkpoint
/// dirs found in the model directories.
fn available_models() -> Value {
    let mut names: BTreeSet<String> = BTreeSet::new();
    for m in crate::store::all() {
        names.insert(m.name);
    }
    for d in model_dirs() {
        if let Ok(rd) = std::fs::read_dir(d) {
            for e in rd.flatten() {
                let p = e.path();
                let nm = e.file_name().to_string_lossy().into_owned();
                let ext = p.extension().and_then(|x| x.to_str());
                let is_model_file = ext == Some("gguf") || ext == Some("hos");
                let is_hf = p.is_dir() && p.join("config.json").exists();
                if is_model_file || is_hf {
                    names.insert(nm);
                }
            }
        }
    }
    json!({ "data": names.into_iter().collect::<Vec<_>>() })
}

fn print_banner(model: &str, family: &str, addr: &str) {
    println!("  +------------------------------------------------------------+");
    println!("  |  flwr serve  ·  one body, listening                        |");
    println!("  +------------------------------------------------------------+");
    println!("    model    {model}");
    println!("    dialect  {family}");
    println!("    chat UI  http://{addr}/            (open in a browser)");
    println!("    endpoint http://{addr}/v1/chat/completions");
    println!("    models   switch the resident model live from the UI dropdown");
    println!("    (OpenAI-compatible · POST messages · stream:true for SSE)");
    println!("    (connections handled concurrently; generation serialized)");
    println!();
}

/// Read and route one request on a connection thread. GET endpoints answer here
/// directly; a chat POST is parsed and handed to the engine thread.
fn dispatch(
    mut stream: TcpStream,
    model: Arc<Mutex<String>>,
    tx: Sender<Job>,
) -> std::io::Result<()> {
    let model_name = model.lock().map(|g| g.clone()).unwrap_or_default();
    let mut reader = BufReader::new(stream.try_clone()?);

    // request line
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Ok(());
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();

    // headers — we only need Content-Length
    let mut content_len = 0usize;
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h)? == 0 {
            break;
        }
        let t = h.trim_end();
        if t.is_empty() {
            break;
        }
        if let Some(v) = t.to_ascii_lowercase().strip_prefix("content-length:") {
            content_len = v.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0u8; content_len];
    if content_len > 0 {
        reader.read_exact(&mut body)?;
    }

    match (method.as_str(), path.as_str()) {
        // The browser UI — a single self-contained page (no framework, no CDN).
        ("GET", "/") => write_html(&mut stream, CHAT_HTML),
        ("GET", "/health") => write_text(
            &mut stream,
            200,
            "OK",
            "flwr is alive — one body, listening\n",
        ),
        ("GET", "/v1/models") => {
            let payload = json!({
                "object": "list",
                "data": [ { "id": model_name, "object": "model", "owned_by": "flwr" } ]
            });
            write_json(&mut stream, 200, &payload)
        }
        ("POST", "/v1/chat/completions") => match parse_chat(&body) {
            Ok((msgs, params, want_stream)) => {
                // Hand the connection to the engine thread; it writes the reply.
                let _ = tx.send(Job::Chat(ChatJob {
                    stream,
                    msgs,
                    params,
                    want_stream,
                }));
                Ok(())
            }
            Err(msg) => write_json(&mut stream, 400, &json!({ "error": { "message": msg } })),
        },

        // Model selection: list available models, and switch the resident one.
        ("GET", "/models/available") => write_json(&mut stream, 200, &available_models()),
        ("POST", "/model") => {
            let v: Value = serde_json::from_slice(&body).unwrap_or_else(|_| json!({}));
            match v["name"].as_str() {
                Some(name) => {
                    let (rtx, rrx) = mpsc::channel();
                    let _ = tx.send(Job::Switch {
                        name: name.to_string(),
                        reply: rtx,
                    });
                    match rrx.recv() {
                        Ok(Ok(n)) => {
                            write_json(&mut stream, 200, &json!({ "ok": true, "model": n }))
                        }
                        Ok(Err(e)) => {
                            write_json(&mut stream, 400, &json!({ "ok": false, "error": e }))
                        }
                        Err(_) => write_json(
                            &mut stream,
                            500,
                            &json!({ "ok": false, "error": "engine gone" }),
                        ),
                    }
                }
                None => write_json(
                    &mut stream,
                    400,
                    &json!({ "error": { "message": "need a model name" } }),
                ),
            }
        }

        // Saved chat transcripts (provenance-bearing JSON in ~/.hos/chats).
        ("POST", "/chats") => save_chat(&model_name, &body, &mut stream),
        ("GET", "/chats") => write_json(&mut stream, 200, &crate::chats::list()),
        ("GET", p) if p.starts_with("/chats/") => {
            match crate::chats::safe_id(&p["/chats/".len()..]).and_then(|i| crate::chats::load(&i))
            {
                Some(s) => write_json_raw(&mut stream, 200, &s),
                None => write_text(&mut stream, 404, "Not Found", "no such chat\n"),
            }
        }
        ("DELETE", p) if p.starts_with("/chats/") => {
            let ok = crate::chats::safe_id(&p["/chats/".len()..])
                .map(|i| crate::chats::delete(&i))
                .unwrap_or(false);
            write_json(
                &mut stream,
                if ok { 200 } else { 404 },
                &json!({ "ok": ok }),
            )
        }

        _ => write_text(&mut stream, 404, "Not Found", "no such organ\n"),
    }
}

/// Persist a conversation. The client sends `{ id, messages }`; the server stamps
/// the model provenance (name + content-hash id + source + quant, looked up in
/// the store) and the sampling params, so the saved record is reproducible.
fn save_chat(model_name: &str, body: &[u8], stream: &mut TcpStream) -> std::io::Result<()> {
    let v: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => {
            return write_json(
                stream,
                400,
                &json!({ "error": { "message": format!("bad JSON: {e}") } }),
            )
        }
    };
    let id = v["id"]
        .as_str()
        .and_then(crate::chats::safe_id)
        .unwrap_or_else(crate::chats::new_id);
    let model = match crate::store::lookup(model_name) {
        Some(m) => {
            json!({ "name": m.name, "id": m.identity, "source": m.source, "quant": m.quant })
        }
        None => json!({ "name": model_name }),
    };
    let d = Params::default();
    let params = json!({
        "temperature": d.temp, "top_p": d.top_p, "top_k": d.top_k,
        "seed": d.seed, "repeat_penalty": d.rep_penalty, "max_tokens": d.max_tokens
    });
    match crate::chats::save(&id, &v["messages"], model, params) {
        Ok(meta) => write_json(stream, 200, &meta),
        Err(e) => write_json(
            stream,
            500,
            &json!({ "error": { "message": e.to_string() } }),
        ),
    }
}

/// Parse a chat-completions body into messages + sampling params.
fn parse_chat(body: &[u8]) -> Result<(Vec<Message>, Params, bool), String> {
    let req: Value = serde_json::from_slice(body).map_err(|e| format!("bad JSON: {e}"))?;
    let msgs: Vec<Message> = req["messages"]
        .as_array()
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
        .unwrap_or_default();
    if msgs.is_empty() {
        return Err("no messages".into());
    }
    let mut p = Params::default();
    if let Some(v) = req["max_tokens"].as_u64() {
        p.max_tokens = v as usize;
    }
    if let Some(v) = req["temperature"].as_f64() {
        p.temp = v as f32;
    }
    if let Some(v) = req["top_p"].as_f64() {
        p.top_p = v as f32;
    }
    if let Some(v) = req["seed"].as_u64() {
        p.seed = v;
    }
    let want_stream = req["stream"].as_bool().unwrap_or(false);
    Ok((msgs, p, want_stream))
}

/// Generate a reply on the engine thread and write it to the job's connection.
fn run_chat(eng: &mut Backend, model_name: &str, mut job: ChatJob) -> std::io::Result<()> {
    let p = &job.params;
    eprint!("  · POST /v1/chat/completions  {} msg", job.msgs.len());
    let started = std::time::Instant::now();
    let id = "chatcmpl-flwr";

    if job.want_stream {
        // Server-Sent Events: one delta per token, then [DONE].
        let head = "HTTP/1.1 200 OK\r\n\
                    Content-Type: text/event-stream\r\n\
                    Cache-Control: no-cache\r\n\
                    Connection: close\r\n\r\n";
        job.stream.write_all(head.as_bytes())?;
        let mut sink = job.stream.try_clone()?;
        let n = eng.chat(
            &job.msgs,
            p.max_tokens,
            p.temp,
            p.top_k,
            p.top_p,
            p.rep_penalty,
            p.repeat_last_n,
            p.seed,
            |piece| {
                let chunk = json!({
                    "id": id, "object": "chat.completion.chunk", "model": model_name,
                    "choices": [ { "index": 0, "delta": { "content": piece }, "finish_reason": Value::Null } ]
                });
                let _ = write!(sink, "data: {chunk}\r\n\r\n");
                let _ = sink.flush();
            },
        );
        let done = json!({
            "id": id, "object": "chat.completion.chunk", "model": model_name,
            "choices": [ { "index": 0, "delta": {}, "finish_reason": "stop" } ]
        });
        write!(sink, "data: {done}\r\n\r\n")?;
        sink.write_all(b"data: [DONE]\r\n\r\n")?;
        sink.flush()?;
        eprintln!(
            "  -> {n} tok  {:.2}s (stream)",
            started.elapsed().as_secs_f64()
        );
        Ok(())
    } else {
        let mut text = String::new();
        let n = eng.chat(
            &job.msgs,
            p.max_tokens,
            p.temp,
            p.top_k,
            p.top_p,
            p.rep_penalty,
            p.repeat_last_n,
            p.seed,
            |piece| text.push_str(piece),
        );
        eprintln!("  -> {n} tok  {:.2}s", started.elapsed().as_secs_f64());
        let payload = json!({
            "id": id, "object": "chat.completion", "model": model_name,
            "choices": [ {
                "index": 0,
                "message": { "role": "assistant", "content": text },
                "finish_reason": "stop"
            } ],
            "usage": { "completion_tokens": n }
        });
        write_json(&mut job.stream, 200, &payload)
    }
}

fn write_text(stream: &mut TcpStream, code: u16, reason: &str, body: &str) -> std::io::Result<()> {
    let resp = format!(
        "HTTP/1.1 {code} {reason}\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(resp.as_bytes())?;
    stream.flush()
}

fn write_json(stream: &mut TcpStream, code: u16, payload: &Value) -> std::io::Result<()> {
    let body = payload.to_string();
    let resp = format!(
        "HTTP/1.1 {code} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(resp.as_bytes())?;
    stream.flush()
}

fn write_json_raw(stream: &mut TcpStream, code: u16, body: &str) -> std::io::Result<()> {
    let resp = format!(
        "HTTP/1.1 {code} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(resp.as_bytes())?;
    stream.flush()
}

fn write_html(stream: &mut TcpStream, html: &str) -> std::io::Result<()> {
    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{html}",
        html.len()
    );
    stream.write_all(resp.as_bytes())?;
    stream.flush()
}

/// The browser chat UI: one self-contained page, no framework or CDN — styled to
/// match the HOS whitepaper (serif masthead, a heavy rule, navy accent, a paper
/// column, an anatomical footer). It talks to this server's own
/// `/v1/chat/completions` (streaming) and `/v1/models`.
const CHAT_HTML: &str = r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>flwr</title>
<style>
  /* BRUTALIST — palette in CSS vars; swap these 5 lines for your exact colors. */
  :root{ --bg:#f2f0eb; --ink:#111111; --accent:#ff4a1c; --accent-fg:#ffffff; --line:#111111; --soft:#ffffff; }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:15px/1.6 ui-monospace,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace}
  .wrap{max-width:820px;margin:0 auto;padding:0 20px 170px}
  header{border-bottom:3px solid var(--line);padding:30px 0 14px}
  h1{font-size:46px;font-weight:800;letter-spacing:-1.5px;margin:0;text-transform:lowercase}
  .sub{margin-top:8px;text-transform:uppercase;letter-spacing:.18em;font-size:11px}
  .meta{margin-top:6px;font-size:11px;color:#666;text-transform:uppercase;letter-spacing:.1em}
  .turn{border-bottom:1px solid var(--line);padding:16px 0}
  .role{display:inline-block;text-transform:uppercase;letter-spacing:.14em;font-size:10px;
    font-weight:700;padding:2px 9px;border:2px solid var(--ink);margin-bottom:9px}
  .turn.you .role{background:var(--ink);color:var(--bg)}
  .turn.flwr .role{background:var(--accent);color:var(--accent-fg);border-color:var(--accent)}
  .text{white-space:pre-wrap;word-wrap:break-word}
  .empty{padding:28px 0;text-transform:uppercase;letter-spacing:.12em;font-size:12px;color:#666}
  .composer{position:fixed;left:0;right:0;bottom:0;background:var(--bg);border-top:3px solid var(--line)}
  .composer .inner{max-width:820px;margin:0 auto;padding:12px 20px;display:flex;gap:12px}
  textarea{flex:1;resize:none;border:2px solid var(--ink);border-radius:0;background:var(--soft);
    padding:11px 12px;font:inherit;min-height:46px;max-height:170px;box-shadow:5px 5px 0 var(--ink)}
  textarea:focus{outline:none;border-color:var(--accent);box-shadow:5px 5px 0 var(--accent)}
  button{border:2px solid var(--ink);border-radius:0;background:var(--accent);color:var(--accent-fg);
    padding:0 24px;font:inherit;font-weight:700;text-transform:uppercase;letter-spacing:.1em;
    cursor:pointer;box-shadow:5px 5px 0 var(--ink)}
  button:active{transform:translate(5px,5px);box-shadow:none}
  button:disabled{opacity:.5;cursor:default;box-shadow:none}
  footer{border-top:3px solid var(--line);text-align:center;padding:16px 0;font-size:10px;
    text-transform:uppercase;letter-spacing:.22em}
  .themebar{margin-top:13px;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
  .themebar .lbl{font-size:10px;letter-spacing:.16em;color:#666}
  .themebar button{box-shadow:none;padding:3px 11px;font-size:10px;background:var(--soft);
    color:var(--ink);border:2px solid var(--ink)}
  .themebar button.on{background:var(--ink);color:var(--bg)}
  .app{display:flex;align-items:flex-start}
  .side{flex:0 0 240px;width:240px;border-right:3px solid var(--line);height:100vh;
    position:sticky;top:0;overflow:auto}
  .sidehead{display:flex;justify-content:space-between;align-items:center;
    border-bottom:3px solid var(--line);padding:18px 14px;text-transform:uppercase;
    letter-spacing:.14em;font-size:11px;font-weight:700}
  .sidehead button{box-shadow:none;padding:4px 10px;font-size:10px}
  #chatlist{display:flex;flex-direction:column}
  .chatitem{display:block;width:100%;text-align:left;font:inherit;color:inherit;cursor:pointer;
    background:none;border:0;border-bottom:1px solid var(--line);border-left:4px solid transparent;
    border-radius:0;box-shadow:none;padding:11px 14px;text-transform:none;letter-spacing:0}
  .chatitem:hover{background:var(--soft)}
  .chatitem.on{border-left-color:var(--accent)}
  .chatitem .t{font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .chatitem .m{font-size:10px;color:#777;text-transform:uppercase;letter-spacing:.08em;margin-top:3px}
  .chatitem .del{float:right;color:#999;font-weight:700;padding-left:10px}
  .chatitem .del:hover{color:var(--accent)}
  .composer{left:240px}
  .modelrow{margin-top:11px;display:flex;gap:8px;align-items:center}
  .modelrow .lbl{font-size:10px;letter-spacing:.16em;color:#666}
  select#modelsel{font:inherit;font-size:12px;border:2px solid var(--ink);border-radius:0;
    background:var(--soft);color:var(--ink);padding:4px 8px;box-shadow:3px 3px 0 var(--ink);
    max-width:520px}
  select#modelsel:disabled{opacity:.5}
</style>
</head>
<body>
<div class="app">
  <aside class="side">
    <div class="sidehead"><span>CHATS</span><button id="newchat">+ NEW</button></div>
    <div id="chatlist"></div>
  </aside>
  <div class="wrap">
    <header>
      <h1>flwr</h1>
      <div class="sub">one body — a local model on the HOS engine, talking.</div>
      <div class="meta" id="meta">connecting…</div>
      <div class="modelrow"><span class="lbl">MODEL</span><select id="modelsel"></select></div>
      <div class="themebar" id="themebar"><span class="lbl">THEME</span></div>
    </header>
    <div id="log"><div class="empty" id="empty">Ask it something below.</div></div>
  </div>
</div>
<div class="composer"><div class="inner">
  <textarea id="in" placeholder="Message flwr…  (Enter to send · Shift+Enter for a newline)"></textarea>
  <button id="send">Send</button>
</div></div>
<footer>HOS — one body, not a pile of organs</footer>
<script>
const log=document.getElementById('log'), input=document.getElementById('in'),
      send=document.getElementById('send'), meta=document.getElementById('meta'),
      messages=[];
let chatId=newId();
// In-app brutalist palettes. Each sets the CSS variables; pick one live.
const PALETTES={
  bone:{name:'BONE',v:{'--bg':'#f2f0eb','--ink':'#111111','--accent':'#ff4a1c','--accent-fg':'#ffffff','--line':'#111111','--soft':'#ffffff'}},
  concrete:{name:'CONCRETE',v:{'--bg':'#e4e4e4','--ink':'#0a0a0a','--accent':'#0000ee','--accent-fg':'#ffffff','--line':'#0a0a0a','--soft':'#ffffff'}},
  acid:{name:'ACID',v:{'--bg':'#fafaf0','--ink':'#111111','--accent':'#d6f500','--accent-fg':'#111111','--line':'#111111','--soft':'#ffffff'}},
  ink:{name:'INK',v:{'--bg':'#0e0e0e','--ink':'#f5f5f5','--accent':'#ff4a1c','--accent-fg':'#ffffff','--line':'#f5f5f5','--soft':'#1a1a1a'}}
};
function applyTheme(key){
  const p=PALETTES[key]; if(!p) return;
  for(const k in p.v){ document.documentElement.style.setProperty(k,p.v[k]); }
  try{localStorage.setItem('flwr-theme',key);}catch(e){}
  document.querySelectorAll('#themebar button').forEach(b=>b.classList.toggle('on',b.dataset.k===key));
}
(function(){
  const bar=document.getElementById('themebar');
  for(const k in PALETTES){
    const b=document.createElement('button'); b.textContent=PALETTES[k].name; b.dataset.k=k;
    b.onclick=()=>applyTheme(k); bar.appendChild(b);
  }
  let saved='bone'; try{saved=localStorage.getItem('flwr-theme')||'bone';}catch(e){}
  applyTheme(saved);
})();
const modelsel=document.getElementById('modelsel');
let activeModel='';
async function initModels(){
  let cur='model';
  try{ const d=await (await fetch('/v1/models')).json(); cur=(d.data&&d.data[0]&&d.data[0].id)||'model'; }
  catch(e){ meta.textContent='model: (offline)'; return; }
  activeModel=cur; meta.textContent='model: '+cur;
  try{
    const a=await (await fetch('/models/available')).json();
    const opts=(a.data||[]).slice(); if(!opts.includes(cur)) opts.unshift(cur);
    modelsel.innerHTML='';
    for(const n of opts){ const o=document.createElement('option'); o.value=n; o.textContent=n;
      if(n===cur) o.selected=true; modelsel.appendChild(o); }
  }catch(e){}
}
modelsel.onchange=async ()=>{
  const name=modelsel.value; if(name===activeModel) return;
  meta.textContent='switching to '+name+' … (loading)'; send.disabled=true; modelsel.disabled=true;
  try{
    const r=await (await fetch('/model',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name})})).json();
    if(r.ok){ activeModel=r.model; meta.textContent='model: '+r.model; }
    else { meta.textContent='switch failed: '+(r.error||'?'); modelsel.value=activeModel; }
  }catch(e){ meta.textContent='switch failed'; modelsel.value=activeModel; }
  send.disabled=false; modelsel.disabled=false; input.focus();
};
initModels();
function turn(cls,label){
  const e=log.querySelector('.empty'); if(e){e.remove();}
  const t=document.createElement('div'); t.className='turn '+cls;
  const r=document.createElement('div'); r.className='role'; r.textContent=label;
  const b=document.createElement('div'); b.className='text';
  t.appendChild(r); t.appendChild(b); log.appendChild(t);
  window.scrollTo(0,document.body.scrollHeight); return b;
}
async function ask(){
  const text=input.value.trim(); if(!text) return;
  input.value=''; send.disabled=true;
  turn('you','You').textContent=text;
  messages.push({role:'user',content:text});
  const out=turn('flwr','flwr'); out.textContent='…'; let acc='';
  try{
    const resp=await fetch('/v1/chat/completions',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({messages,stream:true,max_tokens:512})});
    const reader=resp.body.getReader(), dec=new TextDecoder(); let buf='';
    while(true){
      const {done,value}=await reader.read(); if(done) break;
      buf+=dec.decode(value,{stream:true}).replace(/\r/g,''); let i;
      while((i=buf.indexOf('\n\n'))>=0){
        const line=buf.slice(0,i).trim(); buf=buf.slice(i+2);
        if(!line.startsWith('data:')) continue;
        const data=line.slice(5).trim(); if(data==='[DONE]') continue;
        try{const j=JSON.parse(data); const p=((j.choices[0]||{}).delta||{}).content||'';
          acc+=p; out.textContent=acc; window.scrollTo(0,document.body.scrollHeight);}catch(e){}
      }
    }
    if(!acc) out.textContent='[no response]';
  }catch(e){ out.textContent='[error talking to flwr: '+e+']'; }
  messages.push({role:'assistant',content:acc});
  saveCurrent();
  send.disabled=false; input.focus();
}
send.onclick=ask;
input.addEventListener('keydown',e=>{ if(e.key==='Enter'&&!e.shiftKey){e.preventDefault(); ask();} });
input.focus();

// ---- saved chats (provenance-bearing transcripts on the server) ----
function newId(){return 'c'+Date.now().toString(36)+Math.random().toString(36).slice(2,6);}
function fmtAgo(u){ if(!u) return ''; let s=Math.floor(Date.now()/1000)-u;
  if(s<60)return s+'s'; if(s<3600)return Math.floor(s/60)+'m';
  if(s<86400)return Math.floor(s/3600)+'h'; return Math.floor(s/86400)+'d'; }
function renderMessages(){
  log.innerHTML='';
  if(!messages.length){ log.innerHTML='<div class="empty">Ask it something below.</div>'; return; }
  for(const m of messages){ turn(m.role==='user'?'you':'flwr', m.role==='user'?'You':'flwr').textContent=m.content; }
}
async function refreshChats(){
  try{
    const d=await (await fetch('/chats')).json();
    const list=document.getElementById('chatlist'); list.innerHTML='';
    for(const c of (d.data||[])){
      const it=document.createElement('button'); it.className='chatitem'+(c.id===chatId?' on':'');
      it.innerHTML='<span class="del" title="delete">×</span><div class="t"></div><div class="m"></div>';
      it.querySelector('.t').textContent=c.title||'untitled';
      it.querySelector('.m').textContent=(c.messages||0)+' msg · '+fmtAgo(c.updated);
      it.onclick=(e)=>{ if(e.target.classList.contains('del')){delChat(c.id);return;} loadChat(c.id); };
      list.appendChild(it);
    }
  }catch(e){}
}
async function loadChat(id){
  try{
    const c=await (await fetch('/chats/'+encodeURIComponent(id))).json();
    messages.length=0; for(const m of (c.messages||[])) messages.push(m);
    chatId=id; renderMessages(); refreshChats(); window.scrollTo(0,document.body.scrollHeight);
  }catch(e){}
}
async function delChat(id){
  try{ await fetch('/chats/'+encodeURIComponent(id),{method:'DELETE'}); }catch(e){}
  if(id===chatId){ newChat(); } else { refreshChats(); }
}
function newChat(){ chatId=newId(); messages.length=0; renderMessages(); refreshChats(); input.focus(); }
async function saveCurrent(){
  if(!messages.length) return;
  try{ await fetch('/chats',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({id:chatId,messages})}); refreshChats(); }catch(e){}
}
document.getElementById('newchat').onclick=newChat;
refreshChats();
</script>
</body>
</html>"##;
