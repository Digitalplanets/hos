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
    Qwen35(Box<hos::qwen35::ChatSession>),
}

impl Backend {
    fn family_label(&self) -> &'static str {
        match self {
            Backend::Engine(e) => e.chat_family().label(),
            Backend::Gemma4 { .. } => "gemma",
            Backend::Qwen35(s) => s.family_label(),
        }
    }

    /// Generate a reply to `msgs`, streaming `Chunk`s to `on` (reasoning vs answer).
    /// Non-reasoning backends emit everything as `Chunk::Answer`; only qwen35 splits
    /// out `<think>` reasoning. `think` is ignored by non-reasoning backends.
    #[allow(clippy::too_many_arguments)]
    fn chat(
        &mut self,
        msgs: &[Message],
        image: Option<&[u8]>,
        think: hos::qwen35::Think,
        max_tokens: usize,
        temp: f32,
        top_k: usize,
        top_p: f32,
        rep_penalty: f32,
        repeat_last_n: usize,
        seed: u64,
        mut on: impl FnMut(hos::qwen35::Chunk),
    ) -> usize {
        use hos::qwen35::Chunk;
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
                |piece| on(Chunk::Answer(piece)),
            ),
            Backend::Gemma4 { sess, rng } => {
                if seed != 0 {
                    *rng = seed;
                }
                let hist: Vec<(String, String)> = msgs
                    .iter()
                    .map(|m| {
                        let role = if m.role == "assistant" {
                            "model"
                        } else {
                            "user"
                        };
                        (role.to_string(), m.content.clone())
                    })
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
                sess.generate(&hist, &gp, rng, |piece| on(Chunk::Answer(piece)))
            }
            Backend::Qwen35(s) => {
                // Encode the image (if any + a vision tower is attached) and splice
                // it into the turn; otherwise this is a plain text chat.
                let emb = match (image, s.has_vision()) {
                    (Some(bytes), true) => match s.encode_image_bytes(bytes) {
                        Ok(e) => Some(e),
                        Err(err) => {
                            eprintln!("  · image encode failed: {err}");
                            None
                        }
                    },
                    _ => None,
                };
                s.chat_img(
                    msgs,
                    emb.as_deref(),
                    think,
                    max_tokens,
                    temp,
                    top_k,
                    top_p,
                    rep_penalty,
                    repeat_last_n,
                    seed,
                    on,
                )
            }
        }
    }

    /// Load a backend for `path`, picking the gemma4 or qwen35 session for those
    /// custom arches and the generic `Engine` for everything else.
    pub fn load(path: &Path, gpu: bool) -> Result<Backend, String> {
        // Route BOTH a Gemma-4 HF checkpoint dir AND a native `.hos` gemma4 capsule
        // to the native decoder. `Session::load` dispatches on `is_gemma4_capsule`
        // internally (via `Gemma4::from_capsule`); serve previously only checked the
        // directory form, so a gemma4 `.hos` fell through to the generic GGUF loader
        // and failed with "missing model metadata: tokenizer.ggml.tokens".
        if crate::gemma4_chat::is_gemma4(path) || crate::gemma4_chat::is_gemma4_capsule(path) {
            return crate::gemma4_chat::Session::load(path, gpu)
                .map(|sess| Backend::Gemma4 { sess, rng: 42 })
                .map_err(|e| e.to_string());
        }
        // qwen35 hybrid: a GGUF whose arch is the Gated-DeltaNet hybrid.
        if let Ok(g) = hos::gguf::Gguf::open(path) {
            if hos::model::Arch::detect(&g) == hos::model::Arch::Qwen35Hybrid {
                let tok = hos::tokenizer::Tokenizer::from_gguf(&g).map_err(|e| e.to_string())?;
                let mut s = hos::qwen35::ChatSession::load(&g, tok, gpu).map_err(|e| e.to_string())?;
                // Auto-attach the vision tower if a sibling mmproj-*.gguf exists, so
                // the OpenAI image_url path works out of the box.
                if let Some(mmp) = crate::find_sibling_mmproj(path) {
                    if let Ok(mg) = hos::gguf::Gguf::open(std::path::Path::new(&mmp)) {
                        match s.attach_vision(&mg) {
                            Ok(()) => eprintln!("    vision  mmproj attached ({mmp})"),
                            Err(e) => eprintln!("    vision  mmproj load failed: {e}"),
                        }
                    }
                }
                return Ok(Backend::Qwen35(Box::new(s)));
            }
        }
        Engine::load(path, gpu)
            .map(Backend::Engine)
            .map_err(|e| e.to_string())
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
    think: hos::qwen35::Think,
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
            // A fresh seed per request when the client doesn't send one, so replies
            // vary (OpenAI convention: omitting `seed` is non-deterministic). A
            // client that passes `seed` overrides this and gets reproducible output.
            seed: crate::random_seed(),
            think: hos::qwen35::Think::default(),
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
    image: Option<Vec<u8>>,
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

/// Live activity for the `GET /monitor` endpoint + the web UI's bottom meter.
#[derive(Default)]
struct ServeStats {
    requests: u64,
    tokens: u64,
    last_tok_s: f64,
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
    let stats = Arc::new(Mutex::new(ServeStats::default()));
    let started = std::time::Instant::now();
    let (tx, rx) = mpsc::channel::<Job>();
    let accept_model = model.clone();
    let accept_stats = stats.clone();
    // Accept + parse off the engine thread: one reader thread per connection.
    thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(stream) = conn else { continue };
            let tx = tx.clone();
            let m = accept_model.clone();
            let st = accept_stats.clone();
            thread::spawn(move || {
                if let Err(e) = dispatch(stream, m, tx, st, started) {
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
                let t0 = std::time::Instant::now();
                match run_chat(&mut backend, &name, c) {
                    Ok(n) => {
                        let secs = t0.elapsed().as_secs_f64();
                        if let Ok(mut s) = stats.lock() {
                            s.requests += 1;
                            s.tokens += n as u64;
                            s.last_tok_s = n as f64 / secs.max(1e-9);
                        }
                    }
                    Err(e) => eprintln!("  ! generation write error: {e}"),
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
    stats: Arc<Mutex<ServeStats>>,
    started: std::time::Instant,
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
            Ok((msgs, params, want_stream, image)) => {
                // Hand the connection to the engine thread; it writes the reply.
                let _ = tx.send(Job::Chat(ChatJob {
                    stream,
                    msgs,
                    params,
                    want_stream,
                    image,
                }));
                Ok(())
            }
            Err(msg) => write_json(&mut stream, 400, &json!({ "error": { "message": msg } })),
        },

        // Live activity meter feed (the web UI's bottom strip polls this).
        ("GET", "/monitor") => {
            let cur = model.lock().map(|g| g.clone()).unwrap_or_default();
            let (req, tok, tps) = stats
                .lock()
                .map(|s| (s.requests, s.tokens, s.last_tok_s))
                .unwrap_or((0, 0, 0.0));
            write_json(
                &mut stream,
                200,
                &json!({
                    "model": cur,
                    "requests": req,
                    "tokens": tok,
                    "last_tok_s": (tps * 10.0).round() / 10.0,
                    "uptime_s": started.elapsed().as_secs(),
                    "memory_mb": crate::current_rss_mb(),
                }),
            )
        }
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

        // Reasoning receipts — the agent-facing read path. Fetch a model's full
        // reasoning trace by the content-addressed id returned from a chat call.
        ("GET", p) if p.starts_with("/v1/reasoning/") => {
            let rid = &p["/v1/reasoning/".len()..];
            match crate::reasoning::get(rid) {
                Some(r) => write_json(&mut stream, 200, &r.to_json()),
                None => write_json(
                    &mut stream,
                    404,
                    &json!({ "error": { "message": "no such reasoning receipt" } }),
                ),
            }
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
    let messages = &v["messages"];
    let memory = crate::memory::summarize(&crate::memory::messages_from_json(messages));
    match crate::chats::save(
        &id,
        messages,
        model,
        params,
        crate::memory::to_json(&memory),
    ) {
        Ok(meta) => write_json(stream, 200, &meta),
        Err(e) => write_json(
            stream,
            500,
            &json!({ "error": { "message": e.to_string() } }),
        ),
    }
}

/// Parse a chat-completions body into messages + sampling params.
/// Minimal, dependency-free base64 decoder for `data:...;base64,` image URLs.
fn b64_decode(s: &str) -> Option<Vec<u8>> {
    let val = |c: u8| -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    };
    let (mut out, mut buf, mut bits) = (Vec::new(), 0u32, 0u32);
    for &c in s.as_bytes() {
        if matches!(c, b'=' | b'\n' | b'\r' | b' ') {
            continue;
        }
        buf = (buf << 6) | val(c)? as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
        }
    }
    Some(out)
}

/// OpenAI content can be a plain string or an array of `{type:text|image_url}`.
/// The visible text is the concatenation of the text parts.
fn content_text(content: &Value) -> String {
    if let Some(s) = content.as_str() {
        return s.to_string();
    }
    let mut t = String::new();
    if let Some(arr) = content.as_array() {
        for item in arr {
            if item["type"] == "text" {
                if let Some(s) = item["text"].as_str() {
                    t.push_str(s);
                }
            }
        }
    }
    t
}

/// Extract the first image from an OpenAI content array: a `data:...;base64,`
/// URL (decoded) or a local file path.
fn extract_image(content: &Value) -> Option<Vec<u8>> {
    for item in content.as_array()? {
        if item["type"] == "image_url" {
            let url = item["image_url"]["url"].as_str()?;
            if let Some(idx) = url.find("base64,") {
                return b64_decode(&url[idx + 7..]);
            }
            if let Ok(b) = std::fs::read(url) {
                return Some(b);
            }
        }
    }
    None
}

fn parse_chat(body: &[u8]) -> Result<(Vec<Message>, Params, bool, Option<Vec<u8>>), String> {
    let req: Value = serde_json::from_slice(body).map_err(|e| format!("bad JSON: {e}"))?;
    let mut image: Option<Vec<u8>> = None;
    let msgs: Vec<Message> = req["messages"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .map(|m| {
                    if image.is_none() {
                        image = extract_image(&m["content"]);
                    }
                    Message::new(m["role"].as_str().unwrap_or("user"), &content_text(&m["content"]))
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
    // Reasoning controls (qwen35): OpenAI-style `reasoning_effort`, plus
    // `enable_thinking` (Qwen convention). Ignored by non-reasoning models.
    if let Some(v) = req["enable_thinking"].as_bool() {
        p.think.on = v;
    }
    if let Some(s) = req["reasoning_effort"].as_str() {
        if let Some(e) = hos::qwen35::Effort::parse(s) {
            p.think.effort = e;
        }
    }
    let want_stream = req["stream"].as_bool().unwrap_or(false);
    Ok((msgs, p, want_stream, image))
}

/// Build + persist a content-addressed reasoning receipt for a served turn, and
/// return its id. Returns `None` when there was no reasoning (non-reasoning model
/// or thinking disabled), so callers omit the field gracefully.
fn save_reasoning_receipt(
    model_name: &str,
    msgs: &[Message],
    reasoning: &str,
    answer: &str,
    think: hos::qwen35::Think,
    n: usize,
) -> Option<String> {
    if reasoning.trim().is_empty() {
        return None;
    }
    let query = msgs
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.as_str())
        .unwrap_or("");
    let effort = match think.effort {
        hos::qwen35::Effort::Low => "low",
        hos::qwen35::Effort::Medium => "medium",
        hos::qwen35::Effort::Xhigh => "xhigh",
    };
    let rr = crate::reasoning::ReasoningReceipt::new(
        "serve",
        msgs.len(),
        query,
        effort,
        reasoning,
        answer,
        n,
        json!({ "name": model_name }),
    );
    crate::reasoning::save(&rr).ok().flatten()
}

/// Generate a reply on the engine thread and write it to the job's connection.
fn run_chat(eng: &mut Backend, model_name: &str, mut job: ChatJob) -> std::io::Result<usize> {
    let p = &job.params;
    let bundle = crate::memory::assemble(&job.msgs);
    let chat_msgs = if bundle.omitted_messages > 0 {
        &bundle.messages
    } else {
        &job.msgs
    };
    eprint!("  · POST /v1/chat/completions  {} msg", job.msgs.len());
    if bundle.omitted_messages > 0 {
        eprint!(
            "  compacted={} prompt~{}tok",
            bundle.omitted_messages, bundle.estimated_tokens
        );
    }
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
        let mut reasoning = String::new();
        let mut answer = String::new();
        let n = eng.chat(
            chat_msgs,
            job.image.as_deref(),
            p.think,
            p.max_tokens,
            p.temp,
            p.top_k,
            p.top_p,
            p.rep_penalty,
            p.repeat_last_n,
            p.seed,
            |chunk| {
                // Reasoning streams as `reasoning_content` (OpenAI reasoning-model
                // convention) so the visible `content` stays the clean answer.
                let delta = match chunk {
                    hos::qwen35::Chunk::Reasoning(t) => {
                        reasoning.push_str(t);
                        json!({ "reasoning_content": t })
                    }
                    hos::qwen35::Chunk::Answer(t) => {
                        answer.push_str(t);
                        json!({ "content": t })
                    }
                };
                let evt = json!({
                    "id": id, "object": "chat.completion.chunk", "model": model_name,
                    "choices": [ { "index": 0, "delta": delta, "finish_reason": Value::Null } ]
                });
                let _ = write!(sink, "data: {evt}\r\n\r\n");
                let _ = sink.flush();
            },
        );
        // Persist the reasoning receipt and announce its id as a final delta.
        // Regular (non-reasoning) models get None here, so the delta stays empty —
        // byte-identical to the pre-receipts behavior.
        let rr_id = save_reasoning_receipt(model_name, &job.msgs, &reasoning, &answer, p.think, n);
        let done_delta = match &rr_id {
            Some(rid) => json!({ "reasoning_id": rid }),
            None => json!({}),
        };
        let done = json!({
            "id": id, "object": "chat.completion.chunk", "model": model_name,
            "choices": [ { "index": 0, "delta": done_delta, "finish_reason": "stop" } ]
        });
        write!(sink, "data: {done}\r\n\r\n")?;
        sink.write_all(b"data: [DONE]\r\n\r\n")?;
        sink.flush()?;
        eprintln!(
            "  -> {n} tok  {:.2}s (stream)",
            started.elapsed().as_secs_f64()
        );
        Ok(n)
    } else {
        let mut text = String::new();
        let mut reasoning = String::new();
        let n = eng.chat(
            chat_msgs,
            job.image.as_deref(),
            p.think,
            p.max_tokens,
            p.temp,
            p.top_k,
            p.top_p,
            p.rep_penalty,
            p.repeat_last_n,
            p.seed,
            |chunk| match chunk {
                hos::qwen35::Chunk::Reasoning(t) => reasoning.push_str(t),
                hos::qwen35::Chunk::Answer(t) => text.push_str(t),
            },
        );
        eprintln!("  -> {n} tok  {:.2}s", started.elapsed().as_secs_f64());
        let mut message = json!({ "role": "assistant", "content": text });
        if !reasoning.is_empty() {
            message["reasoning_content"] = json!(reasoning.trim());
        }
        // Persist a content-addressed reasoning receipt; expose its id so a client
        // or agent can fetch the full trace later via GET /v1/reasoning/{id}.
        let rr_id = save_reasoning_receipt(model_name, &job.msgs, &reasoning, &text, p.think, n);
        if let Some(rid) = &rr_id {
            message["reasoning_id"] = json!(rid);
        }
        let payload = json!({
            "id": id, "object": "chat.completion", "model": model_name,
            "choices": [ {
                "index": 0,
                "message": message,
                "finish_reason": "stop"
            } ],
            "usage": { "completion_tokens": n }
        });
        write_json(&mut job.stream, 200, &payload)?;
        Ok(n)
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
    // The UI (HTML + inline JS) ships inside the binary and changes with each build,
    // so never let the browser cache it — a stale cached page is why a rebuilt server
    // can still show an old UI (and an old, broken model switcher). Always fresh.
    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nCache-Control: no-store, must-revalidate\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{html}",
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
  /* flwr serve — the terminal-blockprint look of flwr.systems, in CSS vars so
     the theme switcher can flip it live. */
  :root{ --bg:#0d1218; --ink:#ced2dc; --accent:#d98faa; --accent-fg:#150c11; --line:rgba(140,152,172,.30); --soft:#0f151c; }
  *{box-sizing:border-box}
  ::selection{background:var(--accent);color:var(--accent-fg)}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:14px/1.65 ui-monospace,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
    -webkit-font-smoothing:antialiased}
  .app{display:flex;align-items:flex-start}
  .side{flex:0 0 236px;width:236px;border-right:1px solid var(--line);height:100vh;
    position:sticky;top:0;overflow:auto;background:var(--soft)}
  .sidehead{display:flex;justify-content:space-between;align-items:center;
    border-bottom:1px solid var(--line);padding:16px 14px;text-transform:uppercase;
    letter-spacing:.16em;font-size:10px;font-weight:700;color:var(--accent)}
  .sidehead button{padding:4px 10px;font-size:10px}
  #chatlist{display:flex;flex-direction:column}
  .chatitem{display:block;width:100%;text-align:left;font:inherit;color:var(--ink);cursor:pointer;
    background:none;border:0;border-bottom:1px solid var(--line);border-left:2px solid transparent;
    padding:11px 14px}
  .chatitem:hover{background:var(--bg)}
  .chatitem.on{border-left-color:var(--accent);background:var(--bg)}
  .chatitem .t{font-size:12.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .chatitem .m{font-size:10px;color:var(--accent);opacity:.7;text-transform:uppercase;letter-spacing:.08em;margin-top:3px}
  .chatitem .del{float:right;opacity:.5;font-weight:700;padding-left:10px}
  .chatitem .del:hover{opacity:1;color:var(--accent)}
  .wrap{flex:1;max-width:840px;margin:0 auto;padding:0 22px 180px}
  header{border-bottom:1px solid var(--line);padding:26px 0 16px}
  h1{font-size:34px;font-weight:700;letter-spacing:-1px;margin:0;text-transform:lowercase;color:var(--accent)}
  .sub{margin-top:8px;opacity:.75;font-size:13px}
  .meta{margin-top:8px;font-size:10px;opacity:.55;text-transform:uppercase;letter-spacing:.14em}
  .turn{border-bottom:1px solid var(--line);padding:16px 0}
  .role{display:inline-block;text-transform:uppercase;letter-spacing:.16em;font-size:9px;
    font-weight:700;padding:2px 8px;border:1px solid var(--line);border-radius:3px;margin-bottom:9px;opacity:.9}
  .turn.you .role{color:var(--ink);border-color:var(--line)}
  .turn.flwr .role{background:var(--accent);color:var(--accent-fg);border-color:var(--accent)}
  .text{white-space:pre-wrap;word-wrap:break-word;line-height:1.7}
  .empty{padding:30px 0;text-transform:uppercase;letter-spacing:.14em;font-size:11px;opacity:.5}
  .composer{position:fixed;left:236px;right:0;bottom:0;background:var(--bg);border-top:1px solid var(--line)}
  .composer .inner{max-width:840px;margin:0 auto;padding:14px 22px;display:flex;gap:12px}
  textarea{flex:1;resize:none;border:1px solid var(--line);border-radius:5px;background:var(--soft);
    color:var(--ink);padding:11px 13px;font:inherit;min-height:46px;max-height:170px}
  textarea:focus{outline:none;border-color:var(--accent)}
  textarea::placeholder{color:var(--ink);opacity:.4}
  button{border:1px solid var(--accent);border-radius:5px;background:var(--accent);color:var(--accent-fg);
    padding:0 22px;font:inherit;font-weight:700;text-transform:uppercase;letter-spacing:.1em;cursor:pointer}
  button:hover{filter:brightness(1.07)}
  button:active{transform:translateY(1px)}
  button:disabled{opacity:.5;cursor:default}
  footer{border-top:1px solid var(--line);text-align:center;padding:16px 0;font-size:10px;
    opacity:.5;text-transform:uppercase;letter-spacing:.24em}
  .themebar{margin-top:13px;display:flex;gap:7px;flex-wrap:wrap;align-items:center}
  .themebar .lbl,.modelrow .lbl{font-size:10px;letter-spacing:.16em;opacity:.5}
  .themebar button{padding:3px 11px;font-size:9px;background:var(--soft);color:var(--ink);
    border:1px solid var(--line);letter-spacing:.12em}
  .themebar button.on{background:var(--accent);color:var(--accent-fg);border-color:var(--accent)}
  .modelrow{margin-top:12px;display:flex;gap:8px;align-items:center}
  select#modelsel{font:inherit;font-size:12px;border:1px solid var(--line);border-radius:5px;
    background:var(--soft);color:var(--ink);padding:5px 9px;max-width:520px}
  select#modelsel:disabled{opacity:.5}
  .sysrow{margin-top:12px}
  .systoggle{padding:3px 11px;font-size:9px;background:var(--soft);color:var(--ink);
    border:1px solid var(--line);letter-spacing:.12em;text-transform:uppercase;cursor:pointer}
  .systoggle.on{border-color:var(--accent);color:var(--accent)}
  .sysbox{display:none;width:100%;margin-top:9px;resize:vertical;border:1px solid var(--line);
    border-radius:5px;background:var(--soft);color:var(--ink);padding:10px 12px;font:inherit;min-height:54px}
  .sysbox:focus{outline:none;border-color:var(--accent)}
  .sysbox.show{display:block}
  /* tiny activity meter, pinned above the composer */
  .meter{position:fixed;right:14px;bottom:84px;z-index:6;display:flex;gap:7px;align-items:center;
    font-size:11px;letter-spacing:.03em;color:var(--accent);background:var(--soft);
    border:1px solid var(--line);border-radius:999px;padding:4px 11px;opacity:.9;
    font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
  .meter .sep{opacity:.4}
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
      <div class="sub">one body, a local model on the HOS engine, talking.</div>
      <div class="meta" id="meta">connecting…</div>
      <div class="modelrow"><span class="lbl">MODEL</span><select id="modelsel"></select></div>
      <div class="themebar" id="themebar"><span class="lbl">THEME</span></div>
      <div class="sysrow"><button class="systoggle" id="systoggle">&#9881; instruction</button></div>
      <textarea id="sys" class="sysbox" placeholder="Optional: give the model a role or a core instruction. It persists for this conversation."></textarea>
    </header>
    <div id="log"><div class="empty" id="empty">Ask it something below.</div></div>
  </div>
</div>
<div class="composer"><div class="inner">
  <textarea id="in" placeholder="Message flwr…  (Enter to send · Shift+Enter for a newline)"></textarea>
  <button id="send">Send</button>
</div></div>
<div class="meter" id="meter">❀ idle</div>
<footer>HOS — one body, not a pile of organs</footer>
<script>
const log=document.getElementById('log'), input=document.getElementById('in'),
      send=document.getElementById('send'), meta=document.getElementById('meta'),
      messages=[];
let chatId=newId();
// In-app brutalist palettes. Each sets the CSS variables; pick one live.
const PALETTES={
  flwr:{name:'FLWR',v:{'--bg':'#0d1218','--ink':'#ced2dc','--accent':'#d98faa','--accent-fg':'#150c11','--line':'rgba(140,152,172,.30)','--soft':'#0f151c'}},
  bloom:{name:'BLOOM',v:{'--bg':'#0d1218','--ink':'#ced2dc','--accent':'#d6ba86','--accent-fg':'#1a130a','--line':'rgba(140,152,172,.30)','--soft':'#0f151c'}},
  moss:{name:'MOSS',v:{'--bg':'#0d1218','--ink':'#ced2dc','--accent':'#8fae86','--accent-fg':'#0d1512','--line':'rgba(140,152,172,.30)','--soft':'#0f151c'}},
  paper:{name:'PAPER',v:{'--bg':'#f6f4ef','--ink':'#1a1d24','--accent':'#b06e8c','--accent-fg':'#ffffff','--line':'rgba(40,44,54,.20)','--soft':'#ffffff'}}
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
  let saved='flwr'; try{const s=localStorage.getItem('flwr-theme'); if(s&&PALETTES[s]) saved=s;}catch(e){}
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
    if(r.ok){ activeModel=r.model; meta.textContent='model: '+r.model;
      const b=turn('flwr','flwr'); b.textContent='· switched to '+r.model; }
    else { meta.textContent='switch failed'; modelsel.value=activeModel;
      const b=turn('flwr','flwr'); b.textContent='could not load '+name+' — '+(r.error||'unknown'); }
  }catch(e){ meta.textContent='switch failed'; modelsel.value=activeModel;
    const b=turn('flwr','flwr'); b.textContent='could not switch models — '+e; }
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
  setSystem(sysBox.value);
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

// ---- system instruction (a role that persists for this conversation) ----
const sysBox=document.getElementById('sys'), sysToggle=document.getElementById('systoggle');
function setSystem(text){
  text=(text||'').trim();
  const hasSys=messages[0]&&messages[0].role==='system';
  if(text){ if(hasSys) messages[0].content=text; else messages.unshift({role:'system',content:text}); }
  else if(hasSys){ messages.shift(); }
}
function syncSystemBox(){ const s=messages.find(m=>m.role==='system'); sysBox.value=s?s.content:''; if(s){ sysBox.classList.add('show'); sysToggle.classList.add('on'); } }
sysToggle.onclick=()=>{ const on=sysBox.classList.toggle('show'); sysToggle.classList.toggle('on',on); if(on) sysBox.focus(); };
sysBox.addEventListener('input',()=>setSystem(sysBox.value));

// ---- saved chats (provenance-bearing transcripts on the server) ----
function newId(){return 'c'+Date.now().toString(36)+Math.random().toString(36).slice(2,6);}
function fmtAgo(u){ if(!u) return ''; let s=Math.floor(Date.now()/1000)-u;
  if(s<60)return s+'s'; if(s<3600)return Math.floor(s/60)+'m';
  if(s<86400)return Math.floor(s/3600)+'h'; return Math.floor(s/86400)+'d'; }
function renderMessages(){
  log.innerHTML='';
  const visible=messages.filter(m=>m.role!=='system');
  if(!visible.length){ log.innerHTML='<div class="empty">Ask it something below.</div>'; return; }
  for(const m of visible){ turn(m.role==='user'?'you':'flwr', m.role==='user'?'You':'flwr').textContent=m.content; }
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
    chatId=id; syncSystemBox(); renderMessages(); refreshChats(); window.scrollTo(0,document.body.scrollHeight);
  }catch(e){}
}
async function delChat(id){
  try{ await fetch('/chats/'+encodeURIComponent(id),{method:'DELETE'}); }catch(e){}
  if(id===chatId){ newChat(); } else { refreshChats(); }
}
function newChat(){ chatId=newId(); messages.length=0; setSystem(sysBox.value); renderMessages(); refreshChats(); input.focus(); }
async function saveCurrent(){
  if(!messages.some(m=>m.role!=='system')) return;
  try{ await fetch('/chats',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({id:chatId,messages})}); refreshChats(); }catch(e){}
}
document.getElementById('newchat').onclick=newChat;
refreshChats();

// ---- bottom activity meter: poll /monitor and render a tiny live strip ----
const meterEl=document.getElementById('meter');
async function pollMeter(){
  try{
    const m=await (await fetch('/monitor')).json();
    const sp=m.last_tok_s>0?(m.last_tok_s.toFixed(1)+' tok/s'):'idle';
    const up=m.uptime_s<60?m.uptime_s+'s':Math.floor(m.uptime_s/60)+'m';
    meterEl.innerHTML='❀ '+(m.model||'model')
      +' <span class="sep">·</span> '+sp
      +' <span class="sep">·</span> '+(m.memory_mb||0)+' MB'
      +' <span class="sep">·</span> '+(m.requests||0)+' req'
      +' <span class="sep">·</span> up '+up;
  }catch(e){ meterEl.textContent='❀ offline'; }
}
pollMeter(); setInterval(pollMeter, 2000);
</script>
</body>
</html>"##;
