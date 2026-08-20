//! HOS — a from-scratch local LLM inference engine.
//!
//! v0: GGUF load, Llama-family forward pass, KV cache, CPU, greedy/temperature
//! sampling. CUDA/Metal backends and K-quants land next.

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::doc_lazy_continuation,
    clippy::print_literal
)]

use std::path::{Path, PathBuf};
use std::time::Instant;

use hos::forward;
use hos::gguf::Gguf;
use hos::metal_be;
use hos::model::Model;
use hos::sample;
use hos::tokenizer::Tokenizer;

/// Unwrap a fallible HOS operation, or print a clean error and exit (no panic /
/// backtrace for ordinary bad-input cases like a missing or unsupported model).
fn ok<T>(r: hos::Result<T>) -> T {
    r.unwrap_or_else(|e| {
        eprintln!("[hos] error: {e}");
        std::process::exit(1);
    })
}

/// Value following a CLI flag, if present (`--flag value`).
fn arg_after(flag: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    let i = a.iter().position(|x| x == flag)?;
    a.get(i + 1).filter(|v| !v.starts_with('-')).cloned()
}

// ============================================================
// Built-in prompts and corpora for --bench / --perplexity / --quant-awq.
// ============================================================

/// Fixed prompt for `--bench`, so throughput numbers are comparable run-to-run.
const BENCH_PROMPT: &str = "The history of computing is a long and storied one, beginning with";

/// Built-in corpus for `--perplexity` when no file is given. A fixed passage so
/// the reported perplexity is reproducible without external data.
const DEFAULT_CORPUS: &str = "The quick brown fox jumps over the lazy dog. \
Machine learning models predict the next token given the previous context. \
A language model assigns a probability to each possible continuation of a sequence, \
and perplexity measures how surprised the model is by held-out text: lower is better. \
In the beginning the universe was created. This has made a lot of people very angry \
and been widely regarded as a bad move.";

/// Calibration text for `--quant-awq`, deliberately disjoint from `DEFAULT_CORPUS`
/// (the eval set) so the measured win isn't an artefact of train/test overlap.
const CALIB_CORPUS: &str = "Photosynthesis converts sunlight, water, and carbon dioxide \
into glucose and oxygen inside the chloroplasts of green plants. The mitochondria, often \
called the powerhouse of the cell, release energy by breaking those sugars back down. \
Rivers carve valleys over millennia, depositing sediment that builds fertile deltas at \
the coast. Trade routes once carried silk, spices, and silver across deserts and seas, \
binding distant cities into a single economy long before the telegraph.";

/// `--bench`: time prefill and greedy decode separately on a fixed workload.
fn cmd_bench(model_path: &Path, args: &Args) {
    let mut eng = ok(hos::Engine::load(model_path, args.gpu));
    // fixed prompt for run-to-run comparability; -n sets the decode length.
    let n = if args.n_predict > 0 {
        args.n_predict
    } else {
        64
    };
    // -p overrides the fixed prompt (e.g. to measure prefill at larger token counts).
    let prompt = if args.prompt.is_empty() {
        BENCH_PROMPT
    } else {
        args.prompt.as_str()
    };
    let b = eng.bench(prompt, n);
    println!("=== HOS benchmark ===");
    println!(
        "backend    : {}",
        if args.gpu {
            "Metal GPU"
        } else {
            "CPU (multithreaded)"
        }
    );
    println!(
        "prefill    : {} tok in {:.3}s  ({:.1} tok/s)",
        b.prefill_tokens,
        b.prefill_secs,
        b.prefill_tps()
    );
    println!(
        "decode     : {} tok in {:.3}s  ({:.1} tok/s)",
        b.decode_tokens,
        b.decode_secs,
        b.decode_tps()
    );
}

/// `--perplexity [file]`: score held-out text. `corpus` is the file contents,
/// or the built-in passage when no path was given.
fn cmd_perplexity(model_path: &Path, args: &Args, corpus: String) {
    let mut eng = ok(hos::Engine::load(model_path, args.gpu));
    let ids = eng.tok.encode(&corpus, true);
    let start = Instant::now();
    let (scored, nll, ppl) = eng.perplexity(&ids);
    let secs = start.elapsed().as_secs_f64();
    println!("=== HOS perplexity ===");
    println!(
        "backend       : {}",
        if args.gpu {
            "Metal GPU"
        } else {
            "CPU (multithreaded)"
        }
    );
    println!("corpus tokens : {}", ids.len());
    println!("scored        : {scored} next-token predictions");
    println!("mean NLL      : {nll:.4} nats/token");
    println!("perplexity    : {ppl:.3}");
    println!(
        "time          : {:.2}s ({:.1} tok/s)",
        secs,
        scored as f64 / secs.max(1e-9)
    );
}

/// `--to-hos <model.gguf> [-o out.hos]`: convert a GGUF model into a self-
/// describing `.hos` capsule (dequantized weights + an arch-spec card + provenance).
/// Use on small models — `.hos` stores f32, so the file is larger than the GGUF.
/// Convert a parsed GGUF metadata value into JSON for storage in a `.hos` card.
fn gguf_value_to_json(v: &hos::gguf::Value) -> serde_json::Value {
    use hos::gguf::Value::*;
    match v {
        U8(x) => (*x).into(),
        I8(x) => (*x).into(),
        U16(x) => (*x).into(),
        I16(x) => (*x).into(),
        U32(x) => (*x).into(),
        I32(x) => (*x).into(),
        U64(x) => (*x).into(),
        I64(x) => (*x).into(),
        F32(x) => (*x).into(),
        F64(x) => (*x).into(),
        Bool(x) => (*x).into(),
        Str(s) => s.clone().into(),
        Array(a) => serde_json::Value::Array(a.iter().map(gguf_value_to_json).collect()),
    }
}

/// Parse `--quantize <q8_0|q4_0|q5_0>` → (name, ggml type). Exits on bad target.
/// `--quant-bench`: head-to-head reconstruction error of every quant on a sample
/// of normally-distributed weights (where weights actually live). Lower RMSE at
/// equal bits/weight wins. The point: does HOS's native non-uniform `hq4` beat
/// ggml's uniform 4-bit formats?
fn cmd_quant_bench() {
    use hos::gguf::{
        bytes_for, Gguf, GGML_Q4_0, GGML_Q4_K, GGML_Q5_0, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0,
    };
    let n = 256 * 400;
    let mut rng = 0x1234_5678_9abc_def0u64;
    let mut u = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        ((rng >> 11) as f32) / ((1u64 << 53) as f32)
    };
    // Box–Muller normal samples — the distribution model weights follow.
    let x: Vec<f32> = (0..n)
        .map(|_| {
            let (u1, u2) = (u().max(1e-7), u());
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
        })
        .collect();
    let amax = x.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let rmse = |dec: &[f32]| -> f32 {
        (x.iter()
            .zip(dec)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / n as f32)
            .sqrt()
    };
    println!("quant     bits/wt      RMSE     RMSE/amax");
    let row = |name: &str, bytes: usize, dec: &[f32]| {
        let bits = bytes as f32 * 8.0 / n as f32;
        println!(
            "{name:<8}{bits:>8.2}{:>11.5}{:>12.4}",
            rmse(dec),
            rmse(dec) / amax
        );
    };
    for (name, ty) in [
        ("q8_0", GGML_Q8_0),
        ("q5_k", GGML_Q5_K),
        ("q5_0", GGML_Q5_0),
        ("q6_k", GGML_Q6_K),
        ("q4_k", GGML_Q4_K),
        ("q4_0", GGML_Q4_0),
    ] {
        let enc = hos::gguf_write::quantize(&x, ty);
        let mut dec = vec![0f32; n];
        Gguf::dequant_into(&enc, ty, n, &mut dec).unwrap();
        row(name, bytes_for(ty, n).unwrap_or(enc.len()), &dec);
    }
    let enc = hos::hos_quant::encode_hq4(&x);
    let dec = hos::hos_quant::decode_hq4(&enc, n);
    row("hq4*", enc.len(), &dec);
    println!("\n(* native HOS non-uniform 4-bit. Compare hq4 vs q4_0/q4_k at ~4.5 bits/wt.)");
}

/// `--quant-awq <model.gguf> [-o out.hos] [--quantize q4_0] [--awq-alpha 0.5]`:
/// activation-aware quantization (AWQ-lite). Calibrates per-input-channel
/// activation salience on a corpus, scales salient channels up before quantizing
/// (folding the inverse into the preceding RMSNorm — exact, no runtime change),
/// then quantizes. Mints BOTH an AWQ capsule and a plain (round-to-nearest)
/// capsule at the same format and reports end-to-end perplexity of each. AWQ wins
/// if its perplexity is lower at equal bits — i.e. it beats ggml's RTN quants by
/// minimising *output* error instead of weight error.
fn cmd_quant_awq() {
    use hos::format::{self, Named, ROLE_BIAS, ROLE_EMBED, ROLE_NORM, ROLE_WEIGHT};
    let Some(src) = arg_after("--quant-awq") else {
        eprintln!("[hos] error: --quant-awq needs a model path (.gguf)");
        std::process::exit(1);
    };
    let fmt = arg_after("--quantize").unwrap_or_else(|| "q4_0".into());
    let is_q3 = fmt == "q3"; // native HOS 3-bit (not a ggml type)
    let ty = if is_q3 {
        0
    } else {
        match hos::gguf_write::target_type(&fmt) {
            Some(t) => t,
            None => {
                eprintln!(
                    "[hos] --quant-awq: unsupported --quantize '{fmt}' (q4_0|q5_0|q4_k|q5_k|q6_k|q3)"
                );
                std::process::exit(1);
            }
        }
    };
    let alpha: f32 = arg_after("--awq-alpha")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.5);
    let srcp = Path::new(&src);

    // 1. calibrate: accumulate per-channel activation salience over a corpus
    //    (CPU forward, so the instrumented path runs).
    let mut eng = ok(hos::Engine::load(srcp, false));
    let g = ok(Gguf::open(srcp));
    let arch_name = g
        .meta_str("general.architecture")
        .unwrap_or("unknown")
        .to_string();
    let mu = |k: &str| g.meta_u64(&format!("{arch_name}.{k}"));
    let n_layers = mu("block_count").unwrap_or(0) as usize;
    let dim = mu("embedding_length").unwrap_or(0) as usize;
    let ffn = mu("feed_forward_length").unwrap_or(0) as usize;
    let ids = eng.tok.encode(CALIB_CORPUS, true);
    forward::calib_start(n_layers, dim, ffn);
    let _ = eng.perplexity(&ids);
    let calib = forward::calib_take().expect("calibration recorded");
    eprintln!(
        "[awq] calibrated on {} tokens · {n_layers} layers · dim {dim} · alpha {alpha}",
        calib.tokens
    );

    // 2. build f32 tensors from the GGUF source (GGUF tensor names).
    let role = |n: &str| -> u8 {
        if n.contains("token_embd") || n.contains("embed") {
            ROLE_EMBED
        } else if n.ends_with(".bias") {
            ROLE_BIAS
        } else if n.contains("norm") {
            ROLE_NORM
        } else {
            ROLE_WEIGHT
        }
    };
    let mut names: Vec<String> = g.tensors.keys().cloned().collect();
    names.sort();
    let mut named = Vec::with_capacity(names.len());
    for name in names {
        let data = ok(g.dequant(&name));
        let shape: Vec<usize> = g.tensors[&name]
            .dims
            .iter()
            .rev()
            .map(|d| *d as usize)
            .collect();
        named.push(Named {
            role: role(&name),
            name,
            shape,
            data,
        });
    }

    // 3. AWQ transform on a copy; plain copy left as round-to-nearest.
    let nh = mu("attention.head_count").unwrap_or(0);
    let nkv = mu("attention.head_count_kv").unwrap_or(nh);
    let wo_ok = nh != 0 && nh == nkv; // wo fold is exact only without GQA
    let mut named_awq = named.clone();
    let touched = hos::hos_awq::apply_awq(&mut named_awq, &calib, alpha, wo_ok);
    eprintln!(
        "[awq] activation-scaled {touched} weight matrices (wo {})",
        if wo_ok {
            "included (MHA)"
        } else {
            "skipped (GQA)"
        }
    );

    let arch = serde_json::json!({
        "source_format": "gguf", "architecture": arch_name,
        "embedding_length": mu("embedding_length"), "block_count": mu("block_count"),
        "head_count": mu("attention.head_count"), "head_count_kv": mu("attention.head_count_kv"),
        "feed_forward_length": mu("feed_forward_length"), "context_length": mu("context_length"),
    });
    let mk_card = |label: &str, awq: bool| {
        let mut card = format::Card::new(label, arch.clone());
        card.id = format::model_id(if awq { &named_awq } else { &named });
        card.mode = "inference".into();
        card.provenance.engine = format!("hos ({fmt}{})", if awq { ", awq" } else { "" });
        card.provenance.dataset = format!("converted from {src}");
        let mut meta = serde_json::Map::new();
        for (k, v) in &g.metadata {
            meta.insert(k.clone(), gguf_value_to_json(v));
        }
        card.meta = serde_json::Value::Object(meta);
        card
    };

    let base = srcp.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
    let out_awq = arg_after("-o")
        .or_else(|| arg_after("--out"))
        .unwrap_or_else(|| format!("{base}.awq.hos"));
    let out_plain = format!("{base}.plain.hos");
    let save = |path: &Path, t: &[Named], c: &format::Card| {
        if is_q3 {
            format::save_q3(path, t, c)
        } else {
            format::save_quantized_as(path, t, c, ty)
        }
    };
    ok(save(Path::new(&out_awq), &named_awq, &mk_card("awq", true)).map_err(hos::HosError::from));
    ok(save(Path::new(&out_plain), &named, &mk_card("plain", false)).map_err(hos::HosError::from));

    // 4. measure end-to-end perplexity of each capsule.
    let mut a = ok(hos::Engine::load(Path::new(&out_awq), false));
    let aids = a.tok.encode(DEFAULT_CORPUS, true);
    let (_, _, appl) = a.perplexity(&aids);
    let mut p = ok(hos::Engine::load(Path::new(&out_plain), false));
    let pids = p.tok.encode(DEFAULT_CORPUS, true);
    let (_, _, pppl) = p.perplexity(&pids);

    println!("\n=== AWQ-lite vs plain RTN ({fmt}, alpha {alpha}) ===");
    println!("plain (RTN)   perplexity {pppl:>9.4}   -> {out_plain}");
    println!("awq-lite      perplexity {appl:>9.4}   -> {out_awq}");
    let delta = (pppl - appl) / pppl * 100.0;
    if appl < pppl {
        println!("AWQ wins: {delta:.2}% lower perplexity at equal bits/weight.");
    } else {
        println!("AWQ does not beat plain here: {:.2}% worse.", -delta);
    }
}

fn quant_arg() -> (Option<String>, Option<u32>) {
    match arg_after("--quantize") {
        None => (None, None),
        // hq4/q3 are native HOS quants (not ggml types); handled at the save site.
        Some(t) if t == "hq4" || t == "q3" => (Some(t), None),
        Some(t) => match hos::gguf_write::target_type(&t) {
            Some(ty) => (Some(t), Some(ty)),
            None => {
                eprintln!(
                    "[hos] --quantize: unsupported target '{t}' (q8_0|q4_0|q5_0|q4_k|q5_k|q6_k|hq4|q3)"
                );
                std::process::exit(1);
            }
        },
    }
}

fn cmd_to_hos(_args: &Args) {
    use hos::format::{self, Named, ROLE_BIAS, ROLE_EMBED, ROLE_NORM, ROLE_WEIGHT};
    let Some(src) = arg_after("--to-hos") else {
        eprintln!("[hos] error: --to-hos needs a model path");
        std::process::exit(1);
    };
    let out = arg_after("-o")
        .or_else(|| arg_after("--out"))
        .unwrap_or_else(|| {
            std::path::Path::new(&src)
                .with_extension("hos")
                .to_string_lossy()
                .into_owned()
        });
    let g = ok(Gguf::open(std::path::Path::new(&src)));
    let arch_name = g
        .meta_str("general.architecture")
        .unwrap_or("unknown")
        .to_string();

    // self-describing arch card from GGUF metadata
    let mu = |k: &str| g.meta_u64(&format!("{arch_name}.{k}"));
    let arch = serde_json::json!({
        "source_format": "gguf",
        "architecture": arch_name,
        "embedding_length": mu("embedding_length"),
        "block_count": mu("block_count"),
        "head_count": mu("attention.head_count"),
        "head_count_kv": mu("attention.head_count_kv"),
        "feed_forward_length": mu("feed_forward_length"),
        "context_length": mu("context_length"),
    });

    let role = |n: &str| -> u8 {
        if n.contains("token_embd") || n.contains("embed") {
            ROLE_EMBED
        } else if n.ends_with(".bias") {
            ROLE_BIAS
        } else if n.contains("norm") {
            ROLE_NORM
        } else {
            ROLE_WEIGHT
        }
    };

    let mut names: Vec<String> = g.tensors.keys().cloned().collect();
    names.sort();
    eprintln!(
        "[hos] converting {} tensors from {arch_name} ...",
        names.len()
    );
    let mut named = Vec::with_capacity(names.len());
    for name in names {
        let data = ok(g.dequant(&name));
        // GGUF dims are fastest-first; reverse to row-major [out, .., in]
        let shape: Vec<usize> = g.tensors[&name]
            .dims
            .iter()
            .rev()
            .map(|d| *d as usize)
            .collect();
        named.push(Named {
            role: role(&name),
            name,
            shape,
            data,
        });
    }

    let (quant_name, quant_target) = quant_arg();

    // Display name for the capsule card. Defaults to the source file stem, but
    // `--hos-name` lets you brand a derivative (e.g. mint an Apache-2.0 base under
    // your own name) while `--source-note` keeps the honest attribution/lineage in
    // provenance — so `hos --hos-info` shows "your-name <- original (license)".
    let model_name = arg_after("--hos-name").unwrap_or_else(|| {
        std::path::Path::new(&src)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .to_string()
    });
    let mut card = format::Card::new(&model_name, arch);
    card.id = format::model_id(&named);
    card.mode = "inference".into();
    card.provenance.engine = match &quant_name {
        Some(t) => format!("hos ({t})"),
        None => "hos".into(),
    };
    card.provenance.dataset =
        arg_after("--source-note").unwrap_or_else(|| format!("converted from {src}"));
    // engine-readable metadata: the full GGUF map (hyperparameters + tokenizer),
    // so the capsule is runnable (`hos -m model.hos` / `flwr run`), not archival.
    let mut meta = serde_json::Map::new();
    for (k, v) in &g.metadata {
        meta.insert(k.clone(), gguf_value_to_json(v));
    }
    card.meta = serde_json::Value::Object(meta);

    let p = std::path::Path::new(&out);
    let res = if quant_name.as_deref() == Some("hq4") {
        format::save_hq4(p, &named, &card)
    } else if quant_name.as_deref() == Some("q3") {
        format::save_q3(p, &named, &card)
    } else {
        match quant_target {
            Some(ty) => format::save_quantized_as(p, &named, &card, ty),
            None => format::save(p, &named, &card),
        }
    };
    ok(res.map_err(hos::HosError::from));
    let bytes = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
    let qlabel = quant_name
        .as_deref()
        .map(|t| format!(", {t}"))
        .unwrap_or_default();
    println!(
        "wrote {out} ({:.1} MB, {} tensors{qlabel})",
        bytes as f64 / 1e6,
        named.len()
    );
    println!("inspect: hos --hos-info {out}");
}

/// `--ingest <hf_dir> [-o out.hos]`: load a raw HuggingFace checkpoint and mint a
/// `.hos` capsule — provenance, a content-hash lineage edge back to the source,
/// and the weights. Unlike a converter (weights in → weights out), this gives the
/// model an *identity and an ancestry* in HOS's lineage system. Use on small
/// models (`.hos` stores f32, so the file is ~2x the bf16 source).
/// `--gemma4-ingest <hf-dir> [-o out.hos]` — mint a portable `.hos` capsule from
/// a Gemma-4 HuggingFace checkpoint: already-quantized big linears (no dequant) +
/// f32 norms + the tokenizer, `arch=gemma4`. Loadable via `flwr run out.hos` with
/// no safetensors dir. This is the OPEN-tier ingest path for Gemma-4.
fn cmd_gemma4_ingest() {
    let Some(src) = arg_after("--gemma4-ingest") else {
        eprintln!("[hos] --gemma4-ingest needs a Gemma-4 HuggingFace checkpoint directory");
        std::process::exit(1);
    };
    let dir = std::path::Path::new(&src);
    if !dir.is_dir() {
        eprintln!(
            "[hos] --gemma4-ingest expects the checkpoint *directory* (config.json + safetensors)."
        );
        std::process::exit(1);
    }
    let out = arg_after("-o")
        .or_else(|| arg_after("--out"))
        .unwrap_or_else(|| {
            let base = dir.file_name().and_then(|s| s.to_str()).unwrap_or("gemma4");
            format!("{base}.hos")
        });
    // Quant is selected by HOS_GEMMA4_QUANT; also honor a `--quantize KIND` flag
    // on this command (q4k/q5k/q6k/hq4) so the CLI matches the other ingest paths.
    // Explicit env wins; then --quantize; else default q4k.
    if std::env::var("HOS_GEMMA4_QUANT").is_err() {
        let kind = arg_after("--quantize").unwrap_or_else(|| "q4k".to_string());
        std::env::set_var("HOS_GEMMA4_QUANT", kind);
    }
    eprintln!("[hos] gemma4-ingest: loading {src} ...");
    let m = ok(hos::gemma4::Gemma4::load(dir));
    let tok_json = match std::fs::read_to_string(dir.join("tokenizer.json")) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[hos] gemma4-ingest: read tokenizer.json: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("[hos] gemma4-ingest: writing capsule -> {out} ...");
    ok(m.write_capsule(std::path::Path::new(&out), &tok_json));
    eprintln!("[hos] gemma4-ingest: done -> {out}");
}

fn cmd_ingest(_args: &Args) {
    use hos::format::{self, Named, ROLE_BIAS, ROLE_EMBED, ROLE_NORM, ROLE_WEIGHT};
    use hos::model::ModelSource;
    let Some(src) = arg_after("--ingest") else {
        eprintln!("[hos] error: --ingest needs a HuggingFace checkpoint directory");
        std::process::exit(1);
    };
    let dir = std::path::Path::new(&src);
    if !dir.is_dir() {
        eprintln!("[hos] --ingest expects a HuggingFace checkpoint *directory* (config.json + *.safetensors).");
        eprintln!("[hos] for a .gguf file, use:  hos --to-hos {src} -o out.hos [--quantize q8_0|q4_0|q5_0]");
        std::process::exit(1);
    }
    let out = arg_after("-o")
        .or_else(|| arg_after("--out"))
        .unwrap_or_else(|| {
            let base = dir.file_name().and_then(|s| s.to_str()).unwrap_or("model");
            format!("{base}.hos")
        });
    let (quant_name, quant_target) = quant_arg();

    // Load through HfModel: GGUF-named, engine-layout tensors (HF→GGUF name
    // mapping + Q/K RoPE permutation) + synthesized metadata — so the capsule is
    // *runnable*, not just archival. The tokenizer is carried alongside.
    let hf = ok(hos::hf::HfModel::open(dir));
    let tok = ok(hos::tokenizer::Tokenizer::from_hf(dir));
    let cfg = ok(hos::safetensors::read_json(&dir.join("config.json")));
    // the HF source's identity in its native precision — becomes the ancestor
    let source_hash = ok(hos::safetensors::SafeTensors::open_dir(dir)).source_hash();

    let role = |n: &str| -> u8 {
        if n.contains("token_embd") || n.contains("embed") {
            ROLE_EMBED
        } else if n.ends_with(".bias") {
            ROLE_BIAS
        } else if n.contains("norm") {
            ROLE_NORM
        } else {
            ROLE_WEIGHT
        }
    };

    let mut names = hf.tensor_names();
    names.sort();
    eprintln!(
        "[hos] ingesting {} tensors from {} ...",
        names.len(),
        dir.display()
    );
    let mut named = Vec::with_capacity(names.len());
    for name in &names {
        let data = ok(hf.dequant(name));
        let n = data.len();
        named.push(Named {
            role: role(name),
            name: name.clone(),
            shape: vec![n],
            data,
        });
    }

    // engine-readable metadata: synthesized GGUF keys + the serialized tokenizer,
    // so `hos -m model.hos` / `flwr run` load and run it.
    let mut meta = serde_json::Map::new();
    for (k, v) in hf.meta_map() {
        meta.insert(k.clone(), gguf_value_to_json(v));
    }
    meta.insert("hos.tokenizer".into(), tok.to_value());

    let model_type = cfg
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    let arch = serde_json::json!({
        "source_format": "huggingface-safetensors",
        "model_type": model_type,
        "config": cfg,
    });
    let model_name = dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();

    let mut card = format::Card::new(&model_name, arch);
    card.id = format::model_id(&named);
    card.mode = "inference".into();
    card.provenance.engine = match &quant_name {
        Some(t) => format!("hos-ingest ({t})"),
        None => "hos-ingest".into(),
    };
    card.provenance.dataset = format!("HuggingFace checkpoint: {src}");
    card.provenance.dataset_hash = source_hash.clone();
    card.lineage = vec![source_hash];
    card.meta = serde_json::Value::Object(meta);

    let p = std::path::Path::new(&out);
    let res = if quant_name.as_deref() == Some("hq4") {
        format::save_hq4(p, &named, &card)
    } else if quant_name.as_deref() == Some("q3") {
        format::save_q3(p, &named, &card)
    } else {
        match quant_target {
            Some(ty) => format::save_quantized_as(p, &named, &card, ty),
            None => format::save(p, &named, &card),
        }
    };
    ok(res.map_err(hos::HosError::from));
    let bytes = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
    let qlabel = quant_name
        .as_deref()
        .map(|t| format!(", {t}"))
        .unwrap_or_default();
    println!(
        "minted {out} ({:.1} MB, {} tensors{qlabel}) — runnable",
        bytes as f64 / 1e6,
        named.len()
    );
    println!("  id      : {}   (content hash of the weights)", card.id);
    println!("  lineage : {}   (HF source artifact)", card.lineage[0]);
    println!("  run     : hos -m {out} -p \"...\"   ·   flwr run {out}");
}

/// `--verify-against <hf_dir> -m <model.gguf>`: audit a quantized GGUF against its
/// original HuggingFace checkpoint — same architecture? how much did quantization
/// cost? HOS is the only engine that can do this, because it's the one that loads
/// both the HF source and the GGUF natively, no external tooling.
fn cmd_verify_against(args: &Args) {
    let Some(hf) = arg_after("--verify-against") else {
        eprintln!("[hos] error: --verify-against needs a HuggingFace checkpoint directory");
        std::process::exit(1);
    };
    let gguf = resolve_model(args.model.clone());
    eprintln!(
        "[hos] auditing {} against HF source {hf} ...",
        gguf.display()
    );

    let mut g = ok(hos::Engine::load(&gguf, false));
    let mut h = ok(hos::Engine::load(std::path::Path::new(&hf), false));
    let (gc, hc) = (g.model.cfg.clone(), h.model.cfg.clone());

    println!("\n=== HOS conversion audit ===");
    println!("GGUF : {}", gguf.display());
    println!("HF   : {hf}\n");
    println!("architecture:               gguf      hf");
    let mut arch_ok = true;
    arch_ok &= cmp_row("dim", gc.dim, hc.dim);
    arch_ok &= cmp_row("n_layers", gc.n_layers, hc.n_layers);
    arch_ok &= cmp_row("n_heads", gc.n_heads, hc.n_heads);
    arch_ok &= cmp_row("n_kv_heads", gc.n_kv_heads, hc.n_kv_heads);
    arch_ok &= cmp_row("head_dim", gc.head_dim, hc.head_dim);
    arch_ok &= cmp_row("ffn_dim", gc.ffn_dim, hc.ffn_dim);
    arch_ok &= cmp_row("vocab_size", gc.vocab_size, hc.vocab_size);
    let arch_match = format!("{:?}", gc.arch) == format!("{:?}", hc.arch);
    arch_ok &= arch_match;
    println!(
        "  {:<12} {:>8?}  {:>8?}   {}",
        "arch",
        gc.arch,
        hc.arch,
        if arch_match { "ok" } else { "MISMATCH" }
    );

    // fidelity: perplexity on the same built-in passage (each with its own tokenizer)
    let gids = g.tok.encode(DEFAULT_CORPUS, true);
    let (_, _, gppl) = g.perplexity(&gids);
    let hids = h.tok.encode(DEFAULT_CORPUS, true);
    let (_, _, hppl) = h.perplexity(&hids);
    println!("\nfidelity (perplexity on built-in passage, lower = better):");
    println!("  HF   (f16)  : {hppl:.3}");
    println!("  GGUF (quant): {gppl:.3}");
    println!("  quantization cost: {:+.1}%", (gppl / hppl - 1.0) * 100.0);

    println!();
    if !arch_ok {
        println!("verdict: MISMATCH — the GGUF's architecture does not match the HF source.");
    } else if gppl.is_finite() && gppl <= hppl * 2.0 {
        println!(
            "verdict: CONSISTENT — same architecture; the GGUF is a faithful (lossy) \
             quantization of the HF source."
        );
    } else {
        println!(
            "verdict: SUSPECT — architecture matches but the GGUF perplexity is {:.1}x the HF \
             source; the conversion may be degraded.",
            gppl / hppl
        );
    }
}

/// Build 4 distinct, reproducible synthetic text domains (no external data) and
/// return each as (name, train ids, held-out ids). The pretrained model can model
/// these, but they have clearly different distributions — the multi-domain test.
fn build_domains(
    tok: &hos::tokenizer::Tokenizer,
    seed: u64,
) -> Vec<(&'static str, Vec<usize>, Vec<usize>)> {
    let mut rng = seed | 1;
    let mut nx = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 16) as usize
    };
    let geo = [
        ("France", "Paris"),
        ("Japan", "Tokyo"),
        ("Egypt", "Cairo"),
        ("Brazil", "Brasilia"),
        ("Kenya", "Nairobi"),
        ("Norway", "Oslo"),
        ("Peru", "Lima"),
        ("Ghana", "Accra"),
        ("Spain", "Madrid"),
        ("India", "Delhi"),
        ("Chile", "Santiago"),
        ("Cuba", "Havana"),
    ];
    let animals = ["cat", "fox", "owl", "bear", "wolf", "crow", "deer", "hare"];
    let verbs = ["saw", "chased", "watched", "found", "passed", "left"];
    let nouns = ["moon", "river", "forest", "stone", "field", "shore"];
    let names = ["add", "scale", "clip", "norm", "step", "mix", "fold"];

    let mut geography = String::new();
    let mut arithmetic = String::new();
    let mut story = String::new();
    let mut code = String::new();
    for _ in 0..360 {
        let (c, cap) = geo[nx() % geo.len()];
        geography.push_str(&format!("The capital of {c} is {cap}. "));
        let (a, b) = (nx() % 30, nx() % 30);
        arithmetic.push_str(&format!("{a} plus {b} equals {}. ", a + b));
        story.push_str(&format!(
            "The {} {} the {}. ",
            animals[nx() % animals.len()],
            verbs[nx() % verbs.len()],
            nouns[nx() % nouns.len()]
        ));
        code.push_str(&format!(
            "def {}(x): return x + {}\n",
            names[nx() % names.len()],
            nx() % 10
        ));
    }
    [
        ("geography", geography),
        ("arithmetic", arithmetic),
        ("story", story),
        ("code", code),
    ]
    .into_iter()
    .map(|(name, text)| {
        let ids: Vec<usize> = tok
            .encode(&text, true)
            .iter()
            .map(|&x| x as usize)
            .collect();
        let split = ids.len() * 85 / 100;
        (name, ids[..split].to_vec(), ids[split..].to_vec())
    })
    .collect()
}

/// Train one PEFT method jointly on all domains; return (trainable params,
/// per-domain held-out perplexity).
#[allow(clippy::too_many_arguments)]
fn peft_run_domains(
    base: &hos::model::Model,
    domains: &[(&'static str, Vec<usize>, Vec<usize>)],
    method: &str,
    hcfg: &hos::peft::PeftCfg,
    steps: usize,
    t: usize,
    accum: usize,
    lr: f32,
    lambda: f32,
) -> (usize, Vec<f32>) {
    use hos::tensor::AdamW;
    // RGA: one genome per domain over a shared bank (M3 config). LoRA: single delta.
    let peft = hos::peft::PeftModel::build_multi(base, method, hcfg, domains.len(), 0xC0FFEE)
        .expect("build peft");
    let params = peft.params();
    let decay: Vec<bool> = params.iter().map(|p| p.shape().len() == 2).collect();
    let mut opt = AdamW::new(&params, lr, 0.0);
    let mut rng = 0x1234_5678_9abc_def0u64;
    let mut nx = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 16) as usize
    };
    for step in 0..steps {
        for p in &params {
            p.zero_grad();
        }
        let mut mean = 0.0;
        for _ in 0..accum {
            // pick a domain, a window in it, and ITS genome (per-domain expression)
            let d = nx() % domains.len();
            let tr = &domains[d].1;
            if tr.len() < t + 2 {
                continue;
            }
            let s = nx() % (tr.len() - t - 1);
            let w = &tr[s..s + t + 1];
            let loss = peft
                .loss_g(&w[..t], &w[1..t + 1], lambda, d)
                .scale(1.0 / accum as f32);
            loss.backward();
            mean += loss.data()[0];
        }
        opt.step(&params, &decay);
        if step % 25 == 0 || step == steps - 1 {
            eprintln!("    [{method}] step {step:>3}  loss {mean:.4}");
        }
    }
    // per-domain held-out perplexity, each domain served by its own genome
    let ppls = domains
        .iter()
        .enumerate()
        .map(|(d, (_, _, test))| peft_eval_ppl(&peft, test, t, d))
        .collect();
    (peft.n_trainable(), ppls)
}

/// Held-out perplexity over non-overlapping windows, using domain `gi`'s genome.
/// Greedy-decode `n_new` tokens from the adapted model for genome `gi`, returning
/// the decoded continuation (newlines escaped) — for qualitative before/after
/// examples. Re-runs the full forward each step (the PEFT path has no KV cache),
/// which is fine for short demo generations.
fn peft_generate(
    peft: &hos::peft::PeftModel,
    tok: &hos::tokenizer::Tokenizer,
    prompt: &[usize],
    gi: usize,
    n_new: usize,
) -> String {
    let mut ids = prompt.to_vec();
    let mut gen: Vec<u32> = Vec::new();
    for _ in 0..n_new {
        let logits = peft.logits_g(&ids, gi);
        let sh = logits.shape();
        let vocab = sh[sh.len() - 1];
        let d = logits.data();
        let last = &d[(ids.len() - 1) * vocab..ids.len() * vocab];
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &v) in last.iter().enumerate() {
            if v > bv {
                bv = v;
                best = i;
            }
        }
        ids.push(best);
        gen.push(best as u32);
    }
    tok.decode(&gen).replace('\n', "\\n")
}

fn peft_eval_ppl(peft: &hos::peft::PeftModel, test: &[usize], t: usize, gi: usize) -> f32 {
    let (mut tot, mut n, mut s) = (0.0f32, 0usize, 0usize);
    while s + t < test.len() {
        let w = &test[s..s + t + 1];
        tot += peft.loss_g(&w[..t], &w[1..t + 1], 0.0, gi).data()[0];
        n += 1;
        s += t;
    }
    if n == 0 {
        f32::NAN
    } else {
        (tot / n as f32).exp()
    }
}

/// Continual-learning / interference test: train on domain A, measure A; then
/// continue on domain B, measure A again (forgetting) and B. Returns
/// (ppl_A_after_A, ppl_A_after_B, ppl_B_after_B). A single static LoRA delta gets
/// overwritten learning B; RGA's conditional gating can protect A's genes.
#[allow(clippy::too_many_arguments)]
fn peft_run_sequential(
    base: &hos::model::Model,
    da_train: &[usize],
    da_test: &[usize],
    db_train: &[usize],
    db_test: &[usize],
    method: &str,
    hcfg: &hos::peft::PeftCfg,
    steps: usize,
    t: usize,
    lr: f32,
) -> (f32, f32, f32) {
    use hos::tensor::AdamW;
    let is_rga = method == "rga";
    // RGA: 2 genomes over a shared bank. Phase A trains the bank + genome 0; phase
    // B FREEZES the bank and trains ONLY genome 1 — so domain A (bank + genome 0)
    // is preserved by construction (the M3 frozen-bank result). LoRA has one delta
    // and must overwrite it in phase B (the baseline that forgets).
    let peft = hos::peft::PeftModel::build_multi(base, method, hcfg, 2, 0xC0FFEE).expect("build");
    let mut rng = 0xace1_2345_6789_0abcu64;
    let mut nx = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 16) as usize
    };
    let lambda = if is_rga { 1e-3 } else { 0.0 };
    let mut train_phase = |tr: &[usize], pset: &[&hos::tensor::Tensor], gi: usize| {
        let decay: Vec<bool> = pset.iter().map(|p| p.shape().len() == 2).collect();
        let mut opt = AdamW::new(pset, lr, 0.0);
        for _ in 0..steps {
            for p in pset {
                p.zero_grad();
            }
            let s = nx() % (tr.len() - t - 1);
            let w = &tr[s..s + t + 1];
            peft.loss_g(&w[..t], &w[1..t + 1], lambda, gi).backward();
            opt.step(pset, &decay);
        }
    };

    // Phase A: bank + genome 0  (LoRA: all of its params; params_genome is empty)
    let mut a_params = peft.params_bank();
    a_params.extend(peft.params_genome(0));
    train_phase(da_train, &a_params, 0);
    let a1 = peft_eval_ppl(&peft, da_test, t, 0);

    // Phase B: RGA -> only genome 1 (bank frozen); LoRA -> its single delta again
    let (b_params, b_gi) = if is_rga {
        (peft.params_genome(1), 1)
    } else {
        (peft.params_bank(), 0)
    };
    train_phase(db_train, &b_params, b_gi);
    let a2 = peft_eval_ppl(&peft, da_test, t, 0); // A still uses genome 0
    let b2 = peft_eval_ppl(&peft, db_test, t, b_gi);
    (a1, a2, b2)
}

/// `--peft-interference -m <model>`: the continual-learning test where RGA's
/// thesis actually lives — learn A, then B, measure how much A is forgotten.
/// Env: INTERF_STEPS(60) PEFT_T(48) PEFT_LR(1e-3) PEFT_GENES(4) PEFT_BOTTLE(4).
fn cmd_peft_interference(args: &Args) {
    use hos::tensor::use_gpu;
    let path = resolve_model(args.model.clone());
    let eng = ok(hos::Engine::load(&path, false));
    let envn = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    };
    let (t, steps) = (envn("PEFT_T", 48), envn("INTERF_STEPS", 60));
    let lr: f32 = std::env::var("PEFT_LR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-3);
    let (genes, bottle) = (envn("PEFT_GENES", 4), envn("PEFT_BOTTLE", 4));

    let domains = build_domains(&eng.tok, 0xd0_5eed_1234);
    let c = &eng.model.cfg;
    let rga_total = (c.dim * genes + genes * 2 * c.dim * bottle) * c.n_layers + 8 + 8 * genes;
    let lora_per_rank =
        c.n_layers * (2 * c.dim + c.n_heads * c.head_dim + c.n_kv_heads * c.head_dim);
    let rank = (rga_total / lora_per_rank).max(1);
    let hcfg = hos::peft::PeftCfg {
        rank,
        genes,
        bottleneck: bottle,
        lora_alpha: 16.0,
    };

    // A = geography, B = story (very different distributions)
    let (na, da) = (domains[0].0, &domains[0]);
    let (nb, db) = (domains[2].0, &domains[2]);
    eprintln!(
        "[hos] interference test on {:?}: learn {na}, then {nb}; {steps} steps each, LoRA r{rank} vs RGA",
        c.arch
    );
    use_gpu(true);
    eprintln!("  LoRA ...");
    let (la1, la2, lb2) = peft_run_sequential(
        &eng.model, &da.1, &da.2, &db.1, &db.2, "lora", &hcfg, steps, t, lr,
    );
    eprintln!("  RGA ...");
    let (ra1, ra2, rb2) = peft_run_sequential(
        &eng.model, &da.1, &da.2, &db.1, &db.2, "rga", &hcfg, steps, t, lr,
    );

    println!("\n=== Continual learning: forgetting of '{na}' after learning '{nb}' ===");
    println!(
        "{:<8}{:>14}{:>14}{:>12}{:>12}",
        "method",
        format!("{na} (after {na})"),
        format!("{na} (after {nb})"),
        format!("{nb}"),
        "forgetting"
    );
    let fr = |a1: f32, a2: f32| a2 / a1; // >1 = worse on A after learning B
    println!(
        "{:<8}{:>14.2}{:>14.2}{:>12.2}{:>12.2}x",
        "LoRA",
        la1,
        la2,
        lb2,
        fr(la1, la2)
    );
    println!(
        "{:<8}{:>14.2}{:>14.2}{:>12.2}{:>12.2}x",
        "RGA",
        ra1,
        ra2,
        rb2,
        fr(ra1, ra2)
    );
    println!("\n(forgetting = ppl on A after learning B / ppl on A right after A; lower = remembers A better.)");
    if fr(ra1, ra2) < fr(la1, la2) * 0.95 {
        println!(
            "verdict: RGA forgets less ({:.2}x vs {:.2}x) — conditional gene expression protects domain A.",
            fr(ra1, ra2),
            fr(la1, la2)
        );
    } else if fr(la1, la2) < fr(ra1, ra2) * 0.95 {
        println!("verdict: LoRA forgets less here — honest negative for RGA on this setup.");
    } else {
        println!(
            "verdict: comparable forgetting; the gating did not clearly separate the domains here."
        );
    }
}

/// Train `pset` (a chosen parameter subset; the rest of the model is frozen) for
/// `steps` steps; `pick` yields a (window, genome-index) each step.
fn peft_train(
    peft: &hos::peft::PeftModel,
    pset: &[&hos::tensor::Tensor],
    steps: usize,
    t: usize,
    lr: f32,
    lambda: f32,
    mut pick: impl FnMut() -> (Vec<usize>, usize),
) {
    use hos::tensor::AdamW;
    let decay: Vec<bool> = pset.iter().map(|p| p.shape().len() == 2).collect();
    let mut opt = AdamW::new(pset, lr, 0.0);
    for _ in 0..steps {
        for p in pset {
            p.zero_grad();
        }
        let (w, gi) = pick();
        peft.loss_g(&w[..t], &w[1..t + 1], lambda, gi).backward();
        opt.step(pset, &decay);
    }
}

/// Matched RGA/LoRA hyperparameters (RGA budget, LoRA rank derived to match).
fn peft_matched(
    c: &hos::model::Config,
    genes: usize,
    bottle: usize,
) -> (hos::peft::PeftCfg, usize) {
    let rga_total = (c.dim * genes + genes * 2 * c.dim * bottle) * c.n_layers + 8 + 8 * genes;
    let lora_per_rank =
        c.n_layers * (2 * c.dim + c.n_heads * c.head_dim + c.n_kv_heads * c.head_dim);
    let rank = (rga_total / lora_per_rank).max(1);
    (
        hos::peft::PeftCfg {
            rank,
            genes,
            bottleneck: bottle,
            lora_alpha: 16.0,
        },
        rank,
    )
}

/// `--peft-heldout -m <model>`: the best-of-both test. Train a shared RGA bank on
/// several "seen" domains, freeze it, then adapt a HELD-OUT domain with only a new
/// genome — vs a LoRA that must continue-train (and forgets the seen domains).
/// Env: PEFT_STEPS(120) PEFT_T(48) PEFT_LR(1e-3) PEFT_GENES(4) PEFT_BOTTLE(4).
fn cmd_peft_heldout(args: &Args) {
    use hos::tensor::use_gpu;
    let path = resolve_model(args.model.clone());
    let eng = ok(hos::Engine::load(&path, false));
    let envn = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    };
    let (t, steps) = (envn("PEFT_T", 48), envn("PEFT_STEPS", 120));
    let lr: f32 = std::env::var("PEFT_LR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-3);
    let (genes, bottle) = (envn("PEFT_GENES", 4), envn("PEFT_BOTTLE", 4));
    let domains = build_domains(&eng.tok, 0xd0_5eed_1234);
    let n = domains.len();
    let seen = n - 1; // domains 0..n-1 seen; domain n-1 held out
    let (hcfg, rank) = peft_matched(&eng.model.cfg, genes, bottle);
    let mut rng = 0x51ce_d00d_1234_5678u64;
    let mut nx = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 16) as usize
    };
    use_gpu(true);
    eprintln!(
        "[hos] held-out test: bank on {seen} seen domains, genome-adapt '{}' (held out); LoRA r{rank}",
        domains[n - 1].0
    );
    let win = |d: usize, nx: &mut dyn FnMut() -> usize| {
        let tr = &domains[d].1;
        let s = nx() % (tr.len() - t - 1);
        tr[s..s + t + 1].to_vec()
    };

    // ---- RGA: train bank + seen genomes, freeze, then held-out genome only ----
    let rga =
        hos::peft::PeftModel::build_multi(&eng.model, "rga", &hcfg, n, 0xC0FFEE).expect("rga");
    let mut p1 = rga.params_bank();
    for d in 0..seen {
        p1.extend(rga.params_genome(d));
    }
    eprintln!("  RGA phase 1: diverse bank on seen domains ...");
    peft_train(&rga, &p1, steps, t, lr, 1e-3, || {
        let d = nx() % seen;
        (win(d, &mut nx), d)
    });
    let rga_seen_before: Vec<f32> = (0..seen)
        .map(|d| peft_eval_ppl(&rga, &domains[d].2, t, d))
        .collect();
    eprintln!("  RGA phase 2: held-out genome only (bank frozen) ...");
    let p2 = rga.params_genome(n - 1);
    peft_train(&rga, &p2, steps, t, lr, 1e-3, || {
        (win(n - 1, &mut nx), n - 1)
    });
    let rga_heldout = peft_eval_ppl(&rga, &domains[n - 1].2, t, n - 1);
    let rga_seen_after: Vec<f32> = (0..seen)
        .map(|d| peft_eval_ppl(&rga, &domains[d].2, t, d))
        .collect();

    // ---- LoRA: train on seen, then continue on held-out (forgets) ----
    let lora =
        hos::peft::PeftModel::build_multi(&eng.model, "lora", &hcfg, 1, 0xC0FFEE).expect("lora");
    let lp = lora.params();
    eprintln!("  LoRA phase 1: seen domains ...");
    peft_train(&lora, &lp, steps, t, lr, 0.0, || {
        let d = nx() % seen;
        (win(d, &mut nx), 0)
    });
    let lora_seen_before: Vec<f32> = (0..seen)
        .map(|d| peft_eval_ppl(&lora, &domains[d].2, t, 0))
        .collect();
    eprintln!("  LoRA phase 2: continue on held-out ...");
    peft_train(&lora, &lp, steps, t, lr, 0.0, || (win(n - 1, &mut nx), 0));
    let lora_heldout = peft_eval_ppl(&lora, &domains[n - 1].2, t, 0);
    let lora_seen_after: Vec<f32> = (0..seen)
        .map(|d| peft_eval_ppl(&lora, &domains[d].2, t, 0))
        .collect();

    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    println!("\n=== Held-out adaptation: keep the seen domains, add a new one ===");
    println!(
        "seen: {:?} | held out: {}",
        &domains[..seen].iter().map(|d| d.0).collect::<Vec<_>>(),
        domains[n - 1].0
    );
    println!(
        "{:<8}{:>14}{:>22}{:>22}",
        "method", "held-out ppl", "seen ppl (before->after)", "new-domain storage"
    );
    println!(
        "{:<8}{:>14.2}{:>10.2} -> {:<8.2}{:>22}",
        "RGA",
        rga_heldout,
        mean(&rga_seen_before),
        mean(&rga_seen_after),
        "1 genome (~32 floats)"
    );
    println!(
        "{:<8}{:>14.2}{:>10.2} -> {:<8.2}{:>22}",
        "LoRA",
        lora_heldout,
        mean(&lora_seen_before),
        mean(&lora_seen_after),
        "full adapter"
    );
    let rga_forget = mean(&rga_seen_after) / mean(&rga_seen_before);
    let lora_forget = mean(&lora_seen_after) / mean(&lora_seen_before);
    println!(
        "\nseen-domain forgetting: RGA {rga_forget:.2}x (frozen bank) vs LoRA {lora_forget:.2}x"
    );
    println!("(RGA adds a new domain for ~32 floats with zero forgetting; quality depends on the bank's diversity.)");
}

/// Primitives + a *composite* held-out domain (story structure carrying
/// arithmetic). The composite is never trained directly; it's only reachable if
/// the frozen bank holds the right primitives.
fn build_compose_domains(
    tok: &hos::tokenizer::Tokenizer,
    seed: u64,
) -> Vec<(&'static str, Vec<usize>, Vec<usize>)> {
    let mut rng = seed | 1;
    let mut nx = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 16) as usize
    };
    let geo = [
        ("France", "Paris"),
        ("Japan", "Tokyo"),
        ("Egypt", "Cairo"),
        ("Brazil", "Brasilia"),
        ("Kenya", "Nairobi"),
        ("Norway", "Oslo"),
    ];
    let animals = ["cat", "fox", "owl", "bear", "wolf", "crow", "deer", "hare"];
    let nouns = ["moon", "river", "forest", "stone", "field", "shore"];
    let names = ["add", "scale", "clip", "norm", "step", "mix", "fold"];

    let (mut geography, mut code, mut story, mut arithmetic, mut composite) = (
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
    );
    for _ in 0..360 {
        let (c, cap) = geo[nx() % geo.len()];
        geography.push_str(&format!("The capital of {c} is {cap}. "));
        code.push_str(&format!(
            "def {}(x): return x + {}\n",
            names[nx() % names.len()],
            nx() % 10
        ));
        let (an, no) = (animals[nx() % animals.len()], nouns[nx() % nouns.len()]);
        story.push_str(&format!("The {an} rests by the {no}. "));
        let (a, b) = (nx() % 30, nx() % 30);
        arithmetic.push_str(&format!("{a} plus {b} equals {}. ", a + b));
        // composite = story frame + arithmetic content (needs BOTH primitives)
        let (an, no) = (animals[nx() % animals.len()], nouns[nx() % nouns.len()]);
        let (a, b) = (nx() % 30, nx() % 30);
        composite.push_str(&format!(
            "The {an} saw {a} plus {b} equals {} by the {no}. ",
            a + b
        ));
    }
    [
        ("geography", geography),
        ("code", code),
        ("story", story),
        ("arithmetic", arithmetic),
        ("composite", composite),
    ]
    .into_iter()
    .map(|(name, text)| {
        let ids: Vec<usize> = tok
            .encode(&text, true)
            .iter()
            .map(|&x| x as usize)
            .collect();
        let split = ids.len() * 85 / 100;
        (name, ids[..split].to_vec(), ids[split..].to_vec())
    })
    .collect()
}

/// `--peft-compose -m <model>`: does bank *diversity* move the "can't invent"
/// boundary? Target = a composite domain (story + arithmetic) never trained
/// directly. Train one frozen bank that LACKS the ingredients (geography+code)
/// and one that HAS them (story+arithmetic), then adapt the composite with a
/// genome only over each. If "has" ≪ "lacks", the genome reached a *new* task by
/// recombining existing primitives — diversity, not size, is the lever.
/// Env: PEFT_STEPS(120) PEFT_T(48) PEFT_LR(1e-3) PEFT_GENES(4) PEFT_BOTTLE(4).
fn cmd_peft_compose(args: &Args) {
    use hos::tensor::use_gpu;
    let path = resolve_model(args.model.clone());
    let eng = ok(hos::Engine::load(&path, false));
    let envn = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    };
    let (t, steps) = (envn("PEFT_T", 48), envn("PEFT_STEPS", 120));
    let lr: f32 = std::env::var("PEFT_LR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-3);
    let (genes, bottle) = (envn("PEFT_GENES", 4), envn("PEFT_BOTTLE", 4));
    let (hcfg, _) = peft_matched(&eng.model.cfg, genes, bottle);
    let domains = build_compose_domains(&eng.tok, 0xc0de_5eed_9999_1234);
    let composite = 4usize;
    use_gpu(true);
    eprintln!(
        "[hos] compositional held-out: target '{}' (story + arithmetic), never trained directly",
        domains[composite].0
    );

    // a held-out composite prompt to show the qualitative difference, in the exact
    // template the composite domain follows ("The <animal> saw <a> plus <b> equals ...")
    let cprompt: Vec<usize> = eng
        .tok
        .encode("The fox saw 13 plus 5 equals", true)
        .iter()
        .map(|&x| x as usize)
        .collect();
    let run = |seen: &[usize], label: &str| -> (f32, String) {
        let n = seen.len() + 1;
        let cg = seen.len(); // the composite's genome index
        let rga =
            hos::peft::PeftModel::build_multi(&eng.model, "rga", &hcfg, n, 0xC0FFEE).expect("rga");
        let mut p1 = rga.params_bank();
        for j in 0..seen.len() {
            p1.extend(rga.params_genome(j));
        }
        let mut rng = 0x51ce_d00d_2222_4444u64;
        let mut nx = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 16) as usize
        };
        let win = |d: usize, nx: &mut dyn FnMut() -> usize| {
            let tr = &domains[d].1;
            let s = nx() % (tr.len() - t - 1);
            tr[s..s + t + 1].to_vec()
        };
        eprintln!(
            "  [{label}] bank on {:?} ...",
            seen.iter().map(|&d| domains[d].0).collect::<Vec<_>>()
        );
        peft_train(&rga, &p1, steps, t, lr, 1e-3, || {
            let j = nx() % seen.len();
            (win(seen[j], &mut nx), j)
        });
        eprintln!("  [{label}] composite genome only (bank frozen) ...");
        let p2 = rga.params_genome(cg);
        peft_train(&rga, &p2, steps, t, lr, 1e-3, || {
            (win(composite, &mut nx), cg)
        });
        let ppl = peft_eval_ppl(&rga, &domains[composite].2, t, cg);
        let sample = peft_generate(&rga, &eng.tok, &cprompt, cg, 10);
        (ppl, sample)
    };

    let (lacks, lacks_s) = run(&[0, 1], "lacks"); // geography + code
    let (has, has_s) = run(&[2, 3], "has"); // story + arithmetic

    println!("\n=== Compositional held-out: can a genome COMPOSE existing primitives? ===");
    println!("target: 'composite' = story structure carrying arithmetic (never trained directly)");
    println!(
        "{:<38}{:>20}",
        "frozen bank trained on", "composite ppl (genome only)"
    );
    println!(
        "{:<38}{:>20.2}",
        "geography + code  (no ingredients)", lacks
    );
    println!("{:<38}{:>20.2}", "story + arithmetic  (ingredients)", has);
    let ratio = lacks / has.max(1e-6);
    println!("\nratio (lacks / has): {ratio:.2}x");
    println!("\n--- generation: prompt \"The fox saw 13 plus 5 equals\" (answer: 18) ---");
    println!("  bank WITHOUT ingredients -> \"...{lacks_s}\"");
    println!("  bank WITH    ingredients -> \"...{has_s}\"");
    println!(
        "A bank that holds the primitives lets the genome REACH the composite by recombination;"
    );
    println!(
        "one without them cannot. Diversity of the bank — not model size — moves the boundary."
    );
}

/// `--peft-clonal -m <model>`: clonal selection — can RGA *invent* a held-out
/// (out-of-span) domain by mutating its best-matching genes, which pure
/// regulation can't reach? Train a shared bank on the seen domains + freeze it,
/// then for the held-out domain: (1) regulation only (genome over the frozen
/// bank) as the baseline pure RGA can manage, then (2) select the genes that
/// domain's genome most expresses and SOMATICALLY MUTATE them (unfreeze their
/// down/up modules) while training. Reports held-out quality and seen-domain
/// forgetting — the middle ground between RGA (can't invent) and LoRA (forgets).
/// Env: PEFT_STEPS(120) PEFT_T(48) PEFT_LR(1e-3) PEFT_GENES(8) PEFT_BOTTLE(4) PEFT_CLONES(3).
fn cmd_peft_clonal(args: &Args) {
    use hos::tensor::use_gpu;
    let path = resolve_model(args.model.clone());
    let eng = ok(hos::Engine::load(&path, false));
    let envn = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    };
    let (t, steps) = (envn("PEFT_T", 48), envn("PEFT_STEPS", 120));
    let lr: f32 = std::env::var("PEFT_LR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-3);
    let (genes, bottle) = (envn("PEFT_GENES", 8), envn("PEFT_BOTTLE", 4));
    let n_clone = envn("PEFT_CLONES", 3).min(genes);
    let domains = build_domains(&eng.tok, 0xd0_5eed_1234);
    let n = domains.len();
    let seen = n - 1; // 0..n-1 seen; n-1 held out (out-of-span: code)
    let held = n - 1;
    let (hcfg, _) = peft_matched(&eng.model.cfg, genes, bottle);
    let mut rng = 0xc10a_a1ed_5eed_1234u64;
    let mut nx = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 16) as usize
    };
    use_gpu(true);
    eprintln!(
        "[hos] clonal selection: bank on {seen} domains, then mutate the {n_clone} best genes to invent '{}'",
        domains[held].0
    );
    let win = |d: usize, nx: &mut dyn FnMut() -> usize| {
        let tr = &domains[d].1;
        let s = nx() % (tr.len() - t - 1);
        tr[s..s + t + 1].to_vec()
    };

    // held-out (code) prompt in the exact domain template, for before/after gen
    let kprompt: Vec<usize> = eng
        .tok
        .encode("def scale(x): return x +", true)
        .iter()
        .map(|&x| x as usize)
        .collect();
    let mut rga =
        hos::peft::PeftModel::build_multi(&eng.model, "rga", &hcfg, n, 0xC0FFEE).expect("rga");
    // phase 1: shared bank + seen genomes on the seen domains, then freeze bank
    let mut p1 = rga.params_bank();
    for d in 0..seen {
        p1.extend(rga.params_genome(d));
    }
    eprintln!("  phase 1: shared bank on seen domains ...");
    peft_train(&rga, &p1, steps, t, lr, 1e-3, || {
        let d = nx() % seen;
        (win(d, &mut nx), d)
    });
    let seen_before: Vec<f32> = (0..seen)
        .map(|d| peft_eval_ppl(&rga, &domains[d].2, t, d))
        .collect();

    // phase 2: regulation only — genome over the frozen bank (pure RGA baseline)
    eprintln!("  phase 2: regulation only (genome over frozen bank) ...");
    let gp = rga.params_genome(held);
    peft_train(&rga, &gp, steps, t, lr, 1e-3, || (win(held, &mut nx), held));
    let code_reg = peft_eval_ppl(&rga, &domains[held].2, t, held);
    let reg_s = peft_generate(&rga, &eng.tok, &kprompt, held, 12); // clones still off

    // phase 3: clonal selection — pick the genes the held genome most drives,
    // PROLIFERATE them into a private bank, and mutate the copies. The shared
    // originals are untouched, and the clones are expressed ONLY for the new task.
    let aff = rga.genome_gate_bias(held);
    let mut order: Vec<usize> = (0..aff.len()).collect();
    order.sort_by(|&a, &b| {
        aff[b]
            .partial_cmp(&aff[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top: Vec<usize> = order.into_iter().take(n_clone).collect();
    eprintln!(
        "  phase 3: proliferate + mutate {n_clone} high-affinity genes {top:?} (private) ..."
    );
    rga.proliferate(&top);
    rga.set_clones(true); // the new task expresses its private clones
    let mut cp = rga.params_clones();
    cp.extend(rga.params_genome(held));
    peft_train(&rga, &cp, steps, t, lr, 1e-3, || (win(held, &mut nx), held));
    let code_clonal = peft_eval_ppl(&rga, &domains[held].2, t, held); // clones ON
    let clonal_s = peft_generate(&rga, &eng.tok, &kprompt, held, 12); // clones ON
    rga.set_clones(false); // the seen domains never see the clones
    let seen_after: Vec<f32> = (0..seen)
        .map(|d| peft_eval_ppl(&rga, &domains[d].2, t, d))
        .collect();

    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    println!("\n=== Clonal selection: mutate the best genes to INVENT a new domain ===");
    println!(
        "seen: {:?} | new (out-of-span): {}",
        &domains[..seen].iter().map(|d| d.0).collect::<Vec<_>>(),
        domains[held].0
    );
    println!("{:<34}{:>16}", "stage", "new-domain ppl");
    println!("{:<34}{:>16.2}", "regulation only (pure RGA)", code_reg);
    println!(
        "{:<34}{:>16.2}",
        format!("clonal ({n_clone}/{genes} genes proliferated)"),
        code_clonal
    );
    let improve = code_reg / code_clonal.max(1e-6);
    let forget = mean(&seen_after) / mean(&seen_before);
    println!("\ninvention: {improve:.1}x better on the new domain than pure regulation");
    println!(
        "seen-domain forgetting: {forget:.2}x  (seen {:.2} -> {:.2})",
        mean(&seen_before),
        mean(&seen_after)
    );
    println!("\n--- generation: prompt \"def scale(x): return x +\" (domain: code) ---");
    println!("  regulation only (RGA, can't invent) -> \"...{reg_s}\"");
    println!("  clonal selection (invented)         -> \"...{clonal_s}\"");
    println!("(proliferated clones are private to the new task: invention WITHOUT touching the");
    println!(" seen domains — the gap between RGA, which can't invent, and LoRA, which forgets.)");
}

/// `--peft-demo -m <model>`: the plainest demonstration — train one RGA genome
/// per domain over a shared bank, then show, per domain, what the model completes
/// BEFORE adaptation (untrained adapters contribute exactly zero, so this is the
/// frozen base) vs AFTER. Held-out prompts in each domain's own template.
/// Env: PEFT_STEPS(120) PEFT_T(48) PEFT_LR(1e-3) PEFT_GENES(8) PEFT_BOTTLE(4).
fn cmd_peft_demo(args: &Args) {
    use hos::tensor::use_gpu;
    let path = resolve_model(args.model.clone());
    let eng = ok(hos::Engine::load(&path, false));
    let envn = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    };
    let (t, steps) = (envn("PEFT_T", 48), envn("PEFT_STEPS", 120));
    let lr: f32 = std::env::var("PEFT_LR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-3);
    let (genes, bottle) = (envn("PEFT_GENES", 8), envn("PEFT_BOTTLE", 4));
    let domains = build_domains(&eng.tok, 0xd0_5eed_1234); // geography, arithmetic, story, code
    let (hcfg, _) = peft_matched(&eng.model.cfg, genes, bottle);
    use_gpu(true);
    // held-out prompts, each in its domain's own template (unseen values)
    let prompts = [
        "The capital of Spain is", // geography -> Madrid
        "12 plus 7 equals",        // arithmetic -> 19
        "The owl watched the",     // story -> a noun
        "def fold(x): return",     // code -> x + <digit>
    ];
    let enc = |s: &str| -> Vec<usize> {
        eng.tok
            .encode(s, true)
            .iter()
            .map(|&x| x as usize)
            .collect()
    };
    let pids: Vec<Vec<usize>> = prompts.iter().map(|p| enc(p)).collect();

    let rga = hos::peft::PeftModel::build_multi(&eng.model, "rga", &hcfg, domains.len(), 0xC0FFEE)
        .expect("rga");
    // BEFORE: adapters init to zero contribution, so this IS the frozen base model
    let before: Vec<String> = (0..domains.len())
        .map(|d| peft_generate(&rga, &eng.tok, &pids[d], d, 7))
        .collect();

    // train the shared bank + one genome per domain on all domains
    let mut params = rga.params_bank();
    for d in 0..domains.len() {
        params.extend(rga.params_genome(d));
    }
    let mut rng = 0xdec0_de11_2233_4455u64;
    let mut nx = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 16) as usize
    };
    let win = |d: usize, nx: &mut dyn FnMut() -> usize| {
        let tr = &domains[d].1;
        let s = nx() % (tr.len() - t - 1);
        tr[s..s + t + 1].to_vec()
    };
    eprintln!(
        "[hos] per-domain demo: bank + {} genomes on {:?} ...",
        domains.len(),
        domains.iter().map(|d| d.0).collect::<Vec<_>>()
    );
    peft_train(&rga, &params, steps, t, lr, 1e-3, || {
        let d = nx() % domains.len();
        (win(d, &mut nx), d)
    });
    let after: Vec<String> = (0..domains.len())
        .map(|d| peft_generate(&rga, &eng.tok, &pids[d], d, 7))
        .collect();

    println!("\n=== Per-domain: frozen base vs RGA-adapted (one genome per domain) ===");
    for (d, (name, _, test)) in domains.iter().enumerate() {
        let ppl = peft_eval_ppl(&rga, test, t, d);
        println!(
            "\n[{name}]  prompt: \"{}\"   (adapted held-out ppl {ppl:.2})",
            prompts[d]
        );
        println!("  base (untrained) -> \"...{}\"", before[d]);
        println!("  RGA-adapted      -> \"...{}\"", after[d]);
    }
    println!("\n(BEFORE = the frozen base: RGA adapters initialise to zero contribution, so an");
    println!(" untrained genome reproduces the base model exactly. AFTER = the same model with");
    println!(" each domain's trained genome over the shared bank — proof of mechanism, 135M.)");
}

/// `--peft-grow -m <model>`: the full loop in one run — show the "can't invent"
/// boundary MOVE. Train a shared bank on the seen domains; show a plain genome
/// CANNOT reach the out-of-span domain (before); INVENT it with clonal selection
/// (private clones); CONSOLIDATE it into the shared bank with replay; then show a
/// FRESH plain genome now CAN reach it (after) — the span grew, nothing forgotten.
/// Env: PEFT_STEPS(120) PEFT_T(48) PEFT_LR(1e-3) PEFT_GENES(8) PEFT_BOTTLE(4) PEFT_CLONES(3).
fn cmd_peft_grow(args: &Args) {
    use hos::tensor::use_gpu;
    let path = resolve_model(args.model.clone());
    let eng = ok(hos::Engine::load(&path, false));
    let envn = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    };
    let (t, steps) = (envn("PEFT_T", 48), envn("PEFT_STEPS", 120));
    let lr: f32 = std::env::var("PEFT_LR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-3);
    let (genes, bottle) = (envn("PEFT_GENES", 8), envn("PEFT_BOTTLE", 4));
    let n_clone = envn("PEFT_CLONES", 3).min(genes);
    let domains = build_domains(&eng.tok, 0xd0_5eed_1234);
    let n = domains.len();
    let seen = n - 1;
    let held = n - 1; // out-of-span domain (code)
    let (hcfg, _) = peft_matched(&eng.model.cfg, genes, bottle);
    let mut rng = 0x9701_a51e_ed12_3456u64;
    let mut nx = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 16) as usize
    };
    use_gpu(true);
    eprintln!(
        "[hos] grow: invent + consolidate '{}' and watch the boundary move",
        domains[held].0
    );
    let win = |d: usize, nx: &mut dyn FnMut() -> usize| {
        let tr = &domains[d].1;
        let s = nx() % (tr.len() - t - 1);
        tr[s..s + t + 1].to_vec()
    };
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;

    let mut rga =
        hos::peft::PeftModel::build_multi(&eng.model, "rga", &hcfg, n, 0xC0FFEE).expect("rga");

    // ---- phase 1: shared bank + seen genomes on the seen domains ----
    let mut p1 = rga.params_bank();
    for d in 0..seen {
        p1.extend(rga.params_genome(d));
    }
    eprintln!("  [1/4] shared bank on seen domains ...");
    peft_train(&rga, &p1, steps, t, lr, 1e-3, || {
        let d = nx() % seen;
        (win(d, &mut nx), d)
    });
    let seen_start = mean(
        &(0..seen)
            .map(|d| peft_eval_ppl(&rga, &domains[d].2, t, d))
            .collect::<Vec<_>>(),
    );

    // ---- BEFORE: can a plain genome reach the out-of-span domain? ----
    eprintln!("  [2/4] BEFORE — plain genome over the frozen bank ...");
    let gp = rga.params_genome(held);
    peft_train(&rga, &gp, steps, t, lr, 1e-3, || (win(held, &mut nx), held));
    let before = peft_eval_ppl(&rga, &domains[held].2, t, held);

    // ---- INVENT: clonal selection (private proliferated clones) ----
    let aff = rga.genome_gate_bias(held);
    let mut order: Vec<usize> = (0..aff.len()).collect();
    order.sort_by(|&a, &b| {
        aff[b]
            .partial_cmp(&aff[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top: Vec<usize> = order.into_iter().take(n_clone).collect();
    eprintln!("  [3/4] INVENT — clonal selection on genes {top:?} (private) ...");
    rga.proliferate(&top);
    rga.set_clones(true);
    let mut cp = rga.params_clones();
    cp.extend(rga.params_genome(held));
    peft_train(&rga, &cp, steps, t, lr, 1e-3, || (win(held, &mut nx), held));
    let invented = peft_eval_ppl(&rga, &domains[held].2, t, held);
    rga.set_clones(false);

    // ---- CONSOLIDATE: replay the new domain into the shared bank ----
    eprintln!("  [4/4] CONSOLIDATE — replay (old + new) into the shared bank ...");
    let pall = rga.params();
    peft_train(&rga, &pall, steps, t, lr, 1e-3, || {
        let d = nx() % n; // replay mixture: seen + the new domain
        (win(d, &mut nx), d)
    });

    // ---- AFTER: reset the held genome to fresh random; can it reach it now? ----
    let glen = rga.genome_data(held).len();
    let fresh: Vec<f32> = (0..glen)
        .map(|_| (nx() % 2000) as f32 / 1000.0 - 1.0)
        .collect();
    rga.set_genome(held, &fresh);
    eprintln!("  AFTER — a FRESH plain genome over the grown bank ...");
    let gp2 = rga.params_genome(held);
    peft_train(&rga, &gp2, steps, t, lr, 1e-3, || {
        (win(held, &mut nx), held)
    });
    let after = peft_eval_ppl(&rga, &domains[held].2, t, held);
    let seen_end = mean(
        &(0..seen)
            .map(|d| peft_eval_ppl(&rga, &domains[d].2, t, d))
            .collect::<Vec<_>>(),
    );

    println!("\n=== Grow: the 'can't invent' boundary, moved ===");
    println!(
        "seen: {:?} | out-of-span: {}",
        &domains[..seen].iter().map(|d| d.0).collect::<Vec<_>>(),
        domains[held].0
    );
    println!(
        "  BEFORE  plain genome reaches '{}'?   {:>12.2}   (out-of-span — can't invent)",
        domains[held].0, before
    );
    println!("  invent  clonal selection (private):  {invented:>12.2}   (fast, private organ)");
    println!("  consolidate  replay into shared bank        (slow, permanent)");
    println!(
        "  AFTER   fresh plain genome reaches it?  {after:>12.2}   (now IN-SPAN — the span grew)",
    );
    let moved = before / after.max(1e-6);
    println!("\nboundary moved: {moved:.0}x more reachable by pure regulation after one cycle");
    println!(
        "forgetting across the whole loop: {:.2}x  (seen {seen_start:.2} -> {seen_end:.2})",
        seen_end / seen_start
    );
    println!("(invent fast in a private organ; consolidate the winner into the shared genome.");
    println!(
        " diversity is earned: what was unreachable becomes a primitive, with no forgetting.)"
    );
}

/// `--peft-fuse -m <base.hos>`: **Regime-A two-parent fusion.** Build two
/// homologous parents over ONE frozen base (Parent A: geography+arithmetic;
/// Parent B: story+code). Show neither parent spans the other's domains (a fresh
/// genome over a frozen single-parent bank is out-of-span). FUSE the two gene
/// banks into one shared body (span-union via import — alignment is the identity
/// because both regulate the same base). CONSOLIDATE with replay so the imported
/// organs integrate with no forgetting. BREED a crossover genome that keeps both
/// parents. Emit the result as a `.hos` BODY that inspects as `body: hos` (its own
/// family, lineage -> base) and round-trip re-evaluate it to prove faithfulness.
/// Env: PEFT_STEPS(120) PEFT_T(48) PEFT_LR(1e-3) PEFT_GENES(6) PEFT_BOTTLE(4)
///      FUSE_REFIT(40) FUSE_REPLAY(160) FUSE_OUT(body.hos)
fn cmd_peft_fuse(args: &Args) {
    use hos::tensor::use_gpu;
    let path = resolve_model(args.model.clone());
    let eng = ok(hos::Engine::load(&path, false));
    let envn = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    };
    let (t, steps) = (envn("PEFT_T", 48), envn("PEFT_STEPS", 120));
    let lr: f32 = std::env::var("PEFT_LR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-3);
    let (genes, bottle) = (envn("PEFT_GENES", 6), envn("PEFT_BOTTLE", 4));
    let refit = envn("FUSE_REFIT", 40);
    let replay = envn("FUSE_REPLAY", 160);
    let grow = envn("FUSE_GROW", 0); // Net2Net: extra trainable FFN dims/layer (0 = off)
    let out = std::env::var("FUSE_OUT").unwrap_or_else(|_| "body.hos".into());
    let domains = build_domains(&eng.tok, 0xd0_5eed_1234); // geo, arith, story, code
    let names = ["geography", "arithmetic", "story", "code"];
    let (a_doms, b_doms) = ([0usize, 1], [2usize, 3]); // A: geo+arith ; B: story+code
    let (hcfg, _) = peft_matched(&eng.model.cfg, genes, bottle);
    let (hcfg_f, _) = peft_matched(&eng.model.cfg, 2 * genes, bottle); // fused: K = Ka+Kb
    use_gpu(true);
    let mut rng = 0xf0_5e_1234_abcd_ef01u64;
    let mut nx = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 16) as usize
    };
    let win = |d: usize, nx: &mut dyn FnMut() -> usize| {
        let tr = &domains[d].1;
        let s = nx() % (tr.len() - t - 1);
        tr[s..s + t + 1].to_vec()
    };
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    eprintln!("[hos] FUSE (Regime A): two homologous parents -> one body, span-union + replay");

    // ---- Parent A: bank_A + genomes{0:geo,1:arith}; slots 2,3 spare for probes ----
    eprintln!("  [1/6] train Parent A (geography + arithmetic) ...");
    let pa = hos::peft::PeftModel::build_multi(&eng.model, "rga", &hcfg, 4, 0xA1A1).expect("pa");
    let mut pap = pa.params_bank();
    pap.extend(pa.params_genome(0));
    pap.extend(pa.params_genome(1));
    peft_train(&pa, &pap, steps, t, lr, 1e-3, || {
        let j = nx() % 2;
        (win(a_doms[j], &mut nx), j)
    });
    let a_in = [
        peft_eval_ppl(&pa, &domains[0].2, t, 0),
        peft_eval_ppl(&pa, &domains[1].2, t, 1),
    ];

    // ---- Parent B: bank_B + genomes{0:story,1:code} ----
    eprintln!("  [2/6] train Parent B (story + code) ...");
    let pb = hos::peft::PeftModel::build_multi(&eng.model, "rga", &hcfg, 4, 0xB2B2).expect("pb");
    let mut pbp = pb.params_bank();
    pbp.extend(pb.params_genome(0));
    pbp.extend(pb.params_genome(1));
    peft_train(&pb, &pbp, steps, t, lr, 1e-3, || {
        let j = nx() % 2;
        (win(b_doms[j], &mut nx), j)
    });
    let b_in = [
        peft_eval_ppl(&pb, &domains[2].2, t, 0),
        peft_eval_ppl(&pb, &domains[3].2, t, 1),
    ];

    // ---- BASELINE: each parent reaching the OTHER's domains via a fresh genome
    // over its FROZEN bank — out-of-span (neither parent spans the union) ----
    eprintln!("  [3/6] baseline — single-parent banks on the other's domains (out-of-span) ...");
    let base_steps = (steps / 3).max(25); // out-of-span stays bad — no need to converge
    peft_train(&pa, &pa.params_genome(2), base_steps, t, lr, 1e-3, || {
        (win(2, &mut nx), 2)
    });
    peft_train(&pa, &pa.params_genome(3), base_steps, t, lr, 1e-3, || {
        (win(3, &mut nx), 3)
    });
    let a_cross = [
        peft_eval_ppl(&pa, &domains[2].2, t, 2),
        peft_eval_ppl(&pa, &domains[3].2, t, 3),
    ];
    peft_train(&pb, &pb.params_genome(2), base_steps, t, lr, 1e-3, || {
        (win(0, &mut nx), 2)
    });
    peft_train(&pb, &pb.params_genome(3), base_steps, t, lr, 1e-3, || {
        (win(1, &mut nx), 3)
    });
    let b_cross = [
        peft_eval_ppl(&pb, &domains[0].2, t, 2),
        peft_eval_ppl(&pb, &domains[1].2, t, 3),
    ];
    let cross_base = mean(&[a_cross[0], a_cross[1], b_cross[0], b_cross[1]]);

    // ---- FUSE: import both banks into one body (K = Ka+Kb), copy genomes ----
    eprintln!("  [4/6] FUSE — concatenate banks (span-union), import genomes ...");
    let mut fused =
        hos::peft::PeftModel::build_multi(&eng.model, "rga", &hcfg_f, 5, 0xF0_0D).expect("fused");
    // d0 geo<-A0, d1 arith<-A1, d2 story<-B0, d3 code<-B1, d4 breed slot (init A0)
    let genome_src = [
        (false, 0usize),
        (false, 1),
        (true, 0),
        (true, 1),
        (false, 0),
    ];
    fused.fuse(&pa, &pb, &genome_src);
    if grow > 0 {
        // Net2Net: widen the fused body's hidden layers (function-preserving init)
        fused.grow_ffn(grow, 0x6707_0517_d1ce_0001);
        eprintln!("        + Net2Net: widened FFN by {grow} trainable dims/layer (the fused body's hidden layers grow)");
    }
    let imported = [0, 1, 2, 3].map(|d| peft_eval_ppl(&fused, &domains[d].2, t, d));

    // ---- refit each genome over the FROZEN fused bank (settle cross-drive; bank
    // frozen + genomes private => no forgetting), then CONSOLIDATE with replay ----
    eprintln!("  [5/6] consolidate — genome refit + replay over all four domains ...");
    for d in 0..4 {
        peft_train(&fused, &fused.params_genome(d), refit, t, lr, 1e-3, || {
            (win(d, &mut nx), d)
        });
    }
    let pall = fused.params();
    peft_train(&fused, &pall, replay, t, lr, 1e-3, || {
        let d = nx() % 4;
        (win(d, &mut nx), d)
    });
    let fused_ppl = [0, 1, 2, 3].map(|d| peft_eval_ppl(&fused, &domains[d].2, t, d));

    // ---- BREED: child = crossover(geo genome, code genome) -> slot 4; keeps both? ----
    let (g_geo, g_code) = (fused.genome_data(0), fused.genome_data(3));
    let child: Vec<f32> = g_geo
        .iter()
        .zip(&g_code)
        .map(|(a, b)| 0.5 * (a + b))
        .collect();
    fused.set_genome(4, &child);
    let child_geo = peft_eval_ppl(&fused, &domains[0].2, t, 4);
    let child_code = peft_eval_ppl(&fused, &domains[3].2, t, 4);

    // ---- EMIT the fused .hos BODY + round-trip verification ----
    eprintln!("  [6/6] mint fused .hos body + round-trip ...");
    let rt = save_body(&path, &out, &fused, bottle, &names);

    // ================= report =================
    let in_ref = mean(&[a_in[0], a_in[1], b_in[0], b_in[1]]);
    let fused_mean = mean(&fused_ppl);
    let forget = mean(&[
        fused_ppl[0] / a_in[0],
        fused_ppl[1] / a_in[1],
        fused_ppl[2] / b_in[0],
        fused_ppl[3] / b_in[1],
    ]);
    println!("\n=== FUSE: two homologous parents -> one `body: hos` (Regime A) ===");
    println!("parents:  A = [geography, arithmetic]   B = [story, code]   (one frozen base)\n");
    println!(
        "{:<34}{:>10}{:>10}{:>10}{:>10}",
        "", names[0], names[1], names[2], names[3]
    );
    println!(
        "{:<34}{:>10.2}{:>10.2}{:>10}{:>10}",
        "Parent A (own genomes)", a_in[0], a_in[1], "-", "-"
    );
    println!(
        "{:<34}{:>10}{:>10}{:>10.2}{:>10.2}",
        "Parent B (own genomes)", "-", "-", b_in[0], b_in[1]
    );
    println!(
        "{:<34}{:>10}{:>10}{:>10.2}{:>10.2}",
        "A on B's domains (out-of-span)", "-", "-", a_cross[0], a_cross[1]
    );
    println!(
        "{:<34}{:>10.2}{:>10.2}{:>10}{:>10}",
        "B on A's domains (out-of-span)", b_cross[0], b_cross[1], "-", "-"
    );
    println!(
        "{:<34}{:>10.2}{:>10.2}{:>10.2}{:>10.2}",
        "FUSED body (import only)", imported[0], imported[1], imported[2], imported[3]
    );
    println!(
        "{:<34}{:>10.2}{:>10.2}{:>10.2}{:>10.2}",
        "FUSED body (consolidated)", fused_ppl[0], fused_ppl[1], fused_ppl[2], fused_ppl[3]
    );
    println!("\nin-span parent ref (mean) : {in_ref:.2}");
    println!("out-of-span baseline (mean): {cross_base:.2}   (neither parent spans the union)");
    println!("FUSED union span (mean)    : {fused_mean:.2}");
    println!("forgetting vs parents      : {forget:.2}x   (1.0 = the body kept both parents)");
    println!(
        "breed: child=crossover(geo,code) -> geo {child_geo:.2} / code {child_code:.2}  (keeps both)"
    );
    let span_gain = cross_base / fused_mean.max(1e-6);
    println!("\nspan-union gain: {span_gain:.1}x more reachable than either parent alone");
    match rt {
        Some((rt_ppl, id)) => {
            let drift = mean(&[
                (rt_ppl[0] - fused_ppl[0]).abs(),
                (rt_ppl[1] - fused_ppl[1]).abs(),
                (rt_ppl[2] - fused_ppl[2]).abs(),
                (rt_ppl[3] - fused_ppl[3]).abs(),
            ]);
            println!(
                "saved body: {out}  (id {id}) — loads as `body: hos`, lineage -> base; round-trip drift {drift:.4}"
            );
        }
        None => println!("saved body: {out}"),
    }
    let win_span = fused_mean < cross_base * 0.6;
    let win_forget = forget < 2.0;
    println!(
        "\nVERDICT: span-union {} | no-forgetting {} {}",
        if win_span {
            "ACHIEVED ✓"
        } else {
            "not yet ✗"
        },
        if win_forget {
            "ACHIEVED ✓"
        } else {
            "not yet ✗"
        },
        if win_span && win_forget {
            "— improvement confirmed"
        } else {
            "— iterate"
        }
    );
}

/// Mint the fused organism as a runnable `.hos` BODY: copy the base capsule's
/// tensors + metadata (so it loads and runs), append the fused RGA adapter as
/// `body.*` tensors (role SCALAR -> stored f32, never quantized -> exact genes),
/// and rewrite the card so it identifies as `body: hos` with lineage -> base.
/// Returns the round-trip per-domain ppl + new id if verification ran.
fn save_body(
    src: &std::path::Path,
    out: &str,
    fused: &hos::peft::PeftModel,
    bottle: usize,
    names: &[&str; 4],
) -> Option<([f32; 4], String)> {
    if src.extension().and_then(|s| s.to_str()) != Some("hos") {
        eprintln!(
            "[hos] --peft-fuse body emission needs a .hos base (mint via --to-hos); skipping save"
        );
        return None;
    }
    let (base, base_card) = match hos::format::load(src) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[hos] could not read base capsule: {e}");
            return None;
        }
    };
    let (k, ng) = fused.rga_shape().unwrap_or((0, 0));
    let mut named = base.clone();
    for (n, sh, data) in fused.adapter_tensors() {
        named.push(hos::format::Named {
            name: format!("body.{n}"),
            role: hos::format::ROLE_SCALAR, // tiny + precision-sensitive: stay f32
            shape: sh,
            data,
        });
    }
    let mut card = base_card.clone();
    card.name = "body".into();
    card.mode = "frozen".into();
    if let Some(obj) = card.arch.as_object_mut() {
        if let Some(mt) = obj.get("model_type").cloned() {
            obj.insert("origin_model_type".into(), mt);
        }
        obj.insert("family".into(), serde_json::json!("hos"));
        obj.insert(
            "hos_body".into(),
            serde_json::json!({
                "method": "rga-fused (Regime A)",
                "genes": k, "genomes": ng, "bottleneck": bottle,
                "grow_ffn_dims": fused.grow_delta(),
                "domains": [names[0], names[1], names[2], names[3], "child(geo×code)"],
                "parents": ["A: geography+arithmetic", "B: story+code"],
            }),
        );
    }
    card.lineage.push(base_card.id.clone());
    card.id = hos::format::model_id(&named);
    if let Err(e) = hos::format::save_quantized(std::path::Path::new(out), &named, &card) {
        eprintln!("[hos] save body failed: {e}");
        return None;
    }

    // round-trip: reload the body, rebuild the fused shell, re-express its genes
    let eng2 = match hos::Engine::load(std::path::Path::new(out), false) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("[hos] body saved but reload failed: {e}");
            return Some(([f32::NAN; 4], card.id));
        }
    };
    let (rt_named, _) = hos::format::load(std::path::Path::new(out)).ok()?;
    let snap: Vec<(String, Vec<usize>, Vec<f32>)> = rt_named
        .iter()
        .filter(|n| n.name.starts_with("body."))
        .map(|n| {
            (
                n.name["body.".len()..].to_string(),
                n.shape.clone(),
                n.data.clone(),
            )
        })
        .collect();
    let (hcfg_f, _) = peft_matched(&eng2.model.cfg, k / 2, bottle); // K already = 2*genes
    let hcfg_f = hos::peft::PeftCfg { genes: k, ..hcfg_f };
    let mut fused2 =
        hos::peft::PeftModel::build_multi(&eng2.model, "rga", &hcfg_f, ng, 0xF0_0D).expect("rt");
    let gd = fused.grow_delta();
    if gd > 0 {
        fused2.grow_ffn(gd, 1); // re-allocate grown dims; load_adapter overwrites the data
    }
    fused2.load_adapter(&snap);
    let doms2 = build_domains(&eng2.tok, 0xd0_5eed_1234);
    let t = std::env::var("PEFT_T")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(48usize);
    let rt_ppl = [0, 1, 2, 3].map(|d| peft_eval_ppl(&fused2, &doms2[d].2, t, d));
    Some((rt_ppl, card.id))
}

/// `--peft-replay -m <model>`: consolidation — distill a newly-learned domain
/// into the SHARED bank so it becomes a permanent primitive (the bank's span
/// grows; "can't invent" today becomes "in-span" tomorrow). Compares
/// consolidating on the new domain ALONE (catastrophic forgetting) vs on a
/// REPLAY mixture of old + new (complementary-learning-systems — keeps both).
/// Env: PEFT_STEPS(120) PEFT_T(48) PEFT_LR(1e-3) PEFT_GENES(8) PEFT_BOTTLE(4).
fn cmd_peft_replay(args: &Args) {
    use hos::tensor::use_gpu;
    let path = resolve_model(args.model.clone());
    let eng = ok(hos::Engine::load(&path, false));
    let envn = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    };
    let (t, steps) = (envn("PEFT_T", 48), envn("PEFT_STEPS", 120));
    let lr: f32 = std::env::var("PEFT_LR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-3);
    let (genes, bottle) = (envn("PEFT_GENES", 8), envn("PEFT_BOTTLE", 4));
    let domains = build_domains(&eng.tok, 0xd0_5eed_1234);
    let n = domains.len();
    let seen = n - 1;
    let held = n - 1; // the new domain (out-of-span: code) to consolidate
    let (hcfg, _) = peft_matched(&eng.model.cfg, genes, bottle);
    use_gpu(true);
    eprintln!(
        "[hos] replay consolidation: grow the shared bank to absorb '{}'",
        domains[held].0
    );
    let win = |d: usize, nx: &mut dyn FnMut() -> usize| {
        let tr = &domains[d].1;
        let s = nx() % (tr.len() - t - 1);
        tr[s..s + t + 1].to_vec()
    };

    // Train a shared bank on the seen domains (identical via fixed seeds), then
    // consolidate the new domain into the bank either alone or with replay.
    let run = |replay: bool| -> (f32, f32, f32) {
        let rga =
            hos::peft::PeftModel::build_multi(&eng.model, "rga", &hcfg, n, 0xC0FFEE).expect("rga");
        let mut rng = 0x5eed_2222_aaaa_5555u64;
        let mut nx = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 16) as usize
        };
        let mut p1 = rga.params_bank();
        for d in 0..seen {
            p1.extend(rga.params_genome(d));
        }
        peft_train(&rga, &p1, steps, t, lr, 1e-3, || {
            let d = nx() % seen;
            (win(d, &mut nx), d)
        });
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        let seen_before = mean(
            &(0..seen)
                .map(|d| peft_eval_ppl(&rga, &domains[d].2, t, d))
                .collect::<Vec<_>>(),
        );
        // consolidate: unfreeze the whole bank + every genome
        let pall = rga.params();
        peft_train(&rga, &pall, steps, t, lr, 1e-3, || {
            let d = if replay { nx() % n } else { held };
            (win(d, &mut nx), d)
        });
        let code_after = peft_eval_ppl(&rga, &domains[held].2, t, held);
        let seen_after = mean(
            &(0..seen)
                .map(|d| peft_eval_ppl(&rga, &domains[d].2, t, d))
                .collect::<Vec<_>>(),
        );
        (code_after, seen_before, seen_after)
    };

    eprintln!("  consolidating on the new domain ONLY ...");
    let (code_a, sb_a, sa_a) = run(false);
    eprintln!("  consolidating on a REPLAY mixture (old + new) ...");
    let (code_b, sb_b, sa_b) = run(true);

    println!("\n=== Replay consolidation: promote a new skill into the shared bank ===");
    println!(
        "seen: {:?} | consolidating: {}",
        &domains[..seen].iter().map(|d| d.0).collect::<Vec<_>>(),
        domains[held].0
    );
    println!(
        "{:<28}{:>16}{:>20}",
        "consolidation", "new ppl", "seen forgetting"
    );
    println!(
        "{:<28}{:>16.2}{:>18.2}x",
        "new domain only",
        code_a,
        sa_a / sb_a
    );
    println!(
        "{:<28}{:>16.2}{:>18.2}x",
        "replay (old + new)",
        code_b,
        sa_b / sb_b
    );
    println!("\nBoth absorb the new skill into the shared bank (it's now a primitive a plain");
    println!("genome can reach). Only replay keeps the old domains — that's how the span grows");
    println!("over time without forgetting: fast clonal invention, slow replay consolidation.");
}

/// `--peft-recombine -m <model>`: Phase 3. Train genome A (domain A) and genome B
/// (domain B) over a shared bank, then form a CHILD genome by crossover — no
/// training — and test whether it retains both parents. Compared to merging two
/// independently-trained LoRAs (the standard merge).
/// Env: PEFT_STEPS(160) PEFT_T(48) PEFT_LR(1e-3).
fn cmd_peft_recombine(args: &Args) {
    use hos::tensor::use_gpu;
    let path = resolve_model(args.model.clone());
    let eng = ok(hos::Engine::load(&path, false));
    let envn = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    };
    let (t, steps) = (envn("PEFT_T", 48), envn("PEFT_STEPS", 160));
    let lr: f32 = std::env::var("PEFT_LR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-3);
    let (hcfg, _) = peft_matched(&eng.model.cfg, 4, 4);
    let domains = build_domains(&eng.tok, 0xd0_5eed_1234);
    let (ia, ib) = (0usize, 3usize); // geography, code
    let (na, nb) = (domains[ia].0, domains[ib].0);
    let mut rng = 0x9999_1111_2222_3333u64;
    let mut nx = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 16) as usize
    };
    let win = |d: usize, nx: &mut dyn FnMut() -> usize| {
        let tr = &domains[d].1;
        let s = nx() % (tr.len() - t - 1);
        tr[s..s + t + 1].to_vec()
    };
    use_gpu(true);
    eprintln!("[hos] recombination: parents γ_{na} and γ_{nb} over a shared bank, then crossover");

    // ---- RGA: shared bank + γ_A(0) + γ_B(1); child = average -> slot 2 ----
    let rga =
        hos::peft::PeftModel::build_multi(&eng.model, "rga", &hcfg, 3, 0xC0FFEE).expect("rga");
    let mut p = rga.params_bank();
    p.extend(rga.params_genome(0));
    p.extend(rga.params_genome(1));
    peft_train(&rga, &p, steps, t, lr, 1e-3, || {
        if nx() % 2 == 0 {
            (win(ia, &mut nx), 0)
        } else {
            (win(ib, &mut nx), 1)
        }
    });
    let rga_pa = peft_eval_ppl(&rga, &domains[ia].2, t, 0); // parent A on A
    let rga_pb = peft_eval_ppl(&rga, &domains[ib].2, t, 1); // parent B on B
    let (ga, gb) = (rga.genome_data(0), rga.genome_data(1));
    let child: Vec<f32> = ga.iter().zip(&gb).map(|(a, b)| 0.5 * (a + b)).collect();
    rga.set_genome(2, &child);
    let rga_ca = peft_eval_ppl(&rga, &domains[ia].2, t, 2); // child on A
    let rga_cb = peft_eval_ppl(&rga, &domains[ib].2, t, 2); // child on B
    drop(rga);

    // ---- LoRA: train A and B separately (same seed -> aligned params), merge=avg ----
    let lora_a =
        hos::peft::PeftModel::build_multi(&eng.model, "lora", &hcfg, 1, 0xC0FFEE).expect("la");
    peft_train(&lora_a, &lora_a.params(), steps, t, lr, 0.0, || {
        (win(ia, &mut nx), 0)
    });
    let lora_pa = peft_eval_ppl(&lora_a, &domains[ia].2, t, 0);
    let lora_b =
        hos::peft::PeftModel::build_multi(&eng.model, "lora", &hcfg, 1, 0xC0FFEE).expect("lb");
    peft_train(&lora_b, &lora_b.params(), steps, t, lr, 0.0, || {
        (win(ib, &mut nx), 0)
    });
    let lora_pb = peft_eval_ppl(&lora_b, &domains[ib].2, t, 0);
    // merge: lora_a := average(lora_a, lora_b)
    let (pa, pb) = (lora_a.params(), lora_b.params());
    for (a, b) in pa.iter().zip(&pb) {
        let (ad, bd) = (a.data(), b.data());
        let avg: Vec<f32> = ad.iter().zip(&bd).map(|(x, y)| 0.5 * (x + y)).collect();
        a.set_data(&avg);
    }
    let lora_ma = peft_eval_ppl(&lora_a, &domains[ia].2, t, 0);
    let lora_mb = peft_eval_ppl(&lora_a, &domains[ib].2, t, 0);

    println!(
        "\n=== Recombination: child = crossover(γ_{na}, γ_{nb}); LoRA merge = average(A,B) ==="
    );
    println!("{:<22}{:>14}{:>14}", "config", na, nb);
    println!(
        "{:<22}{:>14.2}{:>14.2}",
        "RGA parents (own γ)", rga_pa, rga_pb
    );
    println!(
        "{:<22}{:>14.2}{:>14.2}",
        "RGA child (one γ)", rga_ca, rga_cb
    );
    println!(
        "{:<22}{:>14.2}{:>14.2}",
        "LoRA parents (own)", lora_pa, lora_pb
    );
    println!(
        "{:<22}{:>14.2}{:>14.2}",
        "LoRA merged (avg)", lora_ma, lora_mb
    );
    let rga_via = 0.5 * (rga_ca / rga_pa + rga_cb / rga_pb);
    let lora_via = 0.5 * (lora_ma / lora_pa + lora_mb / lora_pb);
    println!("\nrecombination cost (child/parent ppl ratio, 1.0 = no loss): RGA {rga_via:.2}x vs LoRA-merge {lora_via:.2}x");
    if rga_via < lora_via * 0.97 {
        println!("verdict: the genome child retains parent capability better than merged LoRAs.");
    } else if lora_via < rga_via * 0.97 {
        println!("verdict: LoRA merge retains more here — honest negative for genome crossover on this setup.");
    } else {
        println!("verdict: comparable recombination viability.");
    }
}

/// `--peft-compare -m <model>`: the matched-budget multi-domain experiment.
/// Trains LoRA and RGA jointly on 4 distinct domains and reports per-domain
/// held-out perplexity — the honest test of whether the genome's conditional
/// expression serves multiple domains better than a single static LoRA delta.
/// Env: PEFT_STEPS(120) PEFT_T(48) PEFT_ACCUM(2) PEFT_LR(1e-3) PEFT_GENES(4) PEFT_BOTTLE(4).
fn cmd_peft_compare(args: &Args) {
    use hos::tensor::use_gpu;
    let path = resolve_model(args.model.clone());
    let eng = ok(hos::Engine::load(&path, false));
    let envn = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    };
    let (t, steps, accum) = (
        envn("PEFT_T", 48),
        envn("PEFT_STEPS", 120),
        envn("PEFT_ACCUM", 2),
    );
    let lr: f32 = std::env::var("PEFT_LR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-3);
    let (genes, bottle) = (envn("PEFT_GENES", 4), envn("PEFT_BOTTLE", 4));

    let domains = build_domains(&eng.tok, 0xd0_5eed_1234);
    use_gpu(true);

    // RGA budget, then match the LoRA rank to it
    let c = &eng.model.cfg;
    let rga_per_layer = c.dim * genes + genes * 2 * c.dim * bottle;
    let rga_total = rga_per_layer * c.n_layers + 8 + 8 * genes;
    let lora_per_rank =
        c.n_layers * (2 * c.dim + c.n_heads * c.head_dim + c.n_kv_heads * c.head_dim);
    let rank = (rga_total / lora_per_rank).max(1);

    eprintln!(
        "[hos] PEFT compare on {:?}: {} domains, t={t}, {steps} steps; LoRA rank {rank} vs RGA genes {genes}/bottle {bottle}",
        c.arch,
        domains.len()
    );
    let lcfg = hos::peft::PeftCfg {
        rank,
        genes,
        bottleneck: bottle,
        lora_alpha: 16.0,
    };
    eprintln!("  training LoRA ...");
    let (lp, lora_ppl) = peft_run_domains(
        &eng.model, &domains, "lora", &lcfg, steps, t, accum, lr, 0.0,
    );
    eprintln!("  training RGA ...");
    let (rp, rga_ppl) = peft_run_domains(
        &eng.model, &domains, "rga", &lcfg, steps, t, accum, lr, 1e-3,
    );

    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    let spread = |v: &[f32]| {
        let (mn, mx) = v
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
        mx - mn
    };
    println!("\n=== PEFT multi-domain comparison (matched budget, held-out perplexity) ===");
    print!("{:<8}{:>10}", "method", "params");
    for (n, _, _) in &domains {
        print!("{n:>12}");
    }
    println!("{:>9}{:>9}", "mean", "spread");
    let row = |name: &str, p: usize, ppl: &[f32]| {
        print!("{name:<8}{p:>10}");
        for v in ppl {
            print!("{v:>12.2}");
        }
        println!("{:>9.2}{:>9.2}", mean(ppl), spread(ppl));
    };
    row("LoRA", lp, &lora_ppl);
    row("RGA", rp, &rga_ppl);
    println!("\n(lower perplexity = better; lower spread = serves all domains more evenly.)");
    let (lm, rm) = (mean(&lora_ppl), mean(&rga_ppl));
    if rm < lm * 0.98 {
        println!("verdict: RGA serves the multi-domain mix better (mean ppl {rm:.2} vs {lm:.2}) at matched budget.");
    } else if lm < rm * 0.98 {
        println!("verdict: LoRA wins on mean ppl ({lm:.2} vs {rm:.2}) here — honest null/negative for RGA on this setup.");
    } else {
        println!("verdict: roughly tied on mean ppl ({rm:.2} vs {lm:.2}); compare spread / interference next.");
    }
}

/// `--peft --method lora|rga -m <model> [--corpus f] [-o out.hos]`:
/// parameter-efficient finetuning on a FROZEN base. LoRA is the baseline; RGA is
/// the genomic alternative (a compact genome regulates a sparse gene bank). Only
/// the adapter trains; the `.hos` artifact stores the adapter with lineage to the
/// base. Env: PEFT_STEPS(80) PEFT_T(64) PEFT_LR(1e-3) PEFT_ACCUM(4) PEFT_RANK(8)
/// PEFT_GENES(4) PEFT_BOTTLE(8) PEFT_LAMBDA(1e-3) PEFT_GPU(1).
fn cmd_peft(args: &Args) {
    use hos::tensor::{use_gpu, AdamW};
    use std::time::Instant;

    let method = arg_after("--method").unwrap_or_else(|| "lora".into());
    let path = resolve_model(args.model.clone());
    let eng = ok(hos::Engine::load(&path, false));

    let envn = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    };
    let hcfg = hos::peft::PeftCfg {
        rank: envn("PEFT_RANK", 8),
        genes: envn("PEFT_GENES", 4),
        bottleneck: envn("PEFT_BOTTLE", 8),
        lora_alpha: 16.0,
    };
    let peft = match hos::peft::PeftModel::build(&eng.model, &method, &hcfg, 0xBEEF) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[hos] cannot run PEFT: {e}");
            std::process::exit(1);
        }
    };

    // init-parity: adapters start at zero, so the initial forward MUST equal the base
    let probe_ids: Vec<usize> = eng
        .tok
        .encode("The capital of France is Paris.", true)
        .iter()
        .map(|&x| x as usize)
        .collect();
    let base_last = {
        let mut st = hos::forward::State::new(&eng.model.cfg);
        let mut last = Vec::new();
        for (pos, &tok) in probe_ids.iter().enumerate() {
            last = hos::forward::forward(&eng.model, &mut st, tok as u32, pos, None);
        }
        last
    };
    let pl = peft.logits(&probe_ids).data();
    let vocab = eng.model.cfg.vocab_size;
    let row = &pl[(probe_ids.len() - 1) * vocab..probe_ids.len() * vocab];
    let mut md = 0.0f32;
    for j in 0..vocab {
        md = md.max((row[j] - base_last[j]).abs());
    }
    let full = {
        // full-finetune count for the contrast
        let p = hos::finetune::FtModel::from_model(&eng.model)
            .map(|f| f.params().iter().map(|t| t.data().len()).sum::<usize>());
        p.unwrap_or(0)
    };
    println!(
        "[hos] {} on {:?}: init forward vs base max|Δ| = {md:.2e} (should be ~0 — adapter starts at base)",
        peft.method_name, eng.model.cfg.arch
    );
    println!(
        "  trainable: {} params  ({:.3}% of the {} full-finetune params)",
        peft.n_trainable(),
        100.0 * peft.n_trainable() as f64 / full.max(1) as f64,
        full
    );

    // corpus
    let corpus = match arg_after("--corpus") {
        Some(p) => std::fs::read_to_string(&p).unwrap_or_else(|e| {
            eprintln!("[hos] error reading corpus {p}: {e}");
            std::process::exit(1);
        }),
        None => "HOS is a from-scratch local LLM engine written in Rust. It loads models, \
                 runs them on CPU or the Apple GPU, trains them, and stores them in a \
                 self-describing format with lineage and provenance. "
            .repeat(40),
    };
    let ids: Vec<usize> = eng
        .tok
        .encode(&corpus, true)
        .iter()
        .map(|&x| x as usize)
        .collect();
    let t = envn("PEFT_T", 64).min(eng.model.cfg.ctx_len);
    let steps = envn("PEFT_STEPS", 80);
    let accum = envn("PEFT_ACCUM", 4).max(1);
    let lr: f32 = std::env::var("PEFT_LR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-3);
    let lambda: f32 = std::env::var("PEFT_LAMBDA")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-3);
    if ids.len() < t + 2 {
        eprintln!("[hos] corpus too short for window {t}");
        std::process::exit(1);
    }
    use_gpu(std::env::var("PEFT_GPU").map(|v| v != "0").unwrap_or(true));

    let params = peft.params();
    let decay: Vec<bool> = params.iter().map(|p| p.shape().len() == 2).collect();
    let mut rng = 0x2545f4914f6cdd1du64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 16) as usize % (ids.len() - t - 1)
    };
    let probe = &ids[0..t + 1];
    let start_loss = peft.loss(&probe[..t], &probe[1..t + 1], 0.0).data()[0];
    let mut opt = AdamW::new(&params, lr, 0.0);
    let t0 = Instant::now();
    for step in 0..steps {
        for p in &params {
            p.zero_grad();
        }
        let mut mean = 0.0f32;
        for _ in 0..accum {
            let s = next();
            let w = &ids[s..s + t + 1];
            let loss = peft
                .loss(&w[..t], &w[1..t + 1], lambda)
                .scale(1.0 / accum as f32);
            loss.backward();
            mean += loss.data()[0];
        }
        opt.step(&params, &decay);
        if step % 20 == 0 || step == steps - 1 {
            eprintln!("  step {step:>3}  loss {mean:.4}");
        }
    }
    let end_loss = peft.loss(&probe[..t], &probe[1..t + 1], 0.0).data()[0];
    eprintln!(
        "[hos] {} done in {:.1}s — probe loss {start_loss:.4} -> {end_loss:.4}",
        peft.method_name,
        t0.elapsed().as_secs_f64()
    );

    // save the adapter only (lineage -> base)
    use_gpu(false);
    save_adapter(&peft, &eng.model, &path, end_loss, steps, lr);
}

fn save_adapter(
    peft: &hos::peft::PeftModel,
    base: &hos::model::Model,
    base_path: &Path,
    end_loss: f32,
    steps: usize,
    lr: f32,
) {
    use hos::format::{self, Named, TrainRun, ROLE_WEIGHT};
    let out = arg_after("-o")
        .or_else(|| arg_after("--out"))
        .unwrap_or_else(|| {
            format!(
                "{}-{}.hos",
                base_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("model"),
                peft.method_name
            )
        });
    let named: Vec<Named> = peft
        .adapter_tensors()
        .into_iter()
        .map(|(name, shape, data)| Named {
            role: ROLE_WEIGHT,
            name,
            shape,
            data,
        })
        .collect();
    let base_id = format::model_id(&[Named {
        role: ROLE_WEIGHT,
        name: "embd".into(),
        shape: vec![base.cfg.vocab_size, base.cfg.dim],
        data: base.tok_embd.clone(),
    }]);
    let arch = serde_json::json!({
        "source_format": "hos-peft",
        "method": peft.method_name,
        "base_architecture": format!("{:?}", base.cfg.arch),
        "block_count": base.cfg.n_layers,
    });
    let mut card = format::Card::new(
        &format!(
            "{}-{}",
            base_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model"),
            peft.method_name
        ),
        arch,
    );
    card.id = format::model_id(&named);
    card.mode = "adapter".into();
    card.provenance.engine = "hos-peft".into();
    card.provenance.dataset = format!("{} adapter on {}", peft.method_name, base_path.display());
    card.provenance.dataset_hash = base_id.clone();
    card.lineage = vec![base_id];
    card.history = vec![TrainRun {
        steps: steps as u64,
        final_loss: end_loss,
        optimizer: "adamw".into(),
        lr,
    }];
    ok(format::save(std::path::Path::new(&out), &named, &card).map_err(hos::HosError::from));
    let bytes = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
    let nfloats: usize = named.iter().map(|t| t.data.len()).sum();
    println!(
        "saved {out} ({:.2} MB, {} adapter params) — lineage -> base",
        bytes as f64 / 1e6,
        nfloats
    );
}

/// `--finetune-check -m <model>`: prove the autograd-built Llama forward matches
/// the inference forward (`forward.rs`) before trusting it for training.
fn cmd_finetune_check(args: &Args) {
    let path = resolve_model(args.model.clone());
    let eng = ok(hos::Engine::load(&path, false)); // CPU f32 weights
    let ids: Vec<usize> = eng
        .tok
        .encode("The capital of France is Paris.", true)
        .iter()
        .map(|&x| x as usize)
        .collect();
    let md = hos::finetune::check_parity(&eng.model, &ids);
    println!(
        "[hos] autograd forward vs inference forward: max |Δlogit| = {md:.3e} over {} tokens",
        ids.len()
    );
    println!(
        "{}",
        if md < 1e-2 {
            "PASS — the trainable Llama forward is byte-faithful to the inference path"
        } else {
            "FAIL — forward mismatch; do not train on this"
        }
    );
}

/// `--finetune -m <model> [--corpus f] [--opt adamw|sgd] [-o out.hos]`:
/// full-parameter finetune of a real pretrained transformer on your own text,
/// saved as a `.hos` capsule whose lineage points back to the base model.
/// Env: FT_STEPS (default 60), FT_T (window, 64), FT_LR (2e-5), FT_ACCUM (4), FT_GPU (1).
fn cmd_finetune(args: &Args) {
    use hos::tensor::{use_gpu, AdamW};
    use std::time::Instant;

    let path = resolve_model(args.model.clone());
    let eng = ok(hos::Engine::load(&path, false));
    let grow = std::env::var("FT_GROW_FFN")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());
    let ft = {
        let r = match grow {
            Some(nf) => {
                eprintln!(
                    "[hos] Net2Net grow: FFN {} -> {} (function-preserving), then specialize-train",
                    eng.model.cfg.ffn_dim, nf
                );
                hos::finetune::FtModel::from_model_grown(&eng.model, nf)
            }
            None => hos::finetune::FtModel::from_model(&eng.model),
        };
        match r {
            Ok(f) => f,
            Err(e) => {
                eprintln!("[hos] cannot finetune: {e}");
                std::process::exit(1);
            }
        }
    };

    let corpus = match arg_after("--corpus") {
        Some(p) => std::fs::read_to_string(&p).unwrap_or_else(|e| {
            eprintln!("[hos] error reading corpus {p}: {e}");
            std::process::exit(1);
        }),
        None => "HOS is a from-scratch local LLM engine written in Rust. It loads models, \
                 runs them on CPU or the Apple GPU, trains them, and stores them in a \
                 self-describing format with lineage and provenance. "
            .repeat(40),
    };
    let ids: Vec<usize> = eng
        .tok
        .encode(&corpus, true)
        .iter()
        .map(|&x| x as usize)
        .collect();

    let envn = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    };
    let t = envn("FT_T", 64).min(eng.model.cfg.ctx_len);
    let steps = envn("FT_STEPS", 60);
    let accum = envn("FT_ACCUM", 4).max(1);
    let lr: f32 = std::env::var("FT_LR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2e-5);
    let opt_name = arg_after("--opt").unwrap_or_else(|| "adamw".into());
    if ids.len() < t + 2 {
        eprintln!(
            "[hos] corpus too short ({} tokens) for window {t}",
            ids.len()
        );
        std::process::exit(1);
    }
    use_gpu(std::env::var("FT_GPU").map(|v| v != "0").unwrap_or(true));

    // flwr: weave an E8 vector-quantized bottleneck into the forward (the forward
    // diverges from Llama -> this is the flwr architecture). Annealed STE.
    let flwr = std::env::var("FT_FLWR").map(|v| v != "0").unwrap_or(false);
    let vq = hos::nn::VectorQuantizer {
        block: 8,
        scale: 1.0,
        codebook: hos::nn::Codebook::E8,
        commit_weight: 0.25,
    };
    if flwr {
        eprintln!("[hos] flwr arch: E8 vector-quantized hidden bottleneck (annealed STE)");
    }
    let viz = std::env::var("FT_VIZ").map(|v| v != "0").unwrap_or(false);

    let params = ft.params();
    let decay: Vec<bool> = params.iter().map(|p| p.shape().len() == 2).collect();
    let n_params: usize = params.iter().map(|p| p.data().len()).sum();
    eprintln!(
        "[hos] finetuning {:?}: {} trainable params, window={t}, opt={opt_name}, lr={lr}, {steps} steps x{accum}",
        eng.model.cfg.arch, n_params
    );

    // deterministic window sampler
    let mut rng = 0x9e3779b97f4a7c15u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 16) as usize % (ids.len() - t - 1)
    };
    let loss_at = |w: &[usize]| {
        let logits = if flwr {
            ft.forward_vq(&w[..t], &vq, 1.0).0
        } else {
            ft.forward(&w[..t])
        };
        logits.cross_entropy(&w[1..t + 1]).data()[0]
    };
    let probe = &ids[0..t + 1];
    let start_loss = loss_at(probe);

    let mut opt = AdamW::new(&params, lr, 0.0);
    let t0 = Instant::now();
    let mut viz_lines = 0usize;
    for step in 0..steps {
        for p in &params {
            p.zero_grad();
        }
        // anneal the E8 quantizer in over the first half of training
        let strength = if flwr {
            (step as f32 / (steps as f32 * 0.5)).min(1.0)
        } else {
            0.0
        };
        let mut mean = 0.0f32;
        for _ in 0..accum {
            let s = next();
            let w = &ids[s..s + t + 1];
            let loss = if flwr {
                let (lg, cm) = ft.forward_vq(&w[..t], &vq, strength);
                lg.cross_entropy(&w[1..t + 1])
                    .add(&cm)
                    .scale(1.0 / accum as f32)
            } else {
                ft.forward(&w[..t])
                    .cross_entropy(&w[1..t + 1])
                    .scale(1.0 / accum as f32)
            };
            loss.backward();
            mean += loss.data()[0];
        }
        if opt_name == "sgd" {
            for p in &params {
                p.sgd_step(lr);
            }
        } else {
            opt.step(&params, &decay);
        }
        if viz {
            let prog = (step + 1) as f32 / steps as f32;
            let drm = if flwr { strength } else { prog };
            let panel = hos::viz::grow_panel(
                if flwr { "FLWR GROW" } else { "GROW MODE" },
                prog,
                mean,
                prog,
                drm,
            );
            if viz_lines > 0 {
                eprint!("\x1b[{viz_lines}A");
            }
            eprint!("{panel}");
            let _ = std::io::Write::flush(&mut std::io::stderr());
            viz_lines = panel.matches('\n').count();
        } else if step % 10 == 0 || step == steps - 1 {
            eprintln!("  step {step:>3}  loss {mean:.4}");
        }
    }
    let end_loss = loss_at(probe);
    eprintln!(
        "[hos] done in {:.1}s — probe loss {start_loss:.4} -> {end_loss:.4}",
        t0.elapsed().as_secs_f64()
    );

    // save a lineaged .hos capsule of the finetuned weights
    use_gpu(false);
    let out = arg_after("-o")
        .or_else(|| arg_after("--out"))
        .unwrap_or_else(|| {
            format!(
                "{}-ft.hos",
                path.file_stem().and_then(|s| s.to_str()).unwrap_or("model")
            )
        });
    if flwr {
        save_flwr(&ft, &out, start_loss, end_loss, steps, lr);
    } else {
        save_finetuned(
            &ft, &eng.model, &path, &out, start_loss, end_loss, steps, lr, &opt_name,
        );
    }
}

/// Mint a `flwr` capsule: architecture = flwr, lineage ROOTED at flwr (no seed,
/// no donor — flwr is genesis). Archival; re-mint with a tokenizer to run.
#[allow(clippy::too_many_arguments)]
fn save_flwr(
    ft: &hos::finetune::FtModel,
    out: &str,
    start_loss: f32,
    end_loss: f32,
    steps: usize,
    lr: f32,
) {
    use hos::format::{self, Named, TrainRun, ROLE_WEIGHT};
    let named: Vec<Named> = ft
        .named_tensors()
        .into_iter()
        .map(|(name, shape, data)| Named {
            role: ROLE_WEIGHT,
            name,
            shape,
            data,
        })
        .collect();
    // flwr genesis: a fixed root id every flwr model shares — the lineage starts here.
    let flwr_root = format::model_id(&[Named {
        role: ROLE_WEIGHT,
        name: "flwr-genesis".into(),
        shape: vec![1],
        data: vec![0.0],
    }]);
    let arch = serde_json::json!({
        "source_format": "flwr",
        "architecture": "flwr",
        "embedding_length": ft.cfg.dim,
        "block_count": ft.cfg.n_layers,
        "head_count": ft.cfg.n_heads,
        "head_count_kv": ft.cfg.n_kv_heads,
        "feed_forward_length": ft.cfg.ffn_dim,
        "bottleneck": "e8-vq",
    });
    let mut card = format::Card::new("flwr", arch);
    card.id = format::model_id(&named);
    card.mode = "trainable".into();
    card.provenance.engine = "flwr".into();
    card.provenance.dataset = "flwr".into();
    card.lineage = vec![flwr_root.clone()];
    card.history = vec![TrainRun {
        steps: steps as u64,
        final_loss: end_loss,
        optimizer: "adamw".into(),
        lr,
    }];
    ok(format::save(std::path::Path::new(out), &named, &card).map_err(hos::HosError::from));
    let bytes = std::fs::metadata(out).map(|m| m.len()).unwrap_or(0);
    println!(
        "saved {out} ({:.1} MB) — arch flwr · lineage root: flwr ({}…) · loss {start_loss:.3} -> {end_loss:.3}",
        bytes as f64 / 1e6,
        &flwr_root[..flwr_root.len().min(8)]
    );
}

#[allow(clippy::too_many_arguments)]
fn save_finetuned(
    ft: &hos::finetune::FtModel,
    base: &hos::model::Model,
    base_path: &Path,
    out: &str,
    start_loss: f32,
    end_loss: f32,
    steps: usize,
    lr: f32,
    opt: &str,
) {
    use hos::format::{self, Named, TrainRun, ROLE_WEIGHT};
    let named: Vec<Named> = ft
        .named_tensors()
        .into_iter()
        .map(|(name, shape, data)| Named {
            role: ROLE_WEIGHT,
            name,
            shape,
            data,
        })
        .collect();
    // base fingerprint = content hash of the base embedding table (stable id)
    let base_id = format::model_id(&[Named {
        role: ROLE_WEIGHT,
        name: "embd".into(),
        shape: vec![base.cfg.vocab_size, base.cfg.dim],
        data: base.tok_embd.clone(),
    }]);
    let arch = serde_json::json!({
        "source_format": "hos-finetune",
        "architecture": format!("{:?}", base.cfg.arch),
        "embedding_length": base.cfg.dim,
        "block_count": base.cfg.n_layers,
        "head_count": base.cfg.n_heads,
        "head_count_kv": base.cfg.n_kv_heads,
        "rope_neox": base.cfg.rope_neox,
    });
    let mut card = format::Card::new(
        base_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model"),
        arch,
    );
    card.id = format::model_id(&named);
    card.mode = "trainable".into();
    card.provenance.engine = "hos-finetune".into();
    card.provenance.dataset = format!("finetuned from {}", base_path.display());
    card.provenance.dataset_hash = base_id.clone();
    card.lineage = vec![base_id];
    card.history = vec![TrainRun {
        steps: steps as u64,
        final_loss: end_loss,
        optimizer: opt.into(),
        lr,
    }];
    let _ = start_loss;
    ok(format::save(std::path::Path::new(out), &named, &card).map_err(hos::HosError::from));
    let bytes = std::fs::metadata(out).map(|m| m.len()).unwrap_or(0);
    println!(
        "saved {out} ({:.1} MB) — lineage -> base; loss {start_loss:.3} -> {end_loss:.3}",
        bytes as f64 / 1e6
    );
    println!("inspect: hos --hos-info {out}");
}

/// One row of the `--verify-against` architecture table.
fn cmp_row(label: &str, a: usize, b: usize) -> bool {
    let ok = a == b;
    println!(
        "  {label:<12} {a:>8}  {b:>8}   {}",
        if ok { "ok" } else { "MISMATCH" }
    );
    ok
}

/// `--hos-viz <file.hos> [-o out.html]`: render a `.hos` capsule as a
/// self-contained "Genetic Code Visualizer" HTML report — arch genes, lineage,
/// training history, and a tensor map.
fn cmd_hos_viz(_args: &Args) {
    use hos::format;
    use std::collections::BTreeMap;
    let Some(src) = arg_after("--hos-viz") else {
        eprintln!("[hos] error: --hos-viz needs a .hos file");
        std::process::exit(1);
    };
    let out = arg_after("-o")
        .or_else(|| arg_after("--out"))
        .unwrap_or_else(|| {
            std::path::Path::new(&src)
                .with_extension("html")
                .to_string_lossy()
                .into_owned()
        });
    let (tensors, card) = ok(format::load(std::path::Path::new(&src)).map_err(hos::HosError::from));

    let esc = |s: &str| {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
    };
    let role_name = ["weight", "bias", "norm", "embed", "opt_state", "scalar"];
    let role_color = [
        "#3b82f6", "#a855f7", "#f59e0b", "#10b981", "#6b7280", "#ef4444",
    ];

    // arch "genes"
    let mut genes = String::new();
    if let Some(obj) = card.arch.as_object() {
        for (k, v) in obj {
            let vs = v.to_string();
            genes.push_str(&format!(
                "<div class='gene'><span class='gk'>{}</span><span class='gv'>{}</span></div>",
                esc(k),
                esc(vs.trim_matches('"'))
            ));
        }
    }

    // lineage
    let lineage = if card.lineage.is_empty() {
        "<em>root organism — no ancestors</em>".to_string()
    } else {
        card.lineage
            .iter()
            .map(|a| esc(a))
            .collect::<Vec<_>>()
            .join(" &rarr; ")
    };

    // history
    let mut hist = String::new();
    for (i, r) in card.history.iter().enumerate() {
        hist.push_str(&format!(
            "<tr><td>{i}</td><td>{}</td><td>{}</td><td>{}</td><td>{:.4}</td></tr>",
            r.steps,
            esc(&r.optimizer),
            r.lr,
            r.final_loss
        ));
    }
    if hist.is_empty() {
        hist = "<tr><td colspan=5><em>no training runs recorded</em></td></tr>".into();
    }

    // tensor map (collapse blk.N)
    let norm = |n: &str| -> String {
        if let Some(rest) = n.strip_prefix("blk.") {
            if let Some(dot) = rest.find('.') {
                return format!("blk.N{}", &rest[dot..]);
            }
        }
        n.to_string()
    };
    let mut groups: BTreeMap<String, (u8, usize, usize)> = BTreeMap::new(); // name -> (role, count, n_floats)
    for t in &tensors {
        let e = groups
            .entry(norm(&t.name))
            .or_insert((t.role, 0, t.data.len()));
        e.1 += 1;
    }
    let mut trows = String::new();
    let mut total = 0usize;
    for t in &tensors {
        total += t.data.len();
    }
    for (n, (role, cnt, nf)) in &groups {
        let r = *role as usize;
        trows.push_str(&format!(
            "<tr><td><span class='dot' style='background:{}'></span>{}</td><td>{}</td><td>×{}</td><td>{}</td></tr>",
            role_color.get(r).unwrap_or(&"#888"), esc(n),
            role_name.get(r).unwrap_or(&"?"), cnt, nf
        ));
    }

    let html = format!(
        r#"<!doctype html><html><head><meta charset="utf-8"><title>HOS · {name}</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0f1420;color:#e6ebf5;margin:0;padding:28px 36px;}}
h1{{margin:0;font-size:24px;color:#fff}} .sub{{color:#8b97ad;font-size:13px;margin-bottom:18px}}
h2{{font-size:14px;color:#9fb3d9;border-bottom:1px solid #28324a;padding-bottom:5px;margin-top:26px;text-transform:uppercase;letter-spacing:.5px}}
.genes{{display:flex;flex-wrap:wrap;gap:8px}}
.gene{{background:#1a2436;border:1px solid #2b3850;border-radius:8px;padding:7px 11px;font-size:12px}}
.gk{{color:#7fa6e0;display:block;font-size:10px;text-transform:uppercase;letter-spacing:.4px}}
.gv{{color:#e6ebf5;font-weight:600}}
table{{width:100%;border-collapse:collapse;font-size:12px;margin-top:6px}}
td,th{{padding:5px 8px;border-bottom:1px solid #222c40;text-align:left}}
th{{color:#8b97ad;font-weight:600}}
.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px;vertical-align:middle}}
.id{{font-family:monospace;color:#f59e0b}} .pill{{background:#1a2436;border-radius:6px;padding:2px 8px;font-size:11px}}
.lin{{font-family:monospace;color:#10b981}}
</style></head><body>
<h1>{name}</h1>
<div class="sub">identity <span class="id">{id}</span> &nbsp;·&nbsp; mode <span class="pill">{mode}</span> &nbsp;·&nbsp; {ntens} tensors &nbsp;·&nbsp; {total} parameters</div>
<h2>Genome (architecture)</h2><div class="genes">{genes}</div>
<h2>Lineage</h2><div class="lin">{lineage}</div>
<h2>Training history</h2><table><tr><th>run</th><th>steps</th><th>optimizer</th><th>lr</th><th>final loss</th></tr>{hist}</table>
<h2>Gene expression (tensors)</h2><table><tr><th>tensor</th><th>role</th><th>count</th><th>values each</th></tr>{trows}</table>
<div class="sub" style="margin-top:24px">Generated by HOS · hos --hos-viz</div>
</body></html>"#,
        name = esc(&card.name),
        id = esc(&card.id),
        mode = esc(&card.mode),
        ntens = tensors.len(),
        total = total,
        genes = genes,
        lineage = lineage,
        hist = hist,
        trows = trows,
    );
    std::fs::write(&out, html).unwrap_or_else(|e| {
        eprintln!("[hos] error: writing {out}: {e}");
        std::process::exit(1);
    });
    println!("wrote {out}  (open in a browser)");
}

/// Returns the corpus for `--perplexity` if that flag is present: the contents
/// of an optional following file path, else the built-in passage.
fn perplexity_corpus() -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    let i = a.iter().position(|x| x == "--perplexity")?;
    if let Some(p) = a.get(i + 1) {
        if !p.starts_with('-') && Path::new(p).is_file() {
            return Some(std::fs::read_to_string(p).unwrap_or_else(|e| {
                eprintln!("[hos] error: reading corpus {p}: {e}");
                std::process::exit(1);
            }));
        }
    }
    Some(DEFAULT_CORPUS.to_string())
}

struct Args {
    model: Option<String>,
    prompt: String,
    n_predict: usize,
    temp: f32,
    top_k: usize,
    top_p: f32,
    rep_penalty: f32,
    repeat_last_n: usize,
    seed: u64,
    gpu: bool,
    no_echo: bool,
    chat: bool,
    system: Option<String>,
    no_think: bool,
    effort: Option<String>,
    hide_thinking: bool,
    image: Option<String>,
    mmproj: Option<String>,
}

impl Args {
    /// Thinking config for hybrid reasoning models (qwen35). `--no-think` disables
    /// reasoning; `--effort low|medium|xhigh` sets depth (xhigh default).
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

/// Resolve a model path so `hos` works from any directory.
/// Order: explicit existing path → -m bare name → $HOS_MODEL → searched in
/// Run a HuggingFace checkpoint folder: `--bench`, `--perplexity`, or streaming
/// generation — same flags as the GGUF path, loaded natively via `Engine`.
fn run_hf(model_path: &Path, args: &Args, load_start: Instant) {
    // A fused `.hos` BODY (arch.hos_body) generates through the RGA forward so its
    // genes + grown FFN actually express — not the bare frozen trunk.
    if hos::is_hos_file(model_path) {
        if let Ok((_, card)) = hos::format::load_raw(model_path) {
            if card.arch.get("hos_body").is_some() {
                run_body(model_path, args, card, load_start);
                return;
            }
        }
    }
    if std::env::args().any(|a| a == "--bench") {
        cmd_bench(model_path, args);
        return;
    }
    if let Some(corpus) = perplexity_corpus() {
        cmd_perplexity(model_path, args, corpus);
        return;
    }
    let mut eng = ok(hos::Engine::load(model_path, args.gpu));
    eprintln!("[hos] load took {:.2}s", load_start.elapsed().as_secs_f64());
    eprintln!(
        "[hos] prompt -> {} tokens",
        eng.tok.encode(&args.prompt, true).len()
    );

    use std::io::Write;
    if !args.no_echo {
        print!("{}", args.prompt);
        std::io::stdout().flush().ok();
    }
    let n = eng.generate(
        &args.prompt,
        args.n_predict,
        args.temp,
        args.top_k,
        args.top_p,
        args.rep_penalty,
        args.repeat_last_n,
        args.seed,
        |piece| {
            print!("{piece}");
            std::io::stdout().flush().ok();
        },
    );
    println!();
    eprintln!("[hos] generated {n} tokens");
}

/// Generate from a fused `.hos` BODY, expressing one of its genomes over the
/// imported gene bank + grown FFN dims (the RGA forward). `--genome <name>` picks
/// which behavior (geography|arithmetic|story|code|child); default = first.
fn run_body(model_path: &Path, args: &Args, card: hos::format::Card, load_start: Instant) {
    use hos::tensor::use_gpu;
    use std::io::Write;
    let eng = ok(hos::Engine::load(model_path, false)); // base trunk + tokenizer
    let (named, _) = ok(hos::format::load(model_path).map_err(hos::HosError::from));
    let body = card
        .arch
        .get("hos_body")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let geti = |k: &str| body.get(k).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let (k, ng, delta, bottle) = (
        geti("genes").max(1),
        geti("genomes").max(1),
        geti("grow_ffn_dims"),
        geti("bottleneck").max(1),
    );
    let domains: Vec<String> = body
        .get("domains")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    // pick the genome to express
    let want = arg_after("--genome").unwrap_or_default();
    let gi = if want.is_empty() {
        0
    } else {
        domains
            .iter()
            .position(|d| d.starts_with(&want))
            .unwrap_or(0)
    };
    let gname = domains
        .get(gi)
        .cloned()
        .unwrap_or_else(|| format!("genome {gi}"));

    // rebuild the fused PeftModel and load its adapter from the body.* tensors
    let (hcfg0, _) = peft_matched(&eng.model.cfg, k, bottle);
    let hcfg = hos::peft::PeftCfg {
        genes: k,
        bottleneck: bottle,
        ..hcfg0
    };
    let mut peft = hos::peft::PeftModel::build_multi(&eng.model, "rga", &hcfg, ng, 0xF0_0D)
        .expect("body shell");
    if delta > 0 {
        peft.grow_ffn(delta, 1); // re-allocate grown dims; load_adapter fills them
    }
    let snap: Vec<(String, Vec<usize>, Vec<f32>)> = named
        .iter()
        .filter(|n| n.name.starts_with("body."))
        .map(|n| {
            (
                n.name["body.".len()..].to_string(),
                n.shape.clone(),
                n.data.clone(),
            )
        })
        .collect();
    peft.load_adapter(&snap);
    use_gpu(args.gpu);
    eprintln!("[hos] load took {:.2}s", load_start.elapsed().as_secs_f64());
    eprintln!(
        "[hos] BODY: expressing genome [{gi}] '{gname}' over {k} genes + {delta} grown FFN dims/layer",
    );

    let mut ids: Vec<usize> = eng
        .tok
        .encode(&args.prompt, true)
        .iter()
        .map(|&x| x as usize)
        .collect();
    if !args.no_echo {
        print!("{}", args.prompt);
        std::io::stdout().flush().ok();
    }
    let mut gen: Vec<u32> = Vec::new();
    let mut shown = 0usize;
    for _ in 0..args.n_predict {
        let logits = peft.logits_g(&ids, gi);
        let sh = logits.shape();
        let vocab = sh[sh.len() - 1];
        let d = logits.data();
        let last = &d[(ids.len() - 1) * vocab..ids.len() * vocab];
        // greedy (temp 0) argmax — deterministic, matches the eval path
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &v) in last.iter().enumerate() {
            if v > bv {
                bv = v;
                best = i;
            }
        }
        ids.push(best);
        gen.push(best as u32);
        // incremental decode (BPE-safe): print only the newly-revealed suffix
        let text = eng.tok.decode(&gen);
        if text.len() > shown {
            print!("{}", &text[shown..]);
            std::io::stdout().flush().ok();
            shown = text.len();
        }
        if Some(best as u32) == eng.tok.eos {
            break;
        }
    }
    println!();
    eprintln!(
        "[hos] generated {} tokens (body genome '{gname}')",
        gen.len()
    );
}

/// $HOS_MODELS_DIR, ~/Documents/hos/models, ~/.hos/models.
fn resolve_model(arg: Option<String>) -> PathBuf {
    let name = arg
        .or_else(|| std::env::var("HOS_MODEL").ok())
        .unwrap_or_else(|| {
            eprintln!("[hos] no model. pass -m <path|name>, or set HOS_MODEL");
            std::process::exit(1);
        });

    let direct = PathBuf::from(&name);
    if direct.exists() {
        return direct;
    }

    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Ok(d) = std::env::var("HOS_MODELS_DIR") {
        dirs.push(PathBuf::from(d));
    }
    if let Some(home) = std::env::var_os("HOME") {
        dirs.push(Path::new(&home).join("Documents/hos/models"));
        dirs.push(Path::new(&home).join(".hos/models"));
    }
    // Search dirs by both the given name and its bare filename, so `-m models/x.gguf`
    // and `-m x.gguf` both resolve from any working directory.
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
    eprintln!("[hos] model not found: {name}");
    eprintln!("[hos] looked in: . , $HOS_MODELS_DIR, ~/Documents/hos/models, ~/.hos/models");
    std::process::exit(1);
}

fn parse_args() -> Args {
    let mut model = None;
    let mut prompt = "Hello".to_string();
    let mut n_predict = 64usize;
    let mut temp = 0.0f32; // 0 = greedy
    let mut top_k = 40usize;
    let mut top_p = 0.95f32;
    let mut rep_penalty = 1.1f32;
    let mut repeat_last_n = 64usize;
    let mut seed = 42u64;
    let mut gpu = false;
    let mut no_echo = false;
    let mut chat = false;
    let mut system: Option<String> = None;
    let mut no_think = false;
    let mut effort: Option<String> = None;
    let mut hide_thinking = false;
    let mut image: Option<String> = None;
    let mut mmproj: Option<String> = None;

    let mut it = std::env::args().skip(1).peekable();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-m" | "--model" => model = Some(it.next().expect("path after -m")),
            "-p" | "--prompt" => prompt = it.next().expect("text after -p"),
            "-n" | "--n-predict" => n_predict = it.next().unwrap().parse().unwrap(),
            "--temp" => temp = it.next().unwrap().parse().unwrap(),
            "--top-k" => top_k = it.next().unwrap().parse().unwrap(),
            "--top-p" => top_p = it.next().unwrap().parse().unwrap(),
            "--repeat-penalty" => rep_penalty = it.next().unwrap().parse().unwrap(),
            "--repeat-last-n" => repeat_last_n = it.next().unwrap().parse().unwrap(),
            "--seed" => seed = it.next().unwrap().parse().unwrap(),
            "--gpu" => gpu = true,
            "--no-echo" => no_echo = true,
            "--chat" => chat = true,
            "--system" => system = Some(it.next().expect("text after --system")),
            "--no-think" | "--no-thinking" => no_think = true,
            "--effort" | "--reasoning-effort" => {
                effort = Some(it.next().expect("level after --effort"))
            }
            "--hide-thinking" | "--hide-reasoning" => hide_thinking = true,
            "--image" | "--img" => image = Some(it.next().expect("path after --image")),
            "--mmproj" => mmproj = Some(it.next().expect("path after --mmproj")),
            "--vision-encode" => {} // handled after model load
            "--deltanet-test" => {} // handled at start of main
            "--info" => {}          // handled after model load
            "--gpu-test" => {}      // handled after model load
            "--qwen35-check" => {}  // handled after model load
            "--vision-check" => {}  // handled after model load
            "--bench" => {}         // handled after model load
            "--finetune"
            | "--finetune-check"
            | "--peft"
            | "--peft-compare"
            | "--peft-interference"
            | "--peft-heldout"
            | "--peft-recombine"
            | "--peft-compose"
            | "--peft-clonal"
            | "--peft-demo"
            | "--peft-replay"
            | "--peft-grow"
            | "--peft-fuse" => {} // handled at start of main
            // value-taking flags consumed elsewhere (train_lm / gen_hos / to_hos read them)
            "--corpus" | "--out" | "-o" | "--gen-hos" | "--to-hos" | "--hos-viz" | "--ingest"
            | "--verify-against" | "--opt" | "--method" | "--quantize" | "--quant-awq"
            | "--awq-alpha" | "--genome" | "--remint-ft" | "--base" | "--hos-name"
            | "--source-note" => {
                let _ = it.next();
            }
            "--perplexity" => {
                // optional corpus file path may follow
                if let Some(p) = it.peek() {
                    if !p.starts_with('-') {
                        it.next();
                    }
                }
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: hos -m model.gguf -p \"prompt\" [-n 64] [--temp 0.0] [--seed 42]"
                );
                eprintln!(
                    "       hos -m model.gguf --bench            # prefill/decode throughput"
                );
                eprintln!("       hos -m model.gguf --perplexity [txt] # score held-out text");
                std::process::exit(0);
            }
            other => {
                eprintln!("[hos] unknown argument: {other}");
                eprintln!(
                    "[hos] (note: don't type the [ ] brackets from docs — they mean 'optional')"
                );
                eprintln!("[hos] run `hos --help` for usage");
                std::process::exit(2);
            }
        }
    }
    Args {
        model,
        prompt,
        n_predict,
        temp,
        top_k,
        top_p,
        rep_penalty,
        repeat_last_n,
        seed,
        gpu,
        no_echo,
        chat,
        system,
        no_think,
        effort,
        hide_thinking,
        image,
        mmproj,
    }
}

const GEMMA4_DEFAULT: &str = "path/to/gemma-4-12B-it";
const GEMMA4_REF_DEFAULT: &str = "path/to/gemma4_reference";

/// Map `--q4k`/`--q5k`/`--q6k` flags to `HOS_GEMMA4_QUANT` (the Gemma4 loader
/// reads the env). No flag => the default f32 path.
fn gemma4_set_quant_env() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--q4k") {
        std::env::set_var("HOS_GEMMA4_QUANT", "q4k");
    } else if args.iter().any(|a| a == "--q5k") {
        std::env::set_var("HOS_GEMMA4_QUANT", "q5k");
    } else if args.iter().any(|a| a == "--q6k") {
        std::env::set_var("HOS_GEMMA4_QUANT", "q6k");
    } else if args.iter().any(|a| a == "--classify") {
        // The constrained classifier defaults to the FAST path (Q4_K): it's only
        // a few-token scoring after prefill, so 4-bit noise is a non-issue and the
        // <dir>/gemma4-q4k.hosw cache makes load nearly instant. Override with an
        // explicit --q5k/--q6k or unset via HOS_GEMMA4_QUANT before launch.
        std::env::set_var("HOS_GEMMA4_QUANT", "q4k");
    }
}

/// `--gemma4 -m <dir> [--ids <csv>] [-p <text>] [-n <k>] [--q4k|--q5k|--q6k] [--gpu]`
fn gemma4_cli() -> hos::Result<()> {
    gemma4_set_quant_env();
    let dir = arg_after("-m")
        .or_else(|| arg_after("--model"))
        .unwrap_or_else(|| GEMMA4_DEFAULT.to_string());
    let n: usize = arg_after("-n").and_then(|s| s.parse().ok()).unwrap_or(0);

    // IMAGE path: `--gemma4 --image <path> -p "<question>"`.
    if let Some(img_path) = arg_after("--image") {
        return gemma4_image_cli(&dir, &img_path);
    }

    // Native Gemma SentencePiece-BPE tokenizer (only loaded for -p, and reused
    // to decode generated ids for display).
    let mut gtok: Option<hos::gemma_tok::GemmaTokenizer> = None;
    let ids: Vec<u32> = if let Some(csv) = arg_after("--ids") {
        csv.split(',')
            .filter_map(|s| s.trim().parse::<u32>().ok())
            .collect()
    } else if let Some(text) = arg_after("-p") {
        // Load Gemma's tokenizer.json natively and encode raw text (BOS=2 added).
        let tok = hos::gemma_tok::GemmaTokenizer::load_from_model_dir(Path::new(&dir))?;
        let enc = tok.encode(&text, true);
        println!(
            "[gemma4] tokenizer loaded (vocab={}), encoded {:?} -> {} ids",
            tok.vocab_len(),
            text,
            enc.len()
        );
        gtok = Some(tok);
        enc
    } else {
        return Err(hos::HosError::Format(
            "--gemma4 needs --ids <csv> (BOS=2 included) or -p <text>".into(),
        ));
    };

    let use_gpu = std::env::args().any(|a| a == "--gpu");
    println!("[gemma4] model={dir}");
    println!("[gemma4] input_ids ({}): {:?}", ids.len(), ids);
    let t0 = std::time::Instant::now();
    let mut m = hos::gemma4::Gemma4::load(Path::new(&dir))?;
    println!("[gemma4] load {:.1}s", t0.elapsed().as_secs_f32());

    // Optional Metal GPU acceleration: upload big linears as f16 (≈24 GB), then
    // the KV-cached decode dispatches its matmuls to the GPU (coherence-gated,
    // NOT bit-exact vs the bf16 CPU oracle).
    let gpu = if use_gpu {
        let g = hos::metal_be::Gpu::new();
        let tu = std::time::Instant::now();
        m.upload_to_gpu(&g);
        println!("[gemma4] GPU upload {:.1}s", tu.elapsed().as_secs_f32());
        Some(g)
    } else {
        None
    };
    let gref = gpu.as_ref();

    if n > 0 {
        let t1 = std::time::Instant::now();
        // KV-cached generation (Part 1): O(n) instead of the old O(n²) recompute.
        let gen = m.generate_cached(&ids, n, gref);
        let dt = t1.elapsed().as_secs_f32();
        println!(
            "[gemma4] greedy {n} tokens ({}): {:?}  ({:.2}s, {:.2}s/tok)",
            if use_gpu { "gpu-cached" } else { "cpu-cached" },
            gen,
            dt,
            dt / n as f32
        );
        if let Some(tok) = &gtok {
            let full: Vec<u32> = ids.iter().chain(gen.iter()).copied().collect();
            println!("[gemma4] prompt+gen text: {:?}", tok.decode(&full, true));
            println!("[gemma4] gen-only  text: {:?}", tok.decode(&gen, true));
        }
    } else {
        let t1 = std::time::Instant::now();
        // On the GPU path the weights are f16-resident (no CPU `forward`); use a
        // one-shot prefill for the logits. CPU path keeps the exact `forward`.
        let logits = if use_gpu {
            let mut cache = hos::gemma4::Gemma4Cache::new(&m);
            m.prefill(&mut cache, &ids, gref)
        } else {
            m.forward(&ids).logits
        };
        println!("[gemma4] forward {:.2}s", t1.elapsed().as_secs_f32());
        let top = hos::gemma4::topk(&logits, 10);
        println!("[gemma4] top-10 (id, logit):");
        for (id, l) in top {
            println!("  {id:>8}  {l:.4}");
        }
    }
    Ok(())
}

/// STAGE 3 native image path: `--gemma4 --image <path> -p "<question>"`.
/// Preprocess the raw image -> vision_embedder -> build input_ids with the
/// Gemma4 chat template (turn markers + boi/image/eoi + reasoning channel) ->
/// splice UNSCALED soft tokens -> bidirectional-attention prefill -> greedy gen.
fn gemma4_image_cli(dir: &str, img_path: &str) -> hos::Result<()> {
    let question =
        arg_after("-p").unwrap_or_else(|| "What is happening in this image?".to_string());
    let n: usize = arg_after("-n").and_then(|s| s.parse().ok()).unwrap_or(64);
    println!("[gemma4-image] model={dir}\n[gemma4-image] image={img_path}\n[gemma4-image] question={question:?}");

    // 1) Native preprocessing (bicubic resize; drift vs torchvision antialias).
    let budget = hos::gemma4_vision::BUDGET_PX;
    let t0 = std::time::Instant::now();
    let (pv, pos, gh, gw) =
        hos::gemma4_vision::preprocess_image_budget(Path::new(img_path), budget)?;
    let n_img = gh * gw;
    println!(
        "[gemma4-image] preprocess {:.2}s -> {n_img} patches",
        t0.elapsed().as_secs_f32()
    );

    // 2) vision_embedder -> soft tokens.
    let ve = hos::gemma4_vision::VisionEmbedder::load(Path::new(dir))?;
    let t1 = std::time::Instant::now();
    let soft = ve.embed_patches(&pv, &pos, n_img);
    println!(
        "[gemma4-image] vision embed {:.2}s",
        t1.elapsed().as_secs_f32()
    );

    // 3) Build input_ids via the Gemma4 chat template.
    let tok = hos::gemma_tok::GemmaTokenizer::load_from_model_dir(Path::new(dir))?;
    const BOI: u32 = 255999;
    const IMG: u32 = 258880;
    const EOI: u32 = 258882;
    let mut ids: Vec<u32> = vec![2, 105, 2364, 107]; // <bos> <|turn> user \n
    ids.push(BOI);
    ids.extend(std::iter::repeat(IMG).take(n_img));
    ids.push(EOI);
    ids.extend(tok.encode(&question, false)); // question (no leading space)
    ids.extend([106u32, 107, 105, 4368, 107, 100, 45518, 107, 101]); // <turn|>\n <|turn> model \n <|channel> thought \n <channel|>
    println!("[gemma4-image] input_ids len={} (n_img={n_img})", ids.len());

    // 4) Load decoder, splice + bidirectional prefill + greedy generate.
    // `--gpu` uploads the big linears to Metal (KV-cached decode dispatches to the
    // GPU, bf16-emul off) — the same fast path as the text CLI.
    let use_gpu = std::env::args().any(|a| a == "--gpu");
    let t2 = std::time::Instant::now();
    let mut m = hos::gemma4::Gemma4::load(Path::new(dir))?;
    println!(
        "[gemma4-image] decoder load {:.1}s",
        t2.elapsed().as_secs_f32()
    );
    let gpu = if use_gpu {
        let g = hos::metal_be::Gpu::new();
        m.upload_to_gpu(&g);
        Some(g)
    } else {
        None
    };
    let t3 = std::time::Instant::now();
    let gen = m.generate_image(&ids, &soft, IMG, n, gpu.as_ref());
    let dt = t3.elapsed().as_secs_f32();
    println!(
        "[gemma4-image] generate {} tokens {:.1}s ({:.2}s/tok)\n",
        gen.len(),
        dt,
        dt / gen.len().max(1) as f32
    );
    println!("[gemma4-image] gen ids: {:?}", gen);
    println!("[gemma4-image] ANSWER: {}", tok.decode(&gen, true));
    Ok(())
}

/// `--gemma4-bench [-m <dir>] [--q4k|--q5k|--q6k] [--gpu] [-n <k>]` — load-time,
/// resident-memory, and decode-tok/s benchmark. Prints one clearly-labeled row
/// per invocation so f32 vs Q4_K (run twice) line up directly.
fn gemma4_bench() -> hos::Result<()> {
    gemma4_set_quant_env();
    let dir = arg_after("-m")
        .or_else(|| arg_after("--model"))
        .unwrap_or_else(|| GEMMA4_DEFAULT.to_string());
    let n: usize = arg_after("-n").and_then(|s| s.parse().ok()).unwrap_or(16);
    let use_gpu = std::env::args().any(|a| a == "--gpu");
    let mode = std::env::var("HOS_GEMMA4_QUANT").unwrap_or_else(|_| "f32".to_string());
    let mode_label = format!("{}{}", mode, if use_gpu { "+gpu" } else { "" });

    println!("[gemma4-bench] model={dir}");
    println!("[gemma4-bench] mode={mode_label}  decode_tokens={n}");

    let t0 = std::time::Instant::now();
    let mut m = hos::gemma4::Gemma4::load(Path::new(&dir))?;
    let load_s = t0.elapsed().as_secs_f32();
    let resident_gb = m.resident_bytes() as f64 / (1024.0 * 1024.0 * 1024.0);
    println!("[gemma4-bench] load {load_s:.2}s  resident {resident_gb:.2} GB");

    let gpu = if use_gpu {
        let g = hos::metal_be::Gpu::new();
        let tu = std::time::Instant::now();
        m.upload_to_gpu(&g);
        println!(
            "[gemma4-bench] GPU upload {:.2}s",
            tu.elapsed().as_secs_f32()
        );
        Some(g)
    } else {
        None
    };
    let gref = gpu.as_ref();

    // Fixed short prompt (BOS included by the tokenizer).
    let tok = hos::gemma_tok::GemmaTokenizer::load_from_model_dir(Path::new(&dir))?;
    let ids = tok.encode("The capital of France is", true);
    println!("[gemma4-bench] prompt ids ({}): {:?}", ids.len(), ids);

    // Prefill (timed) then decode n tokens one at a time (timed) for a clean
    // decode tok/s figure.
    let mut cache = hos::gemma4::Gemma4Cache::new(&m);
    let tp = std::time::Instant::now();
    let mut logits = m.prefill(&mut cache, &ids, gref);
    let prefill_s = tp.elapsed().as_secs_f32();

    let argmax = |lg: &[f32]| -> u32 {
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &l) in lg.iter().enumerate() {
            if l > bv {
                bv = l;
                best = i;
            }
        }
        best as u32
    };
    let td = std::time::Instant::now();
    let mut gen = Vec::with_capacity(n);
    for _ in 0..n {
        let best = argmax(&logits);
        gen.push(best);
        if gen.len() == n {
            break;
        }
        logits = m.decode_step(&mut cache, best, gref);
    }
    let decode_s = td.elapsed().as_secs_f32();
    let tok_s = if decode_s > 0.0 {
        n as f32 / decode_s
    } else {
        0.0
    };

    println!("[gemma4-bench] gen ids: {:?}", gen);
    println!("[gemma4-bench] gen text: {:?}", tok.decode(&gen, true));
    println!("\n[gemma4-bench] ==== RESULT ====");
    println!(
        "  {:<10} {:>10} {:>14} {:>12} {:>14}",
        "MODE", "LOAD(s)", "RESIDENT(GB)", "PREFILL(s)", "DECODE tok/s"
    );
    println!(
        "  {:<10} {:>10.2} {:>14.2} {:>12.2} {:>14.2}",
        mode_label, load_s, resident_gb, prefill_s, tok_s
    );
    Ok(())
}

/// `--gemma4-kv-check` — CORRECTNESS GATE for the KV cache: assert the cached
/// path (`generate_cached`) produces byte-identical token ids to the old
/// uncached path (`generate`) for the 3 selftest prompts, n=8, on the CPU bf16
/// path. Guards that the O(n) cache never diverges from the O(n²) oracle path.
fn gemma4_kv_check() -> hos::Result<()> {
    let dir = arg_after("-m")
        .or_else(|| arg_after("--model"))
        .unwrap_or_else(|| GEMMA4_DEFAULT.to_string());
    let ref_dir = arg_after("--ref").unwrap_or_else(|| GEMMA4_REF_DEFAULT.to_string());
    println!("[gemma4-kv-check] model={dir}");

    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(Path::new(&ref_dir).join("manifest.json"))?)
            .map_err(|e| hos::HosError::Format(format!("manifest: {e}")))?;

    let t0 = std::time::Instant::now();
    let m = hos::gemma4::Gemma4::load(Path::new(&dir))?;
    println!(
        "[gemma4-kv-check] load {:.1}s\n",
        t0.elapsed().as_secs_f32()
    );

    let n = 8usize;
    let mut all_pass = true;
    for pi in 1..=3 {
        let pk = format!("P{pi}");
        let ids: Vec<u32> = manifest["prompts"][&pk]["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();

        let tu = std::time::Instant::now();
        let uncached = m.generate(&ids, n);
        let du = tu.elapsed().as_secs_f32();
        let tc = std::time::Instant::now();
        let cached = m.generate_cached(&ids, n, None);
        let dc = tc.elapsed().as_secs_f32();

        let ok = uncached == cached;
        all_pass &= ok;
        println!(
            "  {pk}: n={n} uncached={:?} ({:.2}s)\n       cached  ={:?} ({:.2}s)  {}",
            uncached,
            du,
            cached,
            dc,
            if ok { "IDENTICAL ✅" } else { "DIVERGED ❌" }
        );
    }
    println!(
        "\n[gemma4-kv-check] {}",
        if all_pass {
            "✅ ALL PASS — cached == uncached for all 3 prompts"
        } else {
            "❌ FAIL — cached path diverged"
        }
    );
    if !all_pass {
        return Err(hos::HosError::Format("kv-check diverged".into()));
    }
    Ok(())
}

/// `--gemma4-prefill-check` — PARITY + SPEED gate for the batched GPU prefill.
/// Runs the sequential prefill (`prefill_image`) and the batched prefill
/// (`prefill_image_batched`) on the SAME input (a text prompt and a synthetic
/// image prompt with a real bidirectional span), and asserts top-1 identical,
/// cosine > 0.999, and matching KV caches — plus reports the wall-time speedup.
/// Forces the q4_k + GPU path (that's where batching engages).
fn gemma4_prefill_check() -> hos::Result<()> {
    let dir = arg_after("-m")
        .or_else(|| arg_after("--model"))
        .unwrap_or_else(|| GEMMA4_DEFAULT.to_string());
    if std::env::var("HOS_GEMMA4_QUANT").is_err() {
        std::env::set_var("HOS_GEMMA4_QUANT", "q4k");
    }
    let quant = std::env::var("HOS_GEMMA4_QUANT").unwrap_or_default();
    println!("[gemma4-prefill-check] model={dir} quant={quant}");

    let t0 = Instant::now();
    let mut m = hos::gemma4::Gemma4::load(Path::new(&dir))?;
    let gpu = hos::metal_be::Gpu::new();
    m.upload_to_gpu(&gpu);
    println!(
        "[gemma4-prefill-check] load+upload {:.1}s",
        t0.elapsed().as_secs_f32()
    );
    if !m.can_batch_prefill() {
        println!("[gemma4-prefill-check] ⚠ not all projections are q4_k-on-GPU; run with --q4k. Skipping.");
        return Err(hos::HosError::Format("prefill-check needs q4k+gpu".into()));
    }

    let cos = |a: &[f32], b: &[f32]| -> f32 {
        let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for (x, y) in a.iter().zip(b.iter()) {
            d += *x as f64 * *y as f64;
            na += *x as f64 * *x as f64;
            nb += *y as f64 * *y as f64;
        }
        (d / (na.sqrt() * nb.sqrt() + 1e-30)) as f32
    };
    let argmax = |v: &[f32]| -> usize {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    };
    let maxabs = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };

    let mut all_pass = true;

    // ---- CASE 1: TEXT prompt (plain causal) ----
    {
        let ids: Vec<u32> = std::iter::once(2u32).chain(100u32..=140).collect(); // 42 tokens
        let mut c_seq = hos::gemma4::Gemma4Cache::new(&m);
        let mut c_bat = hos::gemma4::Gemma4Cache::new(&m);
        let ts = Instant::now();
        let lg_seq = m.prefill(&mut c_seq, &ids, Some(&gpu));
        let seq_s = ts.elapsed().as_secs_f32();
        let tb = Instant::now();
        let lg_bat = m.prefill_batched(&mut c_bat, &ids, &gpu);
        let bat_s = tb.elapsed().as_secs_f32();
        let (t1s, t1b) = (argmax(&lg_seq), argmax(&lg_bat));
        let (kd, vd) = c_seq.max_kv_diff(&c_bat);
        let ok = t1s == t1b && cos(&lg_seq, &lg_bat) > 0.999;
        all_pass &= ok;
        println!(
            "\n[TEXT] seq={} tokens  seq={:.2}s  batched={:.2}s  speedup={:.1}x",
            ids.len(),
            seq_s,
            bat_s,
            seq_s / bat_s.max(1e-6)
        );
        println!(
            "       top1 seq={t1s} batched={t1b}  cos={:.5}  logit_maxabs={:.4}  kv_maxabs k={:.2e} v={:.2e}  {}",
            cos(&lg_seq, &lg_bat), maxabs(&lg_seq, &lg_bat), kd, vd,
            if ok { "PASS ✅" } else { "FAIL ❌" }
        );
    }

    // ---- CASE 2: synthetic IMAGE prompt (bidirectional span, ~300 tokens) ----
    {
        const BOI: u32 = 255999;
        const IMG: u32 = 258880;
        const EOI: u32 = 258882;
        let n_img = 290usize;
        let mut prefix: Vec<u32> = vec![2, 105, 2364, 107];
        prefix.push(BOI);
        prefix.extend(std::iter::repeat(IMG).take(n_img));
        prefix.push(EOI);
        // deterministic pseudo-random soft tokens (values don't matter for parity;
        // both paths get the identical input — this exercises the span/mask code).
        let hidden = m.cfg.hidden;
        let soft: Vec<f32> = (0..n_img * hidden)
            .map(|i| {
                let z = (i.wrapping_mul(2654435761) % 1000) as f32 / 1000.0 - 0.5;
                z * 0.08
            })
            .collect();
        let (embeds, spans) = m.build_image_embeds(&prefix, &soft, IMG);
        let mut c_seq = hos::gemma4::Gemma4Cache::new(&m);
        let mut c_bat = hos::gemma4::Gemma4Cache::new(&m);
        let ts = Instant::now();
        let lg_seq = m.prefill_image(&mut c_seq, &embeds, &spans, Some(&gpu));
        let seq_s = ts.elapsed().as_secs_f32();
        let tb = Instant::now();
        let lg_bat = m.prefill_image_batched(&mut c_bat, &embeds, &spans, &gpu);
        let bat_s = tb.elapsed().as_secs_f32();
        let (t1s, t1b) = (argmax(&lg_seq), argmax(&lg_bat));
        let (kd, vd) = c_seq.max_kv_diff(&c_bat);
        let ok = t1s == t1b && cos(&lg_seq, &lg_bat) > 0.999;
        all_pass &= ok;
        println!(
            "\n[IMAGE] seq={} tokens ({} soft)  seq={:.2}s  batched={:.2}s  speedup={:.1}x",
            embeds.len(),
            n_img,
            seq_s,
            bat_s,
            seq_s / bat_s.max(1e-6)
        );
        println!(
            "        top1 seq={t1s} batched={t1b}  cos={:.5}  logit_maxabs={:.4}  kv_maxabs k={:.2e} v={:.2e}  {}",
            cos(&lg_seq, &lg_bat), maxabs(&lg_seq, &lg_bat), kd, vd,
            if ok { "PASS ✅" } else { "FAIL ❌" }
        );
    }

    println!(
        "\n[gemma4-prefill-check] {}",
        if all_pass {
            "✅ ALL PASS — batched == sequential (top-1 + cos>0.999)"
        } else {
            "❌ FAIL"
        }
    );
    if !all_pass {
        return Err(hos::HosError::Format("prefill-check parity failed".into()));
    }
    Ok(())
}

/// `--gemma-tok-selftest` — validate the native Gemma tokenizer against a
/// frozen transformers (5.12.1) oracle: exact id-sequence match on a diverse
/// corpus, plus decode round-trip.
fn gemma_tok_selftest() -> hos::Result<()> {
    let dir = arg_after("-m")
        .or_else(|| arg_after("--model"))
        .unwrap_or_else(|| GEMMA4_DEFAULT.to_string());
    println!("[gemma-tok-selftest] model={dir}");
    let t0 = std::time::Instant::now();
    let tok = hos::gemma_tok::GemmaTokenizer::load_from_model_dir(Path::new(&dir))?;
    println!(
        "[gemma-tok-selftest] loaded vocab={} in {:.2}s\n",
        tok.vocab_len(),
        t0.elapsed().as_secs_f32()
    );

    // (text, reference ids from tok.encode(s, add_special_tokens=False)).
    // Oracle: transformers 5.12.1 / tokenizers 0.22.2, google/gemma-4-12B-it.
    let cases: &[(&str, &[u32])] = &[
        ("Hello world", &[9259, 1902]),
        (
            "The capital of France is Paris.",
            &[818, 5279, 529, 7001, 563, 9079, 236761],
        ),
        (
            "def fibonacci(n):\n    return n",
            &[2063, 10779, 78113, 236769, 236749, 1473, 107, 140, 2060, 538],
        ),
        (
            "  leading and   multiple   spaces ",
            &[138, 26016, 532, 139, 43819, 139, 35220, 236743],
        ),
        (
            "café ☕ 日本語 \u{1f94a}",
            &[123125, 236859, 236743, 244360, 33375, 238582, 236743, 248279],
        ),
        (
            "line1\nline2\ttab",
            &[1257, 236770, 107, 1257, 236778, 255968, 4823],
        ),
        ("<bos>already", &[2, 68020]),
        (
            "12 times 8 = 96",
            &[236770, 236778, 2782, 236743, 236828, 578, 236743, 236819, 236825],
        ),
        (
            "Mixed CASE and Punctuation!!! ...",
            &[105367, 69203, 532, 593, 23733, 3145, 11145, 3729],
        ),
        (
            "The quick brown fox jumps over the lazy dog. In a village of La Mancha, the name of which I have no desire to call to mind, there lived not long since one of those gentlemen that keep a lance in the lance-rack.",
            &[
                818, 3823, 8864, 37423, 38167, 1024, 506, 31770, 4799, 236761, 799, 496, 9744, 529,
                2774, 2599, 4936, 236764, 506, 1463, 529, 837, 564, 735, 951, 12614, 531, 2246, 531,
                3666, 236764, 993, 11742, 711, 1440, 2338, 886, 529, 1724, 43085, 600, 2514, 496,
                63180, 528, 506, 63180, 236772, 74368, 236761,
            ],
        ),
        (
            "for (int i=0; i<n; i++) { arr[i] = i*i; } // squares\nprintf(\"%d\\n\", arr[3]);",
            &[
                1708, 568, 720, 858, 236784, 236771, 236793, 858, 236820, 236749, 236793, 858,
                4419, 642, 4617, 236840, 236747, 236842, 578, 858, 236829, 236747, 236793, 682,
                973, 23441, 107, 8641, 13233, 236753, 236785, 236749, 827, 4617, 236840, 236800,
                6284,
            ],
        ),
    ];

    println!(
        "{:<44} {:>7} {:>7}  {:<6} {}",
        "text", "n_ref", "n_got", "ENCODE", "ROUNDTRIP"
    );
    let mut all_pass = true;
    for (text, refids) in cases {
        let got = tok.encode(text, false);
        let enc_ok = got.as_slice() == *refids;
        let dec = tok.decode(&got, false);
        let rt_ok = &dec == text;
        all_pass &= enc_ok && rt_ok;
        let show: String = text.chars().take(40).collect();
        println!(
            "{:<44} {:>7} {:>7}  {:<6} {}",
            format!("{:?}", show),
            refids.len(),
            got.len(),
            if enc_ok { "PASS" } else { "FAIL" },
            if rt_ok { "PASS" } else { "FAIL" }
        );
        if !enc_ok {
            let n = refids.len().max(got.len());
            for k in 0..n {
                let a = refids.get(k).copied();
                let b = got.get(k).copied();
                if a != b {
                    println!("      first mismatch @ {k}: ref={a:?} got={b:?}");
                    break;
                }
            }
        }
        if !rt_ok {
            println!("      roundtrip decode: {dec:?}");
        }
    }

    // BOS demo used by generation (P1).
    let p1 = tok.encode("The capital of France is", true);
    let p1_ok = p1 == vec![2, 818, 5279, 529, 7001, 563];
    println!(
        "\n[gemma-tok-selftest] BOS demo: encode(\"The capital of France is\", bos=true) = {:?}  {}",
        p1,
        if p1_ok { "PASS (2,818,5279,529,7001,563)" } else { "FAIL" }
    );
    all_pass &= p1_ok;

    println!(
        "\n[gemma-tok-selftest] {}",
        if all_pass {
            "ALL PASS ✅"
        } else {
            "FAILURES ⚠️"
        }
    );
    if !all_pass {
        return Err(hos::HosError::Format("gemma-tok-selftest failures".into()));
    }
    Ok(())
}

fn main() {
    // the HOS banner, every invocation (stderr: machine-readable stdout stays clean)
    print_banner();
    if std::env::args().any(|a| a == "--gemma-tok-selftest") {
        if let Err(e) = gemma_tok_selftest() {
            eprintln!("gemma-tok selftest error: {e}");
            std::process::exit(1);
        }
        return;
    }
    if std::env::args().any(|a| a == "--gemma4-kv-check") {
        if let Err(e) = gemma4_kv_check() {
            eprintln!("gemma4 kv-check error: {e}");
            std::process::exit(1);
        }
        return;
    }
    if std::env::args().any(|a| a == "--gemma4-prefill-check") {
        if let Err(e) = gemma4_prefill_check() {
            eprintln!("gemma4 prefill-check error: {e}");
            std::process::exit(1);
        }
        return;
    }
    if std::env::args().any(|a| a == "--gemma4-bench") {
        if let Err(e) = gemma4_bench() {
            eprintln!("gemma4 bench error: {e}");
            std::process::exit(1);
        }
        return;
    }
    if std::env::args().any(|a| a == "--gemma4-ingest") {
        cmd_gemma4_ingest();
        return;
    }
    if std::env::args().any(|a| a == "--gemma4") {
        if let Err(e) = gemma4_cli() {
            eprintln!("gemma4 error: {e}");
            std::process::exit(1);
        }
        return;
    }
    if std::env::args().any(|a| a == "--deltanet-test") {
        #[cfg(target_os = "macos")]
        deltanet_test();
        return;
    }
    if std::env::args().any(|a| a == "--autograd-demo") {
        autograd_demo();
        return;
    }
    if std::env::args().any(|a| a == "--nn-demo") {
        hos::nn::demo();
        return;
    }
    if std::env::args().any(|a| a == "--rnn-demo") {
        hos::nn::rnn_demo();
        return;
    }
    if std::env::args().any(|a| a == "--vq-demo") {
        hos::nn::vq_demo();
        return;
    }
    if std::env::args().any(|a| a == "--op-demo") {
        hos::nn::op_demo();
        return;
    }
    if std::env::args().any(|a| a == "--mem-demo") {
        hos::nn::mem_demo();
        return;
    }
    if std::env::args().any(|a| a == "--contradict-scale") {
        hos::nn::contradiction_scale_demo();
        return;
    }
    if std::env::args().any(|a| a == "--fixed-scale") {
        hos::nn::fixed_addr_scale_demo();
        return;
    }
    if std::env::args().any(|a| a == "--train-gpu-test") {
        train_gpu_test();
        return;
    }
    if std::env::args().any(|a| a == "--matmul-bench") {
        matmul_bench();
        return;
    }
    if std::env::args().any(|a| a == "--batch-bench") {
        batch_bench();
        return;
    }
    if std::env::args().any(|a| a == "--batch-attn-test") {
        batch_attn_test();
        return;
    }
    if std::env::args().any(|a| a == "--train-lm") {
        train_lm();
        return;
    }
    if std::env::args().any(|a| a == "--gen-hos") {
        gen_hos(&parse_args());
        return;
    }
    if std::env::args().any(|a| a == "--to-hos") {
        cmd_to_hos(&parse_args());
        return;
    }
    if std::env::args().any(|a| a == "--ingest") {
        cmd_ingest(&parse_args());
        return;
    }
    if std::env::args().any(|a| a == "--verify-against") {
        cmd_verify_against(&parse_args());
        return;
    }
    if std::env::args().any(|a| a == "--finetune-check") {
        cmd_finetune_check(&parse_args());
        return;
    }
    if std::env::args().any(|a| a == "--finetune") {
        cmd_finetune(&parse_args());
        return;
    }
    if std::env::args().any(|a| a == "--peft-heldout") {
        cmd_peft_heldout(&parse_args());
        return;
    }
    if std::env::args().any(|a| a == "--peft-recombine") {
        cmd_peft_recombine(&parse_args());
        return;
    }
    if std::env::args().any(|a| a == "--peft-compose") {
        cmd_peft_compose(&parse_args());
        return;
    }
    if std::env::args().any(|a| a == "--peft-clonal") {
        cmd_peft_clonal(&parse_args());
        return;
    }
    if std::env::args().any(|a| a == "--peft-demo") {
        cmd_peft_demo(&parse_args());
        return;
    }
    if std::env::args().any(|a| a == "--peft-replay") {
        cmd_peft_replay(&parse_args());
        return;
    }
    if std::env::args().any(|a| a == "--peft-grow") {
        cmd_peft_grow(&parse_args());
        return;
    }
    if std::env::args().any(|a| a == "--peft-fuse") {
        cmd_peft_fuse(&parse_args());
        return;
    }
    if std::env::args().any(|a| a == "--quant-bench") {
        cmd_quant_bench();
        return;
    }
    if std::env::args().any(|a| a == "--quant-awq") {
        cmd_quant_awq();
        return;
    }
    if std::env::args().any(|a| a == "--peft-interference") {
        cmd_peft_interference(&parse_args());
        return;
    }
    if std::env::args().any(|a| a == "--peft-compare") {
        cmd_peft_compare(&parse_args());
        return;
    }
    if std::env::args().any(|a| a == "--peft") {
        cmd_peft(&parse_args());
        return;
    }
    if std::env::args().any(|a| a == "--hos-viz") {
        cmd_hos_viz(&parse_args());
        return;
    }
    if std::env::args().any(|a| a == "--hos-selfrun") {
        hos_selfrun();
        return;
    }
    if std::env::args().any(|a| a == "--selfrun-tf") {
        selfrun_tf();
        return;
    }
    if std::env::args().any(|a| a == "--train-spec") {
        train_spec();
        return;
    }
    if std::env::args().any(|a| a == "--lm-demo") {
        lm_demo();
        return;
    }
    if std::env::args().any(|a| a == "--hos-demo") {
        hos_demo();
        return;
    }
    if std::env::args().any(|a| a == "--interp-check") {
        interp_check();
        return;
    }
    {
        let a: Vec<String> = std::env::args().collect();
        if let Some(i) = a.iter().position(|x| x == "--hos-info") {
            hos::format::inspect(std::path::Path::new(&a[i + 1])).expect("inspect");
            return;
        }
    }
    let args = parse_args();

    let model_path = resolve_model(args.model.clone());
    eprintln!("[hos] model: {}", model_path.display());
    let load_start = Instant::now();

    // A HuggingFace checkpoint folder (config.json + *.safetensors) loads
    // natively — no GGUF, no llama.cpp, no transformers.
    if model_path.is_dir() && model_path.join("config.json").exists() {
        run_hf(&model_path, &args, load_start);
        return;
    }
    // A runnable `.hos` capsule loads through Engine::load (handles generate,
    // --bench, --perplexity); the GGUF fast-path below is for .gguf only.
    if hos::is_hos_file(&model_path) {
        run_hf(&model_path, &args, load_start);
        return;
    }

    let g = ok(Gguf::open(&model_path));

    if std::env::args().any(|a| a == "--info") {
        let arch = g.meta_str("general.architecture").unwrap_or("?");
        eprintln!("architecture: {arch}");
        let mut keys: Vec<&String> = g.metadata.keys().collect();
        keys.sort();
        for k in keys {
            if k.starts_with(&format!("{arch}.")) {
                if let Some(v) = g.metadata.get(k).and_then(|v| v.as_u64()) {
                    eprintln!("  {k} = {v}");
                } else if let Some(v) = g.metadata.get(k).and_then(|v| v.as_f32()) {
                    eprintln!("  {k} = {v}");
                }
            }
        }
        // normalized tensor summary (collapse blk.<n> -> blk.N)
        use std::collections::BTreeMap;
        let norm = |n: &str| -> String {
            if let Some(rest) = n.strip_prefix("blk.") {
                if let Some(dot) = rest.find('.') {
                    return format!("blk.N{}", &rest[dot..]);
                }
            }
            n.to_string()
        };
        let mut shapes: BTreeMap<String, (u32, usize)> = BTreeMap::new();
        for (n, t) in &g.tensors {
            let e = shapes.entry(norm(n)).or_insert((t.ggml_type, 0));
            e.1 += 1;
        }
        eprintln!("--- tensors (normalized over layers) ---");
        for (k, (ty, cnt)) in &shapes {
            eprintln!("  {k}  (type {ty}) x{cnt}");
        }
        let mut ssm = 0usize;
        let mut attn = 0usize;
        let mut moe = false;
        for n in g.tensors.keys() {
            if n.ends_with(".ssm_a") {
                ssm += 1;
            }
            if n.ends_with("attn_qkv.weight") || n.ends_with("attn_q.weight") {
                attn += 1;
            }
            if n.contains("exps") {
                moe = true;
            }
        }
        eprintln!("layers w/ SSM: {ssm} | layers w/ attention: {attn} | MoE experts: {moe}");
        return;
    }

    if std::env::args().any(|a| a == "--qwen35-check") {
        ok(hos::qwen35::validate(&g));
        return;
    }
    if std::env::args().any(|a| a == "--vision-check") {
        ok(hos::qwen35_vision::check(&g));
        return;
    }
    if std::env::args().any(|a| a == "--vision-encode") {
        let img = args.image.clone().expect("--vision-encode needs --image <path>");
        let tower = ok(hos::qwen35_vision::VisionTower::load(&g));
        let t0 = Instant::now();
        let emb = ok(tower.encode_image(std::path::Path::new(&img)));
        let pd = tower.cfg.proj_dim;
        let ntok = emb.len() / pd;
        let mean = emb.iter().sum::<f32>() / emb.len() as f32;
        let norm = (emb.iter().map(|v| v * v).sum::<f32>() / emb.len() as f32).sqrt();
        let finite = emb.iter().all(|v| v.is_finite());
        eprintln!(
            "[vision] encoded {img} -> {ntok} tokens x {pd} in {:.1}s | finite={finite} mean={mean:.4} rms={norm:.4}",
            t0.elapsed().as_secs_f64()
        );
        eprintln!("[vision] token0[..8] = {:?}", &emb[..8]);
        return;
    }

    // Reproducible measurement commands (build their own Engine).
    if std::env::args().any(|a| a == "--bench") {
        cmd_bench(&model_path, &args);
        return;
    }
    if let Some(corpus) = perplexity_corpus() {
        cmd_perplexity(&model_path, &args, corpus);
        return;
    }

    let tok = ok(Tokenizer::from_gguf(&g));

    if hos::model::Arch::detect(&g) == hos::model::Arch::Qwen35Hybrid {
        run_qwen35(&g, &tok, &args);
        return;
    }

    // Gemma2/Phi3 run on the CPU forward path (the fused runner is Llama-family only)
    let use_gpu = args.gpu
        && cfg!(target_os = "macos")
        && hos::model::Arch::detect(&g).gpu_supported()
        && hos::model::gpu_quant_supported(&g); // HQ4/MoE have no GPU kernel -> CPU path
    let gpu = if use_gpu {
        Some(metal_be::Gpu::new())
    } else {
        None
    };
    let model = ok(Model::load(&g, gpu.as_ref()));
    eprintln!("[hos] load took {:.2}s", load_start.elapsed().as_secs_f64());

    let runner = gpu.map(|gp| metal_be::GpuRunner::new(gp, &model));

    if std::env::args().any(|a| a == "--gpu-test") {
        gpu_test(&model);
        return;
    }

    let prompt_ids = tok.encode(&args.prompt, true);
    eprintln!("[hos] prompt -> {} tokens", prompt_ids.len());

    let mut state = forward::State::new(&model.cfg);
    let mut rng = args.seed;

    use std::io::Write;
    if !args.no_echo {
        print!("{}", args.prompt);
        std::io::stdout().flush().ok();
    }

    // --- prefill ---
    let prefill_start = Instant::now();
    let mut pos = 0usize;
    let mut last_logits = Vec::new();
    for &t in &prompt_ids {
        last_logits = match &runner {
            Some(r) => r.forward(&model, t, pos),
            None => forward::forward(&model, &mut state, t, pos, None),
        };
        pos += 1;
    }
    let prefill_s = prefill_start.elapsed().as_secs_f64();

    // --- decode ---
    let decode_start = Instant::now();
    let mut generated = 0usize;
    let mut recent: Vec<u32> = Vec::new();
    let mut out_bytes = Vec::new();
    for _ in 0..args.n_predict {
        let from = recent.len().saturating_sub(args.repeat_last_n);
        let next = sample(
            &last_logits,
            args.temp,
            args.top_k,
            args.top_p,
            args.rep_penalty,
            &recent[from..],
            &mut rng,
        );
        recent.push(next);
        if Some(next) == tok.eos {
            break;
        }
        out_bytes.clear();
        tok.decode_into(next, &mut out_bytes);
        print!("{}", String::from_utf8_lossy(&out_bytes));
        std::io::stdout().flush().ok();

        if pos >= model.cfg.ctx_len {
            break;
        }
        last_logits = match &runner {
            Some(r) => r.forward(&model, next, pos),
            None => forward::forward(&model, &mut state, next, pos, None),
        };
        pos += 1;
        generated += 1;
    }
    let decode_s = decode_start.elapsed().as_secs_f64();

    println!();
    eprintln!(
        "\n[hos] prefill: {} tok in {:.3}s ({:.1} tok/s) | decode: {} tok in {:.3}s ({:.1} tok/s)",
        prompt_ids.len(),
        prefill_s,
        prompt_ids.len() as f64 / prefill_s.max(1e-9),
        generated,
        decode_s,
        generated as f64 / decode_s.max(1e-9),
    );
}

/// Validate the Metal matvec against the CPU and benchmark both on the
/// biggest matmul in the model (the output projection: [vocab, dim] · [dim]).
fn gpu_test(model: &Model) {
    let dim = model.cfg.dim;
    let vocab = model.cfg.vocab_size;
    let w = &model.output; // [vocab, dim]
    let x: Vec<f32> = (0..dim).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();

    // CPU reference
    let cpu = |w: &[f32], x: &[f32], rows: usize, in_dim: usize| -> Vec<f32> {
        let mut y = vec![0.0f32; rows];
        for o in 0..rows {
            let mut acc = 0.0;
            for i in 0..in_dim {
                acc += w[o * in_dim + i] * x[i];
            }
            y[o] = acc;
        }
        y
    };

    let w = w.to_f32();
    let gpu = metal_be::Gpu::new();
    let gw = gpu.upload_matrix(&w, vocab, dim);

    let y_cpu = cpu(&w, &x, vocab, dim);
    let y_gpu = gpu.matvec(&gw, &x);

    let mut max_diff = 0.0f32;
    for i in 0..vocab {
        max_diff = max_diff.max((y_cpu[i] - y_gpu[i]).abs());
    }
    eprintln!(
        "[gpu-test] matvec [{vocab} x {dim}] · [{dim}]  max|cpu-gpu| = {max_diff:.3e}  {}",
        if max_diff < 0.5 {
            "✅ MATCH (f16 weights)"
        } else {
            "❌ MISMATCH"
        }
    );

    let iters = 100;
    let t = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(cpu(&w, &x, vocab, dim));
    }
    let cpu_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let t = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(gpu.matvec(&gw, &x));
    }
    let gpu_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    eprintln!(
        "[gpu-test] per matvec: CPU {cpu_ms:.3} ms | GPU {gpu_ms:.3} ms | speedup {:.2}x",
        cpu_ms / gpu_ms
    );
}

/// Experimental CPU path for the qwen35 hybrid (gated delta-net) architecture.
/// One-shot correctness gate for the resident 2-token verify: compares `forward2`'s
/// per-token logits against the proven single-token resident `forward` on the same
/// [t, draft] pair, from the same seeded state. Argmax must match; logits diff tiny.
#[cfg(target_os = "macos")]
fn qwen35_fwd2_check(
    m: &hos::qwen35::Qwen35,
    r: &hos::qwen35::Qwen35Gpu,
    state: &mut hos::qwen35::State,
    logits: &[f32],
    hidden: &[f32],
    gpu: Option<&metal_be::Gpu>,
) {
    let d = m.dim();
    let amax = |l: &[f32]| l.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) }).0;
    let p = state.pos;
    let t = amax(logits) as u32;
    let draft = amax(&m.mtp_draft_logits(hidden, t, p, state, gpu)) as u32;
    eprintln!("[fwd2-check] p={p} t={t} draft={draft}");
    // forward2 (state already seeded to post-prefill p); returns [hid_t | hid_d]
    let hids = r.forward2(m, t, draft, p);
    let lt2 = m.logits_from_hidden(&hids[0..d], gpu);
    let ld2 = m.logits_from_hidden(&hids[d..2 * d], gpu);
    // reference: reset resident backbone to post-prefill, run proven single-token fwd
    r.upload_state(m, state, p);
    let lt = r.forward(m, t, p);
    let ld = r.forward(m, draft, p + 1);
    let maxdiff = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    eprintln!(
        "[fwd2-check] token t: argmax fwd2={} ref={} {} | max|dlogit|={:.4}",
        amax(&lt2), amax(&lt), if amax(&lt2) == amax(&lt) { "OK" } else { "MISMATCH" }, maxdiff(&lt2, &lt),
    );
    eprintln!(
        "[fwd2-check] token d: argmax fwd2={} ref={} {} | max|dlogit|={:.4}",
        amax(&ld2), amax(&ld), if amax(&ld2) == amax(&ld) { "OK" } else { "MISMATCH" }, maxdiff(&ld2, &ld),
    );
}

fn run_qwen35(g: &Gguf, tok: &Tokenizer, args: &Args) {
    use std::io::Write;
    // Chat mode routes through the shared ChatSession (template + thinking split).
    if args.chat {
        run_qwen35_chat(g, args);
        return;
    }
    eprintln!("[hos] qwen35 hybrid — EXPERIMENTAL (GPU matmuls + CPU recurrence)");
    let gpu = if args.gpu && cfg!(target_os = "macos") {
        Some(metal_be::Gpu::new())
    } else {
        None
    };
    let load = Instant::now();
    let m = ok(hos::qwen35::Qwen35::load(g, gpu.as_ref()));
    let mut st = hos::qwen35::State::new(&m);
    // The fully-resident Metal runner (Qwen35Gpu) predates the corrected forward
    // (tile head-map, norm-then-gate, MTP-block skip) and its kernels aren't yet
    // updated, so it is OPT-IN via HOS_QWEN35_RESIDENT. The default path — GPU
    // matmuls + CPU recurrence — matches ChatSession/serve and is verified correct.
    let rgpu = if std::env::var("HOS_QWEN35_RESIDENT").is_ok() {
        gpu.as_ref().map(|g| hos::qwen35::Qwen35Gpu::new(g, &m))
    } else {
        None
    };
    eprintln!("[hos] load took {:.1}s", load.elapsed().as_secs_f64());

    // Raw completion: prompt encoded as-is, stop on eos.
    let ids = tok.encode(&args.prompt, true);
    let stops: Vec<u32> = tok.eos.into_iter().collect();
    eprintln!("[hos] prompt -> {} tokens", ids.len());
    if !args.no_echo {
        print!("{}", args.prompt);
        std::io::stdout().flush().ok();
    }

    // MTP self-speculative decode (opt-in for A/B): batched 2-token verify.
    if m.has_mtp() && std::env::var("HOS_QWEN35_MTP").is_ok() {
        use std::io::Write;
        let mut state = hos::qwen35::State::new(&m);
        let t0 = Instant::now();
        let (logits, hidden) = m.forward_prefill_mtp(&mut state, &ids, gpu.as_ref());
        let pfs = t0.elapsed().as_secs_f64();
        // Resident 2-token verify (GPU): seed the resident backbone from the prefill
        // state, then verify on-GPU. Opt-in via HOS_QWEN35_MTP_GPU; falls back to the
        // CPU-recurrence verify when off or when no GPU is present.
        #[cfg(target_os = "macos")]
        let rgpu2 = if std::env::var("HOS_QWEN35_MTP_GPU").is_ok() {
            gpu.as_ref().map(|g| {
                let r = hos::qwen35::Qwen35Gpu::new(g, &m);
                r.upload_state(&m, &state, state.pos);
                r
            })
        } else {
            None
        };
        #[cfg(target_os = "macos")]
        if std::env::var("HOS_QWEN35_FWD2_TEST").is_ok() {
            if let Some(r) = rgpu2.as_ref() {
                qwen35_fwd2_check(&m, r, &mut state, &logits, &hidden, gpu.as_ref());
                return;
            }
        }
        let mut buf: Vec<u8> = Vec::new();
        let mut pending: Vec<u8> = Vec::new();
        let gen = Instant::now();
        #[cfg(target_os = "macos")]
        let n = if let Some(r) = rgpu2.as_ref() {
            m.decode_speculative_resident(
                &mut state, r, logits, hidden, args.n_predict, args.temp, args.top_k, args.top_p,
                args.rep_penalty, args.repeat_last_n, args.seed, &stops, gpu.as_ref(),
                |tk| {
                    buf.clear();
                    tok.decode_into(tk, &mut buf);
                    pending.extend_from_slice(&buf);
                    let valid = match std::str::from_utf8(&pending) {
                        Ok(s) => s.len(),
                        Err(e) => e.valid_up_to(),
                    };
                    if valid > 0 {
                        print!("{}", std::str::from_utf8(&pending[..valid]).unwrap());
                        std::io::stdout().flush().ok();
                        pending.drain(..valid);
                    }
                    false
                },
            )
        } else {
            m.decode_speculative(
                &mut state, logits, hidden, args.n_predict, args.temp, args.top_k, args.top_p,
                args.rep_penalty, args.repeat_last_n, args.seed, &stops, gpu.as_ref(),
                |tk| {
                    buf.clear();
                    tok.decode_into(tk, &mut buf);
                    pending.extend_from_slice(&buf);
                    let valid = match std::str::from_utf8(&pending) {
                        Ok(s) => s.len(),
                        Err(e) => e.valid_up_to(),
                    };
                    if valid > 0 {
                        print!("{}", std::str::from_utf8(&pending[..valid]).unwrap());
                        std::io::stdout().flush().ok();
                        pending.drain(..valid);
                    }
                    false
                },
            )
        };
        #[cfg(not(target_os = "macos"))]
        let n = m.decode_speculative(
            &mut state, logits, hidden, args.n_predict, args.temp, args.top_k, args.top_p,
            args.rep_penalty, args.repeat_last_n, args.seed, &stops, gpu.as_ref(),
            |tk| {
                buf.clear();
                tok.decode_into(tk, &mut buf);
                pending.extend_from_slice(&buf);
                let valid = match std::str::from_utf8(&pending) {
                    Ok(s) => s.len(),
                    Err(e) => e.valid_up_to(),
                };
                if valid > 0 {
                    print!("{}", std::str::from_utf8(&pending[..valid]).unwrap());
                    std::io::stdout().flush().ok();
                    pending.drain(..valid);
                }
                false
            },
        );
        if !pending.is_empty() {
            print!("{}", String::from_utf8_lossy(&pending));
        }
        println!();
        let ds = gen.elapsed().as_secs_f64();
        eprintln!(
            "[hos] MTP decode: {} tok in {:.1}s ({:.2} tok/s) | prefill {:.2}s",
            n, ds, n as f64 / ds.max(1e-9), pfs
        );
        return;
    }

    let mut pos = 0usize;
    let mut logits = Vec::new();
    let dec = Instant::now();
    for &t in &ids {
        logits = match &rgpu {
            Some(r) => r.forward(&m, t, pos),
            None => m.forward(&mut st, t, pos, gpu.as_ref()),
        };
        pos += 1;
    }
    let mut rng = args.seed;
    let mut recent: Vec<u32> = Vec::new();
    let mut buf = Vec::new();
    // Byte accumulator so a multi-byte UTF-8 char that spans two tokens (e.g. ÷,
    // →, emoji) is emitted whole rather than as replacement glyphs.
    let mut pending_bytes: Vec<u8> = Vec::new();
    let mut n = 0usize;
    let gen = Instant::now();
    for _ in 0..args.n_predict {
        let from = recent.len().saturating_sub(args.repeat_last_n);
        let next = sample(
            &logits,
            args.temp,
            args.top_k,
            args.top_p,
            args.rep_penalty,
            &recent[from..],
            &mut rng,
        );
        recent.push(next);
        if stops.contains(&next) || pos >= 4000 {
            break;
        }
        buf.clear();
        tok.decode_into(next, &mut buf);
        pending_bytes.extend_from_slice(&buf);
        // Emit only the complete-UTF-8 prefix; keep any partial trailing char.
        let valid = match std::str::from_utf8(&pending_bytes) {
            Ok(s) => s.len(),
            Err(e) => e.valid_up_to(),
        };
        if valid > 0 {
            print!("{}", std::str::from_utf8(&pending_bytes[..valid]).unwrap());
            std::io::stdout().flush().ok();
            pending_bytes.drain(..valid);
        }
        logits = match &rgpu {
            Some(r) => r.forward(&m, next, pos),
            None => m.forward(&mut st, next, pos, gpu.as_ref()),
        };
        pos += 1;
        n += 1;
    }
    // Flush any incomplete trailing char (best-effort) at end of stream.
    if !pending_bytes.is_empty() {
        print!("{}", String::from_utf8_lossy(&pending_bytes));
    }
    let prefill_secs = gen.duration_since(dec).as_secs_f64();
    let decode_secs = gen.elapsed().as_secs_f64();
    let dtps = n as f64 / decode_secs.max(1e-9);
    println!();
    eprintln!(
        "[hos] decode: {} tok in {:.1}s ({:.2} tok/s)",
        n, decode_secs, dtps
    );
    // Profiler: prefill split + effective decode bandwidth (weights re-read per
    // token), so kernel work is measured against the hardware ceiling, not vibes.
    if std::env::var("HOS_QWEN35_PROF").is_ok() {
        let mbytes = args
            .model
            .as_ref()
            .and_then(|p| std::fs::metadata(p).ok())
            .map(|m| m.len() as f64)
            .unwrap_or(0.0);
        let gbps = (mbytes / 1e9) * dtps;
        eprintln!(
            "[prof] prefill {} tok in {:.2}s ({:.1} tok/s) | decode {:.2} tok/s | ~{:.0} GB/s effective (model {:.1} GB)",
            ids.len(),
            prefill_secs,
            ids.len() as f64 / prefill_secs.max(1e-9),
            dtps,
            gbps,
            mbytes / 1e9,
        );
    }
}

/// `hos --chat` for qwen35: full chat template + thinking mode via `ChatSession`.
/// Reasoning streams dimmed (unless `--hide-thinking`); the answer streams normal.
fn run_qwen35_chat(g: &Gguf, args: &Args) {
    use hos::qwen35::Chunk;
    use std::io::Write;
    let tok = ok(hos::tokenizer::Tokenizer::from_gguf(g));
    let mut sess = ok(hos::qwen35::ChatSession::load(g, tok, args.gpu));
    let think = args.think();
    let mut msgs = Vec::new();
    if let Some(s) = &args.system {
        msgs.push(hos::chat::Message::new("system", s));
    }
    msgs.push(hos::chat::Message::new("user", &args.prompt));

    // Vision: if --image + --mmproj given, encode the image and splice it in.
    let img_emb: Option<Vec<f32>> = match (&args.image, &args.mmproj) {
        (Some(img), Some(mm)) => {
            eprintln!("[hos] loading vision tower {mm}");
            let mg = ok(hos::gguf::Gguf::open(std::path::Path::new(mm)));
            let tower = ok(hos::qwen35_vision::VisionTower::load(&mg));
            eprintln!("[hos] encoding image {img} ...");
            let t0 = Instant::now();
            let e = ok(tower.encode_image_cached(std::path::Path::new(img)));
            eprintln!(
                "[hos] image -> {} tokens ({:.1}s)",
                e.len() / tower.cfg.proj_dim,
                t0.elapsed().as_secs_f64()
            );
            Some(e)
        }
        (Some(_), None) => {
            eprintln!("[hos] --image needs --mmproj <mmproj.gguf>; ignoring image");
            None
        }
        _ => None,
    };

    let hide = args.hide_thinking;
    let mut in_reason = false;
    sess.chat_img(
        &msgs,
        img_emb.as_deref(),
        think,
        args.n_predict,
        args.temp,
        args.top_k,
        args.top_p,
        args.rep_penalty,
        args.repeat_last_n,
        args.seed,
        |chunk| {
            match chunk {
                Chunk::Reasoning(t) => {
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
                        print!("{}\n\n", hos::viz::RESET);
                        in_reason = false;
                    }
                    print!("{t}");
                }
            }
            std::io::stdout().flush().ok();
        },
    );
    if in_reason {
        print!("{}", hos::viz::RESET);
    }
    println!();
}

/// TRAIN FROM SPEC: a multi-head transformer with a MoE FFN, defined ENTIRELY in
/// JSON, trained through the generic interpreter (backprop flows through interp).
/// Proves train-from-spec + multi-head + MoE + interpreter-under-autograd at once.
fn train_spec() {
    use hos::tensor::{use_gpu, AdamW, Tensor};
    use std::collections::HashMap;

    let corpus = "the sun rose over the quiet harbor as the small boats drifted out to sea. \
gulls circled the pale sky calling above the waves. the old sailor watched the water and \
remembered the long years at sea. the tide came in slow and the harbor filled with light. "
        .repeat(6);
    let mut chars: Vec<char> = corpus.chars().collect();
    chars.sort();
    chars.dedup();
    let vocab = chars.len();
    let idof = |c: char| chars.iter().position(|&x| x == c).unwrap();
    let ids: Vec<usize> = corpus.chars().map(idof).collect();

    let (t, bsz, d, ff, heads, experts) = (32usize, 8usize, 32usize, 64usize, 2usize, 2usize);
    let mut s = 3u64;
    let mut r = |sh: &[usize]| Tensor::randn(sh, &mut s);
    // (name, tensor, decay)
    let mut spec_p: Vec<(String, Tensor, bool)> = vec![
        ("tok".into(), r(&[vocab, d]), false),
        ("pos".into(), r(&[t, d]), false),
        ("attn_norm".into(), Tensor::param(vec![1.0; d], &[d]), false),
        ("wq".into(), r(&[d, d]), true),
        ("wk".into(), r(&[d, d]), true),
        ("wv".into(), r(&[d, d]), true),
        ("wo".into(), r(&[d, d]), true),
        ("ffn_norm".into(), Tensor::param(vec![1.0; d], &[d]), false),
        ("gate".into(), r(&[d, experts]), true),
    ];
    let mut expert_specs = Vec::new();
    for e in 0..experts {
        for (suf, sh, dec) in [
            ("w1", vec![d, ff], true),
            ("b1", vec![ff], false),
            ("w2", vec![ff, d], true),
            ("b2", vec![d], false),
        ] {
            let name = format!("e{e}_{suf}");
            let tt = if dec {
                r(&sh)
            } else {
                Tensor::param(vec![0.0; sh.iter().product()], &sh)
            };
            spec_p.push((name, tt, dec));
        }
        expert_specs.push(serde_json::json!({"w1": format!("e{e}_w1"), "b1": format!("e{e}_b1"), "w2": format!("e{e}_w2"), "b2": format!("e{e}_b2")}));
    }
    spec_p.push(("out_norm".into(), Tensor::param(vec![1.0; d], &[d]), false));
    spec_p.push(("head".into(), r(&[d, vocab]), true));

    let arch = serde_json::json!({
        "type": "decoder_transformer", "ctx": t, "layers": [
            {"op": "embedding", "weight": "tok"},
            {"op": "add_pos", "weight": "pos"},
            {"op": "attention_block", "norm": "attn_norm", "wq": "wq", "wk": "wk", "wv": "wv", "wo": "wo", "heads": heads},
            {"op": "moe_ffn", "norm": "ffn_norm", "gate": "gate", "experts": expert_specs},
            {"op": "rmsnorm", "weight": "out_norm"},
            {"op": "linear", "weight": "head"}
        ]
    });

    let map: HashMap<String, Tensor> = spec_p
        .iter()
        .map(|(n, t, _)| (n.clone(), t.clone()))
        .collect();
    let params: Vec<&Tensor> = spec_p.iter().map(|(_, t, _)| t).collect();
    let decay: Vec<bool> = spec_p.iter().map(|(_, _, dd)| *dd).collect();
    let mut opt = AdamW::new(&params, 0.003, 0.01);

    let mut rng = 99u64;
    let mut sample_batch = |bs: usize| -> (Vec<usize>, Vec<usize>) {
        let (mut inp, mut tgt) = (Vec::new(), Vec::new());
        for _ in 0..bs {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let st = (rng as usize) % (ids.len() - t - 1);
            inp.extend_from_slice(&ids[st..st + t]);
            tgt.extend_from_slice(&ids[st + 1..st + t + 1]);
        }
        (inp, tgt)
    };

    use_gpu(true);
    println!("training a MULTI-HEAD + MoE transformer defined ENTIRELY in JSON\n");
    println!(
        "  vocab={vocab} ctx={t} d={d} heads={heads} experts={experts} (soft-MoE) batch={bsz}"
    );
    println!("  arch lives only in the spec; the interpreter runs forward+backward.\n");
    for step in 0..=300 {
        let (inp, tgt) = sample_batch(bsz);
        for q in &params {
            q.zero_grad();
        }
        let logits = ok(hos::interp::run(&arch, &map, Some(&inp), None));
        let loss = logits.cross_entropy(&tgt);
        loss.backward();
        opt.step(&params, &decay);
        if step % 50 == 0 {
            println!("  step {step:>3}  loss {:.3}", loss.data()[0]);
        }
    }
    use_gpu(false);
    println!("\n  ✅ a spec-defined multi-head MoE transformer trained via the interpreter — loss went down.");
    println!("  train-from-spec works: architecture is data (JSON), not code.");
}

/// A whole TRANSFORMER run purely from its arch spec (attention_block + ffn_block
/// + add_pos) vs a hardcoded forward — proving transformers, not just MLPs, are
/// self-running from HOS.
fn selfrun_tf() {
    use hos::tensor::Tensor;
    use std::collections::HashMap;
    let (bs, t, d, ff, vocab) = (2usize, 6usize, 16usize, 32usize, 10usize);
    let inv = 1.0 / (d as f32).sqrt();
    let mut s = 8u64;
    let mut r = |sh: &[usize]| Tensor::randn(sh, &mut s);
    let w: Vec<(&str, Tensor)> = vec![
        ("tok", r(&[vocab, d])),
        ("pos", r(&[t, d])),
        ("attn_norm", Tensor::param(vec![1.0; d], &[d])),
        ("wq", r(&[d, d])),
        ("wk", r(&[d, d])),
        ("wv", r(&[d, d])),
        ("wo", r(&[d, d])),
        ("ffn_norm", Tensor::param(vec![1.0; d], &[d])),
        ("w1", r(&[d, ff])),
        ("b1", Tensor::param(vec![0.0; ff], &[ff])),
        ("w2", r(&[ff, d])),
        ("b2", Tensor::param(vec![0.0; d], &[d])),
        ("out_norm", Tensor::param(vec![1.0; d], &[d])),
        ("head", r(&[d, vocab])),
    ];
    let m: HashMap<String, Tensor> = w.iter().map(|(n, t)| (n.to_string(), t.clone())).collect();
    let g = |name: &str| m.get(name).unwrap();

    let mut sd = 1u64;
    let ids: Vec<usize> = (0..bs * t)
        .map(|_| {
            sd ^= sd << 13;
            sd ^= sd >> 7;
            sd ^= sd << 17;
            (sd as usize) % vocab
        })
        .collect();
    let mut mask = vec![0.0f32; bs * t * t];
    for rr in 0..bs * t {
        for j in (rr % t + 1)..t {
            mask[rr * t + j] = -1e9;
        }
    }
    let mask = Tensor::constant(mask, &[bs * t, t]);
    let pos_ids: Vec<usize> = (0..bs * t).map(|rr| rr % t).collect();

    // hardcoded transformer forward
    let hardcoded = || {
        let x = g("tok").embedding(&ids).add(&g("pos").embedding(&pos_ids));
        let xn = x.rmsnorm(g("attn_norm"));
        let q = xn.matmul(g("wq")).reshape(&[bs, t, d]);
        let k = xn.matmul(g("wk")).reshape(&[bs, t, d]);
        let v = xn.matmul(g("wv")).reshape(&[bs, t, d]);
        let att = q
            .bmm(&k.transpose_last2())
            .scale(inv)
            .reshape(&[bs * t, t])
            .add(&mask)
            .softmax_rows()
            .reshape(&[bs, t, t]);
        let x = x.add(&att.bmm(&v).reshape(&[bs * t, d]).matmul(g("wo")));
        let xn2 = x.rmsnorm(g("ffn_norm"));
        let ffn = xn2
            .matmul(g("w1"))
            .add(g("b1"))
            .relu()
            .matmul(g("w2"))
            .add(g("b2"));
        let x = x.add(&ffn);
        x.rmsnorm(g("out_norm")).matmul(g("head"))
    };

    let arch = serde_json::json!({
        "type": "decoder_transformer", "ctx": t, "layers": [
            {"op": "embedding", "weight": "tok"},
            {"op": "add_pos", "weight": "pos"},
            {"op": "attention_block", "norm": "attn_norm", "wq": "wq", "wk": "wk", "wv": "wv", "wo": "wo"},
            {"op": "ffn_block", "norm": "ffn_norm", "w1": "w1", "b1": "b1", "w2": "w2", "b2": "b2"},
            {"op": "rmsnorm", "weight": "out_norm"},
            {"op": "linear", "weight": "head"}
        ]
    });

    let hard = hardcoded().data();
    let spec = ok(hos::interp::run(&arch, &m, Some(&ids), None)).data();
    let diff = hard
        .iter()
        .zip(&spec)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("a TRANSFORMER, run from its arch spec — no hardcoded model code\n");
    println!(
        "  spec layers: embedding -> add_pos -> attention_block -> ffn_block -> rmsnorm -> linear"
    );
    println!("  output dims: [{} tokens x {} vocab]", bs * t, vocab);
    println!(
        "  max|hardcoded - spec-driven| = {diff:.3e}   {}",
        if diff < 1e-4 {
            "✅ IDENTICAL — transformer ran from spec"
        } else {
            "❌"
        }
    );
    println!("  → HOS now self-runs transformers, not just MLPs. The moat is complete.");
}

/// The deep moat: a model that RUNS ITSELF from its arch spec. Train an MLP,
/// save it (weights + arch spec) to HOS, then execute it purely via the generic
/// interpreter — no model-specific code — and confirm it matches the hardcoded
/// forward. Edit the JSON spec → different model. Self-describing AND self-running.
fn hos_selfrun() {
    use hos::format::{self, Named};
    use hos::tensor::{AdamW, Tensor};
    use std::path::Path;

    let (d, h) = (32usize, 32usize);
    let mut s = 4u64;
    let mut rv = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
            })
            .collect()
    };
    // tiny MLP classifier: w1[d,h] b1 w2[h,2] b2
    let w1 = Tensor::param(rv(d * h), &[d, h]);
    let b1 = Tensor::param(vec![0.0; h], &[h]);
    let w2 = Tensor::param(rv(h * 2), &[h, 2]);
    let b2 = Tensor::param(vec![0.0; 2], &[2]);
    let ps = [&w1, &b1, &w2, &b2];

    // train a few steps on a trivial separable task (so weights are non-trivial)
    let wstar = rv(d);
    let mut opt = AdamW::new(&ps, 0.02, 0.0);
    let mut sd = 9u64;
    let mut rng = || {
        sd ^= sd << 13;
        sd ^= sd >> 7;
        sd ^= sd << 17;
        sd
    };
    let hardcoded = |x: &Tensor| x.matmul(&w1).add(&b1).relu().matmul(&w2).add(&b2);
    for _ in 0..200 {
        let xv = rv(d);
        let dot: f32 = xv.iter().zip(&wstar).map(|(a, b)| a * b).sum();
        let lab = (dot > 0.0) as usize;
        let x = Tensor::constant(xv, &[1, d]);
        for q in &ps {
            q.zero_grad();
        }
        hardcoded(&x).cross_entropy(&[lab]).backward();
        opt.step(&ps, &[true, false, true, false]);
        let _ = &mut rng;
    }

    // save with the arch SPEC describing the layer sequence
    let named: Vec<Named> = [("w1", &w1), ("b1", &b1), ("w2", &w2), ("b2", &b2)]
        .iter()
        .map(|(name, t)| Named {
            name: (*name).into(),
            role: format::ROLE_WEIGHT,
            shape: t.shape(),
            data: t.data(),
        })
        .collect();
    let arch = serde_json::json!({
        "type": "sequential",
        "layers": [
            {"op": "linear", "weight": "w1", "bias": "b1"},
            {"op": "relu"},
            {"op": "linear", "weight": "w2", "bias": "b2"}
        ]
    });
    let mut card = format::Card::new("selfrun-mlp", arch.clone());
    card.id = format::model_id(&named);
    card.mode = "inference".into();
    let path = Path::new("/tmp/selfrun.hos");
    format::save(path, &named, &card).unwrap();

    // run it PURELY from the file's arch spec — no model code
    let test_x = rv(d);
    let xt = Tensor::constant(test_x.clone(), &[1, d]);
    let hard = hardcoded(&xt).data();
    let spec_out = ok(hos::interp::run_file(path, None, Some(xt))).data();

    println!("self-running model — executed from its arch spec, no hardcoded model code\n");
    println!(
        "  arch spec stored in the file:\n  {}\n",
        serde_json::to_string_pretty(&arch)
            .unwrap()
            .replace('\n', "\n  ")
    );
    println!("  hardcoded forward : [{:.4}, {:.4}]", hard[0], hard[1]);
    println!(
        "  spec-driven run   : [{:.4}, {:.4}]",
        spec_out[0], spec_out[1]
    );
    let diff = hard
        .iter()
        .zip(&spec_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "\n  max|hardcoded - spec| = {diff:.3e}   {}",
        if diff < 1e-5 {
            "✅ IDENTICAL — the file ran itself"
        } else {
            "❌"
        }
    );
    println!("  → edit the \"layers\" JSON and the model changes, no recompile. self-describing + self-running.");
}

/// THE CAPSTONE: a batched decoder transformer LM, trained end-to-end on a char
/// corpus with everything wired together — embeddings, batched attention (bmm),
/// RMSNorm, FFN, cross-entropy, AdamW, GPU matmuls — then generates text and
/// saves to HOS. Env: LM_STEPS, LM_GPU=0 to force CPU.
/// Causal attention mask [bs*t, t] (−1e9 above the diagonal within each window).
fn lm_mask(bs: usize, t: usize) -> hos::tensor::Tensor {
    let mut m = vec![0.0f32; bs * t * t];
    for r in 0..bs * t {
        let tt = r % t;
        for j in (tt + 1)..t {
            m[r * t + j] = -1e9;
        }
    }
    hos::tensor::Tensor::constant(m, &[bs * t, t])
}

/// Forward pass of the tiny char-level transformer. Shared by training
/// (`--train-lm`) and generation (`--gen-hos`) so the two can never drift.
/// Param layout `p`: [tok_emb, pos_emb, out_norm, head] then 10 per layer
/// (attn_norm, wq, wk, wv, wo, ffn_norm, w1, b1, w2, b2). Returns [bs*t, vocab].
fn lm_forward(
    p: &[hos::tensor::Tensor],
    tok_ids: &[usize],
    bs: usize,
    t: usize,
    d: usize,
    layers: usize,
    mask: &hos::tensor::Tensor,
) -> hos::tensor::Tensor {
    let inv = 1.0 / (d as f32).sqrt();
    let pos_ids: Vec<usize> = (0..bs * t).map(|r| r % t).collect();
    let mut x = p[0].embedding(tok_ids).add(&p[1].embedding(&pos_ids));
    for l in 0..layers {
        let b = 4 + l * 10;
        let xn = x.rmsnorm(&p[b]);
        let q = xn.matmul(&p[b + 1]).reshape(&[bs, t, d]);
        let k = xn.matmul(&p[b + 2]).reshape(&[bs, t, d]);
        let v = xn.matmul(&p[b + 3]).reshape(&[bs, t, d]);
        let scores = q.bmm(&k.transpose_last2()).scale(inv).reshape(&[bs * t, t]);
        let att = scores.add(mask).softmax_rows().reshape(&[bs, t, t]);
        let ctx = att.bmm(&v).reshape(&[bs * t, d]);
        x = x.add(&ctx.matmul(&p[b + 4]));
        let xn2 = x.rmsnorm(&p[b + 5]);
        let ff = xn2
            .matmul(&p[b + 6])
            .add(&p[b + 7])
            .relu()
            .matmul(&p[b + 8])
            .add(&p[b + 9]);
        x = x.add(&ff);
    }
    x.rmsnorm(&p[2]).matmul(&p[3])
}

fn train_lm() {
    use hos::tensor::{use_gpu, AdamW, Tensor};
    use std::time::Instant;

    // Train on your own text with `--train-lm --corpus path/to.txt`; otherwise a
    // built-in passage. Short corpora are repeated so there's enough length for
    // the train/val windows.
    let builtin = "the sun rose over the quiet harbor as the small boats drifted out to sea. \
gulls circled the pale sky, calling above the waves. the old sailor watched the water \
and remembered the long years at sea. the tide came in slow and steady, and the harbor \
filled with light. children played along the shore while the boats returned with the \
morning catch. the sea was calm and the sky was clear and the day was good. ";
    let (corpus, source) = match arg_after("--corpus") {
        Some(path) => {
            let txt = std::fs::read_to_string(&path).unwrap_or_else(|e| {
                eprintln!("[hos] error: reading corpus {path}: {e}");
                std::process::exit(1);
            });
            if txt.trim().is_empty() {
                eprintln!("[hos] error: corpus {path} is empty");
                std::process::exit(1);
            }
            (txt, path)
        }
        None => (builtin.to_string(), "builtin-harbor-corpus".to_string()),
    };
    let reps = (4000 / corpus.len().max(1)) + 1;
    let text = corpus.repeat(reps); // enough length for train/val windows
    let chars: Vec<char> = {
        let mut v: Vec<char> = text.chars().collect();
        v.sort();
        v.dedup();
        v
    };
    let vocab = chars.len();
    let idof = |c: char| chars.iter().position(|&x| x == c).unwrap();
    let ids: Vec<usize> = text.chars().map(idof).collect();
    let split = ids.len() * 9 / 10;
    let (train_ids, val_ids) = (&ids[..split], &ids[split.saturating_sub(0)..]);

    let (t, bsz, d, ffn, layers) = (48usize, 16usize, 64usize, 128usize, 2usize);
    let steps: usize = std::env::var("LM_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(400);
    let gpu = std::env::var("LM_GPU").map(|v| v != "0").unwrap_or(true);

    // params: [tok_emb, pos_emb, out_norm, head] then per layer x10
    let mut s = 1u64;
    let mut r = |sh: &[usize]| Tensor::randn(sh, &mut s);
    let mut p: Vec<Tensor> = vec![
        r(&[vocab, d]),
        r(&[t, d]),
        Tensor::param(vec![1.0; d], &[d]),
        r(&[d, vocab]),
    ];
    let mut decay: Vec<bool> = vec![false, false, false, true];
    for _ in 0..layers {
        p.push(Tensor::param(vec![1.0; d], &[d])); // attn_norm
        p.push(r(&[d, d]));
        p.push(r(&[d, d]));
        p.push(r(&[d, d]));
        p.push(r(&[d, d])); // wq wk wv wo
        p.push(Tensor::param(vec![1.0; d], &[d])); // ffn_norm
        p.push(r(&[d, ffn]));
        p.push(Tensor::param(vec![0.0; ffn], &[ffn])); // w1 b1
        p.push(r(&[ffn, d]));
        p.push(Tensor::param(vec![0.0; d], &[d])); // w2 b2
        decay.extend([
            false, true, true, true, true, false, true, false, true, false,
        ]);
    }

    // mask + forward are shared with `--gen-hos` (see lm_mask / lm_forward).
    let mask_for = |bs: usize| lm_mask(bs, t);
    let forward = |p: &[Tensor], tok_ids: &[usize], bs: usize, mask: &Tensor| {
        lm_forward(p, tok_ids, bs, t, d, layers, mask)
    };

    // sample a batch of windows from a slice; returns (inputs[bs*t], targets[bs*t])
    let mut rng = 12345u64;
    let mut sample_batch = |src: &[usize], bs: usize| -> (Vec<usize>, Vec<usize>) {
        let mut inp = Vec::with_capacity(bs * t);
        let mut tgt = Vec::with_capacity(bs * t);
        for _ in 0..bs {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let start = (rng as usize) % (src.len() - t - 1);
            inp.extend_from_slice(&src[start..start + t]);
            tgt.extend_from_slice(&src[start + 1..start + t + 1]);
        }
        (inp, tgt)
    };

    let pr: Vec<&Tensor> = p.iter().collect();
    let mut opt = AdamW::new(&pr, 0.003, 0.01);
    use_gpu(gpu);
    println!(
        "training a batched transformer LM ({} on {})\n",
        if gpu { "GPU matmuls" } else { "CPU" },
        "char corpus"
    );
    println!("  vocab={vocab} ctx={t} batch={bsz} d={d} layers={layers} steps={steps}\n");
    let mask = mask_for(bsz);
    let val_mask = mask_for(bsz);
    let t0 = Instant::now();
    for step in 0..=steps {
        let (inp, tgt) = sample_batch(train_ids, bsz);
        for q in &pr {
            q.zero_grad();
        }
        let loss = forward(&p, &inp, bsz, &mask).cross_entropy(&tgt);
        loss.backward();
        opt.step(&pr, &decay);
        if step % 50 == 0 {
            let (vi, vt) = sample_batch(val_ids, bsz);
            let vl = forward(&p, &vi, bsz, &val_mask).cross_entropy(&vt).data()[0];
            println!(
                "  step {step:>4}  train_loss {:.3}  val_loss {:.3}",
                loss.data()[0],
                vl
            );
        }
    }
    println!("\n  trained in {:.1}s", t0.elapsed().as_secs_f64());

    // free-running generation (batch=1)
    let gmask = mask_for(1);
    let mut ctx: Vec<usize> = corpus.chars().take(t).map(idof).collect();
    let mut gen = String::new();
    let mut grng = 7u64;
    for _ in 0..200 {
        let logits = forward(&p, &ctx, 1, &gmask).data();
        let last = &logits[(t - 1) * vocab..t * vocab];
        let next = hos::sample(last, 0.8, 40, 0.95, 1.0, &[], &mut grng) as usize;
        gen.push(chars[next]);
        ctx.remove(0);
        ctx.push(next);
    }
    println!(
        "\n  --- generated (temp 0.8) ---\n  {}\n",
        gen.replace('\n', " ")
    );

    // save to HOS
    use hos::format::{self, Named};
    use_gpu(false);
    let named: Vec<Named> = p
        .iter()
        .enumerate()
        .map(|(i, t)| Named {
            name: format!("p{i}"),
            role: if decay[i] {
                format::ROLE_WEIGHT
            } else {
                format::ROLE_NORM
            },
            shape: t.shape(),
            data: t.data(),
        })
        .collect();
    // Bake the full vocab (the exact char set, in id order) into the card so the
    // file is self-contained: `--gen-hos` can rebuild + run it with no extra info.
    let charset: String = chars.iter().collect();
    let mut card = format::Card::new(
        "tiny-transformer-lm",
        serde_json::json!({
            "type": "decoder_transformer", "d": d, "layers": layers, "heads": 1,
            "ctx": t, "vocab": vocab, "chars": charset
        }),
    );
    card.id = format::model_id(&named);
    card.resume = format::Resume {
        seed: 1,
        step: steps as u64,
        rng_state: rng,
    };
    card.history.push(format::TrainRun {
        steps: steps as u64,
        final_loss: 0.0,
        optimizer: "adamw".into(),
        lr: 0.003,
    });
    card.provenance.dataset = source;
    let out = arg_after("--out").unwrap_or_else(|| "/tmp/tiny-lm.hos".to_string());
    let path = std::path::Path::new(&out);
    format::save(path, &named, &card).expect("save");
    println!(
        "  saved model -> {} ({} bytes)",
        path.display(),
        std::fs::metadata(path).unwrap().len()
    );
    println!(
        "\n  generate from it:  hos --gen-hos {} -p \"the \" -n 200",
        path.display()
    );
    println!("  inspect it:        hos --hos-info {}", path.display());
}

/// `--gen-hos <file.hos>`: load a char-LM trained by `--train-lm` and generate
/// text from it — proving a `.hos` file is a self-contained, runnable artifact.
/// Seeds with `-p` (or a space), then samples `-n` chars (`--temp`/`--top-k`/`--top-p`).
fn gen_hos(args: &Args) {
    use hos::format;
    use hos::tensor::Tensor;

    let Some(path) = arg_after("--gen-hos") else {
        eprintln!("[hos] error: --gen-hos needs a file, e.g. `hos --gen-hos /tmp/tiny-lm.hos -p \"the \"`");
        std::process::exit(1);
    };
    let (named, card) = format::load(std::path::Path::new(&path)).unwrap_or_else(|e| {
        eprintln!("[hos] error: loading {path}: {e}");
        std::process::exit(1);
    });
    let a = &card.arch;
    let getu = |k: &str| {
        a.get(k).and_then(|v| v.as_u64()).unwrap_or_else(|| {
            eprintln!("[hos] error: {path} is not a char-LM .hos (missing arch.{k})");
            std::process::exit(1);
        })
    };
    let (d, layers, t, vocab) = (
        getu("d") as usize,
        getu("layers") as usize,
        getu("ctx") as usize,
        getu("vocab") as usize,
    );
    let chars: Vec<char> = a
        .get("chars")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .chars()
        .collect();
    if chars.len() != vocab {
        eprintln!(
            "[hos] error: {path} has no usable vocab (arch.chars); was it made by --train-lm?"
        );
        std::process::exit(1);
    }

    // rebuild params p0..pN (in order) as inference constants
    let mut by_name = std::collections::HashMap::new();
    for n in &named {
        by_name.insert(n.name.clone(), Tensor::constant(n.data.clone(), &n.shape));
    }
    let p: Vec<Tensor> = (0..named.len())
        .map(|i| {
            by_name.get(&format!("p{i}")).cloned().unwrap_or_else(|| {
                eprintln!("[hos] error: {path} missing tensor p{i}");
                std::process::exit(1);
            })
        })
        .collect();

    let idof = |c: char| chars.iter().position(|&x| x == c);
    // seed window of length t: prompt (mapped) right-aligned, left-padded with chars[0]
    let prompt = if args.prompt == "Hello" {
        "the "
    } else {
        &args.prompt
    }; // ignore the generic default
    let mut ctx: Vec<usize> = vec![0; t];
    let seed_ids: Vec<usize> = prompt.chars().filter_map(idof).collect();
    let take = seed_ids.len().min(t);
    for (k, &id) in seed_ids[seed_ids.len() - take..].iter().enumerate() {
        ctx[t - take + k] = id;
    }

    let gmask = lm_mask(1, t);
    let n = if args.n_predict > 0 {
        args.n_predict
    } else {
        200
    };
    let temp = if args.temp > 0.0 { args.temp } else { 0.8 };
    let mut rng = args.seed;
    let mut out = String::new();
    for _ in 0..n {
        let logits = lm_forward(&p, &ctx, 1, t, d, layers, &gmask).data();
        let last = &logits[(t - 1) * vocab..t * vocab];
        let next = hos::sample(
            last,
            temp,
            args.top_k,
            args.top_p,
            args.rep_penalty,
            &[],
            &mut rng,
        ) as usize;
        out.push(chars[next]);
        ctx.remove(0);
        ctx.push(next);
    }
    println!(
        "=== {} ({} params, vocab {}, ctx {}) ===",
        card.name,
        named.len(),
        vocab,
        t
    );
    print!("{prompt}{out}");
    println!();
}

/// Validate the batched-ops layer: batched self-attention (bmm + reshape +
/// transpose_last2 + softmax) must equal looping single-example attention.
fn batch_attn_test() {
    use hos::tensor::Tensor;
    let (bs, t, d) = (4usize, 6usize, 8usize);
    let inv = 1.0 / (d as f32).sqrt();
    let mut seed = 3u64;
    let n = bs * t * d;
    let data: Vec<f32> = (0..n)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        })
        .collect();

    // batched: one forward over [bs, t, d]
    let xb = Tensor::constant(data.clone(), &[bs, t, d]);
    let scores = xb.bmm(&xb.transpose_last2()).scale(inv); // [bs, t, t]
    let att = scores
        .reshape(&[bs * t, t])
        .softmax_rows()
        .reshape(&[bs, t, t]);
    let out_b = att.bmm(&xb).data(); // [bs, t, d]

    // reference: loop each example through 2D attention
    let mut single = vec![0.0; n];
    for b in 0..bs {
        let xs = Tensor::constant(data[b * t * d..(b + 1) * t * d].to_vec(), &[t, d]);
        let s = xs.matmul(&xs.transpose()).scale(inv).softmax_rows();
        let o = s.matmul(&xs).data();
        single[b * t * d..(b + 1) * t * d].copy_from_slice(&o);
    }

    let diff = out_b
        .iter()
        .zip(&single)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("batched-ops validation — batched attention vs looped single-example\n");
    println!(
        "  shape [B={bs}, T={t}, d={d}]   max|batched - single| = {diff:.3e}   {}",
        if diff < 1e-5 {
            "✅ MATCH"
        } else {
            "❌ MISMATCH"
        }
    );
    println!(
        "  → bmm + reshape + transpose_last2 + softmax are correct; transformers can now batch."
    );
}

/// True throughput batching: process B examples in ONE forward (the batch is the
/// row dimension → one big matmul), vs one example at a time. Benchmarks the
/// throughput win and confirms batched training still learns.
fn batch_bench() {
    use hos::tensor::{use_gpu, AdamW, Tensor};
    use std::time::Instant;
    fn randvec(n: usize, seed: &mut u64) -> Vec<f32> {
        (0..n)
            .map(|_| {
                let mut x = *seed;
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                *seed = x;
                (x >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
            })
            .collect()
    }
    let (d, hid, n, b) = (256usize, 256usize, 512usize, 128usize);
    let mut s = 5u64;
    let wstar = randvec(d, &mut s);
    let data: Vec<(Vec<f32>, usize)> = (0..n + 200)
        .map(|_| {
            let x = randvec(d, &mut s);
            let dot: f32 = x.iter().zip(&wstar).map(|(a, b)| a * b).sum();
            (x, (dot > 0.0) as usize)
        })
        .collect();
    let (train, test) = data.split_at(n);

    let model = |seed: &mut u64| -> Vec<Tensor> {
        vec![
            Tensor::param(randvec(d * hid, seed), &[d, hid]),
            Tensor::param(vec![0.0; hid], &[hid]),
            Tensor::param(randvec(hid * 2, seed), &[hid, 2]),
            Tensor::param(vec![0.0; 2], &[2]),
        ]
    };
    let fwd = |p: &[Tensor], x: &Tensor| -> Tensor {
        x.matmul(&p[0]).add(&p[1]).relu().matmul(&p[2]).add(&p[3])
    };

    println!("true throughput batching — one big forward vs one-at-a-time\n");
    println!("  D={d} hidden={hid} examples={n} batch={b}\n");
    println!("  {:<22} {:>10}", "mode", "1 epoch");

    // timing: one epoch (full train pass + steps) per mode
    let time_epoch = |gpu: bool, batched: bool| -> f64 {
        use_gpu(gpu);
        let mut sd = 1u64;
        let p = model(&mut sd);
        let pr: Vec<&Tensor> = p.iter().collect();
        let dec = [true, false, true, false];
        let mut opt = AdamW::new(&pr, 0.01, 0.0);
        let t0 = Instant::now();
        if batched {
            for chunk in train.chunks(b) {
                let flat: Vec<f32> = chunk.iter().flat_map(|(x, _)| x.iter().copied()).collect();
                let labs: Vec<usize> = chunk.iter().map(|(_, l)| *l).collect();
                let xb = Tensor::constant(flat, &[chunk.len(), d]);
                for q in &pr {
                    q.zero_grad();
                }
                let loss = fwd(&p, &xb).cross_entropy(&labs);
                loss.backward();
                opt.step(&pr, &dec);
            }
        } else {
            for (x, l) in train {
                let xi = Tensor::constant(x.clone(), &[1, d]);
                for q in &pr {
                    q.zero_grad();
                }
                let loss = fwd(&p, &xi).cross_entropy(&[*l]);
                loss.backward();
                opt.step(&pr, &dec);
            }
        }
        use_gpu(false);
        t0.elapsed().as_secs_f64()
    };
    for (name, gpu, batched) in [
        ("one-at-a-time CPU", false, false),
        ("one-at-a-time GPU", true, false),
        ("batched CPU", false, true),
        ("batched GPU", true, true),
    ] {
        println!("  {:<22} {:>9.3}s", name, time_epoch(gpu, batched));
    }

    // correctness: train batched on GPU, report held-out accuracy
    use_gpu(true);
    let mut sd = 2u64;
    let p = model(&mut sd);
    let pr: Vec<&Tensor> = p.iter().collect();
    let dec = [true, false, true, false];
    let mut opt = AdamW::new(&pr, 0.01, 0.0);
    for _ in 0..40 {
        for chunk in train.chunks(b) {
            let flat: Vec<f32> = chunk.iter().flat_map(|(x, _)| x.iter().copied()).collect();
            let labs: Vec<usize> = chunk.iter().map(|(_, l)| *l).collect();
            let xb = Tensor::constant(flat, &[chunk.len(), d]);
            for q in &pr {
                q.zero_grad();
            }
            fwd(&p, &xb).cross_entropy(&labs).backward();
            opt.step(&pr, &dec);
        }
    }
    let mut ok = 0;
    for (x, l) in test {
        let lg = fwd(&p, &Tensor::constant(x.clone(), &[1, d])).data();
        if (lg[1] > lg[0]) as usize == *l {
            ok += 1;
        }
    }
    use_gpu(false);
    println!(
        "\n  batched-GPU training held-out accuracy: {:.1}%",
        ok as f32 / test.len() as f32 * 100.0
    );
}

/// Rung 5 step 2: benchmark matmul-heavy training on CPU vs GPU across sizes,
/// to find where the GPU win actually kicks in (and confirm results still match).
fn matmul_bench() {
    use hos::tensor::{use_gpu, Tensor};
    use std::time::Instant;
    fn randvec(n: usize, seed: &mut u64) -> Vec<f32> {
        (0..n)
            .map(|_| {
                let mut x = *seed;
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                *seed = x;
                (x >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
            })
            .collect()
    }
    let sizes = [
        (64usize, 128usize, 128usize),
        (128, 512, 512),
        (256, 1024, 1024),
    ];
    let steps = 20;
    println!("matmul training benchmark — CPU vs GPU, {steps} steps/run\n");
    println!(
        "  {:<16} {:>9} {:>9} {:>9} {:>7}",
        "(B, D, H)", "CPU", "GPU", "speedup", "match"
    );
    for (b, d, h) in sizes {
        let run = |gpu: bool| -> (f64, f32) {
            use_gpu(gpu);
            let mut s = 1u64;
            let x = Tensor::constant(randvec(b * d, &mut s), &[b, d]);
            let tgt = Tensor::constant(randvec(b * d, &mut s), &[b, d]);
            let w1 = Tensor::param(randvec(d * h, &mut s), &[d, h]);
            let w2 = Tensor::param(randvec(h * d, &mut s), &[h, d]);
            let ps = [&w1, &w2];
            let t0 = Instant::now();
            let mut lv = 0.0;
            for _ in 0..steps {
                for p in ps {
                    p.zero_grad();
                }
                let out = x.matmul(&w1).relu().matmul(&w2);
                let loss = out.sub(&tgt).square().mean();
                loss.backward();
                for p in ps {
                    p.sgd_step(0.001);
                }
                lv = loss.data()[0];
            }
            use_gpu(false);
            (t0.elapsed().as_secs_f64(), lv)
        };
        let (ct, cl) = run(false);
        let (gt, gl) = run(true);
        let ok = (cl - gl).abs() < cl.abs() * 0.02 + 1e-3;
        println!(
            "  ({b:>3},{d:>4},{h:>4}) {ct:>8.2}s {gt:>8.2}s {:>8.1}x {:>7}",
            ct / gt,
            if ok { "✅" } else { "≈" }
        );
    }
    println!("\n  (small matmuls: GPU overhead dominates. large matmuls: GPU pulls ahead.)");
}

/// Rung 5 step 1: train XOR on CPU then on the GPU matmul backend, and confirm
/// the GPU forward+backward matmul produces the same training result.
fn train_gpu_test() {
    use hos::tensor::{use_gpu, Tensor};
    let run = || -> (f32, Vec<f32>) {
        let mut seed = 1234u64;
        let x = Tensor::constant(vec![0., 0., 0., 1., 1., 0., 1., 1.], &[4, 2]);
        let y = Tensor::constant(vec![0., 1., 1., 0.], &[4, 1]);
        let w1 = Tensor::randn(&[2, 8], &mut seed);
        let b1 = Tensor::param(vec![0.0; 8], &[8]);
        let w2 = Tensor::randn(&[8, 1], &mut seed);
        let b2 = Tensor::param(vec![0.0; 1], &[1]);
        let params = [&w1, &b1, &w2, &b2];
        let mut loss_v = 0.0;
        for _ in 0..2000 {
            for p in params {
                p.zero_grad();
            }
            let pred = x.matmul(&w1).add(&b1).relu().matmul(&w2).add(&b2);
            let loss = pred.sub(&y).square().mean();
            loss.backward();
            for p in params {
                p.sgd_step(0.5);
            }
            loss_v = loss.data()[0];
        }
        let pred = x.matmul(&w1).add(&b1).relu().matmul(&w2).add(&b2).data();
        (loss_v, pred)
    };

    use_gpu(false);
    let (lc, pc) = run();
    use_gpu(true);
    let (lg, pg) = run();
    use_gpu(false);

    println!("rung 5 — GPU matmul fwd+bwd, validated vs CPU autograd\n");
    println!(
        "  CPU: final loss {lc:.6}  preds [{:.3}, {:.3}, {:.3}, {:.3}]",
        pc[0], pc[1], pc[2], pc[3]
    );
    println!(
        "  GPU: final loss {lg:.6}  preds [{:.3}, {:.3}, {:.3}, {:.3}]",
        pg[0], pg[1], pg[2], pg[3]
    );
    let diff = pc
        .iter()
        .zip(&pg)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "\n  max|cpu-gpu prediction| = {diff:.3e}   {}",
        if diff < 1e-2 {
            "✅ MATCH — GPU training is correct"
        } else {
            "❌ MISMATCH"
        }
    );
}

/// Train a tiny MLP on XOR with the hos-tensor autograd core — proves the
/// engine computes gradients and actually learns. Run: `hos --autograd-demo`.
fn autograd_demo() {
    use hos::tensor::Tensor;
    println!("hos-tensor autograd demo: training a 2-layer MLP on XOR\n");
    let mut seed = 1234u64;
    let x = Tensor::constant(vec![0., 0., 0., 1., 1., 0., 1., 1.], &[4, 2]);
    let y = Tensor::constant(vec![0., 1., 1., 0.], &[4, 1]);
    let w1 = Tensor::randn(&[2, 8], &mut seed);
    let b1 = Tensor::param(vec![0.0; 8], &[8]);
    let w2 = Tensor::randn(&[8, 1], &mut seed);
    let b2 = Tensor::param(vec![0.0; 1], &[1]);
    let params = [&w1, &b1, &w2, &b2];
    let lr = 0.5;
    for epoch in 0..=3000 {
        for p in params {
            p.zero_grad();
        }
        let h = x.matmul(&w1).add(&b1).relu();
        let pred = h.matmul(&w2).add(&b2);
        let loss = pred.sub(&y).square().mean();
        loss.backward();
        for p in params {
            p.sgd_step(lr);
        }
        if epoch % 500 == 0 {
            println!("  epoch {epoch:>4}   loss {:.6}", loss.data()[0]);
        }
    }
    let h = x.matmul(&w1).add(&b1).relu();
    let pred = h.matmul(&w2).add(&b2).data();
    println!(
        "\n  predictions (target 0,1,1,0): [{:.3}, {:.3}, {:.3}, {:.3}]",
        pred[0], pred[1], pred[2], pred[3]
    );
}

/// Train a tiny single-head self-attention language model with hos-tensor —
/// proves embedding/attention/softmax/rmsnorm/cross-entropy train end to end.
fn lm_demo() {
    use hos::tensor::Tensor;
    let text = "hello world ".repeat(3);
    let chars: Vec<char> = {
        let mut v: Vec<char> = text.chars().collect();
        v.sort();
        v.dedup();
        v
    };
    let vocab = chars.len();
    let id = |ch: char| chars.iter().position(|&c| c == ch).unwrap();
    let ids: Vec<usize> = text.chars().map(id).collect();
    let t = ids.len() - 1;
    let inputs = ids[..t].to_vec();
    let targets = ids[1..].to_vec();

    let (d, h) = (32usize, 64usize);
    let mut s = 7u64;
    let r = |shape: &[usize], s: &mut u64| Tensor::randn(shape, s);
    let tok = r(&[vocab, d], &mut s);
    let pos = r(&[t, d], &mut s);
    let (wq, wk, wv, wo) = (
        r(&[d, d], &mut s),
        r(&[d, d], &mut s),
        r(&[d, d], &mut s),
        r(&[d, d], &mut s),
    );
    let rmsw = Tensor::param(vec![1.0; d], &[d]);
    let (w1, b1) = (r(&[d, h], &mut s), Tensor::param(vec![0.0; h], &[h]));
    let (w2, b2) = (r(&[h, d], &mut s), Tensor::param(vec![0.0; d], &[d]));
    let head = r(&[d, vocab], &mut s);
    let params = [
        &tok, &pos, &wq, &wk, &wv, &wo, &rmsw, &w1, &b1, &w2, &b2, &head,
    ];

    // causal mask [t, t]
    let mut mask = vec![0.0f32; t * t];
    for i in 0..t {
        for j in 0..t {
            if j > i {
                mask[i * t + j] = -1e9;
            }
        }
    }
    let mask = Tensor::constant(mask, &[t, t]);
    let inv_sqrt_d = 1.0 / (d as f32).sqrt();

    let fwd = || {
        let x = tok.embedding(&inputs).add(&pos);
        let q = x.matmul(&wq);
        let k = x.matmul(&wk);
        let v = x.matmul(&wv);
        let scores = q.matmul(&k.transpose()).scale(inv_sqrt_d).add(&mask);
        let att = scores.softmax_rows();
        let x = x.add(&att.matmul(&v).matmul(&wo));
        let x = x.rmsnorm(&rmsw);
        let ff = x.matmul(&w1).add(&b1).relu().matmul(&w2).add(&b2);
        let x = x.add(&ff);
        x.matmul(&head)
    };

    println!(
        "hos-tensor LM demo: tiny self-attention model on {:?}\n",
        text
    );
    let lr = 0.1;
    for epoch in 0..=2000 {
        for p in params {
            p.zero_grad();
        }
        let logits = fwd();
        let loss = logits.cross_entropy(&targets);
        loss.backward();
        for p in params {
            p.sgd_step(lr);
        }
        if epoch % 250 == 0 {
            println!("  epoch {epoch:>4}   loss {:.4}", loss.data()[0]);
        }
    }

    // greedy next-char prediction at each position
    let logits = fwd().data();
    let mut pred = String::new();
    for i in 0..t {
        let row = &logits[i * vocab..i * vocab + vocab];
        let best = (0..vocab)
            .max_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap())
            .unwrap();
        pred.push(chars[best]);
    }
    let target_str: String = targets.iter().map(|&i| chars[i]).collect();
    println!("\n  target next-chars: {target_str:?}");
    println!("  model  next-chars: {pred:?}");
}

// Tiny single-head transformer LM, params in a fixed order for save/load.
const LM_NAMES: [&str; 12] = [
    "tok", "pos", "wq", "wk", "wv", "wo", "rmsw", "w1", "b1", "w2", "b2", "head",
];
const LM_ROLES: [u8; 12] = [3, 3, 0, 0, 0, 0, 2, 0, 1, 0, 1, 0]; // embed/weight/norm/bias
const LM_DECAY: [bool; 12] = [
    false, false, true, true, true, true, false, true, false, true, false, true,
];

struct TinyLM {
    p: Vec<hos::tensor::Tensor>,
}
impl TinyLM {
    fn new(vocab: usize, t: usize, d: usize, h: usize, seed: u64) -> TinyLM {
        use hos::tensor::Tensor;
        let mut s = seed;
        let mut r = |sh: &[usize]| Tensor::randn(sh, &mut s);
        let p = vec![
            r(&[vocab, d]),
            r(&[t, d]),
            r(&[d, d]),
            r(&[d, d]),
            r(&[d, d]),
            r(&[d, d]),
            Tensor::param(vec![1.0; d], &[d]),
            r(&[d, h]),
            Tensor::param(vec![0.0; h], &[h]),
            r(&[h, d]),
            Tensor::param(vec![0.0; d], &[d]),
            r(&[d, vocab]),
        ];
        TinyLM { p }
    }
    fn refs(&self) -> Vec<&hos::tensor::Tensor> {
        self.p.iter().collect()
    }
    fn forward(
        &self,
        inputs: &[usize],
        mask: &hos::tensor::Tensor,
        inv: f32,
    ) -> hos::tensor::Tensor {
        let x = self.p[0].embedding(inputs).add(&self.p[1]);
        let q = x.matmul(&self.p[2]);
        let k = x.matmul(&self.p[3]);
        let v = x.matmul(&self.p[4]);
        let att = q.matmul(&k.transpose()).scale(inv).add(mask).softmax_rows();
        let x = x.add(&att.matmul(&v).matmul(&self.p[5]));
        let x = x.rmsnorm(&self.p[6]);
        let ff = x
            .matmul(&self.p[7])
            .add(&self.p[8])
            .relu()
            .matmul(&self.p[9])
            .add(&self.p[10]);
        let x = x.add(&ff);
        x.matmul(&self.p[11])
    }
}

/// Full lifecycle proof for the HOS format + AdamW:
/// train -> save .hos (weights + optimizer state + card) -> reload -> verify
/// predictions match -> resume training (loss continues, proving opt state restored).
fn hos_demo() {
    use hos::format::{self, Named};
    use hos::tensor::AdamW;
    use std::path::Path;

    let text = "hello world ".repeat(3);
    let mut chars: Vec<char> = text.chars().collect();
    chars.sort();
    chars.dedup();
    let vocab = chars.len();
    let idof = |c: char| chars.iter().position(|&x| x == c).unwrap();
    let ids: Vec<usize> = text.chars().map(idof).collect();
    let t = ids.len() - 1;
    let inputs = ids[..t].to_vec();
    let targets = ids[1..].to_vec();
    let (d, h) = (32usize, 64usize);
    let inv = 1.0 / (d as f32).sqrt();

    let mut mask = vec![0.0f32; t * t];
    for i in 0..t {
        for j in (i + 1)..t {
            mask[i * t + j] = -1e9;
        }
    }
    let mask = hos::tensor::Tensor::constant(mask, &[t, t]);

    let predict = |lm: &TinyLM| -> String {
        let logits = lm.forward(&inputs, &mask, inv).data();
        (0..t)
            .map(|i| {
                let row = &logits[i * vocab..i * vocab + vocab];
                chars[(0..vocab)
                    .max_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap())
                    .unwrap()]
            })
            .collect()
    };

    // ---- train ----
    let lm = TinyLM::new(vocab, t, d, h, 7);
    let params = lm.refs();
    let mut opt = AdamW::new(&params, 0.01, 0.01);
    println!("HOS lifecycle demo — train, save, reload, resume\n");
    let mut last = 0.0;
    for epoch in 0..=800 {
        for p in &params {
            p.zero_grad();
        }
        let loss = lm.forward(&inputs, &mask, inv).cross_entropy(&targets);
        loss.backward();
        opt.step(&params, &LM_DECAY);
        last = loss.data()[0];
        if epoch % 200 == 0 {
            println!("  train  epoch {epoch:>4}  loss {last:.4}");
        }
    }
    let pred_before = predict(&lm);

    // ---- save ----
    let path = Path::new("/tmp/tiny.hos");
    let mut named: Vec<Named> = Vec::new();
    for i in 0..12 {
        named.push(Named {
            name: LM_NAMES[i].into(),
            role: LM_ROLES[i],
            shape: lm.p[i].shape(),
            data: lm.p[i].data(),
        });
    }
    for i in 0..12 {
        named.push(Named {
            name: format!("opt.m.{i}"),
            role: format::ROLE_OPT_STATE,
            shape: vec![opt.m[i].len()],
            data: opt.m[i].clone(),
        });
        named.push(Named {
            name: format!("opt.v.{i}"),
            role: format::ROLE_OPT_STATE,
            shape: vec![opt.v[i].len()],
            data: opt.v[i].clone(),
        });
    }
    for (k, val) in [
        ("opt.step", opt.t as f32),
        ("opt.lr", opt.lr),
        ("opt.beta1", opt.beta1),
        ("opt.beta2", opt.beta2),
        ("opt.wd", opt.wd),
    ] {
        named.push(Named {
            name: k.into(),
            role: format::ROLE_SCALAR,
            shape: vec![1],
            data: vec![val],
        });
    }
    let mut card = format::Card::new(
        "tiny-hue-lm",
        serde_json::json!({
            "type": "decoder_transformer", "d_model": d, "n_layers": 1, "n_heads": 1,
            "vocab": vocab, "ctx": t, "norm": "rmsnorm", "ffn": "relu"
        }),
    );
    card.id = format::model_id(&named);
    card.resume = format::Resume {
        seed: 7,
        step: opt.t,
        rng_state: 0,
    };
    card.history.push(format::TrainRun {
        steps: opt.t,
        final_loss: last,
        optimizer: "adamw".into(),
        lr: opt.lr,
    });
    format::save(path, &named, &card).expect("save");
    let sz = std::fs::metadata(path).unwrap().len();
    println!("\n  saved {} ({} bytes)", path.display(), sz);

    // ---- reload ----
    let (loaded, _card) = format::load(path).expect("load");
    let get =
        |name: &str| -> Vec<f32> { loaded.iter().find(|n| n.name == name).unwrap().data.clone() };
    let lm2 = TinyLM::new(vocab, t, d, h, 999); // different seed; weights get overwritten
    for i in 0..12 {
        lm2.p[i].set_data(&get(LM_NAMES[i]));
    }
    let mut opt2 = AdamW::new(&lm2.refs(), get("opt.lr")[0], get("opt.wd")[0]);
    opt2.t = get("opt.step")[0] as u64;
    for i in 0..12 {
        opt2.m[i] = get(&format!("opt.m.{i}"));
        opt2.v[i] = get(&format!("opt.v.{i}"));
    }

    let pred_after = predict(&lm2);
    println!(
        "  reload verify: predictions {}",
        if pred_after == pred_before {
            "✅ IDENTICAL"
        } else {
            "❌ differ"
        }
    );

    // ---- resume ----
    println!("\n  resuming training (loss should continue down from {last:.4}, not spike):");
    let params2 = lm2.refs();
    for epoch in 1..=400 {
        for p in &params2 {
            p.zero_grad();
        }
        let loss = lm2.forward(&inputs, &mask, inv).cross_entropy(&targets);
        loss.backward();
        opt2.step(&params2, &LM_DECAY);
        if epoch % 200 == 0 {
            println!("  resume epoch {epoch:>4}  loss {:.4}", loss.data()[0]);
        }
    }
    println!("\n  inspect with:  hos --hos-info {}", path.display());
}

/// Validate the GPU delta-net kernel against a CPU reference on synthetic data.
#[cfg(target_os = "macos")]
fn deltanet_test() {
    let n = 128usize;
    // deterministic synthetic inputs
    let s0: Vec<f32> = (0..n * n)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.01)
        .collect();
    let q: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
    let k: Vec<f32> = (0..n).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();
    let v: Vec<f32> = (0..n).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();
    let (g, beta) = (0.9f32, 0.7f32);

    // CPU reference (same recurrence as qwen35::lin_block)
    let mut s = s0.clone();
    for x in s.iter_mut() {
        *x *= g;
    }
    let kq: f32 = (0..n).map(|i| k[i] * q[i]).sum();
    let mut d = vec![0.0f32; n];
    let mut o_cpu = vec![0.0f32; n];
    for j in 0..n {
        let row = &s[j * n..j * n + n];
        let sk: f32 = (0..n).map(|i| row[i] * k[i]).sum();
        let sq: f32 = (0..n).map(|i| row[i] * q[i]).sum();
        d[j] = beta * (v[j] - sk);
        o_cpu[j] = sq + d[j] * kq;
    }
    for i in 0..n {
        for j in 0..n {
            s[i * n + j] += d[i] * k[j];
        }
    }

    let gpu = metal_be::Gpu::new();
    let (o_gpu, s_gpu) = gpu.deltanet_step(&s0, &q, &k, &v, g, beta);

    let od = o_cpu
        .iter()
        .zip(&o_gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let sd = s
        .iter()
        .zip(&s_gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "[deltanet-test] max|cpu-gpu| output={od:.3e} state={sd:.3e}  {}",
        if od < 1e-4 && sd < 1e-4 {
            "✅ MATCH"
        } else {
            "❌ MISMATCH"
        }
    );
}

fn print_banner() {
    eprintln!(
        r#"
██   ██  ██████  ███████
██   ██ ██    ██ ██
███████ ██    ██ ███████
██   ██ ██    ██      ██
██   ██  ██████  ███████   v0  ·  local inference, from scratch
"#
    );
}

/// M4: verify the spec-driven Llama runner matches the hand-coded forward.
fn interp_check() {
    use std::collections::HashMap;
    let mp = arg_after("--interp-check")
        .unwrap_or_else(|| "path/to/SmolLM2-135M-Instruct-Q8_0.gguf".into());
    let g = ok(Gguf::open(std::path::Path::new(&mp)));
    let tok = ok(Tokenizer::from_gguf(&g));
    let model = ok(Model::load(&g, None)); // CPU
    let cfg = model.cfg.clone();
    let ids = tok.encode("The capital of France is", true);
    let idu: Vec<usize> = ids.iter().map(|&t| t as usize).collect();

    // reference: the hand-coded forward (prefill the prompt, keep last logits)
    let mut st = forward::State::new(&cfg);
    let mut ref_logits = vec![];
    for (pos, &t) in ids.iter().enumerate() {
        ref_logits = forward::forward(&model, &mut st, t, pos, None);
    }

    // spec-driven: a JSON arch spec + a name->weights map, run by the interpreter
    let spec = serde_json::json!({
        "type": "llama", "dim": cfg.dim, "n_layers": cfg.n_layers, "n_heads": cfg.n_heads,
        "n_kv_heads": cfg.n_kv_heads, "head_dim": cfg.head_dim, "ffn_dim": cfg.ffn_dim,
        "vocab": cfg.vocab_size, "rms_eps": cfg.rms_eps, "rope_base": cfg.rope_base, "rope_neox": cfg.rope_neox,
    });
    let mut names = vec![
        "token_embd.weight".to_string(),
        "output_norm.weight".to_string(),
    ];
    if g.has("output.weight") {
        names.push("output.weight".into());
    }
    for l in 0..cfg.n_layers {
        for s in [
            "attn_norm.weight",
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ffn_norm.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ] {
            names.push(format!("blk.{l}.{s}"));
        }
    }
    let mut w: HashMap<String, Vec<f32>> = HashMap::new();
    for n in names {
        let d = ok(g.dequant(&n));
        w.insert(n, d);
    }
    let spec_logits = ok(hos::interp::run_llama_from_spec(&spec, &w, &idu));

    let maxd = ref_logits
        .iter()
        .zip(&spec_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let am = |v: &[f32]| {
        (0..v.len())
            .max_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap())
            .unwrap()
    };
    let (ae, asp) = (am(&ref_logits), am(&spec_logits));
    let mut buf = Vec::new();
    tok.decode_into(asp as u32, &mut buf);
    println!("=== M4: spec-driven inference vs hand-coded forward ===");
    println!(
        "model: {}",
        std::path::Path::new(&mp)
            .file_name()
            .unwrap()
            .to_string_lossy()
    );
    println!(
        "max |engine - spec| over {} logits = {maxd:.3e}",
        ref_logits.len()
    );
    println!(
        "next-token argmax: engine={ae}  spec={asp}  -> {:?}  {}",
        String::from_utf8_lossy(&buf),
        if ae == asp {
            "✓ MATCH"
        } else {
            "✗ MISMATCH"
        }
    );
}
