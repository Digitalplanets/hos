//! Golden token-stream regression net (safety net, part 1).
//!
//! For a fixed prompt + greedy (temp 0) decode, the engine must reproduce a
//! committed token-id stream byte-for-byte. This is an end-to-end pin: tokenizer
//! -> prefill -> per-step forward -> argmax sampling all have to stay numerically
//! stable, or an id flips and the test fails. It is deliberately about *ids*, not
//! decoded text, so a tokenizer-rendering change can't mask a model-numeric drift.
//!
//! One of the goldens is a **flwr** model. flwr snaps the final post-output-norm
//! hidden onto the E8 lattice (`forward::flwr_e8_quant` / `nn::nearest_e8`) right
//! before the lm-head, which turns sub-ULP floating-point drift into a *visible*
//! token flip. So this fixture is the engine's floating-point canary: if a CPU
//! kernel reorder (or, on macOS, a GPU path that's meant to match the CPU path)
//! perturbs the hidden enough to cross a lattice cell boundary, the stream
//! diverges and we catch it here.
//!
//! Models live outside the repo (multi-GB, machine-local), so each case skips
//! gracefully when its model file isn't found — exactly like `harness.rs`. The
//! fixtures themselves ARE committed. Set `HOS_GOLDEN_RECORD=1` to (re)generate a
//! fixture instead of asserting against it.

use std::path::{Path, PathBuf};

/// Locate `models/<name>` by walking up from the crate dir. Works both in a
/// normal checkout (models/ next to Cargo.toml) and in a git worktree, where the
/// real `models/` lives a few ancestors up at the shared repo root.
fn find_model(name: &str) -> Option<PathBuf> {
    let start = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for anc in start.ancestors() {
        let cand = anc.join("models").join(name);
        if cand.is_file() {
            return Some(cand);
        }
    }
    None
}

fn fixture_path(key: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(format!("golden_{key}.txt"))
}

fn read_fixture(key: &str) -> Option<Vec<u32>> {
    let txt = std::fs::read_to_string(fixture_path(key)).ok()?;
    let ids = txt
        .lines()
        .filter(|l| !l.trim_start().starts_with('#') && !l.trim().is_empty())
        .flat_map(|l| {
            l.split([',', ' ', '\t'])
                .filter(|s| !s.is_empty())
                .map(|s| s.parse::<u32>())
        })
        .collect::<Result<Vec<u32>, _>>()
        .ok()?;
    Some(ids)
}

fn write_fixture(key: &str, model: &str, prompt: &str, n: usize, seed: u64, ids: &[u32]) {
    let mut s = String::new();
    s.push_str(&format!(
        "# golden token-id stream for the HOS regression safety net\n"
    ));
    s.push_str(&format!(
        "# model={model} prompt={prompt:?} max_tokens={n} temp=0 seed={seed}\n"
    ));
    s.push_str(&format!("# {} ids, comma-separated:\n", ids.len()));
    let joined = ids
        .iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(",");
    s.push_str(&joined);
    s.push('\n');
    std::fs::write(fixture_path(key), s).expect("write fixture");
}

/// Run one golden case: load the model (skip if absent), greedily generate, and
/// either assert against or record the committed id stream.
fn run_case(key: &str, model_file: &str, prompt: &str, n: usize, seed: u64) {
    let Some(path) = find_model(model_file) else {
        eprintln!(
            "[skip] golden '{key}': model '{model_file}' not found in any ancestor models/ dir"
        );
        return;
    };
    let mut eng = hos::Engine::load(&path, false).expect("load model");
    let ids = eng.generate_ids(prompt, n, 0.0, seed);
    assert!(!ids.is_empty(), "golden '{key}': engine produced no tokens");

    let record = std::env::var("HOS_GOLDEN_RECORD").is_ok();
    match read_fixture(key) {
        Some(expected) if !record => {
            assert_eq!(
                ids, expected,
                "golden '{key}' DRIFTED.\n  model={model_file}\n  prompt={prompt:?}\n  got     ={ids:?}\n  expected={expected:?}\n\
                 (flwr goldens are the fp canary: a crossed E8 lattice cell flips a token. \
                 If this is an intended change, re-record with HOS_GOLDEN_RECORD=1.)"
            );
        }
        _ => {
            write_fixture(key, model_file, prompt, n, seed, &ids);
            eprintln!(
                "[record] golden '{key}': wrote {} ids -> {:?}",
                ids.len(),
                fixture_path(key)
            );
        }
    }
}

/// Determinism guard: generate_ids on a fresh engine must be repeatable. Cheap
/// (uses the smallest model present) and independent of the committed fixtures.
fn smallest_available() -> Option<(&'static str, PathBuf)> {
    for name in [
        "SmolLM2-135M-Instruct-Q8_0.gguf",
        "smol135q8.hos",
        "llama1b.q4k.hos",
    ] {
        if let Some(p) = find_model(name) {
            return Some((name, p));
        }
    }
    None
}

#[test]
fn golden_smollm_q8_greedy_stream() {
    // standard Llama-family golden: small, fast, exercises the GGUF + Q8_0 path.
    run_case(
        "smollm135_q8",
        "SmolLM2-135M-Instruct-Q8_0.gguf",
        "The history of computing began with",
        16,
        42,
    );
}

#[test]
fn golden_flwr_1b_greedy_stream_fp_canary() {
    // flwr fp canary: terminal E8 lattice snap makes this stream fp-sensitive.
    run_case(
        "flwr_1b",
        "flwr_1b_run.hos",
        "The capital of France is",
        16,
        42,
    );
}

/// flwr GPU-vs-CPU token parity (the fp-canary invariant, regression-protected).
/// flwr's terminal E8 snap is fp-sensitive, so the Metal head must reproduce the
/// CPU snap exactly or tokens diverge. Loads the same model on CPU and GPU and
/// asserts identical greedy ids. macOS-only (the GPU path is Metal); skips if the
/// model isn't present, like the goldens.
#[cfg(target_os = "macos")]
fn flwr_gpu_cpu_parity(model_file: &str) {
    let Some(path) = find_model(model_file) else {
        eprintln!("[skip] flwr GPU/CPU parity: model '{model_file}' not found");
        return;
    };
    let (prompt, n, seed) = (
        "Explain who Black Americans are, historically.",
        32usize,
        42u64,
    );
    let cpu = hos::Engine::load(&path, false)
        .expect("load cpu")
        .generate_ids(prompt, n, 0.0, seed);
    let gpu = hos::Engine::load(&path, true)
        .expect("load gpu")
        .generate_ids(prompt, n, 0.0, seed);
    assert_eq!(
        cpu, gpu,
        "flwr GPU tokens diverged from CPU for {model_file} — the Metal head's E8 snap \
         must match the CPU path bit-for-bit.\n  cpu={cpu:?}\n  gpu={gpu:?}"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn flwr_gpu_matches_cpu_f32() {
    // f32 capsule -> per-token GPU head snap path.
    flwr_gpu_cpu_parity("flwr_1b_run.hos");
}

#[cfg(target_os = "macos")]
#[test]
fn flwr_gpu_matches_cpu_q4k() {
    // Q4_K capsule -> exercises the BATCHED-prefill snap branch (can_batch == true).
    flwr_gpu_cpu_parity("flwr_1b_q4k.hos");
}

#[test]
fn generate_ids_is_deterministic() {
    let Some((name, path)) = smallest_available() else {
        eprintln!("[skip] determinism: no small model present");
        return;
    };
    let mut e1 = hos::Engine::load(&path, false).expect("load");
    let a = e1.generate_ids("Once upon a time", 12, 0.0, 7);
    let mut e2 = hos::Engine::load(&path, false).expect("load");
    let b = e2.generate_ids("Once upon a time", 12, 0.0, 7);
    assert_eq!(
        a, b,
        "greedy generate_ids must be deterministic (model={name})"
    );
    assert!(!a.is_empty());
}
