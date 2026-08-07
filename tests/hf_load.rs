//! HuggingFace safetensors load smoke test. Runs only when an HF checkpoint is
//! present — set `HOS_HF_MODEL=<dir>`, or drop a checkpoint folder (with a
//! `config.json`) under `models/`. Skips gracefully otherwise (e.g. in CI), like
//! the GGUF harness. Asserts the load + tokenizer + scoring are well-formed, not
//! specific numbers (which depend on the model).

use std::path::PathBuf;

fn find_hf() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("HOS_HF_MODEL") {
        let p = PathBuf::from(d);
        if p.join("config.json").is_file() {
            return Some(p);
        }
    }
    let models = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models");
    std::fs::read_dir(&models)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .find(|p| p.is_dir() && p.join("config.json").is_file())
}

#[test]
fn hf_checkpoint_loads_and_scores() {
    let Some(dir) = find_hf() else {
        eprintln!("[skip] no HF checkpoint (set HOS_HF_MODEL=<dir> or put one under models/)");
        return;
    };
    let mut eng = hos::Engine::load(&dir, false).expect("load HF checkpoint");

    let ids = eng
        .tok
        .encode("The quick brown fox jumps over the lazy dog.", true);
    assert!(!ids.is_empty(), "tokenizer produced no ids");

    let (n, nll, ppl) = eng.perplexity(&ids);
    assert!(n > 0, "should score at least one prediction");
    assert!(
        ppl.is_finite() && ppl >= 1.0,
        "perplexity must be finite and >= 1, got {ppl}"
    );
    assert!(nll >= 0.0, "mean NLL must be non-negative, got {nll}");
}
