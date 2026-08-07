//! Benchmark + perplexity harness tests. Run against a real model file when one
//! is present in models/ (skipped gracefully otherwise) — they assert the
//! metrics are well-formed and reproducible, not specific numbers (which depend
//! on the model).

use std::path::PathBuf;

fn find_model() -> Option<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models");
    let preferred = dir.join("SmolLM2-135M-Instruct-Q8_0.gguf");
    if preferred.is_file() {
        return Some(preferred);
    }
    std::fs::read_dir(&dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().and_then(|s| s.to_str()) == Some("gguf"))
}

#[test]
fn perplexity_is_finite_and_deterministic() {
    let Some(path) = find_model() else {
        eprintln!("[skip] no model in models/ for perplexity test");
        return;
    };
    let mut eng = hos::Engine::load(&path, false).unwrap();
    let ids = eng
        .tok
        .encode("The quick brown fox jumps over the lazy dog.", true);

    let (n1, nll1, ppl1) = eng.perplexity(&ids);
    let (n2, nll2, ppl2) = eng.perplexity(&ids);

    assert!(n1 > 0, "should score at least one prediction");
    assert_eq!(n1, n2, "scored-token count must be stable");
    assert!(
        ppl1.is_finite() && ppl1 >= 1.0,
        "perplexity must be finite and >= 1, got {ppl1}"
    );
    assert!(nll1 >= 0.0, "mean NLL must be non-negative, got {nll1}");
    // perplexity resets its own KV cache, so repeated calls are identical
    assert!(
        (nll1 - nll2).abs() < 1e-9,
        "NLL not deterministic: {nll1} vs {nll2}"
    );
    assert!(
        (ppl1 - ppl2).abs() < 1e-6,
        "perplexity not deterministic: {ppl1} vs {ppl2}"
    );
}

#[test]
fn bench_reports_positive_throughput() {
    let Some(path) = find_model() else {
        eprintln!("[skip] no model in models/ for bench test");
        return;
    };
    let mut eng = hos::Engine::load(&path, false).unwrap();
    let b = eng.bench("The history of computing began with", 8);

    assert!(b.prefill_tokens > 0, "prefill should ingest the prompt");
    assert!(b.decode_tokens > 0, "should decode at least one token");
    assert!(b.prefill_tps() > 0.0 && b.prefill_tps().is_finite());
    assert!(b.decode_tps() > 0.0 && b.decode_tps().is_finite());
}
