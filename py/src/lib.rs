//! Python bindings for HOS (via PyO3, abi3). Exposes the high-level engine —
//! load a GGUF, generate text, score perplexity, benchmark — plus `.hos` card
//! inspection. Built as a separate cdylib crate so the main engine build is
//! untouched. Build: `maturin build --release` from this directory.

use std::path::Path;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// A loaded model + tokenizer + backend.
#[pyclass]
struct Engine {
    inner: hos_engine::Engine,
}

#[pymethods]
impl Engine {
    /// Engine(path, gpu=True) — load a GGUF model.
    #[new]
    #[pyo3(signature = (path, gpu=true))]
    fn new(path: &str, gpu: bool) -> PyResult<Self> {
        hos_engine::Engine::load(Path::new(path), gpu)
            .map(|inner| Engine { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Generate text from a prompt and return it as a string.
    #[pyo3(signature = (prompt, max_tokens=128, temp=0.0, top_k=40, top_p=0.95,
                        rep_penalty=1.1, repeat_last_n=64, seed=42))]
    #[allow(clippy::too_many_arguments)]
    fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temp: f32,
        top_k: usize,
        top_p: f32,
        rep_penalty: f32,
        repeat_last_n: usize,
        seed: u64,
    ) -> String {
        let mut out = String::new();
        self.inner.generate(
            prompt, max_tokens, temp, top_k, top_p, rep_penalty, repeat_last_n, seed,
            |piece| out.push_str(piece),
        );
        out
    }

    /// Text embedding: final hidden state (post-output-norm, pre-lm-head),
    /// pooled over the prompt. pooling="mean" averages every token's final
    /// hidden (the untrained-decoder-as-embedder default); "last" returns the
    /// last token's vector. Runs the CPU representation tap on a fresh KV
    /// cache, so generation state is untouched.
    #[pyo3(signature = (text, pooling="mean"))]
    fn embed(&mut self, text: &str, pooling: &str) -> PyResult<Vec<f32>> {
        let ids = self.inner.tok.encode(text, true);
        match pooling {
            "mean" => Ok(self.inner.mean_hidden(&ids)),
            "last" => Ok(self.inner.last_hidden(&ids)),
            _ => Err(PyValueError::new_err("pooling must be 'mean' or 'last'")),
        }
    }

    /// Encode `text` and return (scored_tokens, mean_nll_nats, perplexity).
    fn perplexity(&mut self, text: &str) -> (usize, f64, f64) {
        let ids = self.inner.tok.encode(text, true);
        self.inner.perplexity(&ids)
    }

    /// Time prefill + greedy decode. Returns
    /// (prefill_tokens, prefill_secs, prefill_tps, decode_tokens, decode_secs, decode_tps).
    #[pyo3(signature = (prompt, decode_tokens=64))]
    fn bench(&mut self, prompt: &str, decode_tokens: usize) -> (usize, f64, f64, usize, f64, f64) {
        let b = self.inner.bench(prompt, decode_tokens);
        (
            b.prefill_tokens, b.prefill_secs, b.prefill_tps(),
            b.decode_tokens, b.decode_secs, b.decode_tps(),
        )
    }

    /// Token ids for `text` (with BOS).
    fn tokenize(&self, text: &str) -> Vec<u32> {
        self.inner.tok.encode(text, true)
    }

    /// Vocabulary size.
    fn vocab_size(&self) -> usize {
        self.inner.tok.vocab_size()
    }
}

/// Inspect a `.hos` capsule's card: (name, id, mode, arch_json, n_tensors, n_values).
#[pyfunction]
fn inspect_hos(path: &str) -> PyResult<(String, String, String, String, usize, usize)> {
    let (tensors, card) =
        hos_engine::format::load(Path::new(path)).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let values: usize = tensors.iter().map(|t| t.data.len()).sum();
    Ok((card.name, card.id, card.mode, card.arch.to_string(), tensors.len(), values))
}

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn hos(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    m.add_function(wrap_pyfunction!(inspect_hos, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
