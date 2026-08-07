//! Robustness fuzzing for the GGUF parser: the engine must never *panic* on a
//! malformed, truncated, or adversarial file — it must return `Err`. We throw
//! random bytes, truncated valid prefixes, and byte-mutated real models at
//! `Gguf::open` (and `dequant`/`raw` on anything that opens), asserting under
//! `catch_unwind` that nothing aborts the process.

use std::panic::{self, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

use hos::gguf::Gguf;

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn temp_path() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!("hos_fuzz_{}_{}.gguf", std::process::id(), n));
    p
}

/// Tiny deterministic PRNG (xorshift) — no rand crate, reproducible runs.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn byte(&mut self) -> u8 {
        (self.next() >> 24) as u8
    }
}

/// Open the bytes as a GGUF and, if it parses, try to decode every tensor.
/// The whole thing runs under catch_unwind: a panic fails the test.
fn open_and_probe(bytes: &[u8]) {
    let path = temp_path();
    std::fs::write(&path, bytes).unwrap();
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        if let Ok(g) = Gguf::open(&path) {
            // Probe a handful of tensors through both decode paths (enough to
            // hit the offset/size/quant-type guards; probing all of a real model
            // would just re-dequant valid data, which the codec tests cover).
            let names: Vec<String> = g.tensors.keys().take(8).cloned().collect();
            for name in names {
                let _ = g.dequant(&name);
                let _ = g.raw(&name);
            }
        }
    }));
    let _ = std::fs::remove_file(&path);
    assert!(
        result.is_ok(),
        "GGUF parsing panicked on adversarial input (len {})",
        bytes.len()
    );
}

#[test]
fn random_bytes_never_panic() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for len in [0usize, 1, 4, 8, 24, 64, 200, 4096] {
        for _ in 0..40 {
            let buf: Vec<u8> = (0..len).map(|_| rng.byte()).collect();
            open_and_probe(&buf);
        }
    }
}

#[test]
fn gguf_magic_then_garbage_never_panics() {
    // Looks like a GGUF (right magic) but the rest is random — exercises the
    // header/metadata/tensor-info parsing paths, not just the magic check.
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    for _ in 0..200 {
        let mut buf = b"GGUF".to_vec();
        let len = (rng.next() % 512) as usize;
        buf.extend((0..len).map(|_| rng.byte()));
        open_and_probe(&buf);
    }
}

#[test]
fn truncated_and_mutated_real_models_never_panic() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        eprintln!("[skip] no models/ dir");
        return;
    };
    let model = entries
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().and_then(|s| s.to_str()) == Some("gguf"));
    let Some(model) = model else {
        eprintln!("[skip] no .gguf model to mutate");
        return;
    };
    let full = std::fs::read(&model).unwrap();
    let mut rng = Rng(0xCAFE_F00D_BAD0_BEEF);

    // The header + tensor-info region is where parsing logic lives; truncating
    // and mutating within it exercises every length/offset/type guard. (Cutting
    // deep into the multi-hundred-MB data section only yields out-of-bounds
    // tensors — cheap Errs — and re-dequanting valid data is the codec tests'
    // job, so we keep all inputs small and the suite fast.)
    let region = full.len().min(128 * 1024);

    // Truncations at many offsets within the header/info region.
    for _ in 0..40 {
        let cut = (rng.next() as usize) % region;
        open_and_probe(&full[..cut]);
    }
    // Single/double byte flips across the same region.
    for _ in 0..80 {
        let mut buf = full[..region].to_vec();
        let flips = 1 + (rng.next() % 2) as usize;
        for _ in 0..flips {
            let i = (rng.next() as usize) % buf.len();
            buf[i] ^= rng.byte();
        }
        open_and_probe(&buf);
    }
}
