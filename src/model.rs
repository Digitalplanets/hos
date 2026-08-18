//! Llama-family model: config + weights loaded from GGUF.
//!
//! Covers the Llama architecture family (Llama, Mistral, SmolLM2, and Qwen-ish
//! variants that share the same block structure). Weights are dequantized to f32
//! at load time — fine for small models; v1 keeps them quantized for big ones.

use crate::error::{HosError, Result};
use crate::gguf::Gguf;
use crate::metal_be::{Gpu, GpuMatrix};

/// Known architectures. Drives config + capability selection per model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Arch {
    Llama,
    Qwen2,
    Mistral,
    Gemma,
    Phi3,
    OlMoe,
    Qwen35Hybrid,
    /// flwr — Llama-family trunk with an E8 vector-quantized hidden bottleneck
    /// before the lm-head. Runs on the CPU forward (which applies the lattice quant).
    Flwr,
    Other,
}

impl Arch {
    pub fn detect<S: ModelSource>(g: &S) -> Arch {
        // hybrid SSM is identified structurally, not just by name
        if g.has("blk.0.ssm_a") {
            return Arch::Qwen35Hybrid;
        }
        match g.meta_str("general.architecture").unwrap_or("llama") {
            "llama" => Arch::Llama,
            "qwen2" | "qwen2moe" => Arch::Qwen2,
            "mistral" => Arch::Mistral,
            "gemma" | "gemma2" => Arch::Gemma,
            "phi3" => Arch::Phi3,
            "olmoe" => Arch::OlMoe,
            "qwen35" => Arch::Qwen35Hybrid,
            "flwr" => Arch::Flwr,
            _ => Arch::Other,
        }
    }

    /// Standard transformer families HOS runs correctly today.
    pub fn is_transformer(&self) -> bool {
        matches!(
            self,
            Arch::Llama
                | Arch::Qwen2
                | Arch::Mistral
                | Arch::Gemma
                | Arch::Phi3
                | Arch::OlMoe
                | Arch::Flwr
        )
    }

    /// Architectures with a fused/optimized GPU runner. Gemma2/Phi3 run on the
    /// CPU path (correctness-first; their structure differs from the Llama runner).
    /// Flwr is a pure Llama-family trunk; its ONLY divergence is a terminal E8
    /// snap on the last token's post-output-norm hidden, which the GPU head
    /// replays on the CPU (see `needs_terminal_hidden_snap`).
    pub fn gpu_supported(&self) -> bool {
        matches!(self, Arch::Llama | Arch::Qwen2 | Arch::Mistral | Arch::Flwr)
    }

    /// True iff this arch quantizes the final post-output-norm hidden onto a
    /// lattice before the lm-head (flwr's E8 bottleneck). The GPU head MUST
    /// reproduce this CPU-side snap bit-for-bit, or its tokens will diverge from
    /// the CPU path; `false` for every standard transformer. A new arch that adds
    /// such a bottleneck must flip this AND be wired into the GPU head — a
    /// `debug_assert` in metal_be guards against silently skipping it.
    pub fn needs_terminal_hidden_snap(&self) -> bool {
        matches!(self, Arch::Flwr)
    }

    /// NEOX-style RoPE (rotate halves) vs Llama's interleaved-pair RoPE.
    pub fn rope_neox(&self) -> bool {
        matches!(self, Arch::Qwen2 | Arch::Gemma | Arch::Phi3 | Arch::OlMoe)
    }
}

/// Preflight: true iff every linear weight the GPU runner would upload has a
/// Metal matvec kernel, and the model is not MoE. HQ4 (our native NF4 format) now
/// has a GPU kernel (`matvec_hq4_co`) and is accepted here; any other non-standard
/// quant still has no GPU kernel, and the fused runner has no MoE expert branch —
/// so such models must take the (correct) CPU forward instead of hard-exiting in
/// `enc_matvec`. Scans every layer, since a mixed-quant capsule could carry a
/// single offending tensor.
pub fn gpu_quant_supported<S: ModelSource>(g: &S) -> bool {
    use crate::gguf::{
        GGML_F16, GGML_F32, GGML_Q4_0, GGML_Q4_K, GGML_Q5_0, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0,
    };
    if g.has("blk.0.ffn_gate_exps.weight") {
        return false; // MoE: no GPU expert dispatch in the fused runner
    }
    let kernel_ok = |ty: u32| {
        // matvec_hq4_co exists, but the .hos loader currently materializes HQ4 to f32
        // before it reaches the GPU, so HQ4 never arrives as HQ4_TYPE. Keep HQ4 on the
        // CPU path until the resident loader is wired — then re-enable HQ4_TYPE here and
        // add a flwr+HQ4 CPU-vs-GPU token-parity test (see docs/briefs).
        matches!(
            ty,
            GGML_F32
                | GGML_F16
                | GGML_Q4_0
                | GGML_Q5_0
                | GGML_Q8_0
                | GGML_Q4_K
                | GGML_Q5_K
                | GGML_Q6_K
        )
    };
    let names = [
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_output",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
    ];
    let mut i = 0usize;
    while g.has(&format!("blk.{i}.attn_q.weight")) {
        for nm in names {
            let key = format!("blk.{i}.{nm}.weight");
            if g.has(&key) {
                if let Ok((_, ty, _)) = g.raw(&key) {
                    if !kernel_ok(ty) {
                        return false;
                    }
                }
            }
        }
        i += 1;
    }
    true
}

/// A source of model weights + metadata the loader can read, abstracted over the
/// on-disk container. `Gguf` is one implementation; `hf::HfModel` is another that
/// presents a raw HuggingFace `config.json` + `*.safetensors` checkpoint through
/// the *same* GGUF-style keys and tensor names — so `Model::load` runs every
/// architecture unchanged regardless of where the weights came from.
pub trait ModelSource {
    fn meta_str(&self, key: &str) -> Option<&str>;
    fn meta_u64(&self, key: &str) -> Option<u64>;
    fn meta_f32(&self, key: &str) -> Option<f32>;
    fn has(&self, name: &str) -> bool;
    /// Tensor → flat f32 (row-major).
    fn dequant(&self, name: &str) -> Result<Vec<f32>>;
    /// Raw (still-encoded) bytes + ggml type + element count, for GPU upload.
    fn raw(&self, name: &str) -> Result<(&[u8], u32, usize)>;
}

impl ModelSource for Gguf {
    fn meta_str(&self, key: &str) -> Option<&str> {
        Gguf::meta_str(self, key)
    }
    fn meta_u64(&self, key: &str) -> Option<u64> {
        Gguf::meta_u64(self, key)
    }
    fn meta_f32(&self, key: &str) -> Option<f32> {
        Gguf::meta_f32(self, key)
    }
    fn has(&self, name: &str) -> bool {
        Gguf::has(self, name)
    }
    fn dequant(&self, name: &str) -> Result<Vec<f32>> {
        Gguf::dequant(self, name)
    }
    fn raw(&self, name: &str) -> Result<(&[u8], u32, usize)> {
        Gguf::raw(self, name)
    }
}

/// A linear weight matrix [rows(out), cols(in)] that lives on CPU or GPU.
/// `matvec` computes y = W·x on whichever backend holds the weights.
pub enum Weight {
    Cpu {
        data: Vec<f32>,
        rows: usize,
        cols: usize,
    },
    /// Weight kept in its native quantized bytes (GGUF block types, or the native
    /// `hq4`). `matvec` dequantizes one row at a time into thread-local scratch and
    /// dots it — so we read the compressed bytes (≈4-bit) instead of a 4-8× larger
    /// f32 expansion, the dominant cost in decode. `ggml_type == HQ4_TYPE` selects
    /// the native HOS decoder.
    Quant {
        bytes: Vec<u8>,
        ggml_type: u32,
        rows: usize,
        cols: usize,
    },
    Gpu(GpuMatrix),
}

/// Sentinel `ggml_type` for the native HOS `hq4` quant (not a real GGUF type;
/// chosen well above the GGUF type range so it can never collide).
pub const HQ4_TYPE: u32 = 0xF000_00A4;

/// Sentinel `ggml_type` for the native HOS `e8` E8-lattice quant (not a real GGUF
/// type; chosen well above the GGUF type range so it can never collide).
pub const E8_TYPE: u32 = 0xF000_00E8;

impl Weight {
    pub fn cpu(data: Vec<f32>, cols: usize) -> Weight {
        let rows = data.len() / cols;
        Weight::Cpu { data, rows, cols }
    }

    /// Build a CPU weight from a source tensor: keep it quantized (and fuse the
    /// dequant into matvec) when it's a fusable block type with a block-aligned
    /// row; otherwise fall back to a plain f32 matrix.
    pub fn cpu_from_source<S: ModelSource>(g: &S, name: &str, cols: usize) -> Result<Weight> {
        // Quantized-resident is a MEMORY mode (≈4-bit bytes vs a 4-8× f32 blow-up),
        // the only way to fit a large model in RAM. It is currently SLOWER than the
        // f32 path because the per-row dequant isn't SIMD-fused, so it's opt-in via
        // HOS_QRESIDENT=1. The f32 path (default) is faster on models that fit.
        let resident = std::env::var("HOS_QRESIDENT").is_ok_and(|v| v != "0" && !v.is_empty());
        let (bytes, ty, n) = g.raw(name)?;
        let fusable = ty == HQ4_TYPE
            || ty == E8_TYPE
            || crate::gguf::block_elems(ty).is_some_and(|b| cols % b == 0);
        if resident && fusable && n % cols == 0 {
            Ok(Weight::Quant {
                bytes: bytes.to_vec(),
                ggml_type: ty,
                rows: n / cols,
                cols,
            })
        } else {
            Ok(Weight::cpu(g.dequant(name)?, cols))
        }
    }

    pub fn cpu_data(&self) -> &[f32] {
        match self {
            Weight::Cpu { data, .. } => data,
            Weight::Quant { .. } => panic!("cpu_data on quantized weight — use to_f32()"),
            Weight::Gpu(_) => panic!("cpu_data on GPU weight"),
        }
    }

    /// Owned f32 view of the weight (clones `Cpu`, dequantizes `Quant`) — for the
    /// training paths (PEFT/finetune) that build f32 autograd constants from the
    /// frozen base.
    pub fn to_f32(&self) -> Vec<f32> {
        match self {
            Weight::Cpu { data, .. } => data.clone(),
            Weight::Quant {
                bytes,
                ggml_type,
                rows,
                cols,
            } => {
                let n = rows * cols;
                let mut out = vec![0.0f32; n];
                if *ggml_type == HQ4_TYPE {
                    crate::hos_quant::decode_hq4_into(bytes, n, &mut out);
                } else if *ggml_type == E8_TYPE {
                    crate::hos_quant::decode_e8_into(bytes, n, &mut out);
                } else {
                    let _ = crate::gguf::Gguf::dequant_into(bytes, *ggml_type, n, &mut out);
                }
                out
            }
            Weight::Gpu(_) => panic!("to_f32 on GPU weight"),
        }
    }

    pub fn as_gpu(&self) -> &GpuMatrix {
        match self {
            Weight::Gpu(m) => m,
            _ => panic!("as_gpu on non-GPU weight"),
        }
    }

    /// Resident byte footprint of this weight (f32 data, quantized bytes, or the
    /// GPU-side buffer size). Used by the Gemma4 bench to report memory.
    pub fn nbytes(&self) -> usize {
        match self {
            Weight::Cpu { data, .. } => data.len() * 4,
            Weight::Quant { bytes, .. } => bytes.len(),
            Weight::Gpu(m) => {
                if m.ggml_type == crate::gguf::GGML_F16 {
                    m.n_rows * m.in_dim * 2
                } else {
                    crate::gguf::bytes_for(m.ggml_type, m.n_rows * m.in_dim).unwrap_or(0)
                }
            }
        }
    }

    /// y = W · x  (out.len() == rows, x.len() == cols)
    pub fn matvec(&self, gpu: Option<&Gpu>, x: &[f32], out: &mut [f32]) {
        match self {
            Weight::Cpu { data, .. } => cpu_matmul(out, data, x),
            Weight::Quant {
                bytes,
                ggml_type,
                cols,
                ..
            } => quant_matvec(bytes, *ggml_type, *cols, x, out),
            Weight::Gpu(m) => {
                gpu.expect("gpu weight needs a Gpu context")
                    .matvec_into(m, x, out);
            }
        }
    }

    pub fn rows(&self) -> usize {
        match self {
            Weight::Cpu { rows, .. } | Weight::Quant { rows, .. } => *rows,
            Weight::Gpu(m) => m.n_rows,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Weight::Cpu { cols, .. } | Weight::Quant { cols, .. } => *cols,
            Weight::Gpu(m) => m.in_dim,
        }
    }

    /// Batched prefill matvec: `Y[ntok,rows] = X[ntok,cols] @ Wᵀ`, reading each
    /// weight ONCE across all tokens when eligible (q4_k GPU weight → batched
    /// kernel). Every other type/quant (q6_k, f16, CPU, native quants) falls back
    /// to a per-token `matvec` loop — correct, just not batched. Purely additive:
    /// no existing path changes, and any model/quant still works.
    pub fn matvec_batch(&self, gpu: Option<&Gpu>, x: &[f32], ntok: usize, out: &mut [f32]) {
        // Batched GPU paths, by token count:
        //  - 2..=8 tokens (e.g. MTP's 2-token verify): register-tiled kernel that
        //    reads the quantized weight ONCE, no f32 scratch — the small-batch win.
        //  - >=32 tokens (prefill): dequant-to-f32-scratch kernel, amortized.
        // Between/at 1: per-token matvec (fallthrough).
        if let Weight::Gpu(m) = self {
            let g = gpu.expect("gpu weight needs a Gpu context");
            if ntok == 2 {
                // efficient coalesced 2-token path (MTP verify)
                if m.ggml_type == crate::gguf::GGML_Q4_K {
                    g.matmul_q4k_2tok_into(m, x, out);
                    return;
                }
                if m.ggml_type == crate::gguf::GGML_Q6_K {
                    g.matmul_q6k_2tok_into(m, x, out);
                    return;
                }
            }
            if (2..=8).contains(&ntok) {
                if m.ggml_type == crate::gguf::GGML_Q4_K {
                    g.matmul_q4k_small_into(m, x, ntok, out);
                    return;
                }
                if m.ggml_type == crate::gguf::GGML_Q6_K {
                    g.matmul_q6k_small_into(m, x, ntok, out);
                    return;
                }
            }
            if ntok >= 32 {
                if m.ggml_type == crate::gguf::GGML_Q4_K {
                    g.matmul_q4k_prefill_into(m, x, ntok, out);
                    return;
                }
                if m.ggml_type == crate::gguf::GGML_Q6_K {
                    g.matmul_q6k_prefill_into(m, x, ntok, out);
                    return;
                }
            }
        }
        // CPU (or a GPU weight with no batched kernel for this ntok): a quantized/f32
        // weight batches through `cpu_matmat`, which dequantizes each row ONCE and
        // dots it against all `ntok` tokens — the prompt-processing win on x86/Windows
        // (per-token `matvec` would re-dequantize the whole weight for every token).
        // Bit-identical to the per-token loop (same dequant + same `dot_f32`). A GPU
        // weight reaching here (no kernel for this ntok/type) still goes per-token.
        match self {
            Weight::Gpu(_) => {
                let (cols, rows) = (self.cols(), self.rows());
                for t in 0..ntok {
                    self.matvec(
                        gpu,
                        &x[t * cols..t * cols + cols],
                        &mut out[t * rows..(t + 1) * rows],
                    );
                }
            }
            _ => cpu_matmat(out, self, x, ntok),
        }
    }
}

/// Fused quantized matvec: read the compressed weight bytes and dequantize one
/// row at a time into per-thread scratch, then dot with `x`. Reading ≈4-bit bytes
/// instead of an f32 expansion is the bandwidth win; `for_each_init` keeps one
/// scratch buffer per worker thread. Bit-identical to dequant-then-`cpu_matmul`.
pub fn quant_matvec(bytes: &[u8], ggml_type: u32, cols: usize, x: &[f32], out: &mut [f32]) {
    use rayon::prelude::*;
    let row_bytes = if ggml_type == HQ4_TYPE {
        crate::hos_quant::hq4_bytes(cols)
    } else if ggml_type == E8_TYPE {
        crate::hos_quant::e8_bytes(cols)
    } else {
        crate::gguf::bytes_for(ggml_type, cols).expect("row byte size")
    };
    out.par_iter_mut().enumerate().for_each_init(
        || vec![0.0f32; cols],
        |scratch, (o, yo)| {
            let rb = &bytes[o * row_bytes..(o + 1) * row_bytes];
            if ggml_type == HQ4_TYPE {
                crate::hos_quant::decode_hq4_into(rb, cols, scratch);
            } else if ggml_type == E8_TYPE {
                crate::hos_quant::decode_e8_into(rb, cols, scratch);
            } else {
                let _ = crate::gguf::Gguf::dequant_into(rb, ggml_type, cols, scratch);
            }
            *yo = dot_f32(scratch, x);
        },
    );
}

/// Dot product of two equal-length f32 slices. On aarch64 (Apple Silicon) this
/// uses NEON with two 4-wide FMA accumulators — the scalar `Σ a*b` loop is *not*
/// autovectorised by LLVM (f32 addition isn't associative), so this is the real
/// speed lever for the matvec inner loop. On x86_64 (consumer Windows/Intel/AMD)
/// the same two-accumulator design is ported to AVX2+FMA (two 256-bit YMM
/// accumulators, 16 floats/iter), runtime-gated with a scalar fallback so it
/// stays correct on pre-Haswell CPUs. Both SIMD paths reorder the summation, so
/// their parity target is the high-precision reference (see tests/parity.rs),
/// not the bit-exact scalar loop.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let n = a.len();
        let (ap, bp) = (a.as_ptr(), b.as_ptr());
        let (mut acc0, mut acc1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        let mut i = 0;
        while i + 8 <= n {
            acc0 = vfmaq_f32(acc0, vld1q_f32(ap.add(i)), vld1q_f32(bp.add(i)));
            acc1 = vfmaq_f32(acc1, vld1q_f32(ap.add(i + 4)), vld1q_f32(bp.add(i + 4)));
            i += 8;
        }
        let mut acc = vaddvq_f32(vaddq_f32(acc0, acc1));
        while i < n {
            acc += *ap.add(i) * *bp.add(i);
            i += 1;
        }
        acc
    }
    #[cfg(target_arch = "x86_64")]
    {
        if dot_x86_avx2_fma() {
            unsafe { dot_f32_avx2(a, b) }
        } else {
            dot_f32_scalar(a, b)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        dot_f32_scalar(a, b)
    }
}

/// Portable reference summation (also the fallback when SIMD is unavailable).
#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

/// Cached AVX2+FMA gate with a RUNTIME SELF-CHECK. Detect-once, never bypass
/// (calling the `target_feature` fn without the features is UB). Beyond detecting
/// the features, this verifies the AVX2 dot actually matches the scalar reference
/// on THIS cpu — the AVX2 path can't be runtime-tested when cross-compiled from
/// aarch64, so we validate it on the end user's hardware at first use. If it
/// diverges (a kernel bug, a quirky core), we fall back to scalar PERMANENTLY:
/// slower, but never wrong.
#[cfg(target_arch = "x86_64")]
#[inline]
fn dot_x86_avx2_fma() -> bool {
    use std::sync::OnceLock;
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return false;
        }
        // n not a multiple of 16 -> exercises the 8-wide block AND the scalar tail.
        let n = 37usize;
        let a: Vec<f32> = (0..n).map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.5).collect();
        let b: Vec<f32> = (0..n).map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.25).collect();
        let simd = unsafe { dot_f32_avx2(&a, &b) };
        let scalar = dot_f32_scalar(&a, &b);
        (simd - scalar).abs() <= 1e-3 + 1e-4 * scalar.abs()
    })
}

/// AVX2+FMA port of the NEON two-accumulator dot. 16 floats/iter via two YMM
/// FMA accumulators; horizontal-reduce, then a scalar tail. Only ever reached
/// through `dot_x86_avx2_fma()` so the enabled features are guaranteed present.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    let n = a.len();
    let (ap, bp) = (a.as_ptr(), b.as_ptr());
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= n {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(i)), _mm256_loadu_ps(bp.add(i)), acc0);
        acc1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(ap.add(i + 8)),
            _mm256_loadu_ps(bp.add(i + 8)),
            acc1,
        );
        i += 16;
    }
    // reduce the eight lanes of acc0+acc1 to a scalar.
    let sum8 = _mm256_add_ps(acc0, acc1);
    let lo = _mm256_castps256_ps128(sum8);
    let hi = _mm256_extractf128_ps(sum8, 1);
    let mut q = _mm_add_ps(lo, hi); // 4 lanes
    q = _mm_hadd_ps(q, q);
    q = _mm_hadd_ps(q, q);
    let mut acc = _mm_cvtss_f32(q);
    // one more 8-wide block, then scalar tail (covers non-multiple-of-16 lengths).
    while i + 8 <= n {
        let p = _mm256_mul_ps(_mm256_loadu_ps(ap.add(i)), _mm256_loadu_ps(bp.add(i)));
        let plo = _mm256_castps256_ps128(p);
        let phi = _mm256_extractf128_ps(p, 1);
        let mut r = _mm_add_ps(plo, phi);
        r = _mm_hadd_ps(r, r);
        r = _mm_hadd_ps(r, r);
        acc += _mm_cvtss_f32(r);
        i += 8;
    }
    while i < n {
        acc += *ap.add(i) * *bp.add(i);
        i += 1;
    }
    acc
}

/// Batched matvec for prefill: `out[T, O] = X[T, in] · Wᵀ`. The key win over T
/// separate matvecs is **weight reuse** — each weight row is read (and, for a
/// quantized weight, dequantized) exactly ONCE and dotted against all T prompt
/// tokens, instead of re-reading the whole matrix T times. Parallel over output
/// rows into a transposed `[O, T]` scratch (lock-free), then transposed back.
pub fn cpu_matmat(out: &mut [f32], w: &Weight, x: &[f32], t: usize) {
    use rayon::prelude::*;
    let (in_dim, out_dim) = match w {
        Weight::Cpu { cols, rows, .. } | Weight::Quant { cols, rows, .. } => (*cols, *rows),
        Weight::Gpu(_) => panic!("cpu_matmat on GPU weight"),
    };
    let mut tr = vec![0.0f32; out_dim * t]; // [out_dim, T]
    match w {
        Weight::Cpu { data, .. } => {
            tr.par_chunks_mut(t).enumerate().for_each(|(o, dst)| {
                let wrow = &data[o * in_dim..(o + 1) * in_dim];
                for (tok, d) in dst.iter_mut().enumerate() {
                    *d = dot_f32(wrow, &x[tok * in_dim..(tok + 1) * in_dim]);
                }
            });
        }
        Weight::Quant {
            bytes, ggml_type, ..
        } => {
            let row_bytes = if *ggml_type == HQ4_TYPE {
                crate::hos_quant::hq4_bytes(in_dim)
            } else if *ggml_type == E8_TYPE {
                crate::hos_quant::e8_bytes(in_dim)
            } else {
                crate::gguf::bytes_for(*ggml_type, in_dim).expect("row bytes")
            };
            tr.par_chunks_mut(t).enumerate().for_each_init(
                || vec![0.0f32; in_dim],
                |scratch, (o, dst)| {
                    let rb = &bytes[o * row_bytes..(o + 1) * row_bytes];
                    if *ggml_type == HQ4_TYPE {
                        crate::hos_quant::decode_hq4_into(rb, in_dim, scratch);
                    } else if *ggml_type == E8_TYPE {
                        crate::hos_quant::decode_e8_into(rb, in_dim, scratch);
                    } else {
                        let _ = crate::gguf::Gguf::dequant_into(rb, *ggml_type, in_dim, scratch);
                    }
                    for (tok, d) in dst.iter_mut().enumerate() {
                        *d = dot_f32(scratch, &x[tok * in_dim..(tok + 1) * in_dim]);
                    }
                },
            );
        }
        Weight::Gpu(_) => unreachable!(),
    }
    // transpose [out_dim, T] -> [T, out_dim]
    for o in 0..out_dim {
        for tok in 0..t {
            out[tok * out_dim + o] = tr[o * t + tok];
        }
    }
}

/// y[o] = sum_i w[o*in + i] * x[i]  (w is [out, in] row-major), parallel over rows.
pub fn cpu_matmul(y: &mut [f32], w: &[f32], x: &[f32]) {
    use rayon::prelude::*;
    let in_dim = x.len();
    y.par_iter_mut().enumerate().for_each(|(o, yo)| {
        *yo = dot_f32(&w[o * in_dim..o * in_dim + in_dim], x);
    });
}

#[derive(Debug, Clone)]
pub struct Config {
    pub dim: usize, // embedding length (d_model)
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub vocab_size: usize,
    pub ctx_len: usize,
    pub rms_eps: f32,
    pub rope_base: f32,
    pub arch: Arch,
    pub attn_bias: bool, // Qwen2-family has q/k/v bias; Llama/Mistral do not
    pub rope_neox: bool, // NEOX rope (Qwen2) vs interleaved (Llama)
    // architecture knobs (defaults reproduce the Llama path)
    pub embed_scale: f32,      // Gemma scales embeddings by sqrt(dim); else 1.0
    pub norm_add_one: bool,    // Gemma RMSNorm uses (1 + weight)
    pub geglu: bool,           // Gemma FFN uses GELU gate instead of SiLU
    pub attn_softcap: f32,     // Gemma2 attention logit soft-cap (0 = off)
    pub final_softcap: f32,    // Gemma2 final logit soft-cap (0 = off)
    pub n_experts: usize,      // MoE: total experts (0 = dense FFN)
    pub n_experts_used: usize, // MoE: experts routed per token (top-k)
}

/// A stack of per-expert weight matrices, kept in their native quantized bytes.
/// Expert `e` occupies a contiguous `expert_bytes` slice; it is dequantized to
/// f32 on demand (only the routed experts each token), so an N-billion-parameter
/// MoE costs its compressed size in memory, not the f32 blow-up.
pub struct QExperts {
    pub bytes: Vec<u8>,
    pub ggml_type: u32,
    pub n_experts: usize,
    pub rows: usize, // per-expert output dim
    pub cols: usize, // per-expert input dim
    pub expert_bytes: usize,
}

impl QExperts {
    /// Dequantize expert `e` (a [rows, cols] matrix) into `out[..rows*cols]`.
    pub fn dequant_expert(&self, e: usize, out: &mut [f32]) {
        let off = e * self.expert_bytes;
        let _ = crate::gguf::Gguf::dequant_into(
            &self.bytes[off..off + self.expert_bytes],
            self.ggml_type,
            self.rows * self.cols,
            &mut out[..self.rows * self.cols],
        );
    }
}

/// Mixture-of-experts FFN weights for one layer. The router is tiny (kept f32);
/// the experts stay quantized and are dequantized on demand.
pub struct MoeLayer {
    pub router: Vec<f32>, // [n_experts, dim]
    pub gate: QExperts,   // [n_experts, ffn, dim]
    pub up: QExperts,     // [n_experts, ffn, dim]
    pub down: QExperts,   // [n_experts, dim, ffn]
}

pub struct Layer {
    pub attn_norm: Vec<f32>,
    pub wq: Weight, // [n_heads*head_dim, dim]
    pub wk: Weight, // [n_kv_heads*head_dim, dim]
    pub wv: Weight,
    pub wo: Weight, // [dim, n_heads*head_dim]
    pub bq: Option<Vec<f32>>,
    pub bk: Option<Vec<f32>>,
    pub bv: Option<Vec<f32>>,
    pub ffn_norm: Vec<f32>,
    pub w_gate: Weight, // [ffn_dim, dim]
    pub w_up: Weight,   // [ffn_dim, dim]
    pub w_down: Weight, // [dim, ffn_dim]
    // Gemma2 "sandwich" norms applied to the attn/ffn block outputs (else None)
    pub post_attn_norm: Option<Vec<f32>>,
    pub post_ffn_norm: Option<Vec<f32>>,
    // OLMoE QK-norm (RMSNorm on the full q/k projections before RoPE)
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    // mixture-of-experts FFN (replaces the dense w_gate/w_up/w_down when present)
    pub moe: Option<MoeLayer>,
}

pub struct Model {
    pub cfg: Config,
    pub tok_embd: Vec<f32>, // [vocab, dim] (kept on CPU for embedding lookup)
    pub layers: Vec<Layer>,
    pub output_norm: Vec<f32>,
    pub output: Weight, // [vocab, dim] (lm head; tied to tok_embd if absent)
}

impl Model {
    pub fn load<S: ModelSource>(g: &S, gpu: Option<&Gpu>) -> Result<Model> {
        let arch_name = g
            .meta_str("general.architecture")
            .unwrap_or("llama")
            .to_string();
        let arch = Arch::detect(g);
        let k = |s: &str| format!("{arch_name}.{s}");

        // Hybrid SSM/Mamba architectures need the SSM stack (B2, in progress).
        if !arch.is_transformer() {
            return Err(HosError::UnsupportedArch(format!(
                "'{arch_name}' ({arch:?}) is a hybrid SSM/Mamba model — not runnable yet. \
                 HOS runs standard transformers today (Llama / Qwen2 / Mistral / Gemma / Phi3). \
                 Inspect its structure with `hos --qwen35-check -m <model>`."
            )));
        }
        // Gemma2/Phi3 run on the CPU path; ignore a GPU context for them.
        let gpu = if arch.gpu_supported() { gpu } else { None };
        let is_gemma = matches!(arch, Arch::Gemma);
        let is_phi3 = matches!(arch, Arch::Phi3);
        let attn_bias = g.has("blk.0.attn_q.bias");
        let attn_softcap = g.meta_f32(&k("attn_logit_softcapping")).unwrap_or(0.0);
        let final_softcap = g.meta_f32(&k("final_logit_softcapping")).unwrap_or(0.0);

        let need = |key: &str| {
            g.meta_u64(&k(key))
                .ok_or_else(|| HosError::MissingMeta(k(key)))
        };
        let dim = need("embedding_length")? as usize;
        let n_layers = need("block_count")? as usize;
        let n_heads = need("attention.head_count")? as usize;
        let n_kv_heads = g
            .meta_u64(&k("attention.head_count_kv"))
            .unwrap_or(n_heads as u64) as usize;
        let ffn_dim = need("feed_forward_length")? as usize;
        let ctx_len = g.meta_u64(&k("context_length")).unwrap_or(2048).min(8192) as usize;
        let rms_eps = g
            .meta_f32(&k("attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);
        let rope_base = g.meta_f32(&k("rope.freq_base")).unwrap_or(10000.0);
        let head_dim = g
            .meta_u64(&k("attention.key_length"))
            .map(|v| v as usize)
            .or_else(|| g.meta_u64(&k("rope.dimension_count")).map(|v| v as usize))
            .unwrap_or(dim / n_heads);

        let tok_embd = g.dequant("token_embd.weight")?;
        let vocab_size = tok_embd.len() / dim;

        let q_dim = n_heads * head_dim; // attn output projection input width

        // Linear weights go to the GPU in native quant form when `gpu` is set,
        // else dequantized to f32 on CPU. Norms + embeddings always stay on CPU.
        let mkw = |name: &str, cols: usize| -> Result<Weight> {
            Ok(match gpu {
                Some(gp) => {
                    let (bytes, ty, n) = g.raw(name)?;
                    Weight::Gpu(gp.upload_quant(bytes, ty, n / cols, cols))
                }
                None => Weight::cpu_from_source(g, name, cols)?,
            })
        };

        let bias = |name: &str| -> Result<Option<Vec<f32>>> {
            Ok(if attn_bias && g.has(name) {
                Some(g.dequant(name)?)
            } else {
                None
            })
        };
        // optional norm tensor (Gemma2 sandwich norms)
        let onorm = |name: &str| -> Result<Option<Vec<f32>>> {
            Ok(if g.has(name) {
                Some(g.dequant(name)?)
            } else {
                None
            })
        };
        let kv_dim = n_kv_heads * head_dim;
        // Mixture-of-experts (e.g. OLMoE): expert_count > 0 and ffn_*_exps tensors.
        let n_experts = g.meta_u64(&k("expert_count")).unwrap_or(0) as usize;
        let n_experts_used = g.meta_u64(&k("expert_used_count")).unwrap_or(0) as usize;
        let is_moe = n_experts > 0 && g.has("blk.0.ffn_gate_exps.weight");
        let dummy = || Weight::cpu(vec![0.0], 1); // placeholder for the unused dense FFN slot

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = |s: &str| format!("blk.{i}.{s}");
            // Phi3 fuses Q/K/V into attn_qkv; split by rows (each row is
            // independently quantized, so dequant then slice).
            let (wq, wk, wv) = if is_phi3 {
                let qkv = g.dequant(&p("attn_qkv.weight"))?;
                (
                    Weight::cpu(qkv[0..q_dim * dim].to_vec(), dim),
                    Weight::cpu(qkv[q_dim * dim..(q_dim + kv_dim) * dim].to_vec(), dim),
                    Weight::cpu(
                        qkv[(q_dim + kv_dim) * dim..(q_dim + 2 * kv_dim) * dim].to_vec(),
                        dim,
                    ),
                )
            } else {
                (
                    mkw(&p("attn_q.weight"), dim)?,
                    mkw(&p("attn_k.weight"), dim)?,
                    mkw(&p("attn_v.weight"), dim)?,
                )
            };
            // FFN: dense SwiGLU/GeGLU, or mixture-of-experts.
            let (w_gate, w_up, w_down, moe) = if is_moe {
                // experts stay quantized (dequantized on demand per routed expert)
                let experts = |name: &str, rows: usize, cols: usize| -> Result<QExperts> {
                    let (bytes, ty, _) = g.raw(name)?;
                    Ok(QExperts {
                        bytes: bytes.to_vec(),
                        ggml_type: ty,
                        n_experts,
                        rows,
                        cols,
                        expert_bytes: crate::gguf::bytes_for(ty, rows * cols)?,
                    })
                };
                let moe = MoeLayer {
                    router: g.dequant(&p("ffn_gate_inp.weight"))?,
                    gate: experts(&p("ffn_gate_exps.weight"), ffn_dim, dim)?,
                    up: experts(&p("ffn_up_exps.weight"), ffn_dim, dim)?,
                    down: experts(&p("ffn_down_exps.weight"), dim, ffn_dim)?,
                };
                (dummy(), dummy(), dummy(), Some(moe))
            } else if is_phi3 {
                let gu = g.dequant(&p("ffn_up.weight"))?;
                (
                    Weight::cpu(gu[0..ffn_dim * dim].to_vec(), dim),
                    Weight::cpu(gu[ffn_dim * dim..2 * ffn_dim * dim].to_vec(), dim),
                    mkw(&p("ffn_down.weight"), ffn_dim)?,
                    None,
                )
            } else {
                (
                    mkw(&p("ffn_gate.weight"), dim)?,
                    mkw(&p("ffn_up.weight"), dim)?,
                    mkw(&p("ffn_down.weight"), ffn_dim)?,
                    None,
                )
            };
            layers.push(Layer {
                attn_norm: g.dequant(&p("attn_norm.weight"))?,
                wq,
                wk,
                wv,
                wo: mkw(&p("attn_output.weight"), q_dim)?,
                bq: bias(&p("attn_q.bias"))?,
                bk: bias(&p("attn_k.bias"))?,
                bv: bias(&p("attn_v.bias"))?,
                ffn_norm: g.dequant(&p("ffn_norm.weight"))?,
                w_gate,
                w_up,
                w_down,
                post_attn_norm: onorm(&p("post_attention_norm.weight"))?,
                post_ffn_norm: onorm(&p("post_ffw_norm.weight"))?,
                q_norm: onorm(&p("attn_q_norm.weight"))?,
                k_norm: onorm(&p("attn_k_norm.weight"))?,
                moe,
            });
        }

        let output_norm = g.dequant("output_norm.weight")?;
        let output = if g.has("output.weight") {
            mkw("output.weight", dim)?
        } else {
            mkw("token_embd.weight", dim)? // tied embeddings
        };

        let cfg = Config {
            dim,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            ffn_dim,
            vocab_size,
            ctx_len,
            rms_eps,
            rope_base,
            arch,
            attn_bias,
            rope_neox: arch.rope_neox(),
            embed_scale: if is_gemma { (dim as f32).sqrt() } else { 1.0 },
            // NB: the HF->GGUF converter already bakes Gemma's (1+w) into the norm
            // weights, so we must NOT add 1 again at runtime.
            norm_add_one: false,
            geglu: is_gemma,
            attn_softcap,
            final_softcap,
            n_experts,
            n_experts_used,
        };

        eprintln!(
            "[hos] loaded {arch_name} ({arch:?}): dim={dim} layers={n_layers} heads={n_heads} kv_heads={n_kv_heads} head_dim={head_dim} ffn={ffn_dim} vocab={vocab_size} ctx={ctx_len} attn_bias={attn_bias}"
        );

        Ok(Model {
            cfg,
            tok_embd,
            layers,
            output_norm,
            output,
        })
    }
}
