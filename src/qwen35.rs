//! Experimental: Qwen3.5 (`qwen35`) hybrid architecture.
//!
//! 32 blocks = QK-norm attention every `full_attention_interval`-th layer +
//! Gated-DeltaNet-style SSM blocks for the rest, each followed by a SwiGLU FFN.
//!
//! Status: B2 step 1 — config + per-block structure validation. The forward
//! pass (esp. the gated delta-rule recurrence) is WIP and built/verified
//! incrementally. Kept fully separate from the supported transformer path.

use crate::error::{HosError, Result};
use crate::gguf::Gguf;

#[derive(Debug)]
pub struct Cfg {
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub vocab: usize,
    pub rms_eps: f32,
    pub rope_base: f32,
    pub rope_dim: usize,
    pub full_attn_interval: usize,
    // ssm
    pub conv_kernel: usize,
    pub state_size: usize,
    pub group_count: usize,
    pub inner_size: usize,
    pub time_step_rank: usize,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum BlockKind {
    Attention,
    Ssm,
}

impl Cfg {
    pub fn from_gguf(g: &Gguf) -> Result<Cfg> {
        let k = |s: &str| format!("qwen35.{s}");
        let need = |key: &str| {
            g.meta_u64(&k(key))
                .ok_or_else(|| HosError::MissingMeta(k(key)))
        };
        let dim = need("embedding_length")? as usize;
        Ok(Cfg {
            dim,
            n_layers: need("block_count")? as usize,
            n_heads: need("attention.head_count")? as usize,
            n_kv_heads: need("attention.head_count_kv")? as usize,
            head_dim: g
                .meta_u64(&k("attention.key_length"))
                .unwrap_or(dim as u64 / 16) as usize,
            ffn_dim: need("feed_forward_length")? as usize,
            vocab: g
                .tensors
                .get("token_embd.weight")
                .map(|t| t.dims[1] as usize)
                .unwrap_or(0),
            rms_eps: g
                .meta_f32(&k("attention.layer_norm_rms_epsilon"))
                .unwrap_or(1e-6),
            rope_base: g.meta_f32(&k("rope.freq_base")).unwrap_or(1e7),
            rope_dim: g.meta_u64(&k("rope.dimension_count")).unwrap_or(64) as usize,
            full_attn_interval: g.meta_u64(&k("full_attention_interval")).unwrap_or(4) as usize,
            conv_kernel: g.meta_u64(&k("ssm.conv_kernel")).unwrap_or(4) as usize,
            state_size: g.meta_u64(&k("ssm.state_size")).unwrap_or(128) as usize,
            group_count: g.meta_u64(&k("ssm.group_count")).unwrap_or(16) as usize,
            inner_size: g.meta_u64(&k("ssm.inner_size")).unwrap_or(0) as usize,
            time_step_rank: g.meta_u64(&k("ssm.time_step_rank")).unwrap_or(0) as usize,
        })
    }
}

/// Detect each block's kind by which tensors it carries.
pub fn block_kinds(g: &Gguf, n_layers: usize) -> Vec<BlockKind> {
    (0..n_layers)
        .map(|i| {
            if g.has(&format!("blk.{i}.ssm_a")) {
                BlockKind::Ssm
            } else {
                BlockKind::Attention
            }
        })
        .collect()
}

/// Load + validate structure: confirm every expected tensor exists per block
/// kind and report the layout. Foundation for the forward pass.
pub fn validate(g: &Gguf) -> Result<()> {
    let cfg = Cfg::from_gguf(g)?;
    eprintln!("[qwen35] {cfg:#?}");

    let kinds = block_kinds(g, cfg.n_layers);
    let n_attn = kinds.iter().filter(|k| **k == BlockKind::Attention).count();
    let n_ssm = kinds.iter().filter(|k| **k == BlockKind::Ssm).count();

    let pattern: String = kinds
        .iter()
        .map(|k| if *k == BlockKind::Attention { 'A' } else { 's' })
        .collect();
    eprintln!(
        "[qwen35] {} layers: {n_attn} attention, {n_ssm} SSM",
        cfg.n_layers
    );
    eprintln!("[qwen35] layout (A=attn, s=ssm): {pattern}");

    let attn_tensors = [
        "attn_norm.weight",
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_q_norm.weight",
        "attn_k_norm.weight",
        "attn_output.weight",
        "post_attention_norm.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
    ];
    let ssm_tensors = [
        "attn_norm.weight",
        "attn_qkv.weight",
        "attn_gate.weight",
        "ssm_a",
        "ssm_conv1d.weight",
        "ssm_dt.bias",
        "ssm_alpha.weight",
        "ssm_beta.weight",
        "ssm_norm.weight",
        "ssm_out.weight",
        "post_attention_norm.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
    ];

    let mut missing = 0;
    for (i, kind) in kinds.iter().enumerate() {
        let expected: &[&str] = if *kind == BlockKind::Attention {
            &attn_tensors
        } else {
            &ssm_tensors
        };
        for t in expected {
            let name = format!("blk.{i}.{t}");
            if !g.has(&name) {
                eprintln!("[qwen35] MISSING blk.{i} ({kind:?}): {t}");
                missing += 1;
            }
        }
    }

    // print one example of each block kind's tensor shapes
    for &(kind, idx) in &[
        (
            BlockKind::Attention,
            kinds.iter().position(|k| *k == BlockKind::Attention),
        ),
        (
            BlockKind::Ssm,
            kinds.iter().position(|k| *k == BlockKind::Ssm),
        ),
    ]
    .iter()
    .filter_map(|(k, o)| o.map(|i| (*k, i)))
    .collect::<Vec<_>>()
    {
        eprintln!("[qwen35] --- example {kind:?} block (blk.{idx}) shapes ---");
        let mut names: Vec<&String> = g
            .tensors
            .keys()
            .filter(|n| n.starts_with(&format!("blk.{idx}.")))
            .collect();
        names.sort();
        for n in names {
            let t = &g.tensors[n];
            eprintln!(
                "[qwen35]   {}  dims={:?} type={}",
                n.strip_prefix(&format!("blk.{idx}.")).unwrap(),
                t.dims,
                t.ggml_type
            );
        }
    }

    if missing == 0 {
        eprintln!(
            "[qwen35] ✅ structure validated — all expected tensors present for every block."
        );
    } else {
        eprintln!(
            "[qwen35] ⚠️  {missing} expected tensors missing — layout assumption needs revisiting."
        );
    }
    Ok(())
}

// ============================================================================
// Forward pass (CPU). Port of llama.cpp's qwen35 + gated delta-net.
// ============================================================================

use crate::metal_be::Gpu;
use crate::model::{cpu_matmul, Weight};

// Big matmul weights are `Weight` (CPU f32 or GPU f16, runtime-selected).
// Norms, conv, ssm_a/dt, and the tiny beta/alpha projections stay on CPU.
struct AttnW {
    attn_norm: Vec<f32>,
    wq: Weight,
    wk: Weight,
    wv: Weight,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    wo: Weight,
    post_norm: Vec<f32>,
    ffn_gate: Weight,
    ffn_up: Weight,
    ffn_down: Weight,
}
struct LinW {
    attn_norm: Vec<f32>,
    wqkv: Weight,
    wz: Weight,
    ssm_beta: Vec<f32>,
    ssm_alpha: Vec<f32>,
    ssm_a: Vec<f32>,
    ssm_dt: Vec<f32>,
    conv1d: Vec<f32>,
    ssm_norm: Vec<f32>,
    ssm_out: Weight,
    post_norm: Vec<f32>,
    ffn_gate: Weight,
    ffn_up: Weight,
    ffn_down: Weight,
}
enum Block {
    Attn(AttnW),
    Lin(LinW),
}

pub struct Qwen35 {
    pub cfg: Cfg,
    tok_embd: Vec<f32>,
    blocks: Vec<Block>,
    output_norm: Vec<f32>,
    output: Weight,
}

pub struct State {
    k_cache: Vec<Vec<f32>>, // per layer (attn): max_seq * n_kv_heads*head_dim
    v_cache: Vec<Vec<f32>>,
    conv: Vec<Vec<f32>>, // per layer (lin): conv_channels * (K-1)
    ssm: Vec<Vec<f32>>,  // per layer (lin): num_v_heads * head_v_dim^2
    pub pos: usize,
}

const MAX_SEQ: usize = 4096;

impl State {
    pub fn new(m: &Qwen35) -> State {
        let c = &m.cfg;
        let kv_dim = c.n_kv_heads * c.head_dim;
        let conv_ch = c.inner_size + 2 * c.group_count * c.state_size;
        let hv = c.inner_size / c.time_step_rank; // head_v_dim
        State {
            k_cache: vec![vec![0.0; MAX_SEQ * kv_dim]; c.n_layers],
            v_cache: vec![vec![0.0; MAX_SEQ * kv_dim]; c.n_layers],
            conv: vec![vec![0.0; conv_ch * (c.conv_kernel - 1)]; c.n_layers],
            ssm: vec![vec![0.0; c.time_step_rank * hv * hv]; c.n_layers],
            pos: 0,
        }
    }
}

fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let ss = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let s = 1.0 / (ss + eps).sqrt();
    (0..n).map(|i| x[i] * s * w[i]).collect()
}
fn rmsnorm_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    let n = x.len();
    let ss = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let s = 1.0 / (ss + eps).sqrt();
    for i in 0..n {
        x[i] = x[i] * s * w[i];
    }
}
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
fn mm(w: &[f32], x: &[f32], out_dim: usize) -> Vec<f32> {
    let mut y = vec![0.0; out_dim];
    cpu_matmul(&mut y, w, x);
    y
}
fn mmw(w: &Weight, x: &[f32], out_dim: usize, gpu: Option<&Gpu>) -> Vec<f32> {
    let mut y = vec![0.0; out_dim];
    w.matvec(gpu, x, &mut y);
    y
}
// NEOX partial rope: rotate first n_rot dims of each head (pairs i, i+n_rot/2)
fn rope(v: &mut [f32], n_heads: usize, head_dim: usize, n_rot: usize, pos: usize, base: f32) {
    let half = n_rot / 2;
    for h in 0..n_heads {
        let off = h * head_dim;
        for i in 0..half {
            let freq = 1.0 / base.powf(2.0 * i as f32 / n_rot as f32);
            let (s, c) = (pos as f32 * freq).sin_cos();
            let (a, b) = (off + i, off + i + half);
            let (x0, x1) = (v[a], v[b]);
            v[a] = x0 * c - x1 * s;
            v[b] = x0 * s + x1 * c;
        }
    }
}

impl Qwen35 {
    pub fn load(g: &Gguf, gpu: Option<&Gpu>) -> Result<Qwen35> {
        let cfg = Cfg::from_gguf(g)?;
        let kinds = block_kinds(g, cfg.n_layers);
        let tok_embd = g.dequant("token_embd.weight")?;
        // big matmul weight: GPU (f16 resident) when available, else CPU f32
        let w = |name: &str, cols: usize| -> Result<Weight> {
            Ok(match gpu {
                // upload native quantized bytes (coalesced K-quant kernels handle them)
                Some(gp) => {
                    let (bytes, ty, n) = g.raw(name)?;
                    Weight::Gpu(gp.upload_quant(bytes, ty, n / cols, cols))
                }
                None => Weight::cpu(g.dequant(name)?, cols),
            })
        };
        let dim = cfg.dim;
        let ffn = cfg.ffn_dim;
        let conv_ch = cfg.inner_size + 2 * cfg.group_count * cfg.state_size;

        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for (i, kind) in kinds.iter().enumerate() {
            let p = |s: &str| format!("blk.{i}.{s}");
            blocks.push(match kind {
                BlockKind::Attention => Block::Attn(AttnW {
                    attn_norm: g.dequant(&p("attn_norm.weight"))?,
                    wq: w(&p("attn_q.weight"), dim)?,
                    wk: w(&p("attn_k.weight"), dim)?,
                    wv: w(&p("attn_v.weight"), dim)?,
                    q_norm: g.dequant(&p("attn_q_norm.weight"))?,
                    k_norm: g.dequant(&p("attn_k_norm.weight"))?,
                    wo: w(&p("attn_output.weight"), cfg.n_heads * cfg.head_dim)?,
                    post_norm: g.dequant(&p("post_attention_norm.weight"))?,
                    ffn_gate: w(&p("ffn_gate.weight"), dim)?,
                    ffn_up: w(&p("ffn_up.weight"), dim)?,
                    ffn_down: w(&p("ffn_down.weight"), ffn)?,
                }),
                BlockKind::Ssm => Block::Lin(LinW {
                    attn_norm: g.dequant(&p("attn_norm.weight"))?,
                    wqkv: w(&p("attn_qkv.weight"), dim)?,
                    wz: w(&p("attn_gate.weight"), dim)?,
                    ssm_beta: g.dequant(&p("ssm_beta.weight"))?,
                    ssm_alpha: g.dequant(&p("ssm_alpha.weight"))?,
                    ssm_a: g.dequant(&p("ssm_a"))?,
                    ssm_dt: g.dequant(&p("ssm_dt.bias"))?,
                    conv1d: g.dequant(&p("ssm_conv1d.weight"))?,
                    ssm_norm: g.dequant(&p("ssm_norm.weight"))?,
                    ssm_out: w(&p("ssm_out.weight"), cfg.inner_size)?,
                    post_norm: g.dequant(&p("post_attention_norm.weight"))?,
                    ffn_gate: w(&p("ffn_gate.weight"), dim)?,
                    ffn_up: w(&p("ffn_up.weight"), dim)?,
                    ffn_down: w(&p("ffn_down.weight"), ffn)?,
                }),
            });
        }
        let _ = conv_ch;
        let output_norm = g.dequant("output_norm.weight")?;
        let output = if g.has("output.weight") {
            w("output.weight", dim)?
        } else {
            w("token_embd.weight", dim)?
        };
        eprintln!(
            "[qwen35] loaded: {} blocks, vocab {} ({})",
            cfg.n_layers,
            cfg.vocab,
            if gpu.is_some() { "GPU matmuls" } else { "CPU" }
        );
        Ok(Qwen35 {
            cfg,
            tok_embd,
            blocks,
            output_norm,
            output,
        })
    }

    fn ffn(
        &self,
        x: &[f32],
        gate: &Weight,
        up: &Weight,
        down: &Weight,
        gpu: Option<&Gpu>,
    ) -> Vec<f32> {
        let f = self.cfg.ffn_dim;
        let mut g = mmw(gate, x, f, gpu);
        let u = mmw(up, x, f, gpu);
        for i in 0..f {
            g[i] = silu(g[i]) * u[i];
        }
        mmw(down, &g, self.cfg.dim, gpu)
    }

    fn attn_block(
        &self,
        w: &AttnW,
        x: &[f32],
        st: &mut State,
        il: usize,
        pos: usize,
        gpu: Option<&Gpu>,
    ) -> Vec<f32> {
        let c = &self.cfg;
        let (hd, nh, nkv) = (c.head_dim, c.n_heads, c.n_kv_heads);
        let kv_dim = nkv * hd;
        let qfull = mmw(&w.wq, x, nh * hd * 2, gpu);
        let mut q = vec![0.0f32; nh * hd];
        let mut gate = vec![0.0f32; nh * hd];
        for h in 0..nh {
            let base = h * hd * 2;
            q[h * hd..h * hd + hd].copy_from_slice(&qfull[base..base + hd]);
            gate[h * hd..h * hd + hd].copy_from_slice(&qfull[base + hd..base + 2 * hd]);
        }
        for h in 0..nh {
            rmsnorm_inplace(&mut q[h * hd..h * hd + hd], &w.q_norm, c.rms_eps);
        }
        let mut k = mmw(&w.wk, x, kv_dim, gpu);
        let v = mmw(&w.wv, x, kv_dim, gpu);
        for h in 0..nkv {
            rmsnorm_inplace(&mut k[h * hd..h * hd + hd], &w.k_norm, c.rms_eps);
        }
        rope(&mut q, nh, hd, c.rope_dim, pos, c.rope_base);
        rope(&mut k, nkv, hd, c.rope_dim, pos, c.rope_base);
        st.k_cache[il][pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&k);
        st.v_cache[il][pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&v);
        let scale = 1.0 / (hd as f32).sqrt();
        let kv_mul = nh / nkv;
        let mut att = vec![0.0f32; nh * hd];
        for h in 0..nh {
            let kvh = h / kv_mul;
            let qh = &q[h * hd..h * hd + hd];
            let mut scores = vec![0.0f32; pos + 1];
            for t in 0..=pos {
                let kh = &st.k_cache[il][t * kv_dim + kvh * hd..t * kv_dim + kvh * hd + hd];
                scores[t] = qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f32>() * scale;
            }
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for s in scores.iter_mut() {
                *s = (*s - mx).exp();
                sum += *s;
            }
            let oh = &mut att[h * hd..h * hd + hd];
            for t in 0..=pos {
                let vh = &st.v_cache[il][t * kv_dim + kvh * hd..t * kv_dim + kvh * hd + hd];
                let wt = scores[t] / sum;
                for i in 0..hd {
                    oh[i] += wt * vh[i];
                }
            }
        }
        for i in 0..nh * hd {
            att[i] *= 1.0 / (1.0 + (-gate[i]).exp());
        } // * sigmoid(gate)
        mmw(&w.wo, &att, c.dim, gpu)
    }

    fn lin_block(
        &self,
        w: &LinW,
        x: &[f32],
        st: &mut State,
        il: usize,
        gpu: Option<&Gpu>,
    ) -> Vec<f32> {
        let c = &self.cfg;
        let hk = c.state_size; // head_k_dim = 128
        let nk = c.group_count; // num_k_heads = 16
        let nv = c.time_step_rank; // num_v_heads = 32
        let hv = c.inner_size / nv; // head_v_dim = 128
        let conv_ch = c.inner_size + 2 * nk * hk; // 8192
        let kk = c.conv_kernel; // 4

        let qkv = mmw(&w.wqkv, x, conv_ch, gpu);
        let z = mmw(&w.wz, x, c.inner_size, gpu);
        let beta: Vec<f32> = mm(&w.ssm_beta, x, nv)
            .iter()
            .map(|a| 1.0 / (1.0 + (-a).exp()))
            .collect();
        let alpha = mm(&w.ssm_alpha, x, nv);
        let decay: Vec<f32> = (0..nv)
            .map(|h| {
                let sp = {
                    let z = alpha[h] + w.ssm_dt[h];
                    // softplus
                    if z > 20.0 {
                        z
                    } else {
                        (1.0 + z.exp()).ln()
                    }
                };
                w.ssm_a[h] * sp
            })
            .collect();

        // causal depthwise conv1d over conv_ch channels, kernel kk, then silu
        let cs = &mut st.conv[il]; // conv_ch * (kk-1)
        let mut conv_out = vec![0.0f32; conv_ch];
        for ch in 0..conv_ch {
            let mut acc = 0.0;
            // window = [state(kk-1) , new]
            for t in 0..kk - 1 {
                acc += cs[ch * (kk - 1) + t] * w.conv1d[ch * kk + t];
            }
            acc += qkv[ch] * w.conv1d[ch * kk + (kk - 1)];
            conv_out[ch] = silu(acc);
            // shift state: keep last (kk-1) of [state.., new]
            for t in 0..kk - 2 {
                cs[ch * (kk - 1) + t] = cs[ch * (kk - 1) + t + 1];
            }
            cs[ch * (kk - 1) + (kk - 2)] = qkv[ch];
        }

        // split q (nk*hk), k (nk*hk), v (nv*hv)
        let q0 = 0;
        let k0 = nk * hk;
        let v0 = 2 * nk * hk;
        let l2 = |s: &[f32]| {
            let n = s.iter().map(|x| x * x).sum::<f32>();
            let inv = 1.0 / (n + c.rms_eps).sqrt();
            s.iter().map(|x| x * inv).collect::<Vec<f32>>()
        };
        let qscale = 1.0 / (hk as f32).sqrt();
        let mut out = vec![0.0f32; nv * hv];
        for hvh in 0..nv {
            let khh = hvh % nk; // ggml_repeat tiles k-heads (j % nk), not interleaves
            let qh: Vec<f32> = l2(&conv_out[q0 + khh * hk..q0 + khh * hk + hk])
                .iter()
                .map(|x| x * qscale)
                .collect();
            let kh = l2(&conv_out[k0 + khh * hk..k0 + khh * hk + hk]);
            let vh = &conv_out[v0 + hvh * hv..v0 + hvh * hv + hv];
            let s = &mut st.ssm[il][hvh * hv * hv..(hvh + 1) * hv * hv];
            let g = decay[hvh].exp();
            for x in s.iter_mut() {
                *x *= g;
            }
            let kq: f32 = (0..hk).map(|i| kh[i] * qh[i]).sum();
            let mut d = vec![0.0f32; hv];
            let mut o = vec![0.0f32; hv];
            for j in 0..hv {
                let row = &s[j * hv..j * hv + hv];
                let sk: f32 = (0..hk).map(|i| row[i] * kh[i]).sum();
                let sq: f32 = (0..hk).map(|i| row[i] * qh[i]).sum();
                d[j] = beta[hvh] * (vh[j] - sk);
                o[j] = sq + d[j] * kq;
            }
            for i in 0..hv {
                let di = d[i];
                let si = &mut s[i * hv..i * hv + hv];
                for j in 0..hv {
                    si[j] += di * kh[j];
                }
            }
            let on = rmsnorm(&o, &w.ssm_norm, c.rms_eps);
            let zh = &z[hvh * hv..hvh * hv + hv];
            for j in 0..hv {
                out[hvh * hv + j] = on[j] * silu(zh[j]);
            }
        }
        mmw(&w.ssm_out, &out, c.dim, gpu)
    }

    pub fn forward(&self, st: &mut State, token: u32, pos: usize, gpu: Option<&Gpu>) -> Vec<f32> {
        let c = &self.cfg;
        let mut x = self.tok_embd[token as usize * c.dim..(token as usize + 1) * c.dim].to_vec();
        for il in 0..c.n_layers {
            let (cur, post, fg, fu, fd) = match &self.blocks[il] {
                Block::Attn(w) => {
                    let xn = rmsnorm(&x, &w.attn_norm, c.rms_eps);
                    (
                        self.attn_block(w, &xn, st, il, pos, gpu),
                        &w.post_norm,
                        &w.ffn_gate,
                        &w.ffn_up,
                        &w.ffn_down,
                    )
                }
                Block::Lin(w) => {
                    let xn = rmsnorm(&x, &w.attn_norm, c.rms_eps);
                    (
                        self.lin_block(w, &xn, st, il, gpu),
                        &w.post_norm,
                        &w.ffn_gate,
                        &w.ffn_up,
                        &w.ffn_down,
                    )
                }
            };
            for i in 0..c.dim {
                x[i] += cur[i];
            }
            let xpn = rmsnorm(&x, post, c.rms_eps);
            let ffn = self.ffn(&xpn, fg, fu, fd, gpu);
            for i in 0..c.dim {
                x[i] += ffn[i];
            }
        }
        rmsnorm_inplace(&mut x, &self.output_norm, c.rms_eps);
        mmw(&self.output, &x, c.vocab, gpu)
    }
}

// ============================================================================
// Qwen35Gpu: fully GPU-resident runner. Activations + KV/conv/ssm state live on
// the GPU; one command buffer + one encoder per token (no CPU round-trips).
// ============================================================================

/// CPU-only stub of the qwen35 GPU runner for non-macOS targets — never built
/// (the qwen35 GPU gate is `cfg(target_os="macos")`), present only to type-check.
#[cfg(not(target_os = "macos"))]
pub struct Qwen35Gpu;
#[cfg(not(target_os = "macos"))]
impl Qwen35Gpu {
    pub fn new(_gpu: &Gpu, _m: &Qwen35) -> Qwen35Gpu {
        unreachable!("the qwen35 GPU runner is macOS-only")
    }
    pub fn forward(&self, _m: &Qwen35, _token: u32, _pos: usize) -> Vec<f32> {
        unreachable!("the qwen35 GPU runner is macOS-only")
    }
}

#[cfg(target_os = "macos")]
use crate::metal_be::GpuMatrix;
#[cfg(target_os = "macos")]
use metal::{
    Buffer, BufferRef, CommandQueue, ComputeCommandEncoderRef, ComputePipelineState, Device,
    MTLResourceOptions, MTLSize, ResourceRef,
};
#[cfg(target_os = "macos")]
use std::ffi::c_void;

#[cfg(target_os = "macos")]
pub struct Qwen35Gpu {
    #[allow(dead_code)] // held to keep the Metal device alive for the runner's lifetime
    device: Device,
    queue: CommandQueue,
    p_mv_f16: ComputePipelineState,
    p_mv_q4k: ComputePipelineState,
    p_mv_q5k: ComputePipelineState,
    p_mv_q6k: ComputePipelineState,
    p_mv_q8: ComputePipelineState,
    p_rms: ComputePipelineState,
    p_rms_h: ComputePipelineState,
    p_l2_h: ComputePipelineState,
    p_sig_mul: ComputePipelineState,
    p_sig_ip: ComputePipelineState,
    p_decay: ComputePipelineState,
    p_conv: ComputePipelineState,
    p_delta: ComputePipelineState,
    p_gnorm: ComputePipelineState,
    p_rope: ComputePipelineState,
    p_store: ComputePipelineState,
    p_attn: ComputePipelineState,
    p_swiglu: ComputePipelineState,
    p_add: ComputePipelineState,
    p_qgate: ComputePipelineState,
    // activation buffers
    x: Buffer,
    xb: Buffer,
    qfull: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    att: Buffer,
    gate: Buffer,
    qkv: Buffer,
    z: Buffer,
    conv_out: Buffer,
    beta: Buffer,
    decay: Buffer,
    dout: Buffer,
    gbuf: Buffer,
    ubuf: Buffer,
    logits: Buffer,
    // per-layer norms / small params (None where not applicable)
    attn_norm: Vec<Buffer>,
    post_norm: Vec<Buffer>,
    q_norm: Vec<Option<Buffer>>,
    k_norm: Vec<Option<Buffer>>,
    ssm_norm: Vec<Option<Buffer>>,
    conv_w: Vec<Option<Buffer>>,
    ssm_a: Vec<Option<Buffer>>,
    ssm_dt: Vec<Option<Buffer>>,
    ssm_beta: Vec<Option<GpuMatrix>>,
    ssm_alpha: Vec<Option<GpuMatrix>>,
    output_norm: Buffer,
    // state
    kcache: Vec<Buffer>,
    vcache: Vec<Buffer>,
    conv_state: Vec<Buffer>,
    ssm_state: Vec<Buffer>,
}

#[cfg(target_os = "macos")]
impl Qwen35Gpu {
    pub fn new(gpu: &Gpu, m: &Qwen35) -> Qwen35Gpu {
        let device = gpu.device().clone();
        let queue = gpu.queue().clone();
        let lib = gpu.fused_library();
        let pl = |n: &str| {
            device
                .new_compute_pipeline_state_with_function(&lib.get_function(n, None).unwrap())
                .unwrap()
        };
        let c = &m.cfg;
        let nb =
            |n: usize| device.new_buffer((n * 4) as u64, MTLResourceOptions::StorageModeShared);
        let up = |d: &[f32]| {
            device.new_buffer_with_data(
                d.as_ptr() as *const c_void,
                (d.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };

        let kv_dim = c.n_kv_heads * c.head_dim;
        let conv_ch = c.inner_size + 2 * c.group_count * c.state_size;
        let hv = c.inner_size / c.time_step_rank;
        let kinds = block_kinds_from(m);

        let mut attn_norm = Vec::new();
        let mut post_norm = Vec::new();
        let mut q_norm = Vec::new();
        let mut k_norm = Vec::new();
        let mut ssm_norm = Vec::new();
        let mut conv_w = Vec::new();
        let mut ssm_a = Vec::new();
        let mut ssm_dt = Vec::new();
        let mut ssm_beta = Vec::new();
        let mut ssm_alpha = Vec::new();
        for (i, blk) in m.blocks.iter().enumerate() {
            let _ = i;
            match blk {
                Block::Attn(w) => {
                    attn_norm.push(up(&w.attn_norm));
                    post_norm.push(up(&w.post_norm));
                    q_norm.push(Some(up(&w.q_norm)));
                    k_norm.push(Some(up(&w.k_norm)));
                    ssm_norm.push(None);
                    conv_w.push(None);
                    ssm_a.push(None);
                    ssm_dt.push(None);
                    ssm_beta.push(None);
                    ssm_alpha.push(None);
                }
                Block::Lin(w) => {
                    attn_norm.push(up(&w.attn_norm));
                    post_norm.push(up(&w.post_norm));
                    q_norm.push(None);
                    k_norm.push(None);
                    ssm_norm.push(Some(up(&w.ssm_norm)));
                    conv_w.push(Some(up(&w.conv1d)));
                    ssm_a.push(Some(up(&w.ssm_a)));
                    ssm_dt.push(Some(up(&w.ssm_dt)));
                    ssm_beta.push(Some(gpu.upload_matrix(
                        &w.ssm_beta,
                        c.time_step_rank,
                        c.dim,
                    )));
                    ssm_alpha.push(Some(gpu.upload_matrix(
                        &w.ssm_alpha,
                        c.time_step_rank,
                        c.dim,
                    )));
                }
            }
        }
        let mut kcache = Vec::new();
        let mut vcache = Vec::new();
        let mut conv_state = Vec::new();
        let mut ssm_state = Vec::new();
        for kind in &kinds {
            match kind {
                BlockKind::Attention => {
                    kcache.push(nb(MAX_SEQ * kv_dim));
                    vcache.push(nb(MAX_SEQ * kv_dim));
                    conv_state.push(nb(1));
                    ssm_state.push(nb(1));
                }
                BlockKind::Ssm => {
                    kcache.push(nb(1));
                    vcache.push(nb(1));
                    conv_state.push(nb(conv_ch * (c.conv_kernel - 1)));
                    ssm_state.push(nb(c.time_step_rank * hv * hv));
                }
            }
        }
        eprintln!("[qwen35] GPU-resident runner ready");
        Qwen35Gpu {
            p_mv_f16: pl("matvec_f16"),
            p_mv_q4k: pl("matvec_q4k_co"),
            p_mv_q5k: pl("matvec_q5k_co"),
            p_mv_q6k: pl("matvec_q6k_co"),
            p_mv_q8: pl("matvec_q8_0_co"),
            p_rms: pl("rmsnorm"),
            p_rms_h: pl("rmsnorm_heads"),
            p_l2_h: pl("l2norm_heads"),
            p_sig_mul: pl("sigmoid_mul"),
            p_sig_ip: pl("sigmoid_inplace"),
            p_decay: pl("ssm_decay"),
            p_conv: pl("conv1d"),
            p_delta: pl("deltanet_multi"),
            p_gnorm: pl("gated_norm"),
            p_rope: pl("rope_partial"),
            p_store: pl("store_kv"),
            p_attn: pl("attention"),
            p_swiglu: pl("swiglu"),
            p_add: pl("add_inplace"),
            p_qgate: pl("extract_qgate"),
            x: nb(c.dim),
            xb: nb(c.dim),
            qfull: nb(c.n_heads * c.head_dim * 2),
            q: nb(c.n_heads * c.head_dim),
            k: nb(kv_dim),
            v: nb(kv_dim),
            att: nb(c.n_heads * c.head_dim),
            gate: nb(c.n_heads * c.head_dim),
            qkv: nb(conv_ch),
            z: nb(c.inner_size),
            conv_out: nb(conv_ch),
            beta: nb(c.time_step_rank),
            decay: nb(c.time_step_rank),
            dout: nb(c.inner_size),
            gbuf: nb(c.ffn_dim),
            ubuf: nb(c.ffn_dim),
            logits: nb(c.vocab),
            attn_norm,
            post_norm,
            q_norm,
            k_norm,
            ssm_norm,
            conv_w,
            ssm_a,
            ssm_dt,
            ssm_beta,
            ssm_alpha,
            output_norm: up(&m.output_norm),
            kcache,
            vcache,
            conv_state,
            ssm_state,
            device,
            queue,
        }
    }

    fn bar(enc: &ComputeCommandEncoderRef, outs: &[&BufferRef]) {
        let r: Vec<&ResourceRef> = outs
            .iter()
            .map(|&b| {
                let x: &ResourceRef = b;
                x
            })
            .collect();
        enc.memory_barrier_with_resources(&r);
    }
    fn set_u(enc: &ComputeCommandEncoderRef, idx: u64, val: u32) {
        enc.set_bytes(idx, 4, &val as *const u32 as *const c_void);
    }
    fn set_f(enc: &ComputeCommandEncoderRef, idx: u64, val: f32) {
        enc.set_bytes(idx, 4, &val as *const f32 as *const c_void);
    }
    // matvec y = W x ; coalesced (32-lane/row) for K-quants, scalar for f16
    fn mv(&self, enc: &ComputeCommandEncoderRef, w: &GpuMatrix, x: &BufferRef, y: &BufferRef) {
        use crate::gguf::{GGML_F16, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0};
        let (p, coalesced) = match w.ggml_type {
            GGML_Q4_K => (&self.p_mv_q4k, true),
            GGML_Q5_K => (&self.p_mv_q5k, true),
            GGML_Q6_K => (&self.p_mv_q6k, true),
            GGML_Q8_0 => (&self.p_mv_q8, true),
            GGML_F16 => (&self.p_mv_f16, false),
            t => {
                eprintln!("[qwen35] no resident matvec for type {t}");
                std::process::exit(2);
            }
        };
        enc.set_compute_pipeline_state(p);
        enc.set_buffer(0, Some(w.buffer()), 0);
        enc.set_buffer(1, Some(x), 0);
        enc.set_buffer(2, Some(y), 0);
        Self::set_u(enc, 3, w.in_dim as u32);
        if coalesced {
            // coalesced kernels process NDST=2 rows per 32-lane simdgroup and
            // need n_rows in buffer(4) for the tail guard (must match metal_be).
            const NDST: u64 = 2;
            Self::set_u(enc, 4, w.n_rows as u32);
            let simdgroups = (w.n_rows as u64).div_ceil(NDST);
            enc.dispatch_threads(MTLSize::new(simdgroups * 32, 1, 1), MTLSize::new(32, 1, 1));
        } else {
            enc.dispatch_threads(MTLSize::new(w.n_rows as u64, 1, 1), MTLSize::new(256, 1, 1));
        }
        Self::bar(enc, &[y]);
    }
    fn rms(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &BufferRef,
        w: &BufferRef,
        out: &BufferRef,
        n: usize,
        eps: f32,
    ) {
        enc.set_compute_pipeline_state(&self.p_rms);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(w), 0);
        enc.set_buffer(2, Some(out), 0);
        Self::set_u(enc, 3, n as u32);
        Self::set_f(enc, 4, eps);
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        Self::bar(enc, &[out]);
    }
    fn elt(
        &self,
        enc: &ComputeCommandEncoderRef,
        p: &ComputePipelineState,
        a: &BufferRef,
        b: Option<&BufferRef>,
        n: usize,
    ) {
        enc.set_compute_pipeline_state(p);
        enc.set_buffer(0, Some(a), 0);
        if let Some(bb) = b {
            enc.set_buffer(1, Some(bb), 0);
        }
        let tg = p.max_total_threads_per_threadgroup().min(256);
        enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(tg, 1, 1));
        Self::bar(enc, &[a]);
    }

    pub fn forward(&self, m: &Qwen35, token: u32, pos: usize) -> Vec<f32> {
        let c = &m.cfg;
        let (dim, hd, nh, nkv) = (c.dim, c.head_dim, c.n_heads, c.n_kv_heads);
        let kv_dim = nkv * hd;
        let kv_mul = nh / nkv;
        let hv = c.inner_size / c.time_step_rank;
        let conv_ch = c.inner_size + 2 * c.group_count * c.state_size;
        let q0b = 0u64;
        let k0b = (c.group_count * c.state_size * 4) as u64;
        let v0b = (2 * c.group_count * c.state_size * 4) as u64;
        let f4 = 4u64;

        // embedding -> x
        let row = &m.tok_embd[token as usize * dim..(token as usize + 1) * dim];
        unsafe {
            std::ptr::copy_nonoverlapping(row.as_ptr(), self.x.contents() as *mut f32, dim);
        }

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        for (il, blk) in m.blocks.iter().enumerate() {
            self.rms(enc, &self.x, &self.attn_norm[il], &self.xb, dim, c.rms_eps);
            match blk {
                Block::Attn(w) => {
                    self.mv(enc, w.wq.as_gpu(), &self.xb, &self.qfull);
                    // extract q + gate
                    enc.set_compute_pipeline_state(&self.p_qgate);
                    enc.set_buffer(0, Some(&self.qfull), 0);
                    enc.set_buffer(1, Some(&self.q), 0);
                    enc.set_buffer(2, Some(&self.gate), 0);
                    Self::set_u(enc, 3, hd as u32);
                    enc.dispatch_threads(
                        MTLSize::new((nh * hd) as u64, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    Self::bar(enc, &[&self.q, &self.gate]);
                    self.mv(enc, w.wk.as_gpu(), &self.xb, &self.k);
                    self.mv(enc, w.wv.as_gpu(), &self.xb, &self.v);
                    // qk-norm per head
                    enc.set_compute_pipeline_state(&self.p_rms_h);
                    enc.set_buffer(0, Some(&self.q), 0);
                    enc.set_buffer(1, Some(self.q_norm[il].as_ref().unwrap()), 0);
                    Self::set_u(enc, 2, hd as u32);
                    Self::set_f(enc, 3, c.rms_eps);
                    enc.dispatch_thread_groups(
                        MTLSize::new(nh as u64, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    Self::bar(enc, &[&self.q]);
                    enc.set_compute_pipeline_state(&self.p_rms_h);
                    enc.set_buffer(0, Some(&self.k), 0);
                    enc.set_buffer(1, Some(self.k_norm[il].as_ref().unwrap()), 0);
                    Self::set_u(enc, 2, hd as u32);
                    Self::set_f(enc, 3, c.rms_eps);
                    enc.dispatch_thread_groups(
                        MTLSize::new(nkv as u64, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    Self::bar(enc, &[&self.k]);
                    // partial rope
                    for (buf, heads) in [(&self.q, nh), (&self.k, nkv)] {
                        enc.set_compute_pipeline_state(&self.p_rope);
                        enc.set_buffer(0, Some(buf), 0);
                        Self::set_u(enc, 1, hd as u32);
                        Self::set_u(enc, 2, c.rope_dim as u32);
                        Self::set_u(enc, 3, pos as u32);
                        Self::set_f(enc, 4, c.rope_base);
                        enc.dispatch_threads(
                            MTLSize::new((heads * c.rope_dim / 2) as u64, 1, 1),
                            MTLSize::new(64, 1, 1),
                        );
                        Self::bar(enc, &[buf]);
                    }
                    // store kv
                    enc.set_compute_pipeline_state(&self.p_store);
                    enc.set_buffer(0, Some(&self.k), 0);
                    enc.set_buffer(1, Some(&self.v), 0);
                    enc.set_buffer(2, Some(&self.kcache[il]), 0);
                    enc.set_buffer(3, Some(&self.vcache[il]), 0);
                    Self::set_u(enc, 4, kv_dim as u32);
                    Self::set_u(enc, 5, pos as u32);
                    enc.dispatch_threads(
                        MTLSize::new(kv_dim as u64, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    Self::bar(enc, &[&self.kcache[il], &self.vcache[il]]);
                    // attention
                    enc.set_compute_pipeline_state(&self.p_attn);
                    enc.set_buffer(0, Some(&self.q), 0);
                    enc.set_buffer(1, Some(&self.kcache[il]), 0);
                    enc.set_buffer(2, Some(&self.vcache[il]), 0);
                    enc.set_buffer(3, Some(&self.att), 0);
                    Self::set_u(enc, 4, hd as u32);
                    Self::set_u(enc, 5, kv_dim as u32);
                    Self::set_u(enc, 6, kv_mul as u32);
                    Self::set_u(enc, 7, pos as u32);
                    // new attention kernel: one threadgroup per head, hd threads
                    enc.dispatch_thread_groups(
                        MTLSize::new(nh as u64, 1, 1),
                        MTLSize::new(hd as u64, 1, 1),
                    );
                    Self::bar(enc, &[&self.att]);
                    self.elt(enc, &self.p_sig_mul, &self.att, Some(&self.gate), nh * hd);
                    self.mv(enc, w.wo.as_gpu(), &self.att, &self.xb);
                }
                Block::Lin(w) => {
                    self.mv(enc, w.wqkv.as_gpu(), &self.xb, &self.qkv);
                    self.mv(enc, w.wz.as_gpu(), &self.xb, &self.z);
                    self.mv(
                        enc,
                        self.ssm_beta[il].as_ref().unwrap(),
                        &self.xb,
                        &self.beta,
                    );
                    self.elt(enc, &self.p_sig_ip, &self.beta, None, c.time_step_rank);
                    self.mv(
                        enc,
                        self.ssm_alpha[il].as_ref().unwrap(),
                        &self.xb,
                        &self.decay,
                    );
                    enc.set_compute_pipeline_state(&self.p_decay);
                    enc.set_buffer(0, Some(&self.decay), 0);
                    enc.set_buffer(1, Some(self.ssm_a[il].as_ref().unwrap()), 0);
                    enc.set_buffer(2, Some(self.ssm_dt[il].as_ref().unwrap()), 0);
                    enc.dispatch_threads(
                        MTLSize::new(c.time_step_rank as u64, 1, 1),
                        MTLSize::new(32, 1, 1),
                    );
                    Self::bar(enc, &[&self.decay]);
                    // conv1d
                    enc.set_compute_pipeline_state(&self.p_conv);
                    enc.set_buffer(0, Some(&self.qkv), 0);
                    enc.set_buffer(1, Some(&self.conv_state[il]), 0);
                    enc.set_buffer(2, Some(self.conv_w[il].as_ref().unwrap()), 0);
                    enc.set_buffer(3, Some(&self.conv_out), 0);
                    Self::set_u(enc, 4, c.conv_kernel as u32);
                    enc.dispatch_threads(
                        MTLSize::new(conv_ch as u64, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    Self::bar(enc, &[&self.conv_out, &self.conv_state[il]]);
                    // l2 norm q and k regions
                    for off in [q0b, k0b] {
                        enc.set_compute_pipeline_state(&self.p_l2_h);
                        enc.set_buffer(0, Some(&self.conv_out), off);
                        Self::set_u(enc, 1, c.state_size as u32);
                        Self::set_f(enc, 2, c.rms_eps);
                        enc.dispatch_thread_groups(
                            MTLSize::new(c.group_count as u64, 1, 1),
                            MTLSize::new(128, 1, 1),
                        );
                        Self::bar(enc, &[&self.conv_out]);
                    }
                    // deltanet
                    enc.set_compute_pipeline_state(&self.p_delta);
                    enc.set_buffer(0, Some(&self.ssm_state[il]), 0);
                    enc.set_buffer(1, Some(&self.conv_out), q0b);
                    enc.set_buffer(2, Some(&self.conv_out), k0b);
                    enc.set_buffer(3, Some(&self.conv_out), v0b);
                    enc.set_buffer(4, Some(&self.decay), 0);
                    enc.set_buffer(5, Some(&self.beta), 0);
                    enc.set_buffer(6, Some(&self.dout), 0);
                    Self::set_u(enc, 7, hv as u32);
                    Self::set_u(enc, 8, c.state_size as u32);
                    Self::set_u(enc, 9, c.group_count as u32);
                    Self::set_f(enc, 10, 1.0 / (c.state_size as f32).sqrt());
                    enc.dispatch_thread_groups(
                        MTLSize::new(c.time_step_rank as u64, 1, 1),
                        MTLSize::new(hv as u64, 1, 1),
                    );
                    Self::bar(enc, &[&self.dout, &self.ssm_state[il]]);
                    // gated norm
                    enc.set_compute_pipeline_state(&self.p_gnorm);
                    enc.set_buffer(0, Some(&self.dout), 0);
                    enc.set_buffer(1, Some(self.ssm_norm[il].as_ref().unwrap()), 0);
                    enc.set_buffer(2, Some(&self.z), 0);
                    Self::set_u(enc, 3, hv as u32);
                    Self::set_f(enc, 4, c.rms_eps);
                    enc.dispatch_thread_groups(
                        MTLSize::new(c.time_step_rank as u64, 1, 1),
                        MTLSize::new(hv as u64, 1, 1),
                    );
                    Self::bar(enc, &[&self.dout]);
                    self.mv(enc, w.ssm_out.as_gpu(), &self.dout, &self.xb);
                }
            }
            self.elt(enc, &self.p_add, &self.x, Some(&self.xb), dim);
            // FFN
            let (fg, fu, fd) = match blk {
                Block::Attn(w) => (&w.ffn_gate, &w.ffn_up, &w.ffn_down),
                Block::Lin(w) => (&w.ffn_gate, &w.ffn_up, &w.ffn_down),
            };
            self.rms(enc, &self.x, &self.post_norm[il], &self.xb, dim, c.rms_eps);
            self.mv(enc, fg.as_gpu(), &self.xb, &self.gbuf);
            self.mv(enc, fu.as_gpu(), &self.xb, &self.ubuf);
            self.elt(enc, &self.p_swiglu, &self.gbuf, Some(&self.ubuf), c.ffn_dim);
            self.mv(enc, fd.as_gpu(), &self.gbuf, &self.xb);
            self.elt(enc, &self.p_add, &self.x, Some(&self.xb), dim);
            let _ = f4;
        }
        self.rms(enc, &self.x, &self.output_norm, &self.xb, dim, c.rms_eps);
        self.mv(enc, m.output.as_gpu(), &self.xb, &self.logits);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        unsafe { std::slice::from_raw_parts(self.logits.contents() as *const f32, c.vocab) }
            .to_vec()
    }
}

#[cfg(target_os = "macos")]
fn block_kinds_from(m: &Qwen35) -> Vec<BlockKind> {
    m.blocks
        .iter()
        .map(|b| match b {
            Block::Attn(_) => BlockKind::Attention,
            Block::Lin(_) => BlockKind::Ssm,
        })
        .collect()
}
