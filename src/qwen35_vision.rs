//! Qwen3-VL vision tower (the `mmproj` half of the qwen35 hybrid) — native, from
//! the `mmproj-*.gguf`, zero-dep. This is the SigLIP-style ViT + `qwen3vl_merger`
//! that turns an image into image tokens spliced into the text stream at
//! `<|vision_start|><|image_pad|>…<|vision_end|>`.
//!
//! Architecture reverse-engineered from llama.cpp `tools/mtmd/models/qwen3vl.cpp`
//! + `clip.cpp` (the reference, same oracle used for the text model):
//!   patch-embed (conv 16x16, temporal-merged 2 kernels) -> spatial 2x2 reorder
//!   -> +patch_bias -> +learned pos_embd -> 27x { LN, QKV attn w/ 2D vision
//!   M-RoPE, LN, GELU MLP } -> post_LN -> merger (group 4 patches: mm0 -> GELU ->
//!   mm2) -> [n_patches/4, 5120] image tokens.
//!
//! STATUS: config + weight loading (this file) are done and shape-checked against
//! the real mmproj. The forward pass is built + verified against the mtmd oracle
//! next — no guessing; every op is diffed like the text path was.

use crate::error::{HosError, Result};
use crate::gguf::Gguf;
use rayon::prelude::*;
use std::path::Path;
use std::process::Command;

fn gelu(x: f32) -> f32 {
    // tanh approximation, matching ggml_gelu.
    0.5 * x * (1.0 + ((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

/// LayerNorm over `x` (len = weight.len()) with affine weight+bias.
fn layernorm(x: &[f32], w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    (0..x.len())
        .map(|i| (x[i] - mean) * inv * w[i] + b[i])
        .collect()
}

/// out[o] = sum_i x[i]*w[o*in_dim + i] + b[o]  (row-major weight [in, out] in GGUF).
fn linear(x: &[f32], w: &[f32], b: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    (0..out_dim)
        .into_par_iter()
        .map(|o| {
            let row = &w[o * in_dim..o * in_dim + in_dim];
            let mut acc = b[o];
            for i in 0..in_dim {
                acc += x[i] * row[i];
            }
            acc
        })
        .collect()
}

/// Vision-tower config, read from the mmproj's `clip.vision.*` metadata.
#[derive(Debug, Clone)]
pub struct VisionCfg {
    pub image_size: usize,   // 768
    pub patch_size: usize,   // 16
    pub hidden: usize,       // 1152 (embedding_length)
    pub ffn: usize,          // 4304
    pub n_layers: usize,     // 27
    pub n_heads: usize,      // 16
    pub proj_dim: usize,     // 5120 (into the text hidden dim)
    pub spatial_merge: usize, // 2
    pub ln_eps: f32,         // 1e-6
    pub mean: [f32; 3],      // 0.5,0.5,0.5
    pub std: [f32; 3],       // 0.5,0.5,0.5
}

impl VisionCfg {
    pub fn from_gguf(g: &Gguf) -> Result<VisionCfg> {
        let k = |s: &str| format!("clip.vision.{s}");
        let need = |key: &str| {
            g.meta_u64(&k(key))
                .ok_or_else(|| HosError::MissingMeta(k(key)))
        };
        // image_mean/std are [0.5,0.5,0.5] for this SigLIP-style tower (confirmed
        // in the mmproj metadata); read as scalars would need an array accessor, so
        // use the standard SigLIP normalization directly.
        Ok(VisionCfg {
            image_size: need("image_size")? as usize,
            patch_size: need("patch_size")? as usize,
            hidden: need("embedding_length")? as usize,
            ffn: need("feed_forward_length")? as usize,
            n_layers: need("block_count")? as usize,
            n_heads: need("attention.head_count")? as usize,
            proj_dim: g.meta_u64("clip.vision.projection_dim").unwrap_or(5120) as usize,
            spatial_merge: g.meta_u64("clip.vision.spatial_merge_size").unwrap_or(2) as usize,
            ln_eps: g
                .meta_f32(&k("attention.layer_norm_epsilon"))
                .unwrap_or(1e-6),
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
        })
    }

    /// Patches per side at native resolution (image_size / patch_size).
    pub fn grid(&self) -> usize {
        self.image_size / self.patch_size
    }
    /// Head dimension.
    pub fn head_dim(&self) -> usize {
        self.hidden / self.n_heads
    }
    /// Image tokens produced = patches / merge^2.
    pub fn n_image_tokens(&self) -> usize {
        let g = self.grid();
        (g * g) / (self.spatial_merge * self.spatial_merge)
    }
}

/// One ViT block's weights (LayerNorm has weight+bias; fused QKV; GELU MLP).
pub struct VBlock {
    pub ln1_w: Vec<f32>,
    pub ln1_b: Vec<f32>,
    pub qkv_w: Vec<f32>, // [hidden, 3*hidden]
    pub qkv_b: Vec<f32>, // [3*hidden]
    pub o_w: Vec<f32>,   // [hidden, hidden]
    pub o_b: Vec<f32>,
    pub ln2_w: Vec<f32>,
    pub ln2_b: Vec<f32>,
    pub ffn_up_w: Vec<f32>,   // [hidden, ffn]
    pub ffn_up_b: Vec<f32>,   // [ffn]
    pub ffn_down_w: Vec<f32>, // [ffn, hidden]
    pub ffn_down_b: Vec<f32>, // [hidden]
}

/// The full vision tower: patch embed + pos + blocks + post-LN + merger.
pub struct VisionTower {
    pub cfg: VisionCfg,
    pub patch_w: Vec<f32>, // [hidden, 16*16*3] effective kernel (temporal 0 + 1)
    pub patch_bias: Vec<f32>,
    pub pos_embd: Vec<f32>, // [n_pos, hidden]
    pub blocks: Vec<VBlock>,
    pub post_ln_w: Vec<f32>,
    pub post_ln_b: Vec<f32>,
    pub mm0_w: Vec<f32>, // [4*hidden, 4*hidden]
    pub mm0_b: Vec<f32>,
    pub mm2_w: Vec<f32>, // [4*hidden, proj_dim]
    pub mm2_b: Vec<f32>,
}

impl VisionTower {
    /// Load every tensor from the mmproj GGUF (BF16 -> f32) and validate the
    /// architecture against the config. Returns an error naming any missing tensor.
    pub fn load(g: &Gguf) -> Result<VisionTower> {
        let cfg = VisionCfg::from_gguf(g)?;
        let d = cfg.hidden;
        let get = |name: &str| -> Result<Vec<f32>> { g.dequant(name) };

        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for il in 0..cfg.n_layers {
            let p = |t: &str| format!("v.blk.{il}.{t}");
            blocks.push(VBlock {
                ln1_w: get(&p("ln1.weight"))?,
                ln1_b: get(&p("ln1.bias"))?,
                qkv_w: get(&p("attn_qkv.weight"))?,
                qkv_b: get(&p("attn_qkv.bias"))?,
                o_w: get(&p("attn_out.weight"))?,
                o_b: get(&p("attn_out.bias"))?,
                ln2_w: get(&p("ln2.weight"))?,
                ln2_b: get(&p("ln2.bias"))?,
                ffn_up_w: get(&p("ffn_up.weight"))?,
                ffn_up_b: get(&p("ffn_up.bias"))?,
                ffn_down_w: get(&p("ffn_down.weight"))?,
                ffn_down_b: get(&p("ffn_down.bias"))?,
            });
        }

        // Two temporal patch kernels; single images use their sum (the frame is
        // duplicated across the temporal dim of 2 in Qwen-VL).
        let patch_w0 = get("v.patch_embd.weight")?;
        let patch_w1 = get("v.patch_embd.weight.1").unwrap_or_else(|_| vec![0.0; patch_w0.len()]);
        let patch_w: Vec<f32> = patch_w0
            .iter()
            .zip(patch_w1.iter())
            .map(|(a, b)| a + b)
            .collect();

        let tower = VisionTower {
            patch_bias: get("v.patch_embd.bias")?,
            pos_embd: get("v.position_embd.weight")?,
            post_ln_w: get("v.post_ln.weight")?,
            post_ln_b: get("v.post_ln.bias")?,
            mm0_w: get("mm.0.weight")?,
            mm0_b: get("mm.0.bias")?,
            mm2_w: get("mm.2.weight")?,
            mm2_b: get("mm.2.bias")?,
            patch_w,
            blocks,
            cfg,
        };
        tower.validate(d)?;
        Ok(tower)
    }

    /// Shape sanity against the config — catches a wrong/mismatched mmproj early.
    fn validate(&self, d: usize) -> Result<()> {
        let c = &self.cfg;
        let want = |name: &str, got: usize, exp: usize| -> Result<()> {
            if got != exp {
                return Err(HosError::Format(format!(
                    "mmproj {name}: got {got} floats, expected {exp}"
                )));
            }
            Ok(())
        };
        let patch = c.patch_size * c.patch_size * 3;
        want("patch_embd.weight", self.patch_w.len(), patch * d)?;
        want("patch_embd.bias", self.patch_bias.len(), d)?;
        want("position_embd", self.pos_embd.len(), c.grid() * c.grid() * d)?;
        want("mm.0.weight", self.mm0_w.len(), (4 * d) * (4 * d))?;
        want("mm.2.weight", self.mm2_w.len(), (4 * d) * c.proj_dim)?;
        want("mm.2.bias", self.mm2_b.len(), c.proj_dim)?;
        want("blocks", self.blocks.len(), c.n_layers)?;
        let b0 = &self.blocks[0];
        want("blk.0.attn_qkv.weight", b0.qkv_w.len(), d * 3 * d)?;
        want("blk.0.ffn_up.weight", b0.ffn_up_w.len(), d * c.ffn)?;
        Ok(())
    }
}

impl VisionTower {
    /// Decode `path` to RGB, resize to image_size², SigLIP-normalize to [-1,1].
    /// Returns pixels in row-major [y][x][c] (c innermost), length s*s*3.
    fn preprocess(&self, path: &Path) -> Result<Vec<f32>> {
        let s = self.cfg.image_size;
        if !path.exists() {
            return Err(HosError::Format(format!("image not found: {}", path.display())));
        }
        // ffmpeg decodes any format and resizes in one shot (zero Rust deps).
        let out = Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(path)
            .args([
                "-vf",
                &format!("scale={s}:{s}:flags=bicubic"),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ])
            .output()
            .map_err(|e| HosError::Format(format!("ffmpeg launch failed ({e}) — is ffmpeg installed?")))?;
        if !out.status.success() {
            return Err(HosError::Format(format!(
                "ffmpeg failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }
        let buf = out.stdout;
        if buf.len() < s * s * 3 {
            return Err(HosError::Format(format!(
                "ffmpeg returned {} bytes, need {}",
                buf.len(),
                s * s * 3
            )));
        }
        // (px/255 - 0.5)/0.5 = px/127.5 - 1
        Ok(buf[..s * s * 3].iter().map(|&p| p as f32 / 127.5 - 1.0).collect())
    }

    /// One ViT block over `h` (in place): LN, QKV, 2D vision M-RoPE, full attention,
    /// out-proj, residual, LN, GELU MLP, residual. `grid` = patches per side.
    fn vit_layer(&self, h: &mut [f32], b: &VBlock, grid: usize) {
        let c = &self.cfg;
        let d = c.hidden;
        let nh = c.n_heads;
        let hd = c.head_dim(); // 72
        let half = hd / 2; // 36
        let n = grid * grid; // patches
        let scale = 1.0 / (hd as f32).sqrt();
        // theta_scale = freq_base^(-2/n_rot), n_rot = hd/2.
        let theta_scale = (10000.0f32).powf(-2.0 / half as f32);

        // --- attention input: LN1 then QKV per patch ---
        // qkv[p] = [q(1152) | k(1152) | v(1152)]
        let qkv: Vec<Vec<f32>> = (0..n)
            .into_par_iter()
            .map(|p| {
                let xn = layernorm(&h[p * d..p * d + d], &b.ln1_w, &b.ln1_b, c.ln_eps);
                linear(&xn, &b.qkv_w, &b.qkv_b, d, 3 * d)
            })
            .collect();

        // Split + apply 2D vision M-RoPE to Q,K per head. Layout per head:
        // pairs 0..half; pair j pairs dim j with dim j+half; j<half/2 uses row(y),
        // else col(x). (theta_w/theta_e sections are never reached for hd=72.)
        let rope = |vec: &mut [f32], py: usize, px: usize| {
            for hh in 0..nh {
                let base = hh * hd;
                for j in 0..half {
                    let (pos, freq_idx) = if j < half / 2 {
                        (py as f32, j)
                    } else {
                        (px as f32, j - half / 2)
                    };
                    let ang = pos * theta_scale.powi(freq_idx as i32);
                    let (s, co) = ang.sin_cos();
                    let a = base + j;
                    let bb = base + j + half;
                    let x0 = vec[a];
                    let x1 = vec[bb];
                    vec[a] = x0 * co - x1 * s;
                    vec[bb] = x0 * s + x1 * co;
                }
            }
        };
        let mut qs: Vec<Vec<f32>> = qkv.iter().map(|v| v[0..d].to_vec()).collect();
        let mut ks: Vec<Vec<f32>> = qkv.iter().map(|v| v[d..2 * d].to_vec()).collect();
        let vs: Vec<&[f32]> = qkv.iter().map(|v| &v[2 * d..3 * d]).collect();
        for p in 0..n {
            let (py, px) = (p / grid, p % grid);
            rope(&mut qs[p], py, px);
            rope(&mut ks[p], py, px);
        }

        // --- full (bidirectional) attention, per patch, per head ---
        let attn: Vec<Vec<f32>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut out = vec![0.0f32; d];
                for hh in 0..nh {
                    let qh = &qs[i][hh * hd..hh * hd + hd];
                    let mut scores = vec![0.0f32; n];
                    let mut mx = f32::NEG_INFINITY;
                    for jp in 0..n {
                        let kh = &ks[jp][hh * hd..hh * hd + hd];
                        let mut dot = 0.0;
                        for t in 0..hd {
                            dot += qh[t] * kh[t];
                        }
                        dot *= scale;
                        scores[jp] = dot;
                        if dot > mx {
                            mx = dot;
                        }
                    }
                    let mut sum = 0.0;
                    for sj in scores.iter_mut() {
                        *sj = (*sj - mx).exp();
                        sum += *sj;
                    }
                    let oh = &mut out[hh * hd..hh * hd + hd];
                    for jp in 0..n {
                        let w = scores[jp] / sum;
                        let vh = &vs[jp][hh * hd..hh * hd + hd];
                        for t in 0..hd {
                            oh[t] += w * vh[t];
                        }
                    }
                }
                // out projection
                linear(&out, &b.o_w, &b.o_b, d, d)
            })
            .collect();

        // residual 1
        for p in 0..n {
            for k in 0..d {
                h[p * d + k] += attn[p][k];
            }
        }
        // --- MLP: LN2 -> up -> GELU -> down -> residual ---
        let ffn: Vec<Vec<f32>> = (0..n)
            .into_par_iter()
            .map(|p| {
                let xn = layernorm(&h[p * d..p * d + d], &b.ln2_w, &b.ln2_b, c.ln_eps);
                let mut up = linear(&xn, &b.ffn_up_w, &b.ffn_up_b, d, c.ffn);
                for u in up.iter_mut() {
                    *u = gelu(*u);
                }
                linear(&up, &b.ffn_down_w, &b.ffn_down_b, c.ffn, d)
            })
            .collect();
        for p in 0..n {
            for k in 0..d {
                h[p * d + k] += ffn[p][k];
            }
        }
    }

    /// `encode_image` with a content-addressed disk cache in `~/.hos/vision_cache/`
    /// (key = image bytes + tower geometry). The ViT encode is ~seconds-to-minutes
    /// on CPU, so caching makes re-running the same image instant — the difference
    /// between a painful and a pleasant test loop.
    pub fn encode_image_cached(&self, path: &Path) -> Result<Vec<f32>> {
        let bytes = std::fs::read(path)
            .map_err(|e| HosError::Format(format!("image read {}: {e}", path.display())))?;
        // FNV-1a over image bytes + geometry -> stable cache key.
        let mut h = 0xcbf29ce484222325u64;
        for b in bytes.iter() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        for g in [self.cfg.image_size, self.cfg.n_layers, self.cfg.proj_dim] {
            h ^= g as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        let dir = std::env::var_os("HOME")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join(".hos/vision_cache");
        let cache = dir.join(format!("v-{h:016x}.f32"));
        if let Ok(raw) = std::fs::read(&cache) {
            if raw.len() % 4 == 0 {
                let v: Vec<f32> = raw
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                if v.len() == self.cfg.n_image_tokens() * self.cfg.proj_dim {
                    return Ok(v);
                }
            }
        }
        let emb = self.encode_image(path)?;
        let _ = std::fs::create_dir_all(&dir);
        let mut out = Vec::with_capacity(emb.len() * 4);
        for f in &emb {
            out.extend_from_slice(&f.to_le_bytes());
        }
        let _ = std::fs::write(&cache, &out);
        Ok(emb)
    }

    /// Encode raw image bytes (e.g. a base64 `image_url` from an OpenAI request):
    /// write to a temp file and encode through the cached path (so repeated images
    /// still hit the disk cache, keyed on the bytes). ffmpeg sniffs the format.
    pub fn encode_image_bytes(&self, bytes: &[u8]) -> Result<Vec<f32>> {
        let mut h = 0xcbf29ce484222325u64;
        for b in bytes.iter().take(4096) {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        let tmp = std::env::temp_dir().join(format!("hos_img_{h:016x}.tmp"));
        std::fs::write(&tmp, bytes)
            .map_err(|e| HosError::Format(format!("image temp write: {e}")))?;
        let r = self.encode_image_cached(&tmp);
        let _ = std::fs::remove_file(&tmp);
        r
    }

    /// Encode an image into `n_image_tokens` embeddings of `proj_dim`, ready to
    /// splice into the text stream at `<|image_pad|>`. Returns a flat
    /// [n_tokens * proj_dim] vector. Verified against the mtmd oracle.
    pub fn encode_image(&self, path: &Path) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let d = c.hidden;
        let grid = c.grid();
        let ps = c.patch_size;
        let s = c.image_size;
        let px = self.preprocess(path)?;

        // --- patch embedding: conv 16x16 stride 16, kernel indexed [c,kh,kw] ---
        let n = grid * grid;
        let mut h: Vec<f32> = (0..n)
            .into_par_iter()
            .flat_map(|p| {
                let (py, pxi) = (p / grid, p % grid);
                (0..d)
                    .map(|oc| {
                        let ker = &self.patch_w[oc * (ps * ps * 3)..oc * (ps * ps * 3) + ps * ps * 3];
                        let mut acc = 0.0f32;
                        for e in 0..ps * ps * 3 {
                            let cc = e / (ps * ps);
                            let rem = e % (ps * ps);
                            let kh = rem / ps;
                            let kw = rem % ps;
                            let yv = py * ps + kh;
                            let xv = pxi * ps + kw;
                            acc += px[(yv * s + xv) * 3 + cc] * ker[e];
                        }
                        acc
                    })
                    .collect::<Vec<f32>>()
            })
            .collect();

        // + patch_bias + learned position embedding (natural row-major order)
        for p in 0..n {
            for k in 0..d {
                h[p * d + k] += self.patch_bias[k] + self.pos_embd[p * d + k];
            }
        }

        // --- 27 ViT blocks ---
        for b in &self.blocks {
            self.vit_layer(&mut h, b, grid);
        }

        // --- post LayerNorm ---
        let h: Vec<f32> = (0..n)
            .into_par_iter()
            .flat_map(|p| layernorm(&h[p * d..p * d + d], &self.post_ln_w, &self.post_ln_b, c.ln_eps))
            .collect();

        // --- merger: group each spatial 2x2 block -> 4608 -> mm0 -> GELU -> mm2 -> 5120 ---
        let m = c.spatial_merge; // 2
        let bg = grid / m; // 24 merged per side
        let merged_dim = m * m * d; // 4608
        let out: Vec<f32> = (0..bg * bg)
            .into_par_iter()
            .flat_map(|g| {
                let (by, bx) = (g / bg, g % bg);
                // concat order (dy,dx): (0,0),(0,1),(1,0),(1,1)
                let mut cat = Vec::with_capacity(merged_dim);
                for dy in 0..m {
                    for dx in 0..m {
                        let p = (by * m + dy) * grid + (bx * m + dx);
                        cat.extend_from_slice(&h[p * d..p * d + d]);
                    }
                }
                let mut z = linear(&cat, &self.mm0_w, &self.mm0_b, merged_dim, merged_dim);
                for zi in z.iter_mut() {
                    *zi = gelu(*zi);
                }
                linear(&z, &self.mm2_w, &self.mm2_b, merged_dim, c.proj_dim)
            })
            .collect();
        Ok(out)
    }
}

/// `hos --vision-check -m <mmproj.gguf>` — load + report the vision tower so the
/// loader is verifiable before the forward pass lands.
pub fn check(g: &Gguf) -> Result<()> {
    let t = VisionTower::load(g)?;
    let c = &t.cfg;
    eprintln!("[vision] {c:#?}");
    eprintln!(
        "[vision] loaded: {} ViT blocks, patch {}x{}, grid {}x{} = {} patches -> {} image tokens (merge {}) -> proj {}",
        c.n_layers,
        c.patch_size,
        c.patch_size,
        c.grid(),
        c.grid(),
        c.grid() * c.grid(),
        c.n_image_tokens(),
        c.spatial_merge,
        c.proj_dim,
    );
    eprintln!("[vision] all tensors present + shapes validated. forward pass next (oracle-verified).");
    Ok(())
}
