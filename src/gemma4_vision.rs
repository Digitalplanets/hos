//! Gemma 4 12B "unified" — IMAGE path: the ENCODER-FREE `vision_embedder`.
//!
//! STRICTLY ADDITIVE. A new HOS module (mirrors the `gemma4.rs` pattern) that
//! loads the ~10 vision tensors straight from the same `model.safetensors` and
//! implements the exact per-patch forward from `models/gemma4/VISION_SPEC.md`:
//!
//!   x = LayerNorm(patch_ln1, eps=1e-5)                 # over 6912
//!   x = Linear(patch_dense[3840,6912], +bias)          # 6912 -> 3840
//!   x = LayerNorm(patch_ln2, eps=1e-5)
//!   pos = pos_embedding[C,0,:] + pos_embedding[R,1,:]  # factorized 2D
//!   x = x + pos
//!   x = LayerNorm(pos_norm, eps=1e-5)
//!   x = RMSNorm(no scale, eps=1e-6, fp32)              # Gemma4RMSNorm with_scale=false
//!   x = Linear(embedding_projection[3840,3840], no bias)
//!   -> soft tokens [N,3840]
//!
//! There is NO ViT tower — patches (48x48x3 = 6912 flattened, channel-innermost)
//! go straight through this projector. The reference ran in fp32; this module is
//! pure f32 (bf16 weights are widened losslessly on load).

use std::path::Path;
use std::process::Command;

use crate::error::{HosError, Result};
use crate::model::Weight;
use crate::safetensors::SafeTensors;

pub const SIDE: usize = 48; // pooling(3) * patch(16)
pub const MAX_SOFT_TOKENS: usize = 280;
/// budget_px = max_soft_tokens(280) * pooling(3)^2 * patch(16)^2 = 645120.
pub const BUDGET_PX: usize = MAX_SOFT_TOKENS * 9 * 256;

/// VIDEO path: the Gemma4 video processor uses `max_soft_tokens = 70` (NOT the
/// image path's 280), so the per-frame pixel budget is 4x smaller.
pub const VIDEO_MAX_SOFT_TOKENS: usize = 70;
/// budget_px = 70 * pooling(3)^2 * patch(16)^2 = 161280.
pub const VIDEO_BUDGET_PX: usize = VIDEO_MAX_SOFT_TOKENS * 9 * 256;

pub const PATCH_DIM: usize = 6912; // 48*48*3
pub const HIDDEN: usize = 3840;
pub const POS_EMB_LN_EPS: f32 = 1e-5;
pub const RMS_EPS: f32 = 1e-6;

/// Loaded vision_embedder weights (all f32).
pub struct VisionEmbedder {
    patch_ln1_w: Vec<f32>, // [6912]
    patch_ln1_b: Vec<f32>,
    patch_dense: Weight, // [3840, 6912]
    patch_dense_b: Vec<f32>,
    patch_ln2_w: Vec<f32>, // [3840]
    patch_ln2_b: Vec<f32>,
    pos_embedding: Vec<f32>, // [1120, 2, 3840] row-major
    pos_emb_size: usize,     // 1120
    pos_norm_w: Vec<f32>,    // [3840]
    pos_norm_b: Vec<f32>,
    proj: Weight, // embedding_projection [3840, 3840], no bias
}

/// Standard LayerNorm over the whole vector (population variance, affine).
fn layernorm(x: &[f32], w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mean = x.iter().sum::<f32>() / n as f32;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv = (var + eps).sqrt().recip();
    (0..n).map(|i| (x[i] - mean) * inv * w[i] + b[i]).collect()
}

/// Scale-free RMSNorm (Gemma4RMSNorm with_scale=false): just normalize.
fn rmsnorm_noscale(x: &mut [f32], eps: f32) {
    let n = x.len();
    let ms = x.iter().map(|v| v * v).sum::<f32>() / n as f32 + eps;
    let s = ms.sqrt().recip();
    for v in x.iter_mut() {
        *v *= s;
    }
}

fn matvec(w: &Weight, x: &[f32], out_dim: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; out_dim];
    w.matvec(None, x, &mut y);
    y
}

impl VisionEmbedder {
    pub fn load(dir: &Path) -> Result<VisionEmbedder> {
        let st = SafeTensors::open_dir(dir)?;
        Self::load_from(&st)
    }

    /// Load from an already-open SafeTensors (shares the mmap with the decoder).
    pub fn load_from(st: &SafeTensors) -> Result<VisionEmbedder> {
        let p = "model.vision_embedder.";
        let pos_embedding = st.to_f32(&format!("{p}pos_embedding"))?;
        let pos_shape = st.shape(&format!("{p}pos_embedding"))?;
        let pos_emb_size = pos_shape[0];
        eprintln!(
            "[gemma4-vision] loading vision_embedder (bf16->f32); pos_embedding {:?}",
            pos_shape
        );
        Ok(VisionEmbedder {
            patch_ln1_w: st.to_f32(&format!("{p}patch_ln1.weight"))?,
            patch_ln1_b: st.to_f32(&format!("{p}patch_ln1.bias"))?,
            patch_dense: Weight::cpu(st.to_f32(&format!("{p}patch_dense.weight"))?, PATCH_DIM),
            patch_dense_b: st.to_f32(&format!("{p}patch_dense.bias"))?,
            patch_ln2_w: st.to_f32(&format!("{p}patch_ln2.weight"))?,
            patch_ln2_b: st.to_f32(&format!("{p}patch_ln2.bias"))?,
            pos_embedding,
            pos_emb_size,
            pos_norm_w: st.to_f32(&format!("{p}pos_norm.weight"))?,
            pos_norm_b: st.to_f32(&format!("{p}pos_norm.bias"))?,
            proj: Weight::cpu(
                st.to_f32("model.embed_vision.embedding_projection.weight")?,
                HIDDEN,
            ),
        })
    }

    /// `pos_embedding[idx, axis, :]` where axis in {0,1}. Flat layout [size,2,3840].
    #[inline]
    fn pos_row(&self, idx: usize, axis: usize) -> &[f32] {
        let base = (idx * 2 + axis) * HIDDEN;
        &self.pos_embedding[base..base + HIDDEN]
    }

    /// Embed `n` patches. `pixel_values` is `[n, 6912]` row-major (f32, [0,1]),
    /// `position_ids[i] = (x=col=C, y=row=R)`. Returns `[n, 3840]` soft tokens.
    pub fn embed_patches(
        &self,
        pixel_values: &[f32],
        position_ids: &[(i32, i32)],
        n: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; n * HIDDEN];
        for i in 0..n {
            let patch = &pixel_values[i * PATCH_DIM..(i + 1) * PATCH_DIM];
            // 1) LN over 6912
            let x = layernorm(patch, &self.patch_ln1_w, &self.patch_ln1_b, POS_EMB_LN_EPS);
            // 2) Linear 6912 -> 3840 (+bias)
            let mut x = matvec(&self.patch_dense, &x, HIDDEN);
            for j in 0..HIDDEN {
                x[j] += self.patch_dense_b[j];
            }
            // 3) LN over 3840
            let mut x = layernorm(&x, &self.patch_ln2_w, &self.patch_ln2_b, POS_EMB_LN_EPS);
            // 4) factorized 2D positional embedding (x=col axis0, y=row axis1)
            let (cx, ry) = position_ids[i];
            let c = (cx.max(0) as usize).min(self.pos_emb_size - 1);
            let r = (ry.max(0) as usize).min(self.pos_emb_size - 1);
            let px = self.pos_row(c, 0);
            let py = self.pos_row(r, 1);
            for j in 0..HIDDEN {
                x[j] += px[j] + py[j];
            }
            // 5) LN(pos_norm)
            let mut x = layernorm(&x, &self.pos_norm_w, &self.pos_norm_b, POS_EMB_LN_EPS);
            // 6) scale-free RMSNorm (fp32, eps 1e-6)
            rmsnorm_noscale(&mut x, RMS_EPS);
            // 7) Linear 3840 -> 3840 (no bias)
            let y = matvec(&self.proj, &x, HIDDEN);
            out[i * HIDDEN..(i + 1) * HIDDEN].copy_from_slice(&y);
        }
        out
    }
}

// ======================================================================
// Native image preprocessing (raw PNG/JPG -> pixel_values [N,6912])
// ======================================================================

/// Probe an image's (width, height) via ffprobe.
fn probe_image_dims(path: &Path) -> Result<(usize, usize)> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .map_err(|e| HosError::Format(format!("image: ffprobe launch failed ({e})")))?;
    if !out.status.success() {
        return Err(HosError::Format(format!(
            "image: ffprobe failed on {}: {}",
            path.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    let txt = String::from_utf8_lossy(&out.stdout);
    let field = |key: &str| -> Option<usize> {
        let pat = format!("\"{key}\"");
        let ki = txt.find(&pat)?;
        let rest = &txt[ki + pat.len()..];
        let colon = rest.find(':')?;
        let after = rest[colon + 1..].trim_start();
        let end = after.find([',', '\n', '}']).unwrap_or(after.len());
        after[..end].trim().parse().ok()
    };
    let w = field("width").ok_or_else(|| HosError::Format("image: ffprobe no width".into()))?;
    let h = field("height").ok_or_else(|| HosError::Format("image: ffprobe no height".into()))?;
    Ok((w, h))
}

/// Decode an image to native-resolution RGB HWC u8 via ffmpeg.
fn decode_image_rgb(path: &Path) -> Result<(Vec<u8>, usize, usize)> {
    if !path.exists() {
        return Err(HosError::Format(format!(
            "image: not found: {}",
            path.display()
        )));
    }
    let (w, h) = probe_image_dims(path)?;
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .output()
        .map_err(|e| HosError::Format(format!("image: ffmpeg launch failed ({e})")))?;
    if !out.status.success() {
        return Err(HosError::Format(format!(
            "image: ffmpeg failed on {}: {}",
            path.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    let mut buf = out.stdout;
    if buf.len() < w * h * 3 {
        return Err(HosError::Format(format!(
            "image: ffmpeg returned {} bytes, need {} ({}x{}x3)",
            buf.len(),
            w * h * 3,
            w,
            h
        )));
    }
    buf.truncate(w * h * 3);
    Ok((buf, w, h))
}

/// Cubic convolution kernel (Keys, a = -0.5 — the PIL/torchvision default).
fn cubic(x: f32) -> f32 {
    const A: f32 = -0.5;
    let x = x.abs();
    if x <= 1.0 {
        (A + 2.0) * x * x * x - (A + 3.0) * x * x + 1.0
    } else if x < 2.0 {
        A * x * x * x - 5.0 * A * x * x + 8.0 * A * x - 4.0 * A
    } else {
        0.0
    }
}

/// Bicubic resize (half-pixel centers, edge-clamped, NO antialias) of an RGB HWC
/// u8 image `(w,h)` -> f32 RGB HWC `(out_w,out_h)` in [0,255]. Antialias is off,
/// so downscaling drifts vs torchvision `resample=BICUBIC, antialias=True` — the
/// known Stage-3 numeric gap.
fn bicubic_resize(frame: &[u8], w: usize, h: usize, out_w: usize, out_h: usize) -> Vec<f32> {
    let sx = w as f32 / out_w as f32;
    let sy = h as f32 / out_h as f32;
    let clampi = |v: i64, hi: usize| -> usize { v.clamp(0, hi as i64 - 1) as usize };
    let mut out = vec![0f32; out_w * out_h * 3];
    for oy in 0..out_h {
        let fy = (oy as f32 + 0.5) * sy - 0.5;
        let iy = fy.floor() as i64;
        let dy = fy - iy as f32;
        let wy = [cubic(dy + 1.0), cubic(dy), cubic(dy - 1.0), cubic(dy - 2.0)];
        for ox in 0..out_w {
            let fx = (ox as f32 + 0.5) * sx - 0.5;
            let ix = fx.floor() as i64;
            let dx = fx - ix as f32;
            let wx = [cubic(dx + 1.0), cubic(dx), cubic(dx - 1.0), cubic(dx - 2.0)];
            for c in 0..3 {
                let mut acc = 0f32;
                for m in 0..4 {
                    let yy = clampi(iy - 1 + m as i64, h);
                    let mut row = 0f32;
                    for nn in 0..4 {
                        let xx = clampi(ix - 1 + nn as i64, w);
                        row += wx[nn] * frame[(yy * w + xx) * 3 + c] as f32;
                    }
                    acc += wy[m] * row;
                }
                out[(oy * out_w + ox) * 3 + c] = acc;
            }
        }
    }
    out
}

/// Patchify a raw native-resolution RGB HWC u8 frame into
/// `(pixel_values[N,6912], position_ids[N], gh, gw)` under `budget_px`:
/// aspect-preserving bicubic resize to (Gh*48, Gw*48), /255, then 48x48x3 blocks
/// flattened [row,col,channel] (channel innermost), positions (x=col, y=row).
/// Shared by the IMAGE (`BUDGET_PX`) and VIDEO (`VIDEO_BUDGET_PX`) paths — the
/// only per-path difference is the pixel budget. Layout verified bit-identical
/// to the HF `convert_video_to_patches`+`patches_merge` model-patch ordering.
pub fn patchify_rgb_frame(
    frame: &[u8],
    w: usize,
    h: usize,
    budget_px: usize,
) -> (Vec<f32>, Vec<(i32, i32)>, usize, usize) {
    let factor = ((budget_px as f32) / (h as f32 * w as f32)).sqrt();
    let mut th = ((factor * h as f32 / SIDE as f32).floor() as usize) * SIDE;
    let mut tw = ((factor * w as f32 / SIDE as f32).floor() as usize) * SIDE;
    if th == 0 {
        th = SIDE;
    }
    if tw == 0 {
        tw = SIDE;
    }
    let gh = th / SIDE;
    let gw = tw / SIDE;
    let n = gh * gw;
    // Bicubic resize to (th, tw), values in [0,255].
    let img = bicubic_resize(frame, w, h, tw, th);
    let mut pixel_values = vec![0f32; n * PATCH_DIM];
    let mut pos = Vec::with_capacity(n);
    for r in 0..gh {
        for cc in 0..gw {
            let pi = r * gw + cc; // patch index (row-major)
            let base = pi * PATCH_DIM;
            for row in 0..SIDE {
                for col in 0..SIDE {
                    let sy = r * SIDE + row;
                    let sx = cc * SIDE + col;
                    for ch in 0..3 {
                        let v = img[(sy * tw + sx) * 3 + ch] / 255.0;
                        // flatten [row, col, channel]  (channel innermost)
                        pixel_values[base + (row * SIDE + col) * 3 + ch] = v.clamp(0.0, 1.0);
                    }
                }
            }
            pos.push((cc as i32, r as i32)); // (x=col, y=row)
        }
    }
    (pixel_values, pos, gh, gw)
}

/// Full native preprocessing: raw image path -> `(pixel_values[N,6912],
/// position_ids[N], gh, gw)`. Aspect-preserving resize to (Gh*48, Gw*48) under
/// the 645120px budget, /255, then 48x48x3 blocks flattened [row,col,channel].
pub fn preprocess_image(path: &Path) -> Result<(Vec<f32>, Vec<(i32, i32)>, usize, usize)> {
    preprocess_image_budget(path, BUDGET_PX)
}

/// Budget-parametrized `preprocess_image`: same pipeline, but the caller picks the
/// pixel budget (e.g. `VIDEO_BUDGET_PX` for a faster, coarser ~70-token grid used
/// by the constrained classifier's fast per-clip loop). Additive — the default
/// `preprocess_image` still uses the full `BUDGET_PX` (280-token) grid.
pub fn preprocess_image_budget(
    path: &Path,
    budget_px: usize,
) -> Result<(Vec<f32>, Vec<(i32, i32)>, usize, usize)> {
    let (frame, w, h) = decode_image_rgb(path)?;
    let (pixel_values, pos, gh, gw) = patchify_rgb_frame(&frame, w, h, budget_px);
    eprintln!(
        "[gemma4-vision] preprocess {}: native {}x{} -> resize {}x{} (Gh={} Gw={}, N={})",
        path.display(),
        w,
        h,
        gw * SIDE,
        gh * SIDE,
        gh,
        gw,
        gh * gw
    );
    Ok((pixel_values, pos, gh, gw))
}
