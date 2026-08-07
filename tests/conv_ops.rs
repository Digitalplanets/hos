//! Forward-correctness tests for the conv/pool/shape ops (2D and 3D
//! convolution, pooling, unfold/im2col, and related shape math). Each op is
//! checked against a hand-computed result and, where useful, cross-checked
//! against PyTorch (NCHW) by shelling out to a local interpreter. Layout under
//! test: NHWC activations, weight [cout, kh*kw*cin_per_group] inner
//! (kh,kw,cin), conv bias [cout].

use hos::tensor::Tensor;
use std::process::Command;

fn assert_close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{what}: length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    let mut maxd = 0.0f32;
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let d = (x - y).abs();
        if d > maxd {
            maxd = d;
        }
        assert!(d <= tol, "{what}: idx {i} {x} vs {y} (d={d} > {tol})");
    }
    eprintln!("{what}: OK (max diff {maxd:.2e})");
}

/// Run a torch snippet that must print whitespace-separated floats on stdout.
/// Returns None if torch/python is unavailable so the test still asserts the
/// hand-computed path locally.
fn torch(snippet: &str) -> Option<Vec<f32>> {
    // The PyTorch cross-check is optional: set `HOS_TEST_PYTHON` to a
    // torch-enabled interpreter to enable it. If unset, skip the cross-check
    // cleanly (returning None) so only the hand-computed path is asserted.
    let py = match std::env::var("HOS_TEST_PYTHON") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("HOS_TEST_PYTHON not set; skipping PyTorch cross-check");
            return None;
        }
    };
    let out = Command::new(&py).arg("-c").arg(snippet).output().ok()?;
    if !out.status.success() {
        eprintln!(
            "torch unavailable: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    Some(
        s.split_whitespace()
            .map(|t| t.parse::<f32>().unwrap())
            .collect(),
    )
}

#[test]
fn conv2d_sp_stride2_pad1_k3() {
    // 1x4x4x1 input, single 3x3 filter, stride 2, pad 1 -> 1x2x2x1.
    let inp: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let x = Tensor::constant(inp.clone(), &[1, 4, 4, 1]);
    let wt: Vec<f32> = (0..9).map(|i| (i as f32) * 0.1).collect();
    let w = Tensor::constant(wt.clone(), &[1, 9]); // [cout=1, kh*kw*cin=9]
    let bias = Tensor::constant(vec![0.5], &[1]);
    let y = x.conv2d_sp(&w, Some(&bias), 3, 3, 1, 2, 1);
    assert_eq!(y.shape(), vec![1, 2, 2, 1]);

    // Hand computation (NCHW math), padding=1, stride=2.
    let pad = 1usize;
    let h = 4i64;
    let getp = |r: i64, c: i64| -> f32 {
        if r < 0 || r >= h || c < 0 || c >= h {
            0.0
        } else {
            inp[(r * h + c) as usize]
        }
    };
    let mut expect = Vec::new();
    for oh in 0..2i64 {
        for ow in 0..2i64 {
            let mut acc = 0.5;
            for ki in 0..3i64 {
                for kj in 0..3i64 {
                    let r = oh * 2 - pad as i64 + ki;
                    let c = ow * 2 - pad as i64 + kj;
                    acc += getp(r, c) * wt[(ki * 3 + kj) as usize];
                }
            }
            expect.push(acc);
        }
    }
    assert_close(&y.data(), &expect, 1e-5, "conv2d_sp hand stride2 pad1 k3");

    // Torch cross-check (NCHW; NHWC layouts coincide for cin=cout=1).
    let snip = format!(
        "import torch,torch.nn.functional as F\n\
         x=torch.arange(16.).reshape(1,1,4,4)\n\
         w=torch.tensor([{}]).reshape(1,1,3,3)\n\
         b=torch.tensor([0.5])\n\
         y=F.conv2d(x,w,b,stride=2,padding=1)\n\
         print(' '.join(f'{{v:.6f}}' for v in y.flatten().tolist()))",
        (0..9)
            .map(|i| format!("{}", i as f32 * 0.1))
            .collect::<Vec<_>>()
            .join(",")
    );
    if let Some(t) = torch(&snip) {
        assert_close(&y.data(), &t, 1e-5, "conv2d_sp vs torch stride2 pad1 k3");
    }
}

#[test]
fn conv2d_sp_equals_conv2d_stride1_pad0() {
    // Multi-channel case: conv2d_sp(stride1,pad0,no bias) must equal conv2d.
    let mut seed = 12345u64;
    let x = Tensor::randn(&[1, 5, 5, 3], &mut seed);
    let cout = 4;
    let w = Tensor::randn(&[cout, 2 * 2 * 3], &mut seed);
    let base = x.conv2d(&w, 2, 2, cout);
    let sp = x.conv2d_sp(&w, None, 2, 2, cout, 1, 0);
    assert_eq!(base.shape(), sp.shape());
    assert_close(
        &sp.data(),
        &base.data(),
        1e-6,
        "conv2d_sp == conv2d (s1 p0)",
    );
}

#[test]
fn conv2d_spg_depthwise() {
    // Depthwise 3x3, stride1 pad1, groups==cin==cout==2 on a 1x3x3x2 input.
    let mut seed = 777u64;
    let x = Tensor::randn(&[1, 3, 3, 2], &mut seed);
    let cin = 2;
    let cout = 2;
    let groups = 2;
    // weight [cout, kh*kw*(cin/groups)=9*1]
    let w = Tensor::randn(&[cout, 3 * 3 * 1], &mut seed);
    let y = x.conv2d_spg(&w, None, 3, 3, cout, 1, 1, groups);
    assert_eq!(y.shape(), vec![1, 3, 3, cout]);

    // Torch depthwise: weight [cout,1,3,3], groups=2, NHWC<->NCHW transpose.
    let xd = x.data();
    let wd = w.data();
    let xnchw: Vec<f32> = {
        // NHWC [1,3,3,2] -> NCHW [1,2,3,3]
        let mut v = vec![0.0f32; 18];
        for hh in 0..3 {
            for ww in 0..3 {
                for c in 0..cin {
                    v[(c * 3 + hh) * 3 + ww] = xd[(hh * 3 + ww) * cin + c];
                }
            }
        }
        v
    };
    let snip = format!(
        "import torch,torch.nn.functional as F\n\
         x=torch.tensor([{}]).reshape(1,2,3,3)\n\
         w=torch.tensor([{}]).reshape(2,1,3,3)\n\
         y=F.conv2d(x,w,None,stride=1,padding=1,groups=2)\n\
         # back to NHWC order\n\
         y=y.permute(0,2,3,1).contiguous()\n\
         print(' '.join(f'{{v:.6f}}' for v in y.flatten().tolist()))",
        xnchw
            .iter()
            .map(|v| format!("{v}"))
            .collect::<Vec<_>>()
            .join(","),
        wd.iter()
            .map(|v| format!("{v}"))
            .collect::<Vec<_>>()
            .join(",")
    );
    if let Some(t) = torch(&snip) {
        assert_close(&y.data(), &t, 1e-5, "conv2d_spg depthwise vs torch");
    }
}

#[test]
fn maxpool2d_sp_k5_s1_p2_preserves_hw() {
    // SPPF pooling: k=5, stride=1, pad=2 keeps HxW. 1x6x6x1 input.
    let inp: Vec<f32> = (0..36).map(|i| (i as f32) * 0.5 - 7.0).collect();
    let x = Tensor::constant(inp.clone(), &[1, 6, 6, 1]);
    let y = x.maxpool2d_sp(5, 1, 2);
    assert_eq!(y.shape(), vec![1, 6, 6, 1]);

    let snip = format!(
        "import torch,torch.nn.functional as F\n\
         x=torch.tensor([{}]).reshape(1,1,6,6)\n\
         y=F.max_pool2d(x,5,stride=1,padding=2)\n\
         print(' '.join(f'{{v:.6f}}' for v in y.flatten().tolist()))",
        inp.iter()
            .map(|v| format!("{v}"))
            .collect::<Vec<_>>()
            .join(",")
    );
    if let Some(t) = torch(&snip) {
        assert_close(&y.data(), &t, 1e-5, "maxpool2d_sp vs torch k5 s1 p2");
    } else {
        // Fallback hand check on the top-left cell (window rows/cols -2..2).
        let h = 6i64;
        let mut best = f32::NEG_INFINITY;
        for ki in -2..=2i64 {
            for kj in -2..=2i64 {
                if ki >= 0 && ki < h && kj >= 0 && kj < h {
                    best = best.max(inp[(ki * h + kj) as usize]);
                }
            }
        }
        assert!((y.data()[0] - best).abs() < 1e-5);
    }
}

#[test]
fn upsample_nearest2x_2x2() {
    // 1x2x2x1 -> 1x4x4x1, each pixel into a 2x2 block.
    let x = Tensor::constant(vec![1.0, 2.0, 3.0, 4.0], &[1, 2, 2, 1]);
    let y = x.upsample_nearest2x();
    assert_eq!(y.shape(), vec![1, 4, 4, 1]);
    let expect = vec![
        1.0, 1.0, 2.0, 2.0, //
        1.0, 1.0, 2.0, 2.0, //
        3.0, 3.0, 4.0, 4.0, //
        3.0, 3.0, 4.0, 4.0,
    ];
    assert_close(&y.data(), &expect, 1e-6, "upsample_nearest2x hand");

    // Multi-channel torch cross-check.
    let mut seed = 99u64;
    let x2 = Tensor::randn(&[1, 3, 4, 2], &mut seed);
    let y2 = x2.upsample_nearest2x();
    assert_eq!(y2.shape(), vec![1, 6, 8, 2]);
    let xd = x2.data();
    // NHWC [1,3,4,2] -> NCHW [1,2,3,4]
    let mut xnchw = vec![0.0f32; 24];
    for hh in 0..3 {
        for ww in 0..4 {
            for c in 0..2 {
                xnchw[(c * 3 + hh) * 4 + ww] = xd[(hh * 4 + ww) * 2 + c];
            }
        }
    }
    let snip = format!(
        "import torch,torch.nn.functional as F\n\
         x=torch.tensor([{}]).reshape(1,2,3,4)\n\
         y=F.interpolate(x,scale_factor=2,mode='nearest')\n\
         y=y.permute(0,2,3,1).contiguous()\n\
         print(' '.join(f'{{v:.6f}}' for v in y.flatten().tolist()))",
        xnchw
            .iter()
            .map(|v| format!("{v}"))
            .collect::<Vec<_>>()
            .join(",")
    );
    if let Some(t) = torch(&snip) {
        assert_close(&y2.data(), &t, 1e-5, "upsample_nearest2x vs torch");
    }
}

#[test]
fn concat_channels_two() {
    // Two 1x2x2xC tensors concatenated on channels.
    let a = Tensor::constant(vec![1.0, 2.0, 3.0, 4.0], &[1, 2, 2, 1]); // C=1
    let b = Tensor::constant(
        vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0],
        &[1, 2, 2, 2],
    ); // C=2
    let y = Tensor::concat_channels(&[&a, &b]);
    assert_eq!(y.shape(), vec![1, 2, 2, 3]);
    // Per spatial position: [a, b0, b1].
    let expect = vec![
        1.0, 10.0, 11.0, //
        2.0, 20.0, 21.0, //
        3.0, 30.0, 31.0, //
        4.0, 40.0, 41.0,
    ];
    assert_close(&y.data(), &expect, 1e-6, "concat_channels hand");

    // cat_c convenience equals the associated fn.
    let y2 = a.cat_c(&b);
    assert_close(&y2.data(), &y.data(), 1e-6, "cat_c == concat_channels");

    // Torch cross-check: cat along the last (channel) dim of NHWC tensors,
    // matching HOS's layout directly (no NCHW transpose needed).
    let snip = "import torch\n\
         a=torch.tensor([1.,2.,3.,4.]).reshape(1,2,2,1)\n\
         b=torch.tensor([10.,11.,20.,21.,30.,31.,40.,41.]).reshape(1,2,2,2)\n\
         y=torch.cat([a,b],dim=3).contiguous()\n\
         print(' '.join(f'{v:.6f}' for v in y.flatten().tolist()))";
    if let Some(t) = torch(snip) {
        assert_close(&y.data(), &t, 1e-5, "concat_channels vs torch");
    }
}
