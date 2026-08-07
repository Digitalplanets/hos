//! Gradient checks for the autograd core: every op's analytic backward is
//! verified against central finite differences. This is the test that proves
//! training is mathematically sound — if a backward closure is wrong, the
//! numeric gradient won't match.

// `&[x.clone()]` is intentional: x is reused on the following line, so the clone
// is needed (not replaceable with slice::from_ref).
#![allow(clippy::cloned_ref_to_slice_refs)]

use hos::tensor::{AdamW, Tensor};

/// Central finite-difference gradient check.
///
/// `inits` are (data, shape) for each input param. `f` builds a SCALAR output
/// from fresh tensors. We compare backward()'s grads against (f(x+h)-f(x-h))/2h.
fn grad_check(inits: &[(Vec<f32>, Vec<usize>)], f: impl Fn(&[Tensor]) -> Tensor) {
    let mk = |datas: &[Vec<f32>]| -> Vec<Tensor> {
        inits
            .iter()
            .zip(datas)
            .map(|((_, sh), d)| Tensor::param(d.clone(), sh))
            .collect()
    };
    let orig: Vec<Vec<f32>> = inits.iter().map(|(d, _)| d.clone()).collect();

    // analytic gradients
    let ts = mk(&orig);
    let out = f(&ts);
    assert_eq!(out.shape(), vec![1], "grad_check target must be a scalar");
    out.backward();
    let analytic: Vec<Vec<f32>> = ts.iter().map(|t| t.grad()).collect();

    // numeric gradients
    let eps = 1e-3f32;
    for i in 0..orig.len() {
        for k in 0..orig[i].len() {
            let mut up = orig.clone();
            let mut dn = orig.clone();
            up[i][k] += eps;
            dn[i][k] -= eps;
            let lp = f(&mk(&up)).data()[0];
            let lm = f(&mk(&dn)).data()[0];
            let num = (lp - lm) / (2.0 * eps);
            let ana = analytic[i][k];
            let tol = 2e-2 + 2e-2 * ana.abs();
            assert!(
                (num - ana).abs() <= tol,
                "input {i}[{k}]: numeric {num} vs analytic {ana} (tol {tol})"
            );
        }
    }
}

#[test]
fn grad_matmul() {
    grad_check(
        &[
            (vec![0.5, -1.0, 2.0, 0.3, -0.7, 1.1], vec![2, 3]),
            (vec![1.0, -0.5, 0.2, 0.9, -1.3, 0.4], vec![3, 2]),
        ],
        |t| t[0].matmul(&t[1]).mean(),
    );
}

#[test]
fn grad_add_broadcast_bias() {
    grad_check(
        &[
            (vec![0.5, -1.0, 2.0, 0.3, -0.7, 1.1], vec![2, 3]),
            (vec![0.2, -0.4, 0.9], vec![3]),
        ],
        |t| t[0].add(&t[1]).mean(),
    );
}

#[test]
fn grad_mul_and_sub() {
    let a = (vec![0.5, -1.0, 2.0, 0.3], vec![2, 2]);
    let b = (vec![1.0, -0.5, 0.2, 0.9], vec![2, 2]);
    grad_check(&[a.clone(), b.clone()], |t| t[0].mul(&t[1]).mean());
    grad_check(&[a, b], |t| t[0].sub(&t[1]).mean());
}

#[test]
fn grad_relu_and_silu() {
    // avoid exactly 0 inputs for relu (kink)
    let x = (vec![0.7, -1.2, 2.0, -0.4, 1.5, -2.1], vec![2, 3]);
    grad_check(&[x.clone()], |t| t[0].relu().mean());
    grad_check(&[x], |t| t[0].silu().mean());
}

#[test]
fn grad_square_and_scale() {
    let x = (vec![0.7, -1.2, 2.0, -0.4], vec![2, 2]);
    grad_check(&[x.clone()], |t| t[0].square().mean());
    grad_check(&[x], |t| t[0].scale(2.5).mean());
}

#[test]
fn grad_rmsnorm() {
    grad_check(
        &[
            (vec![0.5, -1.0, 2.0, 0.3, -0.7, 1.1], vec![2, 3]),
            (vec![1.0, 0.8, 1.2], vec![3]),
        ],
        |t| t[0].rmsnorm(&t[1]).mean(),
    );
}

#[test]
fn grad_layernorm() {
    // grads wrt input, weight, AND bias are all checked at once.
    grad_check(
        &[
            (vec![0.5, -1.0, 2.0, 0.3, -0.7, 1.1], vec![2, 3]),
            (vec![1.0, 0.8, 1.2], vec![3]),
            (vec![0.1, -0.2, 0.3], vec![3]),
        ],
        // square so the scalar depends nontrivially on every output element
        |t| t[0].layernorm(&t[1], &t[2], 1e-5).square().mean(),
    );
}

#[test]
fn grad_gelu_erf() {
    let x = (vec![0.7, -1.2, 2.0, -0.4, 1.5, -2.1], vec![2, 3]);
    grad_check(&[x], |t| t[0].gelu_erf().mean());
}

#[test]
fn gelu_erf_values() {
    // gelu_erf(0) = 0; large +x ≈ x; large -x ≈ 0.
    let v = Tensor::constant(vec![0.0, 8.0, -8.0], &[1, 3])
        .gelu_erf()
        .data();
    assert!(v[0].abs() < 1e-6, "gelu_erf(0) = {}", v[0]);
    assert!((v[1] - 8.0).abs() < 1e-4, "gelu_erf(8) = {}", v[1]);
    assert!(v[2].abs() < 1e-4, "gelu_erf(-8) = {}", v[2]);
}

#[test]
fn layernorm_normalizes() {
    // before affine (w=1,b=0) each row has ~0 mean and ~unit variance.
    let w = Tensor::constant(vec![1.0; 4], &[4]);
    let b = Tensor::constant(vec![0.0; 4], &[4]);
    let y = Tensor::constant(vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 5.0, 2.0], &[2, 4])
        .layernorm(&w, &b, 1e-5)
        .data();
    for r in 0..2 {
        let row = &y[r * 4..r * 4 + 4];
        let m = row.iter().sum::<f32>() / 4.0;
        let var = row.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / 4.0;
        assert!(m.abs() < 1e-4, "row {r} mean {m}");
        assert!((var - 1.0).abs() < 1e-3, "row {r} var {var}");
    }
}

#[test]
fn grad_softmax_rows() {
    grad_check(
        &[(vec![0.5, -1.0, 2.0, 0.3, -0.7, 1.1], vec![2, 3])],
        // square after softmax so the scalar depends nontrivially on every prob
        |t| t[0].softmax_rows().square().mean(),
    );
}

#[test]
fn grad_cross_entropy() {
    let targets = [2usize, 0usize];
    grad_check(
        &[(vec![0.5, -1.0, 2.0, 0.3, -0.7, 1.1], vec![2, 3])],
        move |t| t[0].cross_entropy(&targets),
    );
}

#[test]
fn grad_embedding() {
    let ids = [2usize, 0usize, 2usize];
    grad_check(
        &[(vec![0.1, 0.2, -0.3, 0.4, 0.5, -0.6], vec![3, 2])],
        move |t| t[0].embedding(&ids).mean(),
    );
}

#[test]
fn grad_bmm() {
    grad_check(
        &[
            (
                vec![0.5, -1.0, 2.0, 0.3, 1.0, -0.5, 0.2, 0.9],
                vec![2, 2, 2],
            ),
            (
                vec![1.0, 0.0, 0.5, -1.0, -0.3, 0.7, 0.2, 1.1],
                vec![2, 2, 2],
            ),
        ],
        |t| t[0].bmm(&t[1]).mean(),
    );
}

#[test]
fn grad_conv1d() {
    // input [L=4, cin=2], weight [cout=2, k*cin = 2*2=4]
    grad_check(
        &[
            (vec![0.5, -1.0, 2.0, 0.3, -0.7, 1.1, 0.2, -0.9], vec![4, 2]),
            (vec![1.0, -0.5, 0.2, 0.9, -1.3, 0.4, 0.6, -0.2], vec![2, 4]),
        ],
        |t| t[0].conv1d(&t[1], 2, 2, 2).mean(),
    );
}

#[test]
fn grad_shape_ops() {
    let x4 = (
        vec![0.5, -1.0, 2.0, 0.3, 1.0, -0.5, 0.2, 0.9],
        vec![2, 2, 2],
    );
    grad_check(&[x4.clone()], |t| t[0].transpose_last2().mean());
    grad_check(&[(x4.0.clone(), vec![1, 2, 2, 2])], |t| {
        t[0].transpose12_4d().mean()
    });
    grad_check(&[(vec![0.5, -1.0, 2.0, 0.3, -0.7, 1.1], vec![2, 3])], |t| {
        t[0].transpose().square().mean()
    });
    grad_check(&[(vec![0.5, -1.0, 2.0, 0.3], vec![2, 2])], |t| {
        t[0].reshape(&[4, 1]).square().mean()
    });
}

#[test]
fn concat_last_works_for_video_rank() {
    let a = Tensor::param((0..12).map(|x| x as f32).collect(), &[1, 2, 2, 1, 3]);
    let b = Tensor::param((0..8).map(|x| 100.0 + x as f32).collect(), &[1, 2, 2, 1, 2]);
    let y = Tensor::concat_last(&[&a, &b]);
    assert_eq!(y.shape(), vec![1, 2, 2, 1, 5]);
    assert_eq!(&y.data()[0..5], &[0.0, 1.0, 2.0, 100.0, 101.0]);
    y.mean().backward();
    assert!(a.grad().iter().all(|&g| (g - 0.05).abs() < 1e-7));
    assert!(b.grad().iter().all(|&g| (g - 0.05).abs() < 1e-7));
}

#[test]
fn grad_col_and_mul_broadcast() {
    grad_check(&[(vec![0.5, -1.0, 2.0, 0.3, -0.7, 1.1], vec![3, 2])], |t| {
        t[0].col(1).square().mean()
    });
    grad_check(
        &[
            (vec![0.5, -1.0, 2.0, 0.3, -0.7, 1.1], vec![3, 2]),
            (vec![0.4, -0.8, 1.2], vec![3, 1]),
        ],
        |t| t[0].mul_broadcast(&t[1]).mean(),
    );
}

#[test]
fn grad_max_rows() {
    // distinct values per column so the argmax is unambiguous
    grad_check(&[(vec![0.5, -1.0, 2.0, 0.3, -0.7, 1.1], vec![3, 2])], |t| {
        t[0].max_rows().square().mean()
    });
}

#[test]
fn end_to_end_mlp_grads() {
    // logits = relu(x @ w1) @ w2 ; loss = cross_entropy. Checks grads flow
    // through a realistic little network (the same shape interp.rs trains).
    let x = vec![0.5, -1.0, 2.0, 0.3, -0.7, 1.1]; // [2,3]
    let w1 = vec![
        0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8, 0.9, 1.0, -1.1, 1.2,
    ]; // [3,4]
    let w2 = vec![0.2, -0.3, 0.5, -0.1, 0.6, -0.7, 0.8, 0.9]; // [4,2]
    let targets = [1usize, 0usize];
    grad_check(
        &[(x, vec![2, 3]), (w1, vec![3, 4]), (w2, vec![4, 2])],
        move |t| {
            t[0].matmul(&t[1])
                .relu()
                .matmul(&t[2])
                .cross_entropy(&targets)
        },
    );
}

#[test]
fn adamw_minimizes_quadratic() {
    // minimize f(w) = mean((w - target)^2); w should converge to target.
    let target = Tensor::constant(vec![3.0, -2.0, 0.5, 7.0], &[1, 4]);
    let w = Tensor::param(vec![0.0, 0.0, 0.0, 0.0], &[1, 4]);
    let params = [&w];
    let mut opt = AdamW::new(&params, 0.1, 0.0);
    for _ in 0..400 {
        w.zero_grad();
        let loss = w.sub(&target).square().mean();
        loss.backward();
        opt.step(&params, &[true]);
    }
    let got = w.data();
    for (g, t) in got.iter().zip([3.0, -2.0, 0.5, 7.0]) {
        assert!((g - t).abs() < 1e-2, "AdamW did not converge: {got:?}");
    }
}

#[test]
fn adamw_weight_decay_shrinks_params() {
    // with zero gradient, decay=true should pull a weight toward 0; decay=false
    // should leave it put. (decoupled weight decay sanity check)
    let decayed = Tensor::param(vec![5.0], &[1, 1]);
    let kept = Tensor::param(vec![5.0], &[1, 1]);
    let dp = [&decayed];
    let kp = [&kept];
    // decoupled decay is scaled by lr; with zero gradient the Adam term is ~0,
    // so the only movement comes from the decay term (decay=true).
    let mut od = AdamW::new(&dp, 0.1, 0.5);
    let mut ok = AdamW::new(&kp, 0.1, 0.5);
    for _ in 0..10 {
        decayed.zero_grad();
        kept.zero_grad();
        od.step(&dp, &[true]);
        ok.step(&kp, &[false]);
    }
    assert!(
        decayed.data()[0] < 5.0,
        "weight decay should shrink the param"
    );
    assert_eq!(kept.data()[0], 5.0, "decay=false must not change the param");
}

#[test]
fn gpu_matmul_matches_cpu() {
    // The training matmul has a GPU path (use_gpu) and a CPU path; they must
    // agree. Skipped gracefully if no Metal device is available.
    let init = std::panic::catch_unwind(|| {
        hos::tensor::use_gpu(true);
    });
    if init.is_err() {
        eprintln!("[skip] no Metal device for gpu_matmul_matches_cpu");
        return;
    }
    let mut seed = 0x1234_5678u64;
    let a = Tensor::randn(&[17, 23], &mut seed);
    let b = Tensor::randn(&[23, 11], &mut seed);

    let gpu_out = a.matmul(&b).data();
    hos::tensor::use_gpu(false);
    let cpu_out = a.matmul(&b).data();

    assert_eq!(gpu_out.len(), cpu_out.len());
    for (g, c) in gpu_out.iter().zip(&cpu_out) {
        assert!((g - c).abs() <= 1e-3 + 1e-3 * c.abs(), "gpu {g} vs cpu {c}");
    }
}

#[test]
fn grad_rope_interleaved() {
    // [rows=2, n_heads=2 * head_dim=4 = 8]; positions 0 and 1
    let x: Vec<f32> = (0..16).map(|i| i as f32 * 0.13 - 1.0).collect();
    grad_check(&[(x, vec![2, 8])], |t| {
        t[0].rope(2, 4, 10000.0, false, &[0, 1]).square().mean()
    });
}

#[test]
fn grad_rope_neox() {
    let x: Vec<f32> = (0..16).map(|i| i as f32 * 0.13 - 1.0).collect();
    grad_check(&[(x, vec![2, 8])], |t| {
        t[0].rope(2, 4, 10000.0, true, &[0, 1]).square().mean()
    });
}

#[test]
fn grad_repeat_interleave_dim0() {
    // [2,3] -> repeat each row twice -> [4,3]
    let x = vec![0.5, -1.0, 2.0, 0.3, -0.7, 1.1];
    grad_check(&[(x, vec![2, 3])], |t| {
        t[0].repeat_interleave_dim0(2).square().mean()
    });
}

#[test]
fn grad_sigmoid() {
    let x = vec![0.5, -1.2, 2.0, 0.0, -0.3, 1.5];
    grad_check(&[(x, vec![2, 3])], |t| t[0].sigmoid().square().mean());
}
