//! Training utilities — optional and composable. Use these, or roll your own
//! loop: the raw `forward` / `backward` / `sgd_step` / `AdamW::step` primitives
//! remain fully available, so you can always drop down to single-example or
//! fully-custom training. Nothing here is mandatory.

use crate::tensor::{AdamW, Tensor};

/// Fisher–Yates shuffle with a seeded xorshift (reproducible across runs).
pub fn shuffle<T>(v: &mut [T], seed: &mut u64) {
    for i in (1..v.len()).rev() {
        let mut x = *seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *seed = x;
        let j = (x as usize) % (i + 1);
        v.swap(i, j);
    }
}

/// One minibatch optimizer step by **gradient accumulation**: averages the
/// gradients over the batch, then takes a single `AdamW` step. Returns the
/// mean loss over the batch.
///
/// `batch.len() == 1` reduces to plain single-example training, so the "small"
/// path costs nothing. `forward_loss(&example) -> scalar loss tensor`.
///
/// Note: this gives the *statistical* benefits of batching (stable averaged
/// gradients, proper shuffled minibatches). It does not yet run the batch in
/// parallel — examples are processed sequentially — so it's about training
/// quality/stability, not raw throughput. Parallel/throughput batching (a batch
/// dimension + batched ops) is a separate, heavier layer.
pub fn train_minibatch<E>(
    batch: &[E],
    params: &[&Tensor],
    opt: &mut AdamW,
    decay: &[bool],
    forward_loss: impl Fn(&E) -> Tensor,
) -> f32 {
    for p in params {
        p.zero_grad();
    }
    let scale = 1.0 / batch.len() as f32;
    let mut mean = 0.0;
    for ex in batch {
        let loss = forward_loss(ex).scale(scale); // scale so accumulated grad = mean grad
        loss.backward();
        mean += loss.data()[0];
    }
    opt.step(params, decay);
    mean
}
