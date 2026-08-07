//! Format conformance — the load-bearing invariant of the hybrid `.hos`/`.flwr`
//! decision: **the `.flwr` label is a cosmetic alias over the sovereign `.hos`
//! capsule.** Every `.flwr` MUST be a byte-valid `.hos`: same magic bytes, ONE
//! writer, ONE reader, NO flwr-private chunks. `arch=flwr` (not the extension) is
//! the discriminator.
//!
//! If this test ever fails, the hybrid format has FORKED and the "everything
//! powered by HOS / traces to me" sovereignty guarantee is void — at which point
//! the correct move is to collapse back to `.hos`-only, not to ship a fork.

use hos::format::{self, Card, Named, MAGIC, ROLE_WEIGHT};

fn tiny_flwr_capsule() -> (Vec<Named>, Card) {
    let tensors = vec![
        Named {
            role: ROLE_WEIGHT,
            name: "token_embd.weight".into(),
            shape: vec![4, 8],
            data: (0..32).map(|i| i as f32 * 0.1).collect(),
        },
        Named {
            role: ROLE_WEIGHT,
            name: "blk.0.attn_q.weight".into(),
            shape: vec![8, 8],
            data: (0..64).map(|i| (i as f32).sin()).collect(),
        },
    ];
    let arch = serde_json::json!({
        "architecture": "flwr",
        "source_format": "flwr",
        "bottleneck": "e8-vq",
    });
    let mut card = Card::new("flwr-conformance", arch);
    card.provenance.engine = "flwr".into();
    (tensors, card)
}

/// A `.flwr` and a `.hos` written from identical content are BYTE-IDENTICAL —
/// proving the extension is cosmetic and there is exactly one writer.
#[test]
fn flwr_is_byte_identical_to_hos() {
    let dir = std::env::temp_dir();
    let p_flwr = dir.join("hos_conf_test.flwr");
    let p_hos = dir.join("hos_conf_test.hos");
    let (tensors, card) = tiny_flwr_capsule();
    format::save(&p_flwr, &tensors, &card).expect("write .flwr");
    format::save(&p_hos, &tensors, &card).expect("write .hos");
    let b_flwr = std::fs::read(&p_flwr).unwrap();
    let b_hos = std::fs::read(&p_hos).unwrap();
    assert_eq!(
        b_flwr, b_hos,
        ".flwr and .hos from identical content MUST be byte-identical (one writer, no per-extension difference)"
    );
    assert_eq!(
        &b_flwr[..4],
        MAGIC,
        "every capsule begins with the HOS magic bytes"
    );
    let _ = std::fs::remove_file(&p_flwr);
    let _ = std::fs::remove_file(&p_hos);
}

/// The stock `.hos` reader loads a `.flwr` file identically (magic-byte path),
/// and `arch=flwr` survives the round-trip as the real discriminator.
#[test]
fn flwr_loads_via_stock_hos_reader() {
    let dir = std::env::temp_dir();
    let p = dir.join("hos_conf_load.flwr");
    let (tensors, card) = tiny_flwr_capsule();
    format::save(&p, &tensors, &card).expect("write .flwr");

    assert!(
        hos::is_hos_file(&p),
        ".flwr must be recognized by is_hos_file via magic bytes, not extension"
    );
    let (rt, rcard) = format::load(&p).expect("the stock .hos reader must load a .flwr");
    assert_eq!(rt.len(), tensors.len(), "tensor count round-trips");
    for (got, want) in rt.iter().zip(&tensors) {
        assert_eq!(got.name, want.name);
        assert_eq!(got.shape, want.shape);
        assert_eq!(got.data, want.data, "tensor data must round-trip exactly");
    }
    assert_eq!(
        rcard.arch.get("architecture").and_then(|v| v.as_str()),
        Some("flwr"),
        "arch=flwr is the discriminator and must survive the round-trip"
    );
    let _ = std::fs::remove_file(&p);
}
