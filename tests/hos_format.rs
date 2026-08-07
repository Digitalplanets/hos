//! HOS ("hos") format tests: a saved model must load back bit-identical,
//! its card metadata must survive the round trip, corruption must be caught by
//! the checksums, and model identity (content hash) must be stable + sensitive.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

use hos::format::{self, Card, Named, ROLE_NORM, ROLE_WEIGHT};

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn temp_path(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!("hos_fmt_{}_{}_{}.hos", std::process::id(), tag, n));
    p
}

fn sample_tensors() -> Vec<Named> {
    vec![
        Named {
            name: "blk.0.w".into(),
            role: ROLE_WEIGHT,
            shape: vec![3, 2],
            data: vec![0.5, -1.0, 2.0, 0.25, -0.75, 1.5],
        },
        Named {
            name: "blk.0.norm".into(),
            role: ROLE_NORM,
            shape: vec![2],
            data: vec![1.0, 0.9],
        },
    ]
}

#[test]
fn round_trip_is_bit_identical() {
    let path = temp_path("rt");
    let tensors = sample_tensors();
    let mut card = Card::new(
        "test-model",
        serde_json::json!({"type":"sequential","layers":[]}),
    );
    card.id = format::model_id(&tensors);
    card.history.push(format::TrainRun {
        steps: 100,
        final_loss: 0.25,
        optimizer: "adamw".into(),
        lr: 1e-3,
    });
    card.lineage.push("deadbeefdeadbeef".into());
    card.resume.step = 100;
    card.resume.seed = 42;

    format::save(&path, &tensors, &card).unwrap();
    let (loaded, lcard) = format::load(&path).unwrap();

    assert_eq!(loaded.len(), tensors.len());
    for (a, b) in loaded.iter().zip(&tensors) {
        assert_eq!(a.name, b.name);
        assert_eq!(a.role, b.role);
        assert_eq!(a.shape, b.shape);
        assert_eq!(a.data, b.data, "tensor data must round-trip exactly");
    }
    assert_eq!(lcard.name, "test-model");
    assert_eq!(lcard.id, card.id);
    assert_eq!(lcard.resume.step, 100);
    assert_eq!(lcard.resume.seed, 42);
    assert_eq!(lcard.history.len(), 1);
    assert_eq!(lcard.history[0].steps, 100);
    assert_eq!(lcard.lineage, vec!["deadbeefdeadbeef".to_string()]);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn corruption_is_detected() {
    let path = temp_path("corrupt");
    // one big tensor so the file is dominated by its data region
    let tensors = vec![Named {
        name: "w".into(),
        role: ROLE_WEIGHT,
        shape: vec![256],
        data: (0..256).map(|i| i as f32 * 0.01).collect(),
    }];
    let card = Card::new("c", serde_json::json!({}));
    format::save(&path, &tensors, &card).unwrap();

    // clean load works
    assert!(format::load(&path).is_ok());

    // flip a byte in the middle (inside the tensor data region) -> checksum fails
    let mut bytes = std::fs::read(&path).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xFF;
    std::fs::write(&path, &bytes).unwrap();

    let err = format::load(&path);
    assert!(
        err.is_err(),
        "corrupted tensor data must be rejected by the checksum"
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn rejects_non_hos_file() {
    let path = temp_path("bad");
    std::fs::write(&path, b"not a hos file at all").unwrap();
    assert!(format::load(&path).is_err());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn model_id_is_stable_and_data_sensitive() {
    let a = sample_tensors();
    let b = sample_tensors();
    assert_eq!(
        format::model_id(&a),
        format::model_id(&b),
        "same weights -> same id"
    );

    let mut c = sample_tensors();
    c[0].data[0] += 1e-4; // tiny change
    assert_ne!(
        format::model_id(&a),
        format::model_id(&c),
        "different weights -> different id"
    );
}
