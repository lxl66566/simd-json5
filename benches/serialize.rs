//! Criterion serialization benchmarks: our writer against `serde_json` (JSON
//! corpora) and `json5` / `serde-json5` (JSON5 corpus).

use std::{fs, hint::black_box, path::PathBuf};

use criterion::{Criterion, criterion_group, criterion_main};

const JSON_CORPORA: [&str; 3] = ["twitter.json", "citm_catalog.json", "canada.json"];

fn corpus(name: &str) -> String {
    let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "benches", "data", name]
        .iter()
        .collect();
    fs::read_to_string(path.as_path()).expect("corpus file must exist; see benches/data")
}

fn bench_json(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialize/json");
    group.sample_size(100);
    for name in JSON_CORPORA {
        let text = corpus(name);
        let serde_value: serde_json::Value = serde_json::from_str(&text).unwrap();
        let our_value = simd_json5::from_str(&text).unwrap();

        group.bench_function(format!("{name}/simd-json5"), |b| {
            b.iter(|| black_box(simd_json5::to_vec(black_box(&our_value))));
        });

        group.bench_function(format!("{name}/serde_json"), |b| {
            b.iter(|| black_box(serde_json::to_vec(black_box(&serde_value))));
        });
    }
    group.finish();
}

fn bench_json5(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialize/json5");
    group.sample_size(100);
    let text = corpus("json5.json5");
    let value: serde_json::Value = json5::from_str(&text).unwrap();
    let our_value = simd_json5::from_str(&text).unwrap();

    group.bench_function("json5.json5/simd-json5", |b| {
        b.iter(|| black_box(simd_json5::to_vec(black_box(&our_value))));
    });

    group.bench_function("json5.json5/json5", |b| {
        b.iter(|| black_box(json5::to_string(black_box(&value)).unwrap()));
    });

    group.bench_function("json5.json5/serde-json5", |b| {
        b.iter(|| black_box(serde_json5::to_string(black_box(&value)).unwrap()));
    });
    group.finish();
}

/// Typed (serde) serialization: the same DOMs through our serde `Serializer`
/// and `serde_json`'s, isolating the serde-layer cost against the groups
/// above.
#[cfg(feature = "serde")]
fn bench_typed(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialize/typed");
    group.sample_size(100);
    for name in JSON_CORPORA {
        let text = corpus(name);
        let serde_value: serde_json::Value = serde_json::from_str(&text).unwrap();
        let our_value = simd_json5::from_str(&text).unwrap();

        group.bench_function(format!("{name}/simd-json5"), |b| {
            b.iter(|| black_box(simd_json5::typed::to_vec(black_box(&our_value)).unwrap()));
        });

        group.bench_function(format!("{name}/serde_json"), |b| {
            b.iter(|| black_box(serde_json::to_vec(black_box(&serde_value))));
        });
    }
    group.finish();
}

fn bench_serialize(c: &mut Criterion) {
    bench_json(c);
    bench_json5(c);
    #[cfg(feature = "serde")]
    bench_typed(c);
}

criterion_group!(benches, bench_serialize);
criterion_main!(benches);
