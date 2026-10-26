//! Criterion parse benchmarks: our crate against `serde_json`, `simd-json`
//! (JSON corpora) and `json5`, `serde-json5` (JSON5 corpus).

use std::{fs, hint::black_box, path::PathBuf};

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};

const JSON_CORPORA: [&str; 3] = ["twitter.json", "citm_catalog.json", "canada.json"];

fn corpus(name: &str) -> String {
    let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "benches", "data", name]
        .iter()
        .collect();
    fs::read_to_string(path.as_path()).expect("corpus file must exist; see benches/data")
}

fn bench_json(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse/json");
    group.sample_size(100);
    for name in JSON_CORPORA {
        let text = corpus(name);
        let bytes = text.as_bytes().to_vec();

        group.bench_function(format!("{name}/serde_json"), |b| {
            b.iter(|| {
                black_box(serde_json::from_str::<serde_json::Value>(black_box(&text)).unwrap())
            });
        });

        // simd-json mutates its input, so each iteration gets a private copy
        // prepared outside the measured routine.
        group.bench_function(format!("{name}/simd-json"), |b| {
            b.iter_batched(
                || bytes.clone(),
                |mut data| {
                    black_box(simd_json::to_owned_value(data.as_mut_slice()).unwrap());
                },
                BatchSize::LargeInput,
            );
        });

        group.bench_function(format!("{name}/simd-json5"), |b| {
            b.iter(|| black_box(simd_json5::from_str(black_box(&text)).unwrap()));
        });
    }
    group.finish();
}

fn bench_json5(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse/json5");
    group.sample_size(100);
    let text = corpus("json5.json5");

    group.bench_function("json5.json5/json5", |b| {
        b.iter(|| black_box(json5::from_str::<serde_json::Value>(black_box(&text)).unwrap()));
    });

    group.bench_function("json5.json5/serde-json5", |b| {
        b.iter(|| black_box(serde_json5::from_str::<serde_json::Value>(black_box(&text)).unwrap()));
    });

    group.bench_function("json5.json5/simd-json5", |b| {
        b.iter(|| black_box(simd_json5::from_str(black_box(&text)).unwrap()));
    });
    group.finish();
}

/// `from_slice` adds whole-input UTF-8 validation on top of `from_str`.
fn bench_slice(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse/slice");
    group.sample_size(100);
    for name in JSON_CORPORA {
        let bytes = corpus(name).into_bytes();
        group.bench_function(format!("{name}/simd-json5"), |b| {
            b.iter(|| black_box(simd_json5::from_slice(black_box(&bytes)).unwrap()));
        });
    }
    group.finish();
}

/// Typed (serde) parsing: our serde layer into a zero-copy `Value` DOM
/// against `serde_json`'s `Value`. The DOM `from_str` group above shows the
/// serde-layer overhead by comparison.
#[cfg(feature = "serde")]
fn bench_typed(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse/typed");
    group.sample_size(100);
    for name in JSON_CORPORA {
        let text = corpus(name);

        group.bench_function(format!("{name}/serde_json"), |b| {
            b.iter(|| {
                black_box(serde_json::from_str::<serde_json::Value>(black_box(&text)).unwrap())
            });
        });

        group.bench_function(format!("{name}/simd-json5"), |b| {
            b.iter(|| {
                black_box(simd_json5::de::from_str::<simd_json5::Value>(black_box(&text)).unwrap())
            });
        });
    }
    group.finish();
}

fn bench_parse(c: &mut Criterion) {
    bench_json(c);
    bench_json5(c);
    bench_slice(c);
    #[cfg(feature = "serde")]
    bench_typed(c);
}

criterion_group!(benches, bench_parse);
criterion_main!(benches);
