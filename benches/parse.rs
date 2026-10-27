//! Criterion parse benchmarks: `simd-json5` against the other Rust JSON5
//! implementations (`json5`, `serde-json5`) and the established JSON
//! implementations (`serde_json`, `simd-json`, `sonic-rs`).
//!
//! Every corpus is valid JSON5 (JSON is a subset); the JSON-only
//! implementations run on the strict-JSON corpora only.

use std::{fs, hint::black_box, path::PathBuf};

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};

/// `(file, is_strict_json)`.
const CORPORA: [(&str, bool); 4] = [
    ("twitter.json", true),
    ("citm_catalog.json", true),
    ("canada.json", true),
    ("json5.json5", false),
];

fn corpus(name: &str) -> String {
    let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "benches", "data", name]
        .iter()
        .collect();
    fs::read_to_string(path.as_path()).expect("corpus file must exist; see benches/data")
}

/// sonic-rs needs AVX2 on x86_64.
#[cfg(target_arch = "x86_64")]
fn sonic_available() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

#[cfg(not(target_arch = "x86_64"))]
fn sonic_available() -> bool {
    true
}

fn bench_parse(c: &mut Criterion) {
    for (name, is_json) in CORPORA {
        let text = corpus(name);
        let bytes = text.as_bytes().to_vec();
        let mut group = c.benchmark_group(format!("parse/{name}"));
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_function("simd-json5", |b| {
            b.iter(|| black_box(simd_json5::from_str(black_box(&text)).unwrap()));
        });

        // `json5` is quadratic on canada.json's long float arrays (~65 s per
        // parse; criterion needs >=10 samples), so it is only measured as a
        // one-shot single sample via examples/one_shot.rs instead.
        if name != "canada.json" {
            group.bench_function("json5", |b| {
                b.iter(|| {
                    black_box(json5::from_str::<serde_json::Value>(black_box(&text)).unwrap())
                });
            });
        }

        group.bench_function("serde-json5", |b| {
            b.iter(|| {
                black_box(serde_json5::from_str::<serde_json::Value>(black_box(&text)).unwrap())
            });
        });

        if is_json {
            group.bench_function("serde_json", |b| {
                b.iter(|| {
                    black_box(serde_json::from_str::<serde_json::Value>(black_box(&text)).unwrap())
                });
            });

            // simd-json mutates its input, so each iteration gets a private
            // copy prepared outside the measured routine.
            group.bench_function("simd-json", |b| {
                b.iter_batched(
                    || bytes.clone(),
                    |mut data| black_box(simd_json::to_owned_value(data.as_mut_slice()).unwrap()),
                    BatchSize::LargeInput,
                );
            });

            if sonic_available() {
                group.bench_function("sonic-rs", |b| {
                    b.iter(|| {
                        black_box(sonic_rs::from_str::<sonic_rs::Value>(black_box(&text)).unwrap())
                    });
                });
            }
        }
        group.finish();
    }
}

/// `from_slice` adds whole-input UTF-8 validation on top of `from_str`.
fn bench_slice(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse/slice");
    for (name, _) in CORPORA {
        let bytes = corpus(name).into_bytes();
        group.throughput(Throughput::Bytes(bytes.len() as u64));
        group.bench_function(format!("{name}/simd-json5"), |b| {
            b.iter(|| black_box(simd_json5::from_slice(black_box(&bytes)).unwrap()));
        });
    }
    group.finish();
}

/// Typed (serde) parsing: our serde layer into a zero-copy `Value` DOM
/// against `serde_json`'s `Value`. The DOM `from_str` groups above show the
/// serde-layer overhead by comparison.
#[cfg(feature = "serde")]
fn bench_typed(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse/typed");
    for (name, is_json) in CORPORA {
        let text = corpus(name);
        group.throughput(Throughput::Bytes(text.len() as u64));

        if is_json {
            group.bench_function(format!("{name}/serde_json"), |b| {
                b.iter(|| {
                    black_box(serde_json::from_str::<serde_json::Value>(black_box(&text)).unwrap())
                });
            });
        }

        group.bench_function(format!("{name}/simd-json5"), |b| {
            b.iter(|| {
                black_box(simd_json5::de::from_str::<simd_json5::Value>(black_box(&text)).unwrap())
            });
        });
    }
    group.finish();
}

fn bench_benches(c: &mut Criterion) {
    bench_parse(c);
    bench_slice(c);
    #[cfg(feature = "serde")]
    bench_typed(c);
}

criterion_group!(benches, bench_benches);
criterion_main!(benches);
