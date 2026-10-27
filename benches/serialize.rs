//! Criterion serialization benchmarks: `simd-json5` against the other Rust
//! JSON5 implementations (`json5`, `serde-json5`) and `serde_json`.
//!
//! simd-json5 and serde_json emit strict JSON; `json5` and `serde-json5` emit
//! JSON5. Every corpus is valid JSON5 (JSON is a subset); `serde_json` runs
//! on the strict-JSON corpora only.

use std::{fs, hint::black_box, path::PathBuf};

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

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

fn bench_serialize(c: &mut Criterion) {
    for (name, is_json) in CORPORA {
        let text = corpus(name);
        // The DOMs are prepared outside the measured routine; any JSON5
        // parser can produce the shared serde_json::Value. Known distortion
        // on the JSON5 corpora: serde_json::Value cannot hold non-finite
        // floats, so their Infinity/NaN literals degrade to Null (a slightly
        // shorter output than what simd-json5 serializes for its own DOM).
        let serde_value: serde_json::Value = serde_json5::from_str(&text).unwrap();
        let our_value = simd_json5::from_str(&text).unwrap();

        let mut group = c.benchmark_group(format!("serialize/{name}"));
        group.sample_size(100);
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_function("simd-json5", |b| {
            b.iter(|| black_box(simd_json5::to_vec(black_box(&our_value))));
        });

        group.bench_function("json5", |b| {
            b.iter(|| black_box(json5::to_string(black_box(&serde_value)).unwrap()));
        });

        group.bench_function("serde-json5", |b| {
            b.iter(|| black_box(serde_json5::to_string(black_box(&serde_value)).unwrap()));
        });

        if is_json {
            group.bench_function("serde_json", |b| {
                b.iter(|| black_box(serde_json::to_vec(black_box(&serde_value))));
            });
        }
        group.finish();
    }
}

/// Typed (serde) serialization: the same DOMs through our serde `Serializer`
/// and `serde_json`'s, isolating the serde-layer cost against the groups
/// above.
#[cfg(feature = "serde")]
fn bench_typed(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialize/typed");
    group.sample_size(100);
    for (name, is_json) in CORPORA {
        let text = corpus(name);
        // Same DOM construction (and non-finite-to-Null distortion) as
        // bench_serialize; serde_json itself only runs on strict JSON.
        let serde_value: serde_json::Value = serde_json5::from_str(&text).unwrap();
        let our_value = simd_json5::from_str(&text).unwrap();
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_function(format!("{name}/simd-json5"), |b| {
            b.iter(|| black_box(simd_json5::typed::to_vec(black_box(&our_value)).unwrap()));
        });

        if is_json {
            group.bench_function(format!("{name}/serde_json"), |b| {
                b.iter(|| black_box(serde_json::to_vec(black_box(&serde_value))));
            });
        }
    }
    group.finish();
}

fn bench_benches(c: &mut Criterion) {
    bench_serialize(c);
    #[cfg(feature = "serde")]
    bench_typed(c);
}

criterion_group!(benches, bench_benches);
criterion_main!(benches);
