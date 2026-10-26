//! Single-sample parse timing for implementations whose cost on some corpus
//! is too large for repeated criterion measurement (e.g. quadratic behavior
//! on long arrays). Prints one JSON line per implementation; the lines merge
//! into `benches/results/one_shot.jsonl` which `benches/report.py` folds into
//! the dated results.
//!
//! Usage: `cargo run --release --example one_shot -- <corpus> <impl>...`

use std::{env, fs, hint::black_box, path::PathBuf, time::Instant};

fn main() {
    let mut args = env::args().skip(1);
    let corpus = args
        .next()
        .unwrap_or_else(|| panic!("usage: one_shot <corpus> <impl>..."));
    let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "benches", "data", &corpus]
        .iter()
        .collect();
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));

    for impl_name in args {
        let start = Instant::now();
        match impl_name.as_str() {
            "simd-json5" => {
                black_box(simd_json5::from_str(black_box(&text)).unwrap());
            },
            "json5" => {
                black_box(json5::from_str::<serde_json::Value>(black_box(&text)).unwrap());
            },
            "serde-json5" => {
                black_box(serde_json5::from_str::<serde_json::Value>(black_box(&text)).unwrap());
            },
            other => panic!("unknown implementation {other}"),
        }
        let mean_ns = u64::try_from(start.elapsed().as_nanos()).expect("duration fits u64");
        println!(
            "{}",
            serde_json::json!({
                "benchmark": "parse",
                "corpus": corpus,
                "impl": impl_name,
                "mean_ns": mean_ns,
                "samples": 1,
                "one_shot": true,
            })
        );
    }
}
