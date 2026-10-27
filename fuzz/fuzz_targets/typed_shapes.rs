#![no_main]

//! Typed deserialization into fixed Rust shapes. The DOM targets only reach
//! `deserialize_any`; these shapes drive the rest of the typed surface:
//! `deserialize_seq`/`tuple`, map keys, `Option`'s O(1) null probe, enum
//! forms, and — via the `OnlyA` struct — the validate-and-skip machinery for
//! members and elements a visitor stops before.
//!
//! Errors are expected for shape-mismatched inputs and ignored; only panics,
//! hangs and depth overflows are bugs. Depth-capped like `de_dom_parity`.

use libfuzzer_sys::fuzz_target;
use serde::Deserialize;

const MAX_DEPTH_PROBE: usize = 200;

fuzz_target!(|data: &[u8]| {
    if depth_estimate(data) > MAX_DEPTH_PROBE {
        return;
    }
    let _ = simd_json5::de::from_slice::<OnlyA>(data);
    let _ = simd_json5::de::from_slice::<Vec<Option<u8>>>(data);
    let _ = simd_json5::de::from_slice::<std::collections::HashMap<String, i64>>(data);
    let _ = simd_json5::de::from_slice::<(bool, f64, String)>(data);
    let _ = simd_json5::de::from_slice::<E>(data);
});

#[derive(Deserialize)]
#[allow(dead_code)]
struct OnlyA {
    a: u8,
}

#[derive(Deserialize)]
#[allow(dead_code)]
enum E {
    A,
    B(u8),
    C { x: u8 },
}

fn depth_estimate(data: &[u8]) -> usize {
    let mut depth = 0usize;
    let mut max = 0usize;
    for &b in data {
        match b {
            b'[' | b'{' => {
                depth += 1;
                max = max.max(depth);
            },
            b']' | b'}' => depth = depth.saturating_sub(1),
            _ => {},
        }
    }
    max
}
