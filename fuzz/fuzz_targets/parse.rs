#![no_main]

//! Public parse/serialize invariants over arbitrary bytes:
//!
//! - `from_slice` either errors (including invalid UTF-8) or yields a DOM; it must never panic.
//! - Serializing a parsed DOM (compact and pretty) must reparse to an equal DOM (NaN-tolerant
//!   equality).
//! - The `Display` impl must render the same bytes as [`simd_json5::to_string`].

use libfuzzer_sys::fuzz_target;
use simd_json5::Value;

fuzz_target!(|data: &[u8]| {
    let Ok(value) = simd_json5::from_slice(data) else {
        return;
    };

    let compact = simd_json5::to_string(&value);
    let reparsed = simd_json5::from_str(&compact).expect("compact output must reparse");
    assert!(
        dom_eq(&value, &reparsed),
        "compact round trip changed the DOM: {compact}"
    );

    let pretty = simd_json5::to_string_pretty(&value);
    let reparsed = simd_json5::from_str(&pretty).expect("pretty output must reparse");
    assert!(
        dom_eq(&value, &reparsed),
        "pretty round trip changed the DOM: {pretty}"
    );

    assert_eq!(format!("{value}"), compact);
});

/// DOM equality with NaN-tolerant float comparison; the crate's `PartialEq`
/// follows f64 semantics (`NaN != NaN`).
fn dom_eq(a: &Value<'_>, b: &Value<'_>) -> bool {
    match (a, b) {
        (Value::F64(x), Value::F64(y)) => (x.is_nan() && y.is_nan()) || x == y,
        (Value::Array(xs), Value::Array(ys)) => {
            xs.len() == ys.len() && xs.iter().zip(ys).all(|(x, y)| dom_eq(x, y))
        },
        (Value::Object(xs), Value::Object(ys)) => {
            xs.len() == ys.len()
                && xs
                    .iter()
                    .zip(ys)
                    .all(|((xk, xv), (yk, yv))| xk.as_str() == yk.as_str() && dom_eq(xv, yv))
        },
        _ => a == b,
    }
}
