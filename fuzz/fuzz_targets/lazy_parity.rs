#![no_main]

//! Lazy `get`/`LazyValue` invariants over arbitrary bytes:
//!
//! - `get` must never panic, whatever the input.
//! - On every document the DOM parser accepts, `get` at the root must succeed and
//!   `LazyValue::to_value` must equal the DOM (NaN-tolerant equality).
//! - Every top-level member/element — and one level below each container — must be
//!   locatable by key/index segment with an equal DOM; misses must be misses; the
//!   kind predicates and accessors must agree with the DOM's.
//!
//! The number of probes per level is capped so one input stays cheap.

use libfuzzer_sys::fuzz_target;
use simd_json5::{LazyValue, Value};

fuzz_target!(|data: &[u8]| {
    let Ok(dom) = simd_json5::from_slice(data) else {
        // Invalid documents only need to not panic.
        let lossy = String::from_utf8_lossy(data);
        drop(simd_json5::get(&lossy, &["a"]));
        drop(simd_json5::get(&lossy, &["a", "0"]));
        return;
    };
    let src = core::str::from_utf8(data).expect("from_slice validates UTF-8");

    let root = simd_json5::get(src, &[])
        .expect("root walk of a valid document")
        .expect("root always locates");
    check_lazy(&root, &dom);

    // Top level: object keys and array indices, with a key guaranteed absent.
    let mut probes: Vec<(String, &Value<'_>)> = Vec::new();
    match &dom {
        Value::Object(members) => {
            for (k, v) in members.iter().take(64) {
                probes.push((k.as_str().to_string(), v));
            }
        },
        Value::Array(items) => {
            for (i, v) in items.iter().enumerate().take(64) {
                probes.push((i.to_string(), v));
            }
        },
        _ => {},
    }
    for (segment, expected) in probes.iter().map(|(s, v)| (s.as_str(), *v)) {
        let lazy = simd_json5::get(src, &[segment])
            .expect("member walk of a valid document")
            .expect("member always locates");
        check_lazy(&lazy, expected);

        // One level below: scalars miss, containers locate members/elements.
        match expected {
            Value::Object(members) => {
                let absent = absent_key(&members.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>());
                for (k, v) in members.iter().take(16) {
                    let lazy = simd_json5::get(src, &[segment, k.as_str()])
                        .expect("nested member walk")
                        .expect("nested member always locates");
                    check_lazy(&lazy, v);
                }
                assert!(
                    simd_json5::get(src, &[segment, absent.as_str()]).unwrap().is_none(),
                    "absent key must miss"
                );
            },
            Value::Array(items) => {
                for (i, v) in items.iter().enumerate().take(16) {
                    let idx = i.to_string();
                    let lazy = simd_json5::get(src, &[segment, idx.as_str()])
                        .expect("array element walk")
                        .expect("array element always locates");
                    check_lazy(&lazy, v);
                }
                let past_end = items.len().to_string();
                assert!(
                    simd_json5::get(src, &[segment, past_end.as_str()]).unwrap().is_none(),
                    "index past the end must miss"
                );
            },
            _ => {
                assert!(
                    simd_json5::get(src, &[segment, "x"]).unwrap().is_none(),
                    "descending into a scalar must miss"
                );
            },
        }
    }
});

/// A key not among `present`: extends a base key until it is absent. No string
/// is unrepresentable as a JSON5 key, so a hardcoded "absent" key would
/// eventually collide with a real one.
fn absent_key(present: &[&str]) -> String {
    let mut key = String::from("absent");
    while present.contains(&key.as_str()) {
        key.push('z');
    }
    key
}

/// The located value and the DOM member must agree on kind, accessors and the
/// reparsed DOM.
fn check_lazy(lazy: &LazyValue<'_>, expected: &Value<'_>) {
    assert_eq!(lazy.is_null(), matches!(expected, Value::Null));
    assert_eq!(lazy.is_bool(), matches!(expected, Value::Bool(_)));
    assert_eq!(
        lazy.is_number(),
        matches!(expected, Value::I64(_) | Value::U64(_) | Value::F64(_))
    );
    assert_eq!(lazy.is_str(), matches!(expected, Value::Str(_)));
    assert_eq!(lazy.is_array(), matches!(expected, Value::Array(_)));
    assert_eq!(lazy.is_object(), matches!(expected, Value::Object(_)));

    assert_eq!(lazy.as_bool().unwrap(), expected.as_bool());
    assert_eq!(lazy.as_str().unwrap().as_deref(), expected.as_str());
    assert_eq!(lazy.as_i64().unwrap(), expected.as_i64());
    assert_eq!(lazy.as_u64().unwrap(), expected.as_u64());
    let (lf, ef) = (lazy.as_f64().unwrap(), expected.as_f64());
    match (lf, ef) {
        (Some(x), Some(y)) => assert!(x.is_nan() == y.is_nan() && (x.is_nan() || x == y)),
        (None, None) => {},
        _ => panic!("f64 accessor disagreement: {lf:?} vs {ef:?}"),
    }
    assert!(
        dom_eq(&lazy.to_value().expect("lazy to_value"), expected),
        "lazy to_value disagrees with the DOM"
    );
}

/// DOM equality with NaN-tolerant float comparison; the crate's `PartialEq`
/// follows f64 semantics (`NaN != NaN`).
fn dom_eq(a: &Value<'_>, b: &Value<'_>) -> bool {
    match (a, b) {
        (Value::F64(x), Value::F64(y)) => (x.is_nan() && y.is_nan()) || x == y,
        (Value::Array(xs), Value::Array(ys)) => {
            xs.len() == ys.len() && xs.iter().zip(ys).all(|(x, y)| dom_eq(x, y))
        }
        (Value::Object(xs), Value::Object(ys)) => {
            xs.len() == ys.len()
                && xs
                    .iter()
                    .zip(ys)
                    .all(|((xk, xv), (yk, yv))| xk.as_str() == yk.as_str() && dom_eq(xv, yv))
        }
        _ => a == b,
    }
}
