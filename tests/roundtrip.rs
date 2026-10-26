//! Round-trip tests: parse -> serialize -> parse must reach an equal DOM.
//! Serialization itself is additionally exercised in `tests/serialize.rs`.

use simd_json5::{Value, from_str, to_string, to_string_pretty};

/// DOM equality with NaN-tolerant float comparison. The crate's `PartialEq`
/// follows f64 semantics (`NaN != NaN`), which would make every round trip
/// through the `NaN` literal fail; this helper exists only for these tests.
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

/// Parse `text`, serialize compact and pretty, re-parse both outputs and
/// require equal DOMs.
fn assert_roundtrip(text: &str) {
    let first = from_str(text).unwrap();
    for output in [to_string(&first), to_string_pretty(&first)] {
        let second = from_str(&output).unwrap();
        assert!(
            dom_eq(&first, &second),
            "round trip changed the DOM: {output}"
        );
    }
}

#[test]
fn roundtrip_json5_kitchen_sink() {
    assert_roundtrip(
        r#"{
            // line comment
            /* block
               comment */
            unquoted: -0x1f,
            'single': [1, 2, 3,],
            "double": {"nested": {"deep": [true, false, null,]}},
            hex: 0xDEADBEEF,
            inf: Infinity,
            ninf: -Infinity,
            nan: NaN,
            exp: 1.5e-3,
            leading_dot: .5,
            trailing_dot: 5.,
            plus: +7,
            esc: "\u0041\x42\n\\\u2028",
        }"#,
    );
}

#[test]
fn roundtrip_top_level_scalars() {
    for text in [
        "null",
        "true",
        "false",
        "42",
        "-42",
        "0",
        "-0",
        "0.0",
        "-0.5",
        "1e3",
        "\"text\"",
        "'text'",
        "Infinity",
        "-Infinity",
        "NaN",
        "-0x10",
    ] {
        assert_roundtrip(text);
    }
}

#[test]
fn roundtrip_containers() {
    assert_roundtrip("{}");
    assert_roundtrip("[]");
    assert_roundtrip("[[[[1]]]]");
    assert_roundtrip("{a:{b:{c:{}}}}");
}

#[test]
fn roundtrip_duplicate_keys_keep_first_position_and_last_value() {
    let value = from_str("{a: 1, 'b': 2, \"a\": 3}").unwrap();
    assert_eq!(to_string(&value), r#"{"a":3,"b":2}"#);
    assert_roundtrip("{a: 1, 'b': 2, \"a\": 3}");
}

#[test]
fn roundtrip_escaped_strings() {
    assert_roundtrip(r#"["\" \\ \/ \b \f \n \r \t \v \0 \x41 \u0042 \u2028 \u00e9"]"#);
    assert_roundtrip(
        r#"["line\
 continuation"]"#,
    );
}

#[test]
fn roundtrip_integer_widths() {
    // Values at and beyond the i64/u64 boundaries become F64 on reparse only
    // when they exceed the integer range; exact integers must stay integral.
    let value = from_str("[18446744073709551615, -9223372036854775808]").unwrap();
    assert_eq!(
        value,
        Value::Array(vec![Value::U64(u64::MAX), Value::I64(i64::MIN)])
    );
    assert_roundtrip("[18446744073709551615, -9223372036854775808]");
}
