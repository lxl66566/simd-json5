//! Exact-output tests for the serializer, plus a cross-check against
//! `serde_json` on JSON-representable values.

use std::fmt::Write as _;

use simd_json5::{Map, Str, Value, to_string, to_string_pretty, to_vec};

fn object(pairs: impl IntoIterator<Item = (&'static str, Value<'static>)>) -> Value<'static> {
    Value::Object(
        pairs
            .into_iter()
            .map(|(k, v)| (Str::Static(k), v))
            .collect::<Map<'_>>(),
    )
}

#[test]
fn scalars() {
    assert_eq!(to_string(&Value::Null), "null");
    assert_eq!(to_string(&Value::Bool(true)), "true");
    assert_eq!(to_string(&Value::Bool(false)), "false");
    assert_eq!(to_string(&Value::I64(0)), "0");
    assert_eq!(to_string(&Value::I64(-1)), "-1");
    assert_eq!(to_string(&Value::I64(i64::MIN)), "-9223372036854775808");
    assert_eq!(to_string(&Value::I64(i64::MAX)), "9223372036854775807");
    assert_eq!(to_string(&Value::U64(u64::MAX)), "18446744073709551615");
    assert_eq!(to_string(&Value::F64(0.0)), "0.0");
    assert_eq!(to_string(&Value::F64(-0.0)), "-0.0");
    assert_eq!(to_string(&Value::F64(2.0)), "2.0");
    assert_eq!(to_string(&Value::F64(0.15625)), "0.15625");
    assert_eq!(to_string(&Value::F64(1e300)), "1e+300");
    assert_eq!(to_string(&Value::F64(5e-324)), "5e-324");
}

#[test]
fn non_finite_literals() {
    assert_eq!(to_string(&Value::F64(f64::NAN)), "NaN");
    assert_eq!(to_string(&Value::F64(f64::INFINITY)), "Infinity");
    assert_eq!(to_string(&Value::F64(f64::NEG_INFINITY)), "-Infinity");
    assert_eq!(
        to_string(&object([
            ("nan", Value::F64(f64::NAN)),
            ("inf", Value::F64(f64::INFINITY)),
            ("ninf", Value::F64(f64::NEG_INFINITY)),
        ])),
        r#"{"nan":NaN,"inf":Infinity,"ninf":-Infinity}"#
    );
}

#[test]
fn string_escapes() {
    let ser = |s: &str| to_string(&Value::Str(Str::Static(s)));
    assert_eq!(ser(""), r#""""#);
    assert_eq!(ser("plain"), r#""plain""#);
    assert_eq!(ser("quote\"inside"), r#""quote\"inside""#);
    assert_eq!(ser("back\\slash"), r#""back\\slash""#);
    assert_eq!(ser("\u{8}\t\n\u{c}\r"), r#""\b\t\n\f\r""#);
    assert_eq!(ser("\u{0}"), r#""\u0000""#);
    assert_eq!(ser("\u{1}"), r#""\u0001""#);
    assert_eq!(ser("\u{1f}"), r#""\u001f""#);
    // 0x7f and everything above 0x20 (including U+2028/U+2029) pass through.
    assert_eq!(ser("\u{7f}"), "\"\u{7f}\"");
    assert_eq!(ser("a\u{2028}b\u{2029}c"), "\"a\u{2028}b\u{2029}c\"");
    assert_eq!(ser("héllo → 😀"), "\"héllo → 😀\"");
    // Escapes interleaved with multi-byte sequences.
    assert_eq!(ser("é\té"), "\"é\\té\"");
}

/// Escapes at every alignment relative to the 8-byte SWAR words of the fast
/// path, including multi-byte sequences straddling word boundaries.
#[test]
fn string_escape_word_boundaries() {
    let ser = |s: &str| to_string(&Value::Str(Str::Static(s)));
    for n in 0..24 {
        let multibyte = "éx".repeat(n);
        assert_eq!(ser(&multibyte), format!("\"{multibyte}\""));
        let quote = format!("{}\"{}", "a".repeat(n), "b".repeat(16));
        assert_eq!(
            ser(&quote),
            format!("\"{}\\\"{}\"", "a".repeat(n), "b".repeat(16))
        );
        let control = format!("{}\n{}", "a".repeat(n), "b".repeat(16));
        assert_eq!(
            ser(&control),
            format!("\"{}\\n{}\"", "a".repeat(n), "b".repeat(16))
        );
    }
}

/// The direct-write escape path (dirty strings) must match `serde_json`
/// byte for byte: escapes inside scanned words, inside the final partial
/// word, and long clean runs between them, at every alignment.
#[test]
fn dirty_string_direct_write_matches_serde_json() {
    let ser = |s: &str| to_string(&Value::Str(Str::Static(s)));
    for n in 0..40 {
        let cases = [
            format!("{}\"{}", "a".repeat(n), "b".repeat(40)),
            format!("{}\\{}", "a".repeat(n), "b".repeat(40)),
            format!("{}\u{1}{}", "a".repeat(n), "b".repeat(n % 9)),
            format!("{}\t{}", "a".repeat(40), "b".repeat(n % 9)),
            format!(
                "{}\"{}\\{}\n{}",
                "a".repeat(n),
                "b".repeat(17),
                "c".repeat(3),
                "d".repeat(n)
            ),
            format!("é{}\"é{}", "é".repeat(n), "é".repeat(5)),
        ];
        for s in cases {
            assert_eq!(ser(&s), serde_json::to_string(&s).unwrap(), "{s:?}");
        }
    }
}

/// Extreme dirty-string distributions on the direct-write path: a single
/// escape inside multi-megabyte strings, every byte a six-byte escape, and
/// long runs between escapes. Pinned against `serde_json` and by a parse
/// round trip.
#[test]
fn dirty_string_extreme_distributions() {
    let ser = |s: &str| to_string(&Value::Str(Str::Static(s)));
    let single = format!("{}\"{}", "a".repeat(3 << 20), "b".repeat(3 << 20));
    let all_six_byte = "\u{1}".repeat(1 << 12);
    let spread = format!(
        "{}\n{}\"{}\\{}\u{2}{}",
        "a".repeat(1000),
        "b".repeat(1000),
        "c".repeat(1000),
        "d".repeat(1000),
        "e".repeat(1000)
    );
    for s in [&single, &all_six_byte, &spread] {
        assert_eq!(ser(s), serde_json::to_string(s).unwrap(), "len {}", s.len());
        assert_eq!(
            simd_json5::from_str(&ser(s)).unwrap(),
            Value::Str(Str::Static(s))
        );
    }
}

#[test]
fn nesting_and_containers() {
    assert_eq!(to_string(&Value::Array(vec![])), "[]");
    assert_eq!(to_string(&object([])), "{}");
    assert_eq!(to_string(&Value::Array(vec![Value::I64(1)])), "[1]");
    let nested = Value::Array(vec![
        Value::I64(1),
        Value::Array(vec![Value::Bool(true), Value::Null]),
        object([("k", Value::F64(-0.5))]),
    ]);
    assert_eq!(to_string(&nested), r#"[1,[true,null],{"k":-0.5}]"#);
    // Insertion order is preserved: keys emitted in first-seen order.
    assert_eq!(
        to_string(&object([
            ("z", Value::I64(26)),
            ("a", Value::I64(1)),
            ("m", Value::I64(13)),
        ])),
        r#"{"z":26,"a":1,"m":13}"#
    );
}

#[test]
fn to_vec_matches_to_string() {
    let value = object([("k", Value::Str(Str::Static("v\"")))]);
    assert_eq!(to_vec(&value), to_string(&value).into_bytes());
}

#[test]
fn display_matches_to_string() {
    let value = sample();
    assert_eq!(format!("{value}"), to_string(&value));
}

/// `Display` streams through a chunked sink; a chunk boundary may split a
/// multi-byte sequence, so long multibyte output must still round the
/// formatter correctly.
#[test]
fn display_streams_long_multibyte_output() {
    // Leading ASCII byte misaligns the chunks so boundaries fall mid-é; the
    // output spans several chunk boundaries.
    let value = Value::Str(Str::Dynamic(format!("x{}", "é".repeat(2000))));
    assert_eq!(format!("{value}"), to_string(&value));
}

/// `fmt::Write` target that fails after `ok_calls` successful `write_str`
/// calls, emulating a formatter with a budget or an underlying I/O error.
struct FailingWrite {
    ok_calls: usize,
}

impl std::fmt::Write for FailingWrite {
    fn write_str(&mut self, _s: &str) -> std::fmt::Result {
        if self.ok_calls == 0 {
            return Err(std::fmt::Error);
        }
        self.ok_calls -= 1;
        Ok(())
    }
}

/// A formatter that fails mid-stream must surface `Err(fmt::Error)` from the
/// `Display` impl instead of panicking or hanging once the sink's buffer
/// stays full (`flush` leaves the buffer undrained on failure). Each document
/// drives one of the three write shapes that can follow a failed flush:
///
/// - `array`: 1-byte extends alternating with pushes; the failing flush sits inside an `extend`,
///   and every later `push` must take the failed early return instead of indexing the full buffer.
/// - `string`: a single large `extend` spanning several flushes, so the failure lands mid-chunk
///   with bytes still left in it (a zero-progress loop would hang).
/// - `panic path`: the output fills the buffer to exactly the chunk size through pushes alone, so
///   the next `push` flushes at entry; a failure there used to index past the buffer end.
#[test]
fn display_survives_formatter_failure() {
    // Matches the FormatterSink chunk size in src/ser.rs.
    const CHUNK: usize = 1024;
    let array = Value::Array(vec![Value::I64(1); 4 * CHUNK]);
    assert!(to_string(&array).len() > 4 * CHUNK);
    let string = Value::Str(Str::Dynamic("x".repeat(6 * CHUNK)));
    // `[` plus the first member's `"xxx…x"` is exactly CHUNK bytes, so the `,`
    // push after it flushes at entry. The tail members only pad the document
    // past four chunk flushes so every budget in 0..4 fails.
    let panic_path = Value::Array(
        [
            Value::Str(Str::Dynamic("x".repeat(CHUNK - 3))),
            Value::Str(Str::Static("")),
        ]
        .into_iter()
        .chain((0..2400).map(Value::I64))
        .collect(),
    );
    assert!(to_string(&panic_path).len() > 4 * CHUNK);
    for (label, value) in [
        ("array", &array),
        ("string", &string),
        ("panic path", &panic_path),
    ] {
        for ok_calls in 0..4 {
            let mut target = FailingWrite { ok_calls };
            let result = write!(target, "{value}");
            assert_eq!(
                result,
                Err(std::fmt::Error),
                "{label}, ok_calls = {ok_calls}"
            );
        }
    }
}

/// A DOM covering every `Value` variant, JSON-representable (finite floats).
fn sample() -> Value<'static> {
    Value::Array(vec![
        Value::Null,
        Value::Bool(true),
        Value::Bool(false),
        Value::I64(i64::MIN),
        Value::I64(-42),
        Value::U64(u64::from(u32::MAX)),
        Value::U64(u64::MAX),
        Value::F64(-0.0),
        Value::F64(0.15625),
        Value::F64(-2.5e-8),
        Value::F64(1e300),
        Value::Str(Str::Static("plain")),
        Value::Str(Str::Dynamic(String::from(
            "with \"quotes\" and \\ backslash",
        ))),
        Value::Str(Str::Static("tab\tnewline\ncr\r")),
        Value::Str(Str::Static("\u{0}\u{1f}\u{7f}")),
        Value::Str(Str::Static("héllo → 😀")),
        Value::Str(Str::Static("linesep\u{2028}\u{2029}chars")),
        Value::Array(vec![]),
        Value::Array(vec![
            Value::I64(1),
            Value::Array(vec![Value::F64(2.0)]),
            object([("inner", Value::Bool(false))]),
        ]),
        object([
            ("", Value::I64(0)),
            ("z", Value::I64(26)),
            ("a", Value::U64(1)),
            (
                "nested",
                object([("deep", Value::Array(vec![Value::Null]))]),
            ),
        ]),
    ])
}

fn to_serde(value: &Value<'_>) -> serde_json::Value {
    match value {
        Value::Null => serde_json::Value::Null,
        Value::Bool(b) => serde_json::Value::Bool(*b),
        Value::I64(n) => (*n).into(),
        Value::U64(n) => (*n).into(),
        Value::F64(n) => {
            serde_json::Value::Number(serde_json::Number::from_f64(*n).expect("finite float"))
        },
        Value::Str(s) => serde_json::Value::String(s.as_str().to_owned()),
        Value::Array(items) => serde_json::Value::Array(items.iter().map(to_serde).collect()),
        Value::Object(members) => serde_json::Value::Object(
            members
                .iter()
                .map(|(k, v)| (k.as_str().to_owned(), to_serde(v)))
                .collect(),
        ),
    }
}

#[test]
fn pretty_formatting() {
    assert_eq!(to_string_pretty(&Value::Null), "null");
    assert_eq!(to_string_pretty(&Value::Array(vec![])), "[]");
    assert_eq!(to_string_pretty(&object([])), "{}");
    assert_eq!(to_string_pretty(&Value::F64(f64::NAN)), "NaN");
    // Empty containers stay inline, exactly like serde_json.
    let value = object([
        ("empty", Value::Array(vec![])),
        ("one", Value::I64(1)),
        (
            "nested",
            Value::Array(vec![object([("deep", Value::Bool(true))])]),
        ),
    ]);
    assert_eq!(
        to_string_pretty(&value),
        concat!(
            "{\n",
            "  \"empty\": [],\n",
            "  \"one\": 1,\n",
            "  \"nested\": [\n",
            "    {\n",
            "      \"deep\": true\n",
            "    }\n",
            "  ]\n",
            "}"
        )
    );
}

#[test]
fn pretty_matches_serde_json() {
    let value = sample();
    let expected = serde_json::to_string_pretty(&to_serde(&value)).unwrap();
    assert_eq!(to_string_pretty(&value), expected);
}

#[test]
fn compact_matches_serde_json() {
    let value = sample();
    let expected = serde_json::to_string(&to_serde(&value)).unwrap();
    assert_eq!(String::from_utf8(to_vec(&value)).unwrap(), expected);
}

/// Pretty indentation is served from a static space cache. Depths past a
/// 64-level cache boundary (and the parser's full depth cap) must still
/// produce the exact two-spaces-per-level layout, byte for byte.
#[test]
fn pretty_indent_deep_nesting() {
    fn expected(depth: usize) -> String {
        let mut out = String::from("[");
        for level in 1..depth {
            out.push('\n');
            out.push_str(&" ".repeat(2 * level));
            out.push('[');
        }
        out.push('\n');
        out.push_str(&" ".repeat(2 * depth));
        out.push('1');
        for level in (0..depth).rev() {
            out.push('\n');
            out.push_str(&" ".repeat(2 * level));
            out.push(']');
        }
        out
    }
    for depth in [65, 70, simd_json5::MAX_DEPTH] {
        let text = format!("{}1{}", "[".repeat(depth), "]".repeat(depth));
        let value = simd_json5::from_str(&text).unwrap();
        assert_eq!(to_string_pretty(&value), expected(depth), "depth = {depth}");
    }
}
