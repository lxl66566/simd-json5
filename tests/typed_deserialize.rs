//! Integration tests for typed (serde) deserialization, feature `serde`.

#![cfg(feature = "serde")]

use std::collections::{BTreeMap, HashMap};

use serde::Deserialize;
use simd_json5::{Error, Value, de, from_str, typed};

#[derive(Deserialize, PartialEq, Debug)]
struct Inner {
    name: String,
    tags: Option<Vec<String>>,
}

#[derive(Deserialize, PartialEq, Debug)]
struct Outer {
    inner: Inner,
    counts: HashMap<String, i64>,
    ratios: BTreeMap<u8, f64>,
    fallback: Option<Inner>,
}

#[derive(Deserialize, PartialEq, Debug)]
struct Newtype(u32);

#[derive(Deserialize, PartialEq, Debug)]
enum E {
    A,
    B(i32),
    C(String),
    D(u8, u8),
    S { x: u8, y: u8 },
}

/// On pure JSON our typed walk must agree with serde_json value for value.
/// `-0` is deliberately absent: the JSON5 grammar pin keeps signless-zero
/// integers in the integer variants (I64(0)), while serde_json renders JSON
/// `-0` as the float -0.0.
#[test]
fn differential_against_serde_json() {
    let docs = [
        "null",
        "true",
        "false",
        "0",
        "12345678901234567890",
        "-9223372036854775808",
        "1.5",
        "-2.25",
        "1e10",
        "1e-3",
        "0.1",
        "1.7976931348623157e308",
        r#""""#,
        r#""abc""#,
        r#""\"\\\/\b\f\n\r\tA""#,
        r#""😀""#,
        "[]",
        "{}",
        "[1,2,3]",
        r#"{"a":1,"b":[true,null,"c"],"d":{"e":[]}}"#,
        "[[[[1]]]]",
        r#"{"x":{"y":{"z":"w"}}}"#,
        "[0.5,-2.25,1e-3]",
        r#"[{"a":1},{"a":2}]"#,
    ];
    for doc in docs {
        let ours: serde_json::Value = de::from_str(doc).unwrap();
        let reference: serde_json::Value = serde_json::from_str(doc).unwrap();
        assert_eq!(ours, reference, "doc {doc}");
    }
}

#[test]
fn nested_structs_and_collections() {
    let src = r#"{
        // nested with comments, unquoted keys and single quotes
        inner: { name: 'hal', tags: ['a', "b",], },
        counts: { x: 1, y: -2, },
        ratios: { "0": .5, "3": 5., "7": +7 },
        fallback: null,
    }"#;
    let v: Outer = de::from_str(src).unwrap();
    let expected = Outer {
        inner: Inner {
            name: "hal".into(),
            tags: Some(vec!["a".into(), "b".into()]),
        },
        counts: [("x".to_string(), 1), ("y".to_string(), -2)].into(),
        ratios: [(0, 0.5), (3, 5.0), (7, 7.0)].into(),
        fallback: None,
    };
    assert_eq!(v, expected);
}

#[test]
fn newtype_struct() {
    assert_eq!(de::from_str::<Newtype>("41").unwrap(), Newtype(41));
    let err = de::from_str::<Newtype>("[42]").unwrap_err();
    assert!(
        matches!(&err, Error::Syntax { message, .. } if message.contains("invalid type: sequence"))
    );
}

/// Newtype-struct and `Option` map keys work like serde_json's: keys are
/// always text (never `null`), so `Option` keys are always `Some`.
#[test]
fn newtype_and_option_map_keys() {
    #[derive(Deserialize, PartialEq, Eq, Hash, Debug)]
    struct Name(String);

    let m: HashMap<Name, u8> = de::from_str(r#"{ "a": 1, 'b': 2 }"#).unwrap();
    assert_eq!(m.get(&Name("a".into())), Some(&1));
    assert_eq!(m.get(&Name("b".into())), Some(&2));

    let m: HashMap<Option<String>, u8> = de::from_str(r#"{ "a": 1, "b": 2 }"#).unwrap();
    assert_eq!(m.get(&Some("a".to_string())), Some(&1));
    assert_eq!(m.len(), 2);
}

#[test]
fn enums_in_every_form() {
    // Unit variants as strings, either quote style.
    assert_eq!(de::from_str::<E>(r#""A""#).unwrap(), E::A);
    assert_eq!(de::from_str::<E>("'A'").unwrap(), E::A);
    // Map-wrapped newtype, tuple and struct variants; quoted and bare keys.
    assert_eq!(de::from_str::<E>(r#"{"B": -1}"#).unwrap(), E::B(-1));
    assert_eq!(de::from_str::<E>("{C: 'x'}").unwrap(), E::C("x".into()));
    assert_eq!(de::from_str::<E>(r#"{"D": [1, 2]}"#).unwrap(), E::D(1, 2));
    assert_eq!(de::from_str::<E>(r#"{"S": {x: 1, y: 2}}"#).unwrap(), E::S {
        x: 1,
        y: 2
    });
}

#[test]
fn enum_errors() {
    // Unknown variant names are serde errors.
    let err = de::from_str::<E>(r#""Z""#).unwrap_err();
    assert!(matches!(err, Error::Syntax { .. }));
    let err = de::from_str::<E>(r#"{"Z": 1}"#).unwrap_err();
    assert!(matches!(err, Error::Syntax { .. }));
    // Unit variant with a payload, payload variant as a bare string.
    let err = de::from_str::<E>(r#"{"A": 1}"#).unwrap_err();
    assert!(matches!(err, Error::Syntax { .. }));
    let err = de::from_str::<E>(r#""B""#).unwrap_err();
    assert!(matches!(err, Error::Syntax { .. }));
    // More than one member in the enum map.
    assert_eq!(
        de::from_str::<E>(r#"{"B":1,"B":2}"#).unwrap_err(),
        Error::TrailingCharacters { offset: 6 }
    );
}

#[test]
fn field_mismatch_is_a_serde_error() {
    #[derive(Deserialize, Debug)]
    struct P {
        x: i32,
    }
    let err = de::from_str::<P>(r#"{"y":1}"#).unwrap_err();
    assert!(matches!(&err, Error::Syntax { message, .. } if message.contains("missing field `x`")));
    let err = de::from_str::<P>(r#"{"x":"s"}"#).unwrap_err();
    assert!(matches!(&err, Error::Syntax { message, .. } if message.contains("invalid type")));
    assert_eq!(de::from_str::<P>(r#"{"x":5}"#).unwrap().x, 5);
}

/// serde visitor errors are re-reported with the read position, so typed
/// type errors carry offsets like every other parse error.
#[test]
fn type_errors_carry_offsets() {
    let err = de::from_str::<bool>("1").unwrap_err();
    assert!(
        matches!(&err, Error::Syntax { message, offset: 1 } if message.contains("invalid type"))
    );
    // Inside containers the cursor stands on the entry after the offending
    // value (here the closing bracket).
    let err = de::from_str::<Vec<bool>>("[true, 1]").unwrap_err();
    assert!(
        matches!(&err, Error::Syntax { message, offset: 8 } if message.contains("invalid type"))
    );
    // Grammar errors keep their own, more precise offsets.
    let err = de::from_str::<serde_json::Value>(r#"{"a": tru}"#).unwrap_err();
    assert!(
        matches!(&err, Error::Syntax { message, offset: 6 } if message.contains("unknown literal"))
    );
}

#[test]
fn json5_numbers_into_f64() {
    #[derive(Deserialize, Debug)]
    struct N {
        v: f64,
    }
    let cases = [
        ("{v: 0x1F}", 31.0),
        ("{v: -0x20}", -32.0),
        ("{v: .5}", 0.5),
        ("{v: 5.}", 5.0),
        ("{v: +7}", 7.0),
        ("{v: 1.e3}", 1000.0),
        ("{v: Infinity}", f64::INFINITY),
        ("{v: -Infinity}", f64::NEG_INFINITY),
        ("{v: 007}", 7.0),
    ];
    for (src, want) in cases {
        assert_eq!(de::from_str::<N>(src).unwrap().v, want, "src {src}");
    }
    assert!(de::from_str::<N>("{v: NaN}").unwrap().v.is_nan());
}

#[test]
fn syntax_error_offsets_survive() {
    let err = de::from_str::<serde_json::Value>("{a: tru}").unwrap_err();
    assert_eq!(err, Error::Syntax {
        message: "unknown literal".into(),
        offset: 4,
    });
    let err = de::from_str::<serde_json::Value>("[1 2]").unwrap_err();
    assert_eq!(err, Error::Syntax {
        message: "expected ',' or ']'".into(),
        offset: 3,
    });
    assert_eq!(
        de::from_str::<serde_json::Value>("'open").unwrap_err(),
        Error::UnterminatedString { offset: 0 }
    );
    assert_eq!(
        de::from_str::<serde_json::Value>("/* x").unwrap_err(),
        Error::UnterminatedComment { offset: 0 }
    );
    assert_eq!(
        de::from_str::<serde_json::Value>("").unwrap_err(),
        Error::UnexpectedEof { offset: 0 }
    );
}

#[test]
fn trailing_input_rejected() {
    assert_eq!(
        de::from_str::<u64>("1 2").unwrap_err(),
        Error::TrailingCharacters { offset: 2 }
    );
    assert_eq!(
        de::from_str::<serde_json::Value>("[1]]").unwrap_err(),
        Error::TrailingCharacters { offset: 3 }
    );
}

#[test]
fn invalid_utf8_in_from_slice() {
    let err = de::from_slice::<serde_json::Value>(b"\xff\xfe").unwrap_err();
    assert_eq!(err, Error::InvalidUtf8 { offset: 0 });
    let ok: Vec<u8> = de::from_slice(b"[1,2]").unwrap();
    assert_eq!(ok, vec![1, 2]);
}

/// Visitor recursion needs stack proportional to nesting, so drive the
/// near-limit documents on a roomy thread.
#[test]
fn depth_limit() {
    std::thread::Builder::new()
        .stack_size(32 << 20)
        .spawn(|| {
            const LIMIT: usize = simd_json5::MAX_DEPTH;
            let flat = format!("{}{}", "[".repeat(LIMIT), "]".repeat(LIMIT));
            let deep = format!("{}{}", "[".repeat(LIMIT + 1), "]".repeat(LIMIT + 1));
            assert!(de::from_str::<serde::de::IgnoredAny>(&flat).is_ok());
            assert_eq!(
                de::from_str::<serde::de::IgnoredAny>(&deep).unwrap_err(),
                Error::DepthLimitExceeded {
                    limit: LIMIT,
                    offset: LIMIT,
                }
            );
        })
        .unwrap()
        .join()
        .unwrap();
}

/// A borrowed `&'de str` field only deserializes when the deserializer
/// offers `visit_borrowed_str`, which our walk does for escape-free strings.
/// Escaped strings error exactly like serde_json (serde 1.0.216+ always
/// owns `Cow`, so `&str` is the observable zero-copy surface).
#[test]
fn zero_copy_borrowed_and_owned() {
    #[derive(Deserialize, Debug)]
    struct S<'a> {
        s: &'a str,
    }
    let borrowed: S = de::from_str(r#"{ s: "borrowed" }"#).unwrap();
    assert_eq!(borrowed.s, "borrowed");
    let err = de::from_str::<S>(r#"{ s: "bo\u0072rowed" }"#).unwrap_err();
    assert!(
        matches!(&err, Error::Syntax { message, .. } if message.contains("expected a borrowed string"))
    );
    // The same document is fine for owned targets.
    let owned: Inner = de::from_str(r#"{ name: "bo\u0072rowed", tags: null }"#).unwrap();
    assert_eq!(owned.name, "borrowed");
}

#[test]
fn typed_facade_delegates_to_de() {
    let v: Vec<u8> = typed::from_str("[1, 2, 3]").unwrap();
    assert_eq!(v, vec![1, 2, 3]);
    let n: u32 = typed::from_slice(b" 7 ").unwrap();
    assert_eq!(n, 7);
    let err = typed::from_str::<u8>("nope").unwrap_err();
    assert_eq!(err, Error::Syntax {
        message: "unknown literal".into(),
        offset: 0,
    });
}

/// The serde walk's number fast path must agree with the DOM parse for every
/// number form, including sign, keyword, hex and overflow handling. Debug
/// renderings are compared because `F64(NaN)` never equals itself.
#[test]
fn number_fast_path_matches_dom() {
    let docs = [
        "0",
        "-0",
        "+0",
        "1",
        "-1",
        "+1",
        "007",
        "0x10",
        "0X1f",
        "-0x10",
        "+0x10",
        "0xFFFFFFFFFFFFFFFF",
        "-0x8000000000000000",
        "0x1FFFFFFFFFFFFFFFF",
        "-0x8000000000000001",
        "18446744073709551615",
        "18446744073709551616",
        "-9223372036854775808",
        "-9223372036854775809",
        ".5",
        "-.5",
        "5.",
        "1.e3",
        "1.5",
        "1e3",
        "1E3",
        "1e+3",
        "1e-3",
        "0.1",
        "9007199254740993.0",
        "1e309",
        "-1e309",
        "1e-400",
        "5e-324",
        "Infinity",
        "+Infinity",
        "-Infinity",
        "NaN",
        "-NaN",
        "+NaN",
        "[.5, 5., 0x10, -Infinity, 18446744073709551616]",
        "{ a: 1.5, b: -0, c: 0x1f }",
    ];
    for d in docs {
        assert_eq!(
            format!("{:?}", de::from_str::<Value>(d).unwrap()),
            format!("{:?}", from_str(d).unwrap()),
            "doc {d}"
        );
    }
}

/// Typed number targets exercise the fast path across every visitor hint:
/// floats via `visit_f64`, integers via `visit_u64`/`visit_i64` and the
/// i128/u128 widening.
#[test]
fn typed_number_fast_path_targets() {
    let cases = [
        ("Infinity", f64::INFINITY),
        ("-Infinity", f64::NEG_INFINITY),
        ("0x10", 16.0),
        ("-0x10", -16.0),
        (".5", 0.5),
        ("5.", 5.0),
        ("1.e3", 1000.0),
        ("1e3", 1000.0),
        ("18446744073709551615", 1.844_674_407_370_955_2e19), // u64::MAX hint
        ("18446744073709551616", 1.844_674_407_370_955_2e19), // overflow -> f64
    ];
    for (src, want) in cases {
        assert_eq!(de::from_str::<f64>(src).unwrap(), want, "src {src}");
    }
    assert!(de::from_str::<f64>("NaN").unwrap().is_nan());
    // `-0` stays in the integer variants per the grammar pin, so the f64
    // value is positive zero.
    let zero = de::from_str::<f64>("-0").unwrap();
    assert!((zero == 0.0) && !zero.is_sign_negative());

    // Integer hints survive the fast path unchanged, including errors.
    assert_eq!(de::from_str::<u64>("0x10").unwrap(), 16);
    assert_eq!(de::from_str::<i64>("-0x10").unwrap(), -16);
    let err = de::from_str::<i64>("1.5").unwrap_err();
    assert!(matches!(&err, Error::Syntax { message, .. } if message.contains("invalid type")));
    // serde's u64 visitor reports a negative i64 hint as a value error.
    let err = de::from_str::<u64>("-1").unwrap_err();
    assert!(matches!(&err, Error::Syntax { message, .. } if message.contains("invalid value")));

    // i128/u128 widen the u64/i64 hints exactly like the slow path.
    assert_eq!(
        de::from_str::<u128>("0xFFFFFFFFFFFFFFFF").unwrap(),
        u128::from(u64::MAX)
    );
    assert_eq!(
        de::from_str::<i128>("-9223372036854775808").unwrap(),
        i128::from(i64::MIN)
    );
}

/// Number grammar errors keep their variant and byte offset on the typed
/// walk, both at the root and inside containers.
#[test]
fn number_error_offsets_match_dom() {
    for d in ["1a", "0x", "1e", "1.2.3", ".e3", "--1", "+"] {
        assert_eq!(
            de::from_str::<f64>(d).unwrap_err(),
            from_str(d).unwrap_err(),
            "doc {d}"
        );
    }
    for d in ["[1a]", "[0x]", "[.e3]", "{a: 1e}", "{a: 1.2.3}"] {
        assert_eq!(
            de::from_str::<serde_json::Value>(d).unwrap_err(),
            from_str(d).unwrap_err(),
            "doc {d}"
        );
    }
    assert_eq!(
        de::from_str::<Vec<f64>>("[1a]").unwrap_err(),
        Error::InvalidNumber { offset: 1 }
    );
}
