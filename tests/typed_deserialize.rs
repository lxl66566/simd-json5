//! Integration tests for typed (serde) deserialization, feature `serde`.

use std::collections::{BTreeMap, HashMap};

use serde::Deserialize;
use simd_json5::{Error, de, typed};

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
    assert!(matches!(&err, Error::Custom(m) if m.contains("invalid type: sequence")));
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
    assert!(matches!(err, Error::Custom(_)));
    let err = de::from_str::<E>(r#"{"Z": 1}"#).unwrap_err();
    assert!(matches!(err, Error::Custom(_)));
    // Unit variant with a payload, payload variant as a bare string.
    let err = de::from_str::<E>(r#"{"A": 1}"#).unwrap_err();
    assert!(matches!(err, Error::Custom(_)));
    let err = de::from_str::<E>(r#""B""#).unwrap_err();
    assert!(matches!(err, Error::Custom(_)));
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
    assert!(matches!(&err, Error::Custom(m) if m.contains("missing field `x`")));
    let err = de::from_str::<P>(r#"{"x":"s"}"#).unwrap_err();
    assert!(matches!(&err, Error::Custom(m) if m.contains("invalid type")));
    assert_eq!(de::from_str::<P>(r#"{"x":5}"#).unwrap().x, 5);
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
    assert!(matches!(&err, Error::Custom(m) if m.contains("expected a borrowed string")));
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
