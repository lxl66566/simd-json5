//! Integration tests for typed (serde) deserialization, feature `serde`.

#![cfg(feature = "serde")]

use std::{
    collections::{BTreeMap, HashMap},
    hash::{Hash, Hasher},
};

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

/// Float-keyed maps need a wrapper: f32/f64 are neither `Hash` nor `Ord`, so
/// no standard collection key type reaches the float map-key branches. A
/// manual `Deserialize` calling the float key methods directly is the way
/// in (the same route a newtype wrapper over a float takes through
/// `deserialize_newtype_struct`).
#[derive(PartialEq, Debug)]
struct F64Key(f64);

impl Eq for F64Key {}

impl Hash for F64Key {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl<'de> Deserialize<'de> for F64Key {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct V;
        impl serde::de::Visitor<'_> for V {
            type Value = F64Key;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a float map key")
            }

            fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E> {
                Ok(F64Key(v))
            }
        }
        deserializer.deserialize_f64(V)
    }
}

/// The f32 flavor: `deserialize_f32` parses the key text at f32 precision
/// directly, so the value arrives through `visit_f32`.
#[derive(PartialEq, Debug)]
struct F32Key(f32);

impl Eq for F32Key {}

impl Hash for F32Key {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl<'de> Deserialize<'de> for F32Key {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct V;
        impl serde::de::Visitor<'_> for V {
            type Value = F32Key;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a float map key")
            }

            fn visit_f32<E>(self, v: f32) -> Result<Self::Value, E> {
                Ok(F32Key(v))
            }
        }
        deserializer.deserialize_f32(V)
    }
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

/// 128-bit integer targets parse exactly past the u64/i64 variants, wherever
/// the literal sits; literals beyond the 128-bit range are number errors,
/// and range/type mismatches stay serde errors.
#[test]
fn wide_integer_targets() {
    #[derive(Deserialize, Debug, PartialEq)]
    struct Wide {
        a: u128,
        b: i128,
    }
    let v: Wide =
        de::from_str("{ a: 18446744073709551616, b: -170141183460469231731687303715884105728, }")
            .unwrap();
    assert_eq!(v, Wide {
        a: 1_u128 << 64,
        b: i128::MIN
    });
    assert_eq!(
        de::from_str::<u128>("340282366920938463463374607431768211455").unwrap(),
        u128::MAX
    );
    assert_eq!(
        de::from_str::<u128>("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF").unwrap(),
        u128::MAX
    );
    assert_eq!(
        de::from_str::<i128>("170141183460469231731687303715884105727").unwrap(),
        i128::MAX
    );
    assert_eq!(
        de::from_str::<Vec<u128>>("[1, 18446744073709551616, 0x20]").unwrap(),
        vec![1, 1_u128 << 64, 32]
    );

    // Past the 128-bit range: InvalidNumber at the token, like grammar errors.
    assert_eq!(
        de::from_str::<u128>("340282366920938463463374607431768211456").unwrap_err(),
        Error::InvalidNumber { offset: 0 }
    );
    assert_eq!(
        de::from_str::<Wide>("{ a: 1, b: -170141183460469231731687303715884105729 }").unwrap_err(),
        Error::InvalidNumber { offset: 11 }
    );
    assert_eq!(
        de::from_str::<u128>("0x100000000000000000000000000000000").unwrap_err(),
        Error::InvalidNumber { offset: 0 }
    );

    // Range and sign mismatches are serde value errors, not parse errors.
    let err = de::from_str::<i128>("340282366920938463463374607431768211455").unwrap_err();
    assert!(
        matches!(&err, Error::Syntax { message, .. } if message.contains("invalid value")),
        "{err:?}"
    );
    let err = de::from_str::<u128>("-1").unwrap_err();
    assert!(
        matches!(&err, Error::Syntax { message, .. } if message.contains("invalid value")),
        "{err:?}"
    );

    // Float and keyword spellings stay type errors for integer visitors.
    for doc in ["1.0", "1e3", "Infinity", "-Infinity", "NaN"] {
        let err = de::from_str::<u128>(doc).unwrap_err();
        assert!(
            matches!(&err, Error::Syntax { message, .. } if message.contains("invalid type")),
            "doc {doc}: {err:?}"
        );
    }
}

/// f32 targets evaluate document numbers at f32 precision in a single
/// rounding, not through an f64 round trip.
#[test]
fn f32_targets_evaluate_at_f32_precision() {
    assert_eq!(de::from_str::<f32>("1.5").unwrap(), 1.5);
    assert_eq!(de::from_str::<f32>(".5").unwrap(), 0.5);
    assert_eq!(de::from_str::<f32>("1.e3").unwrap(), 1000.0);
    assert_eq!(de::from_str::<f32>("0x10").unwrap(), 16.0);
    assert_eq!(
        de::from_str::<f32>("18446744073709551616").unwrap(),
        1.844_674_4e19
    );
    assert_eq!(de::from_str::<f32>("Infinity").unwrap(), f32::INFINITY);
    assert_eq!(de::from_str::<f32>("-Infinity").unwrap(), f32::NEG_INFINITY);
    assert!(de::from_str::<f32>("NaN").unwrap().is_nan());
    // The double-rounding boundary: see the number.rs test of the same name.
    assert_eq!(
        de::from_str::<f32>("1.0000000596046448").unwrap(),
        1.000_000_1
    );
    assert_eq!(
        de::from_str::<Vec<f32>>("[1.0000000596046448, 1.0]").unwrap(),
        vec![1.000_000_1, 1.0]
    );
    // Non-number inputs keep the f64 method's type errors.
    let err = de::from_str::<f32>("'x'").unwrap_err();
    assert!(
        matches!(&err, Error::Syntax { message, .. } if message.contains("invalid type: string"))
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

/// Bytes-expecting methods accept a JSON5 string, like serde_json's: its
/// (unescaped) bytes reach the visitor through `visit_borrowed_bytes`/
/// `visit_bytes`. Which types benefit is serde's choice, exactly as with
/// serde_json: `&[u8]` and `CString` request the bytes methods, while
/// `Vec<u8>`/`Box<[u8]>` deserialize as plain sequences and keep rejecting
/// strings on both sides.
#[test]
fn bytes_accept_string_form() {
    // Escape-free content borrows; either quote style.
    let v: &[u8] = de::from_str(r#""abc""#).unwrap();
    assert_eq!(v, b"abc");
    let v: &[u8] = de::from_str("'xy'").unwrap();
    assert_eq!(v, b"xy");
    assert_eq!(serde_json::from_str::<&[u8]>(r#""abc""#).unwrap(), b"abc");
    // Escaped content unescapes first and arrives as owned bytes, which
    // serde's borrowed-slice visitor rejects — the serde_json outcome.
    for doc in [r#""a\nb""#, r#""a\u0062c""#] {
        let err = de::from_str::<&[u8]>(doc).unwrap_err();
        assert!(
            matches!(&err, Error::Syntax { message, .. } if message
                .contains("invalid type: byte array, expected a borrowed byte array")),
            "doc {doc}: {err:?}"
        );
        assert!(serde_json::from_str::<&[u8]>(doc).is_err(), "doc {doc}");
    }
    // `CString` is the standard type requesting `deserialize_byte_buf`.
    let c: std::ffi::CString = de::from_str(r#""ab""#).unwrap();
    assert_eq!(c.to_bytes(), b"ab");
    assert_eq!(
        serde_json::from_str::<std::ffi::CString>(r#""ab""#)
            .unwrap()
            .to_bytes(),
        b"ab"
    );
    let err = de::from_str::<std::ffi::CString>(r#""a\u0000b""#).unwrap_err();
    assert!(
        matches!(&err, Error::Syntax { message, .. } if message.contains("nul byte")),
        "{err:?}"
    );
    // Sequence targets keep rejecting strings, like serde_json.
    for doc in [r#""abc""#, "'xy'"] {
        let err = de::from_str::<Vec<u8>>(doc).unwrap_err();
        assert!(
            matches!(&err, Error::Syntax { message, .. } if message
                .contains("invalid type: string \"")),
            "doc {doc}: {err:?}"
        );
        assert!(serde_json::from_str::<Vec<u8>>(doc).is_err(), "doc {doc}");
    }
    // The array form still works.
    assert_eq!(de::from_str::<Vec<u8>>("[1, 2, 3]").unwrap(), vec![1, 2, 3]);
    // Other inputs stay serde type errors.
    for doc in ["42", "true", "null", "1.5", "{}"] {
        let err = de::from_str::<Vec<u8>>(doc).unwrap_err();
        assert!(
            matches!(&err, Error::Syntax { message, .. } if message.contains("invalid type")),
            "doc {doc}: {err:?}"
        );
    }
}

/// A map parsed from `doc` (always a literal, so the keys may borrow) must
/// hold exactly one key equal to `expected`.
fn assert_single_key<T>(doc: &'static str, expected: &T)
where
    T: Deserialize<'static> + Hash + Eq + PartialEq + std::fmt::Debug,
{
    let m: HashMap<T, u8> = de::from_str(doc).unwrap();
    assert_eq!(m.len(), 1, "doc {doc}");
    let key = m.into_keys().next().unwrap();
    assert_eq!(&key, expected, "doc {doc}");
}

/// A map whose single key fails to parse as type `$ty` must produce an
/// `invalid value` error naming `$expecting`.
macro_rules! bad_key {
    ($ty:ty, $expecting:expr, $doc:expr) => {{
        let err = de::from_str::<HashMap<$ty, u8>>($doc).unwrap_err();
        assert!(
            matches!(&err, Error::Syntax { message, .. } if message.contains("invalid value")
                && message.contains($expecting)),
            "doc {}: {err:?}",
            $doc
        );
    }};
}

/// The hand-written bool and numeric branches of the map-key deserializer:
/// quoted key text is parsed by the target key type. Numbers use Rust's
/// `FromStr` grammar — looser than serde_json's JSON-grammar numeric keys —
/// and this test pins that behavior, including the divergences.
#[test]
fn map_key_bool_and_number_branches() {
    // Bool keys: quoted or bare identifier, exactly `true`/`false`.
    let m: HashMap<bool, u8> = de::from_str(r#"{ "true": 1, "false": 2 }"#).unwrap();
    assert_eq!((m[&true], m[&false]), (1, 2));
    let m: HashMap<bool, u8> = de::from_str("{ true: 1, false: 2 }").unwrap();
    assert_eq!((m[&true], m[&false]), (1, 2));
    // A failing bool key is an `invalid type` error like serde_json's.
    // (serde_json's own message truncates the key text — it consumes the
    // first byte before giving up — so only the error kind is comparable.)
    let err = de::from_str::<HashMap<bool, u8>>(r#"{ "yes": 1 }"#).unwrap_err();
    assert!(
        matches!(&err, Error::Syntax { message, .. } if message
            .contains("invalid type: string \"yes\", expected a boolean")),
        "{err:?}"
    );
    let reference = serde_json::from_str::<HashMap<bool, u8>>(r#"{ "yes": 1 }"#).unwrap_err();
    assert!(
        reference.to_string().contains("invalid type: string")
            && reference.to_string().contains("expected a boolean"),
        "{reference}"
    );

    // Integer keys of every width, at each boundary.
    assert_single_key::<i8>(r#"{ "127": 1 }"#, &127);
    assert_single_key::<i8>(r#"{ "-128": 1 }"#, &-128);
    assert_single_key::<i16>(r#"{ "-32768": 1 }"#, &-32768);
    assert_single_key::<i32>(r#"{ "2147483647": 1 }"#, &i32::MAX);
    assert_single_key::<i64>(r#"{ "-9223372036854775808": 1 }"#, &i64::MIN);
    assert_single_key::<u8>(r#"{ "255": 1 }"#, &u8::MAX);
    assert_single_key::<u16>(r#"{ "65535": 1 }"#, &u16::MAX);
    assert_single_key::<u32>(r#"{ "4294967295": 1 }"#, &u32::MAX);
    assert_single_key::<u64>(r#"{ "18446744073709551615": 1 }"#, &u64::MAX);
    assert_single_key::<i128>(
        r#"{ "170141183460469231731687303715884105727": 1 }"#,
        &i128::MAX,
    );
    assert_single_key::<u128>(
        r#"{ "340282366920938463463374607431768211455": 1 }"#,
        &u128::MAX,
    );
    // Float keys through the wrapper types, including f64::MAX's decimal
    // form and the f32 direct parse.
    assert_single_key::<F64Key>(r#"{ "1.5e3": 1 }"#, &F64Key(1500.0));
    assert_single_key::<F64Key>(r#"{ "1.7976931348623157e308": 1 }"#, &F64Key(f64::MAX));
    assert_single_key::<F32Key>(r#"{ "2.5": 1 }"#, &F32Key(2.5));
    // The f32 key parse rounds once: this decimal sits just above the f32
    // midpoint 1 + 2^-24, so direct f32 rounds up while an f64 round trip
    // would tie to even (1.0).
    assert_single_key::<F32Key>(r#"{ "1.0000000596046448": 1 }"#, &F32Key(1.000_000_1));

    // FromStr accepts a leading `+` for integers too, and the float
    // spellings `Infinity`/`NaN`; serde_json's numeric keys reject all of
    // these (their grammar is JSON's).
    assert_single_key::<i64>(r#"{ "+1": 1 }"#, &1);
    assert_single_key::<u64>(r#"{ "+1": 1 }"#, &1);
    assert_single_key::<F64Key>(r#"{ "Infinity": 1 }"#, &F64Key(f64::INFINITY));
    assert_single_key::<F64Key>(r#"{ "-Infinity": 1 }"#, &F64Key(f64::NEG_INFINITY));
    let m: HashMap<F64Key, u8> = de::from_str(r#"{ "NaN": 1 }"#).unwrap();
    assert!(m.into_keys().next().unwrap().0.is_nan());
    assert!(serde_json::from_str::<HashMap<i64, u8>>(r#"{ "+1": 1 }"#).is_err());
    assert!(serde_json::from_str::<HashMap<F64Key, u8>>(r#"{ "Infinity": 1 }"#).is_err());
    assert!(serde_json::from_str::<HashMap<F64Key, u8>>(r#"{ "NaN": 1 }"#).is_err());

    // Failures are `invalid value` errors naming the expected type.
    bad_key!(i8, "expected i8", r#"{ "128": 1 }"#);
    bad_key!(u8, "expected u8", r#"{ "-1": 1 }"#);
    bad_key!(i64, "expected i64", r#"{ "x": 1 }"#);
    bad_key!(i64, "expected i64", r#"{ "1.5": 1 }"#);
    bad_key!(i64, "expected i64", r#"{ "Infinity": 1 }"#);
    bad_key!(F64Key, "expected f64", r#"{ "0x10": 1 }"#);
    bad_key!(F64Key, "expected f64", r#"{ "abc": 1 }"#);
    // The f32 key error names f32, not f64.
    bad_key!(F32Key, "expected f32", r#"{ "0x10": 1 }"#);
    bad_key!(F32Key, "expected f32", r#"{ "abc": 1 }"#);
    bad_key!(
        i128,
        "expected i128",
        r#"{ "340282366920938463463374607431768211456": 1 }"#
    );
}

/// Borrowed `&str` map keys take the `visit_borrowed_str` path (escape-free
/// keys); escaped keys own their text and cannot satisfy a `&str` key,
/// exactly like escaped values for `&str` fields.
#[test]
fn borrowed_str_map_keys() {
    let m: HashMap<&str, u8> = de::from_str(r#"{ "a": 1, 'b': 2 }"#).unwrap();
    assert_eq!(m.get("a"), Some(&1));
    assert_eq!(m.get("b"), Some(&2));
    let err = de::from_str::<HashMap<&str, u8>>(r#"{ "a\nb": 1 }"#).unwrap_err();
    assert!(
        matches!(&err, Error::Syntax { message, .. } if message.contains("expected a borrowed string")),
        "{err:?}"
    );
}

/// `#[serde(flatten)]` buffers unknown members through `deserialize_any`
/// and replays them through serde's flat-map adapter; both directions must
/// work on top of our map access.
#[test]
fn flattened_struct_field() {
    #[derive(Deserialize, PartialEq, Debug)]
    struct Flat {
        name: String,
        #[serde(flatten)]
        extra: BTreeMap<String, u32>,
    }

    // Container values inside the flattened tail exercise the buffered
    // value forms (arrays, objects, null, strings, floats).
    #[derive(Deserialize, PartialEq, Debug)]
    struct Deep {
        id: u8,
        #[serde(flatten)]
        rest: BTreeMap<String, serde_json::Value>,
    }

    let v: Flat = de::from_str(r"{ name: 'x', a: 1, b: 2, }").unwrap();
    assert_eq!(v, Flat {
        name: "x".into(),
        extra: [("a".to_string(), 1), ("b".to_string(), 2)].into(),
    });
    let v: Deep = de::from_str(r"{ id: 7, tags: ['a'], nested: { k: null }, }").unwrap();
    assert_eq!(v.id, 7);
    assert_eq!(v.rest["tags"], serde_json::json!(["a"]));
    assert_eq!(v.rest["nested"], serde_json::json!({ "k": null }));

    // A known field still wins over the flattened tail, and a missing one
    // is still an error.
    let err = de::from_str::<Flat>(r#"{ "a": 1 }"#).unwrap_err();
    assert!(
        matches!(&err, Error::Syntax { message, .. } if message.contains("missing field `name`")),
        "{err:?}"
    );
}
