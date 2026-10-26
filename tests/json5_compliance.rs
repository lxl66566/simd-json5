//! Accept/reject compliance table for the JSON5 grammar pin in `DESIGN.md`.

use simd_json5::{Error, Str, Value};

fn obj(pairs: Vec<(&'static str, Value<'static>)>) -> Value<'static> {
    Value::Object(
        pairs
            .into_iter()
            .map(|(k, v)| (Str::Static(k), v))
            .collect(),
    )
}

fn syn(message: &str, offset: usize) -> Error {
    Error::Syntax {
        message: message.into(),
        offset,
    }
}

#[test]
fn accepted_documents() {
    let accepts: Vec<(&str, Value<'static>)> = vec![
        // Keywords and top-level scalars.
        ("null", Value::Null),
        ("true", Value::Bool(true)),
        ("false", Value::Bool(false)),
        ("Infinity", Value::F64(f64::INFINITY)),
        ("-Infinity", Value::F64(f64::NEG_INFINITY)),
        ("+Infinity", Value::F64(f64::INFINITY)),
        ("  null\t", Value::Null),
        // Decimal integers.
        ("0", Value::U64(0)),
        ("1", Value::U64(1)),
        ("-1", Value::I64(-1)),
        ("+1", Value::U64(1)),
        ("-0", Value::I64(0)),
        ("007", Value::U64(7)),
        ("18446744073709551615", Value::U64(u64::MAX)),
        (
            "18446744073709551616",
            Value::F64(1.844_674_407_370_955_2e19),
        ),
        ("-9223372036854775808", Value::I64(i64::MIN)),
        (
            "-9223372036854775809",
            Value::F64(-9.223_372_036_854_776e18),
        ),
        // Hex integers.
        ("0x0", Value::U64(0)),
        ("0x10", Value::U64(16)),
        ("0X1F", Value::U64(31)),
        ("-0xff", Value::I64(-255)),
        ("+0x10", Value::U64(16)),
        ("0xFFFFFFFFFFFFFFFF", Value::U64(u64::MAX)),
        ("-0x8000000000000000", Value::I64(i64::MIN)),
        (
            "0x10000000000000000",
            Value::F64(1.844_674_407_370_955_2e19),
        ),
        // Decimal floats, including the ECMAScript 5.1 elided-digit forms.
        (".5", Value::F64(0.5)),
        ("-.5", Value::F64(-0.5)),
        ("5.", Value::F64(5.0)),
        ("1.e3", Value::F64(1000.0)),
        ("1e3", Value::F64(1000.0)),
        ("1E+3", Value::F64(1000.0)),
        ("1e-3", Value::F64(0.001)),
        ("0.1", Value::F64(0.1)),
        ("1.7976931348623157e308", Value::F64(f64::MAX)),
        // Strings.
        (r#""abc""#, "abc".into()),
        ("'abc'", "abc".into()),
        (r#""""#, "".into()),
        ("''", "".into()),
        (
            r#""\b\f\n\r\t\v\0'\"\\\x41\x7e\u0041\u00e9""#,
            "\u{8}\u{c}\n\r\t\u{b}\0'\"\\A~Aé".into(),
        ),
        (r#""😀""#, "\u{1f600}".into()),
        (r#""\uD83D\uDE00""#, "\u{1f600}".into()),
        (r#""a\qb""#, "aqb".into()),
        (r#""\1\2""#, "12".into()),
        (r#""\0 ""#, "\0 ".into()),
        ("\"a\u{2028}b\u{2029}\"", "a\u{2028}b\u{2029}".into()),
        // Line continuations.
        ("\"a\\\nb\"", "ab".into()),
        ("\"a\\\r\nb\"", "ab".into()),
        ("\"a\\\rb\"", "ab".into()),
        ("\"a\\\u{2028}b\"", "ab".into()),
        ("\"a\\\u{2029}b\"", "ab".into()),
        ("'\\''", "'".into()),
        ("'\"'", "\"".into()),
        // Objects, keys and trailing commas.
        ("{}", obj(vec![])),
        ("{a:1}", obj(vec![("a", Value::U64(1))])),
        ("{ a : 1 }", obj(vec![("a", Value::U64(1))])),
        ("{'a':1}", obj(vec![("a", Value::U64(1))])),
        (r#"{"a":1}"#, obj(vec![("a", Value::U64(1))])),
        (r#"{"":1}"#, obj(vec![("", Value::U64(1))])),
        (
            "{$:1, _:2, if:3, null:4, true:5}",
            obj(vec![
                ("$", Value::U64(1)),
                ("_", Value::U64(2)),
                ("if", Value::U64(3)),
                ("null", Value::U64(4)),
                ("true", Value::U64(5)),
            ]),
        ),
        (
            "{é:1, 日本:2, a\\u0062:3, \\u0061:4}",
            obj(vec![
                ("é", Value::U64(1)),
                ("日本", Value::U64(2)),
                ("ab", Value::U64(3)),
                ("a", Value::U64(4)),
            ]),
        ),
        ("{a:1,}", obj(vec![("a", Value::U64(1))])),
        (
            "{a:1,b:2,}",
            obj(vec![("a", Value::U64(1)), ("b", Value::U64(2))]),
        ),
        // Duplicate keys keep the first position and the last value.
        ("{'a':1,'a':2}", obj(vec![("a", Value::U64(2))])),
        ("{a:1,a:2,a:3}", obj(vec![("a", Value::U64(3))])),
        // Raw control characters other than LF/CR are legal in strings.
        ("\"a\u{0}b\u{1}\"", "a\u{0}b\u{1}".into()),
        (
            "{a:{b:[1,{c:2}]}}",
            obj(vec![(
                "a",
                obj(vec![(
                    "b",
                    vec![Value::U64(1), obj(vec![("c", Value::U64(2))])].into(),
                )]),
            )]),
        ),
        // Arrays.
        ("[]", Value::Array(vec![])),
        ("[1]", vec![Value::U64(1)].into()),
        ("[1,]", vec![Value::U64(1)].into()),
        ("[1,2,]", vec![Value::U64(1), Value::U64(2)].into()),
        (
            "[[1,],[2,]]",
            vec![vec![Value::U64(1)].into(), vec![Value::U64(2)].into()].into(),
        ),
        (
            "[1,'a',{b:2},[3]]",
            vec![
                Value::U64(1),
                "a".into(),
                obj(vec![("b", Value::U64(2))]),
                vec![Value::U64(3)].into(),
            ]
            .into(),
        ),
        // Comments.
        ("// c\n1", Value::U64(1)),
        ("1// c", Value::U64(1)),
        ("/*c*/1", Value::U64(1)),
        ("1/*c*/", Value::U64(1)),
        ("/* multi\nline */[1]", vec![Value::U64(1)].into()),
        ("[1/*c*/,2]", vec![Value::U64(1), Value::U64(2)].into()),
        ("{/*x*/a:1}", obj(vec![("a", Value::U64(1))])),
        ("/* * / */2", Value::U64(2)),
        ("/* // */1", Value::U64(1)),
        ("//\n1", Value::U64(1)),
        // Unicode whitespace between tokens.
        ("[1\u{85},2]", vec![Value::U64(1), Value::U64(2)].into()),
        ("\u{feff}1", Value::U64(1)),
        ("[1\u{a0}]", vec![Value::U64(1)].into()),
        ("{\u{2028}}", obj(vec![])),
    ];
    for (src, want) in accepts {
        let got = simd_json5::from_str(src).unwrap_or_else(|e| panic!("{src:?}: {e}"));
        assert_eq!(got, want, "source {src:?}");
    }
}

#[test]
fn nan_literals_parse_to_nan() {
    for src in ["NaN", "+NaN", "-NaN"] {
        match simd_json5::from_str(src).unwrap() {
            Value::F64(f) => assert!(f.is_nan(), "{src}"),
            v => panic!("{src} parsed as {v:?}"),
        }
    }
}

#[test]
fn from_slice_validates_utf8() {
    let err = simd_json5::from_slice(&[b'"', 0xff, b'"']).unwrap_err();
    assert_eq!(err, Error::InvalidUtf8 { offset: 1 });
    assert_eq!(
        simd_json5::from_slice("\"ok\"".as_bytes()).unwrap(),
        "ok".into()
    );
}

#[test]
fn json5_readme_example() {
    let src = r#"{
  // comments
  unquoted: "and you can quote me on that",
  single: 'and "double" quotes',
  lineBreaks: "Look, Mom! \
No \n or \r required!",
  hexadecimal: 0xdecaf,
  leadingDecimal: .5, trailing: 5.,
  positiveSign: +1,
}"#;
    let v = simd_json5::from_str(src).unwrap();
    let m = v.as_object().unwrap();
    assert_eq!(m.len(), 7);
    assert_eq!(
        m.get("unquoted").unwrap().as_str(),
        Some("and you can quote me on that")
    );
    assert_eq!(
        m.get("single").unwrap().as_str(),
        Some("and \"double\" quotes")
    );
    assert_eq!(
        m.get("lineBreaks").unwrap().as_str(),
        Some("Look, Mom! No \n or \r required!")
    );
    assert_eq!(m.get("hexadecimal").unwrap(), &Value::U64(0xdecaf));
    assert_eq!(m.get("leadingDecimal").unwrap(), &Value::F64(0.5));
    assert_eq!(m.get("trailing").unwrap(), &Value::F64(5.0));
    assert_eq!(m.get("positiveSign").unwrap(), &Value::U64(1));
}

#[test]
fn rejected_documents() {
    let rejects: Vec<(&str, Error)> = vec![
        // Empty or comment-only input.
        ("", Error::UnexpectedEof { offset: 0 }),
        (" ", Error::UnexpectedEof { offset: 1 }),
        ("//", Error::UnexpectedEof { offset: 2 }),
        ("// */ 1", Error::UnexpectedEof { offset: 7 }),
        ("// /*", Error::UnexpectedEof { offset: 5 }),
        // Unknown literals.
        ("tru", syn("unknown literal", 0)),
        ("truex", syn("unknown literal", 0)),
        ("True", syn("unknown literal", 0)),
        ("undefined", syn("unknown literal", 0)),
        ("e3", syn("unknown literal", 0)),
        // Invalid numbers.
        ("1e", Error::InvalidNumber { offset: 0 }),
        ("1e+", Error::InvalidNumber { offset: 0 }),
        (".e3", Error::InvalidNumber { offset: 0 }),
        ("0x", Error::InvalidNumber { offset: 0 }),
        ("0xg", Error::InvalidNumber { offset: 0 }),
        ("0b1", Error::InvalidNumber { offset: 0 }),
        ("1.2.3", Error::InvalidNumber { offset: 0 }),
        ("1a", Error::InvalidNumber { offset: 0 }),
        ("+1x", Error::InvalidNumber { offset: 0 }),
        ("+", Error::InvalidNumber { offset: 0 }),
        ("-", Error::InvalidNumber { offset: 0 }),
        (".", Error::InvalidNumber { offset: 0 }),
        // Container shape.
        ("[1 2]", syn("expected ',' or ']'", 3)),
        ("[5 0x10]", syn("expected ',' or ']'", 3)),
        ("[,]", Error::UnexpectedChar {
            byte: b',',
            offset: 1,
        }),
        ("[1,,2]", Error::UnexpectedChar {
            byte: b',',
            offset: 3,
        }),
        ("{,}", Error::UnexpectedChar {
            byte: b',',
            offset: 1,
        }),
        ("{a:1,,}", Error::UnexpectedChar {
            byte: b',',
            offset: 5,
        }),
        ("[1}", Error::UnexpectedChar {
            byte: b'}',
            offset: 2,
        }),
        ("{a:1]", Error::UnexpectedChar {
            byte: b']',
            offset: 4,
        }),
        ("{a}", syn("expected ':'", 2)),
        ("{a:}", syn("expected value", 3)),
        (r#"{"a" 1}"#, syn("expected ':'", 5)),
        (r#"{"a":1 "b":2}"#, syn("expected ',' or '}'", 7)),
        ("{1:2}", syn("expected identifier or string key", 1)),
        ("{-:1}", syn("expected identifier or string key", 1)),
        ("{.5:1}", syn("expected identifier or string key", 1)),
        ("{a-b:1}", Error::UnexpectedChar {
            byte: b'-',
            offset: 2,
        }),
        // Unterminated containers.
        ("[1", Error::UnexpectedEof { offset: 2 }),
        ("[1,", Error::UnexpectedEof { offset: 3 }),
        ("{a:1", Error::UnexpectedEof { offset: 4 }),
        ("{", Error::UnexpectedEof { offset: 1 }),
        // Exactly one root value.
        ("1 2", Error::TrailingCharacters { offset: 2 }),
        ("5 5", Error::TrailingCharacters { offset: 2 }),
        ("null null", Error::TrailingCharacters { offset: 5 }),
        ("[1] [2]", Error::TrailingCharacters { offset: 4 }),
        // Stray structurals.
        ("}", Error::UnexpectedChar {
            byte: b'}',
            offset: 0,
        }),
        ("]", Error::UnexpectedChar {
            byte: b']',
            offset: 0,
        }),
        (":", Error::UnexpectedChar {
            byte: b':',
            offset: 0,
        }),
        (",", Error::UnexpectedChar {
            byte: b',',
            offset: 0,
        }),
        // Stage 1 rejects: illegal bytes and stray slashes.
        ("!", Error::UnexpectedChar {
            byte: b'!',
            offset: 0,
        }),
        ("[!]", Error::UnexpectedChar {
            byte: b'!',
            offset: 1,
        }),
        ("\u{1}", Error::UnexpectedChar { byte: 1, offset: 0 }),
        ("/x", Error::UnexpectedChar {
            byte: b'/',
            offset: 0,
        }),
        ("1 / 2", Error::UnexpectedChar {
            byte: b'/',
            offset: 2,
        }),
        ("/", Error::UnexpectedChar {
            byte: b'/',
            offset: 0,
        }),
        ("5 /", Error::UnexpectedChar {
            byte: b'/',
            offset: 2,
        }),
        ("/* unterminated", Error::UnterminatedComment { offset: 0 }),
        ("/*/ ", Error::UnterminatedComment { offset: 0 }),
        // Unterminated strings.
        ("\"abc", Error::UnterminatedString { offset: 0 }),
        ("'abc", Error::UnterminatedString { offset: 0 }),
        ("\"a\\\"", Error::UnterminatedString { offset: 0 }),
        // Raw line terminators in strings.
        ("\"a\nb\"", Error::UnexpectedChar {
            byte: b'\n',
            offset: 2,
        }),
        ("'a\rb'", Error::UnexpectedChar {
            byte: b'\r',
            offset: 2,
        }),
        // Invalid escapes inside strings.
        (r#""\01""#, Error::InvalidEscape { offset: 1 }),
        (r#""\x4""#, Error::InvalidEscape { offset: 1 }),
        (r#""\xZZ""#, Error::InvalidEscape { offset: 1 }),
        (r#""\u12""#, Error::InvalidUnicodeEscape { offset: 1 }),
        (r#""\uD800""#, Error::InvalidUnicodeEscape { offset: 1 }),
        (r#""\uD800x""#, Error::InvalidUnicodeEscape { offset: 1 }),
        (r#""\uD800\u0041""#, Error::InvalidUnicodeEscape {
            offset: 1,
        }),
        (r#""\uDC00""#, Error::InvalidUnicodeEscape { offset: 1 }),
        // Invalid escapes inside identifiers.
        (r"{a\x41: 1}", Error::InvalidEscape { offset: 2 }),
        (r"{\uZZZZ: 1}", Error::InvalidUnicodeEscape { offset: 1 }),
    ];
    for (src, want) in rejects {
        let got = simd_json5::from_str(src).unwrap_err();
        assert_eq!(got, want, "source {src:?}");
    }
}

#[test]
fn depth_limit_is_enforced() {
    const MAX_DEPTH: usize = 1024;
    let ok = format!("{}{}", "[".repeat(MAX_DEPTH), "]".repeat(MAX_DEPTH));
    assert!(simd_json5::from_str(&ok).is_ok());
    let deep = format!("{}{}", "[".repeat(MAX_DEPTH + 1), "]".repeat(MAX_DEPTH + 1));
    assert_eq!(
        simd_json5::from_str(&deep).unwrap_err(),
        Error::DepthLimitExceeded {
            limit: MAX_DEPTH,
            offset: MAX_DEPTH,
        }
    );
    // Objects are capped as well, and the cap counts nesting, not siblings.
    let wide = format!("[{},{}]", "[".repeat(MAX_DEPTH), "]".repeat(MAX_DEPTH));
    assert!(simd_json5::from_str(&wide).is_err());
}
