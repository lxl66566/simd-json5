//! Typed (serde) serialization through the `typed` facade: exact output,
//! pretty layout, non-finite literals, map-key rejection, DOM equivalence
//! with the direct serializers, and a differential check against serde_json.

#![cfg(feature = "serde")]

use std::collections::BTreeMap;

use serde::Serialize;
use simd_json5::{Map, Str, Value, to_string, to_string_pretty, typed};

#[derive(Serialize)]
struct Sample {
    name: &'static str,
    escapes: &'static str,
    score: f64,
    ratio: f64,
    tags: Vec<&'static str>,
    empty: Vec<&'static str>,
    parent: Option<&'static str>,
    child: Option<Box<SampleBox>>,
}

#[derive(Serialize)]
struct SampleBox {
    depth: u32,
}

#[derive(Serialize)]
enum Shape {
    Unit,
    Newtype(f64),
    Tuple(f64, f64),
    Struct { radius: u32 },
}

fn sample() -> Sample {
    Sample {
        name: "typed",
        escapes: "tab\t\"quoted\"\n",
        score: 0.25,
        ratio: f64::NAN,
        tags: vec!["a", "b"],
        empty: vec![],
        parent: Some("root"),
        child: Some(Box::new(SampleBox { depth: 1 })),
    }
}

#[test]
fn facade_exact_output() {
    let text = typed::to_string(&sample()).unwrap();
    assert_eq!(
        text,
        concat!(
            r#"{"name":"typed","escapes":"tab\t\"quoted\"\n","score":0.25,"ratio":NaN,"#,
            r#""tags":["a","b"],"empty":[],"parent":"root","child":{"depth":1}}"#
        )
    );
    assert_eq!(typed::to_vec(&sample()).unwrap(), text.into_bytes());
    let text = typed::to_string(&f64::NEG_INFINITY).unwrap();
    assert_eq!(text, "-Infinity");
}

#[test]
fn facade_pretty_output() {
    let text = typed::to_string_pretty(&sample()).unwrap();
    assert_eq!(
        text,
        concat!(
            "{\n",
            "  \"name\": \"typed\",\n",
            "  \"escapes\": \"tab\\t\\\"quoted\\\"\\n\",\n",
            "  \"score\": 0.25,\n",
            "  \"ratio\": NaN,\n",
            "  \"tags\": [\n    \"a\",\n    \"b\"\n  ],\n",
            "  \"empty\": [],\n",
            "  \"parent\": \"root\",\n",
            "  \"child\": {\n    \"depth\": 1\n  }\n",
            "}"
        )
    );
}

#[test]
fn enum_forms_match_serde_json() {
    // All enum forms are JSON-representable; serde_json is the reference.
    let cases: [(&Shape, String); 4] = [
        (&Shape::Unit, serde_json::to_string(&Shape::Unit).unwrap()),
        (
            &Shape::Newtype(0.5),
            serde_json::to_string(&Shape::Newtype(0.5)).unwrap(),
        ),
        (
            &Shape::Tuple(1.0, 2.0),
            serde_json::to_string(&Shape::Tuple(1.0, 2.0)).unwrap(),
        ),
        (
            &Shape::Struct { radius: 3 },
            serde_json::to_string(&Shape::Struct { radius: 3 }).unwrap(),
        ),
    ];
    for (value, expected) in cases {
        assert_eq!(typed::to_string(value).unwrap(), expected);
        assert_eq!(
            typed::to_string_pretty(value).unwrap(),
            serde_json::to_string_pretty(value).unwrap()
        );
    }
}

#[test]
fn non_string_map_keys_rejected() {
    let mut int_keyed = BTreeMap::new();
    int_keyed.insert(1u64, "v");
    let error = typed::to_string(&int_keyed).unwrap_err();
    assert!(
        matches!(&error, simd_json5::Error::Custom(message) if &**message == "map keys must be strings"),
        "unexpected error: {error:?}"
    );
}

/// Object members shared by the DOM fixtures.
fn dom_object() -> Map<'static> {
    let mut object = Map::new();
    object.insert("z".into(), Value::I64(26));
    object.insert("a".into(), Value::F64(-0.5));
    object.insert("s".into(), Value::Str(Str::Static("esc\"aped")));
    object
}

/// DOM built programmatically, covering every variant.
fn dom() -> Value<'static> {
    Value::Array(vec![
        Value::Null,
        Value::Bool(true),
        Value::I64(i64::MIN),
        Value::U64(u64::MAX),
        Value::F64(2.5),
        Value::F64(f64::INFINITY),
        Value::Str(Str::Static("plain")),
        Value::Array(vec![]),
        Value::Object(dom_object()),
    ])
}

#[test]
fn dom_through_serde_equals_direct_serializer() {
    let value = dom();
    // Compact and pretty must be byte-identical between the typed facade
    // (serde machinery) and the direct DOM writers.
    assert_eq!(typed::to_string(&value).unwrap(), to_string(&value));
    assert_eq!(
        typed::to_string_pretty(&value).unwrap(),
        to_string_pretty(&value)
    );
    // Our DOM also serializes through serde_json without conversion.
    // serde_json writes non-finite floats as null, so this DOM is finite.
    let finite = Value::Array(vec![
        Value::I64(1),
        Value::F64(-2.5),
        Value::Str(Str::Static("s")),
        Value::Object(dom_object()),
    ]);
    let through_serde_json = serde_json::to_string(&finite).unwrap();
    assert_eq!(through_serde_json, to_string(&finite));
}

#[test]
fn str_serialize() {
    let text = typed::to_string(&Str::Static("s\tv")).unwrap();
    assert_eq!(text, r#""s\tv""#);
}

/// f32 values take ryu's f32 shortest path, byte-identical to serde_json;
/// widening to f64 first would print `0.1f32` as `0.10000000149011612`.
#[test]
fn f32_matches_serde_json_and_roundtrips() {
    for v in [
        0.1f32,
        -0.1f32,
        1.0 / 3.0,
        f32::MIN_POSITIVE,
        1e-40,
        16_777_217.0,
    ] {
        let text = typed::to_string(&v).unwrap();
        assert_eq!(text, serde_json::to_string(&v).unwrap(), "value {v}");
        assert_eq!(
            simd_json5::de::from_str::<f32>(&text).unwrap(),
            v,
            "text {text}"
        );
    }
    // Non-finite f32 keeps the JSON5 literals.
    assert_eq!(typed::to_string(&f32::NAN).unwrap(), "NaN");
    assert_eq!(typed::to_string(&f32::INFINITY).unwrap(), "Infinity");
    assert_eq!(typed::to_string(&f32::NEG_INFINITY).unwrap(), "-Infinity");
}

/// serde_json formats float exponents through the zmij fork (`1e+300`),
/// while DESIGN pins our floats to ryu (`1e300`); normalize both sides.
fn normalize_exponent(text: &str) -> String {
    text.replace("e+", "e")
}

#[test]
fn differential_against_serde_json_on_values() {
    let fixtures = vec![
        serde_json::json!(null),
        serde_json::json!([1, -2.5, "x", true, null]),
        serde_json::json!({"k": [1.0, {"deep": "v"}], "empty": {}, "arr": []}),
        serde_json::json!(1e300),
        serde_json::json!({"esc": "a\"b\\c\n\t\u{7f}\u{2028}"}),
    ];
    for fixture in fixtures {
        let expected = normalize_exponent(&serde_json::to_string(&fixture).unwrap());
        assert_eq!(
            normalize_exponent(&typed::to_string(&fixture).unwrap()),
            expected
        );
        let expected = normalize_exponent(&serde_json::to_string_pretty(&fixture).unwrap());
        assert_eq!(
            normalize_exponent(&typed::to_string_pretty(&fixture).unwrap()),
            expected
        );
    }
}
