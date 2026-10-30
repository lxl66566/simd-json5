//! DOM accessor semantics: lookups and index fallbacks, cross-variant
//! number conversions, the equality contract, the mutable accessors, and
//! ownership transfer through `into_owned`.

use simd_json5::{Str, Value};

#[test]
fn lookups_are_container_specific() {
    let v = simd_json5::from_str(r"{ a: 1, b: [10, 20] }").unwrap();
    assert_eq!(v.get("a"), Some(&Value::U64(1)));
    assert!(v.get("missing").is_none());
    // get is object-only; get_index is array-only.
    assert!(v.get_index(0).is_none());
    let arr = v.get("b").unwrap();
    assert_eq!(arr.get_index(1), Some(&Value::U64(20)));
    assert!(arr.get_index(2).is_none());
    assert!(arr.get("0").is_none());
    assert!(arr.get("a").is_none());
}

#[test]
fn index_yields_null_outside_containers() {
    let v = simd_json5::from_str(r"{ a: [1, { b: 2 }] }").unwrap();
    assert_eq!(v["a"][0], Value::U64(1));
    assert_eq!(v["a"][1]["b"], Value::U64(2));
    // Missing key, out-of-range index, and indexing scalars or the wrong
    // container kind.
    assert_eq!(v["missing"], Value::Null);
    assert_eq!(v["a"][7], Value::Null);
    assert_eq!(v["a"][0]["x"], Value::Null);
    assert_eq!(v["a"][1][0], Value::Null);
    assert_eq!(v["a"]["a"], Value::Null);
}

#[test]
fn number_conversions_across_variants() {
    // u64 -> i64 and i64 -> u64 only inside the target's range.
    assert_eq!(Value::U64(0).as_i64(), Some(0));
    assert_eq!(Value::U64(i64::MAX as u64).as_i64(), Some(i64::MAX));
    assert_eq!(Value::U64(u64::MAX).as_i64(), None);
    assert_eq!(Value::I64(0).as_u64(), Some(0));
    assert_eq!(Value::I64(-1).as_u64(), None);
    assert_eq!(Value::I64(i64::MIN).as_u64(), None);
    // Floats never narrow to integers; integers widen to f64.
    assert_eq!(Value::F64(1.0).as_i64(), None);
    assert_eq!(Value::F64(1.0).as_u64(), None);
    assert_eq!(Value::I64(-2).as_f64(), Some(-2.0));
    assert_eq!(
        Value::U64(u64::MAX).as_f64(),
        Some(18_446_744_073_709_551_616.0)
    );
    // Non-numbers never convert.
    assert_eq!(Value::Null.as_f64(), None);
    assert_eq!(Value::from("1").as_i64(), None);
}

#[test]
fn integer_equality_across_variants() {
    assert_eq!(Value::I64(5), Value::U64(5));
    // -1 has no u64 encoding; u64::MAX is not it.
    assert_ne!(Value::I64(-1), Value::U64(u64::MAX));
    // Floats never equal integers; NaN equals nothing, itself included.
    assert_ne!(Value::F64(1.0), Value::U64(1));
    let nan = Value::F64(f64::NAN);
    let nan2 = nan.clone();
    assert_ne!(nan, nan2);
}

/// Object equality compares member sets and ignores insertion order
/// (IndexMap semantics, as in serde_json); insertion order still governs
/// iteration and serialization.
#[test]
fn object_equality_ignores_member_order() {
    let v = simd_json5::from_str(r"{ a: 1, b: 2 }").unwrap();
    let same_order = simd_json5::from_str(r"{ a: 1, b: 2 }").unwrap();
    let reordered = simd_json5::from_str(r"{ b: 2, a: 1 }").unwrap();
    assert_eq!(v, same_order);
    assert_eq!(v, reordered);
    let different = simd_json5::from_str(r"{ a: 1, b: 3 }").unwrap();
    assert_ne!(v, different);
}

/// `as_array_mut` lends a mutable slice and `as_object_mut` a mutable map;
/// both reject non-matching variants.
#[test]
fn mut_accessors_are_variant_gated() {
    let mut v = simd_json5::from_str(r"{ a: [1, 2] }").unwrap();
    let root = v.as_object_mut().unwrap();
    if let Some(arr) = root.get_mut("a").and_then(Value::as_array_mut) {
        arr[1] = Value::from(30u64);
    }
    root.insert("c".into(), Value::from(7u64));
    assert_eq!(v["a"][1], Value::U64(30));
    assert_eq!(v["c"], Value::U64(7));

    let mut scalar = Value::from("text");
    assert!(scalar.as_array_mut().is_none());
    assert!(scalar.as_object_mut().is_none());
}

/// `into_owned` deep-converts every borrowed string to an owned one, so
/// the result outlives the input.
#[test]
fn into_owned_detaches_from_input() {
    let text = String::from(r#"{ a: 'plain', b: ["esc\nape"] }"#);
    let borrowed = simd_json5::from_str(&text).unwrap();
    assert!(matches!(
        borrowed.get("a"),
        Some(Value::Str(Str::Static(_)))
    ));
    let owned = borrowed.into_owned();
    drop(text);
    assert!(matches!(owned.get("a"), Some(Value::Str(Str::Dynamic(s))) if s == "plain"));
    assert_eq!(owned["b"][0].as_str(), Some("esc\nape"));
    // Str::into_owned converts borrowed storage the same way.
    assert_eq!(Str::Static("s").into_owned(), "s");
}
