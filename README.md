# simd-json5

SIMD-accelerated JSON5 parser and serializer for Rust. Parsing runs in two
stages — a SIMD scan that indexes structural characters (comments, unquoted
keys and single-quoted strings included), then a semantic pass that builds a
zero-copy DOM borrowing from the input. Serialization emits strict JSON
extended with the JSON5 `NaN` / `Infinity` / `-Infinity` literals, so output
is valid JSON whenever the value contains no non-finite float.

**Status: under active development.** Parsing (SIMD stage 1 on x86_64),
serialization and the serde integration are working; see `DESIGN.md` for the
architecture contract.

## Usage

```rust
// JSON5 input: unquoted keys, comments, trailing commas, hex, Infinity.
let value = simd_json5::from_str("{ a: 1, // comment\n  b: [2, 3,] }").unwrap();
assert_eq!(value["b"][0], simd_json5::Value::I64(2));

// Output is strict JSON plus the JSON5 non-finite float literals.
let text = simd_json5::to_string(&value); // {"a":1,"b":[2,3]}
let pretty = simd_json5::to_string_pretty(&value);

// With the `serde` feature: straight into your types, both directions.
#[derive(serde::Deserialize, serde::Serialize)]
struct Point {
    x: i64,
    y: i64,
}
let p: Point = simd_json5::de::from_str("{ x: 1, y: 2 }").unwrap();
assert_eq!(simd_json5::typed::to_string(&p).unwrap(), r#"{"x":1,"y":2}"#);
```

## Serialization notes

- Map keys must be strings (or string-like: `char`, unit variant names).
  Integer, bool or float keys are rejected with an error instead of being
  rendered as quoted strings the way `serde_json` does.
- Non-finite floats serialize as the JSON5 literals `NaN` / `Infinity` /
  `-Infinity`; all other output is strict JSON.

## Feature flags

| Feature | Default | Description |
| ------- | ------- | ----------- |
| (none)  | yes     | DOM parsing and serialization core. |
| `serde` | no      | serde integration: zero-copy `Deserialize` for `Value`/`Str` and any `Deserialize` type via `de::from_str`, `Serialize` for `Value`/`Str` and any `Serialize` type via `typed::to_string` and friends. |

## MSRV

Rust 1.85 (edition 2024). Development (nightly `rustfmt`/`clippy` settings)
uses the nightly toolchain.

## License

MIT OR Apache-2.0.
