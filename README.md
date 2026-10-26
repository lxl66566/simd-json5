# simd-json5

Fastest JSON5 parser and serializer for Rust, with optional serde support.

## Benchmarks

DOM parse and serialization throughput against the other Rust JSON5 implementations (`json5`, `serde-json5`) and the JSON implementations `serde_json`, `simd-json`, `sonic-rs`.

simd-json5 is the fastest JSON5 implementation on every corpus: in criterion runs it parses 2.6-85x faster than `json5` and 6.5-39x faster than `serde-json5` (`canada.json` 7,000x faster than `json5` crate). Against JSON implementations it parses 1.0-1.5x faster than `serde_json` and serializes on par; the SIMD specialists `simd-json` and `sonic-rs` stay ahead on strict-JSON input (up to ~6x).

Parse throughput, decimal MB/s, higher is better:

| Corpus            | simd-json5 |  json5 | serde-json5 | serde_json | simd-json | sonic-rs |
| ----------------- | ---------: | -----: | ----------: | ---------: | --------: | -------: |
| twitter.json      |    **642** |    196 |          16 |        437 |       625 |     2105 |
| citm_catalog.json |    **839** |    326 |          33 |        585 |       699 |     2354 |
| canada.json       |    **253** | 0.03\* |          39 |        243 |       313 |     1493 |
| json5.json5       |    **277** |    3.3 |          30 |          - |         - |        - |

Serialization throughput, decimal MB/s, higher is better:

| Corpus            | simd-json5 | json5 | serde-json5 | serde_json |
| ----------------- | ---------: | ----: | ----------: | ---------: |
| twitter.json      |       2702 |   234 |         229 |   **2835** |
| citm_catalog.json |       4300 |   432 |         760 |   **5010** |
| canada.json       |   **1094** |   111 |         319 |       1087 |
| json5.json5       |   **1721** |   278 |         368 |          - |

(`*`: `json5` behaves quadratically on long float arrays, making repeated measurement of `canada.json` impractical.)

Reproduce with `cargo bench --bench parse --bench serialize`.

## Usage

DOM parse and serialize:

```rust
// Unquoted keys, comments, trailing commas, hex, Infinity all parse.
let value = simd_json5::from_str("{ a: 1, // comment\n  b: [2, 3,] }").unwrap();
assert_eq!(value["b"][0], simd_json5::Value::I64(2));

let text = simd_json5::to_string(&value); // {"a":1,"b":[2,3]}
let pretty = simd_json5::to_string_pretty(&value);
```

Typed via serde, enabled by the default `serde` feature:

```rust
#[derive(serde::Deserialize, serde::Serialize)]
struct Point {
    x: i64,
    y: i64,
}

let p: Point = simd_json5::de::from_str("{ x: 1, y: 2 }").unwrap();
assert_eq!(simd_json5::typed::to_string(&p).unwrap(), r#"{"x":1,"y":2}"#);
```

## Safety

Production code uses `unsafe` sparingly: 11 blocks, each with a safety argument and its invariant.

## Serialization notes

Parsing runs in two stages — a SIMD scan indexes structural characters (comments, unquoted keys and single-quoted strings included), then a semantic pass builds a zero-copy `Value` borrowing from the input.

- Map keys must be strings (or string-like: `char`, unit variant names). Integer, bool or float keys are rejected with an error instead of being rendered as quoted strings the way `serde_json` does.
- Non-finite floats serialize as the JSON5 literals `NaN` / `Infinity` / `-Infinity`; all other output is strict JSON.

## Feature flags

The DOM core (parsing and serialization) does not depend on serde; disable the default feature for a DOM-only build.

- `serde` (default): zero-copy `Deserialize`/`Serialize` for `Value`/`Str`, plus `de::from_str` / `typed::to_string` for any serde type.
