# simd-json5

A fast, SIMD-accelerated JSON5 parser and serializer for Rust, with optional serde support.

## Benchmarks

simd-json5 is the fastest JSON5 implementation on every corpus: in criterion runs it parses 1.4-3.0x faster than `json5` and 8-43x faster than `serde-json5`. Against JSON implementations it parses 1.2-1.5x faster than `serde_json` and serializes on par with it (ahead on `twitter.json`); the SIMD specialists `simd-json` and `sonic-rs` stay ahead on strict-JSON input (up to ~4.9x).

Parse throughput, decimal MB/s, higher is better:

| Corpus            | simd-json5 | json5 | serde-json5 | serde_json | simd-json | sonic-rs |
| ----------------- | ---------: | ----: | ----------: | ---------: | --------: | -------: |
| twitter.json      |    **683** |   228 |          16 |        441 |       601 |     2099 |
| citm_catalog.json |    **810** |   325 |          32 |        578 |      1330 |     2346 |
| canada.json       |    **306** |   213 |          38 |        256 |       621 |     1485 |
| json5.json5       |    **276** |   186 |          30 |          - |         - |        - |

Serialization throughput, decimal MB/s, higher is better:

| Corpus            | simd-json5 | json5 | serde-json5 | serde_json |
| ----------------- | ---------: | ----: | ----------: | ---------: |
| twitter.json      |   **3009** |   234 |         231 |       2864 |
| citm_catalog.json |       4541 |   423 |         773 |   **5180** |
| canada.json       |       1057 |   112 |         318 |   **1098** |
| json5.json5       |   **1821** |   269 |         375 |          - |

Reproduce with `uv run benches/report.py --run`; `cargo bench --bench parse --bench serialize` runs the full criterion suites.

## Usage

DOM parse and serialize:

```rust
// Unquoted keys, comments, trailing commas, hex, Infinity all parse.
let value = simd_json5::from_str("{ a: 1, // comment\n  b: [2, 3,] }").unwrap();
assert_eq!(value["b"][0], simd_json5::Value::U64(2));

let text = simd_json5::to_string(&value); // {"a":1,"b":[2,3]}
let pretty = simd_json5::to_string_pretty(&value);
```

Point lookups without building a DOM:

```rust
let src = r#"{ artist: { name: 'Helen', tags: ["a", "b",] } }"#;

let name = simd_json5::get(src, &["artist", "name"]).unwrap().unwrap();
assert_eq!(name.as_str().unwrap().as_deref(), Some("Helen"));

let tag = simd_json5::get(src, &["artist", "tags", "1"]).unwrap().unwrap();
assert_eq!(tag.raw(), r#""b""#);
```

Typed via serde with `serde` feature (enabled by default):

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

A 1.83-billion-execution libFuzzer campaign found no crashes (0.2.0); see `fuzz/README.md`.

## Serialization notes

Parsing runs in two stages — a SIMD scan indexes structural characters (comments, unquoted keys and single-quoted strings included), then a semantic pass builds a zero-copy `Value` borrowing from the input.

- Map keys must be strings (or string-like: `char`, unit variant names). Integer, bool or float keys are rejected with an error instead of being rendered as quoted strings the way `serde_json` does.
- Non-finite floats serialize as the JSON5 literals `NaN` / `Infinity` / `-Infinity`; all other output is strict JSON.

## Feature flags

The DOM core does not depend on serde; disable the default feature for a DOM-only build.

- `serde` (default): zero-copy `Deserialize`/`Serialize` for `Value`/`Str`, plus `de::from_str` / `typed::to_string` for any serde type.
