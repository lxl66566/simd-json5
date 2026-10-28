# simd-json5

Fastest JSON5 parser and serializer for Rust, with optional serde support.

## Benchmarks

DOM parse and serialization throughput against the other Rust JSON5 implementations (`json5`, `serde-json5`) and the JSON implementations `serde_json`, `simd-json`, `sonic-rs`.

simd-json5 is the fastest JSON5 implementation on every corpus: in criterion runs it parses 2.4-83x faster than `json5` and 8.4-40x faster than `serde-json5` (`canada.json` 9,300x faster than `json5` crate). Against JSON implementations it parses up to 1.5x faster than `serde_json` (on par on the float-heavy `canada.json`) and serializes within 25% of it; the SIMD specialists `simd-json` and `sonic-rs` stay ahead on strict-JSON input (up to ~4.6x).

Parse throughput, decimal MB/s, higher is better:

| Corpus            | simd-json5 |  json5 | serde-json5 | serde_json | simd-json | sonic-rs |
| ----------------- | ---------: | -----: | ----------: | ---------: | --------: | -------: |
| twitter.json      |    **655** |    200 |          16 |        439 |       602 |     2128 |
| citm_catalog.json |    **787** |    323 |          32 |        580 |      1317 |     2389 |
| canada.json       |    **324** | 0.03\* |          39 |        349 |       616 |     1491 |
| json5.json5       |    **272** |    3.3 |          30 |          - |         - |        - |

Serialization throughput, decimal MB/s, higher is better:

| Corpus            | simd-json5 | json5 | serde-json5 | serde_json |
| ----------------- | ---------: | ----: | ----------: | ---------: |
| twitter.json      |       2675 |   228 |         234 |   **2827** |
| citm_catalog.json |       4011 |   431 |         761 |   **5164** |
| canada.json       |       1048 |   114 |         318 |   **1112** |
| json5.json5       |   **1694** |   276 |         371 |          - |

(`*`: `json5` behaves quadratically on long float arrays, making repeated measurement of `canada.json` impractical.)

Reproduce with `uv run benches/report.py --run` (README comparison groups only, ~3 min); `cargo bench --bench parse --bench serialize` runs the full criterion suites.

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

A 1.83-billion-execution libFuzzer campaign found no crashes; see `fuzz/README.md`.

## Serialization notes

Parsing runs in two stages — a SIMD scan indexes structural characters (comments, unquoted keys and single-quoted strings included), then a semantic pass builds a zero-copy `Value` borrowing from the input.

- Map keys must be strings (or string-like: `char`, unit variant names). Integer, bool or float keys are rejected with an error instead of being rendered as quoted strings the way `serde_json` does.
- Non-finite floats serialize as the JSON5 literals `NaN` / `Infinity` / `-Infinity`; all other output is strict JSON.
- Float formatting follows `serde_json` >= 1.0.154 (both use the `zmij` backend): exponents carry an explicit sign, e.g. `1e+300`. Older `serde_json` (ryu backend) prints `1e300` instead.

## Feature flags

The DOM core (parsing and serialization) does not depend on serde; disable the default feature for a DOM-only build.

- `serde` (default): zero-copy `Deserialize`/`Serialize` for `Value`/`Str`, plus `de::from_str` / `typed::to_string` for any serde type.
