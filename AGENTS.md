# simd-json5

Project policy for agents working in this repository:

- This crate targets the JSON5 spec (https://spec.json5.org) only. Compatibility with serde_json is not a goal.
- Intentional behavioral divergences from serde_json are JavaScript/JSON5 semantics, and each one is recorded in this file so future agents judge changes against the JSON5 spec instead of serde_json.

## Decisions

- i128/u128 visitors parse decimal and hex integer literals exactly, up to 128 bits; larger magnitudes raise `InvalidNumber`. The DOM `Value` and `deserialize_any` still cap integers at 64 bits and fall back to f64, because `Value` has no 128-bit variant.
- f32 targets (values and map keys) round once, decimal to f32; only hex literals wider than 64 bits narrow from their exact f64 value.
- Type-error offsets are the read cursor where the error surfaced, not the start of the offending value.
