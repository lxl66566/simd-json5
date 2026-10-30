# simd-json5

Project policy for agents working in this repository:

- This crate targets the JSON5 spec (https://spec.json5.org) only. Compatibility with serde_json is not a goal.
- Intentional behavioral divergences from serde_json are JavaScript/JSON5 semantics, and each one is recorded in this file so future agents judge changes against the JSON5 spec instead of serde_json.

## Decisions

- i128/u128 visitors parse decimal and hex integer literals exactly, up to 128 bits; larger magnitudes raise `InvalidNumber`. The DOM `Value` and `deserialize_any` still cap integers at 64 bits and fall back to f64, because `Value` has no 128-bit variant.
- f32 targets (values and map keys) round once, decimal to f32; only hex literals wider than 64 bits narrow from their exact f64 value.
- Type-error offsets are the read cursor where the error surfaced, not the start of the offending value.
- Serializer map keys must be string-like: `str`, `char`, unit-variant names, and `Some(inner)` of a string-like value. `None` and other non-strings are rejected.
- `SerializeMap` protocol violations (key without value, and similar) raise an error instead of emitting malformed JSON5.
- `Value::Object` equality compares member sets and ignores insertion order; insertion order governs iteration and serialization only.
- `parser::parse` and `de::from_str` draw stage-1 buffers from thread-local pools (the COUNTS_POOL policy); new stage-1 entry points should use `parser::Stage1Buffers::take` instead of allocating.

## Benchmarking

- Criterion wall deltas are dominated by per-build code-layout rolls: ±5-15% per cell, stable per source state, affecting even code a change never touched. Instruction counts are the deterministic truth. Attribute a regression with callgrind Ir per call on the criterion binaries (`valgrind --tool=callgrind <bench-bin> --bench --profile-time 2 '<cell>/simd-json5'`, inclusive Ir of the entry fn over its call count) before believing a wall delta; two independent `ab.py` runs for merge verdicts; discount single-run >4% outliers.
- The release profile keeps `lto = "thin"`: fat LTO measured equal instruction counts with no wall gain and slower builds; no-LTO measured no gain either.
- A stage-1 token-end table was measured and dropped: it trades string-heavy corpora for number-heavy ones (citm/twitter +3-8% Ir versus canada -5.5% Ir). A fused single-push (start, end) index is the design that could pay; see `perf/parser-core-orig`.
