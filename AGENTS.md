# simd-json5

Project policy for agents working in this repository:

- This crate targets the JSON5 spec (https://spec.json5.org) only. Compatibility with serde_json is not a goal.
- Intentional behavioral divergences from serde_json are JavaScript/JSON5 semantics, and each one is recorded in this file so future agents judge changes against the JSON5 spec instead of serde_json.

## Decisions

- Type-error offsets are the read cursor where the error surfaced, not the start of the offending value.
