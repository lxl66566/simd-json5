# simd-json5 architecture contract

This document pins the shared contracts between the parallel development branches. Treat it as read-only: if a contract must change, report the requested change in your final report instead of editing this file.

## Goals

- Correct JSON5 (spec at json5.org) parsing and serialization, zero-copy DOM where possible.
- Parse/serialize performance competitive with simd-json on JSON input, and far ahead of json5 / serde-json5 on JSON5 input.
- Core (DOM parse + serialize) has zero knowledge of serde. Serde integration lives behind the `serde` cargo feature.

## Crate layout and file ownership

- `src/lib.rs` public API and re-exports. Changed only via the coordination noted below.
- `src/error.rs` error type. Owner: core-parser.
- `src/value.rs` DOM (`Value`, `Str`, `Map`, `MAX_DEPTH`). Scalar/accessor part exists already. core-parser may extend it for parse needs; serializer adds `fmt::Display` and any serialize-only helpers in clearly separated impl blocks.
- `src/parser/mod.rs` orchestration. Owner: core-parser. simd-stage1 later rewires `stage1::build_index` dispatch only.
- `src/parser/stage1.rs` structural indexer: scalar reference implementation and dispatch. Owner: core-parser for the scalar path, simd-stage1 for the SIMD backends and dispatch.
- `src/parser/stage2.rs` semantic parser. Owner: core-parser.
- `src/parser/number.rs` number parsing (ints by hand, floats via a fast-float family crate). Owner: core-parser.
- `src/ser.rs` serializer. Owner: serializer.
- `src/de.rs` (feature `serde`) deserializer. Owner: assigned later.
- `tests/` integration tests: each branch adds its own files there.
- `benches/` criterion suite. Owner: serializer.

## Public API pin

- `from_str(&str) -> Result<Value<'_>>`, `from_slice(&[u8]) -> Result<Value<'_>>` (UTF-8 validating).
- `to_string(&Value) -> String`, `to_string_pretty(&Value) -> String`, `to_vec(&Value) -> Vec<u8>`.
- `Value<'a>`: `Null Bool I64 U64 F64 Str Array Object`. Object is `indexmap::IndexMap<Str<'a>, Value<'a>>` (insertion order, JavaScript semantics: duplicate keys keep first position, last value).
- `Str<'a>`: `Static(&'a str)` when the source had no escapes, `Dynamic(String)` when unescaping was required. Content-based `PartialEq`/`Hash`, `Deref<Target = str>`.
- `MAX_DEPTH = 1024` nesting cap for containers, enforced in stage 2 and in the serde deserializer.
- With feature `serde`:
  - `simd_json5::de::from_str::<T: Deserialize<'de>>(&'de str) -> Result<T>` and `de::from_slice`: zero-copy deserializer walking the stage-1 index directly, mirroring stage-2 semantics (identical grammar, errors, `MAX_DEPTH`); borrowed strings via `visit_borrowed_str` when no unescaping was needed; serde_json-equivalent `Deserializer`/`Visitor` coverage including enum forms (string unit variant and map-wrapped variants) and `Error::Custom` for `de::Error::custom`.
  - `simd_json5::typed::to_string::<T: Serialize>(&T) -> String`, `typed::to_vec`, `typed::to_string_pretty`: a serde `Serializer` over the existing writer; non-finite floats use the JSON5 literals, identical escape rules. `typed::from_str`/`typed::from_slice` delegate to `de`.
  - `impl Serialize for Value/Str` lives in `src/ser.rs`, `impl Deserialize` in `src/de.rs`; the DOM root entry points stay unchanged.

## Stage 1 contract

See the module doc in `src/parser/stage1.rs`; it is the authoritative copy. Summary: emit ascending offsets for structurals (`{}[],:`), string opening quotes, and first bytes of bare tokens; token bytes are `A-Za-z0-9$_.+-` plus `\` (identifier `\uXXXX` escapes make backslash a token byte; its validation is stage 2's job) plus non-ASCII bytes that are not Unicode whitespace; classify Unicode whitespace in a scalar slow path; reject illegal bytes, stray `/`, structurally malformed UTF-8, unterminated strings/comments with offsets, and inputs above 4 GiB. SIMD backends must produce byte-identical index output and error variants as the scalar implementation, verified by differential tests.

## Stage 2 semantics (JSON5 grammar pin)

- Values: `null true false`, numbers, strings, arrays, objects. Top-level scalars allowed.
- Objects: keys are identifiers or strings; `:` required; members comma-separated with one optional trailing comma before `}`. Identifiers: `[A-Za-z_$]` plus non-ASCII (accepted leniently as ECMAScript ID_Start/ID_Continue approximations) plus `\uXXXX` escapes; unquoted number keys are invalid.
- Arrays: elements comma-separated, one optional trailing comma before `]`.
- Numbers: `[+-]? ( Infinity | NaN | HexInteger | Decimal )`. HexInteger: `0x [0-9a-fA-F]+`. Decimal per ECMAScript 5.1: `Digits (. Digits?)? ((e|E) [+-]? Digits)?` or `. Digits (...)` — so `5.`, `.5`, `1.e3`, `+1`, `007` are valid; `.e3`, `1e`, `0x` are not. Integers fitting u64 (positive) or i64 (negative) stay integer variants; everything else becomes F64. Hex overflow beyond i64/u64 becomes F64 by exact widening.
- Strings: `"` or `'` delimited. Escapes: `\" \' \\ \b \f \n \r \t \v \0` (`\0` only when not followed by a decimal digit), `\xHH` (exactly 2 hex), `\uHHHH` (exactly 4 hex; surrogate pairs combine; lone surrogates are `InvalidUnicodeEscape`), line continuation `\` + (LF | CR | CRLF | LS | PS) produces nothing, any other `\c` produces `c` except `\1`..`\9`, which are invalid (JSON5's `NonEscapeCharacter` excludes decimal digits). Raw LF/CR are illegal in strings; raw U+2028/U+2029 are legal. Strings containing no backslash borrow; others unescape into `Str::Dynamic`.
- Comments: `// ...` to end of line, `/* ... */` (may nest? No — JSON5 comments do not nest; first `*/` ends the comment).
- Duplicate keys: last value wins, first position kept (IndexMap `insert` semantics).
- Errors use the variant closest to the failure (see `src/error.rs`); `Syntax` with a short message covers grammar states not otherwise named.

## Serializer contract

- Output is strict JSON except: `NaN`, `Infinity`, `-Infinity` literals for non-finite f64.
- Strings double-quoted, UTF-8 passthrough for bytes >= 0x20 except `"` and `\`; short escapes `\b \t \n \f \r`; other C0 controls as `\u00XX`. U+2028/U+2029 pass through raw (valid in both JSON and JSON5 strings).
- Integers via `itoa`, floats via `zmij` (zmij writes `NaN`/`inf` — the writer must special-case them to the literals above).
- Pretty printing: two-space indent, `": "` after keys, newline between items, exactly like `serde_json::to_string_pretty` on JSON-representable values.
- No streaming public API yet: `to_string`/`to_string_pretty`/`to_vec` only.

## SIMD plan

- Backends live in `src/parser/simd.rs`; the parser routes through `simd::build_index` (see `src/parser/mod.rs`). One generic kernel is monomorphized per ISA through `fearless_simd`'s `dispatch!`, which also enables each ISA's target features: SSE2, SSE4.2, AVX2 and Ice-Lake AVX-512 on x86_64, NEON on aarch64, SIMD128 on wasm32. Targets with no supported vector ISA (`Fallback` level) run the canonical scalar `stage1::build_index`, which beats emulated vectors.
- Algorithm: 64-byte unaligned blocks (four 128-bit lanes); a vector whitespace mask yields an attention mask (its complement); a scalar walk of attention bits emits token starts, dispatches to vectorized string skipping (simdjson's odd-backslash-run escape parity with cross-block carry) and vectorized comment skipping (`*`/`/` masks with `prev_star` carry), and takes the Unicode-whitespace slow path on high-bit bytes. The trailing partial block is finished by a scalar tail carrying the in-token flag.
- Byte classification: nibble lookup tables through dynamic byte shuffles (the simdjson pshufb scheme) on ISAs with a native shuffle; compare chains on SSE2 and the fallback, whose dynamic shuffles are emulated per lane.
- Correctness over cleverness: the SIMD path may bail to the scalar path at any construct it cannot handle, but must not produce different output.

## Testing strategy

- Compliance: JSON5 spec examples (json5.org README), ported cases from the reference `json5` (Rust) crate test suite and the official json5-tests repository, both accept and reject directions.
- Differential: SIMD stage1 vs scalar stage1 on random inputs (`rand` dev-dependency; generate from a JSON5-ish alphabet including comments, quotes, escapes, high-bit bytes).
- Oracle: on pure-JSON inputs compare our DOM against `serde_json` (dev-dependency) after normalization.
- Round-trip: parse -> serialize -> parse must reach an equal DOM.

## Benchmark plan

Criterion suite in `benches/`. Corpora: `twitter.json`, `citm_catalog.json`, `canada.json` (JSON, from serde_json or simd-json test data) plus one JSON5 corpus with comments, unquoted keys, single quotes, trailing commas, hex and Infinity/NaN literals. Competitors (dev-dependencies): `serde_json`, `simd-json`, `json5`, `serde-json5`. Only the coordinating manager runs measurement benches; branches never run `cargo bench`.

## Dependency pins

Runtime: `indexmap`, `thiserror`, `itoa`, `zmij`, `memchr` (scalar string/comment scans), `simdutf8` (`from_slice` UTF-8 validation), `fearless_simd` (stage-1 ISA dispatch), `fast-float2 0.2` (float parsing). Optional: `serde`. Dev: `serde_json`, `rand`, `criterion`, the competitor parsers above.

## Process rules

- Work in your assigned git worktree only; commit atomically on your branch (tests and docs travel with the code they cover); never `git add -A`; English comments/commits/docs; no emoji.
- Run every cargo command with `CARGO_TARGET_DIR=C:/programs/Rust/simd-json5-target` (git bash).
- `cargo fmt` before every commit; `cargo clippy --all-targets --all-features -- -D warnings` and `cargo nextest run --all-features` plus `cargo test --all-features --doc` must pass before your final report.
- Temporary files go to `Z:/Temp` and must be cleaned up.
