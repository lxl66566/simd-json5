# Fuzzing

Run a target: `cargo fuzz run <target> -- -dict=fuzz/dict/json5.dict -timeout=5 <max_total_time=N>`.

Targets:

- `parse`: `from_slice` must never panic; compact and pretty serialization of a parsed DOM must reparse to an equal DOM (NaN-tolerant), and `Display` must render the same bytes as `to_string`.
- `de_dom_parity`: the typed (serde) walk and the DOM walk must agree on every input — equal DOMs on success, equal errors on failure. This target found the stage-2 string-arm error-ordering divergence fixed alongside it.
- `typed_shapes`: fixed Rust shapes (`struct` with skipped members, `Vec<Option<u8>>`, `HashMap<String, i64>`, a tuple, an enum) driving the typed surface that `deserialize_any` never reaches: seq/map access, O(1) `Option` null probing, enum forms, and the validate-and-skip machinery. Errors are expected and ignored; only panics and hangs are bugs.

The serde-walking targets (`de_dom_parity`, `typed_shapes`) pre-scan input depth and skip documents deeper than 200 containers: the serde walk recurses through visitor frames (~32 KiB per level) and overflows the fuzz runner's 8 MiB stack by design near `MAX_DEPTH`.

Windows note: with `-Zsanitizer=address` the fuzz binary loads `clang_rt.asan_dynamic-x86_64.dll` from the MSVC installation at runtime; if it is not on `PATH` the process dies with `STATUS_DLL_NOT_FOUND`. Prepend `C:\Program Files\Microsoft Visual Studio\2022\<edition>\VC\Tools\MSVC\<version>\bin\Hostx64\x64` to `PATH` before `cargo fuzz run`.

Corpus directories hold only hand-written seeds; fuzzer-grown inputs and artifacts are gitignored.
