#![no_main]

//! The typed (serde) walk and the DOM walk must agree on every input: equal
//! DOMs on success, equal errors on failure. Mirrors the in-repo
//! `dom_parity_with_stage2` / `error_parity_with_stage2` tests on fuzzer input.
//!
//! Inputs are depth-capped through the shared [`depth`] module: the serde
//! walk would otherwise overflow the runner stack near `MAX_DEPTH` by design.

use libfuzzer_sys::fuzz_target;
use simd_json5_fuzz::depth;

fuzz_target!(|data: &[u8]| {
    if depth::estimate(data) > depth::MAX_PROBE {
        return;
    }
    let dom = simd_json5::from_slice(data);
    let typed = simd_json5::de::from_slice::<simd_json5::Value>(data);
    match (&dom, &typed) {
        (Ok(a), Ok(b)) => {
            assert_eq!(
                format!("{a:?}"),
                format!("{b:?}"),
                "DOM mismatch on {data:?}"
            );
        },
        (Err(a), Err(b)) => assert_eq!(a, b, "error mismatch on {data:?}"),
        _ => panic!("DOM and serde paths disagree on {data:?}"),
    }
});
