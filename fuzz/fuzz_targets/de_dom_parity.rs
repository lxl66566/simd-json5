#![no_main]

//! The typed (serde) walk and the DOM walk must agree on every input: equal
//! DOMs on success, equal errors on failure. Mirrors the in-repo
//! `dom_parity_with_stage2` / `error_parity_with_stage2` tests on fuzzer input.
//!
//! Inputs are depth-capped: the serde walk recurses through visitor frames
//! (about 32 KiB per level per the crate docs) and the fuzz runner's main
//! thread stack is 8 MiB, so documents approaching `MAX_DEPTH` would overflow
//! the stack by design. The crude byte pre-scan only ever over-counts depth
//! (strings can contribute), which merely skips a few inputs.

use libfuzzer_sys::fuzz_target;

/// Skip documents whose container nesting could plausibly exceed this.
const MAX_DEPTH_PROBE: usize = 200;

fuzz_target!(|data: &[u8]| {
    if depth_estimate(data) > MAX_DEPTH_PROBE {
        return;
    }
    let dom = simd_json5::from_slice(data);
    let typed = simd_json5::de::from_slice::<simd_json5::Value>(data);
    match (&dom, &typed) {
        (Ok(a), Ok(b)) => {
            assert_eq!(format!("{a:?}"), format!("{b:?}"), "DOM mismatch on {data:?}");
        },
        (Err(a), Err(b)) => assert_eq!(a, b, "error mismatch on {data:?}"),
        _ => panic!("DOM and serde paths disagree on {data:?}"),
    }
});

/// Upper bound on container nesting (strings inflate it; that is fine).
fn depth_estimate(data: &[u8]) -> usize {
    let mut depth = 0usize;
    let mut max = 0usize;
    for &b in data {
        match b {
            b'[' | b'{' => {
                depth += 1;
                max = max.max(depth);
            },
            b']' | b'}' => depth = depth.saturating_sub(1),
            _ => {},
        }
    }
    max
}
