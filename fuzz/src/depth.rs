//! Input depth pre-scan for the serde-walking targets.
//!
//! The serde walk recurses through visitor frames (about 32 KiB per level per
//! the crate docs) and the fuzz runner's main thread has an 8 MiB stack, so
//! documents approaching the parser's [`simd_json5::MAX_DEPTH`] (1024) would
//! overflow the stack by design. Targets skip inputs whose estimated depth
//! exceeds [`MAX_PROBE`].

/// Skip documents whose container nesting could plausibly exceed this.
pub const MAX_PROBE: usize = 200;

/// Upper bound on container nesting (strings inflate it; that is fine).
pub fn estimate(data: &[u8]) -> usize {
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
