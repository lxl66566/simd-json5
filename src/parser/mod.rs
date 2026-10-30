//! Two-stage parser: SIMD structural indexing followed by semantic parsing.

pub(crate) mod number;
pub(crate) mod simd;
pub(crate) mod stage1;
pub(crate) mod stage2;

use std::cell::RefCell;

use crate::{
    Result,
    parser::stage1::{CLASSES, Span, class},
    value::Value,
};

/// Cap on the stage-1 index pre-allocation, in elements (256 MiB of `u32`).
/// The `len / 8` estimate only has to spare the early reallocation chain on
/// ordinary documents; inputs dense enough to outgrow it (or sparse enough
/// to need a fraction) are served by normal vec growth, while an uncapped
/// estimate would reserve 2 GiB for a 4 GiB input.
const INITIAL_INDEX_MAX_ELEMS: usize = 64 << 20;

/// Initial capacity for a stage-1 index over an input of `len` bytes: one
/// entry per ~8 input bytes covers typical documents up front (twitter
/// needs ~1/11, canada ~1/6), bounded by [`INITIAL_INDEX_MAX_ELEMS`].
#[must_use]
pub(crate) fn initial_index_capacity(len: usize) -> usize {
    (len / 8).min(INITIAL_INDEX_MAX_ELEMS)
}

/// Build a DOM [`Value`] from a JSON5 document.
///
/// # Contract
///
/// `bytes` must be valid UTF-8. All callers guarantee it: `from_str` passes a
/// `&str` and `from_slice` validates before delegating. Stage 2 relies on the
/// contract — string and identifier slices are converted without re-validating
/// (`str_slice`), and `Unescape` copies raw prefixes exactly — so feeding
/// invalid UTF-8 here is undefined behavior for string content, not an
/// `Error::InvalidUtf8`. Debug builds assert the contract at every slice.
pub(crate) fn parse(bytes: &[u8]) -> Result<Value<'_>> {
    let mut bufs = Stage1Buffers::take(bytes.len());
    simd::build_index(bytes, &mut bufs.index, &mut bufs.spans)?;
    stage2::build(bytes, &bufs.index, &bufs.spans)
}

thread_local! {
    /// Reusable stage-1 scratch: the index table.
    /// The same allocator-cycling rationale as [`COUNTS_POOL`] applies, at
    /// several times the size — a twitter-scale parse cycles a
    /// multi-hundred-KiB index block through the allocator every parse.
    static INDEX_POOL: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
    /// Reusable clean-string span scratch; see [`INDEX_POOL`].
    static SPANS_POOL: RefCell<Vec<Span>> = const { RefCell::new(Vec::new()) };
}

/// Stage-1 output buffers borrowed from the thread-local pools, returned on
/// drop. A parse nested while a pool is borrowed simply takes fresh buffers.
pub(crate) struct Stage1Buffers {
    pub(crate) index: Vec<u32>,
    /// Clean-string spans; capacity comes only from reuse, never from a
    /// per-parse estimate (a string contributes at most one span but at
    /// least one index entry, so the index estimate bounds the waste).
    pub(crate) spans: Vec<Span>,
}

impl Stage1Buffers {
    /// Cleared buffers for an input of `len` bytes, reserving the standard
    /// initial index capacity.
    pub(crate) fn take(len: usize) -> Self {
        let hint = initial_index_capacity(len);
        let mut index = INDEX_POOL.with(|p| core::mem::take(&mut *p.borrow_mut()));
        index.clear();
        index.reserve(hint);
        let mut spans = SPANS_POOL.with(|p| core::mem::take(&mut *p.borrow_mut()));
        spans.clear();
        Self { index, spans }
    }
}

impl Drop for Stage1Buffers {
    fn drop(&mut self) {
        // Keep only the largest recent buffers, and only up to the cap.
        // Spans hold 8-byte elements, so their element cap halves for the
        // same byte cap.
        if self.index.capacity() <= POOL_MAX_ELEMS {
            INDEX_POOL.with(|p| {
                let mut p = p.borrow_mut();
                if p.capacity() < self.index.capacity() {
                    *p = core::mem::take(&mut self.index);
                }
            });
        }
        if self.spans.capacity() <= POOL_MAX_ELEMS / 2 {
            SPANS_POOL.with(|p| {
                let mut p = p.borrow_mut();
                if p.capacity() < self.spans.capacity() {
                    *p = core::mem::take(&mut self.spans);
                }
            });
        }
    }
}

/// Documents with fewer index entries return no table: no container in them
/// can hold enough children to amortize a pre-allocation, so the pass and its
/// allocation would be pure overhead.
#[cfg_attr(not(feature = "serde"), allow(dead_code))]
const MIN_COUNT_ENTRIES: usize = 8;

/// Largest scratch buffer kept between parses, in elements (8 MiB of u32):
/// one huge document must not pin memory for every later parse on the thread.
/// Shared by the counts, index and span pools; span elements are 8 bytes, so
/// their element cap halves to keep the same byte bound.
#[cfg_attr(not(feature = "serde"), allow(dead_code))]
const POOL_MAX_ELEMS: usize = 2 << 20;

thread_local! {
    /// Reusable counts-table scratch. Allocating a fresh index-length block
    /// per parse measurably slowed documents parsed after a different-sized
    /// document in the same process (DOM citm right after twitter lost 20%+
    /// to the allocator cycling the freed half-megabyte block every parse),
    /// so steady-state parsing borrows one buffer instead of allocating.
    static COUNTS_POOL: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
    /// Reusable open-container stack for [`container_counts`]; see
    /// [`COUNTS_POOL`] for the pooling rationale.
    #[cfg_attr(not(feature = "serde"), allow(dead_code))]
    static OPENS_POOL: RefCell<Vec<(usize, u32)>> = const { RefCell::new(Vec::new()) };
}

/// Largest open-container stack kept between parses, in entries (64 KiB):
/// valid documents nest no deeper than `MAX_DEPTH`, so only rejected garbage
/// could outgrow it, and it must not pin memory.
#[cfg_attr(not(feature = "serde"), allow(dead_code))]
const OPENS_POOL_MAX_ELEMS: usize = 4096;

/// The element-count hint table produced by [`container_counts`]. Recycles
/// its buffer into the thread-local pool when dropped; a nested parse while
/// the pool is borrowed simply takes a fresh buffer.
#[cfg_attr(not(feature = "serde"), allow(dead_code))]
pub(crate) struct Counts(Vec<u32>);

#[cfg_attr(not(feature = "serde"), allow(dead_code))]
impl Counts {
    pub(crate) fn as_slice(&self) -> &[u32] {
        &self.0
    }
}

impl Drop for Counts {
    fn drop(&mut self) {
        // Keep only the largest recent buffer, and only up to the cap.
        if self.0.capacity() <= POOL_MAX_ELEMS {
            COUNTS_POOL.with(|p| {
                let mut p = p.borrow_mut();
                if p.capacity() < self.0.capacity() {
                    *p = core::mem::take(&mut self.0);
                }
            });
        }
    }
}

/// Element-count hint for every container opening in `index`, by index
/// position: the hint at `at` estimates the direct children of the container
/// whose `[` or `{` is entry `at`; reads at any other position are
/// meaningless. Consumers look up instead of re-scanning, so the whole
/// document pays exactly one linear pass.
///
/// The pass keeps a stack of open containers: every `,` between an opening
/// and its matching `]` / `}` at that nesting level counts toward the
/// innermost open container, and the matching close records `commas + 1`.
/// Strings and bare tokens contribute only their first index entry, so
/// nothing inside them is ever seen here.
///
/// A hint producer only — never fails, never panics, and carries no
/// semantics: on malformed input (stray commas or closers, truncated
/// containers, trailing commas) the values are merely harmless capacity
/// upper bounds, and stage 2 rejects the document regardless. An empty
/// container records 1 and a trailing comma over-counts by one; capacity is
/// invisible, so both are safe. Openings that never close keep 0, a
/// zero-valued hint that reserves no capacity.
///
/// Scratch cost: 4 bytes per index entry plus the pass itself, which only
/// classifies the byte at each entry — far cheaper than the parse.
#[cfg_attr(not(feature = "serde"), allow(dead_code))]
pub(crate) fn container_counts(bytes: &[u8], index: &[u32]) -> Counts {
    if index.len() < MIN_COUNT_ENTRIES {
        return Counts(Vec::new());
    }
    let mut counts = COUNTS_POOL.with(|p| core::mem::take(&mut *p.borrow_mut()));
    counts.clear();
    counts.resize(index.len(), 0);
    // (index position of the `[` / `{`, commas seen at its nesting level).
    // Borrowed from a pool like the counts table: hinted parses run per
    // parse, and valid documents never stack deeper than MAX_DEPTH.
    let mut opens = OPENS_POOL.with(|p| core::mem::take(&mut *p.borrow_mut()));
    opens.clear();
    opens.reserve(16);
    for (at, &entry) in index.iter().enumerate() {
        // Stage-1 entries are always valid input positions (< bytes.len());
        // the widening cast is lossless (usize is at least 32 bits).
        let off = entry as usize;
        // SAFETY: `off` comes from the stage 1 index built over this same
        // `bytes`: every entry is a valid input position (< bytes.len()),
        // the same contract `stage2::build` relies on for its own unchecked
        // read at each entry.
        debug_assert!(off < bytes.len());
        let b = unsafe { *bytes.get_unchecked(off) };
        // Strings, tokens and `:` never separate container children; one
        // table test skips the common case before the structural match.
        if CLASSES[b as usize] & class::STRUCTURAL == 0 {
            continue;
        }
        match b {
            b'[' | b'{' => opens.push((at, 0)),
            b']' | b'}' => {
                // A close with nothing open is malformed; stage 2 rejects
                // it, the hint table just ignores it.
                if let Some((open, commas)) = opens.pop() {
                    counts[open] = commas.saturating_add(1);
                }
            },
            // A comma outside any container is malformed; ignored too.
            b',' => {
                if let Some((_, commas)) = opens.last_mut() {
                    *commas = commas.saturating_add(1);
                }
            },
            _ => {},
        }
    }
    // Return the open-container stack for the next hinted parse on this
    // thread, keeping the larger buffer up to the depth cap (documents past
    // MAX_DEPTH are rejected by the parsers anyway; deeper garbage must not
    // pin an oversized stack).
    if opens.capacity() <= OPENS_POOL_MAX_ELEMS {
        OPENS_POOL.with(|p| {
            let mut p = p.borrow_mut();
            if p.capacity() < opens.capacity() {
                *p = core::mem::take(&mut opens);
            }
        });
    }
    Counts(counts)
}

#[cfg(test)]
mod tests {
    use super::{container_counts, parse};
    use crate::{Value, parser::stage1::build_index};

    fn counts(src: &str) -> super::Counts {
        let mut index = Vec::new();
        build_index(src.as_bytes(), &mut index, &mut Vec::new()).unwrap();
        container_counts(src.as_bytes(), &index)
    }

    /// Consecutive parses borrow and return the pooled stage-1 buffers;
    /// results must be unaffected by stale capacity or contents left by
    /// earlier parses (of any size, success or error).
    #[test]
    fn pooled_buffers_reuse_across_parses() {
        let big = format!(
            "[{}]",
            (0..2000)
                .map(|i| format!("{{n{i}: {i}, s: \"v{i}\"}}"))
                .collect::<Vec<_>>()
                .join(",")
        );
        let v = parse(big.as_bytes()).unwrap();
        assert_eq!(v.as_array().unwrap().len(), 2000);
        assert_eq!(
            parse(b"[1, 2]").unwrap(),
            Value::Array(vec![Value::U64(1), Value::U64(2)])
        );
        // Error paths return the buffers too (Drop runs on `?`).
        assert!(parse(b"{a:").is_err());
        let v = parse(b"{a: 1, b: 'x'}").unwrap();
        let m = v.as_object().unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!(m.get_index(1).unwrap().1.as_str(), Some("x"));
    }

    /// `{a: [1, 2,], b: [], c: {}}`: entries 0 `{`, 3 `[`, 12 `[`, 17 `{`
    /// (19 closes the outer object). The outer object counts its three
    /// member separators exactly; the trailing comma inside the first array
    /// over-counts it by one (a safe upper bound); empty containers record 1.
    #[test]
    fn balanced_documents_hint_every_opening() {
        let table = counts("{a: [1, 2,], b: [], c: {}}");
        let c = table.as_slice();
        assert_eq!(c.len(), 20);
        assert_eq!(c[0], 3);
        assert_eq!(c[3], 3);
        assert_eq!(c[12], 1);
        assert_eq!(c[17], 1);
    }

    /// Malformed documents must not panic and must not produce a hint that
    /// would pre-allocate anything (0 everywhere, or a bounded over-count).
    /// Two consecutive parses also exercise the pooled buffer's reuse.
    #[test]
    fn malformed_input_produces_harmless_hints() {
        // Truncated array: the opening never closes, so its slot stays 0.
        let table = counts("[1, 2, 3, 4, 5, 6, 7, 8");
        let c = table.as_slice();
        assert_eq!(c.len(), 16);
        assert_eq!(c[0], 0);
        // Stray closers and commas with no container open are ignored.
        let c = counts("] , ] , ] , ] , ] ]");
        assert!(c.as_slice().iter().all(|&x| x == 0));
        // An adversarial comma run only inflates the count linearly.
        let c = counts("[, , , , , , ]");
        assert_eq!(c.as_slice()[0], 7);
        // A second scan over a pooled buffer must not see stale entries.
        let c = counts("{a: 1, b: 2, c: 3}");
        assert_eq!(c.as_slice()[0], 3);
        // Tiny documents skip the table entirely (no pass, no allocation).
        let c = counts("[1, 2]");
        assert_eq!(c.as_slice(), [] as [u32; 0]);
    }
}
