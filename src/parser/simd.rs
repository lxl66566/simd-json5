//! SIMD backends for the stage-1 structural indexer.
//!
//! This module implements the stage-1 contract (see the module documentation
//! of [`super::stage1`], the authoritative copy) with vector kernels built on
//! [`fearless_simd`]:
//!
//! - one generic kernel, monomorphized per ISA through `dispatch!`, which also enables each ISA's
//!   target features: SSE2, SSE4.2, AVX2 and Ice-Lake AVX-512 on x86_64, NEON on aarch64, SIMD128
//!   on wasm32;
//! - the scalar `Fallback` level routes to the canonical [`super::stage1::build_index`]: emulated
//!   vectors lose to the scalar scan by a wide margin;
//! - byte classification runs through nibble lookup tables and dynamic byte shuffles (the simdjson
//!   pshufb scheme) where the ISA provides a native shuffle, and through compare chains on SSE2 and
//!   the fallback, whose dynamic shuffles are emulated per lane.
//!
//! Kernel algorithm (one 64-byte block, four 128-bit lanes per iteration):
//!
//! 1. Classify the block, producing a 64-bit whitespace mask; its complement (`attention mask`)
//!    marks every byte that needs a scalar decision.
//! 2. Walk the set bits of the attention mask. Whitespace runs are skipped for free; token starts
//!    are emitted here; `/` dispatches to the vectorized comment skippers and quotes to the
//!    vectorized string skipper (escape parity via simdjson's odd-backslash-run trick, see
//!    [`odd_backslash_ends`]). High-bit bytes take the Unicode slow path (`decode_seq` +
//!    `is_unicode_whitespace`), consuming whole sequences.
//! 3. A skip that crosses the block boundary moves the scan cursor with it; blocks are loaded
//!    unaligned, so arbitrary landings are fine.
//! 4. The final partial block is finished by the scalar tail ([`scan_scalar`]), carrying the
//!    in-token flag across the handoff.
//!
//! All backends produce byte-identical index output, clean-string span
//! tables and error values for the same input, down to the partial contents
//! left behind on error. The differential tests in this module enforce that
//! against the canonical scalar implementation.

// inline(always) on the kernel and every helper is load-bearing: the generic
// body must fold into the feature-enabled closure that `dispatch!` builds
// around each backend, or the vector code would be compiled without the ISA.
#![allow(clippy::inline_always)]

#[cfg(target_arch = "x86_64")]
use fearless_simd::{Avx2, Avx512, u8x32, u8x64};
use fearless_simd::{Level, Simd, dispatch, prelude::*, u8x16};
use memchr::memchr;

use super::stage1::{
    Span, check_input_len, decode_seq, is_unicode_whitespace, scan_scalar, skip_string_tail,
};
use crate::{Error, Result};

/// Dispatch entry for the parser: selects the best kernel for the current
/// CPU.
///
/// - every SIMD level: the generic [`scan_simd`] kernel with the ISA's target features enabled by
///   `dispatch!`;
/// - `Fallback`: the canonical scalar [`super::stage1::build_index`]. Its length check then runs
///   twice (once here, once inside); the duplicate comparison is cheaper than forking the canonical
///   entry.
pub(crate) fn build_index(bytes: &[u8], index: &mut Vec<u32>, spans: &mut Vec<Span>) -> Result<()> {
    check_input_len(bytes)?;
    let level = Level::new();
    if level.is_fallback() {
        return super::stage1::build_index(bytes, index, spans);
    }
    dispatch!(level, simd => scan_simd(simd, bytes, index, spans))
}

/// Mask of the bytes equal to `byte` in one 128-bit lane, in the low 16 bits.
#[inline(always)]
fn eq_mask16<S: Simd>(simd: S, lane: u8x16<S>, byte: u8) -> u64 {
    lane.simd_eq(simd.splat_u8x16(byte)).to_bitmask()
}

/// Mask of the bytes whose unsigned value is `<= bound`, in the low 16 bits.
#[inline(always)]
fn le_mask16<S: Simd>(simd: S, lane: u8x16<S>, bound: u8) -> u64 {
    lane.simd_le(simd.splat_u8x16(bound)).to_bitmask()
}

/// OR of the equality masks for `byte` over the 64 bytes at `i`, loaded at
/// the level's native width: one compare per 64 bytes on AVX-512, two on
/// AVX2, four on 128-bit ISAs. Callers guarantee 64 readable bytes at `i`.
#[inline(always)]
fn eq_mask_native<S: Simd>(simd: S, bytes: &[u8], i: usize, byte: u8) -> u64 {
    let width = <S::u8s as SimdBase<S>>::LEN;
    let needle = <S::u8s as SimdBase<S>>::splat(simd, byte);
    let mut acc = 0_u64;
    let mut k = 0;
    while k < 64 {
        acc |= <S::u8s as SimdBase<S>>::from_slice(simd, &bytes[i + k..i + k + width])
            .simd_eq(needle)
            .to_bitmask()
            << k;
        k += width;
    }
    acc
}

/// Pair of equality masks for `first` and `second` over the 64 bytes at
/// `i`: every native block is loaded once and feeds both compares, half the
/// loads of two [`eq_mask_native`] calls. Callers guarantee 64 readable
/// bytes at `i`.
#[inline(always)]
fn pair_mask_native<S: Simd>(simd: S, bytes: &[u8], i: usize, first: u8, second: u8) -> (u64, u64) {
    let width = <S::u8s as SimdBase<S>>::LEN;
    let needle_first = <S::u8s as SimdBase<S>>::splat(simd, first);
    let needle_second = <S::u8s as SimdBase<S>>::splat(simd, second);
    let mut acc_first = 0_u64;
    let mut acc_second = 0_u64;
    let mut k = 0;
    while k < 64 {
        let v = <S::u8s as SimdBase<S>>::from_slice(simd, &bytes[i + k..i + k + width]);
        acc_first |= v.simd_eq(needle_first).to_bitmask() << k;
        acc_second |= v.simd_eq(needle_second).to_bitmask() << k;
        k += width;
    }
    (acc_first, acc_second)
}

/// Quadruple of equality masks over the 64 bytes at `i`: like
/// [`pair_mask_native`] with four needles — one load per native block feeds
/// all four compares (the string skip wants quote, backslash, LF and CR
/// together). Callers guarantee 64 readable bytes at `i`.
#[inline(always)]
fn quad_mask_native<S: Simd>(
    simd: S,
    bytes: &[u8],
    i: usize,
    first: u8,
    second: u8,
    third: u8,
    fourth: u8,
) -> (u64, u64, u64, u64) {
    let width = <S::u8s as SimdBase<S>>::LEN;
    let needle_first = <S::u8s as SimdBase<S>>::splat(simd, first);
    let needle_second = <S::u8s as SimdBase<S>>::splat(simd, second);
    let needle_third = <S::u8s as SimdBase<S>>::splat(simd, third);
    let needle_fourth = <S::u8s as SimdBase<S>>::splat(simd, fourth);
    let mut acc_first = 0_u64;
    let mut acc_second = 0_u64;
    let mut acc_third = 0_u64;
    let mut acc_fourth = 0_u64;
    let mut k = 0;
    while k < 64 {
        let v = <S::u8s as SimdBase<S>>::from_slice(simd, &bytes[i + k..i + k + width]);
        acc_first |= v.simd_eq(needle_first).to_bitmask() << k;
        acc_second |= v.simd_eq(needle_second).to_bitmask() << k;
        acc_third |= v.simd_eq(needle_third).to_bitmask() << k;
        acc_fourth |= v.simd_eq(needle_fourth).to_bitmask() << k;
        k += width;
    }
    (acc_first, acc_second, acc_third, acc_fourth)
}

/// Load 64 bytes at `i` as four 128-bit lanes, unaligned like the scalar
/// reads they replace. Callers guarantee 64 readable bytes at `i`.
#[inline(always)]
fn load64<S: Simd>(simd: S, bytes: &[u8], i: usize) -> [u8x16<S>; 4] {
    [
        u8x16::from_slice(simd, &bytes[i..i + 16]),
        u8x16::from_slice(simd, &bytes[i + 16..i + 32]),
        u8x16::from_slice(simd, &bytes[i + 32..i + 48]),
        u8x16::from_slice(simd, &bytes[i + 48..i + 64]),
    ]
}

/// Classification strategy for one scan. `Simd::level()` is a real call that
/// cannot fold inside the block loop, so the level is inspected once per
/// scan and the per-block dispatch is a match on this value.
#[derive(Clone, Copy)]
enum Classifier<S: Simd> {
    /// Nibble tables where dynamic shuffles are native, compare chains
    /// otherwise (all levels except the x86 wide lanes).
    Generic { tables: bool, simd: S },
    #[cfg(target_arch = "x86_64")]
    Avx2(Avx2),
    #[cfg(target_arch = "x86_64")]
    Avx512(Avx512),
}

/// Inspect the level once per scan.
fn classifier<S: Simd>(simd: S) -> Classifier<S> {
    #[cfg(target_arch = "x86_64")]
    match simd.level() {
        Level::Avx512(tok) => return Classifier::Avx512(tok),
        Level::Avx2(tok) => return Classifier::Avx2(tok),
        _ => {},
    }
    Classifier::Generic {
        tables: tables_native(simd),
        simd,
    }
}

/// Classify the 64 bytes at `i`: `(whitespace, token, structural)` masks,
/// where whitespace is exactly `0x09..=0x0D` and `0x20`, and the other two
/// are the stage-1 token and structural byte sets (see `CLASSES`). Callers
/// guarantee 64 readable bytes at `i`.
#[inline(always)]
fn classify_with<S: Simd>(c: Classifier<S>, bytes: &[u8], i: usize) -> (u64, u64, u64) {
    match c {
        Classifier::Generic { tables, simd } => {
            let blocks = load64(simd, bytes, i);
            if tables {
                classify_tables(simd, blocks)
            } else {
                classify_compares(simd, blocks)
            }
        },
        #[cfg(target_arch = "x86_64")]
        Classifier::Avx2(tok) => classify_avx2(tok, bytes, i),
        #[cfg(target_arch = "x86_64")]
        Classifier::Avx512(tok) => classify_avx512(tok, bytes, i),
    }
}

/// Whether this level lowers dynamic byte shuffles to a native instruction.
fn tables_native<S: Simd>(simd: S) -> bool {
    if simd.level().is_fallback() {
        return false;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if matches!(simd.level(), Level::Sse2(_)) {
        return false;
    }
    true
}

/// Compare-chain classification: byte classes from unsigned range and
/// equality compares. Used on levels without a native dynamic byte shuffle
/// (SSE2, the fallback), whose emulated shuffles cost more than this whole
/// chain.
#[inline(always)]
fn classify_compares<S: Simd>(simd: S, blocks: [u8x16<S>; 4]) -> (u64, u64, u64) {
    let mut ws = 0_u64;
    let mut tok = 0_u64;
    let mut special = 0_u64;
    for (lane, b) in blocks.into_iter().enumerate() {
        // Token set: [0-9] [A-Z] [a-z] $ _ + - . \ (see CLASSES).
        let dig = le_mask16(simd, b, b'9') & !le_mask16(simd, b, b'/');
        let upper = le_mask16(simd, b, b'Z') & !le_mask16(simd, b, b'@');
        let lower = le_mask16(simd, b, b'z') & !le_mask16(simd, b, b'`');
        let extra = eq_mask16(simd, b, b'$')
            | eq_mask16(simd, b, b'_')
            | eq_mask16(simd, b, b'+')
            | eq_mask16(simd, b, b'-')
            | eq_mask16(simd, b, b'.')
            | eq_mask16(simd, b, b'\\');
        // No range trick here: 0x5C (`\`) and 0x7C (`|`) sit inside the
        // bracket ranges but are not structural.
        let special_lane = eq_mask16(simd, b, b'{')
            | eq_mask16(simd, b, b'}')
            | eq_mask16(simd, b, b'[')
            | eq_mask16(simd, b, b']')
            | eq_mask16(simd, b, b',')
            | eq_mask16(simd, b, b':');
        let ws_lane =
            (le_mask16(simd, b, 0x0d) & !le_mask16(simd, b, 0x08)) | eq_mask16(simd, b, b' ');
        let shift = 16 * lane;
        tok |= (dig | upper | lower | extra) << shift;
        special |= special_lane << shift;
        ws |= ws_lane << shift;
    }
    (ws, tok, special)
}

static WS_LUT: [u8; 16] = [
    0x20, 100, 100, 100, 17, 100, 113, 2, 100, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 100, 100,
];
static LO_LUT: [u8; 16] = [
    0x1a, 0x1e, 0x1e, 0x1e, 0x1f, 0x1e, 0x1e, 0x1e, 0x1e, 0x1e, 0x9c, 0x25, 0x4c, 0x25, 0x05, 0x0c,
];
static HI_LUT: [u8; 16] = [
    0x00, 0x00, 0x41, 0x82, 0x04, 0x28, 0x04, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

/// Duplicate a 16-byte table across both 128-bit lanes of a 32-byte table.
const fn doubled(lut: [u8; 16]) -> [u8; 32] {
    let mut out = [0_u8; 32];
    let mut half = 0;
    while half < 2 {
        let mut k = 0;
        while k < 16 {
            out[half * 16 + k] = lut[k];
            k += 1;
        }
        half += 1;
    }
    out
}

static WS_LUT32: [u8; 32] = doubled(WS_LUT);
static LO_LUT32: [u8; 32] = doubled(LO_LUT);
static HI_LUT32: [u8; 32] = doubled(HI_LUT);

/// Quadruple a 16-byte table across the four 128-bit lanes of a 64-byte
/// table.
const fn quadrupled(lut: [u8; 16]) -> [u8; 64] {
    let mut out = [0_u8; 64];
    let mut quarter = 0;
    while quarter < 4 {
        let mut k = 0;
        while k < 16 {
            out[quarter * 16 + k] = lut[k];
            k += 1;
        }
        quarter += 1;
    }
    out
}

static WS_LUT64: [u8; 64] = quadrupled(WS_LUT);
static LO_LUT64: [u8; 64] = quadrupled(LO_LUT);
static HI_LUT64: [u8; 64] = quadrupled(HI_LUT);

/// AVX2 classification through native 256-bit lanes: two lanes of
/// shuffle-fed nibble lookups (the [`classify_tables`] scheme, see its
/// documentation) at simdjson's original op count.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn classify_avx2(simd: Avx2, bytes: &[u8], i: usize) -> (u64, u64, u64) {
    let ws_lut = u8x32::from_slice(simd, &WS_LUT32);
    let lo_lut = u8x32::from_slice(simd, &LO_LUT32);
    let hi_lut = u8x32::from_slice(simd, &HI_LUT32);
    let zero = simd.splat_u8x32(0);
    let mut ws = 0_u64;
    let mut tok = 0_u64;
    let mut special = 0_u64;
    for (lane, b) in [
        u8x32::from_slice(simd, &bytes[i..i + 32]),
        u8x32::from_slice(simd, &bytes[i + 32..i + 64]),
    ]
    .into_iter()
    .enumerate()
    {
        let lo = b & 0x0f;
        let cls = simd.swizzle_dyn_within_blocks_u8x32(lo_lut, lo)
            & simd.swizzle_dyn_within_blocks_u8x32(hi_lut, b >> 4);
        let ws_lane = b.simd_eq(simd.swizzle_dyn_within_blocks_u8x32(ws_lut, lo));
        let tok_lane = (cls & 0x1f).simd_ne(zero);
        let special_lane = ((cls >> 5) & 0x07).simd_ne(zero);
        let shift = 32 * lane;
        ws |= ws_lane.to_bitmask() << shift;
        tok |= tok_lane.to_bitmask() << shift;
        special |= special_lane.to_bitmask() << shift;
    }
    (ws, tok, special)
}

/// AVX-512 classification: one full-width `vpermb` lookup per table over the
/// whole 64-byte block (the [`classify_tables`] scheme, see its
/// documentation), with the three masks converting through compact `k`
/// registers.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn classify_avx512(simd: Avx512, bytes: &[u8], i: usize) -> (u64, u64, u64) {
    let ws_lut = u8x64::from_slice(simd, &WS_LUT64);
    let lo_lut = u8x64::from_slice(simd, &LO_LUT64);
    let hi_lut = u8x64::from_slice(simd, &HI_LUT64);
    let zero = simd.splat_u8x64(0);
    let b = u8x64::from_slice(simd, &bytes[i..i + 64]);
    let lo = b & 0x0f;
    let cls = simd.swizzle_dyn_u8x64(lo_lut, lo) & simd.swizzle_dyn_u8x64(hi_lut, b >> 4);
    let ws = b.simd_eq(simd.swizzle_dyn_u8x64(ws_lut, lo));
    let tok = (cls & 0x1f).simd_ne(zero);
    let special = ((cls >> 5) & 0x07).simd_ne(zero);
    (ws.to_bitmask(), tok.to_bitmask(), special.to_bitmask())
}

/// Nibble-table classification (the simdjson scheme): one shuffle per nibble
/// replaces the compare chains, producing all three masks in a handful of
/// vector ops per 128-bit lane.
///
/// Whitespace uses a self-match table: the entry a byte's low nibble
/// selects equals the byte itself exactly for the six whitespace bytes
/// (every filler value's own low nibble differs from its index, so it can
/// never compare equal). simdjson's table extended with VT (`0x0B`) and FF
/// (`0x0C`) for JSON5.
///
/// Token and structural bytes come from two rectangle tables: each bit b
/// below owns one rectangle `H_b x L_b` (high-nibble set times low-nibble
/// set) of the target byte sets, and a byte's class is the AND of the bits
/// its two nibbles select:
///
/// ```text
/// bit 0: h{2}   x l{4,B,D,E}    $ + - .
/// bit 1: h{3}   x l{0-9}        0-9
/// bit 2: h{4,6} x l{1-F}        A-O, a-o
/// bit 3: h{5}   x l{0-A,C,F}    P-Z, backslash, _
/// bit 4: h{7}   x l{0-A}        p-z
/// bit 5: h{5,7} x l{B,D}        [ ] { }
/// bit 6: h{2}   x l{C}          ,
/// bit 7: h{3}   x l{A}          :
/// ```
///
/// Bits 0-4 mark token rectangles and bits 5-7 structural ones, so a nonzero
/// low-half byte means token and a nonzero top-three-bits byte means
/// structural. The rectangles are exact products, and the 256-value
/// exhaustive test below pins the tables to `CLASSES`.
///
/// Nibbles are masked explicitly before the lookup: dynamic-shuffle index
/// semantics differ per ISA (x86 `pshufb` indexes by the low 4 bits, NEON
/// `tbl` zeroes out-of-range indices), and masked nibbles make the lookup
/// identical everywhere.
#[inline(always)]
fn classify_tables<S: Simd>(simd: S, blocks: [u8x16<S>; 4]) -> (u64, u64, u64) {
    let ws_lut = u8x16::from_slice(simd, &WS_LUT);
    let lo_lut = u8x16::from_slice(simd, &LO_LUT);
    let hi_lut = u8x16::from_slice(simd, &HI_LUT);
    let zero = simd.splat_u8x16(0);
    let mut ws = 0_u64;
    let mut tok = 0_u64;
    let mut special = 0_u64;
    for (lane, b) in blocks.into_iter().enumerate() {
        let lo = b & 0x0f;
        let cls = simd.swizzle_dyn_within_blocks_u8x16(lo_lut, lo)
            & simd.swizzle_dyn_within_blocks_u8x16(hi_lut, b >> 4);
        let ws_lane = b.simd_eq(simd.swizzle_dyn_within_blocks_u8x16(ws_lut, lo));
        let tok_lane = (cls & 0x1f).simd_ne(zero);
        let special_lane = ((cls >> 5) & 0x07).simd_ne(zero);
        let shift = 16 * lane;
        ws |= ws_lane.to_bitmask() << shift;
        tok |= tok_lane.to_bitmask() << shift;
        special |= special_lane.to_bitmask() << shift;
    }
    (ws, tok, special)
}

/// Vectorized string skip: `(end, clean)` for the string opened at `open`,
/// where `end` is the position just past the closing quote and `clean`
/// marks content with no backslash and no LF/CR byte before it (the
/// stage-1 span contract; escape parity is *not* applied to the LF/CR
/// test — a line continuation's backslash disqualifies on its own, so the
/// conservative answer is exact for clean strings by construction).
#[inline(always)]
fn skip_string_simd<S: Simd>(simd: S, bytes: &[u8], open: usize) -> Result<(usize, bool)> {
    let quote = bytes[open];
    let len = bytes.len();
    let mut i = open + 1;
    let mut ends_odd = false;
    let mut clean = true;
    while i + 64 <= len {
        let (bs, quotes, lf, cr) = quad_mask_native(simd, bytes, i, b'\\', quote, b'\n', b'\r');
        // Any backslash or line terminator before the closer disqualifies
        // the span; bytes past the closer belong to following content.
        let dirt = bs | lf | cr;
        // A quote closes the string only when not escaped. With no
        // backslashes in the block the parity chain degenerates: the only
        // escape shadow is the carry-in on byte 0, and the carry always ends
        // here (matches odd_backslash_ends on a zero mask), so the ~10-op
        // chain is skipped for the common backslash-free block.
        let closers = if bs == 0 {
            let carried = u64::from(ends_odd);
            ends_odd = false;
            quotes & !carried
        } else {
            quotes & !odd_backslash_ends(bs, &mut ends_odd)
        };
        if closers != 0 {
            let t = closers.trailing_zeros() as usize;
            clean &= dirt & ((1_u64 << t) - 1) == 0;
            return Ok((i + t + 1, clean));
        }
        clean &= dirt == 0;
        i += 64;
    }
    // Scalar tail, carrying the pending escape parity and cleanliness.
    skip_string_tail(bytes, open, i, ends_odd, clean)
}

/// Vectorized comment skip mirroring [`skip_comment`].
#[inline(always)]
fn skip_comment_simd<S: Simd>(simd: S, bytes: &[u8], slash: usize) -> Result<usize> {
    match bytes.get(slash + 1) {
        Some(b'/') => Ok(skip_line_simd(simd, bytes, slash + 2)),
        Some(b'*') => skip_block_simd(simd, bytes, slash),
        _ => Err(Error::UnexpectedChar {
            byte: b'/',
            offset: slash,
        }),
    }
}

/// Line comment: position just past the next `\n`, or the end of input.
#[inline(always)]
fn skip_line_simd<S: Simd>(simd: S, bytes: &[u8], from: usize) -> usize {
    let len = bytes.len();
    let mut i = from;
    while i + 64 <= len {
        let lf = eq_mask_native(simd, bytes, i, b'\n');
        if lf != 0 {
            return i + lf.trailing_zeros() as usize + 1;
        }
        i += 64;
    }
    match memchr(b'\n', &bytes[i..]) {
        Some(nl) => i + nl + 1,
        None => len,
    }
}

/// Block comment: position just past the first `*/`, or the canonical
/// unterminated error.
#[inline(always)]
fn skip_block_simd<S: Simd>(simd: S, bytes: &[u8], slash: usize) -> Result<usize> {
    let len = bytes.len();
    let mut i = slash + 2;
    // Set when byte `i - 1` is `*` (a `*/` may straddle a block boundary).
    let mut prev_star = false;
    while i + 64 <= len {
        let (stars, slashes) = pair_mask_native(simd, bytes, i, b'*', b'/');
        let mut closers = (stars << 1) & slashes;
        if prev_star && slashes & 1 != 0 {
            closers |= 1;
        }
        if closers != 0 {
            return Ok(i + closers.trailing_zeros() as usize + 1);
        }
        prev_star = stars >> 63 != 0;
        i += 64;
    }
    // Scalar tail; `prev_star` carries the boundary-straddling `*`.
    while let Some(hit) = memchr(b'/', &bytes[i..]) {
        let at = i + hit;
        // The byte before the first candidate comes from the vector part;
        // every later candidate has its predecessor inside the tail.
        let prev_is_star = if hit == 0 {
            prev_star
        } else {
            bytes[at - 1] == b'*'
        };
        if prev_is_star {
            return Ok(at + 1);
        }
        i = at + 1;
    }
    Err(Error::UnterminatedComment { offset: slash })
}

/// Bit i of the returned mask is set when byte i directly follows an
/// odd-length run of backslashes, i.e. when byte i is escaped. Ported from
/// simdjson's `find_odd_backslash_sequences` (carry chains seeded at run
/// starts with the parity of their position); `ends_odd` carries the pending
/// parity into the next 64-byte block.
fn odd_backslash_ends(bs_bits: u64, ends_odd: &mut bool) -> u64 {
    const EVEN_BITS: u64 = 0x5555_5555_5555_5555;
    const ODD_BITS: u64 = !EVEN_BITS;
    let prev = u64::from(*ends_odd);
    let start_edges = bs_bits & !(bs_bits << 1);
    // An odd-length run ending the previous block flips which parity of run
    // start continues the chain at bit 0.
    let even_start_mask = EVEN_BITS ^ prev;
    let even_starts = start_edges & even_start_mask;
    let odd_starts = start_edges & !even_start_mask;
    let even_carries = bs_bits.wrapping_add(even_starts);
    let (odd_carries, ends_odd_now) = bs_bits.overflowing_add(odd_starts);
    let odd_carries = odd_carries | prev;
    *ends_odd = ends_odd_now;
    let even_carry_ends = even_carries & !bs_bits;
    let odd_carry_ends = odd_carries & !bs_bits;
    (even_carry_ends & ODD_BITS) | (odd_carry_ends & EVEN_BITS)
}

/// Generic 64-byte-block scan; `dispatch!` monomorphizes this for each ISA
/// inside a feature-enabled closure, so the whole body (and everything it
/// calls) must fold in through `#[inline(always)]`.
#[allow(clippy::cast_possible_truncation)] // guarded by check_input_len
#[inline(always)]
fn scan_simd<S: Simd>(
    simd: S,
    bytes: &[u8],
    index: &mut Vec<u32>,
    spans: &mut Vec<Span>,
) -> Result<()> {
    debug_assert!(u32::try_from(bytes.len()).is_ok());
    let len = bytes.len();
    let mut pos = 0usize;
    // True when the previous byte outside strings/comments was a token byte,
    // so the byte at `pos` continues a token run instead of starting one.
    let mut in_token = false;
    let classifier = classifier(simd);
    'blocks: while pos + 64 <= len {
        let (ws, tok, special) = classify_with(classifier, bytes, pos);
        let mut attn = !ws; // attention bits: every non-whitespace byte
        let block_end = pos + 64;
        if attn != 0 {
            // Push bits: structural bytes and token-run starts. A start is a
            // tok bit whose predecessor is not a token byte; adjacency in the
            // mask already encodes whitespace gaps, and a run continuing from
            // the previous block suppresses bit 0.
            let mut emit = special | (tok & !(tok << 1) & !u64::from(in_token));
            // Decision bytes are everything else that is not whitespace:
            // quotes, comments, high-bit sequences and illegal ASCII.
            // `cursor` is one past the last consumed attention byte; a gap
            // between `cursor` and the next decision byte is whitespace and
            // breaks a token run (mirrors the scalar `p > cursor` rule).
            let mut cursor = pos;
            while attn != 0 {
                let rare = attn & !special & !tok;
                let cut = if rare == 0 {
                    64
                } else {
                    rare.trailing_zeros() as usize
                };
                // Bulk-push every structural and token start before the next
                // decision byte: a tight tzcnt loop with no byte loads and no
                // per-byte dispatch.
                let low = if cut == 0 {
                    0
                } else {
                    u64::MAX >> (64 - cut)
                };
                let mut fast = emit & low;
                while fast != 0 {
                    let lowest = fast & fast.wrapping_neg();
                    index.push((pos + lowest.trailing_zeros() as usize) as u32);
                    fast ^= lowest;
                }
                emit &= !low;
                let below = attn & low; // attention bytes consumed above
                attn &= !low;
                if below != 0 {
                    // The tail of the consumed region carries into the
                    // decision byte only when directly adjacent, and only as
                    // a token byte (a structural tail or a trailing
                    // whitespace gap both end the run).
                    let hb = 63 - below.leading_zeros() as usize;
                    cursor = pos + hb + 1;
                    in_token = hb + 1 == cut && tok >> hb & 1 == 1;
                }
                if rare == 0 {
                    // What remains is pushes and token continuations; the
                    // block tail carries a token state iff its last byte is
                    // a token byte (a whitespace tail leaves tok bit 63
                    // clear, which is the same answer).
                    in_token = tok >> 63 & 1 == 1;
                    break;
                }
                let p = pos + cut; // in bounds: p < block_end <= len
                if p > cursor {
                    in_token = false;
                }
                let b = bytes[p];
                let jump = match b {
                    b'"' | b'\'' => {
                        index.push(p as u32);
                        in_token = false;
                        let (jump, clean) = skip_string_simd(simd, bytes, p)?;
                        if clean {
                            spans.push(Span {
                                open: p as u32,
                                close: (jump - 1) as u32,
                            });
                        }
                        jump
                    },
                    b'/' => {
                        in_token = false;
                        skip_comment_simd(simd, bytes, p)?
                    },
                    0x09..=0x0d | b' ' => {
                        // Only reachable if the vector whitespace mask
                        // disagreed with the scalar predicate. Handle it as
                        // whitespace so a bug surfaces as a differential
                        // mismatch, not a panic.
                        debug_assert!(false, "whitespace byte inside the attention mask");
                        in_token = false;
                        p + 1
                    },
                    b if b >= 0x80 => match decode_seq(bytes, p) {
                        None => return Err(Error::InvalidUtf8 { offset: p }),
                        Some((cp, seq_len)) => {
                            if is_unicode_whitespace(cp) {
                                in_token = false;
                            } else {
                                if !in_token {
                                    index.push(p as u32);
                                }
                                in_token = true;
                            }
                            p + seq_len
                        },
                    },
                    // ASCII reaching the decision walk is never a token or
                    // structural byte: it is illegal.
                    b => return Err(Error::UnexpectedChar { byte: b, offset: p }),
                };
                if jump >= block_end {
                    // A string, comment or multi-byte sequence consumed past
                    // the block end: resume scanning at `jump` (loads are
                    // unaligned).
                    pos = jump;
                    continue 'blocks;
                }
                // Consume this byte (and anything the arm swallowed with it).
                let keep = !((1_u64 << (jump - pos)) - 1);
                attn &= keep;
                emit &= keep;
                cursor = jump;
                if b >= 0x80 && in_token {
                    // A high-bit identifier sequence continues a token run,
                    // but the precomputed starts bit at the resume position
                    // only saw its non-token predecessor: suppress it. A
                    // structural byte there must still be emitted.
                    let bit = 1_u64 << (jump - pos);
                    if special & bit == 0 {
                        emit &= !bit;
                    }
                }
            }
        }
        // Whitespace at the block tail means the last byte before the
        // boundary is not a token byte, so any token run broke before it.
        // The `attn == 0` whole-whitespace block and the inner-loop exit on
        // an exhausted attention mask skip the bit-63 recomputation above,
        // leaving a stale carry that would swallow the next token's start.
        // (`continue 'blocks` skips this on purpose: a skip that landed past
        // the boundary already carries the class of its last consumed byte.)
        if ws >> 63 & 1 != 0 {
            in_token = false;
        }
        pos = block_end;
    }
    // Finish the partial trailing block scalar, carrying the token-run state.
    scan_scalar(bytes, pos, in_token, index, spans)
}

#[cfg(test)]
mod tests {
    // Hand-written expected offsets are tiny and cannot truncate.
    #![allow(clippy::cast_possible_truncation)]

    #[cfg(target_arch = "aarch64")]
    use fearless_simd::Neon;
    #[cfg(target_arch = "x86_64")]
    use fearless_simd::{Avx2, Sse2, Sse4_2};

    use super::*;
    use crate::parser::{
        stage1,
        stage1::{CLASSES, class},
    };

    type KernelFn = fn(&[u8], &mut Vec<u32>, &mut Vec<Span>) -> Result<()>;

    /// Per level: classification masks from every strategy.
    type Strategies = Vec<(&'static str, (u64, u64, u64))>;

    fn scan_fallback(bytes: &[u8], index: &mut Vec<u32>, spans: &mut Vec<Span>) -> Result<()> {
        dispatch!(Level::fallback(), simd => scan_simd(simd, bytes, index, spans))
    }

    #[cfg(target_arch = "x86_64")]
    fn scan_sse2(bytes: &[u8], index: &mut Vec<u32>, spans: &mut Vec<Span>) -> Result<()> {
        // SAFETY: SSE2 (with sse/fxsr) is the x86_64 baseline.
        dispatch!(Level::Sse2(unsafe { Sse2::assume_supported() }), simd => scan_simd(simd, bytes, index, spans))
    }

    #[cfg(target_arch = "x86_64")]
    fn scan_sse4_2(bytes: &[u8], index: &mut Vec<u32>, spans: &mut Vec<Span>) -> Result<()> {
        // SAFETY: pushed into `kernels()` only after detecting every feature
        // this token requires.
        dispatch!(Level::Sse4_2(unsafe { Sse4_2::assume_supported() }), simd => scan_simd(simd, bytes, index, spans))
    }

    #[cfg(target_arch = "x86_64")]
    fn scan_avx2(bytes: &[u8], index: &mut Vec<u32>, spans: &mut Vec<Span>) -> Result<()> {
        // SAFETY: pushed into `kernels()` only after detecting every feature
        // this token requires.
        dispatch!(Level::Avx2(unsafe { Avx2::assume_supported() }), simd => scan_simd(simd, bytes, index, spans))
    }

    #[cfg(target_arch = "aarch64")]
    fn scan_neon(bytes: &[u8], index: &mut Vec<u32>, spans: &mut Vec<Span>) -> Result<()> {
        // SAFETY: NEON is the aarch64 baseline.
        dispatch!(Level::Neon(unsafe { Neon::assume_supported() }), simd => scan_simd(simd, bytes, index, spans))
    }

    fn scan_dispatched(bytes: &[u8], index: &mut Vec<u32>, spans: &mut Vec<Span>) -> Result<()> {
        build_index(bytes, index, spans)
    }

    /// All kernels available on this machine, including the dispatch entry.
    fn kernels() -> Vec<(&'static str, KernelFn)> {
        let mut kernels: Vec<(&'static str, KernelFn)> = vec![("fallback", scan_fallback)];
        #[cfg(target_arch = "x86_64")]
        {
            kernels.push(("sse2", scan_sse2));
            // Feature sets must match each token's `assume_supported` contract.
            let sse4_2 = std::is_x86_feature_detected!("sse4.2")
                && std::is_x86_feature_detected!("cmpxchg16b")
                && std::is_x86_feature_detected!("popcnt");
            if sse4_2 {
                kernels.push(("sse4.2", scan_sse4_2));
            }
            let avx2 = std::is_x86_feature_detected!("avx2")
                && std::is_x86_feature_detected!("bmi1")
                && std::is_x86_feature_detected!("bmi2")
                && std::is_x86_feature_detected!("cmpxchg16b")
                && std::is_x86_feature_detected!("f16c")
                && std::is_x86_feature_detected!("fma")
                && std::is_x86_feature_detected!("lzcnt")
                && std::is_x86_feature_detected!("movbe")
                && std::is_x86_feature_detected!("popcnt")
                && std::is_x86_feature_detected!("xsave");
            if avx2 {
                kernels.push(("avx2", scan_avx2));
            }
        }
        #[cfg(target_arch = "aarch64")]
        kernels.push(("neon", scan_neon));
        kernels.push(("dispatch", scan_dispatched));
        kernels
    }

    /// The core differential: every kernel must agree with the canonical
    /// scalar stage 1 on both the result and the full index and span-table
    /// contents (including the partial contents left behind on error).
    fn differential(bytes: &[u8]) {
        let mut oracle = Vec::new();
        let mut oracle_spans = Vec::new();
        let expected = stage1::build_index(bytes, &mut oracle, &mut oracle_spans);
        for (label, kernel) in kernels() {
            let mut got = Vec::new();
            let mut got_spans = Vec::new();
            let result = kernel(bytes, &mut got, &mut got_spans);
            assert!(
                result == expected,
                "{label}: result {result:?} != oracle {expected:?} for {bytes:02x?}"
            );
            assert!(
                got == oracle,
                "{label}: index {got:?} != oracle {oracle:?} for {bytes:02x?}"
            );
            assert!(
                got_spans == oracle_spans,
                "{label}: spans {got_spans:?} != oracle {oracle_spans:?} for {bytes:02x?}"
            );
        }
    }

    /// Dispatched index for a valid input.
    fn idx(bytes: &[u8]) -> Vec<u32> {
        let mut v = Vec::new();
        build_index(bytes, &mut v, &mut Vec::new()).unwrap();
        v
    }

    /// Dispatched error for an invalid input.
    fn err(bytes: &[u8]) -> Error {
        let mut v = Vec::new();
        build_index(bytes, &mut v, &mut Vec::new()).unwrap_err()
    }

    #[test]
    fn empty_and_whitespace_only() {
        assert_eq!(idx(b""), Vec::<u32>::new());
        assert_eq!(idx(b" \t\r\n\x0b\x0c"), Vec::<u32>::new());
        differential(b"");
        differential(b" \t\r\n\x0b\x0c");
    }

    #[test]
    fn token_starts_after_every_separator_kind() {
        // ASCII whitespace and Unicode whitespace separate runs.
        for sep in [
            "\t", "\n", "\r", "\x0b", "\x0c", " ", "\u{a0}", "\u{1680}", "\u{2000}", "\u{200a}",
            "\u{2028}", "\u{2029}", "\u{202f}", "\u{205f}", "\u{3000}", "\u{feff}",
        ] {
            let input = format!("a{sep}b");
            differential(input.as_bytes());
            assert_eq!(idx(input.as_bytes()), vec![0, 1 + sep.len() as u32]);
        }
        // Structural bytes separate and are emitted themselves.
        for sep in ["{", "}", "[", "]", ",", ":"] {
            let input = format!("a{sep}b");
            differential(input.as_bytes());
            assert_eq!(idx(input.as_bytes()), vec![0, 1, 2]);
        }
        // Quote bytes open strings: the opener is emitted, the rest consumed.
        for input in ["a\"b\"", "a'b'", "\"", "'"] {
            differential(input.as_bytes());
        }
        assert_eq!(idx(b"a\"b\""), vec![0, 1]);
        // Slash introduces comments.
        differential(b"a//c\nb");
        assert_eq!(idx(b"a//c\nb"), vec![0, 5]);
        differential(b"a/*c*/b");
        assert_eq!(idx(b"a/*c*/b"), vec![0, 6]);
        // Beginning of input acts as a separator.
        differential(b"abc");
        assert_eq!(idx(b"abc"), vec![0]);
        differential(b" abc");
        assert_eq!(idx(b" abc"), vec![1]);
    }

    #[test]
    fn structurals_inside_strings_and_comments_are_skipped() {
        differential(br#"{"a,b":[1]}"#);
        assert_eq!(idx(br#"{"a,b":[1]}"#), vec![0, 1, 6, 7, 8, 9, 10]);
        differential(b"[1/*c*/,2]");
        assert_eq!(idx(b"[1/*c*/,2]"), vec![0, 1, 7, 8, 9]);
        differential(br#""{},:[]""#);
        assert_eq!(idx(br#""{},:[]""#), vec![0]);
        differential(br"/*{['\']}*/x");
        assert_eq!(idx(br"/*{['\']}*/x"), vec![11]);
    }

    #[test]
    fn comments_line_and_block() {
        differential(b"// c\n1");
        assert_eq!(idx(b"// c\n1"), vec![5]);
        // Line comment may end at end of input without a newline.
        differential(b"// no newline");
        assert_eq!(idx(b"// no newline"), Vec::<u32>::new());
        differential(b"/**/");
        differential(b"/* multi\nline */[1]");
        assert_eq!(idx(b"/* multi\nline */[1]"), vec![16, 17, 18]);
        // Block comments do not nest; first */ ends them.
        differential(b"/* /* */ 1");
        assert_eq!(idx(b"/* /* */ 1"), vec![9]);
        // Unterminated variants.
        differential(b"/*");
        differential(b"/* x");
        assert_eq!(err(b"/* x"), Error::UnterminatedComment { offset: 0 });
        differential(b"a /*");
        assert_eq!(err(b"a /*"), Error::UnterminatedComment { offset: 2 });
        assert_eq!(err(b"/*/"), Error::UnterminatedComment { offset: 0 });
        // Stray slash and slash at end of input.
        differential(b"/x");
        assert_eq!(err(b"/x"), Error::UnexpectedChar {
            byte: b'/',
            offset: 0
        });
        differential(b"5/");
        assert_eq!(err(b"5/"), Error::UnexpectedChar {
            byte: b'/',
            offset: 1
        });
        differential(b"/");
        assert_eq!(err(b"/"), Error::UnexpectedChar {
            byte: b'/',
            offset: 0
        });
    }

    #[test]
    fn strings_quotes_and_escapes() {
        differential(br#""abc""#);
        differential(b"''");
        differential(br#""a\"b""#);
        differential(br"'don\'t'");
        differential(br#""it's""#);
        assert_eq!(idx(br#""it's""#), vec![0]);
        differential(br#"'say "hi"'"#);
        assert_eq!(idx(br#"'say "hi"'"#), vec![0]);
        // Escaped backslash does not escape the closing quote.
        differential(br#""a\\""#);
        assert_eq!(idx(br#""a\\""#), vec![0]);
        // Unterminated strings carry the opening quote offset.
        differential(b"\"abc");
        assert_eq!(err(b"\"abc"), Error::UnterminatedString { offset: 0 });
        differential(b"'a\\'");
        differential(b"x 'abc");
        assert_eq!(err(b"x 'abc"), Error::UnterminatedString { offset: 2 });
        differential(br#""a\"#);
        assert_eq!(err(br#""a\"#), Error::UnterminatedString { offset: 0 });
    }

    #[test]
    fn illegal_bytes_are_rejected() {
        for (input, want) in [
            (&b"!"[..], Error::UnexpectedChar {
                byte: b'!',
                offset: 0,
            }),
            (&b"[1?]"[..], Error::UnexpectedChar {
                byte: b'?',
                offset: 2,
            }),
            (&b"a\x00b"[..], Error::UnexpectedChar { byte: 0, offset: 1 }),
            (&b"("[..], Error::UnexpectedChar {
                byte: b'(',
                offset: 0,
            }),
            (&b"a~b"[..], Error::UnexpectedChar {
                byte: b'~',
                offset: 1,
            }),
            (&b"\x7f"[..], Error::UnexpectedChar {
                byte: 0x7f,
                offset: 0,
            }),
        ] {
            differential(input);
            assert_eq!(err(input), want);
        }
        // Partial index contents on error match the canonical scalar.
        let mut got = Vec::new();
        assert!(build_index(b"a!", &mut got, &mut Vec::new()).is_err());
        assert_eq!(got, vec![0]);
    }

    #[test]
    fn backslash_is_a_token_byte_outside_strings() {
        // `\uXXXX` identifier escapes keep the whole identifier one token.
        differential(br"a\u0062c");
        assert_eq!(idx(br"a\u0062c"), vec![0]);
        differential(br"\key");
        assert_eq!(idx(br"\key"), vec![0]);
        differential(b"a \\ b");
        assert_eq!(idx(b"a \\ b"), vec![0, 2, 4]);
    }

    #[test]
    fn unicode_whitespace_and_non_whitespace_high_bytes() {
        // Lookalikes that are NOT whitespace: single token runs. U+0085 (NEL)
        // is Cc, not Zs: the JSON5 separator set excludes it.
        for cp in [
            0x85u32, 0x84, 0x200b, 0x200c, 0x200d, 0x2060, 0x2010, 0x180e,
        ] {
            let ch = char::from_u32(cp).unwrap();
            let input = format!("a{ch}b");
            differential(input.as_bytes());
            assert_eq!(idx(input.as_bytes()), vec![0]);
        }
        // A Unicode whitespace sequence may interrupt a token run mid-way
        // (valid token sequence first, then U+2028).
        differential(&[0xc3, 0xa9, 0xe2, 0x80, 0xa8]);
        assert_eq!(idx(&[0xc3, 0xa9, 0xe2, 0x80, 0xa8]), vec![0]);
        // A lead byte followed by a non-continuation byte is malformed.
        differential(&[0xc2, 0xe2, 0x80, 0xa8]);
        assert_eq!(err(&[0xc2, 0xe2, 0x80, 0xa8]), Error::InvalidUtf8 {
            offset: 0
        });
        // Non-ASCII identifiers: whole sequences are token units.
        differential("héllo wörld".as_bytes());
        assert_eq!(idx("héllo wörld".as_bytes()), vec![0, 7]);
        differential("日本語".as_bytes());
        assert_eq!(idx("日本語".as_bytes()), vec![0]);
    }

    #[test]
    fn unicode_whitespace_full() {
        let mut cps: Vec<u32> = vec![0xa0, 0x1680];
        cps.extend(0x2000..=0x200a);
        cps.extend([0x2028, 0x2029, 0x202f, 0x205f, 0x3000, 0xfeff]);
        for cp in cps {
            let ch = char::from_u32(cp).unwrap();
            let input = format!("a{ch}b");
            differential(input.as_bytes());
            assert_eq!(
                idx(input.as_bytes()),
                vec![0, 1 + ch.len_utf8() as u32],
                "code point U+{cp:04X}"
            );
            // At the beginning of input it acts like leading whitespace.
            let lead = format!("{ch}b");
            differential(lead.as_bytes());
            assert_eq!(
                idx(lead.as_bytes()),
                vec![ch.len_utf8() as u32],
                "U+{cp:04X} lead"
            );
        }
    }

    #[test]
    fn malformed_utf8_outside_strings_is_rejected() {
        for (input, offset) in [
            (&b"a\x80b"[..], 1usize),
            (&b"\xE2\x41\x80"[..], 0),
            (&b"\xE2\x80"[..], 0),
            (&b"\xF0\x90\x80"[..], 0),
            (&b"\xF8\x80\x80\x80"[..], 0),
        ] {
            differential(input);
            assert_eq!(err(input), Error::InvalidUtf8 { offset });
        }
        // Structurally valid sequences (including overlong forms, per the
        // canonical decode_seq) are single token units.
        differential(b"\xC0\x85x");
        assert_eq!(idx(b"\xC0\x85x"), vec![0]);
        differential(b"a\xF0\x90\x80\x80b");
        assert_eq!(idx(b"a\xF0\x90\x80\x80b"), vec![0]);
        differential("aéb c".as_bytes());
        assert_eq!(idx("aéb c".as_bytes()), vec![0, 5]);
    }

    #[test]
    fn canonical_document_examples() {
        differential(b"{},[]:");
        assert_eq!(idx(b"{},[]:"), vec![0, 1, 2, 3, 4, 5]);
        differential(b"true");
        differential(b"-Infinity");
        differential(b"0x1F +2 1.e3");
        assert_eq!(idx(b"0x1F +2 1.e3"), vec![0, 5, 8]);
        differential(b"1,2");
        differential(b"a-b.c+d");
        differential(b"a$b _c");
        assert_eq!(idx(b"a$b _c"), vec![0, 4]);
        differential(br#"["a", 'b']"#);
        assert_eq!(idx(br#"["a", 'b']"#), vec![0, 1, 4, 6, 9]);
        differential(b"/**/[1,]");
        assert_eq!(idx(b"/**/[1,]"), vec![4, 5, 6, 7]);
        let src = b"{ a: 1, // comment\n  'b': [2, 3,] }";
        differential(src);
        assert_eq!(idx(src), vec![
            0, 2, 3, 5, 6, 21, 24, 26, 27, 28, 30, 31, 32, 34
        ]);
        let s = "1\u{a0}2\u{2028}3\u{3000}4\u{feff}5";
        differential(s.as_bytes());
        assert_eq!(idx(s.as_bytes()), vec![0, 3, 7, 11, 15]);
    }

    /// Systematic 64-byte boundary sweep: closers, escape-parity carries and
    /// token runs crossing block boundaries at every alignment.
    #[test]
    fn block_boundary_sweep() {
        for k in 0..=140usize {
            let x = "x".repeat(k);
            differential(format!("\"{x}\"").as_bytes()); // plain string
            differential(format!("\"{}\"", "\\\"".repeat(k)).as_bytes()); // escaped quotes
            // Raw backslash storms: even count closes the string, odd does not.
            let mut storm = Vec::with_capacity(2 * k + 2);
            storm.push(b'"');
            storm.extend(std::iter::repeat_n(b'\\', k));
            storm.push(b'"');
            differential(&storm);
            // Quote storms alternate open/close.
            differential(&vec![b'"'; k]);
            differential(format!("/*{}*/", "*".repeat(k)).as_bytes());
            differential(format!("//{}\n{x}", "y".repeat(k)).as_bytes());
            differential(format!("//{x}").as_bytes()); // line comment to EOF
            // Token runs and whitespace-separated runs across boundaries.
            differential(x.as_bytes());
            differential(format!("{x} {x}").as_bytes());
            differential(format!("{x},{x}").as_bytes());
            // High-bit sequences straddling the boundary.
            let ws_seq = "\u{2028}".repeat(k);
            differential(format!("a{ws_seq}b").as_bytes());
            let ident = "\u{e9}".repeat(k);
            differential(format!("{ident} {ident}").as_bytes());
        }
        // A whole-block whitespace run between tokens must not swallow the
        // second token's start index (the `in_token` carry resets at the
        // boundary): token block + pure-whitespace block + trailing token.
        differential(format!("{}{}y", "x".repeat(64), " ".repeat(64)).as_bytes());
        // Same with a partial second block: the high-bit sequence ends a
        // block filled with exactly enough spaces to exhaust `attn`.
        differential(format!("\u{e9}{}y", " ".repeat(62)).as_bytes());
        // Mixed tab/newline whitespace blocks (different ws bytes, same run).
        differential(format!("{}\t\ny", "\t\n".repeat(32)).as_bytes());
        differential(format!("x{}\t\ny", "\t\n".repeat(32)).as_bytes());
        // An odd-length backslash run ending a string-skip block (they start
        // at open+1, so byte 64 here), then a backslash-free block: the
        // escape carry must still shadow the next block's byte 0 even though
        // the parity chain is skipped there; an even-length run must not.
        differential(format!("\"{}\\\"{}\"y", "a".repeat(63), "b".repeat(62)).as_bytes());
        differential(format!("\"{}\\\"{}", "a".repeat(63), "b".repeat(63)).as_bytes());
        differential(format!("\"{}\\\\\"{}\"y", "a".repeat(62), "b".repeat(62)).as_bytes());
    }

    /// A vector string skip can end on a block boundary with a pending escape
    /// whose target is not a quote or backslash: the scalar tail must consume
    /// exactly that target byte, never the next quote/backslash hit (a former
    /// bug swallowed the closing quote and reported the string unterminated).
    #[test]
    fn tail_escape_carry_consumes_exactly_one_target() {
        // The backslash lands on the last byte of a 64-byte block for
        // k = 63 + 64n; the sweep covers every alignment.
        for k in 0..=140usize {
            // Plain-letter target, close in the tail.
            differential(format!("\"{}\\n{}\"", "a".repeat(k), "b".repeat(140 - k)).as_bytes());
            // Target is the other quote kind, a backslash, or end of input.
            differential(format!("\"{}\\'{}\"", "a".repeat(k), "b".repeat(140 - k)).as_bytes());
            differential(format!("\"{}\\\\{}\"", "a".repeat(k), "b".repeat(140 - k)).as_bytes());
            differential(format!("\"{}\\n", "a".repeat(k)).as_bytes());
        }
        // The originally failing shape: backslash at byte 64, target 'n'.
        let s = format!("\"{}\\n\"", "a".repeat(63));
        differential(s.as_bytes());
        assert_eq!(idx(s.as_bytes()), vec![0]);
    }

    /// Random bytes from a JSON5-ish alphabet: hits accept paths, every error
    /// variant, and every mask class.
    #[test]
    fn random_bytes_differential() {
        use rand::{RngExt, SeedableRng, rngs::StdRng};

        let pool: Vec<u8> = b" \t\r\n\x0b\x0c{}[],:\"'\\/abcXYZ019$_.+-!?()<>#=%~`^|;&*"
            .iter()
            .copied()
            .chain([
                0x00, 0x01, 0x08, 0x0e, 0x1f, 0x7f, 0xc2, 0x85, 0xa0, 0xc3, 0xa9, 0xc0, 0xc1, 0xe1,
                0x9a, 0x80, 0xe2, 0x81, 0x9f, 0xe3, 0xef, 0xbb, 0xbf, 0xa8, 0xa9, 0xaf, 0x8a, 0x8b,
                0xf0, 0x90, 0xf4, 0x8f, 0xf8, 0xff,
            ])
            .collect();
        let mut ok_seen = 0;
        let mut err_seen = 0;
        for seed in 0..3000u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let len = if seed % 20 == 0 {
                rng.random_range(200..=800)
            } else {
                rng.random_range(0..=160)
            };
            let input: Vec<u8> = (0..len)
                .map(|_| pool[rng.random_range(0..pool.len())])
                .collect();
            let mut probe = Vec::new();
            if stage1::build_index(&input, &mut probe, &mut Vec::new()).is_ok() {
                ok_seen += 1;
            } else {
                err_seen += 1;
            }
            differential(&input);
        }
        assert!(ok_seen > 0 && err_seen > 0, "generator lost diversity");
    }

    /// Structured random documents: valid JSON5-ish fragments glued with
    /// random separators, plus frequent corruption for error coverage.
    #[test]
    fn structured_differential() {
        use rand::{RngExt, SeedableRng, rngs::StdRng};

        for seed in 0..2500u64 {
            let mut rng = StdRng::seed_from_u64(0x5eed_0000 + seed);
            let pieces = rng.random_range(0..=40);
            let mut doc: Vec<u8> = Vec::new();
            for _ in 0..pieces {
                match rng.random_range(0..16) {
                    0 => doc.extend_from_slice(b"{"),
                    1 => doc.extend_from_slice(b"}"),
                    2 => doc.extend_from_slice(b"["),
                    3 => doc.extend_from_slice(b"]"),
                    4 => doc.extend_from_slice(b","),
                    5 => doc.extend_from_slice(b":"),
                    6 => doc.extend_from_slice(b"key"),
                    7 => doc.extend_from_slice(b"$id_1"),
                    8 => doc.extend_from_slice(match rng.random_range(0..6) {
                        0 => &b"0x1F"[..],
                        1 => &b"+1.5e3"[..],
                        2 => &b"-.5"[..],
                        3 => &b"Infinity"[..],
                        4 => &b"NaN"[..],
                        _ => &b"1."[..],
                    }),
                    9 => {
                        // String with mixed escapes; sometimes unterminated.
                        let quote = if rng.random_bool(0.5) {
                            b'"'
                        } else {
                            b'\''
                        };
                        doc.push(quote);
                        for _ in 0..rng.random_range(0..12) {
                            match rng.random_range(0..8) {
                                0 => doc.extend_from_slice(b"\\\""),
                                1 => doc.extend_from_slice(b"\\\\"),
                                2 => doc.extend_from_slice(b"\\u00e9"),
                                3 => doc.extend_from_slice(b"\\n"),
                                4 => doc.push(b','),
                                5 => doc.extend_from_slice("\u{2028}".as_bytes()),
                                6 => doc.push(0xc3), // high-bit noise
                                _ => doc.push(b'x'),
                            }
                        }
                        if rng.random_bool(0.85) {
                            doc.push(quote);
                        } else if rng.random_bool(0.5) {
                            doc.push(b'\\'); // dangling escape
                        }
                    },
                    10 => {
                        // Line comment; sometimes without the newline.
                        doc.extend_from_slice(b"//");
                        doc.extend(std::iter::repeat_n(b'x', rng.random_range(0..12)));
                        if rng.random_bool(0.7) {
                            doc.push(b'\n');
                        }
                    },
                    11 => {
                        // Block comment with inner stars/slashes; sometimes
                        // unterminated.
                        doc.extend_from_slice(b"/*");
                        for _ in 0..rng.random_range(0..12) {
                            match rng.random_range(0..6) {
                                0 => doc.extend_from_slice(b"*/"),
                                1 => doc.extend_from_slice(b"/*"),
                                2 => doc.extend_from_slice(b"**"),
                                _ => doc.push(b'y'),
                            }
                        }
                        if rng.random_bool(0.85) {
                            doc.extend_from_slice(b"*/");
                        }
                    },
                    12 => {
                        // Unicode whitespace separators.
                        let cps = [
                            0x85u32, 0xa0, 0x1680, 0x2003, 0x200a, 0x2028, 0x2029, 0x202f, 0x205f,
                            0x3000, 0xfeff, 0x200b,
                        ];
                        let cp = cps[rng.random_range(0..cps.len())];
                        doc.extend_from_slice(char::from_u32(cp).unwrap().to_string().as_bytes());
                    },
                    13 => {
                        // ASCII whitespace run.
                        for _ in 0..rng.random_range(1..4) {
                            doc.push(b" \t\r\n"[rng.random_range(0..4)]);
                        }
                    },
                    14 => doc.extend_from_slice(match rng.random_range(0..4) {
                        0 => &b"/"[..],
                        1 => &b"/x"[..],
                        2 => &b"!"[..],
                        _ => &b"\\"[..],
                    }),
                    _ => doc.extend_from_slice("\u{e9}\u{65e5}".as_bytes()),
                }
            }
            differential(&doc);
        }
    }

    /// Long inputs: many blocks per skip, tails everywhere.
    #[test]
    fn large_and_pathological_inputs() {
        let mut doc = Vec::new();
        for i in 0..2000 {
            match i % 5 {
                0 => doc.extend_from_slice(b"{key_"),
                1 => doc.extend_from_slice(format!("{i}: 'v {i}', // tail\n").as_bytes()),
                2 => doc.extend_from_slice(b"/* block */ [1.5e3, -0x1F, NaN], "),
                3 => doc.extend_from_slice("\"\u{e9}sc\\\"\" ".as_bytes()),
                _ => doc.extend_from_slice("\u{2028}Infinity,\u{feff}".as_bytes()),
            }
        }
        differential(&doc);

        // Escape storms: odd-length backslash runs crossing many blocks.
        for n in [63usize, 64, 65, 127, 128, 129, 100_000] {
            let mut storm = vec![b'"'];
            storm.extend(std::iter::repeat_n(b'\\', n));
            storm.push(b'"');
            differential(&storm);
        }
        // Star storms inside block comments.
        let mut stars = b"a/*".to_vec();
        stars.extend(std::iter::repeat_n(b'*', 100_000));
        stars.extend_from_slice(b"*/b");
        differential(&stars);
        // A single bare token spanning hundreds of blocks.
        let mut tok = Vec::new();
        tok.extend(std::iter::repeat_n(b'a', 300_000));
        differential(&tok);
        let mut tok2 = tok.clone();
        tok2.push(b' ');
        tok2.push(b'b');
        differential(&tok2);
    }

    /// Exhaustive: every level's classification, both compare-chain and
    /// nibble-table strategy, must agree with the scalar `CLASSES` byte table
    /// (and the exact whitespace set) for all 256 byte values (4 blocks of
    /// 64). This is what catches lookalike bytes inside range tricks (0x5C
    /// inside `[`..`]`, 0x7C inside `{`..=`}`) and nibble-table mistakes,
    /// including on emulated-shuffle levels where the table path is unused.
    #[test]
    fn masks_match_class_table() {
        let mut bytes = [0_u8; 256];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = i as u8;
        }
        let ws_of = |b: usize| matches!(b, 0x09..=0x0d | 0x20);

        let mut levels: Vec<(&'static str, Strategies)> = Vec::new();
        let mut push = |label: &'static str, strategies: Strategies| {
            levels.push((label, strategies));
        };
        macro_rules! level {
            ($label:literal, $level:expr) => {
                dispatch!($level, simd => {
                    let blocks = load64(simd, &bytes, 0);
                    push($label, vec![
                        ("dispatched", classify_with(classifier(simd), &bytes, 0)),
                        ("compares", classify_compares(simd, blocks)),
                        ("tables", classify_tables(simd, blocks)),
                    ]);
                });
            };
        }
        level!("fallback", Level::fallback());
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: SSE2 (with sse/fxsr) is the x86_64 baseline.
            level!("sse2", Level::Sse2(unsafe { Sse2::assume_supported() }));
            // SAFETY: detected alongside every required feature.
            if std::is_x86_feature_detected!("sse4.2")
                && std::is_x86_feature_detected!("cmpxchg16b")
                && std::is_x86_feature_detected!("popcnt")
            {
                level!(
                    "sse4.2",
                    Level::Sse4_2(unsafe { Sse4_2::assume_supported() })
                );
            }
            // SAFETY: detected alongside every required feature.
            if std::is_x86_feature_detected!("avx2")
                && std::is_x86_feature_detected!("bmi1")
                && std::is_x86_feature_detected!("bmi2")
                && std::is_x86_feature_detected!("cmpxchg16b")
                && std::is_x86_feature_detected!("f16c")
                && std::is_x86_feature_detected!("fma")
                && std::is_x86_feature_detected!("lzcnt")
                && std::is_x86_feature_detected!("movbe")
                && std::is_x86_feature_detected!("popcnt")
                && std::is_x86_feature_detected!("xsave")
            {
                level!("avx2", Level::Avx2(unsafe { Avx2::assume_supported() }));
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON is the aarch64 baseline.
            level!("neon", Level::Neon(unsafe { Neon::assume_supported() }));
        }

        for (level_label, strategies) in levels {
            for (strategy, (ws, tok, special)) in strategies {
                for lane in 0..4usize {
                    for i in 0..16usize {
                        let b = lane * 16 + i;
                        let bit = 1_u64 << (lane * 16 + i);
                        assert_eq!(
                            tok & bit != 0,
                            CLASSES[b] & class::TOKEN != 0,
                            "{level_label}/{strategy} token: byte {b:#04x}"
                        );
                        assert_eq!(
                            special & bit != 0,
                            CLASSES[b] & class::STRUCTURAL != 0,
                            "{level_label}/{strategy} structural: byte {b:#04x}"
                        );
                        assert_eq!(
                            ws & bit != 0,
                            ws_of(b),
                            "{level_label}/{strategy} ws: byte {b:#04x}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn kernel_availability_report() {
        // Informational: which kernels this machine verifies. Run with
        // `--no-capture` to see it.
        println!("level: {:?}", Level::new());
        assert!(kernels().iter().any(|(label, _)| *label == "fallback"));
    }
}
