//! SIMD backends for the stage-1 structural indexer.
//!
//! This module implements the stage-1 contract (see the module documentation
//! of [`super::stage1`], the authoritative copy) with vector kernels and a
//! cached dispatch:
//!
//! - x86_64: an SSE2 kernel (baseline ISA, always available). An AVX2 kernel is selected once per
//!   process through a `OnceLock`, so the CPU probe never runs on the hot path.
//! - every other target (including aarch64): the canonical scalar [`super::stage1::build_index`]. A
//!   NEON kernel cannot be run or differentially verified on the machines available to this branch
//!   (NEON has no movemask; a hand-derived bit-extraction chain would be exactly the kind of
//!   subtly-wrong code the differential tests exist to catch), so the canonical scalar scan stands
//!   in for it: correct-and-slower beats subtly-wrong. The non-x86_64 routing is compile-checked
//!   for `aarch64-unknown-linux-gnu`.
//!
//! Kernel algorithm (one 64-byte block per iteration):
//!
//! 1. Classify the block with vector compares, producing a 64-bit whitespace mask; its complement
//!    (`attention mask`) marks every byte that needs a scalar decision.
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
//! All backends produce byte-identical index output and error values for the
//! same input, down to the partial index contents left behind on error. The
//! differential tests in this module enforce that against the canonical
//! scalar implementation.

use super::stage1::{CLASSES, class, decode_seq, is_unicode_whitespace, token_end};
use crate::{Error, Result};

/// Reject inputs whose offsets do not fit `u32`; mirrors the canonical guard.
#[cfg(target_arch = "x86_64")]
fn check_input_len(bytes: &[u8]) -> Result<()> {
    if bytes.len() > u32::MAX as usize {
        return Err(Error::Syntax {
            message: "input larger than 4 GiB is not supported".into(),
            offset: 0,
        });
    }
    Ok(())
}

/// Position just past the string whose opening quote is at `open`, honoring
/// backslash escape parity. Mirrors the private canonical `skip_string`.
#[cfg(target_arch = "x86_64")]
fn skip_string(bytes: &[u8], open: usize) -> Result<usize> {
    skip_string_tail(bytes, open, open + 1, false)
}

/// Scalar string skip from `i` with an incoming escape-parity flag (used when
/// the vector part of a skip ends mid-escape). Mirrors the canonical
/// backslash-consumes-the-next-byte semantics.
#[cfg(target_arch = "x86_64")]
fn skip_string_tail(bytes: &[u8], open: usize, mut i: usize, mut escaped: bool) -> Result<usize> {
    let quote = bytes[open];
    while i < bytes.len() {
        let b = bytes[i];
        if escaped {
            escaped = false;
        } else if b == b'\\' {
            escaped = true;
        } else if b == quote {
            return Ok(i + 1);
        }
        i += 1;
    }
    Err(Error::UnterminatedString { offset: open })
}

/// Position just past the comment whose `/` is at `slash`; mirrors the private
/// canonical `skip_comment` (line comments end after the newline or at end of
/// input; block comments end at the first `*/`, `/*/` stays unterminated).
#[cfg(target_arch = "x86_64")]
fn skip_comment(bytes: &[u8], slash: usize) -> Result<usize> {
    match bytes.get(slash + 1) {
        Some(b'/') => Ok(match bytes[slash + 2..].iter().position(|&b| b == b'\n') {
            Some(nl) => slash + 2 + nl + 1,
            None => bytes.len(),
        }),
        Some(b'*') => {
            let mut i = slash + 2;
            while i + 1 < bytes.len() {
                if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                    return Ok(i + 2);
                }
                i += 1;
            }
            Err(Error::UnterminatedComment { offset: slash })
        },
        _ => Err(Error::UnexpectedChar {
            byte: b'/',
            offset: slash,
        }),
    }
}

/// Scalar scan mirroring the canonical stage-1 main loop from byte `i`, with
/// an `in_token` carry so the SIMD kernels can hand off mid-token-run. When
/// `in_token` is false the loop body is the canonical decision-point loop.
#[allow(clippy::cast_possible_truncation)] // guarded by check_input_len
#[cfg(target_arch = "x86_64")]
fn scan_scalar(bytes: &[u8], mut i: usize, in_token: bool, index: &mut Vec<u32>) -> Result<()> {
    if in_token {
        // Finish the token run that the vector part started: consume it
        // without emitting (its start was already emitted).
        i = token_end(bytes, i);
    }
    while i < bytes.len() {
        match bytes[i] {
            b'{' | b'}' | b'[' | b']' | b',' | b':' => {
                index.push(i as u32);
                i += 1;
            },
            b'"' | b'\'' => {
                index.push(i as u32);
                i = skip_string(bytes, i)?;
            },
            b'/' => i = skip_comment(bytes, i)?,
            b'\t'..=b'\r' | b' ' => i += 1,
            0x80..=0xff => {
                // Scalar slow path: classify the whole sequence before
                // deciding. Structurally malformed input is rejected here.
                let (cp, len) = decode_seq(bytes, i).ok_or(Error::InvalidUtf8 { offset: i })?;
                if is_unicode_whitespace(cp) {
                    i += len;
                } else {
                    index.push(i as u32);
                    i = token_end(bytes, i);
                }
            },
            b if CLASSES[b as usize] & class::TOKEN != 0 => {
                index.push(i as u32);
                i = token_end(bytes, i);
            },
            b => return Err(Error::UnexpectedChar { byte: b, offset: i }),
        }
    }
    Ok(())
}

/// Dispatch entry for the parser: selects the best backend for the current
/// CPU.
///
/// - x86_64: SSE2 kernel (baseline) or the AVX2 kernel, chosen once per process through a cached
///   probe.
/// - other targets: the canonical scalar [`super::stage1::build_index`].
#[cfg(target_arch = "x86_64")]
pub(crate) fn build_index(bytes: &[u8], index: &mut Vec<u32>) -> Result<()> {
    check_input_len(bytes)?;
    x86::dispatch(bytes, index)
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn build_index(bytes: &[u8], index: &mut Vec<u32>) -> Result<()> {
    super::stage1::build_index(bytes, index)
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use core::arch::x86_64::{
        __m128i, __m256i, _mm_cmpeq_epi8, _mm_cmpgt_epi8, _mm_loadu_si128, _mm_movemask_epi8,
        _mm_set1_epi8, _mm_xor_si128, _mm256_cmpeq_epi8, _mm256_cmpgt_epi8, _mm256_loadu_si256,
        _mm256_movemask_epi8, _mm256_set1_epi8, _mm256_xor_si256,
    };
    use std::sync::OnceLock;

    use super::Prims;
    use crate::Result;

    /// SSE2 block primitives: four 16-byte lanes per 64-byte block. SSE2 is
    /// baseline on x86_64, so these need no runtime detection.
    #[derive(Clone, Copy)]
    pub(crate) struct Sse2;

    // Byte lanes are reinterpreted between u8/i8, and movemask results are
    // known to fit the low bits; both are lossless in practice.
    // inline(always) is load-bearing: it folds the primitives into the
    // #[target_feature] kernel that enables the ISA, so no generic copy is
    // ever compiled or called without the feature.
    #[allow(
        clippy::inline_always,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss
    )]
    impl Prims for Sse2 {
        type Block = [__m128i; 4];

        #[inline(always)]
        #[allow(clippy::cast_ptr_alignment)] // loadu needs no alignment
        unsafe fn load64(bytes: &[u8], i: usize) -> Self::Block {
            // SAFETY: the caller (scan loop) guarantees 64 readable bytes at
            // `i`; loadu imposes no alignment requirement.
            unsafe {
                let p = bytes.as_ptr().add(i).cast::<__m128i>();
                [
                    _mm_loadu_si128(p),
                    _mm_loadu_si128(p.add(1)),
                    _mm_loadu_si128(p.add(2)),
                    _mm_loadu_si128(p.add(3)),
                ]
            }
        }

        #[inline(always)]
        unsafe fn eq_mask(block: Self::Block, byte: u8) -> u64 {
            // SAFETY: register-only intrinsics; SSE2 is x86_64 baseline.
            unsafe {
                let needle = _mm_set1_epi8(byte as i8);
                let m = |v: __m128i| u64::from(_mm_movemask_epi8(_mm_cmpeq_epi8(v, needle)) as u32);
                m(block[0]) | (m(block[1]) << 16) | (m(block[2]) << 32) | (m(block[3]) << 48)
            }
        }

        #[inline(always)]
        unsafe fn le_mask(block: Self::Block, bound: u8) -> u64 {
            // SAFETY: register-only intrinsics; SSE2 is x86_64 baseline.
            unsafe {
                // SSE2 byte comparison is signed and core::arch exposes only
                // the strict forms, so unsigned <= is built from signed > on
                // sign-flipped operands plus a bitwise NOT: v <= b is
                // !(v > b).
                let flip = _mm_set1_epi8(-128);
                let needle = _mm_set1_epi8((bound ^ 0x80) as i8);
                let m = |v: __m128i| {
                    _mm_movemask_epi8(_mm_cmpgt_epi8(_mm_xor_si128(v, flip), needle)) as u32
                };
                let gt = u64::from(m(block[0]))
                    | (u64::from(m(block[1])) << 16)
                    | (u64::from(m(block[2])) << 32)
                    | (u64::from(m(block[3])) << 48);
                !gt
            }
        }

        #[inline(always)]
        unsafe fn token_mask(block: Self::Block) -> u64 {
            // SAFETY: register-only intrinsics; SSE2 is x86_64 baseline.
            unsafe {
                // Token set: [0-9] [A-Z] [a-z] $ _ + - . \ (see CLASSES).
                let dig = Self::le_mask(block, b'9') & !Self::le_mask(block, b'/');
                let upper = Self::le_mask(block, b'Z') & !Self::le_mask(block, b'@');
                let lower = Self::le_mask(block, b'z') & !Self::le_mask(block, b'`');
                let extra = Self::eq_mask(block, b'$')
                    | Self::eq_mask(block, b'_')
                    | Self::eq_mask(block, b'+')
                    | Self::eq_mask(block, b'-')
                    | Self::eq_mask(block, b'.')
                    | Self::eq_mask(block, b'\\');
                dig | upper | lower | extra
            }
        }

        #[inline(always)]
        unsafe fn structural_mask(block: Self::Block) -> u64 {
            // SAFETY: register-only intrinsics; SSE2 is x86_64 baseline.
            unsafe {
                // No range trick here: 0x5C ('') and 0x7C ('|') sit inside
                // the bracket ranges but are not structural.
                Self::eq_mask(block, b'{')
                    | Self::eq_mask(block, b'}')
                    | Self::eq_mask(block, b'[')
                    | Self::eq_mask(block, b']')
                    | Self::eq_mask(block, b',')
                    | Self::eq_mask(block, b':')
            }
        }
    }

    /// Append structural offsets for `bytes` to `index` (SSE2 kernel).
    ///
    /// # Safety
    ///
    /// No preconditions: SSE2 is part of the x86_64 baseline. The input must
    /// fit in `u32` offsets (see [`super::check_input_len`]).
    #[target_feature(enable = "sse2")]
    pub(crate) unsafe fn build_index_sse2(bytes: &[u8], index: &mut Vec<u32>) -> Result<()> {
        // SAFETY: SSE2 is the x86_64 baseline ISA.
        unsafe { super::scan_simd::<Sse2>(bytes, index) }
    }

    /// AVX2 block primitives: two 32-byte lanes per 64-byte block. Only
    /// reachable after a successful runtime AVX2 detection.
    #[derive(Clone, Copy)]
    pub(crate) struct Avx2;

    // See the Sse2 impl for the allow rationale.
    #[allow(
        clippy::inline_always,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss
    )]
    impl Prims for Avx2 {
        type Block = [__m256i; 2];

        #[inline(always)]
        #[allow(clippy::cast_ptr_alignment)] // loadu needs no alignment
        unsafe fn load64(bytes: &[u8], i: usize) -> Self::Block {
            // SAFETY: the caller (scan loop) guarantees 64 readable bytes at
            // `i`; loadu imposes no alignment requirement.
            unsafe {
                let p = bytes.as_ptr().add(i).cast::<__m256i>();
                [_mm256_loadu_si256(p), _mm256_loadu_si256(p.add(1))]
            }
        }

        #[inline(always)]
        unsafe fn eq_mask(block: Self::Block, byte: u8) -> u64 {
            // SAFETY: register-only intrinsics; the enabling kernel runs only
            // after a successful AVX2 detection.
            unsafe {
                let needle = _mm256_set1_epi8(byte as i8);
                let m = |v: __m256i| {
                    u64::from(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, needle)) as u32)
                };
                m(block[0]) | (m(block[1]) << 32)
            }
        }

        #[inline(always)]
        unsafe fn le_mask(block: Self::Block, bound: u8) -> u64 {
            // SAFETY: register-only intrinsics; the enabling kernel runs only
            // after a successful AVX2 detection.
            unsafe {
                // Same sign-flip trick as the SSE2 version: unsigned <= is
                // !(signed >) on flipped operands.
                let flip = _mm256_set1_epi8(-128);
                let needle = _mm256_set1_epi8((bound ^ 0x80) as i8);
                let m = |v: __m256i| {
                    _mm256_movemask_epi8(_mm256_cmpgt_epi8(_mm256_xor_si256(v, flip), needle))
                        as u32
                };
                let gt = u64::from(m(block[0])) | (u64::from(m(block[1])) << 32);
                !gt
            }
        }

        #[inline(always)]
        unsafe fn token_mask(block: Self::Block) -> u64 {
            // SAFETY: register-only intrinsics; the enabling kernel runs only
            // after a successful AVX2 detection.
            unsafe {
                // Same token set as the SSE2 impl (see CLASSES).
                let dig = Self::le_mask(block, b'9') & !Self::le_mask(block, b'/');
                let upper = Self::le_mask(block, b'Z') & !Self::le_mask(block, b'@');
                let lower = Self::le_mask(block, b'z') & !Self::le_mask(block, b'`');
                let extra = Self::eq_mask(block, b'$')
                    | Self::eq_mask(block, b'_')
                    | Self::eq_mask(block, b'+')
                    | Self::eq_mask(block, b'-')
                    | Self::eq_mask(block, b'.')
                    | Self::eq_mask(block, b'\\');
                dig | upper | lower | extra
            }
        }

        #[inline(always)]
        unsafe fn structural_mask(block: Self::Block) -> u64 {
            // SAFETY: register-only intrinsics; the enabling kernel runs only
            // after a successful AVX2 detection.
            unsafe {
                // Same byte set as the SSE2 impl (see CLASSES).
                Self::eq_mask(block, b'{')
                    | Self::eq_mask(block, b'}')
                    | Self::eq_mask(block, b'[')
                    | Self::eq_mask(block, b']')
                    | Self::eq_mask(block, b',')
                    | Self::eq_mask(block, b':')
            }
        }
    }

    /// Append structural offsets for `bytes` to `index` (AVX2 kernel).
    ///
    /// # Safety
    ///
    /// AVX2 must be available on the CPU; the cached dispatch below selects
    /// this kernel only after a successful detection. The input must fit in
    /// `u32` offsets (see [`super::check_input_len`]).
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn build_index_avx2(bytes: &[u8], index: &mut Vec<u32>) -> Result<()> {
        // SAFETY: the caller (dispatch) guarantees AVX2 support.
        unsafe { super::scan_simd::<Avx2>(bytes, index) }
    }

    /// Cached runtime dispatch. The CPU probe runs exactly once per process;
    /// a repeated `is_x86_feature_detected!` on the hot path would cost far
    /// more than the dispatch itself.
    type Kernel = unsafe fn(&[u8], &mut Vec<u32>) -> Result<()>;

    static KERNEL: OnceLock<Kernel> = OnceLock::new();

    pub(super) fn dispatch(bytes: &[u8], index: &mut Vec<u32>) -> Result<()> {
        let kernel = *KERNEL.get_or_init(|| {
            if std::is_x86_feature_detected!("avx2") {
                build_index_avx2
            } else {
                build_index_sse2
            }
        });
        // SAFETY: SSE2 is baseline on x86_64; the AVX2 kernel is selected
        // only after a successful feature detection.
        unsafe { kernel(bytes, index) }
    }
}

/// Bit i of the returned mask is set when byte i directly follows an
/// odd-length run of backslashes, i.e. when byte i is escaped. Ported from
/// simdjson's `find_odd_backslash_sequences` (carry chains seeded at run
/// starts with the parity of their position); `ends_odd` carries the pending
/// parity into the next 64-byte block.
#[cfg(target_arch = "x86_64")]
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

/// Vector block primitives for one 64-byte step of the scan. All methods are
/// `#[inline(always)]` so the generic scan folds into the
/// `#[target_feature]` kernel that enables the ISA.
#[cfg(target_arch = "x86_64")]
trait Prims {
    type Block: Copy;

    /// Load 64 bytes at `i`.
    ///
    /// # Safety
    ///
    /// `i + 64 <= bytes.len()` must hold, and the enabling ISA of the calling
    /// kernel must be available on the CPU.
    unsafe fn load64(bytes: &[u8], i: usize) -> Self::Block;

    /// Mask of bytes equal to `byte`.
    ///
    /// # Safety
    ///
    /// The enabling ISA of the calling kernel must be available.
    unsafe fn eq_mask(block: Self::Block, byte: u8) -> u64;

    /// Mask of bytes whose unsigned value is `<= bound`.
    ///
    /// # Safety
    ///
    /// The enabling ISA of the calling kernel must be available.
    unsafe fn le_mask(block: Self::Block, bound: u8) -> u64;

    /// Mask of stage-1 token bytes (`CLASSES[..] & class::TOKEN`): identifier
    /// characters, number characters and the backslash.
    ///
    /// # Safety
    ///
    /// The enabling ISA of the calling kernel must be available.
    unsafe fn token_mask(block: Self::Block) -> u64;

    /// Mask of structural bytes (`CLASSES[..] & class::STRUCTURAL`):
    /// `{` `}` `[` `]` `,` `:`.
    ///
    /// # Safety
    ///
    /// The enabling ISA of the calling kernel must be available.
    unsafe fn structural_mask(block: Self::Block) -> u64;
}

/// Vectorized string skip: returns the position just past the closing quote
/// of the string opened at `open`, or the canonical unterminated error.
///
/// # Safety
///
/// The enabling ISA of `O` must be available on the CPU.
#[cfg(target_arch = "x86_64")]
unsafe fn skip_string_simd<O: Prims>(bytes: &[u8], open: usize) -> Result<usize> {
    let quote = bytes[open];
    let len = bytes.len();
    let mut i = open + 1;
    let mut ends_odd = false;
    while i + 64 <= len {
        // SAFETY: loop guard bounds the load; ISA per the function contract.
        let block = unsafe { O::load64(bytes, i) };
        // SAFETY: register-only primitive ops.
        let bs = unsafe { O::eq_mask(block, b'\\') };
        let quotes = unsafe { O::eq_mask(block, quote) };
        // A quote closes the string only when not escaped.
        let closers = quotes & !odd_backslash_ends(bs, &mut ends_odd);
        if closers != 0 {
            return Ok(i + closers.trailing_zeros() as usize + 1);
        }
        i += 64;
    }
    // Scalar tail, carrying the pending escape parity.
    skip_string_tail(bytes, open, i, ends_odd)
}

/// Vectorized comment skip mirroring [`skip_comment`].
///
/// # Safety
///
/// The enabling ISA of `O` must be available on the CPU.
#[cfg(target_arch = "x86_64")]
unsafe fn skip_comment_simd<O: Prims>(bytes: &[u8], slash: usize) -> Result<usize> {
    match bytes.get(slash + 1) {
        // SAFETY: same ISA as the calling kernel.
        Some(b'/') => Ok(unsafe { skip_line_simd::<O>(bytes, slash + 2) }),
        // SAFETY: same ISA as the calling kernel.
        Some(b'*') => unsafe { skip_block_simd::<O>(bytes, slash) },
        _ => Err(Error::UnexpectedChar {
            byte: b'/',
            offset: slash,
        }),
    }
}

/// Line comment: position just past the next `\n`, or the end of input.
///
/// # Safety
///
/// The enabling ISA of `O` must be available on the CPU.
#[cfg(target_arch = "x86_64")]
unsafe fn skip_line_simd<O: Prims>(bytes: &[u8], from: usize) -> usize {
    let len = bytes.len();
    let mut i = from;
    while i + 64 <= len {
        // SAFETY: loop guard bounds the load; ISA per the function contract.
        let lf = unsafe { O::eq_mask(O::load64(bytes, i), b'\n') };
        if lf != 0 {
            return i + lf.trailing_zeros() as usize + 1;
        }
        i += 64;
    }
    match bytes[i..].iter().position(|&b| b == b'\n') {
        Some(nl) => i + nl + 1,
        None => len,
    }
}

/// Block comment: position just past the first `*/`, or the canonical
/// unterminated error.
///
/// # Safety
///
/// The enabling ISA of `O` must be available on the CPU.
#[cfg(target_arch = "x86_64")]
unsafe fn skip_block_simd<O: Prims>(bytes: &[u8], slash: usize) -> Result<usize> {
    let len = bytes.len();
    let mut i = slash + 2;
    // Set when byte `i - 1` is `*` (a `*/` may straddle a block boundary).
    let mut prev_star = false;
    while i + 64 <= len {
        // SAFETY: loop guard bounds the load; ISA per the function contract.
        let block = unsafe { O::load64(bytes, i) };
        // SAFETY: register-only primitive ops.
        let stars = unsafe { O::eq_mask(block, b'*') };
        let slashes = unsafe { O::eq_mask(block, b'/') };
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
    while i < len {
        if prev_star && bytes[i] == b'/' {
            return Ok(i + 1);
        }
        prev_star = bytes[i] == b'*';
        i += 1;
    }
    Err(Error::UnterminatedComment { offset: slash })
}

/// Generic 64-byte-block scan; the kernel entry points monomorphize this for
/// each ISA. Everything is `#[inline(always)]` down to the primitives, so the
/// whole body code-gens inside the `#[target_feature]` kernel.
///
/// # Safety
///
/// The enabling ISA of `O` must be available on the CPU, and the input must
/// fit in `u32` offsets (see [`check_input_len`]).
#[cfg(target_arch = "x86_64")]
#[allow(clippy::cast_possible_truncation)]
// guarded by check_input_len
// inline(always) is load-bearing here (see the Prims impls): the generic
// body must fold into the #[target_feature] kernel of each backend.
#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn scan_simd<O: Prims>(bytes: &[u8], index: &mut Vec<u32>) -> Result<()> {
    debug_assert!(u32::try_from(bytes.len()).is_ok());
    let len = bytes.len();
    let mut pos = 0usize;
    // True when the previous byte outside strings/comments was a token byte,
    // so the byte at `pos` continues a token run instead of starting one.
    let mut in_token = false;
    'blocks: while pos + 64 <= len {
        // SAFETY: loop guard guarantees 64 readable bytes at `pos`.
        let block = unsafe { O::load64(bytes, pos) };
        // SAFETY: register-only primitive ops. Whitespace is exactly
        // 0x09..=0x0D and 0x20.
        let ws = unsafe {
            (O::le_mask(block, 0x0d) & !O::le_mask(block, 0x08)) | O::eq_mask(block, b' ')
        };
        let mut attn = !ws; // attention bits: every non-whitespace byte
        let block_end = pos + 64;
        if attn != 0 {
            // SAFETY: register-only primitive ops.
            let special = unsafe { O::structural_mask(block) };
            let tok = unsafe { O::token_mask(block) };
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
                        // SAFETY: same ISA as this kernel.
                        unsafe { skip_string_simd::<O>(bytes, p) }?
                    },
                    b'/' => {
                        in_token = false;
                        // SAFETY: same ISA as this kernel.
                        unsafe { skip_comment_simd::<O>(bytes, p) }?
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
        pos = block_end;
    }
    // Finish the partial trailing block scalar, carrying the token-run state.
    scan_scalar(bytes, pos, in_token, index)
}

#[cfg(test)]
mod tests {
    // Hand-written expected offsets are tiny and cannot truncate.
    #![allow(clippy::cast_possible_truncation)]

    use super::*;
    use crate::parser::stage1;

    type KernelFn = unsafe fn(&[u8], &mut Vec<u32>) -> Result<()>;

    /// All kernels available on this machine, including the dispatch entry.
    fn kernels() -> Vec<(&'static str, KernelFn)> {
        #[cfg(target_arch = "x86_64")]
        {
            let mut kernels: Vec<(&'static str, KernelFn)> = vec![("sse2", x86::build_index_sse2)];
            if std::is_x86_feature_detected!("avx2") {
                kernels.push(("avx2", x86::build_index_avx2));
            }
            kernels.push(("dispatch", build_index));
            kernels
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            vec![("scalar", build_index)]
        }
    }

    /// The core differential: every kernel must agree with the canonical
    /// scalar stage 1 on both the result and the full index contents
    /// (including the partial contents left behind on error).
    fn differential(bytes: &[u8]) {
        let mut oracle = Vec::new();
        let expected = stage1::build_index(bytes, &mut oracle);
        for (label, kernel) in kernels() {
            let mut got = Vec::new();
            // SAFETY: x86_64 kernels are baseline-or-detected for this CPU.
            let result = unsafe { kernel(bytes, &mut got) };
            assert!(
                result == expected,
                "{label}: result {result:?} != oracle {expected:?} for {bytes:02x?}"
            );
            assert!(
                got == oracle,
                "{label}: index {got:?} != oracle {oracle:?} for {bytes:02x?}"
            );
        }
    }

    /// Dispatched index for a valid input.
    fn idx(bytes: &[u8]) -> Vec<u32> {
        let mut v = Vec::new();
        build_index(bytes, &mut v).unwrap();
        v
    }

    /// Dispatched error for an invalid input.
    fn err(bytes: &[u8]) -> Error {
        let mut v = Vec::new();
        build_index(bytes, &mut v).unwrap_err()
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
            "\t", "\n", "\r", "\x0b", "\x0c", " ", "\u{85}", "\u{a0}", "\u{1680}", "\u{2000}",
            "\u{200a}", "\u{2028}", "\u{2029}", "\u{202f}", "\u{205f}", "\u{3000}", "\u{feff}",
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
        assert!(build_index(b"a!", &mut got).is_err());
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
        // Lookalikes that are NOT whitespace: single token runs.
        for cp in [0x84u32, 0x200b, 0x200c, 0x200d, 0x2060, 0x2010, 0x180e] {
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
        let mut cps: Vec<u32> = vec![0x85, 0xa0, 0x1680];
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
            if stage1::build_index(&input, &mut probe).is_ok() {
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

    /// Exhaustive: every kernel's `token_mask` / `structural_mask` must agree
    /// with the scalar `CLASSES` byte table for all 256 byte values (4 blocks
    /// of 64). This is what catches lookalike bytes inside range tricks
    /// (0x5C inside `[`..`]`, 0x7C inside `{`..=`}`).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn masks_match_class_table() {
        let mut bytes = [0_u8; 256];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = i as u8;
        }
        for start in (0..256).step_by(64) {
            // SAFETY: SSE2 is the x86_64 baseline.
            let sse_block = unsafe { x86::Sse2::load64(&bytes, start) };
            // SAFETY: as above.
            let mut masks = vec![(
                "sse2",
                unsafe { x86::Sse2::token_mask(sse_block) },
                unsafe { x86::Sse2::structural_mask(sse_block) },
            )];
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: guarded by the feature detection above.
                let block = unsafe { x86::Avx2::load64(&bytes, start) };
                masks.push(("avx2", unsafe { x86::Avx2::token_mask(block) }, unsafe {
                    x86::Avx2::structural_mask(block)
                }));
            }
            for (label, tok, special) in masks {
                for i in 0..64 {
                    let b = start + i;
                    assert_eq!(
                        tok & (1 << i) != 0,
                        CLASSES[b] & class::TOKEN != 0,
                        "{label} token: byte {b:#04x}"
                    );
                    assert_eq!(
                        special & (1 << i) != 0,
                        CLASSES[b] & class::STRUCTURAL != 0,
                        "{label} structural: byte {b:#04x}"
                    );
                }
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn kernel_availability_report() {
        // Informational: which kernels this machine verifies. Run with
        // `--no-capture` to see it.
        let avx2 = std::is_x86_feature_detected!("avx2");
        println!("avx2 detected: {avx2}");
        assert!(kernels().iter().any(|(label, _)| *label == "sse2"));
    }
}
