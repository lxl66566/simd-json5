//! Number parsing for the `[+-]?(Infinity|NaN|Hex|Decimal)`.
//!
//! Decimal forms follow ECMAScript 5.1: `5.`, `.5`, `1.e3`, `+1` and `007`
//! are valid; `.e3`, `1e` and `0x` are not. Integers that fit u64 (positive)
//! or i64 (negative) stay integer variants; anything else becomes an exactly
//! converted f64. Float evaluation is delegated to `fast_float2`; only sign
//! handling, grammar validation and integer accumulation are done here.
//!
//! Hex integers wider than u128 (33 or more hex digits) accumulate in f64
//! step by step (`f * 16.0 + digit`); the stepwise rounding may differ from
//! a correctly rounded conversion by 1 ULP. Inputs that large are out of
//! scope.
//!
//! The serde i128/u128 and f32 visitors get dedicated entries:
//! [`parse_num_wide`] keeps 128-bit integer literals exact (rejecting those
//! past the 128-bit range) and [`parse_num_f32`] evaluates floats at f32
//! precision in a single rounding. The DOM and `deserialize_any` paths use
//! [`parse_num`]/[`Num`] unchanged.

use crate::{Error, Result, value::Value};

/// Numeric value of a hex digit, 0..=15 (shared with `\xHH` / `\uHHHH`
/// escapes).
#[must_use]
pub(crate) fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// A parsed number split by variant, for callers that dispatch on the
/// integer/float kind without building a DOM [`Value`] (the serde fast path).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Num {
    I64(i64),
    U64(u64),
    F64(f64),
}

/// A parsed number for the serde i128/u128 visitors: the [`Num`] variants
/// widened to 128 bits, with integer literals held exactly however long they
/// are (within the 128-bit range).
#[cfg_attr(not(feature = "serde"), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Num128 {
    I128(i128),
    U128(u128),
    F64(f64),
}

/// A parsed number for the serde f32 visitor: the integer hints of [`Num`]
/// with float values evaluated directly at f32 precision.
#[cfg_attr(not(feature = "serde"), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum NumF32 {
    I64(i64),
    U64(u64),
    F32(f32),
}

/// First bytes a number token may start with: exactly the byte set
/// `stage2::token_value` routes to [`parse_num`]. No bare identifier (`null`,
/// `true`, keywords) can begin with one, so the serde fast path uses this to
/// pick [`parse_num`] without the generic token classification.
#[cfg_attr(not(feature = "serde"), allow(dead_code))]
#[must_use]
pub(crate) fn is_num_start(b: u8) -> bool {
    matches!(b, b'+' | b'-' | b'.' | b'0'..=b'9')
}

/// Parse a bare number token. `token` is the full maximal token run reported
/// by stage 1; `offset` is its first byte position in the input, used for
/// errors.
pub(crate) fn parse_num(token: &[u8], offset: usize) -> Result<Num> {
    let (neg, rest) = match token.first() {
        Some(&b'+') => (false, &token[1..]),
        Some(&b'-') => (true, &token[1..]),
        _ => (false, token),
    };
    if rest == b"Infinity" {
        return Ok(Num::F64(if neg {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        }));
    }
    if rest == b"NaN" {
        return Ok(Num::F64(f64::NAN));
    }
    if matches!(rest.first(), Some(&b'0') if matches!(rest.get(1), Some(&(b'x' | b'X')))) {
        return parse_hex(&rest[2..], neg, offset);
    }
    parse_decimal(rest, neg, offset)
}

/// 2^127, the i128 negation boundary (the 128-bit counterpart of parse_hex's
/// and parse_decimal's 2^63 boundaries).
const TWO_POW_127: u128 = 1 << 127;

/// Exact u128 accumulation of digit bytes in `base`, `None` on overflow past
/// u128. Each byte is mapped through `val`; a `None` there aborts too, so
/// callers pair the fold with their digit classifier.
fn accumulate(digits: &[u8], base: u128, val: impl Fn(u8) -> Option<u128>) -> Option<u128> {
    digits.iter().try_fold(0_u128, |acc, &d| {
        acc.checked_mul(base)?.checked_add(val(d)?)
    })
}

/// Sign tail of the wide integer parses. `acc == 2^127` wraps to i128::MIN,
/// whose negation is itself (the same convention as the 64-bit paths); a
/// negative magnitude past 2^127 leaves the 128-bit range and is invalid.
#[allow(clippy::cast_possible_wrap)] // acc <= 2^127, so `as i128` only "wraps" 2^127 to i128::MIN
fn wide_signed(acc: u128, neg: bool, offset: usize) -> Result<Num128> {
    if !neg {
        return Ok(Num128::U128(acc));
    }
    if acc > TWO_POW_127 {
        return Err(invalid_number(offset));
    }
    Ok(Num128::I128((acc as i128).wrapping_neg()))
}

/// Parse a bare number token for the serde i128/u128 visitors, exact across
/// the whole 128-bit range. A maximal token of plain decimal digits (or a
/// `0x` run of hex digits) is a valid integer literal by construction, so it
/// accumulates directly at u128 width instead of overflowing [`Num`]'s u64
/// accumulator into f64; literals past u128::MAX / i128::MIN are
/// [`Error::InvalidNumber`]. Everything else — floats, keywords, malformed
/// tokens — keeps [`parse_num`]'s grammar and F64 value, which the integer
/// visitors reject as a type error.
#[cfg_attr(not(feature = "serde"), allow(dead_code))]
pub(crate) fn parse_num_wide(token: &[u8], offset: usize) -> Result<Num128> {
    let (neg, rest) = match token.first() {
        Some(&b'+') => (false, &token[1..]),
        Some(&b'-') => (true, &token[1..]),
        _ => (false, token),
    };
    if let Some(&first) = rest.first()
        && first.is_ascii_digit()
        && rest[1..].iter().all(u8::is_ascii_digit)
    {
        let acc = accumulate(rest, 10, |d| Some(u128::from(d - b'0')))
            .ok_or_else(|| invalid_number(offset))?;
        return wide_signed(acc, neg, offset);
    }
    if matches!(rest.first(), Some(&b'0') if matches!(rest.get(1), Some(&(b'x' | b'X')))) {
        let digits = &rest[2..];
        if !digits.is_empty() && digits.iter().all(|d| hex_val(*d).is_some()) {
            let acc = accumulate(digits, 16, |d| hex_val(d).map(u128::from))
                .ok_or_else(|| invalid_number(offset))?;
            return wide_signed(acc, neg, offset);
        }
    }
    Ok(match parse_num(token, offset)? {
        Num::I64(i) => Num128::I128(i128::from(i)),
        Num::U64(u) => Num128::U128(u128::from(u)),
        Num::F64(f) => Num128::F64(f),
    })
}

/// Parse a bare number token for the serde f32 visitor. Integer hints match
/// [`parse_num`]; float spellings are evaluated directly at f32 precision —
/// one decimal rounding, instead of the decimal→f64→f32 double rounding the
/// f64 delegation performs (they differ by 1 ULP on f32 tie boundaries).
#[cfg_attr(not(feature = "serde"), allow(dead_code))]
pub(crate) fn parse_num_f32(token: &[u8], offset: usize) -> Result<NumF32> {
    Ok(match parse_num(token, offset)? {
        Num::I64(i) => NumF32::I64(i),
        Num::U64(u) => NumF32::U64(u),
        Num::F64(f) => NumF32::F32(f32_value(token, f)),
    })
}

/// Direct f32 evaluation of a grammar-validated token whose [`parse_num`]
/// value is `f64_value`. fast_float2 accepts every decimal spelling of the
/// grammar (elided forms, signs, the `Infinity`/`NaN` keywords); only hex
/// literals overflow into `Num::F64`, and those narrow their f64 value —
/// hex has no decimal spelling to re-round.
#[allow(
    clippy::cast_precision_loss, // f64 -> f32 narrowing is the intent for hex overflow
    clippy::cast_possible_truncation
)]
fn f32_value(token: &[u8], f64_value: f64) -> f32 {
    fast_float2::parse::<f32, _>(token).unwrap_or(f64_value as f32)
}

/// DOM entry: the grammar in [`parse_num`] mapped onto [`Value`] variants.
pub(crate) fn parse(token: &[u8], offset: usize) -> Result<Value<'static>> {
    Ok(match parse_num(token, offset)? {
        Num::I64(i) => Value::I64(i),
        Num::U64(u) => Value::U64(u),
        Num::F64(f) => Value::F64(f),
    })
}

/// `Error::InvalidNumber` constructor. Cold: keeps the error paths out of
/// the hot digit loops' instruction stream.
#[cold]
fn invalid_number(offset: usize) -> Error {
    Error::InvalidNumber { offset }
}

/// Hexadecimal integer: `0x` has already been consumed.
#[allow(
    clippy::cast_precision_loss, // u128 -> f64 widening is the documented overflow path
    clippy::cast_possible_truncation, // in-range by the adjacent bound checks
    clippy::cast_possible_wrap // acc <= 2^63, so `as i64` only "wraps" 2^63 to i64::MIN
)]
fn parse_hex(digits: &[u8], neg: bool, offset: usize) -> Result<Num> {
    const TWO_POW_63: u128 = 1 << 63;
    const U64_MAX: u128 = u64::MAX as u128;
    if digits.is_empty() {
        return Err(invalid_number(offset));
    }
    let mut acc: u128 = 0;
    // Past u128 range the value keeps accumulating in f64; beyond 2^128 the
    // stepwise rounding may differ from a correctly rounded conversion by an
    // ULP. Inputs with more than 32 hex digits are out of scope.
    let mut wide: Option<f64> = None;
    for &d in digits {
        let Some(v) = hex_val(d) else {
            return Err(invalid_number(offset));
        };
        if let Some(a) = acc
            .checked_mul(16)
            .and_then(|a| a.checked_add(u128::from(v)))
        {
            acc = a;
        } else {
            let f = wide.get_or_insert(acc as f64);
            *f = *f * 16.0 + f64::from(v);
        }
    }
    if let Some(f) = wide {
        return Ok(Num::F64(if neg {
            -f
        } else {
            f
        }));
    }
    if neg {
        if acc <= TWO_POW_63 {
            // acc == 2^63 wraps to i64::MIN, whose negation is itself.
            return Ok(Num::I64((acc as i64).wrapping_neg()));
        }
    } else if acc <= U64_MAX {
        return Ok(Num::U64(acc as u64));
    }
    let f = acc as f64;
    Ok(Num::F64(if neg {
        -f
    } else {
        f
    }))
}

/// 10^n for the SWAR digit batches, indexed by the batch digit count 1..=8.
const POW10: [u64; 9] = [
    1,
    10,
    100,
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
    100_000_000,
];

/// Value and length of the leading ASCII-digit run in an 8-byte word, SWAR
/// style (no per-byte branch). A count of 0 means the first byte is not a
/// digit. Borrow and carry inside the two wrapping word ops can only corrupt
/// bytes at or above the first non-digit byte, which the run never reaches,
/// so the flagged positions are exact. The word is read little-endian over
/// the input slice (see `parse_decimal`), so `from_le_bytes` semantics hold
/// on every target.
#[inline]
fn leading_digits(word: u64) -> (u64, usize) {
    let non_digit = (word.wrapping_add(0x4646_4646_4646_4646)
        | word.wrapping_sub(0x3030_3030_3030_3030))
        & 0x8080_8080_8080_8080;
    if non_digit == 0 {
        return (u64::from(parse_eight_digits(word)), 8);
    }
    let n = (non_digit.trailing_zeros() / 8) as usize;
    if n == 0 {
        return (0, 0);
    }
    // Mask off the bytes past the run, left-justify the digits and combine
    // them with the same shift-and-add chain as `parse_eight_digits`.
    let mut d = word.wrapping_sub(0x3030_3030_3030_3030);
    d &= (1_u64 << (n * 8)) - 1;
    d <<= (8 - n) * 8;
    d = (d.wrapping_mul(10).wrapping_add(d >> 8)) & 0x00ff_00ff_00ff_00ff;
    d = (d.wrapping_mul(100).wrapping_add(d >> 16)) & 0x0000_ffff_0000_ffff;
    d = d.wrapping_mul(10_000).wrapping_add(d >> 32);
    // In range by construction: the chain ends with the 1..=8 digit value in
    // the low 32 bits (at most 99,999,999).
    #[allow(clippy::cast_possible_truncation)]
    (u64::from(d as u32), n)
}

/// Value of exactly 8 ASCII digits in one u64 word, shift-and-add SWAR
/// (the simdjson `parse_eight_digits_unrolled` chain).
#[inline]
#[allow(clippy::cast_possible_truncation)] // eight digits are at most 99,999,999
fn parse_eight_digits(word: u64) -> u32 {
    let mut val = word.wrapping_sub(0x3030_3030_3030_3030);
    val = (val.wrapping_mul(10).wrapping_add(val >> 8)) & 0x00ff_00ff_00ff_00ff;
    val = (val.wrapping_mul(100).wrapping_add(val >> 16)) & 0x0000_ffff_0000_ffff;
    val.wrapping_mul(10_000).wrapping_add(val >> 32) as u32
}

/// Unsigned decimal per ECMAScript 5.1, sign already stripped. One
/// validating pass that also accumulates the integer digits; the frac/exp
/// layout is read off the stop positions afterwards.
fn parse_decimal(s: &[u8], neg: bool, offset: usize) -> Result<Num> {
    // Integer part: consume and accumulate in one pass. SWAR batches of
    // eight digits while at least eight bytes remain, a byte-wise tail for
    // the final stretch. A partial batch (count < 8) hit a non-digit byte,
    // so the integer part ends there; on overflow keep consuming
    // (validation continues) and fall back to lib_parse below.
    let mut i = 0;
    let mut acc: u64 = 0;
    let mut overflow = false;
    loop {
        if s.len() - i >= 8 {
            let word = u64::from_le_bytes(s[i..i + 8].try_into().unwrap());
            let (v, count) = leading_digits(word);
            if count == 0 {
                break;
            }
            if !overflow {
                match acc.checked_mul(POW10[count]).and_then(|a| a.checked_add(v)) {
                    Some(a) => acc = a,
                    None => overflow = true,
                }
            }
            i += count;
            if count < 8 {
                break;
            }
        } else if i < s.len() && s[i].is_ascii_digit() {
            if !overflow {
                let v = u64::from(s[i] - b'0');
                if let Some(a) = acc.checked_mul(10).and_then(|a| a.checked_add(v)) {
                    acc = a;
                } else {
                    overflow = true;
                }
            }
            i += 1;
        } else {
            break;
        }
    }
    let int_digits = i;
    let mut frac_digits = 0;
    let mut had_dot = false;
    if s.get(i) == Some(&b'.') {
        had_dot = true;
        i += 1;
        let from = i;
        // Fraction digits are only counted here: long mantissas (canada's
        // 17-18 significant digits) can never serve the fast path below, so
        // the significant is accumulated in a second pass only when the
        // digit count leaves the fast path reachable.
        while i < s.len() && s[i].is_ascii_digit() {
            i += 1;
        }
        frac_digits = i - from;
        // `5.` and `.5` are valid; a lone `.` (or `.e3`) is not.
        if int_digits + frac_digits == 0 {
            return Err(invalid_number(offset));
        }
    } else if int_digits == 0 {
        // No digits and no dot: `e3`, empty, stray sign.
        return Err(invalid_number(offset));
    }
    let mut had_exp = false;
    let mut exp_value: i64 = 0;
    let mut exp_neg = false;
    if matches!(s.get(i), Some(&(b'e' | b'E'))) {
        i += 1;
        if matches!(s.get(i), Some(&(b'+' | b'-'))) {
            exp_neg = s[i] == b'-';
            i += 1;
        }
        // `1e` and `1e+` are rejected. The value saturates: any exponent
        // outside the fast-path window falls back to lib_parse, which
        // evaluates it exactly.
        let from = i;
        while i < s.len() && s[i].is_ascii_digit() {
            let d = i64::from(s[i] - b'0');
            exp_value = exp_value.saturating_mul(10).saturating_add(d);
            i += 1;
        }
        if i == from {
            return Err(invalid_number(offset));
        }
        if exp_neg {
            exp_value = -exp_value;
        }
        had_exp = true;
    }
    // The token is a maximal run, so any residue (`1a`, `1.2.3`) is invalid.
    if i != s.len() {
        return Err(invalid_number(offset));
    }
    if had_dot || had_exp {
        // Fast path: at most 16 significant digit positions (a conservative
        // bound: leading zeros are counted, and a 17th digit always exceeds
        // 2^53) and no accumulation overflow in the integer part. The
        // significant is accumulated now — never for mantissas that long —
        // and both `significant as f64` and every power of ten up to 10^22
        // are exact in f64 (5^22 still fits the 53-bit mantissa), so the
        // single multiply or divide is correctly rounded to the decimal
        // value.
        if !overflow
            && int_digits.saturating_add(frac_digits) <= 16
            && let Ok(frac) = i64::try_from(frac_digits)
            // The exponent may saturate at the i64 edge; the subtraction
            // itself must not overflow (a saturated exponent is far outside
            // the window, so it just falls back).
            && let Some(dec_exp) = exp_value.checked_sub(frac)
            && (-22..=22).contains(&dec_exp)
        {
            // Second pass over the mantissa (the integer and fraction
            // digits, dot skipped): at most 16 digits, so the accumulation
            // cannot overflow.
            let mantissa_end = int_digits + usize::from(had_dot) + frac_digits;
            let mut significant: u64 = 0;
            for &b in &s[..mantissa_end] {
                if b.is_ascii_digit() {
                    significant = significant * 10 + u64::from(b - b'0');
                }
            }
            if significant <= MAX_SIGNIFICANT_FAST {
                return Ok(Num::F64(signed(neg, pow10_fast(significant, dec_exp))));
            }
        }
        // Fraction layout facts come straight from the validation pass; no
        // second scan for the dot position.
        let leading_dot = int_digits == 0;
        let elided_frac = had_dot && frac_digits == 0;
        return Ok(Num::F64(signed(
            neg,
            float_value(s, leading_dot, elided_frac, offset)?,
        )));
    }
    // Pure integer: keep the integer variant while the magnitude fits.
    if overflow {
        let f = lib_parse(s, offset)?;
        return Ok(Num::F64(signed(neg, f)));
    }
    if !neg {
        return Ok(Num::U64(acc));
    }
    // -(acc) fits i64 iff acc <= 2^63; acc == 2^63 wraps to i64::MIN, whose
    // negation is itself.
    #[allow(clippy::cast_possible_wrap)]
    if acc <= (i64::MAX as u64) + 1 {
        return Ok(Num::I64((acc as i64).wrapping_neg()));
    }
    let f = lib_parse(s, offset)?;
    Ok(Num::F64(signed(neg, f)))
}

fn signed(neg: bool, f: f64) -> f64 {
    if neg {
        -f
    } else {
        f
    }
}

/// Largest mantissa the fast float path accepts: `acc as f64` is exact up to
/// and including 2^53.
const MAX_SIGNIFICANT_FAST: u64 = 1 << 53;

/// Powers of ten, each exactly representable in f64 (5^22 still fits the
/// 53-bit mantissa, 5^23 does not), indexed by the exponent 0..=22.
const POW10_F64: [f64; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16,
    1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
];

/// `significant * 10^exp` for the fast path: `significant <= 2^53` and
/// `|exp| <= 22`, so both operands are exact in f64 and the single multiply
/// or divide (IEEE-correctly rounded on their exact product or quotient,
/// which is the decimal value itself) yields the correctly rounded result.
#[allow(clippy::cast_precision_loss)] // exact: significant <= 2^53
fn pow10_fast(significant: u64, exp: i64) -> f64 {
    let d = significant as f64;
    // In range by the caller's contract: |exp| <= 22.
    let idx = usize::try_from(exp.unsigned_abs()).expect("exp in [-22, 22]");
    if exp >= 0 {
        d * POW10_F64[idx]
    } else {
        d / POW10_F64[idx]
    }
}

/// Evaluate an unsigned, grammar-validated decimal as f64. `leading_dot`
/// marks `.5` forms and `elided_frac` marks `5.` / `1.e3` forms; both are
/// normalized (to `0.5` / `5.0`) because C `from_chars` semantics are not
/// guaranteed for elided digit runs by every backend. Both flags come from
/// the grammar pass, so no byte is scanned twice. A parse failure of both
/// float backends surfaces as `Error::InvalidNumber` at `offset`.
fn float_value(s: &[u8], leading_dot: bool, elided_frac: bool, offset: usize) -> Result<f64> {
    if !leading_dot && !elided_frac {
        return lib_parse(s, offset);
    }
    // The normalized form inserts one byte (`0`); at most one of the two
    // flags can be set (`.` alone is rejected by the grammar). Legal tokens
    // fit the 64-byte stack buffer; only overlong exponents spill to the
    // heap.
    debug_assert!(!(leading_dot && elided_frac));
    if s.len() < 64 {
        let mut buf = [0_u8; 64];
        let mut n = 0;
        if leading_dot {
            buf[n] = b'0';
            n += 1;
        }
        if elided_frac {
            // A single dot with digits before it and none after: `5.` / `1.e3`.
            let dot = s.iter().position(|&b| b == b'.').unwrap_or(0);
            buf[n..=n + dot].copy_from_slice(&s[..=dot]);
            n += dot + 1;
            buf[n] = b'0';
            n += 1;
            buf[n..n + s.len() - dot - 1].copy_from_slice(&s[dot + 1..]);
            n += s.len() - dot - 1;
        } else {
            buf[n..n + s.len()].copy_from_slice(s);
            n += s.len();
        }
        lib_parse(&buf[..n], offset)
    } else {
        let mut buf = Vec::with_capacity(s.len() + 1);
        if leading_dot {
            buf.push(b'0');
        }
        if elided_frac {
            let dot = s.iter().position(|&b| b == b'.').unwrap_or(0);
            buf.extend_from_slice(&s[..=dot]);
            buf.push(b'0');
            buf.extend_from_slice(&s[dot + 1..]);
        } else {
            buf.extend_from_slice(s);
        }
        lib_parse(&buf, offset)
    }
}

fn lib_parse(s: &[u8], offset: usize) -> Result<f64> {
    if let Ok(f) = fast_float2::parse::<f64, _>(s) {
        return Ok(f);
    }
    // Unreachable for grammar-validated input (overflow reports Ok(inf),
    // underflow Ok(0)); the std layer is a second chance, and a failure of
    // both is reported instead of silently corrupting the value.
    core::str::from_utf8(s)
        .ok()
        .and_then(|t| t.parse::<f64>().ok())
        .ok_or_else(|| invalid_number(offset))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn num(t: &str) -> Value<'static> {
        parse(t.as_bytes(), 0).unwrap()
    }

    fn err(t: &str) -> Error {
        parse(t.as_bytes(), 0).unwrap_err()
    }

    #[test]
    fn swar_batch_boundaries() {
        // One batch.
        assert_eq!(num("99999999"), Value::U64(99_999_999));
        assert_eq!(num("10000000"), Value::U64(10_000_000));
        // Batch plus tail digits.
        assert_eq!(num("123456789"), Value::U64(123_456_789));
        assert_eq!(num("123456789012345"), Value::U64(123_456_789_012_345));
        // Two full batches.
        assert_eq!(num("1234567890123456"), Value::U64(1_234_567_890_123_456));
        // Two batches plus tail, filling u64 exactly and overflowing past it.
        assert_eq!(num("18446744073709551615"), Value::U64(u64::MAX));
        assert_eq!(
            num("18446744073709551616"),
            Value::F64(1.844_674_407_370_955_2e19)
        );
        // Non-digit inside a scanned word ends the run at the word.
        assert_eq!(num("1234567.5"), Value::F64(1_234_567.5));
        assert_eq!(num("12345678.5"), Value::F64(12_345_678.5));
        assert_eq!(num("12345678.25"), Value::F64(12_345_678.25));
        // A digit run stopped by an exponent byte mid-word.
        assert_eq!(num("1234567e3"), Value::F64(1_234_567e3));
    }

    /// The SWAR batch accumulation must agree with a plain digit-by-digit
    /// accumulation for every length around the 8- and 16-byte batch
    /// boundaries, including values that overflow u64.
    #[test]
    #[allow(clippy::float_cmp)] // exactness against std parsing is the point
    fn swar_accumulation_matches_naive() {
        let naive = |t: &str| -> Option<u64> {
            let mut acc: u64 = 0;
            for b in t.bytes() {
                let (a, o1) = acc.overflowing_mul(10);
                let (a, o2) = a.overflowing_add(u64::from(b - b'0'));
                if o1 || o2 {
                    return None;
                }
                acc = a;
            }
            Some(acc)
        };
        for len in 1..=22 {
            let t: String = (0..len)
                .map(|k| char::from(b'0' + u8::try_from(k % 10).unwrap()))
                .collect();
            let expect = naive(&t);
            let got = num(&t);
            match expect {
                Some(v) => assert_eq!(got, Value::U64(v), "len {len}"),
                None => assert!(
                    matches!(got, Value::F64(f) if f == t.parse::<f64>().unwrap()),
                    "len {len}"
                ),
            }
            // The same digits followed by a fraction: the integer run stops
            // at the dot, at every alignment against the 8-byte words.
            let split = format!("{t}.5");
            assert_eq!(
                num(&split),
                Value::F64(t.parse::<f64>().unwrap() + 0.5),
                "len {len}"
            );
        }
    }

    /// The power-of-ten fast path must agree with std's correctly rounded
    /// parse across its whole window: every exponent in [-22, 22] on both
    /// the dot and the `e` spelling, mantissas around 2^53, leading and
    /// trailing zeros, and the fallback just outside the window.
    #[test]
    #[allow(clippy::float_cmp)] // exactness against std parsing is the point
    fn pow10_fast_path_matches_std() {
        let mantissas = [
            1_u64,
            2,
            5,
            9,
            12345,
            999_999_999,
            9_007_199_254_740_991,
            1 << 53,
        ];
        for &m in &mantissas {
            for exp in -22_i64..=22 {
                let token = format!("{m}e{exp}");
                let expect: f64 = token.parse().unwrap();
                assert_eq!(num(&token), Value::F64(expect), "{token}");
                // The same value with the dot shifted instead of an exponent.
                let digits = m.to_string();
                let shift = usize::try_from(exp.unsigned_abs()).unwrap();
                let dot = if exp >= 0 {
                    format!("{digits}{}.", "0".repeat(shift))
                } else {
                    let frac = shift;
                    if frac >= digits.len() {
                        format!("0.{}{digits}", "0".repeat(frac - digits.len()))
                    } else {
                        let at = digits.len() - frac;
                        format!("{}.{}", &digits[..at], &digits[at..])
                    }
                };
                assert_eq!(num(&dot), Value::F64(expect), "{dot}");
            }
        }
        // Just outside the window: mantissa and exponent fallbacks, and the
        // saturated exponent parse.
        assert_eq!(num("1e23"), Value::F64(1e23));
        assert_eq!(num("1e-23"), Value::F64(1e-23));
        assert_eq!(
            num("9007199254740993.0"),
            Value::F64(9_007_199_254_740_992.0)
        );
        assert_eq!(num("1e99999999999999999999"), Value::F64(f64::INFINITY));
        assert_eq!(num("1e-99999999999999999999"), Value::F64(0.0));
        // A saturated exponent combined with a fraction digit must not
        // overflow the decimal-exponent subtraction (found by the parse
        // fuzz targets).
        assert_eq!(num("1.5e99999999999999999999"), Value::F64(f64::INFINITY));
        assert_eq!(num("1.5e-99999999999999999999"), Value::F64(0.0));
        assert_eq!(num("-1.5e-99999999999999999999"), Value::F64(-0.0));
        assert_eq!(
            num("1.7976931348623157e-32007199252007199254740993"),
            Value::F64(
                "1.7976931348623157e-32007199252007199254740993"
                    .parse::<f64>()
                    .unwrap()
            )
        );
    }

    #[test]
    fn plain_integers() {
        assert_eq!(num("0"), Value::U64(0));
        assert_eq!(num("1"), Value::U64(1));
        assert_eq!(num("+1"), Value::U64(1));
        assert_eq!(num("-1"), Value::I64(-1));
        assert_eq!(num("-0"), Value::I64(0));
        assert_eq!(num("007"), Value::U64(7));
        assert_eq!(num("18446744073709551615"), Value::U64(u64::MAX));
        assert_eq!(num("-9223372036854775808"), Value::I64(i64::MIN));
    }

    #[test]
    fn integer_overflow_becomes_exact_f64() {
        assert_eq!(
            num("18446744073709551616"),
            Value::F64(1.844_674_407_370_955_2e19)
        );
        assert_eq!(
            num("-9223372036854775809"),
            Value::F64(-9.223_372_036_854_776e18)
        );
    }

    #[test]
    fn hex_integers() {
        assert_eq!(num("0x0"), Value::U64(0));
        assert_eq!(num("0x10"), Value::U64(16));
        assert_eq!(num("0X1f"), Value::U64(31));
        assert_eq!(num("0xaB"), Value::U64(0xab));
        assert_eq!(num("-0x10"), Value::I64(-16));
        assert_eq!(num("+0x10"), Value::U64(16));
        assert_eq!(num("0xFFFFFFFFFFFFFFFF"), Value::U64(u64::MAX));
        assert_eq!(num("-0x8000000000000000"), Value::I64(i64::MIN));
        assert_eq!(
            num("0x1FFFFFFFFFFFFFFFF"),
            Value::F64(3.689_348_814_741_910_3e19)
        );
        assert_eq!(
            num("-0x8000000000000001"),
            Value::F64(-9.223_372_036_854_776e18)
        );
    }

    /// u128 -> f64 is correctly rounded; the lint cannot prove it.
    #[allow(clippy::cast_precision_loss)]
    fn u128_to_f64(v: u128) -> f64 {
        v as f64
    }

    #[test]
    fn hex_overflow_to_f64_is_exact() {
        // u128::MAX needs 32 hex digits and converts exactly to one f64.
        let t = "0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF";
        assert_eq!(num(t), Value::F64(u128_to_f64(u128::MAX)));
    }

    #[test]
    fn special_values() {
        assert_eq!(num("Infinity"), Value::F64(f64::INFINITY));
        assert_eq!(num("+Infinity"), Value::F64(f64::INFINITY));
        assert_eq!(num("-Infinity"), Value::F64(f64::NEG_INFINITY));
        assert!(matches!(num("NaN"), Value::F64(f) if f.is_nan()));
        assert!(matches!(num("-NaN"), Value::F64(f) if f.is_nan()));
        assert!(matches!(num("+NaN"), Value::F64(f) if f.is_nan()));
    }

    #[test]
    fn decimal_floats() {
        assert_eq!(num(".5"), Value::F64(0.5));
        assert_eq!(num("-.5"), Value::F64(-0.5));
        assert_eq!(num("5."), Value::F64(5.0));
        assert_eq!(num("1.e3"), Value::F64(1000.0));
        assert_eq!(num("1.5"), Value::F64(1.5));
        assert_eq!(num("1e3"), Value::F64(1000.0));
        assert_eq!(num("1E3"), Value::F64(1000.0));
        assert_eq!(num("1e+3"), Value::F64(1000.0));
        assert_eq!(num("1e-3"), Value::F64(0.001));
        assert_eq!(num("0.1"), Value::F64(0.1));
        assert_eq!(num("1.7976931348623157e308"), Value::F64(f64::MAX));
        // 2^53 + 1 is midway between two doubles and ties to even.
        assert_eq!(
            num("9007199254740993.0"),
            Value::F64(9_007_199_254_740_992.0)
        );
        // Leading-dot forms around the 64-byte stack-buffer boundary of
        // float_value (one below, one at it) both normalize identically.
        assert_eq!(num(&format!(".5e{}", "0".repeat(60))), Value::F64(0.5));
        assert_eq!(num(&format!(".5e{}", "0".repeat(61))), Value::F64(0.5));
    }

    #[test]
    fn float_range_edges() {
        assert_eq!(num("1e309"), Value::F64(f64::INFINITY));
        assert_eq!(num("-1e309"), Value::F64(f64::NEG_INFINITY));
        assert_eq!(num("1e-400"), Value::F64(0.0));
        assert_eq!(
            num("5e-324"),
            Value::F64(f64::from_bits(1)) // smallest subnormal
        );
    }

    #[test]
    fn invalid_numbers() {
        for t in [
            "",
            "+",
            "-",
            ".",
            ".e3",
            "1e",
            "1e+",
            "1E-",
            "0x",
            "0xg",
            "0x1g",
            "1a",
            "1.2.3",
            "--1",
            ".5.5",
            "1e3.5",
            "+.",
            "Infinityx",
            "-NaNx",
            "e3",
            "E5",
        ] {
            assert_eq!(err(t), Error::InvalidNumber { offset: 0 }, "token {t:?}");
        }
    }

    #[test]
    fn errors_carry_token_offset() {
        assert_eq!(err("1a"), Error::InvalidNumber { offset: 0 });
        assert_eq!(parse(b"1a", 7).unwrap_err(), Error::InvalidNumber {
            offset: 7
        });
    }

    /// The DOM `parse` is a pure `parse_num` -> `Value` mapping; a mismatch
    /// here would silently fork the DOM and serde number semantics.
    #[test]
    fn parse_matches_parse_num_mapping() {
        for t in [
            "0",
            "-1",
            "+1",
            "-0",
            "007",
            "18446744073709551615",
            "-9223372036854775808",
            "18446744073709551616",
            "0x1f",
            "-0x10",
            "0x1FFFFFFFFFFFFFFFF",
            ".5",
            "5.",
            "1.e3",
            "1.5",
            "1e3",
            "Infinity",
            "-Infinity",
            "NaN",
        ] {
            let n = parse_num(t.as_bytes(), 0).unwrap();
            let mapped = match n {
                Num::I64(i) => Value::I64(i),
                Num::U64(u) => Value::U64(u),
                Num::F64(f) => Value::F64(f),
            };
            // Debug renderings: F64(NaN) never equals itself.
            assert_eq!(
                format!("{:?}", parse(t.as_bytes(), 0).unwrap()),
                format!("{mapped:?}"),
                "token {t:?}"
            );
        }
        assert!(matches!(parse_num(b"NaN", 0), Ok(Num::F64(f)) if f.is_nan()));
    }

    #[test]
    fn num_start_covers_token_value_routing() {
        // Every first byte `stage2::token_value` routes to the number parser,
        // and no byte it routes to the literal/keyword arm; the serde fast
        // path leans on this to match the slow path exactly.
        for b in b"0123456789+-." {
            assert!(is_num_start(*b));
        }
        for t in ["Infinity", "NaN", "true", "false", "null", "x", "_", "$"] {
            assert!(!is_num_start(t.as_bytes()[0]), "token {t:?}");
        }
    }

    #[test]
    fn wide_integer_literals_exact() {
        let wide = |t: &str| parse_num_wide(t.as_bytes(), 0).unwrap();
        assert_eq!(wide("0"), Num128::U128(0));
        assert_eq!(wide("+1"), Num128::U128(1));
        assert_eq!(wide("-1"), Num128::I128(-1));
        assert_eq!(wide("-0"), Num128::I128(0));
        assert_eq!(wide("007"), Num128::U128(7));
        // The u64/i64 boundaries and one past them, where `parse_num` overflows to F64.
        assert_eq!(
            wide("18446744073709551615"),
            Num128::U128(u128::from(u64::MAX))
        );
        assert_eq!(wide("18446744073709551616"), Num128::U128(1 << 64));
        assert_eq!(
            wide("-9223372036854775808"),
            Num128::I128(i128::from(i64::MIN))
        );
        assert_eq!(
            wide("-9223372036854775809"),
            Num128::I128(i128::from(i64::MIN) - 1)
        );
        // The 128-bit boundaries.
        assert_eq!(
            wide("340282366920938463463374607431768211455"),
            Num128::U128(u128::MAX)
        );
        assert_eq!(
            wide("-170141183460469231731687303715884105728"),
            Num128::I128(i128::MIN)
        );
        // Hex, both prefixes and signs, at the same boundaries.
        assert_eq!(wide("0x0"), Num128::U128(0));
        assert_eq!(wide("+0x10"), Num128::U128(16));
        assert_eq!(wide("-0x10"), Num128::I128(-16));
        assert_eq!(
            wide("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"),
            Num128::U128(u128::MAX)
        );
        assert_eq!(
            wide("-0x80000000000000000000000000000000"),
            Num128::I128(i128::MIN)
        );
    }

    #[test]
    fn wide_integer_literals_out_of_range() {
        for t in [
            "340282366920938463463374607431768211456",  // 2^128
            "-170141183460469231731687303715884105729", // i128::MIN - 1
            "0x100000000000000000000000000000000",      // 2^128 in hex
            "-0x80000000000000000000000000000001",
            "0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", // 33 hex digits
            "99999999999999999999999999999999999999999999999999", // 50 digits
        ] {
            assert_eq!(
                parse_num_wide(t.as_bytes(), 0),
                Err(Error::InvalidNumber { offset: 0 }),
                "token {t}"
            );
        }
        // The error carries the token's input offset.
        assert_eq!(
            parse_num_wide(b"340282366920938463463374607431768211456", 9).unwrap_err(),
            Error::InvalidNumber { offset: 9 }
        );
    }

    #[test]
    fn wide_entry_matches_parse_num_off_integers() {
        // Float and keyword spellings keep parse_num's F64 value (the 128-bit
        // integer visitors reject them as type errors)...
        assert_eq!(parse_num_wide(b"1.5", 0), Ok(Num128::F64(1.5)));
        assert_eq!(parse_num_wide(b"1e3", 0), Ok(Num128::F64(1000.0)));
        assert_eq!(parse_num_wide(b"5.", 0), Ok(Num128::F64(5.0)));
        assert_eq!(parse_num_wide(b".5", 0), Ok(Num128::F64(0.5)));
        assert!(matches!(parse_num_wide(b"Infinity", 0), Ok(Num128::F64(f)) if f == f64::INFINITY));
        assert!(
            matches!(parse_num_wide(b"-Infinity", 0), Ok(Num128::F64(f)) if f == f64::NEG_INFINITY)
        );
        assert!(matches!(parse_num_wide(b"NaN", 0), Ok(Num128::F64(f)) if f.is_nan()));
        // ...and malformed tokens keep parse_num's exact errors.
        for t in [
            "",
            "+",
            "-",
            ".",
            ".e3",
            "1e",
            "1e+",
            "0x",
            "0xg",
            "0x1g",
            "1a",
            "1.2.3",
            "--1",
            "Infinityx",
            "e3",
            "1_0",
        ] {
            assert_eq!(
                parse_num_wide(t.as_bytes(), 0).unwrap_err(),
                parse_num(t.as_bytes(), 0).unwrap_err(),
                "token {t:?}"
            );
        }
    }

    #[test]
    fn f32_entry_single_rounding() {
        let f32n = |t: &str| parse_num_f32(t.as_bytes(), 0).unwrap();
        assert_eq!(f32n("42"), NumF32::U64(42));
        assert_eq!(f32n("-7"), NumF32::I64(-7));
        assert_eq!(f32n("0x10"), NumF32::U64(16));
        assert_eq!(f32n("1.5"), NumF32::F32(1.5));
        assert_eq!(f32n(".5"), NumF32::F32(0.5));
        assert_eq!(f32n("5."), NumF32::F32(5.0));
        assert_eq!(f32n("1.e3"), NumF32::F32(1000.0));
        assert_eq!(f32n("1e3"), NumF32::F32(1000.0));
        assert_eq!(f32n("Infinity"), NumF32::F32(f32::INFINITY));
        assert_eq!(f32n("-Infinity"), NumF32::F32(f32::NEG_INFINITY));
        assert!(matches!(f32n("NaN"), NumF32::F32(v) if v.is_nan()));
        // This decimal sits just above the f32 midpoint 1 + 2^-24: the direct
        // f32 rounding rounds up, while decimal -> f64 -> f32 lands on the
        // exact midpoint and ties to even (1.0).
        assert_eq!(f32n("1.0000000596046448"), NumF32::F32(1.000_000_1_f32));
        // Hex past the 64-bit variants has no decimal spelling; its f64
        // value (2^65 after the u128 rounding) narrows.
        assert_eq!(f32n("0x1FFFFFFFFFFFFFFFF"), NumF32::F32(3.689_349e19_f32));
        // Grammar errors are parse_num's.
        for t in ["", "1a", "0x", "1.2.3", ".e3"] {
            assert_eq!(
                parse_num_f32(t.as_bytes(), 0).unwrap_err(),
                parse_num(t.as_bytes(), 0).unwrap_err(),
                "token {t:?}"
            );
        }
    }
}
