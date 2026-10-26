//! Number parsing for the DESIGN grammar pin `[+-]?(Infinity|NaN|Hex|Decimal)`.
//!
//! Decimal forms follow ECMAScript 5.1: `5.`, `.5`, `1.e3`, `+1` and `007`
//! are valid; `.e3`, `1e` and `0x` are not. Integers that fit u64 (positive)
//! or i64 (negative) stay integer variants; anything else becomes an exactly
//! converted f64. Float evaluation is delegated to `fast_float2`; only sign
//! handling, grammar validation and integer accumulation are done here.

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

/// Parse a bare number token. `token` is the full maximal token run reported
/// by stage 1; `offset` is its first byte position in the input, used for
/// errors.
pub(crate) fn parse(token: &[u8], offset: usize) -> Result<Value<'static>> {
    let (neg, rest) = match token.first() {
        Some(&b'+') => (false, &token[1..]),
        Some(&b'-') => (true, &token[1..]),
        _ => (false, token),
    };
    if rest == b"Infinity" {
        return Ok(Value::F64(if neg {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        }));
    }
    if rest == b"NaN" {
        return Ok(Value::F64(f64::NAN));
    }
    if matches!(rest.first(), Some(&b'0') if matches!(rest.get(1), Some(&(b'x' | b'X')))) {
        return parse_hex(&rest[2..], neg, offset);
    }
    parse_decimal(rest, neg, offset)
}

/// Hexadecimal integer: `0x` has already been consumed.
#[allow(
    clippy::cast_precision_loss, // u128 -> f64 widening is the documented overflow path
    clippy::cast_possible_truncation, // in-range by the adjacent bound checks
    clippy::cast_possible_wrap // acc <= 2^63, so `as i64` only "wraps" 2^63 to i64::MIN
)]
fn parse_hex(digits: &[u8], neg: bool, offset: usize) -> Result<Value<'static>> {
    const TWO_POW_63: u128 = 1 << 63;
    const U64_MAX: u128 = u64::MAX as u128;
    if digits.is_empty() {
        return Err(Error::InvalidNumber { offset });
    }
    let mut acc: u128 = 0;
    // Past u128 range the value keeps accumulating in f64; beyond 2^128 the
    // stepwise rounding may differ from a correctly rounded conversion by an
    // ULP. Inputs with more than 32 hex digits are out of scope.
    let mut wide: Option<f64> = None;
    for &d in digits {
        let Some(v) = hex_val(d) else {
            return Err(Error::InvalidNumber { offset });
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
        return Ok(Value::F64(if neg {
            -f
        } else {
            f
        }));
    }
    if neg {
        if acc <= TWO_POW_63 {
            // acc == 2^63 wraps to i64::MIN, whose negation is itself.
            return Ok(Value::I64((acc as i64).wrapping_neg()));
        }
    } else if acc <= U64_MAX {
        return Ok(Value::U64(acc as u64));
    }
    let f = acc as f64;
    Ok(Value::F64(if neg {
        -f
    } else {
        f
    }))
}

/// Unsigned decimal per ECMAScript 5.1, sign already stripped.
fn parse_decimal(s: &[u8], neg: bool, offset: usize) -> Result<Value<'static>> {
    let int_digits = count_digits(s, 0);
    let mut i = int_digits;
    let mut frac_digits = 0;
    let mut had_dot = false;
    if s.get(i) == Some(&b'.') {
        had_dot = true;
        i += 1;
        frac_digits = count_digits(s, i);
        i += frac_digits;
        // `5.` and `.5` are valid; a lone `.` (or `.e3`) is not.
        if int_digits + frac_digits == 0 {
            return Err(Error::InvalidNumber { offset });
        }
    } else if int_digits == 0 {
        // No digits and no dot: `e3`, empty, stray sign.
        return Err(Error::InvalidNumber { offset });
    }
    let mut had_exp = false;
    if matches!(s.get(i), Some(&(b'e' | b'E'))) {
        i += 1;
        if matches!(s.get(i), Some(&(b'+' | b'-'))) {
            i += 1;
        }
        // `1e` and `1e+` are rejected.
        let exp_digits = count_digits(s, i);
        if exp_digits == 0 {
            return Err(Error::InvalidNumber { offset });
        }
        i += exp_digits;
        had_exp = true;
    }
    // The token is a maximal run, so any residue (`1a`, `1.2.3`) is invalid.
    if i != s.len() {
        return Err(Error::InvalidNumber { offset });
    }
    if had_dot || had_exp {
        // Fraction layout facts come straight from the validation pass; no
        // second scan for the dot position.
        let leading_dot = int_digits == 0;
        let elided_frac = had_dot && frac_digits == 0;
        return Ok(Value::F64(signed(
            neg,
            float_value(s, leading_dot, elided_frac),
        )));
    }
    // Pure integer: keep the integer variant while the magnitude fits.
    let mut acc: u64 = 0;
    for &d in &s[..int_digits] {
        let v = u64::from(d - b'0');
        // Positive overflow past u64::MAX, or negative past i64::MIN.
        acc = if let Some(a) = acc.checked_mul(10).and_then(|a| a.checked_add(v)) {
            a
        } else {
            let f = lib_parse(s);
            return Ok(Value::F64(signed(neg, f)));
        };
    }
    if !neg {
        return Ok(Value::U64(acc));
    }
    // -(acc) fits i64 iff acc <= 2^63; acc == 2^63 wraps to i64::MIN, whose
    // negation is itself.
    #[allow(clippy::cast_possible_wrap)]
    if acc <= (i64::MAX as u64) + 1 {
        return Ok(Value::I64((acc as i64).wrapping_neg()));
    }
    let f = lib_parse(s);
    Ok(Value::F64(signed(neg, f)))
}

fn signed(neg: bool, f: f64) -> f64 {
    if neg {
        -f
    } else {
        f
    }
}

/// Number of ASCII digits starting at `from`.
fn count_digits(s: &[u8], from: usize) -> usize {
    s[from..].iter().take_while(|b| b.is_ascii_digit()).count()
}

/// Evaluate an unsigned, grammar-validated decimal as f64. `leading_dot`
/// marks `.5` forms and `elided_frac` marks `5.` / `1.e3` forms; both are
/// normalized (to `0.5` / `5.0`) because C `from_chars` semantics are not
/// guaranteed for elided digit runs by every backend. Both flags come from
/// the grammar pass, so no byte is scanned twice.
fn float_value(s: &[u8], leading_dot: bool, elided_frac: bool) -> f64 {
    if !leading_dot && !elided_frac {
        return lib_parse(s);
    }
    let mut buf = Vec::with_capacity(s.len() + 1);
    if leading_dot {
        buf.push(b'0');
    }
    if elided_frac {
        // A single dot with digits before it and none after: `5.` / `1.e3`.
        let dot = s.iter().position(|&b| b == b'.').unwrap_or(0);
        buf.extend_from_slice(&s[..=dot]);
        buf.push(b'0');
        buf.extend_from_slice(&s[dot + 1..]);
    } else {
        buf.extend_from_slice(s);
    }
    lib_parse(&buf)
}

fn lib_parse(s: &[u8]) -> f64 {
    if let Ok(f) = fast_float2::parse::<f64, _>(s) {
        return f;
    }
    // Unreachable for grammar-validated input (overflow reports Ok(inf));
    // if ever hit, defer to std, which also maps underflow to 0.
    core::str::from_utf8(s)
        .ok()
        .and_then(|t| t.parse::<f64>().ok())
        .unwrap_or(f64::INFINITY)
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
}
