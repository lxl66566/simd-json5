//! Stage 1: structural indexer.
//!
//! Contract that every backend (the scalar scan and every SIMD level
//! dispatched in [`super::simd`]) must uphold:
//!
//! - `build_index` appends strictly ascending byte offsets into `bytes`.
//! - An offset is emitted for:
//!   - every structural byte outside strings and comments: `{` `}` `[` `]` `,` `:`;
//!   - the opening quote of every string (`"` or `'`);
//!   - the first byte of every bare token: a maximal run of bytes that are not separators (numbers,
//!     identifiers such as unquoted keys, `true`, `false`, `null`, `Infinity`, `NaN`). The
//!     beginning of the input acts as a separator.
//! - Separator bytes are: ASCII whitespace (`0x09..=0x0D`, `0x20`), structural bytes, quote bytes,
//!   `/` (comment introducer), and Unicode whitespace (`U+00A0`, `U+1680`, `U+2000..=U+200A`,
//!   `U+2028`, `U+2029`, `U+202F`, `U+205F`, `U+3000`, `U+FEFF`). Unicode whitespace is classified
//!   in a scalar slow path; every other non-ASCII byte is an ordinary token byte (identifier
//!   character).
//! - Token bytes are the ASCII characters `A-Z a-z 0-9 $ _ + - . \` plus non-ASCII bytes that are
//!   not Unicode whitespace. `\` is a token byte so that unquoted identifiers may contain `\uXXXX`
//!   escapes (validated in stage 2); the escape never spans token boundaries because no separator
//!   byte may appear inside an escape.
//! - Validation performed here, returning `Err` with the offending offset:
//!   - any byte outside strings/comments that is not a separator and not a token byte (reported as
//!     [`Error::UnexpectedChar`]);
//!   - `/` not followed by `/` or `*` (reported as [`Error::UnexpectedChar`] at the `/`);
//!   - unterminated string or comment.
//! - Inside strings stage 1 only locates the closing quote, honoring backslash escapes; escape
//!   validation happens in stage 2.

use crate::{Error, Result};

/// Append structural offsets for `bytes` to `index`.
///
/// Canonical scalar reference: the production path on targets with no
/// supported vector ISA and the differential-test oracle for the SIMD
/// kernels in [`super::simd`]; elsewhere the parser routes through the SIMD
/// dispatch instead.
#[allow(clippy::cast_possible_truncation)] // guarded by the length check below
pub(crate) fn build_index(bytes: &[u8], index: &mut Vec<u32>) -> Result<()> {
    if bytes.len() > u32::MAX as usize {
        return Err(Error::Syntax {
            message: "input larger than 4 GiB is not supported".into(),
            offset: 0,
        });
    }
    let mut i = 0;
    while i < bytes.len() {
        // Every arm leaves `i` on a decision point, so a token byte seen here
        // always starts a fresh token (token runs are consumed atomically).
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
                // Scalar slow path: classify the code point before deciding.
                let (cp, len) = decode_seq(bytes, i).ok_or(Error::InvalidUtf8 { offset: i })?;
                if is_unicode_whitespace(cp) {
                    i += len;
                } else {
                    index.push(i as u32);
                    i = token_end(bytes, i);
                }
            },
            b if is_ascii_token_byte(b) => {
                index.push(i as u32);
                i = token_end(bytes, i);
            },
            b => return Err(Error::UnexpectedChar { byte: b, offset: i }),
        }
    }
    Ok(())
}

/// End of the bare token starting at `start`: the first position whose byte is
/// not a token byte. Unicode whitespace terminates a token; other non-ASCII
/// sequences continue it. Malformed UTF-8 also terminates (the caller reports
/// it when the main loop reaches the offending byte).
#[must_use]
pub(crate) fn token_end(bytes: &[u8], start: usize) -> usize {
    let mut i = start;
    while i < bytes.len() {
        let b = bytes[i];
        if b < 0x80 {
            if CLASSES[b as usize] & class::TOKEN == 0 {
                break;
            }
            i += 1;
        } else {
            match decode_seq(bytes, i) {
                Some((cp, len)) if !is_unicode_whitespace(cp) => i += len,
                _ => break,
            }
        }
    }
    i
}

/// Byte-class bit flags shared by the hot byte scans of both stages; see
/// [`CLASSES`]. One table lookup replaces a compare chain per byte.
pub(crate) mod class {
    /// Stage-1 token byte: identifier characters, number characters and the
    /// backslash (see [`super::is_ascii_token_byte`]).
    pub const TOKEN: u8 = 0x01;
    /// Identifier continue byte `[A-Za-z0-9_$]`.
    pub const IDENT: u8 = 0x02;
    /// Backslash `\`: string escape introducer and token byte.
    pub const BACKSLASH: u8 = 0x04;
    /// Line feed `\n`: illegal raw inside strings.
    pub const LF: u8 = 0x08;
    /// Carriage return `\r`: illegal raw inside strings.
    pub const CR: u8 = 0x10;
    /// Double quote `"`.
    pub const DQUOTE: u8 = 0x20;
    /// Single quote `'`.
    pub const SQUOTE: u8 = 0x40;
    /// Structural byte: `{` `}` `[` `]` `,` `:`.
    pub const STRUCTURAL: u8 = 0x80;
}

/// 256-entry byte classification: `CLASSES[b] & class::TOKEN != 0` etc. The
/// table is 4 cache lines and stays hot through every scan.
pub(crate) static CLASSES: [u8; 256] = build_classes();

#[allow(clippy::cast_possible_truncation)] // b is the loop index, 0..=255
const fn build_classes() -> [u8; 256] {
    let mut t = [0_u8; 256];
    let mut b = 0_usize;
    while b < 256 {
        let c = b as u8;
        let mut v = 0_u8;
        if matches!(
            c,
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'$' | b'_' | b'+' | b'-' | b'.' | b'\\'
        ) {
            v |= class::TOKEN;
        }
        if matches!(c, b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'$' | b'_') {
            v |= class::IDENT;
        }
        if c == b'\\' {
            v |= class::BACKSLASH;
        }
        if c == b'\n' {
            v |= class::LF;
        }
        if c == b'\r' {
            v |= class::CR;
        }
        if c == b'"' {
            v |= class::DQUOTE;
        }
        if c == b'\'' {
            v |= class::SQUOTE;
        }
        if matches!(c, b'{' | b'}' | b'[' | b']' | b',' | b':') {
            v |= class::STRUCTURAL;
        }
        t[b] = v;
        b += 1;
    }
    t
}

/// Position just past the string whose opening quote is at `open`, honoring
/// backslash escape parity. Escape contents are not validated here.
pub(crate) fn skip_string(bytes: &[u8], open: usize) -> Result<usize> {
    let quote = bytes[open];
    let mut i = open + 1;
    while i < bytes.len() {
        match bytes[i] {
            // May overshoot the end by one; the loop guard then reports
            // the string as unterminated.
            b'\\' => i += 2,
            b if b == quote => return Ok(i + 1),
            _ => i += 1,
        }
    }
    Err(Error::UnterminatedString { offset: open })
}

/// Position just past the comment whose `/` is at `slash`.
pub(crate) fn skip_comment(bytes: &[u8], slash: usize) -> Result<usize> {
    match bytes.get(slash + 1) {
        Some(b'/') => {
            // Line comment: runs to the newline (exclusive; the newline is
            // ordinary whitespace for the next iteration).
            let nl = bytes[slash + 2..]
                .iter()
                .position(|&b| b == b'\n')
                .unwrap_or(bytes.len() - (slash + 2));
            Ok((slash + 2 + nl + 1).min(bytes.len()))
        },
        Some(b'*') => {
            // Block comments do not nest: the first `*/` ends them. The
            // search starts after the opener so `/*/` stays unterminated.
            let mut i = slash + 2;
            while i + 1 < bytes.len() {
                if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                    return Ok(i + 2);
                }
                i += 1;
            }
            Err(Error::UnterminatedComment { offset: slash })
        },
        // A lone `/` at end of input, or one that introduces nothing.
        _ => Err(Error::UnexpectedChar {
            byte: b'/',
            offset: slash,
        }),
    }
}

/// ASCII token bytes: identifier characters, number characters and the
/// backslash that may start a `\uXXXX` identifier escape.
fn is_ascii_token_byte(b: u8) -> bool {
    matches!(b, b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'$' | b'_' | b'+' | b'-' | b'.' | b'\\')
}

/// Unicode whitespace per the stage 1 contract: the JSON5 WhiteSpace and
/// LineTerminator sets minus the ASCII range (Zs, NBSP/BOM and LS/PS). NEL
/// (`U+0085`) is category Cc, not Zs, and is not a JSON5 separator: like any
/// other non-separator code point it continues a bare token.
#[must_use]
pub(crate) fn is_unicode_whitespace(cp: u32) -> bool {
    matches!(
        cp,
        0xa0 | 0x1680 | 0x2000..=0x200a | 0x2028 | 0x2029 | 0x202f | 0x205f | 0x3000 | 0xfeff
    )
}

/// Decode the UTF-8 sequence starting at `i`. Returns `(code_point, length)`.
/// Structural validation only (lead byte, continuation prefixes, slice
/// bounds): inputs reaching the parser are `&str`-validated already, so
/// overlong forms and surrogate encodings are not re-checked.
pub(crate) fn decode_seq(bytes: &[u8], i: usize) -> Option<(u32, usize)> {
    let b = *bytes.get(i)?;
    let len = match b {
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        0xf0..=0xf7 => 4,
        _ => return None,
    };
    if i + len > bytes.len() {
        return None;
    }
    let mut cp = u32::from(b & (0x7f >> len));
    for &c in &bytes[i + 1..i + len] {
        if c & 0xc0 != 0x80 {
            return None;
        }
        cp = (cp << 6) | u32::from(c & 0x3f);
    }
    Some((cp, len))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idx(input: &str) -> Vec<u32> {
        let mut v = Vec::new();
        build_index(input.as_bytes(), &mut v).unwrap();
        v
    }

    fn err(input: &str) -> Error {
        let mut v = Vec::new();
        build_index(input.as_bytes(), &mut v).unwrap_err()
    }

    #[test]
    fn empty_and_whitespace_only_emit_nothing() {
        assert_eq!(idx(""), Vec::<u32>::new());
        assert_eq!(idx(" \t\r\n\x0b\x0c"), Vec::<u32>::new());
    }

    #[test]
    fn structurals_are_emitted_in_order() {
        assert_eq!(idx("{},[]:"), vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn tokens_are_single_entries() {
        assert_eq!(idx("true"), vec![0]);
        assert_eq!(idx("-Infinity"), vec![0]);
        assert_eq!(idx("0x1F +2 1.e3"), vec![0, 5, 8]);
        assert_eq!(idx("1,2"), vec![0, 1, 2]);
    }

    #[test]
    fn token_run_spans_all_token_bytes() {
        assert_eq!(idx("a-b.c+d"), vec![0]);
        assert_eq!(idx("a$b _c"), vec![0, 4]);
        assert_eq!(idx(r"a\u0062c"), vec![0]);
    }

    #[test]
    fn comments_emit_nothing() {
        assert_eq!(idx("// c\n1"), vec![5]);
        assert_eq!(idx("1 // c"), vec![0]);
        // "/* multi\nline */" occupies bytes 0..=15.
        assert_eq!(idx("/* multi\nline */[1]"), vec![16, 17, 18]);
        assert_eq!(idx("/**/[1,]"), vec![4, 5, 6, 7]);
        assert_eq!(idx("[1/*c*/,2]"), vec![0, 1, 7, 8, 9]);
        assert!(build_index(b"/*/", &mut Vec::new()).is_err());
    }

    #[test]
    fn strings_emit_opening_quote_only() {
        assert_eq!(idx(r#""abc""#), vec![0]);
        assert_eq!(idx(r#""a\"b" "#), vec![0]);
        assert_eq!(idx("''"), vec![0]);
        assert_eq!(idx(r"'don\'t'"), vec![0]);
        // Structural bytes inside strings are not indexed.
        assert_eq!(idx(r#""{},:[]""#), vec![0]);
        assert_eq!(idx(r#"["a", 'b']"#), vec![0, 1, 4, 6, 9]);
    }

    #[test]
    fn unicode_whitespace_separates() {
        // U+00A0, U+2028, U+3000, U+FEFF between single-byte tokens.
        let s = "1\u{a0}2\u{2028}3\u{3000}4\u{feff}5";
        assert_eq!(idx(s), vec![0, 3, 7, 11, 15]);
    }

    #[test]
    fn non_ascii_bytes_form_identifiers() {
        assert_eq!(idx("é"), vec![0]);
        assert_eq!(idx("日本語"), vec![0]);
        // é is not whitespace: the token spans both bytes and continues.
        assert_eq!(idx("aéb c"), vec![0, 5]);
    }

    #[test]
    fn illegal_bytes_are_rejected() {
        assert_eq!(err("!"), Error::UnexpectedChar {
            byte: b'!',
            offset: 0
        });
        assert_eq!(err("[1?]"), Error::UnexpectedChar {
            byte: b'?',
            offset: 2
        });
        assert_eq!(err("a\x00b"), Error::UnexpectedChar { byte: 0, offset: 1 });
        assert_eq!(err("("), Error::UnexpectedChar {
            byte: b'(',
            offset: 0
        });
    }

    #[test]
    fn stray_slash_is_rejected() {
        assert_eq!(err("/x"), Error::UnexpectedChar {
            byte: b'/',
            offset: 0
        });
        assert_eq!(err("1 / 2"), Error::UnexpectedChar {
            byte: b'/',
            offset: 2
        });
        assert_eq!(err("/"), Error::UnexpectedChar {
            byte: b'/',
            offset: 0
        });
        assert_eq!(err("5/"), Error::UnexpectedChar {
            byte: b'/',
            offset: 1
        });
    }

    #[test]
    fn unterminated_string_and_comment() {
        assert_eq!(err("\"abc"), Error::UnterminatedString { offset: 0 });
        assert_eq!(err("'a\\'"), Error::UnterminatedString { offset: 0 });
        assert_eq!(err("\"a\\"), Error::UnterminatedString { offset: 0 });
        assert_eq!(err("/* x"), Error::UnterminatedComment { offset: 0 });
        assert_eq!(err("/*/"), Error::UnterminatedComment { offset: 0 });
        // A line comment may end at end of input.
        assert_eq!(idx("// no newline"), Vec::<u32>::new());
    }

    #[test]
    fn mixed_document() {
        // { a : 1 , // comment\n  ' b ' : [ 2 , 3 , ] }
        // 0 2 3   5 6               21 23 24  26 27 28 30 31 32  34
        let src = "{ a: 1, // comment\n  'b': [2, 3,] }";
        assert_eq!(idx(src), vec![
            0, 2, 3, 5, 6, 21, 24, 26, 27, 28, 30, 31, 32, 34
        ]);
    }
}
