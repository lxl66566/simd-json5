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
//! - Side table: for every *clean* string — one whose bytes strictly between the quotes contain no
//!   backslash and no raw LF/CR — one `Span { open, close }` is appended to `spans`, recording the
//!   opening- and closing-quote offsets in ascending `open` order. Stage 2 borrows a spanned
//!   string's content directly instead of re-scanning it; a string without a span takes the full
//!   validating parse, so an over-conservative dirty marking is only slower, never wrong.

use memchr::{memchr, memchr2};

use crate::{Error, Result};

/// A clean string recorded by stage 1 (see the module contract): the string
/// opening at `open` closes at `close` (both quote offsets), and the content
/// `bytes[open + 1..close]` contains no backslash and no raw LF/CR — exactly
/// the strings whose stage-2 parse reduces to borrowing that slice.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Span {
    pub(crate) open: u32,
    pub(crate) close: u32,
}

/// Reject inputs whose offsets do not fit `u32`.
pub(crate) fn check_input_len(bytes: &[u8]) -> Result<()> {
    if bytes.len() > u32::MAX as usize {
        return Err(Error::Syntax {
            message: "input larger than 4 GiB is not supported".into(),
            offset: 0,
        });
    }
    Ok(())
}

/// Append structural offsets for `bytes` to `index` and clean-string spans to
/// `spans`.
///
/// Canonical scalar reference: the production path on targets with no
/// supported vector ISA and the differential-test oracle for the SIMD
/// kernels in [`super::simd`]; elsewhere the parser routes through the SIMD
/// dispatch instead.
pub(crate) fn build_index(bytes: &[u8], index: &mut Vec<u32>, spans: &mut Vec<Span>) -> Result<()> {
    check_input_len(bytes)?;
    scan_scalar(bytes, 0, false, index, spans)
}

/// The canonical scalar scan from byte `i`, with an `in_token` carry so the
/// SIMD kernels can hand off mid-token-run at block boundaries. When
/// `in_token` is false the loop body is the plain decision-point loop;
/// callers enforce the 4 GiB length limit.
#[allow(clippy::cast_possible_truncation)] // guarded by check_input_len in callers
pub(crate) fn scan_scalar(
    bytes: &[u8],
    mut i: usize,
    in_token: bool,
    index: &mut Vec<u32>,
    spans: &mut Vec<Span>,
) -> Result<()> {
    if in_token {
        // Finish the token run that the vector part started: consume it
        // without emitting (its start was already emitted).
        i = token_end(bytes, i);
    }
    while i < bytes.len() {
        // Every arm leaves `i` on a decision point, so a token byte seen here
        // always starts a fresh token (token runs are consumed atomically).
        // `i` moves through opaque callees (skip_string, token_end), so the
        // bounds check on this read does not fold into the loop condition.
        // SAFETY: `i < bytes.len()` is the loop condition.
        let b = unsafe { *bytes.get_unchecked(i) };
        match b {
            b'{' | b'}' | b'[' | b']' | b',' | b':' => {
                index.push(i as u32);
                i += 1;
            },
            b'"' | b'\'' => {
                index.push(i as u32);
                let (end, clean) = skip_string(bytes, i)?;
                if clean {
                    spans.push(Span {
                        open: i as u32,
                        close: (end - 1) as u32,
                    });
                }
                i = end;
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
            b if CLASSES[b as usize] & class::TOKEN != 0 => {
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
        // SAFETY: `i < bytes.len()` is the loop condition; `i` only advances
        // by whole classified steps (1 for ASCII, a decoded sequence length
        // otherwise), so every re-entry satisfies it again.
        let b = unsafe { *bytes.get_unchecked(i) };
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
    /// backslash (the exact set marked `TOKEN` in [`CLASSES`]).
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

/// Position just past the string whose opening quote is at `open`, with its
/// cleanliness flag; see [`skip_string_tail`].
pub(crate) fn skip_string(bytes: &[u8], open: usize) -> Result<(usize, bool)> {
    skip_string_tail(bytes, open, open + 1, false, true)
}

/// Scalar string skip from `i` with an incoming escape-parity flag (used
/// when a vector skip ends mid-escape), returning `(end, clean)`: `end` is
/// just past the closing quote, and `clean` (carried in, carried out) marks
/// content with neither a backslash nor an LF/CR byte so far. A clean string
/// needs no escape validation and no terminator re-check, so stage 2 can use
/// the recorded extent instead of re-scanning. The test is conservative —
/// an escaped LF (line continuation) also clears it, costing only the fast
/// path.
///
/// Mirrors the canonical backslash-consumes-the-next-byte semantics.
pub(crate) fn skip_string_tail(
    bytes: &[u8],
    open: usize,
    mut i: usize,
    mut escaped: bool,
    mut clean: bool,
) -> Result<(usize, bool)> {
    let quote = bytes[open];
    loop {
        if escaped {
            // The pending backslash (the vector part's last byte) consumes
            // exactly the byte at `i`, whatever it is. When that target is
            // not itself a quote or backslash, the next hit below belongs to
            // the string proper and must not be swallowed with it. (A
            // pending escape implies a backslash in the vector part, which
            // already cleared `clean` there.)
            escaped = false;
            i += 1;
        }
        // `i` may overshoot the end by one on a dangling escape; the empty
        // search range then reports the string as unterminated.
        let Some(hit) = memchr2(quote, b'\\', &bytes[i.min(bytes.len())..]) else {
            return Err(Error::UnterminatedString { offset: open });
        };
        let at = i + hit;
        // A raw LF/CR since the last hit disqualifies the span (stage 2's
        // parse reports it). The scan is bounded by `at`, and consecutive
        // ranges are disjoint, so the two scans stay linear together.
        if clean && memchr2(b'\n', b'\r', &bytes[i..at]).is_some() {
            clean = false;
        }
        if bytes[at] == quote {
            return Ok((at + 1, clean));
        }
        clean = false; // the backslash itself
        i = at + 2; // the backslash consumes the byte after it
    }
}

/// Position just past the comment whose `/` is at `slash`.
pub(crate) fn skip_comment(bytes: &[u8], slash: usize) -> Result<usize> {
    match bytes.get(slash + 1) {
        Some(b'/') => {
            // Line comment: runs to the newline (exclusive; the newline is
            // ordinary whitespace for the next iteration), or to the end of
            // input when there is none.
            Ok(match memchr(b'\n', &bytes[slash + 2..]) {
                Some(nl) => slash + 2 + nl + 1,
                None => bytes.len(),
            })
        },
        Some(b'*') => {
            // Block comments do not nest: the first `*/` ends them. The
            // search starts after the opener's `*` so `/*/` stays
            // unterminated; scanning for `*` instead of `/` also keeps
            // slash-dense comments (`/* //// ... */`) from restarting the
            // search on every hit.
            let mut i = slash + 2;
            while let Some(hit) = memchr(b'*', &bytes[i..]) {
                let at = i + hit;
                // A star as the last input byte has no successor and cannot
                // close the comment.
                if bytes.get(at + 1) == Some(&b'/') {
                    return Ok(at + 2);
                }
                i = at + 1;
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
        build_index(input.as_bytes(), &mut v, &mut Vec::new()).unwrap();
        v
    }

    fn spans(input: &str) -> Vec<Span> {
        let mut v = Vec::new();
        let mut s = Vec::new();
        build_index(input.as_bytes(), &mut v, &mut s).unwrap();
        s
    }

    fn err(input: &str) -> Error {
        let mut v = Vec::new();
        build_index(input.as_bytes(), &mut v, &mut Vec::new()).unwrap_err()
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
        assert!(build_index(b"/*/", &mut Vec::new(), &mut Vec::new()).is_err());
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

    /// Clean strings (no backslash, no raw LF/CR before the close) record a
    /// span; escaped strings, strings with raw line terminators and strings
    /// inside comments do not. Raw U+2028/U+2029 are content, not
    /// terminators.
    #[test]
    fn clean_strings_record_spans() {
        let sp = |o: u32, c: u32| Span { open: o, close: c };
        assert_eq!(spans(r#""abc""#), vec![sp(0, 4)]);
        assert_eq!(spans("''"), vec![sp(0, 1)]); // empty content is clean
        assert_eq!(spans(r#"["a", 'b']"#), vec![sp(1, 3), sp(6, 8)]);
        // Escapes disqualify, including line continuations.
        assert_eq!(spans(r#""a\"b""#), Vec::<Span>::new());
        assert_eq!(spans("\"a\\\nb\""), Vec::<Span>::new());
        // Raw LF/CR disqualify (and the parse rejects them in stage 2).
        assert_eq!(spans("\"a\nb\""), Vec::<Span>::new());
        assert_eq!(spans("\"a\rb\""), Vec::<Span>::new());
        // Raw LS/PS are legal content: still clean.
        assert_eq!(spans("\"a\u{2028}b\""), vec![sp(0, 6)]);
        // Strings inside comments are never indexed nor spanned.
        assert_eq!(spans("// \"a\"\n1"), Vec::<Span>::new());
        // Ascending by opening quote.
        assert_eq!(spans("{'a': \"b\"}"), vec![sp(1, 3), sp(6, 8)]);
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
