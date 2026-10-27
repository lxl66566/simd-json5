//! Stage 2: semantic parser over the structural index produced by stage 1.
//!
//! Walks the index with an explicit container stack (no recursion), enforces
//! JSON5 grammar and container shape, and builds a [`Value`] borrowing from
//! the input bytes. See `DESIGN.md` for the grammar pin.
//!
//! Everything between index entries is separators (whitespace or comments),
//! already classified by stage 1, so only bytes at entry offsets plus string
//! and token bodies are inspected here.

use crate::{
    Error, Result,
    parser::{
        number,
        stage1::{CLASSES, Span, class, decode_seq, is_unicode_whitespace, token_end},
    },
    value::{MAX_DEPTH, Map, Str, Value},
};

/// Object member state machine.
#[derive(Clone, Copy, PartialEq)]
enum ObjState {
    /// Before the first key or after a comma; `}` also allowed.
    KeyOrClose,
    /// A key was read; `:` is required next.
    Colon,
    /// The colon was read; a member value is required next.
    Value,
    /// A member value was read; `,` or `}` next.
    CommaOrClose,
}

enum Frame<'a> {
    Array {
        items: Vec<Value<'a>>,
        after_element: bool,
    },
    Object {
        map: Map<'a>,
        state: ObjState,
        pending_key: Option<Str<'a>>,
    },
}

/// Syntax-error constructor. Cold: keeps error paths out of the hot loop's
/// instruction stream.
#[cold]
fn syntax(message: &str, offset: usize) -> Error {
    Error::Syntax {
        message: message.into(),
        offset,
    }
}

/// Build a DOM from the structural `index` and the clean-string `spans` of
/// stage 1 over `bytes`.
///
/// Containers deliberately grow from empty: pre-sizing them from the
/// [`super::container_counts`] hints was tried in five variants (exact
/// hints, min-filtered, power-of-two-rounded, threshold-gated, pooled
/// table) and each one benched DOM citm_catalog 22-27% slower than the
/// grow path when measured right after another document in the same
/// process (first-in-process the same builds won 5-15%), while the serde
/// path using the identical hints and table never showed the effect. The
/// DOM walk's hot loop is much tighter than the serde machinery; whatever
/// the microarchitectural interaction is, pre-allocation from hints does
/// not survive it, so this loop stays allocation-hint-free.
pub(crate) fn build<'a>(bytes: &'a [u8], index: &[u32], spans: &[Span]) -> Result<Value<'a>> {
    let mut root: Option<Value<'a>> = None;
    // Typical documents nest shallower than this; one upfront allocation
    // covers the whole walk.
    let mut stack: Vec<Frame<'a>> = Vec::with_capacity(16);
    // Clean-span cursor: spans are ordered by opening offset and every
    // string open consumes at most the head.
    let mut spans = spans;

    for &entry in index {
        // Stage-1 entries are always valid input positions (< bytes.len());
        // the widening cast is lossless (usize is at least 32 bits).
        let off = entry as usize;
        // A completed root value may only be followed by separators.
        if root.is_some() && stack.is_empty() {
            return Err(Error::TrailingCharacters { offset: off });
        }
        // SAFETY: `off` comes from the stage 1 index built over this same
        // `bytes`: every entry is a valid input position (< bytes.len()).
        // `build` is pub(crate) and only called by `parser::parse`, right
        // after index construction over the same slice.
        let b = unsafe { *bytes.get_unchecked(off) };
        match b {
            b'{' | b'[' => {
                check_value_position(&stack, off)?;
                if stack.len() >= MAX_DEPTH {
                    return Err(Error::DepthLimitExceeded {
                        limit: MAX_DEPTH,
                        offset: off,
                    });
                }
                stack.push(if b == b'[' {
                    Frame::Array {
                        items: Vec::new(),
                        after_element: false,
                    }
                } else {
                    Frame::Object {
                        map: Map::new(),
                        state: ObjState::KeyOrClose,
                        pending_key: None,
                    }
                });
            },
            b'}' => {
                if !matches!(
                    stack.last(),
                    Some(Frame::Object {
                        state: ObjState::KeyOrClose | ObjState::CommaOrClose,
                        ..
                    })
                ) {
                    return Err(match stack.last() {
                        Some(Frame::Object {
                            state: ObjState::Colon,
                            ..
                        }) => syntax("expected ':'", off),
                        Some(Frame::Object {
                            state: ObjState::Value,
                            ..
                        }) => syntax("expected value", off),
                        _ => Error::UnexpectedChar {
                            byte: b'}',
                            offset: off,
                        },
                    });
                }
                match stack.pop() {
                    Some(Frame::Object { map, .. }) => {
                        deliver(&mut stack, &mut root, Value::Object(map));
                    },
                    _ => unreachable!("frame kind checked above"),
                }
            },
            b']' => {
                if !matches!(stack.last(), Some(Frame::Array { .. })) {
                    return Err(match stack.last() {
                        Some(Frame::Object {
                            state: ObjState::Colon,
                            ..
                        }) => syntax("expected ':'", off),
                        Some(Frame::Object {
                            state: ObjState::Value,
                            ..
                        }) => syntax("expected value", off),
                        _ => Error::UnexpectedChar {
                            byte: b']',
                            offset: off,
                        },
                    });
                }
                match stack.pop() {
                    Some(Frame::Array { items, .. }) => {
                        deliver(&mut stack, &mut root, Value::Array(items));
                    },
                    _ => unreachable!("frame kind checked above"),
                }
            },
            b',' => match stack.last_mut() {
                Some(Frame::Array { after_element, .. }) if *after_element => {
                    *after_element = false;
                },
                Some(Frame::Object { state, .. }) if *state == ObjState::CommaOrClose => {
                    *state = ObjState::KeyOrClose;
                },
                Some(Frame::Object {
                    state: ObjState::Colon,
                    ..
                }) => return Err(syntax("expected ':'", off)),
                Some(Frame::Object {
                    state: ObjState::Value,
                    ..
                }) => return Err(syntax("expected value", off)),
                _ => {
                    return Err(Error::UnexpectedChar {
                        byte: b',',
                        offset: off,
                    })
                },
            },
            b':' => match stack.last_mut() {
                Some(Frame::Object { state, .. }) if *state == ObjState::Colon => {
                    *state = ObjState::Value;
                },
                _ => {
                    return Err(Error::UnexpectedChar {
                        byte: b':',
                        offset: off,
                    })
                },
            },
            b'"' | b'\'' => {
                // A quoted string is a key only in key position; every other
                // position is validated before parsing so a misplaced string
                // reports the structural error instead of its own parse error
                // (the serde path checks the state first too).
                let is_key = matches!(
                    stack.last(),
                    Some(Frame::Object {
                        state: ObjState::KeyOrClose,
                        ..
                    })
                );
                if !is_key {
                    check_value_position(&stack, off)?;
                }
                let s = parse_string_spanned(bytes, off, &mut spans)?;
                if is_key {
                    let Some(Frame::Object {
                        state, pending_key, ..
                    }) = stack.last_mut()
                    else {
                        unreachable!("is_key implies an object frame");
                    };
                    *state = ObjState::Colon;
                    *pending_key = Some(s);
                } else {
                    deliver(&mut stack, &mut root, Value::Str(s));
                }
            },
            _ => {
                if let Some(Frame::Object {
                    state, pending_key, ..
                }) = stack.last_mut()
                    && *state == ObjState::KeyOrClose
                {
                    // Unquoted number keys are invalid.
                    if matches!(b, b'+' | b'-' | b'.' | b'0'..=b'9') {
                        return Err(syntax("expected identifier or string key", off));
                    }
                    let end = token_end(bytes, off);
                    let key = parse_identifier(bytes, off, end)?;
                    *state = ObjState::Colon;
                    *pending_key = Some(key);
                    continue;
                }
                check_value_position(&stack, off)?;
                let v = if let Some(v) = keyword_value(bytes, off) {
                    v
                } else {
                    let end = token_end(bytes, off);
                    token_value(&bytes[off..end], off)?
                };
                deliver(&mut stack, &mut root, v);
            },
        }
    }
    if !stack.is_empty() {
        return Err(Error::UnexpectedEof {
            offset: bytes.len(),
        });
    }
    root.ok_or(Error::UnexpectedEof {
        offset: bytes.len(),
    })
}

/// Verify the innermost container accepts the start of a value here.
fn check_value_position(stack: &[Frame<'_>], off: usize) -> Result<()> {
    match stack.last() {
        None
        | Some(
            Frame::Array {
                after_element: false,
                ..
            }
            | Frame::Object {
                state: ObjState::Value,
                ..
            },
        ) => Ok(()),
        Some(Frame::Array { .. }) => Err(syntax("expected ',' or ']'", off)),
        Some(Frame::Object {
            state: ObjState::KeyOrClose,
            ..
        }) => Err(syntax("expected identifier or string key", off)),
        Some(Frame::Object {
            state: ObjState::Colon,
            ..
        }) => Err(syntax("expected ':'", off)),
        Some(Frame::Object {
            state: ObjState::CommaOrClose,
            ..
        }) => Err(syntax("expected ',' or '}'", off)),
    }
}

/// Attach a completed value to its parent, or to the root slot.
fn deliver<'a>(stack: &mut Vec<Frame<'a>>, root: &mut Option<Value<'a>>, v: Value<'a>) {
    match stack.last_mut() {
        None => *root = Some(v),
        Some(Frame::Array {
            items,
            after_element,
        }) => {
            items.push(v);
            *after_element = true;
        },
        Some(Frame::Object {
            map,
            state,
            pending_key,
        }) => {
            let key = pending_key
                .take()
                .expect("member value without a pending key");
            // IndexMap::insert keeps the first position and the last value.
            map.insert(key, v);
            *state = ObjState::CommaOrClose;
        },
    }
}

/// Interpret `true` / `false` / `null` keywords by direct comparison at the
/// entry offset, skipping the token_end scan and slice. `None` for anything
/// else (numbers, longer or shorter misspellings), which falls back to
/// [`token_value`] for the exact canonical behavior.
fn keyword_value(bytes: &[u8], off: usize) -> Option<Value<'static>> {
    let (kw, v): (&[u8], Value<'_>) = match bytes[off] {
        b't' => (b"true", Value::Bool(true)),
        b'f' => (b"false", Value::Bool(false)),
        b'n' => (b"null", Value::Null),
        _ => return None,
    };
    let end = off + kw.len();
    // A keyword must be the whole token: a following token byte (ASCII class
    // or any high byte) means a longer run like `truex` and takes the slow
    // path, which reports it exactly as before.
    let whole = match bytes.get(end) {
        None => true,
        Some(&b) if b < 0x80 => CLASSES[b as usize] & class::TOKEN == 0,
        Some(_) => false,
    };
    if whole && bytes.get(off..end) == Some(kw) {
        Some(v)
    } else {
        None
    }
}

/// Interpret a bare token in value position.
pub(crate) fn token_value(tok: &[u8], off: usize) -> Result<Value<'static>> {
    match tok {
        b"true" => Ok(Value::Bool(true)),
        b"false" => Ok(Value::Bool(false)),
        b"null" => Ok(Value::Null),
        b"Infinity" => Ok(Value::F64(f64::INFINITY)),
        b"NaN" => Ok(Value::F64(f64::NAN)),
        _ => match tok.first() {
            Some(&(b'+' | b'-' | b'.' | b'0'..=b'9')) => number::parse(tok, off),
            // Source-exact keyword matching: escaped spellings do not count.
            _ => Err(syntax("unknown literal", off)),
        },
    }
}

/// Parse the bare token `bytes[off..end]` as an unquoted object key. Starts
/// with `[A-Za-z_$]`, a non-ASCII code point or a `\uXXXX` escape; continues
/// with identifier bytes, non-ASCII code points and more `\uXXXX` escapes.
pub(crate) fn parse_identifier(bytes: &[u8], off: usize, end: usize) -> Result<Str<'_>> {
    // Bytes since the end of the previous escape are appended as one raw
    // segment when the next escape starts, and once more after the loop, so
    // no segment between or after escapes is dropped.
    let mut out: Option<String> = None;
    let mut last = off;
    let mut i = off;
    while i < end {
        let b = bytes[i];
        if b == b'\\' {
            // Only \u escapes are legal in identifiers.
            if bytes.get(i + 1) != Some(&b'u') {
                return Err(Error::InvalidEscape { offset: i });
            }
            let (c, next) = decode_unicode_escape(bytes, i)?;
            // `end - off` is a capacity upper bound: a 6-byte escape decodes
            // to at most 3 UTF-8 bytes (a surrogate pair: 12 to 4).
            let o = out.get_or_insert_with(|| String::with_capacity(end - off));
            o.push_str(str_slice(bytes, last, i));
            o.push(c);
            last = next;
            i = next;
        } else if b < 0x80 {
            if CLASSES[b as usize] & class::IDENT == 0 {
                return Err(Error::UnexpectedChar { byte: b, offset: i });
            }
            i += 1;
        } else {
            let (cp, len) = decode_seq(bytes, i).ok_or(Error::InvalidUtf8 { offset: i })?;
            // Unicode whitespace cannot occur inside a token (stage 1 stops
            // there); reject defensively rather than silently accept.
            if is_unicode_whitespace(cp) {
                return Err(Error::UnexpectedChar { byte: b, offset: i });
            }
            i += len;
        }
    }
    if let Some(o) = out.as_mut() {
        o.push_str(str_slice(bytes, last, end));
    }
    Ok(match out {
        Some(o) => Str::Dynamic(o),
        None => Str::Static(str_slice(bytes, off, end)),
    })
}

/// Parse the string whose opening quote is at `open`, borrowing directly
/// when the stage-1 span cursor's head matches `open`: a clean span's
/// content `bytes[open + 1..close]` is exactly what [`parse_string`] would
/// return (no backslash means no escape processing and no line-continuation
/// or terminator checks can fail), minus its re-scan of the content. The
/// cursor only advances on a match: spans are ordered by opening offset, so
/// the head belongs to some later string whenever it mismatches, and the
/// full parse covers everything else.
pub(crate) fn parse_string_spanned<'a>(
    bytes: &'a [u8],
    open: usize,
    spans: &mut &[Span],
) -> Result<Str<'a>> {
    if let Some(Span { open: o, close }) = spans.first()
        && *o as usize == open
    {
        *spans = &spans[1..];
        return Ok(Str::Static(str_slice(bytes, open + 1, *close as usize)));
    }
    parse_string(bytes, open).map(|(s, _)| s)
}

/// Parse the string whose opening quote is at `open`. Returns the string and
/// the position just past the closing quote.
pub(crate) fn parse_string(bytes: &[u8], open: usize) -> Result<(Str<'_>, usize)> {
    let quote = bytes[open];
    let mut i = open + 1;
    let mut u = Unescape::new(bytes, i);
    loop {
        // Locate the next quote, backslash or raw line terminator.
        let Some(hit) = find_interest(bytes, i, quote) else {
            return Err(Error::UnterminatedString { offset: open });
        };
        u.push_raw(i, hit);
        match bytes[hit] {
            b if b == quote => return Ok((u.finish(hit), hit + 1)),
            // Raw LF/CR are illegal; raw U+2028/U+2029 never trigger this
            // scan and stay in the content.
            b'\n' | b'\r' => {
                return Err(Error::UnexpectedChar {
                    byte: bytes[hit],
                    offset: hit,
                })
            },
            _ => i = process_escape(bytes, hit, &mut u)?,
        }
    }
}

/// Offset of the first byte of interest to the string scanner at or after
/// `from`: the closing `quote`, a backslash, or a raw line terminator.
/// Single SWAR pass over little-endian words, 16 bytes (two u64) per
/// iteration; on a hit the first half with a match is picked and located
/// within by its first set high bit. Class-table tail for the final partial
/// stretch. Measured faster than a memchr3+memchr pair on string-heavy
/// corpora: the CR scan must stay bounded by the first quote/backslash/LF
/// hit (an unbounded one made the parse quadratic), so the memchr form pays
/// two passes and two call setups per string.
fn find_interest(bytes: &[u8], from: usize, quote: u8) -> Option<usize> {
    const ONES: u64 = 0x0101_0101_0101_0101;
    const HIBITS: u64 = 0x8080_8080_8080_8080;
    const BS: u64 = 0x5c5c_5c5c_5c5c_5c5c;
    const LF: u64 = 0x0a0a_0a0a_0a0a_0a0a;
    const CR: u64 = 0x0d0d_0d0d_0d0d_0d0d;
    /// Nonzero in the high bit of every byte that is zero in `x`.
    const fn has_zero(x: u64) -> u64 {
        x.wrapping_sub(ONES) & !x & HIBITS
    }
    /// Interest mask of one word: quote, backslash, LF or CR bytes.
    const fn interest_mask(w: u64, q: u64) -> u64 {
        has_zero(w ^ q) | has_zero(w ^ BS) | has_zero(w ^ LF) | has_zero(w ^ CR)
    }
    let q = ONES * u64::from(quote);
    let mut i = from;
    while i + 16 <= bytes.len() {
        // from_le_bytes pins the word order, so trailing_zeros/8 is the first
        // matching byte regardless of the host endianness.
        let m0 = interest_mask(u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap()), q);
        let m1 = interest_mask(
            u64::from_le_bytes(bytes[i + 8..i + 16].try_into().unwrap()),
            q,
        );
        if (m0 | m1) != 0 {
            // First half with a hit, then the first matching byte in it.
            let (m, k) = if m0 != 0 {
                (m0, 0)
            } else {
                (m1, 8)
            };
            return Some(i + k + (m.trailing_zeros() / 8) as usize);
        }
        i += 16;
    }
    let interest = class::BACKSLASH
        | class::LF
        | class::CR
        | if quote == b'"' {
            class::DQUOTE
        } else {
            class::SQUOTE
        };
    bytes[i..]
        .iter()
        .position(|&b| CLASSES[b as usize] & interest != 0)
        .map(|k| i + k)
}

/// Handle the escape whose backslash is at `esc`; returns the position just
/// past the escape.
fn process_escape(bytes: &[u8], esc: usize, u: &mut Unescape<'_>) -> Result<usize> {
    let n = *bytes
        .get(esc + 1)
        .ok_or(Error::InvalidEscape { offset: esc })?;
    match n {
        b'b' => u.push_char(esc, '\u{8}'),
        b'f' => u.push_char(esc, '\u{c}'),
        b'n' => u.push_char(esc, '\n'),
        b'r' => u.push_char(esc, '\r'),
        b't' => u.push_char(esc, '\t'),
        b'v' => u.push_char(esc, '\u{b}'),
        b'\'' | b'"' | b'\\' => u.push_char(esc, n as char),
        b'0' => {
            // \0 may not be followed by a decimal digit: \01 would be a
            // legacy octal escape, which JSON5 does not have.
            if bytes.get(esc + 2).is_some_and(u8::is_ascii_digit) {
                return Err(Error::InvalidEscape { offset: esc });
            }
            u.push_char(esc, '\0');
        },
        // Decimal digits are not NonEscapeCharacters in the JSON5 grammar
        // (the production excludes EscapeCharacter, which contains
        // DecimalDigit), so \1..\9 are errors, not the digit itself; json5-js
        // rejects them too.
        b'1'..=b'9' => return Err(Error::InvalidEscape { offset: esc }),
        b'x' => {
            let mut v = 0_u32;
            for k in 2..4 {
                let d = bytes
                    .get(esc + k)
                    .copied()
                    .ok_or(Error::InvalidEscape { offset: esc })?;
                v = v * 16
                    + u32::from(number::hex_val(d).ok_or(Error::InvalidEscape { offset: esc })?);
            }
            u.push_char(esc, char::from_u32(v).expect("value below 0xD800"));
            return Ok(esc + 4);
        },
        b'u' => {
            let (c, next) = decode_unicode_escape(bytes, esc)?;
            u.push_char(esc, c);
            return Ok(next);
        },
        // Line continuations: LF, CR (optionally CRLF), LS, PS produce
        // nothing, but they still force the owned buffer: the value no
        // longer equals a contiguous input slice.
        b'\n' => {
            u.split_at(esc);
            return Ok(esc + 2);
        },
        b'\r' => {
            u.split_at(esc);
            return Ok(if bytes.get(esc + 2) == Some(&b'\n') {
                esc + 3
            } else {
                esc + 2
            });
        },
        // Any other \c produces c.
        _ if n < 0x80 => u.push_char(esc, n as char),
        _ => {
            let (cp, len) =
                decode_seq(bytes, esc + 1).ok_or(Error::InvalidUtf8 { offset: esc + 1 })?;
            if cp == 0x2028 || cp == 0x2029 {
                u.split_at(esc);
            } else {
                u.push_char(
                    esc,
                    char::from_u32(cp).ok_or(Error::InvalidUtf8 { offset: esc + 1 })?,
                );
            }
            return Ok(esc + 1 + len);
        },
    }
    Ok(esc + 2)
}

/// Decode a `\uXXXX` escape whose backslash is at `esc`. A high surrogate
/// must be immediately followed by a `\uXXXX` low surrogate; the pair is
/// combined. Returns the char and the position just past the escape.
fn decode_unicode_escape(bytes: &[u8], esc: usize) -> Result<(char, usize)> {
    let invalid = || Error::InvalidUnicodeEscape { offset: esc };
    let hi = hex4(bytes, esc + 2).ok_or_else(invalid)?;
    if (0xd800..=0xdbff).contains(&hi) {
        if bytes.get(esc + 6) != Some(&b'\\') || bytes.get(esc + 7) != Some(&b'u') {
            return Err(invalid());
        }
        let lo = hex4(bytes, esc + 8).ok_or_else(invalid)?;
        if !(0xdc00..=0xdfff).contains(&lo) {
            return Err(invalid());
        }
        let c = 0x1_0000 + u32::from(hi - 0xd800) * 0x400 + u32::from(lo - 0xdc00);
        Ok((
            char::from_u32(c).expect("a surrogate pair decodes to a scalar value"),
            esc + 12,
        ))
    } else if (0xdc00..=0xdfff).contains(&hi) {
        Err(invalid())
    } else {
        Ok((
            char::from_u32(u32::from(hi)).expect("non-surrogate BMP code point"),
            esc + 6,
        ))
    }
}

/// Four hex digits at `at`, as a u16.
fn hex4(bytes: &[u8], at: usize) -> Option<u16> {
    let mut v: u16 = 0;
    for &b in bytes.get(at..at.checked_add(4)?)? {
        v = (v << 4) | u16::from(number::hex_val(b)?);
    }
    Some(v)
}

/// UTF-8 slice of `bytes[a..b]`.
///
/// Unchecked in release: every input reaching the parser is valid UTF-8
/// (`from_str` receives a `&str`, `from_slice` validates at entry), and both
/// ends of every slice taken here are either ASCII bytes or positions
/// produced by `decode_seq`, which consumes whole UTF-8 sequences.
fn str_slice(bytes: &[u8], a: usize, b: usize) -> &str {
    let raw = &bytes[a..b];
    debug_assert!(core::str::from_utf8(raw).is_ok());
    // SAFETY: valid UTF-8 by the contract above; the debug_assert pins it.
    unsafe { core::str::from_utf8_unchecked(raw) }
}

/// Incrementally unescaped string body. The owned buffer is materialized
/// lazily when the first escape appears; otherwise the input is borrowed.
struct Unescape<'a> {
    bytes: &'a [u8],
    start: usize,
    out: Option<String>,
}

impl<'a> Unescape<'a> {
    fn new(bytes: &'a [u8], start: usize) -> Self {
        Self {
            bytes,
            start,
            out: None,
        }
    }

    /// Owned buffer, materialized from the raw prefix `..upto` on first use.
    /// The prefix is valid UTF-8 by the parser's input contract (`from_str`
    /// and `from_slice` validate it), so the conversion is exact. It may
    /// contain raw LS/PS (legal string content); only raw LF/CR are
    /// impossible — the scanner stops at those before any escape.
    fn buf(&mut self, upto: usize) -> &mut String {
        let (bytes, start) = (self.bytes, self.start);
        self.out
            .get_or_insert_with(|| str_slice(bytes, start, upto).to_owned())
    }

    fn push_char(&mut self, upto: usize, c: char) {
        self.buf(upto).push(c);
    }

    /// Materialize the owned buffer without appending anything (used by line
    /// continuations, which contribute no character).
    fn split_at(&mut self, upto: usize) {
        self.buf(upto);
    }

    /// Append the raw bytes `[a..b)` once unescaping has started.
    fn push_raw(&mut self, a: usize, b: usize) {
        // Raw segments are only consumed once unescaping has started;
        // escape-free strings keep the borrowed fast path and never touch a
        // String here.
        if b > a && self.out.is_some() {
            let bytes = self.bytes;
            self.buf(a).push_str(str_slice(bytes, a, b));
        }
    }

    fn finish(self, close: usize) -> Str<'a> {
        match self.out {
            Some(o) => Str::Dynamic(o),
            None => Str::Static(str_slice(self.bytes, self.start, close)),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Error, Value,
        parser::parse,
        value::{MAX_DEPTH, Str},
    };

    fn ok(src: &str) -> Value<'_> {
        parse(src.as_bytes()).unwrap()
    }

    fn err(src: &str) -> Error {
        parse(src.as_bytes()).unwrap_err()
    }

    fn syn(message: &str, offset: usize) -> Error {
        Error::Syntax {
            message: message.into(),
            offset,
        }
    }

    #[test]
    fn top_level_scalars() {
        assert_eq!(ok("null"), Value::Null);
        assert_eq!(ok("true"), Value::Bool(true));
        assert_eq!(ok("  42  "), Value::U64(42));
        assert_eq!(ok("'-0'").as_str(), Some("-0"));
    }

    #[test]
    fn empty_input_is_unexpected_eof() {
        assert_eq!(err(""), Error::UnexpectedEof { offset: 0 });
        assert_eq!(err(" \t "), Error::UnexpectedEof { offset: 3 });
        assert_eq!(err("// just a comment"), Error::UnexpectedEof {
            offset: 17
        });
    }

    #[test]
    fn containers_with_trailing_commas() {
        assert_eq!(
            ok("[1, [2, 3,],]"),
            Value::Array(vec![
                Value::U64(1),
                Value::Array(vec![Value::U64(2), Value::U64(3)]),
            ])
        );
        let v = ok("{ a: 1, 'b': 2, }");
        let m = v.as_object().unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!(m.get_index(0).unwrap().0.as_str(), "a");
        assert_eq!(m.get("b"), Some(&Value::U64(2)));
    }

    #[test]
    fn duplicate_keys_keep_first_position_and_last_value() {
        let v = ok("{a: 1, b: 2, a: 3}");
        let m = v.as_object().unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!(m.get_index(0).unwrap().0.as_str(), "a");
        assert_eq!(m.get_index(1).unwrap().0.as_str(), "b");
        assert_eq!(m.get("a"), Some(&Value::U64(3)));
    }

    #[test]
    fn depth_limit() {
        let flat = format!("{}{}", "[".repeat(MAX_DEPTH), "]".repeat(MAX_DEPTH));
        assert!(parse(flat.as_bytes()).is_ok());
        let deep = format!("{}{}", "[".repeat(MAX_DEPTH + 1), "]".repeat(MAX_DEPTH + 1));
        assert_eq!(
            parse(deep.as_bytes()).unwrap_err(),
            Error::DepthLimitExceeded {
                limit: MAX_DEPTH,
                offset: MAX_DEPTH,
            }
        );
    }

    #[test]
    fn strings_borrow_without_escapes() {
        assert!(matches!(ok(r#""abc""#), Value::Str(Str::Static(_))));
        assert!(matches!(ok(r#""a\nb""#), Value::Str(Str::Dynamic(_))));
    }

    #[test]
    fn full_escape_set() {
        assert_eq!(
            ok(r#""\b\f\n\r\t\v\0\'\"\\\x41\u0042""#).as_str(),
            Some("\u{8}\u{c}\n\r\t\u{b}\0'\"\\A\u{42}")
        );
        // Any other \c produces c; \1..\9 are rejected (see process_escape).
        assert_eq!(ok(r#""\q\ ""#).as_str(), Some("q "));
        assert_eq!(ok(r#""😀""#).as_str(), Some("\u{1f600}"));
    }

    #[test]
    fn surrogate_pairs() {
        assert_eq!(ok(r#""😀""#).as_str(), Some("\u{1f600}"));
        assert_eq!(ok(r#""\uD83D\uDE00""#).as_str(), Some("\u{1f600}"));
        // Raw non-ASCII after a backslash is passed through unchanged.
        assert_eq!(ok("\"\\é\"").as_str(), Some("é"));
    }

    #[test]
    fn line_continuations() {
        assert_eq!(ok("\"a\\\nb\"").as_str(), Some("ab"));
        assert_eq!(ok("\"a\\\r\nb\"").as_str(), Some("ab"));
        assert_eq!(ok("\"a\\\rb\"").as_str(), Some("ab"));
        assert_eq!(ok("\"a\\\u{2028}b\"").as_str(), Some("ab"));
        assert_eq!(ok("\"a\\\u{2029}b\"").as_str(), Some("ab"));
    }

    #[test]
    fn raw_line_separators_are_legal_content() {
        assert_eq!(
            ok("\"a\u{2028}b\u{2029}\"").as_str(),
            Some("a\u{2028}b\u{2029}")
        );
        assert!(matches!(ok("\"a\u{2028}\""), Value::Str(Str::Static(_))));
    }

    #[test]
    fn raw_line_terminators_in_strings_are_rejected() {
        assert_eq!(err("\"a\nb\""), Error::UnexpectedChar {
            byte: b'\n',
            offset: 2,
        });
        assert_eq!(err("\"a\rb\""), Error::UnexpectedChar {
            byte: b'\r',
            offset: 2,
        });
    }

    #[test]
    fn invalid_escapes() {
        assert_eq!(err(r#""\01""#), Error::InvalidEscape { offset: 1 });
        assert_eq!(err(r#""\x4""#), Error::InvalidEscape { offset: 1 });
        assert_eq!(err(r#""\xZZ""#), Error::InvalidEscape { offset: 1 });
        assert_eq!(err(r#""\u12""#), Error::InvalidUnicodeEscape { offset: 1 });
        assert_eq!(err(r#""\uD800""#), Error::InvalidUnicodeEscape {
            offset: 1
        });
        assert_eq!(err(r#""\uD800\u0041""#), Error::InvalidUnicodeEscape {
            offset: 1
        });
        assert_eq!(err(r#""\uDC00""#), Error::InvalidUnicodeEscape {
            offset: 1
        });
        // \1..\9 are not NonEscapeCharacters; \0 followed by a non-digit is
        // fine.
        assert_eq!(err(r#""\1""#), Error::InvalidEscape { offset: 1 });
        assert_eq!(err(r#""\9""#), Error::InvalidEscape { offset: 1 });
        assert_eq!(ok(r#""\0 ""#).as_str(), Some("\0 "));
    }

    #[test]
    fn identifier_keys() {
        let src = "{ if: 1, $x: 2, _y: 3, é: 4, \\u0061: 5, a\\u0062: 6 }";
        let v = ok(src);
        let m = v.as_object().unwrap();
        assert_eq!(m.len(), 6);
        assert_eq!(m.get_index(0).unwrap().0.as_str(), "if");
        assert_eq!(m.get_index(3).unwrap().0.as_str(), "é");
        assert_eq!(m.get_index(4).unwrap().0.as_str(), "a");
        assert_eq!(m.get_index(5).unwrap().0.as_str(), "ab");
    }

    /// Bytes between, after and around identifier escapes must survive; a
    /// former bug dropped every ASCII byte after the first escape.
    #[test]
    fn identifier_bytes_around_escapes_are_kept() {
        let v = ok(r"{a\u0062c: 1, a\u0062b\u0063: 2}");
        let m = v.as_object().unwrap();
        assert_eq!(m.get_index(0).unwrap().0.as_str(), "abc");
        assert_eq!(m.get_index(1).unwrap().0.as_str(), "abbc");
        // Non-ASCII between and after escapes, and an escape-first key.
        let v = ok("{ é\\u0062x: 1, \\u0061é: 2 }");
        let m = v.as_object().unwrap();
        assert_eq!(m.get_index(0).unwrap().0.as_str(), "ébx");
        assert_eq!(m.get_index(1).unwrap().0.as_str(), "aé");
        // Quoted keys always worked; pinned as the contrast oracle.
        let v = ok(r#"{"a\u0062c": 1}"#);
        assert_eq!(
            v.as_object().unwrap().get_index(0).unwrap().0.as_str(),
            "abc"
        );
    }

    #[test]
    fn identifier_escape_errors() {
        assert_eq!(err(r"{a\x41: 1}"), Error::InvalidEscape { offset: 2 });
        assert_eq!(err(r"{a\uD800: 1}"), Error::InvalidUnicodeEscape {
            offset: 2
        });
        // `.`, `+`, `-` are token bytes but not identifier bytes.
        assert_eq!(err("{a-b: 1}"), Error::UnexpectedChar {
            byte: b'-',
            offset: 2,
        });
    }

    #[test]
    fn structural_errors() {
        assert_eq!(err("[1 2]"), syn("expected ',' or ']'", 3));
        assert_eq!(err("[1,,2]"), Error::UnexpectedChar {
            byte: b',',
            offset: 3,
        });
        assert_eq!(err("{a:1,,}"), Error::UnexpectedChar {
            byte: b',',
            offset: 5,
        });
        assert_eq!(err("[1}"), Error::UnexpectedChar {
            byte: b'}',
            offset: 2,
        });
        assert_eq!(err("{a:1]"), Error::UnexpectedChar {
            byte: b']',
            offset: 4,
        });
        assert_eq!(err("{a}"), syn("expected ':'", 2));
        assert_eq!(err("{a:}"), syn("expected value", 3));
        assert_eq!(err("[1"), Error::UnexpectedEof { offset: 2 });
        assert_eq!(err("{a:1"), Error::UnexpectedEof { offset: 4 });
        assert_eq!(err("1 2"), Error::TrailingCharacters { offset: 2 });
        assert_eq!(err("null,"), Error::TrailingCharacters { offset: 4 });
        assert_eq!(err("{1:2}"), syn("expected identifier or string key", 1));
        assert_eq!(err("tru"), syn("unknown literal", 0));
        assert_eq!(err(":"), Error::UnexpectedChar {
            byte: b':',
            offset: 0,
        });
        // A misplaced string reports the structural error, not its own parse
        // error: position is validated before the string body is parsed. The
        // escapes are invalid (`\1`), so the string would fail on its own if
        // it were parsed first.
        assert_eq!(err(r"{a 'b\1': 1}"), syn("expected ':'", 3));
        assert_eq!(err(r"[1 'b\1' 2]"), syn("expected ',' or ']'", 3));
    }

    #[test]
    fn separators_between_tokens() {
        assert_eq!(
            ok("[1\u{a0},\u{2028}2]"),
            Value::Array(vec![Value::U64(1), Value::U64(2)])
        );
        assert_eq!(
            ok("/* c */ [1, // x\n 2]"),
            Value::Array(vec![Value::U64(1), Value::U64(2)])
        );
    }

    #[test]
    fn string_interest_scan_matches_naive_scan() {
        fn naive(bytes: &[u8], from: usize, quote: u8) -> Option<usize> {
            bytes[from..]
                .iter()
                .position(|&b| b == quote || b == b'\\' || b == b'\n' || b == b'\r')
                .map(|k| from + k)
        }
        // Every needle at every alignment around the 8-byte SWAR words, for
        // both quote kinds.
        for pad in 0..=40_usize {
            for &needle in b"\"'\\\n\rx" {
                let mut bytes = vec![b'a'; pad];
                bytes.push(needle);
                bytes.extend_from_slice(b"tail");
                for &quote in b"\"'" {
                    assert_eq!(
                        super::find_interest(&bytes, 1, quote),
                        naive(&bytes, 1, quote)
                    );
                }
            }
        }
        // Needles inside the final partial word, and no needle at all.
        for pad in 0..=8 {
            let bytes = [vec![b'a'; pad], vec![b'"']].concat();
            assert_eq!(super::find_interest(&bytes, 0, b'"'), Some(pad));
        }
        assert_eq!(super::find_interest(b"abc", 0, b'"'), None);
        assert_eq!(super::find_interest(b"", 0, b'"'), None);
    }

    #[test]
    fn nested_mixed_document() {
        let v = ok("{ a: [1, {b: 'c',},], d: null, }");
        let m = v.as_object().unwrap();
        assert_eq!(m.len(), 2);
        let inner = m.get("a").and_then(Value::as_array).unwrap();
        let inner_map = inner[1].as_object().unwrap();
        let b = inner_map.get("b").unwrap();
        assert_eq!(b.as_str(), Some("c"));
        assert!(m.get("d").unwrap().is_null());
    }
}
