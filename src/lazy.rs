//! Lazy single-value access: locate one value by path without building a
//! DOM.
//!
//! [`get`] walks the raw input with an explicit container stack (no
//! recursion; depth-capped like the DOM parser) and returns a
//! [`LazyValue`] holding the value's byte range. Cost is proportional to
//! the bytes walked, not to the document: skipped members are located
//! structurally (string extents escape-aware, token extents by class), and
//! only the accessed value and the keys compared against the path get
//! their full grammar validated. Accessors on the returned value parse on
//! demand and borrow from the input wherever the DOM parser would.

use crate::{
    Error, Result,
    parser::{
        stage1::{decode_seq, is_unicode_whitespace, skip_comment, skip_string, token_end},
        stage2,
    },
    value::{MAX_DEPTH, Str, Value},
};

/// Syntax-error constructor. Cold: keeps the error paths out of the hot
/// loop's instruction stream.
#[cold]
fn syntax(message: &str, offset: usize) -> Error {
    Error::Syntax {
        message: message.into(),
        offset,
    }
}

/// Object member state machine (mirrors stage 2).
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

/// An open container on the walk stack.
enum Open {
    Array { after_element: bool },
    Object { state: ObjState },
}

/// Position of the first decision point (non-separator byte) at or after
/// `from`: ASCII whitespace, comments and Unicode whitespace are skipped.
fn skip_seps(bytes: &[u8], mut i: usize) -> Result<usize> {
    loop {
        match bytes.get(i) {
            None | Some(&b'"' | &b'\'' | &b'{' | &b'}' | &b'[' | &b']' | &b',' | &b':') => {
                return Ok(i);
            },
            Some(b'\t'..=b'\r' | b' ') => i += 1,
            Some(b'/') => i = skip_comment(bytes, i)?,
            Some(&b) if b >= 0x80 => match decode_seq(bytes, i) {
                // Unicode whitespace skips; every other non-ASCII sequence is
                // an ordinary token byte and stops the scan.
                Some((cp, len)) if is_unicode_whitespace(cp) => i += len,
                _ => return Ok(i),
            },
            // Token bytes stop the scan; malformed UTF-8 cannot occur (the
            // input is a `&str`).
            _ => return Ok(i),
        }
    }
}

/// Verify the innermost container accepts the start of a value here
/// (mirrors stage 2).
fn check_value_position(stack: &[Open], off: usize) -> Result<()> {
    match stack.last() {
        None
        | Some(
            Open::Array {
                after_element: false,
            }
            | Open::Object {
                state: ObjState::Value,
            },
        ) => Ok(()),
        Some(Open::Array { .. }) => Err(syntax("expected ',' or ']'", off)),
        Some(Open::Object {
            state: ObjState::KeyOrClose,
        }) => Err(syntax("expected identifier or string key", off)),
        Some(Open::Object {
            state: ObjState::Colon,
        }) => Err(syntax("expected ':'", off)),
        Some(Open::Object {
            state: ObjState::CommaOrClose,
        }) => Err(syntax("expected ',' or '}'", off)),
    }
}

/// Record that a complete value was delivered to the innermost container
/// (mirrors stage 2's `deliver` minus the DOM work).
fn value_delivered(stack: &mut [Open]) {
    match stack.last_mut() {
        Some(Open::Array { after_element }) => *after_element = true,
        Some(Open::Object { state }) => *state = ObjState::CommaOrClose,
        None => {},
    }
}

/// Start and end offsets of the value at or after `from` (leading
/// separators skipped), with its whole structure validated: container
/// pairing and member state, string extents (escape-aware: a backslash
/// consumes the byte after it), and separator placement. `end` is just
/// past the value's last byte. Escape grammar and token spelling are not
/// validated here; the accessors validate whatever they return.
fn value_range(bytes: &[u8], from: usize) -> Result<(usize, usize)> {
    let mut stack: Vec<Open> = Vec::with_capacity(16);
    let mut i = skip_seps(bytes, from)?;
    let start = i;
    if i >= bytes.len() {
        return Err(Error::UnexpectedEof {
            offset: bytes.len(),
        });
    }
    loop {
        // Every arm leaves `i` just past the construct it consumed (or
        // returns); the tail skips the separators after it.
        let b = bytes[i];
        match b {
            b'{' | b'[' => {
                check_value_position(&stack, i)?;
                if stack.len() >= MAX_DEPTH {
                    return Err(Error::DepthLimitExceeded {
                        limit: MAX_DEPTH,
                        offset: i,
                    });
                }
                stack.push(if b == b'[' {
                    Open::Array {
                        after_element: false,
                    }
                } else {
                    Open::Object {
                        state: ObjState::KeyOrClose,
                    }
                });
                i += 1;
            },
            b'}' | b']' => {
                let closable = match stack.last() {
                    Some(Open::Object {
                        state: ObjState::KeyOrClose | ObjState::CommaOrClose,
                    }) => b == b'}',
                    Some(Open::Array { .. }) => b == b']',
                    _ => false,
                };
                if !closable {
                    return Err(match stack.last() {
                        Some(Open::Object {
                            state: ObjState::Colon,
                        }) => syntax("expected ':'", i),
                        Some(Open::Object {
                            state: ObjState::Value,
                        }) => syntax("expected value", i),
                        _ => Error::UnexpectedChar { byte: b, offset: i },
                    });
                }
                stack.pop();
                i += 1;
                if stack.is_empty() {
                    return Ok((start, i));
                }
                value_delivered(&mut stack);
            },
            b',' => {
                match stack.last_mut() {
                    Some(Open::Array { after_element }) if *after_element => *after_element = false,
                    Some(Open::Object { state }) if *state == ObjState::CommaOrClose => {
                        *state = ObjState::KeyOrClose;
                    },
                    Some(Open::Object {
                        state: ObjState::Colon,
                    }) => return Err(syntax("expected ':'", i)),
                    Some(Open::Object {
                        state: ObjState::Value,
                    }) => return Err(syntax("expected value", i)),
                    _ => {
                        return Err(Error::UnexpectedChar {
                            byte: b',',
                            offset: i,
                        });
                    },
                }
                i += 1;
            },
            b':' => {
                match stack.last_mut() {
                    Some(Open::Object { state }) if *state == ObjState::Colon => {
                        *state = ObjState::Value;
                    },
                    _ => {
                        return Err(Error::UnexpectedChar {
                            byte: b':',
                            offset: i,
                        });
                    },
                }
                i += 1;
            },
            b'"' | b'\'' => {
                let is_key = matches!(
                    stack.last(),
                    Some(Open::Object {
                        state: ObjState::KeyOrClose
                    })
                );
                if !is_key {
                    check_value_position(&stack, i)?;
                }
                let (end, _) = skip_string(bytes, i)?;
                if is_key {
                    if let Some(Open::Object { state }) = stack.last_mut() {
                        *state = ObjState::Colon;
                    }
                } else {
                    if stack.is_empty() {
                        return Ok((start, end));
                    }
                    value_delivered(&mut stack);
                }
                i = end;
            },
            _ => {
                // A bare token: consumed as one run. Keys accept only
                // identifier starts; every value position accepts any token
                // run (spelling is validated by the accessors).
                let end = token_end(bytes, i);
                if let Some(Open::Object { state }) = stack.last_mut()
                    && *state == ObjState::KeyOrClose
                {
                    if matches!(b, b'+' | b'-' | b'.' | b'0'..=b'9') {
                        return Err(syntax("expected identifier or string key", i));
                    }
                    *state = ObjState::Colon;
                } else {
                    check_value_position(&stack, i)?;
                    if stack.is_empty() {
                        return Ok((start, end));
                    }
                    value_delivered(&mut stack);
                }
                i = end;
            },
        }
        i = skip_seps(bytes, i)?;
        if i >= bytes.len() {
            return Err(Error::UnexpectedEof {
                offset: bytes.len(),
            });
        }
    }
}

/// A value located by [`get`]: its byte range in the original input plus
/// on-demand accessors, never materializing a DOM.
///
/// Structure was validated by the walk that located the value; the value's
/// own grammar (escape sequences, number or keyword spelling) is validated
/// by the accessor that reads it.
///
/// ```
/// let value = simd_json5::get("{ a: 1, // comment\n  b: [2, 3,] }", &["b", "0"]).unwrap();
/// assert_eq!(value.unwrap().as_i64().unwrap(), Some(2));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LazyValue<'a> {
    bytes: &'a [u8],
    start: usize,
    end: usize,
}

impl<'a> LazyValue<'a> {
    /// Raw JSON5 text of the value, exactly as it appears in the input.
    #[must_use]
    pub fn raw(&self) -> &'a str {
        stage2::str_slice(self.bytes, self.start, self.end)
    }

    #[must_use]
    pub fn is_null(&self) -> bool {
        self.raw() == "null"
    }

    #[must_use]
    pub fn is_bool(&self) -> bool {
        matches!(self.raw(), "true" | "false")
    }

    #[must_use]
    pub fn is_number(&self) -> bool {
        // Every bare token is a number or a keyword, and the only keywords
        // starting with `N`/`I` are `NaN`/`Infinity` — themselves numbers.
        matches!(
            self.bytes[self.start],
            b'+' | b'-' | b'.' | b'0'..=b'9' | b'N' | b'I'
        )
    }

    #[must_use]
    pub fn is_str(&self) -> bool {
        matches!(self.bytes[self.start], b'"' | b'\'')
    }

    #[must_use]
    pub fn is_array(&self) -> bool {
        self.bytes[self.start] == b'['
    }

    #[must_use]
    pub fn is_object(&self) -> bool {
        self.bytes[self.start] == b'{'
    }

    /// The value as a bool, or `None` when it is not a bool. Errors when
    /// the token is not a valid literal.
    pub fn as_bool(&self) -> Result<Option<bool>> {
        match self.scalar()? {
            Some(Value::Bool(b)) => Ok(Some(b)),
            _ => Ok(None),
        }
    }

    /// The value as a string, or `None` when it is not a string. The
    /// string's escapes are fully validated and unescaped; escape-free
    /// strings borrow from the input.
    pub fn as_str(&self) -> Result<Option<Str<'a>>> {
        match self.scalar()? {
            Some(Value::Str(s)) => Ok(Some(s)),
            _ => Ok(None),
        }
    }

    /// The value as `i64`, or `None` when it is not a representable
    /// integer. Errors when a number token is malformed.
    pub fn as_i64(&self) -> Result<Option<i64>> {
        Ok(self.scalar()?.as_ref().and_then(Value::as_i64))
    }

    /// The value as `u64`, or `None` when it is not a representable
    /// integer. Errors when a number token is malformed.
    pub fn as_u64(&self) -> Result<Option<u64>> {
        Ok(self.scalar()?.as_ref().and_then(Value::as_u64))
    }

    /// The value as `f64` (integers widen), or `None` when it is not a
    /// number. Errors when a number token is malformed.
    pub fn as_f64(&self) -> Result<Option<f64>> {
        Ok(self.scalar()?.as_ref().and_then(Value::as_f64))
    }

    /// Locate a member or element one level below this value: an object
    /// member by key, or an array element when `key` is a decimal index.
    /// Duplicate keys follow the DOM: the last value wins.
    pub fn get(&self, key: &str) -> Result<Option<LazyValue<'a>>> {
        descend(self.bytes, self.start, self.end, key)
    }

    /// Parse the whole value into a DOM [`Value`], borrowing from the same
    /// input.
    pub fn to_value(&self) -> Result<Value<'a>> {
        crate::parser::parse(self.raw().as_bytes())
    }

    /// The fully validated scalar value, or `None` for containers. Strings
    /// and tokens go through the stage-2 parsers, so the validation and
    /// borrowed/owned split match the DOM path exactly.
    fn scalar(&self) -> Result<Option<Value<'a>>> {
        match self.bytes[self.start] {
            b'"' | b'\'' => Ok(Some(Value::Str(
                stage2::parse_string(self.bytes, self.start)?.0,
            ))),
            b'{' | b'[' => Ok(None),
            _ => {
                let end = token_end(self.bytes, self.start);
                stage2::token_value(&self.bytes[self.start..end], self.start).map(Some)
            },
        }
    }
}

impl core::fmt::Display for LazyValue<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.raw())
    }
}

/// Locate the value at `path` in a JSON5 document without building a DOM.
///
/// Each path segment selects an object member by key (quoted or unquoted,
/// escapes honored), or an array element when the segment is a decimal
/// index and the container is an array. `Ok(None)` means the path does not
/// exist. Duplicate keys follow the DOM: the last value wins.
///
/// The walk validates the document's structure — container pairing and
/// member state, string extents (a backslash consumes the byte after it),
/// separator placement, nesting depth — and stops there: members that are
/// skipped over are not grammar-checked (an invalid escape inside a
/// non-matching string is not reported). The returned value validates its
/// own grammar through its accessors, exactly like the DOM parser.
///
/// ```
/// let src = r#"{ "artist": { name: 'Helen', tags: ["a", "b",] } }"#;
/// let name = simd_json5::get(src, &["artist", "name"]).unwrap();
/// assert_eq!(name.unwrap().as_str().unwrap().as_deref(), Some("Helen"));
/// let tag = simd_json5::get(src, &["artist", "tags", "1"]).unwrap();
/// assert_eq!(tag.unwrap().as_str().unwrap().as_deref(), Some("b"));
/// assert!(
///     simd_json5::get(src, &["artist", "missing"])
///         .unwrap()
///         .is_none()
/// );
/// ```
pub fn get<'a>(src: &'a str, path: &[&str]) -> Result<Option<LazyValue<'a>>> {
    let bytes = src.as_bytes();
    let (mut start, mut end) = value_range(bytes, 0)?;
    let rest = skip_seps(bytes, end)?;
    if rest != bytes.len() {
        return Err(Error::TrailingCharacters { offset: rest });
    }
    for &segment in path {
        let Some(child) = descend(bytes, start, end, segment)? else {
            return Ok(None);
        };
        start = child.start;
        end = child.end;
    }
    Ok(Some(LazyValue { bytes, start, end }))
}

/// Descend one path segment into the container at `(start, end)`.
fn descend<'a>(
    bytes: &'a [u8],
    start: usize,
    end: usize,
    segment: &str,
) -> Result<Option<LazyValue<'a>>> {
    let found = match bytes[start] {
        b'{' => object_member(bytes, start, end, segment)?,
        b'[' if !segment.is_empty() && segment.bytes().all(|b| b.is_ascii_digit()) => {
            // Saturating parse: an index past `usize::MAX` cannot exist in a
            // document bounded by the input length anyway.
            let index = segment.parse::<usize>().unwrap_or(usize::MAX);
            array_element(bytes, start, end, index)?
        },
        _ => None,
    };
    Ok(found.map(|(start, end)| LazyValue { bytes, start, end }))
}

/// Iterate the members of the object at `(start, end)` and return the
/// range of the member matching `key` — the last match, mirroring the
/// DOM's duplicate-key semantics. Keys are fully validated and unescaped
/// (matching keys are grammar-checked, exactly like the DOM path).
fn object_member(
    bytes: &[u8],
    start: usize,
    end: usize,
    key: &str,
) -> Result<Option<(usize, usize)>> {
    let body_end = end - 1; // at the closing `}`
    let mut i = skip_seps(bytes, start + 1)?;
    let mut found = None;
    while i < body_end {
        // The key: stage-2 parsing gives the exact DOM spelling semantics.
        let (key_str, key_end) = match bytes[i] {
            b'"' | b'\'' => {
                let (s, e) = stage2::parse_string(bytes, i)?;
                (s, e)
            },
            _ => {
                if matches!(bytes[i], b'+' | b'-' | b'.' | b'0'..=b'9') {
                    return Err(syntax("expected identifier or string key", i));
                }
                let e = token_end(bytes, i);
                (stage2::parse_identifier(bytes, i, e)?, e)
            },
        };
        i = skip_seps(bytes, key_end)?;
        if bytes.get(i) != Some(&b':') {
            return Err(syntax("expected ':'", i));
        }
        i = skip_seps(bytes, i + 1)?;
        let (vstart, vend) = value_range(bytes, i)?;
        if key_str.as_str() == key {
            found = Some((vstart, vend));
        }
        i = skip_seps(bytes, vend)?;
        // Validated by the walk: `,` continues, `}` (possibly after a
        // trailing comma) ends the member list.
        match bytes.get(i) {
            Some(b',') => i = skip_seps(bytes, i + 1)?,
            Some(b'}') | None => break,
            _ => return Err(syntax("expected ',' or '}'", i)),
        }
    }
    Ok(found)
}

/// Locate the element at `index` of the array at `(start, end)`.
fn array_element(
    bytes: &[u8],
    start: usize,
    end: usize,
    index: usize,
) -> Result<Option<(usize, usize)>> {
    let body_end = end - 1; // at the closing `]`
    let mut i = skip_seps(bytes, start + 1)?;
    let mut n = 0;
    while i < body_end {
        let (vstart, vend) = value_range(bytes, i)?;
        if n == index {
            return Ok(Some((vstart, vend)));
        }
        n += 1;
        i = skip_seps(bytes, vend)?;
        // Validated by the walk: `,` continues, `]` (possibly after a
        // trailing comma) ends the element list.
        match bytes.get(i) {
            Some(b',') => i = skip_seps(bytes, i + 1)?,
            Some(b']') | None => break,
            _ => return Err(syntax("expected ',' or ']'", i)),
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_ok<'a>(src: &'a str, path: &[&str]) -> Option<LazyValue<'a>> {
        get(src, path).unwrap()
    }

    fn get_err(src: &str, path: &[&str]) -> Error {
        get(src, path).unwrap_err()
    }

    /// The lazy path must agree with the DOM on every valid document: raw
    /// kinds, typed accessors and `to_value` all match a full parse.
    #[test]
    fn agrees_with_dom() {
        let src = r#"{
            // comment
            a: 1, b: 'x', "c d": [true, null, -2.5e3,], 'e': { f: "\u0041\n" },
        }"#;
        let dom = crate::from_str(src).unwrap();
        assert_eq!(
            get_ok(src, &["a"]).and_then(|v| v.as_i64().unwrap()),
            dom.get("a").and_then(Value::as_i64)
        );
        assert_eq!(
            get_ok(src, &["b"])
                .and_then(|v| v.as_str().unwrap())
                .as_deref(),
            dom.get("b").and_then(|v| v.as_str())
        );
        assert_eq!(
            get_ok(src, &["c d", "3"]).and_then(|v| v.as_f64().unwrap()),
            dom.get("c d").unwrap().get_index(3).and_then(Value::as_f64)
        );
        assert_eq!(
            get_ok(src, &["e", "f"])
                .and_then(|v| v.as_str().unwrap())
                .as_deref(),
            dom.get("e").unwrap().get("f").and_then(|v| v.as_str())
        );
        assert_eq!(get_ok(src, &[]).unwrap().to_value().unwrap(), dom);
    }

    #[test]
    fn paths_and_misses() {
        let src = "{a: [1, {b: 2,},],}";
        assert_eq!(get_ok(src, &["a", "1", "b"]).unwrap().raw(), "2");
        assert_eq!(get_ok(src, &["a", "0"]).unwrap().raw(), "1");
        assert!(get_ok(src, &["a", "2"]).is_none());
        assert!(get_ok(src, &["b"]).is_none());
        // Array indices are decimal digits only.
        assert!(get_ok(src, &["a", "x"]).is_none());
        // A numeric key on an array, a string index on an object.
        assert!(get_ok(src, &["0"]).is_none());
        assert!(get_ok(src, &["a", "0", "b"]).is_none());
        // The root itself.
        assert_eq!(get_ok(src, &[]).unwrap().raw(), src);
    }

    /// Duplicate keys keep the last value, like the DOM.
    #[test]
    fn duplicate_keys_take_last_value() {
        let src = "{a: 1, b: 2, a: 3}";
        assert_eq!(get_ok(src, &["a"]).unwrap().raw(), "3");
        let dom = crate::from_str(src).unwrap();
        assert_eq!(
            get_ok(src, &["a"]).unwrap().to_value().unwrap(),
            *dom.get("a").unwrap()
        );
    }

    /// Leading zeros in an index segment still address the element, and an
    /// out-of-range index is a miss, not an error.
    #[test]
    fn array_index_segments() {
        assert_eq!(get_ok("[10, 20]", &["0"]).unwrap().raw(), "10");
        assert_eq!(get_ok("[10, 20]", &["01"]).unwrap().raw(), "20");
        assert!(get_ok("[10, 20]", &["2"]).is_none());
        assert!(get_ok("[10, 20]", &["999999999999999999999999"]).is_none());
    }

    #[test]
    fn scalar_root() {
        assert_eq!(get_ok("42", &[]).unwrap().as_u64().unwrap(), Some(42));
        assert_eq!(get_ok(" -7 ", &[]).unwrap().as_i64().unwrap(), Some(-7));
        assert!(get_ok("true", &[]).unwrap().as_bool().unwrap().unwrap());
        assert!(get_ok("null", &[]).unwrap().is_null());
        assert_eq!(get_ok("'-0'", &[]).unwrap().raw(), "'-0'");
        assert_eq!(
            get_ok("'a\\nb'", &[])
                .unwrap()
                .as_str()
                .unwrap()
                .unwrap()
                .as_str(),
            "a\nb"
        );
        // NaN/Infinity are numbers: bare tokens the number predicate must
        // classify (found by the lazy_parity fuzz target).
        assert!(get_ok("NaN", &[]).unwrap().is_number());
        assert!(get_ok("-Infinity", &[]).unwrap().is_number());
        assert!(get_ok("[NaN, 1, Infinity]", &["2"]).unwrap().is_number());
        assert!(get_ok("NaN", &[]).unwrap().as_f64().unwrap().unwrap().is_nan());
        assert_eq!(
            get_ok("[Infinity]", &["0"]).unwrap().as_f64().unwrap().unwrap(),
            f64::INFINITY
        );
        // Keywords and strings are not numbers.
        assert!(!get_ok("null", &[]).unwrap().is_number());
        assert!(!get_ok("true", &[]).unwrap().is_number());
        assert!(!get_ok("'NaN'", &[]).unwrap().is_number());
    }

    /// Structure errors mirror the DOM parser's.
    #[test]
    fn structure_errors() {
        use crate::Error::*;
        let syn = |m: &str, o: usize| Syntax {
            message: m.into(),
            offset: o,
        };
        assert_eq!(get_err("", &[]), UnexpectedEof { offset: 0 });
        assert_eq!(get_err(" // c", &[]), UnexpectedEof { offset: 5 });
        assert_eq!(get_err("[1", &[]), UnexpectedEof { offset: 2 });
        assert_eq!(get_err("{a:1", &[]), UnexpectedEof { offset: 4 });
        assert_eq!(get_err("{a:", &[]), UnexpectedEof { offset: 3 });
        assert_eq!(get_err("null,", &[]), TrailingCharacters { offset: 4 });
        assert_eq!(get_err("1 2", &[]), TrailingCharacters { offset: 2 });
        assert_eq!(get_err("{a}", &[]), syn("expected ':'", 2));
        assert_eq!(get_err("{a:}", &[]), syn("expected value", 3));
        assert_eq!(get_err("[1 2]", &[]), syn("expected ',' or ']'", 3));
        assert_eq!(get_err("{a:1 2}", &[]), syn("expected ',' or '}'", 5));
        assert_eq!(get_err("{a:1,,}", &[]), UnexpectedChar {
            byte: b',',
            offset: 5
        });
        assert_eq!(get_err("[1}]", &[]), UnexpectedChar {
            byte: b'}',
            offset: 2
        });
        // Descending into a scalar is a miss, not an error.
        assert!(get_ok("[1]", &["0", "x"]).is_none());
        assert_eq!(
            get_err("{1:2}", &[]),
            syn("expected identifier or string key", 1)
        );
        assert_eq!(get_err("'abc", &[]), UnterminatedString { offset: 0 });
        assert_eq!(get_err("/* x", &[]), UnterminatedComment { offset: 0 });
        assert_eq!(get_err("/", &[]), UnexpectedChar {
            byte: b'/',
            offset: 0
        });
    }

    /// Deep nesting hits the depth limit instead of overflowing the stack
    /// (the walk is iterative; sonic-rs's recursive `skip_one` overflows).
    #[test]
    fn depth_limit_is_reported_not_blown() {
        let deep = format!("{}{}", "[".repeat(MAX_DEPTH), "]".repeat(MAX_DEPTH));
        assert!(get(&deep, &[]).is_ok());
        let deeper = format!("{}{}", "[".repeat(MAX_DEPTH + 1), "]".repeat(MAX_DEPTH + 1));
        assert_eq!(get_err(&deeper, &[]), Error::DepthLimitExceeded {
            limit: MAX_DEPTH,
            offset: MAX_DEPTH,
        });
    }

    /// Skipped members are located structurally only: an invalid escape in
    /// a non-matching string is not reported, while the accessed value's
    /// grammar is validated by its accessors.
    #[test]
    fn skipped_members_skip_grammar() {
        let src = r#"{ bad: "\x4", good: 1 }"#;
        assert_eq!(get_ok(src, &["good"]).unwrap().raw(), "1");
        // The accessed value is validated when read.
        assert!(matches!(
            get_ok(src, &["bad"]).unwrap().as_str(),
            Err(Error::InvalidEscape { offset: 8 })
        ));
        // Keys are compared with full validation: an invalid key escape is
        // an error even on a miss.
        assert!(matches!(
            get_err(r"{ 'a\x4': 1 }", &["z"]),
            Error::InvalidEscape { offset: 4 }
        ));
    }

    /// Raw strings borrow from the input; escapes on the accessed value
    /// materialize exactly like the DOM path.
    #[test]
    fn string_access_matches_dom() {
        let src = r#"{ clean: 'abc', dirty: "a\nb" }"#;
        assert!(matches!(
            get_ok(src, &["clean"]).unwrap().as_str().unwrap(),
            Some(Str::Static(_))
        ));
        assert!(matches!(
            get_ok(src, &["dirty"]).unwrap().as_str().unwrap(),
            Some(Str::Dynamic(_))
        ));
        // Line continuations and comments inside the walk.
        let messy = "{ a: 'x\\\n y', /* c */ b: 1 }";
        assert_eq!(
            get_ok(messy, &["a"])
                .unwrap()
                .as_str()
                .unwrap()
                .unwrap()
                .as_str(),
            "x y"
        );
    }

    /// The whole range of a located value is exact: comments and
    /// whitespace around it stay outside, everything inside stays in.
    #[test]
    fn raw_ranges_are_exact() {
        let src = "{ a: [1, 2, /*c*/ 3], b: 'x' }";
        assert_eq!(get_ok(src, &["a"]).unwrap().raw(), "[1, 2, /*c*/ 3]");
        assert_eq!(get_ok(src, &["b"]).unwrap().raw(), "'x'");
        assert_eq!(
            get_ok(src, &["a"]).unwrap().to_value().unwrap(),
            crate::from_str("[1,2,3]").unwrap()
        );
    }

    /// Identifiers with `\uXXXX` escapes match their decoded spelling.
    #[test]
    fn escaped_identifier_keys() {
        let src = r"{ a\u0062c: 1, plain: 2 }";
        assert_eq!(get_ok(src, &["abc"]).unwrap().raw(), "1");
        assert_eq!(get_ok(src, &["plain"]).unwrap().raw(), "2");
        assert!(get_ok(src, &["a"]).is_none());
    }
}
