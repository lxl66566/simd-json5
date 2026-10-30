//! Typed (serde) deserialization: a zero-copy walk over the stage-1 index.
//!
//! [`from_str`] runs stage 1 (`simd::build_index`) and then drives serde
//! visitors across the index entries, mirroring the stage-2 state machine
//! (identical grammar, error variants and byte offsets, including
//! [`crate::value::MAX_DEPTH`]). Strings that need no unescaping reach
//! visitors as borrowed `&str`; escape-containing strings arrive owned. Type
//! mismatches surface as [`Error::Custom`] through the [`serde::de::Error`]
//! impl; [`from_str`] re-reports those with the read position as
//! [`Error::Syntax`], so every typed-path error carries an offset like the
//! DOM path's errors do.
//!
//! The visitor walk recurses through serde frames, so documents near
//! [`crate::value::MAX_DEPTH`] need a large stack — about 32 MiB reaches the
//! limit (the DOM path uses an explicit stack and has no such requirement).
//!
//! ```
//! #[derive(serde::Deserialize)]
//! struct Point {
//!     x: i64,
//!     y: i64,
//! }
//!
//! let p: Point = simd_json5::de::from_str("{ x: 1, y: /* mid */ 2, }").unwrap();
//! assert_eq!((p.x, p.y), (1, 2));
//! ```

use std::cell::OnceCell;

use serde::de::{self, DeserializeSeed, Unexpected, Visitor};

use crate::{
    Error, Result,
    parser::{
        Counts, Stage1Buffers, container_counts,
        number::{self, Num, Num128, NumF32},
        simd,
        stage1::{CLASSES, Span, class, token_end},
        stage2,
    },
    value::{MAX_DEPTH, Map, Str, Value},
};

/// Deserialize `T` from a JSON5 document, borrowing string data from `s`
/// where possible.
pub fn from_str<'de, T>(s: &'de str) -> Result<T>
where
    T: serde::Deserialize<'de>,
{
    let mut bufs = Stage1Buffers::take(s.len());
    simd::build_index(s.as_bytes(), &mut bufs.index, &mut bufs.spans)?;
    let mut de = Deserializer::new(s.as_bytes(), &bufs.index, &bufs.spans);
    let value = T::deserialize(&mut de).map_err(|e| locate(e, &de))?;
    de.end()?;
    Ok(value)
}

/// Attach the current read position to serde visitor errors, which arrive as
/// position-less [`Error::Custom`] (`invalid type`, `missing field`, ...).
/// Re-reported as [`Error::Syntax`] they match the always-positioned errors
/// of the DOM path, like serde_json's located type errors. Errors that
/// already carry an offset pass through unchanged.
///
/// The position is the read cursor where the error surfaced — just past a
/// consumed value for type mismatches, not the value's start. Reporting the
/// start would need bookkeeping on the value-dispatch hot path; the cursor
/// semantic is pinned by `type_errors_carry_offsets`.
fn locate(e: Error, de: &Deserializer<'_, '_>) -> Error {
    match e {
        Error::Custom(message) => Error::Syntax {
            offset: de.offset(),
            message,
        },
        located => located,
    }
}

/// Deserialize `T` from raw JSON5 bytes; fails if `v` is not valid UTF-8,
/// mirroring [`crate::from_slice`].
pub fn from_slice<'de, T>(v: &'de [u8]) -> Result<T>
where
    T: serde::Deserialize<'de>,
{
    let s = simdutf8::compat::from_utf8(v).map_err(|e| Error::InvalidUtf8 {
        offset: e.valid_up_to(),
    })?;
    from_str(s)
}

/// Map serde's `Error` protocol onto [`Error::Custom`].
impl de::Error for Error {
    fn custom<T: core::fmt::Display>(msg: T) -> Self {
        Self::Custom(msg.to_string().into_boxed_str())
    }
}

/// Where a value is being parsed from, deciding the error for misplaced
/// structural bytes. Mirrors `check_value_position` and the `,` / `:` arms of
/// stage 2's main loop.
#[derive(Clone, Copy)]
enum Ctx {
    /// Document root: exactly one value may appear.
    Top,
    /// Between `[` and `]`: an element is expected next.
    Element,
    /// After an object key's `:`.
    MemberValue,
}

/// Error for a structural byte found where a value should start.
fn structural_error(b: u8, off: usize, ctx: Ctx) -> Error {
    match (b, ctx) {
        (b':', _) => Error::UnexpectedChar {
            byte: b,
            offset: off,
        },
        // `,` `]` `}` after an object colon report "expected value" (stage 2
        // routes them through the object state machine).
        (_, Ctx::MemberValue) => syntax("expected value", off),
        _ => Error::UnexpectedChar {
            byte: b,
            offset: off,
        },
    }
}

/// Syntax-error constructor. Cold: keeps error construction out of the serde
/// dispatch methods' code paths.
#[cold]
fn syntax(message: &str, offset: usize) -> Error {
    Error::Syntax {
        message: message.into(),
        offset,
    }
}

/// Pull-based stage-2 mirror: walks index entries and drives serde visitors.
struct Deserializer<'a, 'de> {
    bytes: &'de [u8],
    index: &'a [u32],
    /// Clean-string spans of stage 1, in opening-offset order; the head is
    /// consumed whenever the string it records is parsed (see
    /// [`stage2::parse_string_spanned`]).
    spans: &'a [Span],
    /// Child-count hints keyed by index position of each container opening
    /// (see [`container_counts`]); built by [`Deserializer::hint_counts`] on
    /// the first `size_hint` request, so scalar-root and early-stop parses
    /// never pay the whole-index scan and its allocation. Consulted by the
    /// `size_hint` overrides of [`SeqAccess`] and [`MapAccess`] only.
    counts: OnceCell<Counts>,
    pos: usize,
    depth: usize,
    ctx: Ctx,
    /// Whether the innermost open container has consumed at least one
    /// element/member. Kept in the deserializer because `visit_seq` /
    /// `visit_map` consume their access object; saved and restored around
    /// nested containers.
    started: bool,
}

impl<'a, 'de> Deserializer<'a, 'de> {
    fn new(bytes: &'de [u8], index: &'a [u32], spans: &'a [Span]) -> Self {
        Self {
            bytes,
            index,
            spans,
            counts: OnceCell::new(),
            pos: 0,
            depth: 0,
            ctx: Ctx::Top,
            started: false,
        }
    }

    /// The counts table, computing it on first request. `container_counts`
    /// reads only the immutable input and index, so deferring it cannot
    /// change what it returns.
    fn hint_counts(&self) -> &[u32] {
        self.counts
            .get_or_init(|| container_counts(self.bytes, self.index))
            .as_slice()
    }

    /// String value at the quote entry `off`: borrowed straight from the
    /// clean stage-1 span when its head matches, the full validating parse
    /// otherwise.
    fn string(&mut self, off: usize) -> Result<Str<'de>> {
        stage2::parse_string_spanned(self.bytes, off, &mut self.spans)
    }

    /// Entry offset the cursor stands on, or end of input for EOF errors.
    fn offset(&self) -> usize {
        match self.index.get(self.pos) {
            // Entries are input offsets and stage 1 caps input length at
            // u32::MAX, so the cast never truncates.
            Some(&e) => e as usize,
            None => self.bytes.len(),
        }
    }

    /// Byte and offset of the current entry in one index lookup, or `None`
    /// when the index is exhausted (EOF).
    fn peek_entry(&self) -> Option<(u8, usize)> {
        match self.index.get(self.pos) {
            Some(&e) => {
                let off = e as usize;
                Some((self.bytes[off], off))
            },
            None => None,
        }
    }

    /// Byte at the current entry, or `None` when the index is exhausted.
    fn peek_byte(&self) -> Option<u8> {
        self.peek_entry().map(|(b, _)| b)
    }

    /// Advance past the current entry.
    fn bump(&mut self) {
        self.pos += 1;
    }

    /// Reject trailing input after a complete root value.
    fn end(&mut self) -> Result<()> {
        match self.index.get(self.pos) {
            Some(&e) => Err(Error::TrailingCharacters { offset: e as usize }),
            None => Ok(()),
        }
    }

    /// Parse one scalar (string or bare token) in value position, consuming
    /// its entry. Structural bytes in value position produce the same errors
    /// stage 2 would.
    fn parse_scalar(&mut self) -> Result<Value<'de>> {
        let Some((b, off)) = self.peek_entry() else {
            return Err(Error::UnexpectedEof {
                offset: self.bytes.len(),
            });
        };
        match b {
            b'"' | b'\'' => {
                let s = self.string(off)?;
                self.bump();
                Ok(Value::Str(s))
            },
            b',' | b':' | b']' | b'}' => Err(structural_error(b, off, self.ctx)),
            _ => {
                let end = token_end(self.bytes, off);
                let v = stage2::token_value(&self.bytes[off..end], off)?;
                self.bump();
                Ok(v)
            },
        }
    }

    /// Parse the next value for a scalar-expecting `deserialize_*` method.
    /// Containers fail with `invalid_type` before any input is consumed;
    /// other errors match stage 2 exactly.
    fn parse_scalar_or_type_error<V: Visitor<'de>>(&mut self, visitor: &V) -> Result<Value<'de>> {
        match self.peek_byte() {
            Some(b'[') => Err(de::Error::invalid_type(Unexpected::Seq, visitor)),
            Some(b'{') => Err(de::Error::invalid_type(Unexpected::Map, visitor)),
            _ => self.parse_scalar(),
        }
    }

    /// Parse a value for a bytes-expecting method. Like serde_json, a string
    /// is accepted: its (unescaped) bytes reach the visitor through
    /// `visit_borrowed_bytes`/`visit_bytes`; anything else goes through the
    /// seq path, so arrays of numbers still work and other inputs produce
    /// serde's type errors.
    fn parse_bytes<V: Visitor<'de>>(&mut self, visitor: V) -> Result<V::Value> {
        match self.peek_byte() {
            Some(b'"' | b'\'') => {
                let off = self.offset();
                let s = self.string(off)?;
                self.bump();
                match s {
                    Str::Static(s) => visitor.visit_borrowed_bytes(s.as_bytes()),
                    Str::Dynamic(s) => visitor.visit_bytes(s.as_bytes()),
                }
            },
            _ => self.parse_seq(visitor),
        }
    }

    /// Parse a `[` container for a seq-expecting method (`seq`, `tuple`,
    /// `tuple_struct`, `bytes`); mismatches produce serde type errors.
    fn parse_seq<V: Visitor<'de>>(&mut self, visitor: V) -> Result<V::Value> {
        match self.peek_byte() {
            Some(b'[') => self.parse_value(visitor),
            Some(b'{') => Err(de::Error::invalid_type(Unexpected::Map, &visitor)),
            _ => {
                let v = self.parse_scalar()?;
                Err(invalid_type_value(&v, &visitor))
            },
        }
    }

    /// Parse a `{` container for a map-expecting method.
    fn parse_map<V: Visitor<'de>>(&mut self, visitor: V) -> Result<V::Value> {
        match self.peek_byte() {
            Some(b'{') => self.parse_value(visitor),
            Some(b'[') => Err(de::Error::invalid_type(Unexpected::Seq, &visitor)),
            _ => {
                let v = self.parse_scalar()?;
                Err(invalid_type_value(&v, &visitor))
            },
        }
    }

    /// Parse a struct from `{ ... }` or, like serde_json, from `[ ... ]`.
    fn parse_struct<V: Visitor<'de>>(&mut self, visitor: V) -> Result<V::Value> {
        if matches!(self.peek_byte(), Some(b'{' | b'[')) {
            self.parse_value(visitor)
        } else {
            let v = self.parse_scalar()?;
            Err(invalid_type_value(&v, &visitor))
        }
    }

    /// True when the cursor stands on the bare token `null`. Constant time:
    /// match the four literal bytes, then require a token terminator.
    /// `nullx`-style longer tokens compare unequal and reach the scalar
    /// path, which reports stage 2's unknown literal. Only `token_end`
    /// classifies a high-bit terminator (Unicode whitespace ends a token,
    /// other non-ASCII continues it), so that rare boundary keeps the scan.
    fn peek_is_null(&self) -> bool {
        let Some((b'n', off)) = self.peek_entry() else {
            return false;
        };
        if self.bytes.get(off..off + 4) != Some(b"null") {
            return false;
        }
        match self.bytes.get(off + 4) {
            None => true,
            Some(&b) if b >= 0x80 => &self.bytes[off..token_end(self.bytes, off)] == b"null",
            Some(&b) => CLASSES[b as usize] & class::TOKEN == 0,
        }
    }

    /// Parse a number for an integer-expecting method, hinting by variant
    /// so range checks stay with the visitor.
    fn parse_integer<V: Visitor<'de>>(&mut self, visitor: V) -> Result<V::Value> {
        match self.try_current_num() {
            Some(Ok(n)) => visit_num(n, visitor),
            Some(Err(e)) => Err(e),
            None => {
                let v = self.parse_scalar_or_type_error(&visitor)?;
                visit_number(v, visitor)
            },
        }
    }

    /// Fast path: interpret the entry at `off` as a bare number token with
    /// `parse` and advance past it. The caller has checked
    /// [`number::is_num_start`] on the entry byte, which is exactly the byte
    /// set `stage2::token_value` routes to `number::parse` — so behavior is
    /// identical to the scalar slow path, minus the generic token
    /// classification and intermediate `Value`. The `parse` argument picks
    /// the number.rs entry: [`number::parse_num`] itself, or the f32 / 128-bit
    /// flavors the matching visitors use.
    #[inline]
    fn parse_current_num_with<T>(
        &mut self,
        off: usize,
        parse: fn(&[u8], usize) -> Result<T>,
    ) -> Result<T> {
        let end = token_end(self.bytes, off);
        let n = parse(&self.bytes[off..end], off)?;
        self.bump();
        Ok(n)
    }

    fn parse_current_num(&mut self, off: usize) -> Result<Num> {
        self.parse_current_num_with(off, number::parse_num)
    }

    /// Number fast path for the typed scalar methods: parse the current entry
    /// with `parse` when it can start a number. `None` means the entry cannot
    /// start a number; the caller keeps its scalar slow path, which owns the
    /// `[`/`{` type errors, the structural-byte diagnostics and the bare
    /// `Infinity`/`NaN` keywords.
    fn try_current_num_with<T>(
        &mut self,
        parse: fn(&[u8], usize) -> Result<T>,
    ) -> Option<Result<T>> {
        let (b, off) = self.peek_entry()?;
        if !number::is_num_start(b) {
            return None;
        }
        Some(self.parse_current_num_with(off, parse))
    }

    fn try_current_num(&mut self) -> Option<Result<Num>> {
        self.try_current_num_with(number::parse_num)
    }

    /// Dispatch a value to `visitor` by its entry kind.
    fn parse_value<V: Visitor<'de>>(&mut self, visitor: V) -> Result<V::Value> {
        match self.peek_entry() {
            Some((b'[', _)) => self.parse_array(visitor),
            Some((b'{', _)) => self.parse_object(visitor),
            Some((b, off)) if number::is_num_start(b) => {
                visit_num(self.parse_current_num(off)?, visitor)
            },
            _ => {
                let v = self.parse_scalar()?;
                visit_scalar(v, visitor)
            },
        }
    }

    /// Consume a container opening entry and enter it, guarding the depth
    /// exactly like stage 2 (the limit applies to the enclosing count).
    fn open_container(&mut self, off: usize) -> Result<()> {
        if self.depth >= MAX_DEPTH {
            return Err(Error::DepthLimitExceeded {
                limit: MAX_DEPTH,
                offset: off,
            });
        }
        self.bump();
        self.depth += 1;
        Ok(())
    }

    /// Parse an array through `visit_seq`. Elements past an early visitor
    /// stop are validated and skipped by `finish_seq`.
    fn parse_array<V: Visitor<'de>>(&mut self, visitor: V) -> Result<V::Value> {
        // Capture the `[` entry's index position for the counts table
        // before `open_container` advances past it.
        let open = self.pos;
        self.open_container(self.offset())?;
        let saved = self.started;
        self.started = false;
        let result = visitor.visit_seq(SeqAccess {
            de: &mut *self,
            open,
        });
        let started = self.started;
        self.started = saved;
        match result {
            Ok(v) => {
                let fin = self.finish_seq(started);
                self.depth -= 1;
                fin?;
                Ok(v)
            },
            // A visitor error aborts the whole document; no resync.
            Err(e) => {
                self.depth -= 1;
                Err(e)
            },
        }
    }

    /// Parse an object through `visit_map`. Members past an early visitor
    /// stop are validated and skipped by `finish_map`.
    fn parse_object<V: Visitor<'de>>(&mut self, visitor: V) -> Result<V::Value> {
        let open = self.pos;
        self.open_container(self.offset())?;
        let saved = self.started;
        self.started = false;
        let result = visitor.visit_map(MapAccess {
            de: &mut *self,
            open,
        });
        let started = self.started;
        self.started = saved;
        match result {
            Ok(v) => {
                let fin = self.finish_map(started);
                self.depth -= 1;
                fin?;
                Ok(v)
            },
            Err(e) => {
                self.depth -= 1;
                Err(e)
            },
        }
    }

    /// Advance past one array slot boundary. `Ok(false)` means the cursor
    /// stands on the `]` (not consumed); `Ok(true)` means an element follows
    /// at the cursor with its leading `,` already consumed.
    fn seq_step(&mut self, started: bool) -> Result<bool> {
        match self.peek_byte() {
            Some(b']') => Ok(false),
            Some(b',') if started => {
                self.bump();
                // One optional trailing comma directly before `]`.
                Ok(self.peek_byte() != Some(b']'))
            },
            // `:` and `}` keep stage 2's byte-specific errors after an
            // element; value starts report the missing separator.
            Some(b @ (b':' | b'}')) if started => Err(Error::UnexpectedChar {
                byte: b,
                offset: self.offset(),
            }),
            Some(_) if started => Err(syntax("expected ',' or ']'", self.offset())),
            // Before the first element a `,` stays for `parse_scalar` to
            // reject, matching stage 2's `after_element` logic.
            Some(_) => Ok(true),
            None => Err(Error::UnexpectedEof {
                offset: self.bytes.len(),
            }),
        }
    }

    /// Advance past one object member boundary. `Ok(false)` means the cursor
    /// stands on the `}` (not consumed); `Ok(true)` means a key follows.
    fn map_step(&mut self, started: bool) -> Result<bool> {
        match self.peek_byte() {
            Some(b'}') => Ok(false),
            Some(b',') if started => {
                self.bump();
                // One optional trailing comma directly before `}`.
                Ok(self.peek_byte() != Some(b'}'))
            },
            // `:` and `]` keep stage 2's byte-specific errors after a member.
            Some(b':') if started => Err(Error::UnexpectedChar {
                byte: b':',
                offset: self.offset(),
            }),
            Some(b']') if started => Err(Error::UnexpectedChar {
                byte: b']',
                offset: self.offset(),
            }),
            Some(_) if started => Err(syntax("expected ',' or '}'", self.offset())),
            // Before the first member these are rejected by `next_key`.
            Some(_) => Ok(true),
            None => Err(Error::UnexpectedEof {
                offset: self.bytes.len(),
            }),
        }
    }

    /// Parse an object key (quoted string or unquoted identifier), consuming
    /// its entry. Everything else a key position may contain is rejected the
    /// way stage 2 rejects it.
    fn next_key(&mut self) -> Result<Str<'de>> {
        let Some(b) = self.peek_byte() else {
            return Err(Error::UnexpectedEof {
                offset: self.bytes.len(),
            });
        };
        let off = self.offset();
        match b {
            b'"' | b'\'' => {
                let s = self.string(off)?;
                self.bump();
                Ok(s)
            },
            b',' | b':' | b']' => Err(Error::UnexpectedChar {
                byte: b,
                offset: off,
            }),
            // Containers where a key belongs report the key error, matching
            // stage 2's `check_value_position`.
            b'[' | b'{' => Err(syntax("expected identifier or string key", off)),
            _ => {
                let end = token_end(self.bytes, off);
                // Unquoted number keys are invalid.
                if matches!(b, b'+' | b'-' | b'.' | b'0'..=b'9') {
                    return Err(syntax("expected identifier or string key", off));
                }
                let key = stage2::parse_identifier(self.bytes, off, end)?;
                self.bump();
                Ok(key)
            },
        }
    }

    /// Consume the `:` after a key; any other entry reports stage 2's
    /// `expected ':'` state error.
    fn expect_colon(&mut self) -> Result<()> {
        match self.peek_byte() {
            Some(b':') => {
                self.bump();
                Ok(())
            },
            Some(_) => Err(syntax("expected ':'", self.offset())),
            None => Err(Error::UnexpectedEof {
                offset: self.bytes.len(),
            }),
        }
    }

    /// Validate and skip the rest of an array a visitor stopped consuming,
    /// then consume the closing `]`. A fully drained array still leaves the
    /// `]` at the cursor for this method.
    fn finish_seq(&mut self, started: bool) -> Result<()> {
        let mut started = started;
        loop {
            if !self.seq_step(started)? {
                self.bump();
                return Ok(());
            }
            started = true;
            self.skip_value(Ctx::Element)?;
        }
    }

    /// Validate and skip the rest of an object a visitor stopped consuming,
    /// then consume the closing `}`.
    fn finish_map(&mut self, started: bool) -> Result<()> {
        let mut started = started;
        loop {
            if !self.map_step(started)? {
                self.bump();
                return Ok(());
            }
            started = true;
            self.next_key()?;
            self.expect_colon()?;
            self.skip_value(Ctx::MemberValue)?;
        }
    }

    /// Parse one value and discard it, validating exactly like a visited
    /// parse. Used by `deserialize_ignored_any` and the finish helpers.
    /// Recursive like `parse_value` but with minimal frames; the depth guard
    /// bounds the recursion.
    fn skip_value(&mut self, ctx: Ctx) -> Result<()> {
        let Some(b) = self.peek_byte() else {
            return Err(Error::UnexpectedEof {
                offset: self.bytes.len(),
            });
        };
        let off = self.offset();
        match b {
            b'[' => {
                self.open_container(off)?;
                let r = self.skip_seq_contents();
                self.depth -= 1;
                r
            },
            b'{' => {
                self.open_container(off)?;
                let r = self.skip_map_contents();
                self.depth -= 1;
                r
            },
            b'"' | b'\'' => {
                self.string(off)?;
                self.bump();
                Ok(())
            },
            b',' | b':' | b']' | b'}' => Err(structural_error(b, off, ctx)),
            _ => {
                let end = token_end(self.bytes, off);
                stage2::token_value(&self.bytes[off..end], off)?;
                self.bump();
                Ok(())
            },
        }
    }

    fn skip_seq_contents(&mut self) -> Result<()> {
        let mut started = false;
        loop {
            if !self.seq_step(started)? {
                self.bump();
                return Ok(());
            }
            started = true;
            self.skip_value(Ctx::Element)?;
        }
    }

    fn skip_map_contents(&mut self) -> Result<()> {
        let mut started = false;
        loop {
            if !self.map_step(started)? {
                self.bump();
                return Ok(());
            }
            started = true;
            self.next_key()?;
            self.expect_colon()?;
            self.skip_value(Ctx::MemberValue)?;
        }
    }
}

/// Feed a parsed scalar to `visitor`, borrowing escape-free strings.
fn visit_scalar<'de, V: Visitor<'de>>(v: Value<'de>, visitor: V) -> Result<V::Value> {
    match v {
        Value::Null => visitor.visit_unit(),
        Value::Bool(b) => visitor.visit_bool(b),
        Value::I64(i) => visitor.visit_i64(i),
        Value::U64(u) => visitor.visit_u64(u),
        Value::F64(f) => visitor.visit_f64(f),
        Value::Str(Str::Static(s)) => visitor.visit_borrowed_str(s),
        Value::Str(Str::Dynamic(s)) => visitor.visit_string(s),
        // Container entries are dispatched by `parse_value` before this
        // helper runs; `parse_scalar` never produces them.
        Value::Array(_) | Value::Object(_) => unreachable!("scalars only"),
    }
}

/// serde's `Unexpected` rendering of a parsed value, for type errors.
fn value_unexpected<'a>(v: &'a Value<'_>) -> Unexpected<'a> {
    match v {
        Value::Null => Unexpected::Unit,
        Value::Bool(b) => Unexpected::Bool(*b),
        Value::U64(u) => Unexpected::Unsigned(*u),
        Value::I64(i) => Unexpected::Signed(*i),
        Value::F64(f) => Unexpected::Float(*f),
        Value::Str(s) => Unexpected::Str(s.as_str()),
        // Unreachable: every caller feeds this helper the output of
        // `parse_scalar` or `parse_scalar_or_type_error`, which only produce
        // scalars — their callers dispatch `[`/`{` containers themselves.
        Value::Array(_) => Unexpected::Seq,
        Value::Object(_) => Unexpected::Map,
    }
}

/// `invalid type` error for a parsed value that mismatched the visitor.
fn invalid_type_value<'de, V: Visitor<'de>>(v: &Value<'_>, visitor: &V) -> Error {
    de::Error::invalid_type(value_unexpected(v), visitor)
}

/// Feed a parsed number to `visitor`, hinting by variant like serde_json.
fn visit_number<'de, V: Visitor<'de>>(v: Value<'de>, visitor: V) -> Result<V::Value> {
    match v {
        Value::U64(u) => visitor.visit_u64(u),
        Value::I64(i) => visitor.visit_i64(i),
        Value::F64(f) => visitor.visit_f64(f),
        other => Err(invalid_type_value(&other, &visitor)),
    }
}

/// Feed a fast-path [`Num`] to `visitor` by variant, matching `visit_scalar`
/// and `visit_number` exactly (U64 -> `visit_u64`, I64 -> `visit_i64`,
/// F64 -> `visit_f64`).
fn visit_num<'de, V: Visitor<'de>>(n: Num, visitor: V) -> Result<V::Value> {
    match n {
        Num::U64(u) => visitor.visit_u64(u),
        Num::I64(i) => visitor.visit_i64(i),
        Num::F64(f) => visitor.visit_f64(f),
    }
}

/// Cap a self-computed capacity hint, mirroring serde's internal
/// `size_hint::cautious` (not public in current serde): preallocate at most
/// 1 MiB worth of elements, so a hostile estimate cannot force a huge up-front
/// allocation; beyond the cap collections keep growing by doubling.
fn cautious<E>(hint: Option<usize>) -> usize {
    const MAX_PREALLOC_BYTES: usize = 1024 * 1024;
    let max = MAX_PREALLOC_BYTES / size_of::<E>().max(1);
    hint.unwrap_or(0).min(max)
}

/// Array element feed for `visit_seq`; elements are parsed on demand through
/// the shared comma lookahead in [`Deserializer::seq_step`].
struct SeqAccess<'b, 'a, 'de> {
    de: &'b mut Deserializer<'a, 'de>,
    /// Index position of this array's `[` entry; keys `de.counts`.
    open: usize,
}

impl<'de> de::SeqAccess<'de> for SeqAccess<'_, '_, 'de> {
    type Error = Error;

    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>>
    where
        T: DeserializeSeed<'de>,
    {
        let started = self.de.started;
        if !self.de.seq_step(started)? {
            return Ok(None);
        }
        self.de.started = true;
        self.de.ctx = Ctx::Element;
        seed.deserialize(&mut *self.de).map(Some)
    }

    fn size_hint(&self) -> Option<usize> {
        // The element count recorded when this array opened. Visitors read
        // it before the first element (serde's Vec visitor calls it once);
        // afterwards it stays a safe upper bound (trailing comma). `None`
        // when the table was skipped for a tiny document; an unclosed
        // container keeps a recorded 0 and yields a zero-valued hint.
        self.de.hint_counts().get(self.open).map(|&c| c as usize)
    }
}

/// Object member feed for `visit_map`; keys are parsed eagerly (string or
/// identifier), values on demand.
struct MapAccess<'b, 'a, 'de> {
    de: &'b mut Deserializer<'a, 'de>,
    /// Index position of this object's `{` entry; keys `de.counts`.
    open: usize,
}

impl<'de> de::MapAccess<'de> for MapAccess<'_, '_, 'de> {
    type Error = Error;

    fn next_key_seed<K>(&mut self, seed: K) -> Result<Option<K::Value>>
    where
        K: DeserializeSeed<'de>,
    {
        let started = self.de.started;
        if !self.de.map_step(started)? {
            return Ok(None);
        }
        self.de.started = true;
        let key = self.de.next_key()?;
        seed.deserialize(MapKeyDeserializer { key }).map(Some)
    }

    fn next_value_seed<V>(&mut self, seed: V) -> Result<V::Value>
    where
        V: DeserializeSeed<'de>,
    {
        self.de.expect_colon()?;
        self.de.ctx = Ctx::MemberValue;
        seed.deserialize(&mut *self.de)
    }

    fn size_hint(&self) -> Option<usize> {
        // Like the seq case; duplicate keys can only over-estimate.
        self.de.hint_counts().get(self.open).map(|&c| c as usize)
    }
}

/// Offers an already-parsed key string to a field or map-key seed, borrowed
/// when the source had no escapes. Bool and numeric requests parse the key
/// text — numbers through Rust's `FromStr` grammar, which is looser than
/// serde_json's JSON-grammar numeric keys (a leading `+` and the float
/// spellings `Infinity`/`NaN` are accepted) — and enum requests treat it as
/// the variant name; everything else forwards to `deserialize_any` so
/// mismatched key types still produce standard serde type errors. The
/// numeric and bool behavior is pinned by the map-key tests in
/// `tests/typed_deserialize.rs`.
struct MapKeyDeserializer<'de> {
    key: Str<'de>,
}

impl<'de> MapKeyDeserializer<'de> {
    fn visit<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        match self.key {
            Str::Static(s) => visitor.visit_borrowed_str(s),
            Str::Dynamic(s) => visitor.visit_string(s),
        }
    }

    /// Parse the key text as a number for numeric map-key types, through
    /// Rust's `FromStr` grammar — deliberately looser than the document's
    /// number grammar, since quoted keys carry arbitrary text and `FromStr`
    /// accepts forms `parse_num` never sees (`+1`, `Infinity`). Pinned by
    /// the map-key tests in `tests/typed_deserialize.rs`.
    fn parse_key_number<T: core::str::FromStr>(self, expecting: &'static str) -> Result<T> {
        let s = self.key.as_str();
        s.parse()
            .map_err(|_| de::Error::invalid_value(Unexpected::Str(s), &expecting))
    }
}

impl<'de> serde::Deserializer<'de> for MapKeyDeserializer<'de> {
    type Error = Error;

    serde::forward_to_deserialize_any! {
        char str string bytes byte_buf unit unit_struct seq tuple
        tuple_struct map struct identifier ignored_any
    }

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.visit(visitor)
    }

    /// A map key is always text, never `null`, so an `Option` key is always
    /// `Some` — like serde_json.
    fn deserialize_option<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_some(self)
    }

    /// Newtype keys take their inner type from the key text.
    fn deserialize_newtype_struct<V>(self, _name: &'static str, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_newtype_struct(self)
    }

    fn deserialize_bool<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        match self.key.as_str() {
            "true" => visitor.visit_bool(true),
            "false" => visitor.visit_bool(false),
            // serde_json's key deserializer reports the same invalid_type.
            other => Err(de::Error::invalid_type(Unexpected::Str(other), &visitor)),
        }
    }

    fn deserialize_i8<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.parse_key_number::<i8>("i8")
            .and_then(|v| visitor.visit_i8(v))
    }

    fn deserialize_i16<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.parse_key_number::<i16>("i16")
            .and_then(|v| visitor.visit_i16(v))
    }

    fn deserialize_i32<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.parse_key_number::<i32>("i32")
            .and_then(|v| visitor.visit_i32(v))
    }

    fn deserialize_i64<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.parse_key_number::<i64>("i64")
            .and_then(|v| visitor.visit_i64(v))
    }

    fn deserialize_u8<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.parse_key_number::<u8>("u8")
            .and_then(|v| visitor.visit_u8(v))
    }

    fn deserialize_u16<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.parse_key_number::<u16>("u16")
            .and_then(|v| visitor.visit_u16(v))
    }

    fn deserialize_u32<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.parse_key_number::<u32>("u32")
            .and_then(|v| visitor.visit_u32(v))
    }

    fn deserialize_u64<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.parse_key_number::<u64>("u64")
            .and_then(|v| visitor.visit_u64(v))
    }

    fn deserialize_i128<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.parse_key_number::<i128>("i128")
            .and_then(|v| visitor.visit_i128(v))
    }

    fn deserialize_u128<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.parse_key_number::<u128>("u128")
            .and_then(|v| visitor.visit_u128(v))
    }

    fn deserialize_f32<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.parse_key_number::<f32>("f32")
            .and_then(|v| visitor.visit_f32(v))
    }

    fn deserialize_f64<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.parse_key_number("f64")
            .and_then(|v| visitor.visit_f64(v))
    }

    fn deserialize_enum<V>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_enum(UnitVariantAccess { variant: self.key })
    }
}

/// Enum as a plain string: a unit variant only. Requests for payload forms
/// fail with serde's unit-variant type error, like serde_json.
struct UnitVariantAccess<'de> {
    variant: Str<'de>,
}

impl<'de> de::EnumAccess<'de> for UnitVariantAccess<'de> {
    type Error = Error;
    type Variant = Self;

    fn variant_seed<V>(mut self, seed: V) -> Result<(V::Value, Self::Variant)>
    where
        V: DeserializeSeed<'de>,
    {
        // The VariantAccess impls never read `variant`, so take the name
        // instead of cloning it (an allocation for escaped variant names).
        let value = seed.deserialize(MapKeyDeserializer {
            key: core::mem::take(&mut self.variant),
        })?;
        Ok((value, self))
    }
}

impl<'de> de::VariantAccess<'de> for UnitVariantAccess<'de> {
    type Error = Error;

    fn unit_variant(self) -> Result<()> {
        Ok(())
    }

    fn newtype_variant_seed<T>(self, _seed: T) -> Result<T::Value>
    where
        T: DeserializeSeed<'de>,
    {
        Err(de::Error::invalid_type(
            Unexpected::UnitVariant,
            &"newtype variant",
        ))
    }

    fn tuple_variant<V>(self, _len: usize, _visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        Err(de::Error::invalid_type(
            Unexpected::UnitVariant,
            &"tuple variant",
        ))
    }

    fn struct_variant<V>(self, _fields: &'static [&'static str], _visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        Err(de::Error::invalid_type(
            Unexpected::UnitVariant,
            &"struct variant",
        ))
    }
}

/// Enum as a single-member object `{ Variant: ... }`. The member's key is
/// the variant name (quoted or unquoted identifier); its value is the
/// variant payload.
struct EnumMapAccess<'b, 'a, 'de> {
    de: &'b mut Deserializer<'a, 'de>,
}

impl<'b, 'a, 'de> de::EnumAccess<'de> for EnumMapAccess<'b, 'a, 'de> {
    type Error = Error;
    type Variant = EnumVariantAccess<'b, 'a, 'de>;

    fn variant_seed<V>(self, seed: V) -> Result<(V::Value, Self::Variant)>
    where
        V: DeserializeSeed<'de>,
    {
        // An empty object in enum position is a type-level error (stage 2
        // happily parses `{}`).
        if !self.de.map_step(false)? {
            return Err(de::Error::custom("expected value"));
        }
        let key = self.de.next_key()?;
        let value = seed.deserialize(MapKeyDeserializer { key })?;
        Ok((value, EnumVariantAccess { de: self.de }))
    }
}

/// Payload half of the map enum form; parses the member value and then
/// requires the object to end.
struct EnumVariantAccess<'b, 'a, 'de> {
    de: &'b mut Deserializer<'a, 'de>,
}

impl EnumVariantAccess<'_, '_, '_> {
    /// Consume the closing `}`, accepting the same single optional trailing
    /// comma as `map_step`; anything else means the enum object carried more
    /// than one member.
    fn finish(self) -> Result<()> {
        match self.de.peek_byte() {
            Some(b'}') => {
                self.de.bump();
                Ok(())
            },
            // One optional trailing comma directly before `}`; a comma
            // leading anywhere else is a second member, reported at the
            // comma like the pre-trailing-comma behavior.
            Some(b',') => {
                let comma = self.de.offset();
                self.de.bump();
                match self.de.peek_byte() {
                    Some(b'}') => {
                        self.de.bump();
                        Ok(())
                    },
                    Some(_) => Err(Error::TrailingCharacters { offset: comma }),
                    None => Err(Error::UnexpectedEof {
                        offset: self.de.bytes.len(),
                    }),
                }
            },
            Some(_) => Err(Error::TrailingCharacters {
                offset: self.de.offset(),
            }),
            None => Err(Error::UnexpectedEof {
                offset: self.de.bytes.len(),
            }),
        }
    }
}

impl<'de> de::VariantAccess<'de> for EnumVariantAccess<'_, '_, 'de> {
    type Error = Error;

    fn unit_variant(self) -> Result<()> {
        self.de.expect_colon()?;
        self.de.ctx = Ctx::MemberValue;
        // A unit variant's payload must be `null`, like serde_json.
        <() as serde::Deserialize<'de>>::deserialize(&mut *self.de)?;
        self.finish()
    }

    fn newtype_variant_seed<T>(self, seed: T) -> Result<T::Value>
    where
        T: DeserializeSeed<'de>,
    {
        self.de.expect_colon()?;
        self.de.ctx = Ctx::MemberValue;
        let value = seed.deserialize(&mut *self.de)?;
        self.finish()?;
        Ok(value)
    }

    fn tuple_variant<V>(self, _len: usize, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.de.expect_colon()?;
        self.de.ctx = Ctx::MemberValue;
        let value = self.de.parse_value(visitor)?;
        self.finish()?;
        Ok(value)
    }

    fn struct_variant<V>(self, _fields: &'static [&'static str], visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.de.expect_colon()?;
        self.de.ctx = Ctx::MemberValue;
        let value = self.de.parse_value(visitor)?;
        self.finish()?;
        Ok(value)
    }
}

impl<'de> serde::Deserialize<'de> for Value<'de> {
    fn deserialize<D>(deserializer: D) -> core::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(ValueVisitor)
    }
}

struct ValueVisitor;

impl<'de> Visitor<'de> for ValueVisitor {
    type Value = Value<'de>;

    fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("any valid JSON5 value")
    }

    fn visit_unit<E>(self) -> core::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Null)
    }

    fn visit_bool<E>(self, v: bool) -> core::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Bool(v))
    }

    fn visit_i64<E>(self, v: i64) -> core::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::I64(v))
    }

    fn visit_u64<E>(self, v: u64) -> core::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::U64(v))
    }

    fn visit_f64<E>(self, v: f64) -> core::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::F64(v))
    }

    fn visit_borrowed_str<E>(self, v: &'de str) -> core::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Str(Str::Static(v)))
    }

    fn visit_str<E>(self, v: &str) -> core::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Str(Str::Dynamic(v.to_owned())))
    }

    fn visit_string<E>(self, v: String) -> core::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Str(Str::Dynamic(v)))
    }

    fn visit_seq<A>(self, mut seq: A) -> core::result::Result<Self::Value, A::Error>
    where
        A: de::SeqAccess<'de>,
    {
        // The hint is exact except for a trailing comma; `cautious` bounds
        // the preallocation, beyond it the vec keeps doubling.
        let mut items = Vec::with_capacity(cautious::<Value<'de>>(seq.size_hint()));
        while let Some(v) = seq.next_element()? {
            items.push(v);
        }
        Ok(Value::Array(items))
    }

    fn visit_map<A>(self, mut access: A) -> core::result::Result<Self::Value, A::Error>
    where
        A: de::MapAccess<'de>,
    {
        let mut entries =
            Map::with_capacity(cautious::<(Str<'de>, Value<'de>)>(access.size_hint()));
        // IndexMap::insert keeps the first position and the last value, the
        // JavaScript duplicate-key semantics of stage 2.
        while let Some((k, v)) = access.next_entry()? {
            entries.insert(k, v);
        }
        Ok(Value::Object(entries))
    }
}

impl<'de> serde::Deserialize<'de> for Str<'de> {
    fn deserialize<D>(deserializer: D) -> core::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_str(StrVisitor)
    }
}

struct StrVisitor;

impl<'de> Visitor<'de> for StrVisitor {
    type Value = Str<'de>;

    fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("a JSON5 string")
    }

    fn visit_borrowed_str<E>(self, v: &'de str) -> core::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Str::Static(v))
    }

    fn visit_str<E>(self, v: &str) -> core::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        // Only reached from foreign deserializers; ours offers borrowed or
        // owned strings directly.
        Ok(Str::Dynamic(v.to_owned()))
    }

    fn visit_string<E>(self, v: String) -> core::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Str::Dynamic(v))
    }
}

impl<'de> serde::Deserializer<'de> for &mut Deserializer<'_, 'de> {
    type Error = Error;

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.parse_value(visitor)
    }

    fn deserialize_bool<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        match self.parse_scalar_or_type_error(&visitor)? {
            Value::Bool(b) => visitor.visit_bool(b),
            other => Err(invalid_type_value(&other, &visitor)),
        }
    }

    fn deserialize_i8<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.parse_integer(visitor)
    }

    fn deserialize_i16<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.parse_integer(visitor)
    }

    fn deserialize_i32<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.parse_integer(visitor)
    }

    fn deserialize_i64<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.parse_integer(visitor)
    }

    fn deserialize_u8<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.parse_integer(visitor)
    }

    fn deserialize_u16<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.parse_integer(visitor)
    }

    fn deserialize_u32<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.parse_integer(visitor)
    }

    fn deserialize_u64<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.parse_integer(visitor)
    }

    fn deserialize_i128<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        match self.try_current_num_with(number::parse_num_wide) {
            // Exact across the whole 128-bit range; literals past it are
            // InvalidNumber from the parser, and the visitor owns the
            // narrower-range value errors (a u128 hint on an i128 target).
            Some(Ok(Num128::U128(u))) => visitor.visit_u128(u),
            Some(Ok(Num128::I128(i))) => visitor.visit_i128(i),
            Some(Ok(Num128::F64(f))) => visitor.visit_f64(f),
            Some(Err(e)) => Err(e),
            None => match self.parse_scalar_or_type_error(&visitor)? {
                Value::U64(u) => visitor.visit_u128(u128::from(u)),
                Value::I64(i) => visitor.visit_i128(i128::from(i)),
                Value::F64(f) => visitor.visit_f64(f),
                other => Err(invalid_type_value(&other, &visitor)),
            },
        }
    }

    fn deserialize_u128<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.deserialize_i128(visitor)
    }

    fn deserialize_f32<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        match self.try_current_num_with(number::parse_num_f32) {
            Some(Ok(NumF32::U64(u))) => visitor.visit_u64(u),
            Some(Ok(NumF32::I64(i))) => visitor.visit_i64(i),
            // One decimal rounding, not the f64 double rounding.
            Some(Ok(NumF32::F32(v))) => visitor.visit_f32(v),
            Some(Err(e)) => Err(e),
            // Non-number entries (keywords, strings, containers) share the
            // f64 method's type errors and value mapping.
            None => self.deserialize_f64(visitor),
        }
    }

    fn deserialize_f64<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        match self.try_current_num() {
            // Same variant-to-visitor mapping as the slow path below; only
            // the match order differs.
            Some(Ok(n)) => visit_num(n, visitor),
            Some(Err(e)) => Err(e),
            None => match self.parse_scalar_or_type_error(&visitor)? {
                Value::F64(f) => visitor.visit_f64(f),
                Value::U64(u) => visitor.visit_u64(u),
                Value::I64(i) => visitor.visit_i64(i),
                other => Err(invalid_type_value(&other, &visitor)),
            },
        }
    }

    fn deserialize_char<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        match self.parse_scalar_or_type_error(&visitor)? {
            Value::Str(s) => {
                let mut chars = s.as_str().chars();
                match (chars.next(), chars.next()) {
                    (Some(c), None) => visitor.visit_char(c),
                    _ => Err(de::Error::invalid_value(
                        Unexpected::Str(s.as_str()),
                        &"a single character",
                    )),
                }
            },
            other => Err(invalid_type_value(&other, &visitor)),
        }
    }

    fn deserialize_str<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        match self.parse_scalar_or_type_error(&visitor)? {
            Value::Str(Str::Static(s)) => visitor.visit_borrowed_str(s),
            Value::Str(Str::Dynamic(s)) => visitor.visit_string(s),
            other => Err(invalid_type_value(&other, &visitor)),
        }
    }

    fn deserialize_string<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.deserialize_str(visitor)
    }

    fn deserialize_bytes<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.parse_bytes(visitor)
    }

    fn deserialize_byte_buf<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.parse_bytes(visitor)
    }

    fn deserialize_option<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        if self.peek_is_null() {
            self.bump();
            visitor.visit_none()
        } else {
            visitor.visit_some(self)
        }
    }

    fn deserialize_unit<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        match self.parse_scalar_or_type_error(&visitor)? {
            Value::Null => visitor.visit_unit(),
            other => Err(invalid_type_value(&other, &visitor)),
        }
    }

    fn deserialize_unit_struct<V>(self, _name: &'static str, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.deserialize_unit(visitor)
    }

    fn deserialize_newtype_struct<V>(self, _name: &'static str, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_newtype_struct(self)
    }

    fn deserialize_seq<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.parse_seq(visitor)
    }

    fn deserialize_tuple<V>(self, _len: usize, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.parse_seq(visitor)
    }

    fn deserialize_tuple_struct<V>(
        self,
        _name: &'static str,
        _len: usize,
        visitor: V,
    ) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.parse_seq(visitor)
    }

    fn deserialize_map<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.parse_map(visitor)
    }

    fn deserialize_struct<V>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.parse_struct(visitor)
    }

    fn deserialize_enum<V>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        match self.peek_byte() {
            Some(b'"' | b'\'') => {
                let off = self.offset();
                let variant = self.string(off)?;
                self.bump();
                visitor.visit_enum(UnitVariantAccess { variant })
            },
            Some(b'{') => {
                self.open_container(self.offset())?;
                let result = visitor.visit_enum(EnumMapAccess { de: &mut *self });
                self.depth -= 1;
                result
            },
            Some(b'[') => Err(de::Error::invalid_type(Unexpected::Seq, &visitor)),
            _ => {
                let v = self.parse_scalar()?;
                Err(invalid_type_value(&v, &visitor))
            },
        }
    }

    fn deserialize_identifier<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.deserialize_str(visitor)
    }

    fn deserialize_ignored_any<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        let ctx = self.ctx;
        self.skip_value(ctx)?;
        visitor.visit_unit()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalars() {
        assert_eq!(from_str::<u64>(" 42 ").unwrap(), 42);
        assert_eq!(from_str::<i64>("-7").unwrap(), -7);
        assert_eq!(from_str::<f64>("1.5e3").unwrap(), 1500.0);
        assert!(from_str::<bool>("true").unwrap());
        assert_eq!(from_str::<()>("null").unwrap(), ());
        assert_eq!(from_str::<String>(r#""a""#).unwrap(), "a");
        assert_eq!(from_str::<&str>("'x'").unwrap(), "x");
    }

    #[test]
    fn scalar_errors_match_stage2() {
        assert_eq!(from_str::<u64>("").unwrap_err(), Error::UnexpectedEof {
            offset: 0
        });
        assert_eq!(
            from_str::<u64>("tru").unwrap_err(),
            syntax("unknown literal", 0)
        );
        assert_eq!(
            from_str::<u64>("1 2").unwrap_err(),
            Error::TrailingCharacters { offset: 2 }
        );
    }

    /// The serde walk must produce the exact DOM stage 2 produces, including
    /// string borrowing and duplicate-key semantics. Debug renderings are
    /// compared because `F64(NaN)` never equals itself.
    #[test]
    fn dom_parity_with_stage2() {
        let docs = [
            "null",
            "true",
            "42",
            "-7",
            "0x1F",
            "-0x20",
            "1.5e3",
            ".5",
            "5.",
            "+7",
            "Infinity",
            "NaN",
            "''",
            r#""abc""#,
            r#""a\nb""#,
            "[]",
            "[1,]",
            "{}",
            "{a:1,}",
            "[1,[2,[]],]",
            "{ a: 1, 'b': [2,], c: { d: null, }, }",
            r"{\u0061: 1}",
            "/* c */ [1] // t",
            "{ if: 1, $x: 2, _y: 3, é: 4 }",
            "\"\\uD83D\\uDE00\"",
            "\"a\\\nb\"",
            "[1\u{a0},\u{2028}2]",
            "{a:1,a:2}",
        ];
        for d in docs {
            assert_eq!(
                format!("{:?}", from_str::<Value>(d).unwrap()),
                format!("{:?}", crate::from_str(d).unwrap()),
                "doc {d:?}"
            );
        }
    }

    /// Grammar failures must surface with identical variants and offsets.
    #[test]
    fn error_parity_with_stage2() {
        let docs = [
            "",
            " ",
            "// c",
            "tru",
            "1a",
            "[1 2]",
            "[1,,2]",
            "{a:1,,}",
            "[1}",
            "{a:1]",
            "{a}",
            "{a:}",
            "[1",
            "{a:1",
            "1 2",
            "null,",
            "{1:2}",
            ":",
            "}",
            "]",
            ",",
            "[,]",
            "{,}",
            "[}]",
            "{]}",
            r#"{"a":}"#,
            "{a::1}",
            "[[}]",
            r#"{"a" 1}"#,
            r#"{"a":1 "b":2}"#,
            "{a:1,]",
            "{a:,}",
            "\"a\nb\"",
            "\"open",
            "/* x",
            "/x",
            "!",
            "[1?]",
            r"{a\x41:1}",
            "{a-b:1}",
            "{+a:1}",
            "[1,",
            "{a",
            "[,",
            "{:,}",
            // A misplaced string reports the structural error, not its own
            // parse error (both paths validate position before parsing).
            r"{a 'b\1': 1}",
            "{a 'b",
            r"[1 'b\1' 2]",
        ];
        for d in docs {
            assert_eq!(
                from_str::<Value>(d).unwrap_err(),
                crate::from_str(d).unwrap_err(),
                "doc {d:?}"
            );
        }
    }

    #[test]
    fn depth_limit_matches_stage2() {
        // Deserializing into the DOM recurses through serde visitor frames,
        // so near-limit documents need more than the default thread stack.
        std::thread::Builder::new()
            .stack_size(32 << 20)
            .spawn(|| {
                let flat = format!("{}{}", "[".repeat(MAX_DEPTH), "]".repeat(MAX_DEPTH));
                let deep = format!("{}{}", "[".repeat(MAX_DEPTH + 1), "]".repeat(MAX_DEPTH + 1));
                assert!(from_str::<Value>(&flat).is_ok());
                assert_eq!(
                    from_str::<Value>(&deep).unwrap_err(),
                    Error::DepthLimitExceeded {
                        limit: MAX_DEPTH,
                        offset: MAX_DEPTH,
                    }
                );
                for doc in [&flat, &deep] {
                    let de_r = from_str::<Value>(doc);
                    let dom_r = crate::from_str(doc);
                    assert_eq!(de_r.is_ok(), dom_r.is_ok(), "doc len {}", doc.len());
                    if let (Err(e1), Err(e2)) = (&de_r, &dom_r) {
                        assert_eq!(e1, e2, "doc len {}", doc.len());
                    }
                }
            })
            .expect("spawn depth test thread")
            .join()
            .expect("depth test thread panicked");
    }

    #[test]
    fn value_impl_borrows_strings() {
        let v = from_str::<Value>(r#"{ a: "x", b: "y\nz", '\u0063': 1 }"#).unwrap();
        let m = v.as_object().unwrap();
        assert!(matches!(
            m.get_index(0).map(|(_, v)| v),
            Some(Value::Str(Str::Static(_)))
        ));
        assert!(matches!(
            m.get_index(1).map(|(_, v)| v),
            Some(Value::Str(Str::Dynamic(_)))
        ));
        // Identifier escapes also produce owned keys.
        assert!(matches!(m.get_index(2), Some((Str::Dynamic(_), _))));
    }

    #[derive(serde::Deserialize, PartialEq, Debug)]
    struct Point {
        x: i32,
        y: i32,
    }

    #[derive(serde::Deserialize, PartialEq, Debug)]
    enum E {
        A,
        B(u8),
        C { x: u8 },
    }

    #[test]
    fn typed_structs() {
        assert_eq!(
            from_str::<Point>("{ x: 1, y: /* c */ 2, }").unwrap(),
            Point { x: 1, y: 2 }
        );
        // Structs accept the seq form like serde_json.
        assert_eq!(from_str::<Point>("[3,4]").unwrap(), Point { x: 3, y: 4 });
        let err = from_str::<Point>("{x:1}").unwrap_err();
        assert!(
            matches!(&err, Error::Syntax { message, .. } if message.contains("missing field `y`"))
        );
    }

    #[test]
    fn typed_scalars() {
        assert_eq!(from_str::<Option<u32>>("null").unwrap(), None);
        assert_eq!(from_str::<Option<u32>>("7").unwrap(), Some(7));
        assert_eq!(from_str::<char>("\"a\"").unwrap(), 'a');
        assert_eq!(from_str::<char>("'\\uD83D\\uDE00'").unwrap(), '\u{1f600}');
        assert_eq!(from_str::<Vec<u8>>("[1,2,3]").unwrap(), vec![1, 2, 3]);
        assert_eq!(from_str::<(u8, u8)>("[1,2]").unwrap(), (1, 2));
        // Range and type mismatches surface as positioned serde errors.
        let err = from_str::<u8>("300").unwrap_err();
        assert!(matches!(&err, Error::Syntax { message, .. } if message.contains("invalid value")));
        let err = from_str::<bool>("1").unwrap_err();
        assert!(matches!(&err, Error::Syntax { message, .. } if message.contains("invalid type")));
        let err = from_str::<String>("true").unwrap_err();
        assert!(matches!(&err, Error::Syntax { message, .. } if message.contains("invalid type")));
    }

    /// 128-bit integer targets parse exactly past the 64-bit variants;
    /// literals beyond the range are number errors, not float type errors.
    #[test]
    fn wide_integer_visitors() {
        assert_eq!(
            from_str::<u128>("18446744073709551616").unwrap(),
            1_u128 << 64
        );
        assert_eq!(
            from_str::<i128>("-9223372036854775809").unwrap(),
            i128::from(i64::MIN) - 1
        );
        assert_eq!(
            from_str::<u128>("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF").unwrap(),
            u128::MAX
        );
        assert_eq!(
            from_str::<u128>("340282366920938463463374607431768211456").unwrap_err(),
            Error::InvalidNumber { offset: 0 }
        );
        let err = from_str::<u128>("1.5").unwrap_err();
        assert!(
            matches!(&err, Error::Syntax { message, .. } if message.contains("invalid type: floating point"))
        );
    }

    /// f32 targets evaluate document numbers at f32 precision in a single
    /// rounding.
    #[test]
    fn f32_targets_single_rounding() {
        assert_eq!(from_str::<f32>("1.5").unwrap(), 1.5);
        assert_eq!(from_str::<f32>("0x10").unwrap(), 16.0);
        assert_eq!(from_str::<f32>("Infinity").unwrap(), f32::INFINITY);
        // The double-rounding boundary: see the number.rs test of the same name.
        assert_eq!(
            from_str::<f32>("1.0000000596046448").unwrap(),
            1.000_000_1_f32
        );
    }

    /// serde visitor errors are re-reported with the read position, so typed
    /// type errors carry offsets like every other parse error.
    #[test]
    fn type_errors_carry_offsets() {
        // After the scalar is consumed the cursor is past end of input.
        let err = from_str::<bool>("1").unwrap_err();
        assert!(
            matches!(&err, Error::Syntax { message, offset: 1 } if message.contains("invalid type"))
        );
        // The visitor gave up with the cursor on the closing brace.
        let err = from_str::<Point>("{x:1}").unwrap_err();
        assert!(
            matches!(&err, Error::Syntax { message, offset: 4 } if message.contains("missing field"))
        );
    }

    #[test]
    fn typed_enums() {
        // Unit variants: any quote style.
        assert_eq!(from_str::<E>(r#""A""#).unwrap(), E::A);
        assert_eq!(from_str::<E>("'A'").unwrap(), E::A);
        // Newtype and struct variants in the map form, quoted or not.
        assert_eq!(from_str::<E>("{B: 1}").unwrap(), E::B(1));
        assert_eq!(from_str::<E>(r#"{"B": 2}"#).unwrap(), E::B(2));
        assert_eq!(from_str::<E>(r#"{"C": {x: 2}}"#).unwrap(), E::C { x: 2 });
        // The map form accepts one trailing comma, like any JSON5 object.
        assert_eq!(from_str::<E>("{B: 1,}").unwrap(), E::B(1));
        assert_eq!(from_str::<E>(r#"{"C": {x: 2},}"#).unwrap(), E::C { x: 2 });
        assert_eq!(from_str::<E>(r#"{"A": null,}"#).unwrap(), E::A);
        // A payload on a unit variant is a type error.
        let err = from_str::<E>(r#"{"A": 1}"#).unwrap_err();
        assert!(matches!(&err, Error::Syntax { message, .. } if message.contains("invalid type")));
        // So is a bare string naming a non-unit variant.
        let err = from_str::<E>(r#""B""#).unwrap_err();
        assert!(matches!(&err, Error::Syntax { message, .. } if message.contains("unit variant")));
        // Empty map and extra members.
        assert!(matches!(
            from_str::<E>("{}").unwrap_err(),
            Error::Syntax { .. }
        ));
        assert_eq!(
            from_str::<E>(r#"{"B":1,"B":2}"#).unwrap_err(),
            Error::TrailingCharacters { offset: 6 }
        );
        // A second comma is still the multi-member error, at the comma.
        assert_eq!(
            from_str::<E>(r#"{"B":1,,}"#).unwrap_err(),
            Error::TrailingCharacters { offset: 6 }
        );
    }

    /// `peek_is_null`'s O(1) check must agree with the token scan for every
    /// terminator class: token bytes extend the literal into an unknown
    /// literal, ASCII separators and whitespace end it, and high-bit bytes
    /// need the Unicode-whitespace classification (NBSP ends the token,
    /// other non-ASCII continues it).
    #[test]
    fn option_null_boundary_tokens() {
        for doc in [
            "nullx",
            "null1",
            "null+",
            "null_",
            "null$",
            "null.",
            "nullé",
            r"null\x41",
            "null\u{a0}x",
        ] {
            assert_eq!(
                from_str::<Option<u8>>(doc).unwrap_err(),
                crate::from_str(doc).unwrap_err(),
                "doc {doc:?}"
            );
        }
        for doc in [
            "null",
            "null ",
            "null\t",
            "null// c",
            "null/* c */",
            "null\u{a0}",
        ] {
            assert_eq!(from_str::<Option<u8>>(doc).unwrap(), None, "doc {doc:?}");
        }
        // Boundary classes inside containers, where the next entry follows.
        assert_eq!(from_str::<Vec<Option<u8>>>("[null,null]").unwrap(), vec![
            None, None
        ]);
    }

    #[test]
    fn ignored_values_still_validate_grammar() {
        #[derive(serde::Deserialize, Debug)]
        struct OnlyA {
            a: u8,
        }
        // The unknown member is skipped but its value must still parse.
        let err = from_str::<OnlyA>(r#"{"a":1,"b":[2,3,{"c":}]}"#).unwrap_err();
        assert_eq!(err, syntax("expected value", 21));
        let ok = from_str::<OnlyA>(r#"{"a":1,"b":[2,3,Infinity]}"#).unwrap();
        assert_eq!(ok.a, 1);
    }
}
