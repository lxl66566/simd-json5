//! JSON5 serialization.
//!
//! Output is strict JSON extended with the JSON5 literals `NaN`, `Infinity`
//! and `-Infinity` for non-finite floats. Keys stay double-quoted; strings are
//! emitted as UTF-8 with only the mandatory escapes (`"`, `\`, control
//! characters below `0x20`).
//!
//! Pretty output uses two-space indentation and `": "` after keys, byte
//! compatible with `serde_json::to_string_pretty` on JSON-representable
//! values.

use crate::value::{MAX_DEPTH, Map, Value};

/// Per-byte escape classification for string contents.
///
/// - `0`: pass the byte through raw (everything `>= 0x20`, which covers all multi-byte UTF-8
///   sequences and U+2028/U+2029);
/// - `1..=7`: index into [`SHORT_ESCAPES`] (`\"` `\\` `\b` `\t` `\n` `\f` `\r`);
/// - `8`: other C0 control, emitted as `\u00XX`.
static ESCAPE: [u8; 256] = build_escape_table();

/// Escape characters for classes `1..=7`: quote, backslash, b, t, n, f, r.
/// Index 0 is unused padding so class numbers index directly.
const SHORT_ESCAPES: [u8; 8] = *b"\x00\"\\btnfr";

/// Class of an arbitrary C0 control byte that has no short escape.
const CLASS_UNICODE: u8 = 8;

/// Indent run for pretty mode: two spaces per level, covering every nesting
/// depth the parser accepts ([`MAX_DEPTH`]); deeper hand-built indentation
/// falls back to writing the remainder byte by byte.
const SPACE_CACHE: [u8; 2 * MAX_DEPTH] = [b' '; 2 * MAX_DEPTH];

const fn build_escape_table() -> [u8; 256] {
    let mut table = [0; 256];
    table[b'"' as usize] = 1;
    table[b'\\' as usize] = 2;
    table[0x08] = 3; // \b
    table[b'\t' as usize] = 4;
    table[b'\n' as usize] = 5;
    table[0x0c] = 6; // \f
    table[b'\r' as usize] = 7;
    let mut byte = 0usize;
    while byte < 0x20 {
        if table[byte] == 0 {
            table[byte] = CLASS_UNICODE;
        }
        byte += 1;
    }
    table
}

/// Output target of [`Writer`]: an owned buffer (DOM and serde entry points)
/// or a streaming formatter adapter (the `Display` impl).
trait Sink {
    fn push(&mut self, byte: u8);

    fn extend(&mut self, bytes: &[u8]);
}

impl Sink for Vec<u8> {
    #[inline]
    fn push(&mut self, byte: u8) {
        Vec::push(self, byte);
    }

    #[inline]
    fn extend(&mut self, bytes: &[u8]) {
        self.extend_from_slice(bytes);
    }
}

impl<S: Sink + ?Sized> Sink for &mut S {
    #[inline]
    fn push(&mut self, byte: u8) {
        (**self).push(byte);
    }

    #[inline]
    fn extend(&mut self, bytes: &[u8]) {
        (**self).extend(bytes);
    }
}

/// Growable output buffer. Growth is amortized through `Vec` doubling; the
/// writer never shrinks or reallocates per token.
///
/// Recursion depth follows the DOM. The parser caps nesting at
/// [`crate::MAX_DEPTH`], so parser-produced values cannot overflow the stack;
/// hand-built values deeper than that are the caller's responsibility (the
/// same holds for `Drop` and `PartialEq` on the DOM).
struct Writer<W: Sink = Vec<u8>> {
    out: W,
    pretty: bool,
    indent: usize,
}

impl<W: Sink> Writer<W> {
    fn write_value(&mut self, value: &Value<'_>) {
        match value {
            Value::Null => self.out.extend(b"null"),
            Value::Bool(true) => self.out.extend(b"true"),
            Value::Bool(false) => self.out.extend(b"false"),
            Value::I64(n) => {
                let mut buffer = itoa::Buffer::new();
                self.out.extend(buffer.format(*n).as_bytes());
            },
            Value::U64(n) => {
                let mut buffer = itoa::Buffer::new();
                self.out.extend(buffer.format(*n).as_bytes());
            },
            Value::F64(n) => self.write_f64(*n),
            Value::Str(s) => self.write_escaped(s.as_str()),
            Value::Array(items) => self.write_array(items),
            Value::Object(members) => self.write_object(members),
        }
    }

    fn write_array(&mut self, items: &[Value<'_>]) {
        self.out.push(b'[');
        let mut first = true;
        for item in items {
            self.before_item(&mut first);
            self.write_value(item);
        }
        self.after_items(first, b']');
    }

    fn write_object(&mut self, members: &Map<'_>) {
        self.out.push(b'{');
        let mut first = true;
        for (key, item) in members {
            self.before_item(&mut first);
            self.write_escaped(key.as_str());
            self.write_key_value_separator();
            self.write_value(item);
        }
        self.after_items(first, b'}');
    }

    /// Emit the separator before a container item. The first item opens the
    /// indent level (pretty mode); later items get a comma. Shared by the DOM
    /// writer and the serde serializer so both produce identical bytes.
    fn before_item(&mut self, first: &mut bool) {
        if *first {
            *first = false;
            self.indent += 1;
        } else {
            self.out.push(b',');
        }
        self.newline_indent();
    }

    /// Close a container after its items. Empty containers close inline
    /// (pretty mode emits no newline inside them).
    fn after_items(&mut self, first: bool, close: u8) {
        if !first {
            self.indent -= 1;
            self.newline_indent();
        }
        self.out.push(close);
    }

    /// `:` after a key, followed by a space in pretty mode.
    fn write_key_value_separator(&mut self) {
        self.out.push(b':');
        if self.pretty {
            self.out.push(b' ');
        }
    }

    /// Open an externally tagged variant wrapper: `{"name":` in compact mode,
    /// `{\n<indent>"name": ` in pretty mode. Closed by [`end_variant`].
    // Only the serde Serializer reaches these; the DOM path never emits
    // variants.
    #[cfg(feature = "serde")]
    fn begin_variant(&mut self, name: &str) {
        self.out.push(b'{');
        self.indent += 1;
        self.newline_indent();
        self.write_escaped(name);
        self.write_key_value_separator();
    }

    /// Close the wrapper opened by [`begin_variant`].
    #[cfg(feature = "serde")]
    fn end_variant(&mut self) {
        self.indent -= 1;
        self.newline_indent();
        self.out.push(b'}');
    }

    /// In pretty mode: newline plus two spaces per indent level. Compact mode:
    /// nothing.
    fn newline_indent(&mut self) {
        if self.pretty {
            self.out.push(b'\n');
            let width = 2 * self.indent;
            let cached = width.min(SPACE_CACHE.len());
            self.out.extend(&SPACE_CACHE[..cached]);
            for _ in 0..width - cached {
                self.out.push(b' ');
            }
        }
    }

    /// Non-finite floats use the JSON5 literals; zmij only sees finite values
    /// because it would emit `NaN`/`inf` otherwise.
    fn write_f64(&mut self, n: f64) {
        if n.is_nan() {
            self.out.extend(b"NaN");
        } else if n.is_infinite() {
            let literal = if n.is_sign_negative() {
                &b"-Infinity"[..]
            } else {
                &b"Infinity"[..]
            };
            self.out.extend(literal);
        } else {
            let mut buffer = zmij::Buffer::new();
            self.out.extend(buffer.format_finite(n).as_bytes());
        }
    }

    /// Finite f32 values take zmij's f32 path: widening to f64 first would
    /// print the shortest f64 representation of the widened value (`0.1f32`
    /// as `0.10000000149011612`) instead of the shortest f32 one (`0.1`),
    /// diverging from serde_json. Non-finite values keep the f64 literal
    /// handling.
    #[cfg_attr(not(feature = "serde"), allow(dead_code))]
    fn write_f32(&mut self, n: f32) {
        if n.is_finite() {
            let mut buffer = zmij::Buffer::new();
            self.out.extend(buffer.format_finite(n).as_bytes());
        } else {
            self.write_f64(f64::from(n));
        }
    }

    fn write_escaped(&mut self, s: &str) {
        self.out.push(b'"');
        let bytes = s.as_bytes();
        let mut start = 0;
        let mut i = 0;
        while i + 8 <= bytes.len() {
            // SWAR pre-check on whole words: a word without any byte in
            // {< 0x20, '"', '\\'} is copied as one slice when it is finally
            // flushed. UTF-8 continuation bytes are >= 0x80, so the < 0x20
            // test never trips mid-codepoint and words are safe to skip
            // across. A flagged word falls back to the byte-wise pass below.
            let w = u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap());
            let dangerous = has_zero(w & 0xe0e0_e0e0_e0e0_e0e0)
                | has_zero(w ^ 0x2222_2222_2222_2222)
                | has_zero(w ^ 0x5c5c_5c5c_5c5c_5c5c);
            if dangerous == 0 {
                i += 8;
            } else {
                start = escape_range(&mut self.out, bytes, i, i + 8, start);
                i += 8;
            }
        }
        start = escape_range(&mut self.out, bytes, i, bytes.len(), start);
        self.out.extend(&bytes[start..]);
        self.out.push(b'"');
    }
}

/// Nonzero in the high bit of every byte that is zero in `x`, the SWAR
/// zero-byte test (`x - 0x01..` borrows through zero bytes). A byte flagged
/// only by an incoming borrow sits above a true zero, so `== 0` is exact for
/// "no dangerous byte" and false positives merely trigger the byte-wise
/// fallback.
const fn has_zero(x: u64) -> u64 {
    x.wrapping_sub(0x0101_0101_0101_0101) & !x & 0x8080_8080_8080_8080
}

/// Byte-wise escape pass over `bytes[from..to]`, appending to `out` and
/// returning the first offset not yet flushed (an escape at `i` flushes
/// through `i`, so the new `start` is `i + 1`).
fn escape_range<S: Sink + ?Sized>(
    out: &mut S,
    bytes: &[u8],
    from: usize,
    to: usize,
    start: usize,
) -> usize {
    let mut start = start;
    for i in from..to {
        let byte = bytes[i];
        let class = ESCAPE[usize::from(byte)];
        if class == 0 {
            continue;
        }
        out.extend(&bytes[start..i]);
        out.push(b'\\');
        match class {
            CLASS_UNICODE => {
                const HEX: &[u8; 16] = b"0123456789abcdef";
                out.extend(b"u00");
                out.push(HEX[usize::from(byte >> 4)]);
                out.push(HEX[usize::from(byte & 0x0f)]);
            },
            short => out.push(SHORT_ESCAPES[usize::from(short)]),
        }
        start = i + 1;
    }
    start
}

/// Streaming sink for the `Display` impl: forwards serializer output to a
/// formatter in fixed-size chunks instead of materializing the whole
/// document. Chunks may split a multi-byte sequence; `flush` hands only
/// complete sequences to `write_str`, keeping an incomplete tail buffered,
/// so every `&str` passed to the formatter is valid without assuming chunk
/// boundaries land between code points.
pub(crate) struct FormatterSink<'f, 'a> {
    f: &'f mut core::fmt::Formatter<'a>,
    buf: [u8; 256],
    len: usize,
    failed: bool,
}

impl Sink for FormatterSink<'_, '_> {
    fn push(&mut self, byte: u8) {
        // After a failed `flush` the buffer stays full (`len == buf.len()`);
        // writing on would index out of bounds. Drop the bytes instead:
        // `finish` reports the failure once the walk completes.
        if self.failed {
            return;
        }
        if self.len == self.buf.len() {
            self.flush();
        }
        self.buf[self.len] = byte;
        self.len += 1;
    }

    fn extend(&mut self, mut bytes: &[u8]) {
        // Same as `push`: once failed, `buf.len() - len == 0` would make
        // every iteration copy zero bytes and loop forever.
        if self.failed {
            return;
        }
        while !bytes.is_empty() {
            let take = core::cmp::min(bytes.len(), self.buf.len() - self.len);
            self.buf[self.len..self.len + take].copy_from_slice(&bytes[..take]);
            self.len += take;
            bytes = &bytes[take..];
            if self.len == self.buf.len() {
                self.flush();
            }
        }
    }
}

impl<'f, 'a> FormatterSink<'f, 'a> {
    pub(crate) fn new(f: &'f mut core::fmt::Formatter<'a>) -> Self {
        Self {
            f,
            buf: [0; 256],
            len: 0,
            failed: false,
        }
    }

    /// Write the longest complete-UTF-8 prefix of the buffer; an incomplete
    /// trailing sequence stays buffered for the next chunk.
    fn flush(&mut self) {
        if self.failed {
            return;
        }
        let complete: &str = match core::str::from_utf8(&self.buf[..self.len]) {
            Ok(s) => s,
            Err(e) => match core::str::from_utf8(&self.buf[..e.valid_up_to()]) {
                // The buffer starts at a code-point boundary of a stream that
                // is valid UTF-8 overall (see `write_bytes`), so the only
                // possible error is a sequence cut at the buffer end; its
                // bytes stay buffered for the next chunk.
                Ok(prefix) if e.error_len().is_none() => prefix,
                // Unreachable for this writer; fail the stream rather than
                // emit invalid text.
                _ => {
                    self.failed = true;
                    return;
                },
            },
        };
        if self.f.write_str(complete).is_err() {
            self.failed = true;
            return;
        }
        let written = complete.len();
        self.buf.copy_within(written..self.len, 0);
        self.len -= written;
    }

    /// Flush the final chunk; with the stream complete, the buffer drains
    /// entirely.
    pub(crate) fn finish(mut self) -> core::fmt::Result {
        self.flush();
        if self.failed || self.len != 0 {
            Err(core::fmt::Error)
        } else {
            Ok(())
        }
    }
}

/// Serialize `value` into `f` chunk by chunk (the `Display` path); the output
/// is valid UTF-8 by construction (see [`write_bytes`]).
pub(crate) fn write_streaming(
    value: &Value<'_>,
    f: &mut core::fmt::Formatter<'_>,
) -> core::fmt::Result {
    let mut sink = FormatterSink::new(f);
    let mut writer = Writer {
        out: &mut sink,
        pretty: false,
        indent: 0,
    };
    writer.write_value(value);
    sink.finish()
}

/// Serialize into a byte buffer. The output is valid UTF-8 by construction:
/// passthrough only copies whole input bytes (input strings are valid UTF-8
/// and no escape can split a multi-byte sequence), and every synthesized
/// escape is ASCII.
fn write_bytes(value: &Value<'_>, pretty: bool) -> Vec<u8> {
    let mut writer = Writer {
        // serde_json's `to_vec` starts at 128 as well: covers small documents
        // outright; a full-size hint is not available here (the DOM does not
        // remember its source size), so larger outputs keep growing by doubling.
        out: Vec::with_capacity(128),
        pretty,
        indent: 0,
    };
    writer.write_value(value);
    writer.out
}

/// Serialize to compact JSON5 text.
///
/// ```
/// use simd_json5::{Map, Value, to_string};
///
/// let mut object = Map::new();
/// object.insert("a".into(), Value::I64(1));
/// object.insert("b".into(), Value::F64(f64::INFINITY));
/// assert_eq!(to_string(&Value::Object(object)), r#"{"a":1,"b":Infinity}"#);
/// ```
#[must_use]
pub fn to_string(value: &Value<'_>) -> String {
    // SAFETY: see `write_bytes`; only valid UTF-8 bytes are ever written.
    unsafe { String::from_utf8_unchecked(write_bytes(value, false)) }
}

/// Serialize to pretty-printed JSON5 text: two-space indentation, `": "` after
/// keys, newline after every container item.
///
/// ```
/// use simd_json5::{Map, Value, to_string_pretty};
///
/// let mut object = Map::new();
/// object.insert("a".into(), Value::I64(1));
/// assert_eq!(to_string_pretty(&Value::Object(object)), "{\n  \"a\": 1\n}");
/// ```
#[must_use]
pub fn to_string_pretty(value: &Value<'_>) -> String {
    // SAFETY: see `write_bytes`; only valid UTF-8 bytes are ever written.
    unsafe { String::from_utf8_unchecked(write_bytes(value, true)) }
}

/// Serialize to compact JSON5 bytes.
#[must_use]
pub fn to_vec(value: &Value<'_>) -> Vec<u8> {
    write_bytes(value, false)
}

/// serde integration (feature `serde`): the [`serde::Serializer`] driving the
/// same [`Writer`] as the DOM entry points, producing byte-identical output.
#[cfg(feature = "serde")]
pub(crate) mod serde_impls {
    use serde::{
        Serialize,
        ser::{self, Error as _, Impossible},
    };

    use super::Writer;
    use crate::{
        Error, Result,
        value::{Str, Value},
    };

    /// Map key and serialization failures onto [`Error::Custom`], matching the
    /// `serde::ser::Error` protocol.
    impl ser::Error for Error {
        fn custom<T: core::fmt::Display>(msg: T) -> Self {
            Self::Custom(msg.to_string().into_boxed_str())
        }
    }

    /// Serialize `value` through the serde machinery into a byte buffer.
    pub(crate) fn to_bytes<T>(value: &T, pretty: bool) -> Result<Vec<u8>>
    where
        T: ?Sized + Serialize,
    {
        let mut writer = Writer {
            // Same starting capacity as `write_bytes` / serde_json's `to_vec`.
            out: Vec::with_capacity(128),
            pretty,
            indent: 0,
        };
        value.serialize(Serializer {
            writer: &mut writer,
        })?;
        Ok(writer.out)
    }

    /// The serde serializer. Errors abort the whole serialization; partial
    /// output is discarded together with the writer.
    pub(crate) struct Serializer<'w> {
        writer: &'w mut Writer,
    }

    macro_rules! forward_int {
        ($($method:ident($ty:ty)),* $(,)?) => {
            $(fn $method(self, v: $ty) -> Result<()> {
                self.serialize_i64(i64::from(v))
            })*
        };
    }

    macro_rules! forward_uint {
        ($($method:ident($ty:ty)),* $(,)?) => {
            $(fn $method(self, v: $ty) -> Result<()> {
                self.serialize_u64(u64::from(v))
            })*
        };
    }

    impl<'w> serde::Serializer for Serializer<'w> {
        type Error = Error;
        type Ok = ();
        type SerializeMap = MapCompound<'w>;
        type SerializeSeq = SeqCompound<'w>;
        type SerializeStruct = MapCompound<'w>;
        type SerializeStructVariant = MapCompound<'w>;
        type SerializeTuple = SeqCompound<'w>;
        type SerializeTupleStruct = SeqCompound<'w>;
        type SerializeTupleVariant = SeqCompound<'w>;

        forward_int!(serialize_i8(i8), serialize_i16(i16), serialize_i32(i32),);

        forward_uint!(serialize_u8(u8), serialize_u16(u16), serialize_u32(u32),);

        fn serialize_i64(self, v: i64) -> Result<()> {
            let mut buffer = itoa::Buffer::new();
            self.writer.out.extend(buffer.format(v).as_bytes());
            Ok(())
        }

        fn serialize_u64(self, v: u64) -> Result<()> {
            let mut buffer = itoa::Buffer::new();
            self.writer.out.extend(buffer.format(v).as_bytes());
            Ok(())
        }

        fn serialize_i128(self, v: i128) -> Result<()> {
            let mut buffer = itoa::Buffer::new();
            self.writer.out.extend(buffer.format(v).as_bytes());
            Ok(())
        }

        fn serialize_u128(self, v: u128) -> Result<()> {
            let mut buffer = itoa::Buffer::new();
            self.writer.out.extend(buffer.format(v).as_bytes());
            Ok(())
        }

        fn serialize_f32(self, v: f32) -> Result<()> {
            self.writer.write_f32(v);
            Ok(())
        }

        fn serialize_f64(self, v: f64) -> Result<()> {
            self.writer.write_f64(v);
            Ok(())
        }

        fn serialize_bool(self, v: bool) -> Result<()> {
            self.writer.out.extend(if v {
                &b"true"[..]
            } else {
                &b"false"[..]
            });
            Ok(())
        }

        fn serialize_char(self, v: char) -> Result<()> {
            let mut buffer = [0; 4];
            self.writer.write_escaped(v.encode_utf8(&mut buffer));
            Ok(())
        }

        fn serialize_str(self, v: &str) -> Result<()> {
            self.writer.write_escaped(v);
            Ok(())
        }

        fn serialize_bytes(self, _v: &[u8]) -> Result<()> {
            Err(Error::custom("binary data is not supported"))
        }

        fn serialize_none(self) -> Result<()> {
            self.serialize_unit()
        }

        fn serialize_some<T>(self, value: &T) -> Result<()>
        where
            T: ?Sized + Serialize,
        {
            value.serialize(self)
        }

        fn serialize_unit(self) -> Result<()> {
            self.writer.out.extend(b"null");
            Ok(())
        }

        fn serialize_unit_struct(self, _name: &'static str) -> Result<()> {
            self.serialize_unit()
        }

        /// Externally tagged unit variants serialize as the variant name.
        fn serialize_unit_variant(
            self,
            _name: &'static str,
            _index: u32,
            variant: &'static str,
        ) -> Result<()> {
            self.writer.write_escaped(variant);
            Ok(())
        }

        fn serialize_newtype_struct<T>(self, _name: &'static str, value: &T) -> Result<()>
        where
            T: ?Sized + Serialize,
        {
            value.serialize(self)
        }

        /// Externally tagged newtype variants serialize as a single-key map.
        fn serialize_newtype_variant<T>(
            self,
            _name: &'static str,
            _index: u32,
            variant: &'static str,
            value: &T,
        ) -> Result<()>
        where
            T: ?Sized + Serialize,
        {
            self.writer.begin_variant(variant);
            value.serialize(Serializer {
                writer: self.writer,
            })?;
            self.writer.end_variant();
            Ok(())
        }

        fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq> {
            self.writer.out.push(b'[');
            Ok(self.seq_compound(false))
        }

        fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple> {
            self.writer.out.push(b'[');
            Ok(self.seq_compound(false))
        }

        fn serialize_tuple_struct(
            self,
            _name: &'static str,
            _len: usize,
        ) -> Result<Self::SerializeTupleStruct> {
            self.writer.out.push(b'[');
            Ok(self.seq_compound(false))
        }

        /// Externally tagged tuple variants serialize as `{"name": [..]}`.
        fn serialize_tuple_variant(
            self,
            _name: &'static str,
            _index: u32,
            variant: &'static str,
            _len: usize,
        ) -> Result<Self::SerializeTupleVariant> {
            self.writer.begin_variant(variant);
            self.writer.out.push(b'[');
            Ok(self.seq_compound(true))
        }

        fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap> {
            self.writer.out.push(b'{');
            Ok(self.map_compound(false))
        }

        fn serialize_struct(
            self,
            _name: &'static str,
            _len: usize,
        ) -> Result<Self::SerializeStruct> {
            self.writer.out.push(b'{');
            Ok(self.map_compound(false))
        }

        /// Externally tagged struct variants serialize as `{"name": {..}}`.
        fn serialize_struct_variant(
            self,
            _name: &'static str,
            _index: u32,
            variant: &'static str,
            _len: usize,
        ) -> Result<Self::SerializeStructVariant> {
            self.writer.begin_variant(variant);
            self.writer.out.push(b'{');
            Ok(self.map_compound(true))
        }
    }

    impl<'w> Serializer<'w> {
        fn seq_compound(self, variant: bool) -> SeqCompound<'w> {
            SeqCompound {
                writer: self.writer,
                first: true,
                variant,
            }
        }

        fn map_compound(self, variant: bool) -> MapCompound<'w> {
            MapCompound {
                writer: self.writer,
                first: true,
                variant,
            }
        }
    }

    /// Array-like containers: sequences, tuples, tuple structs and (with
    /// `variant`) externally tagged tuple variants.
    pub(crate) struct SeqCompound<'w> {
        writer: &'w mut Writer,
        first: bool,
        variant: bool,
    }

    impl ser::SerializeSeq for SeqCompound<'_> {
        type Error = Error;
        type Ok = ();

        fn serialize_element<T>(&mut self, value: &T) -> Result<()>
        where
            T: ?Sized + Serialize,
        {
            self.writer.before_item(&mut self.first);
            value.serialize(Serializer {
                writer: self.writer,
            })
        }

        fn end(self) -> Result<()> {
            self.writer.after_items(self.first, b']');
            self.close_variant();
            Ok(())
        }
    }

    impl ser::SerializeTuple for SeqCompound<'_> {
        type Error = Error;
        type Ok = ();

        fn serialize_element<T>(&mut self, value: &T) -> Result<()>
        where
            T: ?Sized + Serialize,
        {
            ser::SerializeSeq::serialize_element(self, value)
        }

        fn end(self) -> Result<()> {
            ser::SerializeSeq::end(self)
        }
    }

    impl ser::SerializeTupleStruct for SeqCompound<'_> {
        type Error = Error;
        type Ok = ();

        fn serialize_field<T>(&mut self, value: &T) -> Result<()>
        where
            T: ?Sized + Serialize,
        {
            ser::SerializeSeq::serialize_element(self, value)
        }

        fn end(self) -> Result<()> {
            ser::SerializeSeq::end(self)
        }
    }

    impl ser::SerializeTupleVariant for SeqCompound<'_> {
        type Error = Error;
        type Ok = ();

        fn serialize_field<T>(&mut self, value: &T) -> Result<()>
        where
            T: ?Sized + Serialize,
        {
            ser::SerializeSeq::serialize_element(self, value)
        }

        fn end(self) -> Result<()> {
            ser::SerializeSeq::end(self)
        }
    }

    impl SeqCompound<'_> {
        /// Close the `{"name":` wrapper opened by `begin_variant`.
        fn close_variant(self) {
            if self.variant {
                self.writer.end_variant();
            }
        }
    }

    /// Object-like containers: maps, structs and (with `variant`) externally
    /// tagged struct variants.
    pub(crate) struct MapCompound<'w> {
        writer: &'w mut Writer,
        first: bool,
        variant: bool,
    }

    impl ser::SerializeMap for MapCompound<'_> {
        type Error = Error;
        type Ok = ();

        fn serialize_key<T>(&mut self, key: &T) -> Result<()>
        where
            T: ?Sized + Serialize,
        {
            self.writer.before_item(&mut self.first);
            key.serialize(KeySerializer {
                writer: self.writer,
            })
        }

        fn serialize_value<T>(&mut self, value: &T) -> Result<()>
        where
            T: ?Sized + Serialize,
        {
            self.writer.write_key_value_separator();
            value.serialize(Serializer {
                writer: self.writer,
            })
        }

        fn end(self) -> Result<()> {
            self.writer.after_items(self.first, b'}');
            self.close_variant();
            Ok(())
        }
    }

    impl ser::SerializeStruct for MapCompound<'_> {
        type Error = Error;
        type Ok = ();

        fn serialize_field<T>(&mut self, key: &'static str, value: &T) -> Result<()>
        where
            T: ?Sized + Serialize,
        {
            self.writer.before_item(&mut self.first);
            self.writer.write_escaped(key);
            self.writer.write_key_value_separator();
            value.serialize(Serializer {
                writer: self.writer,
            })
        }

        fn end(self) -> Result<()> {
            self.writer.after_items(self.first, b'}');
            self.close_variant();
            Ok(())
        }
    }

    impl ser::SerializeStructVariant for MapCompound<'_> {
        type Error = Error;
        type Ok = ();

        fn serialize_field<T>(&mut self, key: &'static str, value: &T) -> Result<()>
        where
            T: ?Sized + Serialize,
        {
            ser::SerializeStruct::serialize_field(self, key, value)
        }

        fn end(self) -> Result<()> {
            ser::SerializeStruct::end(self)
        }
    }

    impl MapCompound<'_> {
        fn close_variant(self) {
            if self.variant {
                self.writer.end_variant();
            }
        }
    }

    /// Key-only serializer. Map keys are restricted to strings and string-like
    /// scalars (chars, unit variant names); everything else is rejected
    /// through [`Error::Custom`].
    struct KeySerializer<'w> {
        writer: &'w mut Writer,
    }

    fn unsupported_key<T>() -> Result<T> {
        Err(Error::custom("map keys must be strings"))
    }

    /// Rejects every non-string key type with the same message.
    macro_rules! reject {
        ($($method:ident($($arg:ty),*)),* $(,)?) => {
            $(fn $method(self, $(_: $arg),*) -> Result<()> {
                unsupported_key()
            })*
        };
    }

    impl serde::Serializer for KeySerializer<'_> {
        type Error = Error;
        type Ok = ();
        type SerializeMap = Impossible<(), Error>;
        type SerializeSeq = Impossible<(), Error>;
        type SerializeStruct = Impossible<(), Error>;
        type SerializeStructVariant = Impossible<(), Error>;
        type SerializeTuple = Impossible<(), Error>;
        type SerializeTupleStruct = Impossible<(), Error>;
        type SerializeTupleVariant = Impossible<(), Error>;

        reject!(
            serialize_bool(bool),
            serialize_i8(i8),
            serialize_i16(i16),
            serialize_i32(i32),
            serialize_i64(i64),
            serialize_i128(i128),
            serialize_u8(u8),
            serialize_u16(u16),
            serialize_u32(u32),
            serialize_u64(u64),
            serialize_u128(u128),
            serialize_f32(f32),
            serialize_f64(f64),
            serialize_bytes(&[u8]),
        );

        fn serialize_str(self, v: &str) -> Result<()> {
            self.writer.write_escaped(v);
            Ok(())
        }

        fn serialize_char(self, v: char) -> Result<()> {
            let mut buffer = [0; 4];
            self.writer.write_escaped(v.encode_utf8(&mut buffer));
            Ok(())
        }

        /// Enum keys serialize as the variant name.
        fn serialize_unit_variant(
            self,
            _name: &'static str,
            _index: u32,
            variant: &'static str,
        ) -> Result<()> {
            self.writer.write_escaped(variant);
            Ok(())
        }

        fn serialize_none(self) -> Result<()> {
            unsupported_key()
        }

        fn serialize_some<T>(self, _value: &T) -> Result<()>
        where
            T: ?Sized + Serialize,
        {
            unsupported_key()
        }

        fn serialize_unit(self) -> Result<()> {
            unsupported_key()
        }

        fn serialize_unit_struct(self, _name: &'static str) -> Result<()> {
            unsupported_key()
        }

        fn serialize_newtype_struct<T>(self, _name: &'static str, _value: &T) -> Result<()>
        where
            T: ?Sized + Serialize,
        {
            unsupported_key()
        }

        fn serialize_newtype_variant<T>(
            self,
            _name: &'static str,
            _index: u32,
            _variant: &'static str,
            _value: &T,
        ) -> Result<()>
        where
            T: ?Sized + Serialize,
        {
            unsupported_key()
        }

        fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq> {
            unsupported_key()
        }

        fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple> {
            unsupported_key()
        }

        fn serialize_tuple_struct(
            self,
            _name: &'static str,
            _len: usize,
        ) -> Result<Self::SerializeTupleStruct> {
            unsupported_key()
        }

        fn serialize_tuple_variant(
            self,
            _name: &'static str,
            _index: u32,
            _variant: &'static str,
            _len: usize,
        ) -> Result<Self::SerializeTupleVariant> {
            unsupported_key()
        }

        fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap> {
            unsupported_key()
        }

        fn serialize_struct(
            self,
            _name: &'static str,
            _len: usize,
        ) -> Result<Self::SerializeStruct> {
            unsupported_key()
        }

        fn serialize_struct_variant(
            self,
            _name: &'static str,
            _index: u32,
            _variant: &'static str,
            _len: usize,
        ) -> Result<Self::SerializeStructVariant> {
            unsupported_key()
        }
    }

    impl Serialize for Str<'_> {
        fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            serializer.serialize_str(self.as_str())
        }
    }

    /// The DOM serializes through the same wire format as
    /// [`crate::ser::to_string`]; routing it through any serde serializer
    /// (including serde_json's) yields the identical DOM representation.
    impl Serialize for Value<'_> {
        fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            use serde::ser::SerializeMap as _;

            match self {
                Value::Null => serializer.serialize_unit(),
                Value::Bool(b) => serializer.serialize_bool(*b),
                Value::I64(n) => serializer.serialize_i64(*n),
                Value::U64(n) => serializer.serialize_u64(*n),
                Value::F64(n) => serializer.serialize_f64(*n),
                Value::Str(s) => serializer.serialize_str(s.as_str()),
                Value::Array(items) => items.serialize(serializer),
                Value::Object(members) => {
                    let mut map = serializer.serialize_map(Some(members.len()))?;
                    for (key, value) in members {
                        map.serialize_entry(key.as_str(), value)?;
                    }
                    map.end()
                },
            }
        }
    }
}

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    use std::collections::BTreeMap;

    use serde::Serialize;

    use super::serde_impls::to_bytes;

    fn compact<T: ?Sized + Serialize>(value: &T) -> String {
        String::from_utf8(to_bytes(value, false).unwrap()).unwrap()
    }

    fn pretty<T: ?Sized + Serialize>(value: &T) -> String {
        String::from_utf8(to_bytes(value, true).unwrap()).unwrap()
    }

    #[derive(Serialize)]
    struct Nested {
        name: &'static str,
        tags: Vec<&'static str>,
        empty: Vec<u8>,
        point: Option<(i32, i32)>,
        missing: Option<u8>,
    }

    #[derive(Serialize)]
    enum Shape {
        Unit,
        Newtype(f64),
        Tuple(f64, f64),
        Struct { radius: u32 },
    }

    #[derive(Serialize)]
    struct Unit;

    #[derive(Serialize)]
    struct Newtype(i64);

    #[derive(Serialize)]
    struct TupleStruct(char, bool);

    #[test]
    fn structs_and_scalars() {
        let value = Nested {
            name: "a\"b",
            tags: vec!["x", "y"],
            empty: vec![],
            point: Some((1, -2)),
            missing: None,
        };
        assert_eq!(
            compact(&value),
            r#"{"name":"a\"b","tags":["x","y"],"empty":[],"point":[1,-2],"missing":null}"#
        );
        assert_eq!(compact(&Unit), "null");
        assert_eq!(compact(&Newtype(i64::MIN)), "-9223372036854775808");
        assert_eq!(compact(&TupleStruct('é', true)), r#"["é",true]"#);
        assert_eq!(
            compact(&u128::MAX),
            "340282366920938463463374607431768211455"
        );
        assert_eq!(compact(&(-1.5f32)), "-1.5");
        // f32 takes zmij's f32 path: `0.1f32` stays `0.1`, not the widened
        // `0.10000000149011612`.
        assert_eq!(compact(&0.1f32), "0.1");
        assert_eq!(compact(&f32::NAN), "NaN");
        assert_eq!(compact(&f32::NEG_INFINITY), "-Infinity");
    }

    #[test]
    fn enum_forms() {
        assert_eq!(compact(&Shape::Unit), r#""Unit""#);
        assert_eq!(compact(&Shape::Newtype(0.5)), r#"{"Newtype":0.5}"#);
        assert_eq!(compact(&Shape::Tuple(1.0, 2.0)), r#"{"Tuple":[1.0,2.0]}"#);
        assert_eq!(
            compact(&Shape::Struct { radius: 3 }),
            r#"{"Struct":{"radius":3}}"#
        );
    }

    #[test]
    fn non_finite_literals() {
        assert_eq!(compact(&f64::NAN), "NaN");
        assert_eq!(compact(&f64::INFINITY), "Infinity");
        assert_eq!(compact(&f64::NEG_INFINITY), "-Infinity");
        assert_eq!(compact(&vec![f64::NAN]), "[NaN]");
    }

    #[test]
    fn pretty_layout() {
        assert_eq!(pretty(&Vec::<u8>::new()), "[]");
        assert_eq!(
            pretty(&Shape::Struct { radius: 3 }),
            "{\n  \"Struct\": {\n    \"radius\": 3\n  }\n}"
        );
        assert_eq!(
            pretty(&Shape::Tuple(1.0, 2.0)),
            "{\n  \"Tuple\": [\n    1.0,\n    2.0\n  ]\n}"
        );
        assert_eq!(pretty(&Shape::Unit), "\"Unit\"");
        let mut map = BTreeMap::new();
        map.insert("k", vec![1]);
        map.insert("empty", vec![]);
        assert_eq!(
            pretty(&map),
            "{\n  \"empty\": [],\n  \"k\": [\n    1\n  ]\n}"
        );
    }

    #[test]
    fn map_keys() {
        let mut map = BTreeMap::new();
        map.insert("k", 1);
        assert_eq!(compact(&map), r#"{"k":1}"#);
        let mut int_keyed = BTreeMap::new();
        int_keyed.insert(7u64, "v");
        let error = to_bytes(&int_keyed, false).unwrap_err();
        assert!(
            matches!(error, crate::Error::Custom(ref message) if &**message == "map keys must be strings"),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn bytes_rejected() {
        // `&[u8]` serializes as a number sequence; `serialize_bytes` is only
        // reached by manual impls (or serde_bytes), so drive it directly.
        struct RawBytes;
        impl Serialize for RawBytes {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                serializer.serialize_bytes(b"x")
            }
        }

        let error = to_bytes(&RawBytes, false).unwrap_err();
        assert!(
            matches!(error, crate::Error::Custom(ref message) if &**message == "binary data is not supported"),
            "unexpected error: {error:?}"
        );
    }
}
