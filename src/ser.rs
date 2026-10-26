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

use crate::value::{Map, Value};

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

/// Growable output buffer. Growth is amortized through `Vec` doubling; the
/// writer never shrinks or reallocates per token.
///
/// Recursion depth follows the DOM. The parser caps nesting at
/// [`crate::MAX_DEPTH`], so parser-produced values cannot overflow the stack;
/// hand-built values deeper than that are the caller's responsibility (the
/// same holds for `Drop` and `PartialEq` on the DOM).
struct Writer {
    buf: Vec<u8>,
    pretty: bool,
    indent: usize,
}

impl Writer {
    fn write_value(&mut self, value: &Value<'_>) {
        match value {
            Value::Null => self.buf.extend_from_slice(b"null"),
            Value::Bool(true) => self.buf.extend_from_slice(b"true"),
            Value::Bool(false) => self.buf.extend_from_slice(b"false"),
            Value::I64(n) => {
                let mut buffer = itoa::Buffer::new();
                self.buf.extend_from_slice(buffer.format(*n).as_bytes());
            },
            Value::U64(n) => {
                let mut buffer = itoa::Buffer::new();
                self.buf.extend_from_slice(buffer.format(*n).as_bytes());
            },
            Value::F64(n) => self.write_f64(*n),
            Value::Str(s) => self.write_escaped(s.as_str()),
            Value::Array(items) => self.write_array(items),
            Value::Object(members) => self.write_object(members),
        }
    }

    fn write_array(&mut self, items: &[Value<'_>]) {
        self.buf.push(b'[');
        let mut first = true;
        for item in items {
            self.before_item(&mut first);
            self.write_value(item);
        }
        self.after_items(first, b']');
    }

    fn write_object(&mut self, members: &Map<'_>) {
        self.buf.push(b'{');
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
            self.buf.push(b',');
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
        self.buf.push(close);
    }

    /// `:` after a key, followed by a space in pretty mode.
    fn write_key_value_separator(&mut self) {
        self.buf.push(b':');
        if self.pretty {
            self.buf.push(b' ');
        }
    }

    /// Open an externally tagged variant wrapper: `{"name":` in compact mode,
    /// `{\n<indent>"name": ` in pretty mode. Closed by [`end_variant`].
    // Only the serde Serializer reaches these; the DOM path never emits
    // variants.
    #[cfg(feature = "serde")]
    fn begin_variant(&mut self, name: &str) {
        self.buf.push(b'{');
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
        self.buf.push(b'}');
    }

    /// In pretty mode: newline plus two spaces per indent level. Compact mode:
    /// nothing.
    fn newline_indent(&mut self) {
        if self.pretty {
            self.buf.push(b'\n');
            self.buf.extend(core::iter::repeat_n(b' ', 2 * self.indent));
        }
    }

    /// Non-finite floats use the JSON5 literals; ryu only sees finite values
    /// because it would emit `NaN`/`inf` otherwise.
    fn write_f64(&mut self, n: f64) {
        if n.is_nan() {
            self.buf.extend_from_slice(b"NaN");
        } else if n.is_infinite() {
            let literal = if n.is_sign_negative() {
                &b"-Infinity"[..]
            } else {
                &b"Infinity"[..]
            };
            self.buf.extend_from_slice(literal);
        } else {
            let mut buffer = ryu::Buffer::new();
            self.buf
                .extend_from_slice(buffer.format_finite(n).as_bytes());
        }
    }

    fn write_escaped(&mut self, s: &str) {
        self.buf.push(b'"');
        let bytes = s.as_bytes();
        let mut start = 0;
        for (i, &byte) in bytes.iter().enumerate() {
            let class = ESCAPE[usize::from(byte)];
            if class == 0 {
                continue;
            }
            self.buf.extend_from_slice(&bytes[start..i]);
            self.buf.push(b'\\');
            match class {
                CLASS_UNICODE => {
                    const HEX: &[u8; 16] = b"0123456789abcdef";
                    self.buf.extend_from_slice(b"u00");
                    self.buf.push(HEX[usize::from(byte >> 4)]);
                    self.buf.push(HEX[usize::from(byte & 0x0f)]);
                },
                short => self.buf.push(SHORT_ESCAPES[usize::from(short)]),
            }
            start = i + 1;
        }
        self.buf.extend_from_slice(&bytes[start..]);
        self.buf.push(b'"');
    }
}

/// Serialize into a byte buffer. The output is valid UTF-8 by construction:
/// passthrough only copies whole input bytes (input strings are valid UTF-8
/// and no escape can split a multi-byte sequence), and every synthesized
/// escape is ASCII.
fn write_bytes(value: &Value<'_>, pretty: bool) -> Vec<u8> {
    let mut writer = Writer {
        buf: Vec::new(),
        pretty,
        indent: 0,
    };
    writer.write_value(value);
    writer.buf
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
            buf: Vec::new(),
            pretty,
            indent: 0,
        };
        value.serialize(Serializer {
            writer: &mut writer,
        })?;
        Ok(writer.buf)
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
            self.writer
                .buf
                .extend_from_slice(buffer.format(v).as_bytes());
            Ok(())
        }

        fn serialize_u64(self, v: u64) -> Result<()> {
            let mut buffer = itoa::Buffer::new();
            self.writer
                .buf
                .extend_from_slice(buffer.format(v).as_bytes());
            Ok(())
        }

        fn serialize_i128(self, v: i128) -> Result<()> {
            let mut buffer = itoa::Buffer::new();
            self.writer
                .buf
                .extend_from_slice(buffer.format(v).as_bytes());
            Ok(())
        }

        fn serialize_u128(self, v: u128) -> Result<()> {
            let mut buffer = itoa::Buffer::new();
            self.writer
                .buf
                .extend_from_slice(buffer.format(v).as_bytes());
            Ok(())
        }

        fn serialize_f32(self, v: f32) -> Result<()> {
            self.serialize_f64(f64::from(v))
        }

        fn serialize_f64(self, v: f64) -> Result<()> {
            self.writer.write_f64(v);
            Ok(())
        }

        fn serialize_bool(self, v: bool) -> Result<()> {
            self.writer.buf.extend_from_slice(if v {
                b"true"
            } else {
                b"false"
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
            self.writer.buf.extend_from_slice(b"null");
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
            self.writer.buf.push(b'[');
            Ok(self.seq_compound(false))
        }

        fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple> {
            self.writer.buf.push(b'[');
            Ok(self.seq_compound(false))
        }

        fn serialize_tuple_struct(
            self,
            _name: &'static str,
            _len: usize,
        ) -> Result<Self::SerializeTupleStruct> {
            self.writer.buf.push(b'[');
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
            self.writer.buf.push(b'[');
            Ok(self.seq_compound(true))
        }

        fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap> {
            self.writer.buf.push(b'{');
            Ok(self.map_compound(false))
        }

        fn serialize_struct(
            self,
            _name: &'static str,
            _len: usize,
        ) -> Result<Self::SerializeStruct> {
            self.writer.buf.push(b'{');
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
            self.writer.buf.push(b'{');
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
