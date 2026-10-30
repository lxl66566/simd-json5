//! SIMD-accelerated JSON5 parser and serializer.
//!
//! Parsing runs in two stages: a SIMD scan that indexes structural characters
//! (comments, unquoted keys and single-quoted strings included), then a
//! semantic pass that builds a zero-copy [`Value`] borrowing from the input.
//! The output side emits strict JSON extended with the JSON5 `NaN` /
//! `Infinity` literals, so serialized data is valid JSON whenever the value
//! contains no non-finite float.
//!
//! Typed serialization (default feature `serde`) accepts only string-like map
//! keys; integer, bool or float keys are rejected with an error instead of
//! being rendered as quoted strings the way `serde_json` does.
//!
//! For point lookups, [`get`] locates a value by a path of object keys and
//! array indices without building a DOM. The walk validates the document's
//! structure; the returned [`LazyValue`] validates its own grammar through
//! its accessors, so skipped members are never parsed.
//!
//! ```
//! let value = simd_json5::from_str("{ a: 1, // comment\n  b: [2, 3,] }").unwrap();
//! assert_eq!(value["b"][0], simd_json5::Value::U64(2));
//!
//! let b = simd_json5::get("{ a: 1, b: [2, 3,] }", &["b", "1"]).unwrap().unwrap();
//! assert_eq!(b.as_i64().unwrap(), Some(3));
//! ```

pub mod error;
pub mod ser;
pub mod value;

#[cfg(feature = "serde")]
pub mod de;
#[cfg(feature = "serde")]
pub mod typed;

mod lazy;
mod parser;

pub use error::{Error, Result};
pub use lazy::{LazyValue, get};
pub use ser::{to_string, to_string_pretty, to_vec};
pub use value::{MAX_DEPTH, Map, Str, Value};

/// Parse a JSON5 document, borrowing string data from `s` where possible.
pub fn from_str(s: &str) -> Result<Value<'_>> {
    parser::parse(s.as_bytes())
}

/// Parse a JSON5 document from raw bytes; fails if `v` is not valid UTF-8.
pub fn from_slice(v: &[u8]) -> Result<Value<'_>> {
    let s = simdutf8::compat::from_utf8(v).map_err(|e| Error::InvalidUtf8 {
        offset: e.valid_up_to(),
    })?;
    parser::parse(s.as_bytes())
}
