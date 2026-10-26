//! Typed (serde) serialization facade.
//!
//! These entry points drive the same writer as the DOM serializers in
//! [`crate::ser`], producing byte-identical output for equal data. Map keys
//! must be strings; non-finite floats serialize as the JSON5 literals `NaN`,
//! `Infinity` and `-Infinity`.
//!
//! The deserialization side lives in [`crate::de`]; [`from_str`] and
//! [`from_slice`] below delegate to it.
//!
//! ```
//! #[derive(serde::Serialize)]
//! struct Point {
//!     x: i64,
//!     y: i64,
//! }
//!
//! let text = simd_json5::typed::to_string(&Point { x: 1, y: 2 }).unwrap();
//! assert_eq!(text, r#"{"x":1,"y":2}"#);
//! ```

use serde::{Deserialize, Serialize};

use crate::{Result, ser::serde_impls::to_bytes};

/// Serialize any serializable value to compact JSON5 text.
///
/// ```
/// assert_eq!(
///     simd_json5::typed::to_string(&vec![1, 2, 3]).unwrap(),
///     "[1,2,3]"
/// );
/// ```
pub fn to_string<T>(value: &T) -> Result<String>
where
    T: ?Sized + Serialize,
{
    to_bytes(value, false).map(|bytes| {
        // SAFETY: the writer only ever appends valid UTF-8 (see
        // `ser::write_bytes`).
        unsafe { String::from_utf8_unchecked(bytes) }
    })
}

/// Serialize any serializable value to pretty-printed JSON5 text (two-space
/// indent, `": "` after keys, empty containers inline).
pub fn to_string_pretty<T>(value: &T) -> Result<String>
where
    T: ?Sized + Serialize,
{
    to_bytes(value, true).map(|bytes| {
        // SAFETY: the writer only ever appends valid UTF-8 (see
        // `ser::write_bytes`).
        unsafe { String::from_utf8_unchecked(bytes) }
    })
}

/// Serialize any serializable value to compact JSON5 bytes.
pub fn to_vec<T>(value: &T) -> Result<Vec<u8>>
where
    T: ?Sized + Serialize,
{
    to_bytes(value, false)
}

/// Deserialize `T` from a JSON5 document; delegates to [`crate::de`].
///
/// ```
/// #[derive(serde::Deserialize)]
/// struct Point {
///     x: i64,
///     y: i64,
/// }
///
/// let p: Point = simd_json5::typed::from_str("[1, 2]").unwrap();
/// assert_eq!((p.x, p.y), (1, 2));
/// ```
pub fn from_str<'de, T>(s: &'de str) -> Result<T>
where
    T: Deserialize<'de>,
{
    crate::de::from_str(s)
}

/// Deserialize `T` from raw JSON5 bytes; delegates to [`crate::de`].
pub fn from_slice<'de, T>(v: &'de [u8]) -> Result<T>
where
    T: Deserialize<'de>,
{
    crate::de::from_slice(v)
}
