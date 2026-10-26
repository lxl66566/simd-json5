//! DOM for parsed JSON5 documents.

use core::{
    borrow::Borrow,
    fmt,
    hash::{Hash, Hasher},
    ops::Index,
};

use indexmap::IndexMap;

/// Object map preserving insertion order, matching JavaScript semantics.
/// Duplicate keys keep the position of the first occurrence and the value of
/// the last.
pub type Map<'a> = IndexMap<Str<'a>, Value<'a>>;

/// Maximum container nesting accepted by the parser. Bounds stack usage of
/// recursive operations (drop, equality, serde deserialization) on the DOM.
pub const MAX_DEPTH: usize = 1024;

/// A JSON5 string. Borrows from the input when no unescaping was needed and
/// owns an allocation otherwise.
#[derive(Debug, Clone, Eq)]
pub enum Str<'a> {
    /// Slice of the original input; the source contained no escapes.
    Static(&'a str),
    /// Unescaped string built from a source containing escapes.
    Dynamic(String),
}

impl Default for Str<'_> {
    fn default() -> Self {
        Self::Static("")
    }
}

impl Str<'_> {
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Static(s) => s,
            Self::Dynamic(s) => s,
        }
    }

    #[must_use]
    pub fn into_owned(self) -> String {
        match self {
            Self::Static(s) => s.to_owned(),
            Self::Dynamic(s) => s,
        }
    }
}

impl PartialEq for Str<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

impl Hash for Str<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}

impl fmt::Display for Str<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl core::ops::Deref for Str<'_> {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

impl AsRef<str> for Str<'_> {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl Borrow<str> for Str<'_> {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl<'a> From<&'a str> for Str<'a> {
    fn from(s: &'a str) -> Self {
        Self::Static(s)
    }
}

impl From<String> for Str<'_> {
    fn from(s: String) -> Self {
        Self::Dynamic(s)
    }
}

/// A JSON5 value. Numbers are stored in the narrowest lossless variant.
#[derive(Debug, Clone, Default)]
pub enum Value<'a> {
    #[default]
    Null,
    Bool(bool),
    I64(i64),
    U64(u64),
    F64(f64),
    Str(Str<'a>),
    Array(Vec<Value<'a>>),
    Object(Map<'a>),
}

/// Equality compares string content and numerically equal integer variants
/// (`I64(1) == U64(1)`). Floats never equal integers; `F64(NaN)` equals
/// nothing.
impl PartialEq for Value<'_> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Null, Self::Null) => true,
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::I64(a), Self::I64(b)) => a == b,
            (Self::U64(a), Self::U64(b)) => a == b,
            (Self::I64(a), Self::U64(b)) | (Self::U64(b), Self::I64(a)) => {
                u64::try_from(*a) == Ok(*b)
            },
            (Self::F64(a), Self::F64(b)) => a == b,
            (Self::Str(a), Self::Str(b)) => a == b,
            (Self::Array(a), Self::Array(b)) => a == b,
            (Self::Object(a), Self::Object(b)) => a == b,
            _ => false,
        }
    }
}

impl<'a> Value<'a> {
    #[must_use]
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    #[must_use]
    pub fn is_bool(&self) -> bool {
        matches!(self, Self::Bool(_))
    }

    #[must_use]
    pub fn is_number(&self) -> bool {
        matches!(self, Self::I64(_) | Self::U64(_) | Self::F64(_))
    }

    #[must_use]
    pub fn is_str(&self) -> bool {
        matches!(self, Self::Str(_))
    }

    #[must_use]
    pub fn is_array(&self) -> bool {
        matches!(self, Self::Array(_))
    }

    #[must_use]
    pub fn is_object(&self) -> bool {
        matches!(self, Self::Object(_))
    }

    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::I64(i) => Some(*i),
            Self::U64(u) => i64::try_from(*u).ok(),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U64(u) => Some(*u),
            Self::I64(i) => u64::try_from(*i).ok(),
            _ => None,
        }
    }

    #[must_use]
    #[allow(clippy::cast_precision_loss)] // widening integers to f64 matches serde_json
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::I64(i) => Some(*i as f64),
            Self::U64(u) => Some(*u as f64),
            Self::F64(f) => Some(*f),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_array(&self) -> Option<&[Value<'a>]> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_array_mut(&mut self) -> Option<&mut Vec<Value<'a>>> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_object(&self) -> Option<&Map<'a>> {
        match self {
            Self::Object(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_object_mut(&mut self) -> Option<&mut Map<'a>> {
        match self {
            Self::Object(m) => Some(m),
            _ => None,
        }
    }

    /// Object member lookup by key, or array element lookup by index.
    pub fn get(&self, key: impl Borrow<str>) -> Option<&Value<'a>> {
        match self {
            Self::Object(m) => m.get(key.borrow()),
            _ => None,
        }
    }

    #[must_use]
    pub fn get_index(&self, index: usize) -> Option<&Value<'a>> {
        match self {
            Self::Array(v) => v.get(index),
            _ => None,
        }
    }

    /// Deep-clone into a value that owns all its data, dropping the borrow on
    /// the input.
    pub fn into_owned(self) -> Value<'static> {
        match self {
            Self::Null => Value::Null,
            Self::Bool(b) => Value::Bool(b),
            Self::I64(i) => Value::I64(i),
            Self::U64(u) => Value::U64(u),
            Self::F64(f) => Value::F64(f),
            Self::Str(s) => Value::Str(Str::Dynamic(s.into_owned())),
            Self::Array(v) => Value::Array(v.into_iter().map(Value::into_owned).collect()),
            Self::Object(m) => Value::Object(
                m.into_iter()
                    .map(|(k, v)| (Str::Dynamic(k.into_owned()), v.into_owned()))
                    .collect(),
            ),
        }
    }
}

impl<'a> Index<&str> for Value<'a> {
    type Output = Value<'a>;

    /// Missing object members and indexing a non-object both yield
    /// [`Value::Null`].
    fn index(&self, key: &str) -> &Self::Output {
        self.get(key).unwrap_or(&Value::Null)
    }
}

impl<'a> Index<usize> for Value<'a> {
    type Output = Value<'a>;

    /// Out-of-range array indices and indexing a non-array both yield
    /// [`Value::Null`].
    fn index(&self, index: usize) -> &Self::Output {
        self.get_index(index).unwrap_or(&Value::Null)
    }
}

impl From<bool> for Value<'_> {
    fn from(b: bool) -> Self {
        Self::Bool(b)
    }
}

impl From<i64> for Value<'_> {
    fn from(i: i64) -> Self {
        Self::I64(i)
    }
}

impl From<u64> for Value<'_> {
    fn from(u: u64) -> Self {
        Self::U64(u)
    }
}

impl From<f64> for Value<'_> {
    fn from(f: f64) -> Self {
        Self::F64(f)
    }
}

impl<'a> From<Str<'a>> for Value<'a> {
    fn from(s: Str<'a>) -> Self {
        Self::Str(s)
    }
}

impl<'a> From<&'a str> for Value<'a> {
    fn from(s: &'a str) -> Self {
        Self::Str(Str::Static(s))
    }
}

impl<'a> From<Vec<Value<'a>>> for Value<'a> {
    fn from(v: Vec<Value<'a>>) -> Self {
        Self::Array(v)
    }
}

impl<'a> From<Map<'a>> for Value<'a> {
    fn from(m: Map<'a>) -> Self {
        Self::Object(m)
    }
}

// ---------------------------------------------------------------------------
// Serialize-side impls. Everything below is owned by the serializer branch;
// keep it clearly separated from the DOM definition above.
// ---------------------------------------------------------------------------

/// Compact serialization, identical to [`crate::to_string`], streamed into
/// the formatter in chunks instead of materializing the whole document in a
/// temporary `String` first.
impl fmt::Display for Value<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        crate::ser::write_streaming(self, f)
    }
}
