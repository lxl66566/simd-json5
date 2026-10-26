//! Error type for parsing and serialization failures.

/// Convenience alias used throughout the crate.
pub type Result<T> = core::result::Result<T, Error>;

/// Parse or serialization error. Offsets are byte positions into the input.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("unexpected end of input at offset {offset}")]
    UnexpectedEof { offset: usize },

    #[error("unexpected byte {byte:#04x} at offset {offset}")]
    UnexpectedChar { byte: u8, offset: usize },

    #[error("invalid number at offset {offset}")]
    InvalidNumber { offset: usize },

    #[error("invalid escape sequence at offset {offset}")]
    InvalidEscape { offset: usize },

    #[error("invalid unicode escape sequence at offset {offset}")]
    InvalidUnicodeEscape { offset: usize },

    #[error("unterminated string starting at offset {offset}")]
    UnterminatedString { offset: usize },

    #[error("unterminated comment starting at offset {offset}")]
    UnterminatedComment { offset: usize },

    #[error("trailing characters at offset {offset}")]
    TrailingCharacters { offset: usize },

    #[error("input is not valid utf-8 before offset {offset}")]
    InvalidUtf8 { offset: usize },

    #[error("nesting depth {limit} exceeded at offset {offset}")]
    DepthLimitExceeded { limit: usize, offset: usize },

    #[error("{message} at offset {offset}")]
    Syntax { message: Box<str>, offset: usize },

    #[error("{0}")]
    Custom(Box<str>),
}

impl Error {
    /// Byte offset into the input where the error was detected, if known.
    #[must_use]
    pub fn offset(&self) -> Option<usize> {
        match *self {
            Self::UnexpectedEof { offset }
            | Self::UnexpectedChar { offset, .. }
            | Self::InvalidNumber { offset }
            | Self::InvalidEscape { offset }
            | Self::InvalidUnicodeEscape { offset }
            | Self::UnterminatedString { offset }
            | Self::UnterminatedComment { offset }
            | Self::TrailingCharacters { offset }
            | Self::InvalidUtf8 { offset }
            | Self::DepthLimitExceeded { offset, .. }
            | Self::Syntax { offset, .. } => Some(offset),
            Self::Custom(_) => None,
        }
    }
}
