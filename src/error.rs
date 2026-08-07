//! Unified error type for HOS.
//!
//! Loading, parsing, and the self-running interpreter return `Result<_,
//! HosError>` instead of panicking — a malformed model file, a truncated GGUF,
//! an unsupported quant type, or a bad arch spec degrades into a clean error
//! the caller can report, never an aborted process. (Genuine internal-invariant
//! violations still panic; those are bugs, not bad input.)

use std::fmt;

#[derive(Debug)]
pub enum HosError {
    /// Underlying I/O failure (file open/read/write).
    Io(std::io::Error),
    /// Malformed container: bad magic, truncation, unexpected EOF, bad version.
    Format(String),
    /// A required tensor was not present in the file.
    MissingTensor(String),
    /// Tensor uses a quantization type HOS does not decode.
    UnsupportedQuant(u32),
    /// Model architecture HOS cannot run (yet).
    UnsupportedArch(String),
    /// Required model metadata key was missing.
    MissingMeta(String),
    /// The interpreter's arch spec was invalid or referenced a missing weight.
    Spec(String),
}

impl fmt::Display for HosError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HosError::Io(e) => write!(f, "io error: {e}"),
            HosError::Format(m) => write!(f, "malformed file: {m}"),
            HosError::MissingTensor(n) => write!(f, "missing tensor '{n}'"),
            HosError::UnsupportedQuant(t) => write!(
                f,
                "unsupported quantization type {t} (supported: F32/F16/Q8_0/Q4_0/Q5_0/Q4_K/Q5_K/Q6_K)"
            ),
            HosError::UnsupportedArch(a) => write!(f, "unsupported architecture: {a}"),
            HosError::MissingMeta(k) => write!(f, "missing model metadata: {k}"),
            HosError::Spec(m) => write!(f, "arch spec error: {m}"),
        }
    }
}

impl std::error::Error for HosError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            HosError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for HosError {
    fn from(e: std::io::Error) -> Self {
        HosError::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, HosError>;
