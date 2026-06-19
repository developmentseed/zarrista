//! Error handling for zarrista.
//!
//! [`ZarristaError`] is a Rust error enum that wraps the various error types
//! returned by `zarrs` (and a few other crates) and converts cleanly into a
//! Python exception via `From<ZarristaError> for PyErr`. Functions that return
//! [`ZarristaResult`] can therefore use `?` directly on those underlying
//! errors instead of sprinkling `.map_err(...)` everywhere.

use pyo3::prelude::*;
use pythonize::PythonizeError;
use thiserror::Error;
use zarrs::array::codec::TransposeOrderError;
use zarrs::array::{ArrayCreateError, ArrayError, CodecError};
use zarrs::filesystem::FilesystemStoreCreateError;
use zarrs::group::GroupCreateError;
use zarrs::node::{NodeCreateError, NodePathError};
use zarrs::plugin::PluginCreateError;
use zarrs::storage::StorageError;

use crate::exceptions as exc;

/// Errors that can occur in zarrista.
///
/// Each variant wraps an error from an upstream crate (or a [`PyErr`] passed
/// through unchanged). The `#[from]` attributes give us `?` on those errors,
/// and the `From<ZarristaError> for PyErr` impl maps every variant onto an
/// appropriate Python exception.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ZarristaError {
    /// No array or group exists at the requested path.
    #[error("{0}")]
    NotFound(String),
    /// An error originating from the Python interpreter, passed through as-is.
    #[error(transparent)]
    Py(#[from] PyErr),
    /// Failed to open an array.
    #[error(transparent)]
    ArrayCreate(#[from] ArrayCreateError),
    /// An error reading from or operating on an array.
    #[error(transparent)]
    Array(#[from] ArrayError),
    /// Failed to open a group.
    #[error(transparent)]
    GroupCreate(#[from] GroupCreateError),
    /// Failed to enumerate or create a child node.
    #[error(transparent)]
    NodeCreate(#[from] NodeCreateError),
    /// An invalid node path.
    #[error(transparent)]
    NodePath(#[from] NodePathError),
    /// An error from the underlying storage backend.
    #[error(transparent)]
    Storage(#[from] StorageError),
    /// Failed to open a filesystem store.
    #[error(transparent)]
    FilesystemStoreCreate(#[from] FilesystemStoreCreateError),
    /// Failed to convert a value to or from Python.
    #[error(transparent)]
    Pythonize(#[from] PythonizeError),
    /// Failed to (de)serialize JSON.
    #[error(transparent)]
    SerdeJson(#[from] serde_json::Error),
    /// Codec error
    #[error(transparent)]
    Codec(#[from] CodecError),
    /// Transpose order error
    #[error(transparent)]
    TransposeOrder(#[from] TransposeOrderError),
    /// Failed to create a codec (or other plugin) from its configuration.
    #[error(transparent)]
    PluginCreate(#[from] PluginCreateError),
}

impl ZarristaError {
    /// Build a [`ZarristaError::NotFound`] for a missing node path.
    pub(crate) fn not_found(path: &str) -> Self {
        Self::NotFound(format!("no array or group found at path {path:?}"))
    }
}

impl From<ZarristaError> for PyErr {
    fn from(error: ZarristaError) -> Self {
        match error {
            ZarristaError::NotFound(msg) => exc::NotFoundError::new_err(msg),
            ZarristaError::Py(err) => err,
            ZarristaError::ArrayCreate(err) => exc::ArrayCreateError::new_err(err.to_string()),
            ZarristaError::Array(err) => exc::ArrayError::new_err(err.to_string()),
            ZarristaError::GroupCreate(err) => exc::GroupCreateError::new_err(err.to_string()),
            ZarristaError::NodeCreate(err) => exc::NodeCreateError::new_err(err.to_string()),
            ZarristaError::NodePath(err) => exc::NodePathError::new_err(err.to_string()),
            ZarristaError::Storage(err) => exc::StorageError::new_err(err.to_string()),
            ZarristaError::FilesystemStoreCreate(err) => {
                exc::StorageError::new_err(err.to_string())
            }
            ZarristaError::Pythonize(err) => exc::SerializationError::new_err(err.to_string()),
            ZarristaError::SerdeJson(err) => exc::SerializationError::new_err(err.to_string()),
            ZarristaError::Codec(err) => exc::CodecError::new_err(err.to_string()),
            ZarristaError::TransposeOrder(err) => {
                exc::TransposeOrderError::new_err(err.to_string())
            }
            ZarristaError::PluginCreate(err) => exc::PluginCreateError::new_err(err.to_string()),
        }
    }
}

/// A `Result` whose error converts into a Python exception.
pub(crate) type ZarristaResult<T> = Result<T, ZarristaError>;
