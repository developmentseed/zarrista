//! Error handling for zarrsita.
//!
//! [`ZarrsitaError`] is a Rust error enum that wraps the various error types
//! returned by `zarrs` (and a few other crates) and converts cleanly into a
//! Python exception via `From<ZarrsitaError> for PyErr`. Functions that return
//! [`ZarrsitaResult`] can therefore use `?` directly on those underlying
//! errors instead of sprinkling `.map_err(...)` everywhere.

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pythonize::PythonizeError;
use thiserror::Error;
use zarrs::array::{ArrayCreateError, ArrayError};
use zarrs::filesystem::FilesystemStoreCreateError;
use zarrs::group::GroupCreateError;
use zarrs::node::{NodeCreateError, NodePathError};
use zarrs::storage::StorageError;

create_exception!(
    zarrsita,
    ZarrsitaException,
    PyException,
    "Base class for all zarrsita errors."
);
create_exception!(
    zarrsita,
    NotFoundError,
    ZarrsitaException,
    "Raised when a node (array or group) does not exist at a path."
);

/// Errors that can occur in zarrsita.
///
/// Each variant wraps an error from an upstream crate (or a [`PyErr`] passed
/// through unchanged). The `#[from]` attributes give us `?` on those errors,
/// and the `From<ZarrsitaError> for PyErr` impl maps every variant onto an
/// appropriate Python exception.
#[derive(Debug, Error)]
#[non_exhaustive]
pub(crate) enum ZarrsitaError {
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
}

impl ZarrsitaError {
    /// Build a [`ZarrsitaError::NotFound`] for a missing node path.
    pub(crate) fn not_found(path: &str) -> Self {
        Self::NotFound(format!("no array or group found at path {path:?}"))
    }
}

impl From<ZarrsitaError> for PyErr {
    fn from(error: ZarrsitaError) -> Self {
        match error {
            ZarrsitaError::NotFound(msg) => NotFoundError::new_err(msg),
            ZarrsitaError::Py(err) => err,
            ZarrsitaError::Pythonize(err) => err.into(),
            ZarrsitaError::ArrayCreate(err) => ZarrsitaException::new_err(err.to_string()),
            ZarrsitaError::Array(err) => ZarrsitaException::new_err(err.to_string()),
            ZarrsitaError::GroupCreate(err) => ZarrsitaException::new_err(err.to_string()),
            ZarrsitaError::NodeCreate(err) => ZarrsitaException::new_err(err.to_string()),
            ZarrsitaError::NodePath(err) => ZarrsitaException::new_err(err.to_string()),
            ZarrsitaError::Storage(err) => ZarrsitaException::new_err(err.to_string()),
            ZarrsitaError::FilesystemStoreCreate(err) => {
                ZarrsitaException::new_err(err.to_string())
            }
            ZarrsitaError::SerdeJson(err) => ZarrsitaException::new_err(err.to_string()),
        }
    }
}

/// A `Result` whose error converts into a Python exception.
pub(crate) type ZarrsitaResult<T> = Result<T, ZarrsitaError>;
