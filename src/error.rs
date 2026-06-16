//! Python exception types and helpers for surfacing zarrs errors.

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::PyErr;
use std::fmt::Display;

create_exception!(
    zarrsita,
    ZarrsitaError,
    PyException,
    "Base class for all zarrsita errors."
);
create_exception!(
    zarrsita,
    NotFoundError,
    ZarrsitaError,
    "Raised when a node (array or group) does not exist at a path."
);

/// Convert any error with a message into a [`ZarrsitaError`].
pub(crate) fn to_py_err<E: Display>(err: E) -> PyErr {
    ZarrsitaError::new_err(err.to_string())
}

/// Build a [`NotFoundError`] for a missing node path.
pub(crate) fn not_found(path: &str) -> PyErr {
    NotFoundError::new_err(format!("no array or group found at path {path:?}"))
}
