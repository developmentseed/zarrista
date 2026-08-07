//! Newtype wrappers of upstream types to implement FromPyObject and IntoPyObject
//!
//! These wrappers are **not** standalone Python classes; they only define serde

use std::convert::Infallible;
use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::PyString;
use zarrs::storage::{StoreKey, StorePrefix};

/// A Zarr abstract store key.
///
/// See <https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#abstract-store-interface>.
pub struct PyStoreKey(StoreKey);

impl PyStoreKey {
    pub fn into_inner(self) -> StoreKey {
        self.0
    }
}

impl FromPyObject<'_, '_> for PyStoreKey {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        StoreKey::new(obj.extract::<String>()?)
            .map(PyStoreKey)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

impl<'py> IntoPyObject<'py> for PyStoreKey {
    type Target = PyString;
    type Error = Infallible;
    type Output = Bound<'py, Self::Target>;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(PyString::new(py, self.0.as_str()))
    }
}

impl<'py> IntoPyObject<'py> for &PyStoreKey {
    type Target = PyString;
    type Error = Infallible;
    type Output = Bound<'py, Self::Target>;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(PyString::new(py, self.0.as_str()))
    }
}

impl AsRef<StoreKey> for PyStoreKey {
    fn as_ref(&self) -> &StoreKey {
        &self.0
    }
}

impl From<PyStoreKey> for StoreKey {
    fn from(py_key: PyStoreKey) -> Self {
        py_key.0
    }
}

impl From<StoreKey> for PyStoreKey {
    fn from(key: StoreKey) -> Self {
        Self(key)
    }
}

/// The directory inside a zip file that a zip store uses as its root.
///
/// The zip storage adapter removes this value from the start of each zip entry
/// name. Entry names always use `/` and never start with `/`, so the value must
/// end with `/` and must not start with `/`. A value that breaks either rule
/// gives an empty store, or store keys that keep a leading `/`. Therefore this
/// extractor normalizes the value, and `nested`, `nested/`, and `/nested/` all
/// select the same directory.
///
/// This is an entry-name prefix and not a filesystem path, so it extracts via `PyStorePrefix`.
pub struct PyZipPath(StorePrefix);

impl FromPyObject<'_, '_> for PyZipPath {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        let path = obj.extract::<PyBackedStr>()?;
        let mut normalized = path.trim_start_matches('/').to_string();
        if !normalized.is_empty() && !normalized.ends_with('/') {
            normalized.push('/');
        }
        // Normalization satisfies `StorePrefix`'s invariant, so this cannot fail.
        let prefix =
            StorePrefix::new(normalized).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self(prefix))
    }
}

impl From<PyZipPath> for PathBuf {
    fn from(path: PyZipPath) -> PathBuf {
        PathBuf::from(path.0.as_str())
    }
}
