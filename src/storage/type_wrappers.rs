//! Newtype wrappers of upstream types to implement FromPyObject and IntoPyObject
//!
//! These wrappers are **not** standalone Python classes; they only define serde

use std::convert::Infallible;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;
use zarrs::storage::StoreKey;

/// A Zarr abstract store key.
///
/// See <https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#abstract-store-interface>.
pub struct PyStoreKey(StoreKey);

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
