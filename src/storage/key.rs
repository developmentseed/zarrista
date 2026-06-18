use std::convert::Infallible;

use pyo3::prelude::*;
use pyo3::types::PyString;
use zarrs::storage::StoreKey;

use crate::error::ZarristaError;

pub struct PyStoreKey(StoreKey);

impl FromPyObject<'_, '_> for PyStoreKey {
    type Error = ZarristaError;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        let key = obj.extract::<String>()?;
        Ok(PyStoreKey(StoreKey::new(key)?))
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

impl From<PyStoreKey> for StoreKey {
    fn from(key: PyStoreKey) -> Self {
        key.0
    }
}

impl From<StoreKey> for PyStoreKey {
    fn from(key: StoreKey) -> Self {
        PyStoreKey(key)
    }
}
