use std::sync::Arc;

use object_store::ObjectStore;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3_bytes::PyBytes;
use pyo3_object_store::AnyObjectStore;
use zarrs::storage::AsyncReadableListableStorageTraits;
use zarrs_icechunk::AsyncIcechunkStore;
use zarrs_object_store::AsyncObjectStore;

pub struct PyAsyncStorage(Arc<dyn AsyncReadableListableStorageTraits>);

impl PyAsyncStorage {
    pub fn into_inner(self) -> Arc<dyn AsyncReadableListableStorageTraits> {
        self.0
    }
}

impl FromPyObject<'_, '_> for PyAsyncStorage {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(store) = obj.extract::<PyAsyncIcechunkStore>() {
            return Ok(Self(Arc::new(store.into_inner())));
        }

        if let Ok(store) = obj.extract::<PyAsyncObjectStore>() {
            return Ok(Self(Arc::new(store.into_inner())));
        }

        Err(PyTypeError::new_err(
            "expected an async compatible storage object",
        ))
    }
}

impl From<PyAsyncStorage> for Arc<dyn AsyncReadableListableStorageTraits> {
    fn from(s: PyAsyncStorage) -> Self {
        s.0
    }
}

pub struct PyAsyncObjectStore(AsyncObjectStore<Arc<dyn ObjectStore>>);

impl PyAsyncObjectStore {
    fn into_inner(self) -> AsyncObjectStore<Arc<dyn ObjectStore>> {
        self.0
    }
}

impl FromPyObject<'_, '_> for PyAsyncObjectStore {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        let store = obj.extract::<AnyObjectStore>()?;
        Ok(Self(AsyncObjectStore::new(store.into_dyn())))
    }
}

pub struct PyAsyncIcechunkStore(AsyncIcechunkStore);

impl PyAsyncIcechunkStore {
    fn into_inner(self) -> AsyncIcechunkStore {
        self.0
    }
}

impl FromPyObject<'_, '_> for PyAsyncIcechunkStore {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        let class = obj.getattr("__class__")?;

        let module = class.getattr("__module__")?.extract::<PyBackedStr>()?;
        let name = class.getattr("__name__")?.extract::<PyBackedStr>()?;

        if module != "icechunk.session" || name != "Session" {
            return Err(PyTypeError::new_err(format!(
                "Expected an icechunk session object, got an instance of {}.{}",
                module, name
            )));
        }

        let serialized_session = obj
            .getattr("_session")?
            .call_method0("as_bytes")?
            .extract::<PyBytes>()?;

        let icechunk_session = icechunk::session::Session::from_bytes(
            serialized_session.as_slice(),
        )
        .map_err(|err| {
            PyValueError::new_err(format!("Failed to reconstruct icechunk Session: {}", err))
        })?;
        Ok(Self(AsyncIcechunkStore::new(icechunk_session)))
    }
}
