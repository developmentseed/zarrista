//! Python-facing store types wrapping zarrs storage backends.

use crate::dtype::DynStorage;
use crate::error::to_py_err;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::sync::Arc;
use zarrs::filesystem::FilesystemStore as ZarrsFilesystemStore;
use zarrs::storage::store::MemoryStore as ZarrsMemoryStore;

/// A read-only handle to some zarrs storage backend.
pub(crate) type Storage = Arc<DynStorage>;

/// A store backed by a local directory.
#[pyclass(module = "zarrsita", frozen)]
pub struct FilesystemStore {
    pub(crate) storage: Storage,
}

#[pymethods]
impl FilesystemStore {
    /// Open a filesystem store rooted at `path`.
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let store = ZarrsFilesystemStore::new(path).map_err(to_py_err)?;
        Ok(Self {
            storage: Arc::new(store),
        })
    }

    fn __repr__(&self) -> String {
        "FilesystemStore(...)".to_string()
    }
}

/// An in-memory store, primarily useful for testing.
#[pyclass(module = "zarrsita", frozen)]
pub struct MemoryStore {
    pub(crate) storage: Storage,
}

#[pymethods]
impl MemoryStore {
    #[new]
    fn new() -> Self {
        Self {
            storage: Arc::new(ZarrsMemoryStore::new()),
        }
    }

    fn __repr__(&self) -> String {
        "MemoryStore()".to_string()
    }
}

/// Pull the inner [`Storage`] out of any zarrsita store object.
pub(crate) fn extract_storage(store: &Bound<'_, PyAny>) -> PyResult<Storage> {
    if let Ok(s) = store.cast::<FilesystemStore>() {
        return Ok(s.get().storage.clone());
    }
    if let Ok(s) = store.cast::<MemoryStore>() {
        return Ok(s.get().storage.clone());
    }
    Err(PyTypeError::new_err(
        "expected a FilesystemStore or MemoryStore",
    ))
}
