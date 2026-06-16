use crate::error::ZarrsitaResult;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::sync::Arc;
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::store::MemoryStore;
use zarrs::storage::ReadableListableStorageTraits;

use zarrs::storage::{ReadableStorageTraits, ReadableWritableStorageTraits, WritableStorageTraits};

#[allow(dead_code)]
pub enum SyncStorage {
    Readable(Arc<dyn ReadableStorageTraits>),
    Writable(Arc<dyn WritableStorageTraits>),
    ReadableWritable(Arc<dyn ReadableWritableStorageTraits>),
}

/// A store backed by a local directory.
#[pyclass(module = "zarrsita", frozen, name = "FilesystemStore")]
pub struct PyFilesystemStore {
    pub(crate) storage: Arc<dyn ReadableListableStorageTraits>,
}

#[pymethods]
impl PyFilesystemStore {
    /// Open a filesystem store rooted at `path`.
    #[new]
    fn new(path: &str) -> ZarrsitaResult<Self> {
        let store = FilesystemStore::new(path)?;
        Ok(Self {
            storage: Arc::new(store),
        })
    }

    fn __repr__(&self) -> String {
        "FilesystemStore(...)".to_string()
    }
}

/// An in-memory store, primarily useful for testing.
#[pyclass(module = "zarrsita", frozen, name = "MemoryStore")]
pub struct PyMemoryStore {
    pub(crate) storage: Arc<dyn ReadableListableStorageTraits>,
}

#[pymethods]
impl PyMemoryStore {
    #[new]
    fn new() -> Self {
        Self {
            storage: Arc::new(MemoryStore::new()),
        }
    }

    fn __repr__(&self) -> String {
        "MemoryStore()".to_string()
    }
}

/// Pull the inner [`Storage`] out of any zarrsita store object.
pub(crate) fn extract_storage(
    store: &Bound<'_, PyAny>,
) -> PyResult<Arc<dyn ReadableListableStorageTraits>> {
    if let Ok(s) = store.cast::<PyFilesystemStore>() {
        return Ok(s.get().storage.clone());
    }
    if let Ok(s) = store.cast::<PyMemoryStore>() {
        return Ok(s.get().storage.clone());
    }
    Err(PyTypeError::new_err(
        "expected a FilesystemStore or MemoryStore",
    ))
}
