use crate::error::ZarristaResult;
use crate::storage::PyStore;
use pyo3::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::store::MemoryStore;
use zarrs::storage::ReadableListableStorageTraits;

/// A store backed by a local directory.
#[pyclass(module = "zarrista", frozen, name = "FilesystemStore")]
pub struct PyFilesystemStore {
    pub(crate) storage: Arc<dyn ReadableListableStorageTraits>,
}

#[pymethods]
impl PyFilesystemStore {
    /// Open a filesystem store rooted at `path`.
    #[new]
    fn new(path: PathBuf) -> ZarristaResult<Self> {
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
#[pyclass(module = "zarrista", frozen, name = "MemoryStore")]
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

/// Pull the inner [`Storage`] out of any zarrista store object.
pub(crate) fn extract_storage(
    store: &Bound<'_, PyAny>,
) -> PyResult<Arc<dyn ReadableListableStorageTraits>> {
    if let Ok(s) = store.cast::<PyFilesystemStore>() {
        return Ok(s.get().storage.clone());
    }
    if let Ok(s) = store.cast::<PyMemoryStore>() {
        return Ok(s.get().storage.clone());
    }
    // Any other object is treated as a duck-typed custom store.
    Ok(Arc::new(PyStore::new(store)))
}
