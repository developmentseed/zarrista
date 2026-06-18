use crate::error::ZarristaResult;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::store::MemoryStore;
use zarrs::storage::ReadableWritableListableStorageTraits;

/// A zarrista sync store object adapted to the maximal `zarrs` storage trait.
pub struct PySyncStorage(Arc<dyn ReadableWritableListableStorageTraits>);

impl FromPyObject<'_, '_> for PySyncStorage {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(s) = obj.cast::<PyFilesystemStore>() {
            return Ok(Self(s.get().storage.clone()));
        }
        if let Ok(s) = obj.cast::<PyMemoryStore>() {
            return Ok(Self(s.get().storage.clone()));
        }
        Err(PyTypeError::new_err(
            "expected a FilesystemStore or MemoryStore",
        ))
    }
}

impl From<PySyncStorage> for Arc<dyn ReadableWritableListableStorageTraits> {
    fn from(s: PySyncStorage) -> Self {
        s.0
    }
}

/// A store backed by a local directory.
#[pyclass(module = "zarrista", frozen, name = "FilesystemStore")]
pub struct PyFilesystemStore {
    pub(crate) storage: Arc<dyn ReadableWritableListableStorageTraits>,
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
    pub(crate) storage: Arc<dyn ReadableWritableListableStorageTraits>,
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
