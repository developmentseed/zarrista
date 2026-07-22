mod filesystem;
mod memory;
mod obspec;
mod read_only;

pub use filesystem::PyFilesystemStore;
pub use memory::PyMemoryStore;
pub use read_only::ReadOnlyStorageAdapter;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::sync::Arc;
use zarrs::storage::ReadableWritableListableStorageTraits;

/// A zarrista sync store object adapted to the maximal `zarrs` storage trait.
#[derive(Clone, IntoPyObject)]
pub enum PySyncStorage {
    Filesystem(PyFilesystemStore),
    MemoryStore(PyMemoryStore),
}

impl PySyncStorage {
    pub fn inner(&self) -> Arc<dyn ReadableWritableListableStorageTraits> {
        match self {
            Self::Filesystem(store) => store.storage.clone(),
            Self::MemoryStore(store) => store.0.clone(),
        }
    }
}

impl FromPyObject<'_, '_> for PySyncStorage {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(s) = obj.cast::<PyFilesystemStore>() {
            return Ok(Self::Filesystem(s.get().clone()));
        }
        if let Ok(s) = obj.cast::<PyMemoryStore>() {
            return Ok(Self::MemoryStore(s.get().clone()));
        }
        Err(PyTypeError::new_err(
            "expected a FilesystemStore or MemoryStore",
        ))
    }
}

impl From<PySyncStorage> for Arc<dyn ReadableWritableListableStorageTraits> {
    fn from(s: PySyncStorage) -> Self {
        match s {
            PySyncStorage::Filesystem(store) => store.storage,
            PySyncStorage::MemoryStore(store) => store.0,
        }
    }
}
