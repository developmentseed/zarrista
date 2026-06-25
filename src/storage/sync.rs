use crate::error::ZarristaResult;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::byte_range::{ByteRange, ByteRangeIterator};
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{
    Bytes, ListableStorageTraits, MaybeBytes, MaybeBytesIterator, OffsetBytesIterator,
    ReadableListableStorageTraits, ReadableStorageTraits, ReadableWritableListableStorageTraits,
    StorageError, StoreKey, StoreKeys, StoreKeysPrefixes, StorePrefix, WritableStorageTraits,
};
/// A zarrista sync store object adapted to the maximal `zarrs` storage trait.
pub struct PySyncStorage(Arc<dyn ReadableWritableListableStorageTraits>);

impl PySyncStorage {
    pub fn into_inner(self) -> Arc<dyn ReadableWritableListableStorageTraits> {
        self.0
    }
}

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

/// Wraps a readable + listable store, rejecting every write with
/// [`StorageError::ReadOnly`].
pub struct ReadOnly(Arc<dyn ReadableListableStorageTraits>);

impl ReadOnly {
    pub fn new(inner: Arc<dyn ReadableListableStorageTraits>) -> Self {
        Self(inner)
    }
}

impl ReadableStorageTraits for ReadOnly {
    fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        self.0.get(key)
    }

    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        self.0.get_partial_many(key, byte_ranges)
    }

    fn get_partial(
        &self,
        key: &StoreKey,
        byte_range: ByteRange,
    ) -> Result<MaybeBytes, StorageError> {
        self.0.get_partial(key, byte_range)
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.0.size_key(key)
    }

    fn supports_get_partial(&self) -> bool {
        self.0.supports_get_partial()
    }
}

impl ListableStorageTraits for ReadOnly {
    fn list(&self) -> Result<StoreKeys, StorageError> {
        self.0.list()
    }

    fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        self.0.list_prefix(prefix)
    }

    fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        self.0.list_dir(prefix)
    }

    fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        self.0.size_prefix(prefix)
    }
}

impl WritableStorageTraits for ReadOnly {
    fn set(&self, _key: &StoreKey, _value: Bytes) -> Result<(), StorageError> {
        Err(StorageError::ReadOnly)
    }

    fn set_partial_many(
        &self,
        _key: &StoreKey,
        _offset_values: OffsetBytesIterator,
    ) -> Result<(), StorageError> {
        Err(StorageError::ReadOnly)
    }

    fn erase(&self, _key: &StoreKey) -> Result<(), StorageError> {
        Err(StorageError::ReadOnly)
    }

    fn erase_prefix(&self, _prefix: &StorePrefix) -> Result<(), StorageError> {
        Err(StorageError::ReadOnly)
    }

    fn supports_set_partial(&self) -> bool {
        false
    }
}
