use std::path::PathBuf;
use std::sync::Arc;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::byte_range::{ByteRange, ByteRangeIterator};
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{
    Bytes, ListableStorageTraits, MaybeBytes, MaybeBytesIterator, OffsetBytesIterator,
    ReadableListableStorageTraits, ReadableStorageTraits, ReadableWritableListableStorageTraits,
    StorageError, StoreKey, StoreKeys, StoreKeysPrefixes, StorePrefix, WritableStorageTraits,
};
use zarrs_zip::ZipStorageAdapter;

use crate::error::ZarristaResult;
use crate::storage::PyStoreKey;

/// A zarrista sync store object adapted to the maximal `zarrs` storage trait.
#[derive(Clone, IntoPyObject)]
pub enum PySyncStorage {
    Filesystem(PyFilesystemStore),
    MemoryStore(PyMemoryStore),
    ZipStore(PyZipStore),
}

impl PySyncStorage {
    pub fn inner(&self) -> Arc<dyn ReadableWritableListableStorageTraits> {
        match self {
            Self::Filesystem(store) => store.storage.clone(),
            Self::MemoryStore(store) => store.0.clone(),
            Self::ZipStore(store) => store.storage.clone(),
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
        if let Ok(s) = obj.cast::<PyZipStore>() {
            return Ok(Self::ZipStore(s.get().clone()));
        }
        Err(PyTypeError::new_err(
            "expected a FilesystemStore, MemoryStore, or ZipStore",
        ))
    }
}

impl From<PySyncStorage> for Arc<dyn ReadableWritableListableStorageTraits> {
    fn from(s: PySyncStorage) -> Self {
        match s {
            PySyncStorage::Filesystem(store) => store.storage,
            PySyncStorage::MemoryStore(store) => store.0,
            PySyncStorage::ZipStore(store) => store.storage,
        }
    }
}

/// A store backed by a local directory.
#[pyclass(
    module = "zarrista",
    frozen,
    name = "FilesystemStore",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyFilesystemStore {
    pub(crate) storage: Arc<FilesystemStore>,
    path: PathBuf,
}

crate::wasm_send_sync!(PyFilesystemStore);

#[pymethods]
impl PyFilesystemStore {
    /// Open a filesystem store rooted at `path`.
    #[new]
    fn new(path: PathBuf) -> ZarristaResult<Self> {
        let store = FilesystemStore::new(&path)?;
        Ok(Self {
            storage: Arc::new(store),
            path,
        })
    }

    fn __repr__(&self) -> String {
        format!("FilesystemStore({})", self.path.display())
    }
}

/// An in-memory store, primarily useful for testing.
#[pyclass(module = "zarrista", frozen, name = "MemoryStore", skip_from_py_object)]
#[derive(Clone)]
pub struct PyMemoryStore(Arc<MemoryStore>);

crate::wasm_send_sync!(PyMemoryStore);

#[pymethods]
impl PyMemoryStore {
    #[new]
    fn new() -> Self {
        Self(Arc::new(MemoryStore::new()))
    }

    fn __repr__(&self) -> String {
        "MemoryStore()".to_string()
    }
}

/// A read-only store backed by a zip file that is held in another store.
#[pyclass(module = "zarrista", frozen, name = "ZipStore", skip_from_py_object)]
#[derive(Clone)]
pub struct PyZipStore {
    storage: Arc<ReadOnlyStorageAdapter>,
    key: StoreKey,
}

crate::wasm_send_sync!(PyZipStore);

#[pymethods]
impl PyZipStore {
    /// Open the zip file that is stored at `key` in `store`.
    #[new]
    #[pyo3(
        signature = (store, key, path = None),
        text_signature = "(store, key, path=None)"
    )]
    fn new(
        py: Python,
        store: PySyncStorage,
        key: PyStoreKey,
        path: Option<PathBuf>,
    ) -> ZarristaResult<Self> {
        let key = key.into_inner();
        let adapter = crate::py::detach(py, || {
            if let Some(path) = path {
                ZipStorageAdapter::new_with_path(store.inner(), key.clone(), path)
            } else {
                ZipStorageAdapter::new(store.inner(), key.clone())
            }
        })?;
        Ok(Self {
            storage: Arc::new(ReadOnlyStorageAdapter::new(Arc::new(adapter))),
            key,
        })
    }

    /// Open the zip file that is stored at `key` in `store`.
    ///
    /// This is an alias for `ZipStore.__init__`
    #[staticmethod]
    #[pyo3(
        signature = (store, key, path = None),
        text_signature = "(store, key, path=None)"
    )]
    fn open(
        py: Python,
        store: PySyncStorage,
        key: PyStoreKey,
        path: Option<PathBuf>,
    ) -> ZarristaResult<Self> {
        Self::new(py, store, key, path)
    }

    fn __repr__(&self) -> String {
        format!("ZipStore({})", self.key.as_str())
    }
}

/// A storage adapter that reads and lists transparently but rejects all writes at runtime.
pub struct ReadOnlyStorageAdapter(Arc<dyn ReadableListableStorageTraits>);

impl ReadOnlyStorageAdapter {
    pub fn new(inner: Arc<dyn ReadableListableStorageTraits>) -> Self {
        Self(inner)
    }
}

impl ReadableStorageTraits for ReadOnlyStorageAdapter {
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

impl ListableStorageTraits for ReadOnlyStorageAdapter {
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

impl WritableStorageTraits for ReadOnlyStorageAdapter {
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
