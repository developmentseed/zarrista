use std::convert::Infallible;
use std::sync::Arc;

use async_trait::async_trait;
use object_store::ObjectStore;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3_async_runtimes::tokio::future_into_py;
use pyo3_bytes::PyBytes;
use pyo3_object_store::AnyObjectStore;
use zarrs::storage::byte_range::ByteRangeIterator;
use zarrs::storage::{
    AsyncListableStorageTraits, AsyncMaybeBytesIterator, AsyncReadableListableStorageTraits,
    AsyncReadableStorageTraits, AsyncReadableWritableListableStorageTraits,
    AsyncWritableStorageTraits, Bytes, MaybeBytes, OffsetBytesIterator, StorageError, StoreKey,
    StoreKeys, StoreKeysPrefixes, StorePrefix,
};
use zarrs_icechunk::AsyncIcechunkStore;
use zarrs_object_store::AsyncObjectStore;
use zarrs_zip::ZipStorageAdapter;

use crate::error::ZarristaError;
use crate::storage::{PyStoreKey, PyZipPath};

#[derive(Clone, IntoPyObject)]
pub enum PyAsyncStorage {
    ObjectStore(PyAsyncObjectStore),
    Icechunk(PyAsyncIcechunkStore),
    ZipStore(PyAsyncZipStore),
}

impl PyAsyncStorage {
    pub fn inner(&self) -> Arc<dyn AsyncReadableWritableListableStorageTraits> {
        match self {
            Self::ObjectStore(store) => store.inner.clone(),
            Self::Icechunk(store) => store.inner.clone(),
            Self::ZipStore(store) => store.storage.clone(),
        }
    }
}

impl FromPyObject<'_, '_> for PyAsyncStorage {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        // Once we know it's an icechunk Session, surface any failure instead of
        // falling through to the generic error below: the caller meant icechunk.
        if is_icechunk_session(obj)? {
            let store = obj.extract::<PyAsyncIcechunkStore>()?;
            return Ok(Self::Icechunk(store));
        }

        if let Ok(store) = obj.extract::<PyAsyncObjectStore>() {
            return Ok(Self::ObjectStore(store));
        }

        if let Ok(s) = obj.cast::<PyAsyncZipStore>() {
            return Ok(Self::ZipStore(s.get().clone()));
        }

        Err(PyTypeError::new_err(
            "expected an async compatible storage object",
        ))
    }
}

impl From<PyAsyncStorage> for Arc<dyn AsyncReadableWritableListableStorageTraits> {
    fn from(s: PyAsyncStorage) -> Self {
        s.inner()
    }
}

#[derive(Clone)]
pub struct PyAsyncObjectStore {
    inner: Arc<AsyncObjectStore<Arc<dyn ObjectStore>>>,
    /// Handle to the original Python object, used for returning the original store to Python
    pyobj: Arc<Py<PyAny>>,
}

impl FromPyObject<'_, '_> for PyAsyncObjectStore {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        let store = obj.extract::<AnyObjectStore>()?;
        Ok(Self {
            inner: Arc::new(AsyncObjectStore::new(store.into_dyn())),
            pyobj: Arc::new(obj.into()),
        })
    }
}

impl<'py> IntoPyObject<'py> for PyAsyncObjectStore {
    type Target = PyAny;
    type Error = Infallible;
    type Output = Bound<'py, Self::Target>;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.pyobj.bind(py).clone())
    }
}

impl<'py> IntoPyObject<'py> for &PyAsyncObjectStore {
    type Target = PyAny;
    type Error = Infallible;
    type Output = Bound<'py, Self::Target>;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.pyobj.bind(py).clone())
    }
}

#[derive(Clone)]
pub struct PyAsyncIcechunkStore {
    inner: Arc<AsyncIcechunkStore>,
    /// Handle to the original Python object, used for returning the original store to Python
    pyobj: Arc<Py<PyAny>>,
}

/// Whether `obj` is an `icechunk.session.Session` instance.
fn is_icechunk_session(obj: Borrowed<'_, '_, PyAny>) -> PyResult<bool> {
    let class = obj.getattr("__class__")?;
    let module = class.getattr("__module__")?.extract::<PyBackedStr>()?;
    let name = class.getattr("__name__")?.extract::<PyBackedStr>()?;
    Ok(module == "icechunk.session" && name == "Session")
}

impl FromPyObject<'_, '_> for PyAsyncIcechunkStore {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        // The session is bridged across the language boundary by serializing it on the
        // Python side and deserializing it with the icechunk crate zarrista links
        // against.
        // That format is only compatible within a major version, so we require
        // the installed icechunk to be 2.x (the version we build against) and fail with
        // an actionable message rather than a cryptic deserialization error.
        let icechunk_version = obj
            .py()
            .import("icechunk")?
            .getattr("__version__")?
            .extract::<PyBackedStr>()?;
        let icechunk_major_version: u32 = icechunk_version
            .split('.')
            .next()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        if icechunk_major_version < 2 {
            return Err(PyValueError::new_err(format!(
                "zarrista's icechunk support requires icechunk >= 2, but found {icechunk_version}. \
                 icechunk 2.x requires Python >= 3.12."
            )));
        }

        // NOTE: this is a bit hacky. We should make an issue in icechunk to ask for a public,
        // stable API to create a new rust Session object from a Python Session object.
        let serialized_session = obj
            .getattr("_session")?
            .call_method0("as_bytes")?
            .extract::<PyBytes>()?;

        let session = icechunk::session::Session::from_bytes(serialized_session.as_slice())
            .map_err(|err| {
                PyValueError::new_err(format!(
                    "Failed to reconstruct icechunk Session: {err}.\nThis can happen when the \
                     installed icechunk is incompatible with the version zarrista was built \
                     against (2.x)."
                ))
            })?;
        Ok(Self {
            inner: Arc::new(AsyncIcechunkStore::new(session)),
            pyobj: Arc::new(obj.into()),
        })
    }
}

impl<'py> IntoPyObject<'py> for PyAsyncIcechunkStore {
    type Target = PyAny;
    type Error = Infallible;
    type Output = Bound<'py, Self::Target>;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.pyobj.bind(py).clone())
    }
}

impl<'py> IntoPyObject<'py> for &PyAsyncIcechunkStore {
    type Target = PyAny;
    type Error = Infallible;
    type Output = Bound<'py, Self::Target>;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.pyobj.bind(py).clone())
    }
}

/// A read-only store backed by a zip file that is held in another store.
#[pyclass(
    module = "zarrista",
    frozen,
    name = "AsyncZipStore",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyAsyncZipStore {
    storage: Arc<AsyncReadOnlyStorageAdapter>,
    key: StoreKey,
}

crate::wasm_send_sync!(PyAsyncZipStore);

#[pymethods]
impl PyAsyncZipStore {
    /// Open the zip file that is stored at `key` in `store`.
    #[staticmethod]
    #[pyo3(
        signature = (store, key, *, path = None),
        text_signature = "(store, key, *, path=None)"
    )]
    fn open_async<'py>(
        py: Python<'py>,
        store: PyAsyncStorage,
        key: PyStoreKey,
        path: Option<PyZipPath>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let key = key.into_inner();

        future_into_py(py, async move {
            let adapter = if let Some(path) = path {
                ZipStorageAdapter::new_with_path_async(store.inner(), key.clone(), path).await
            } else {
                ZipStorageAdapter::new_async(store.inner(), key.clone()).await
            }
            .map_err(ZarristaError::from)?;
            Ok(Self {
                storage: Arc::new(AsyncReadOnlyStorageAdapter::new(Arc::new(adapter))),
                key,
            })
        })
    }

    fn __repr__(&self) -> String {
        format!("AsyncZipStore({})", self.key.as_str())
    }
}

/// An async storage adapter that reads and lists transparently but rejects all writes at runtime.
pub struct AsyncReadOnlyStorageAdapter(Arc<dyn AsyncReadableListableStorageTraits>);

impl AsyncReadOnlyStorageAdapter {
    pub fn new(inner: Arc<dyn AsyncReadableListableStorageTraits>) -> Self {
        Self(inner)
    }
}

#[async_trait]
impl AsyncReadableStorageTraits for AsyncReadOnlyStorageAdapter {
    async fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        self.0.get(key).await
    }

    async fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<AsyncMaybeBytesIterator<'a>, StorageError> {
        self.0.get_partial_many(key, byte_ranges).await
    }

    async fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.0.size_key(key).await
    }

    fn supports_get_partial(&self) -> bool {
        self.0.supports_get_partial()
    }
}

#[async_trait]
impl AsyncListableStorageTraits for AsyncReadOnlyStorageAdapter {
    async fn list(&self) -> Result<StoreKeys, StorageError> {
        self.0.list().await
    }

    async fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        self.0.list_prefix(prefix).await
    }

    async fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        self.0.list_dir(prefix).await
    }

    async fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        self.0.size_prefix(prefix).await
    }
}

#[async_trait]
impl AsyncWritableStorageTraits for AsyncReadOnlyStorageAdapter {
    async fn set(&self, _key: &StoreKey, _value: Bytes) -> Result<(), StorageError> {
        Err(StorageError::ReadOnly)
    }

    async fn set_partial_many<'a>(
        &'a self,
        _key: &StoreKey,
        _offset_values: OffsetBytesIterator<'a>,
    ) -> Result<(), StorageError> {
        Err(StorageError::ReadOnly)
    }

    async fn erase(&self, _key: &StoreKey) -> Result<(), StorageError> {
        Err(StorageError::ReadOnly)
    }

    async fn erase_prefix(&self, _prefix: &StorePrefix) -> Result<(), StorageError> {
        Err(StorageError::ReadOnly)
    }

    fn supports_set_partial(&self) -> bool {
        false
    }
}
