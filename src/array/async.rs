//! The `AsyncArray` Python class: metadata accessors and numpy-style reads.

use std::sync::Arc;

use crate::array::selection::PySelection;
use crate::array::shared::array_metadata_accessors;
use crate::array::PyChunkIndices;
use crate::array_bytes::PyArrayBytes;
use crate::codec::PyCodecOptions;
use crate::decoded_array::DecodedArray;
use crate::error::{ZarristaError, ZarristaResult};
use crate::metadata::PyArrayMetadata;
use crate::node::PyNodePath;
use crate::storage::{AsyncReadOnlyStorageAdapter, PyAsyncStorage};
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use pyo3_bytes::PyBytes;
use zarrs::array::Array;
use zarrs::storage::AsyncReadableWritableListableStorageTraits;

/// A Zarr array.
#[derive(Clone)]
#[pyclass(module = "zarrista", frozen, name = "AsyncArray", from_py_object)]
pub struct PyAsyncArray {
    pub(crate) inner: Arc<Array<dyn AsyncReadableWritableListableStorageTraits>>,
}

impl PyAsyncArray {
    pub(crate) fn new(inner: Arc<Array<dyn AsyncReadableWritableListableStorageTraits>>) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> &Arc<Array<dyn AsyncReadableWritableListableStorageTraits>> {
        &self.inner
    }
}

// Metadata accessors shared with `PyArray`; see `array/shared.rs`.
array_metadata_accessors!(PyAsyncArray);

#[pymethods]
impl PyAsyncArray {
    /// Read a region with numpy-style basic indexing, e.g. `await arr[0:10, :, 5]`.
    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        selection: PySelection,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.retrieve_array_subset(py, selection)
    }

    fn __repr__(&self) -> String {
        format!(
            "AsyncArray(shape={:?}, dtype={:?})",
            self.inner.shape(),
            self.dtype().__repr__()
        )
    }

    /// Use the provided metadata to open a new array at `path` in `store`.
    ///
    /// This does **not** write to the store, use `store_metadata` to write metadata to storage.
    #[staticmethod]
    #[pyo3(
        signature = (metadata, store, path = PyNodePath::root()),
        text_signature = "(metadata, store, path='/')"
    )]
    fn from_metadata(
        metadata: PyArrayMetadata,
        store: PyAsyncStorage,
        path: PyNodePath,
    ) -> ZarristaResult<Self> {
        let inner =
            Array::new_with_metadata(store.into_inner(), path.as_str(), metadata.into_inner())?;
        Ok(Self::new(Arc::new(inner)))
    }

    /// Open the array stored at `path` in `store`.
    #[staticmethod]
    #[pyo3(
        signature = (store, path = PyNodePath::root()),
        text_signature = "(store, path='/')"
    )]
    fn open_async<'py>(
        py: Python<'py>,
        store: PyAsyncStorage,
        path: PyNodePath,
    ) -> PyResult<Bound<'py, PyAny>> {
        let storage = store.into();
        future_into_py(py, async move {
            let inner = Array::async_open(storage, path.as_str())
                .await
                .map_err(ZarristaError::from)?;
            Ok(Self::new(Arc::new(inner)))
        })
    }

    #[pyo3(signature = (chunk_indices, **codec_options))]
    fn compact_chunk<'py>(
        &self,
        py: Python<'py>,
        chunk_indices: PyChunkIndices,
        codec_options: Option<PyCodecOptions>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let codec_options = codec_options
            .map(|opts| opts.into_inner())
            .unwrap_or_default();

        future_into_py(py, async move {
            let result = inner
                .async_compact_chunk(chunk_indices.as_ref(), &codec_options)
                .await
                .map_err(ZarristaError::from)?;
            Ok(result)
        })
    }

    fn erase_chunk<'py>(
        &self,
        py: Python<'py>,
        chunk_indices: PyChunkIndices,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        future_into_py(py, async move {
            inner
                .async_erase_chunk(chunk_indices.as_ref())
                .await
                .map_err(ZarristaError::from)?;
            Ok(())
        })
    }

    /// Return a read-only view of this array; writes raise at runtime.
    fn read_only(&self) -> Self {
        let read_list_storage = self.inner.storage().readable_listable();
        let storage = Arc::new(AsyncReadOnlyStorageAdapter::new(read_list_storage));
        Self::new(Arc::new(self.inner.with_storage(storage)))
    }

    fn erase_metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        future_into_py(py, async move {
            inner
                .async_erase_metadata()
                .await
                .map_err(ZarristaError::from)?;
            Ok(())
        })
    }

    /// Read a region of the array as `Data`, using numpy-style basic indexing.
    fn retrieve_array_subset<'py>(
        &self,
        py: Python<'py>,
        selection: PySelection,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let array_subset = selection.to_array_subset(inner.shape())?;

        future_into_py(py, async move {
            let decoded = inner
                .async_retrieve_array_subset::<DecodedArray>(&array_subset)
                .await
                .map_err(ZarristaError::from)?;
            Ok(decoded)
        })
    }

    #[pyo3(signature = (chunk_indices, **codec_options))]
    fn retrieve_chunk<'py>(
        &self,
        py: Python<'py>,
        chunk_indices: PyChunkIndices,
        codec_options: Option<PyCodecOptions>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let codec_options = codec_options
            .map(|opts| opts.into_inner())
            .unwrap_or_default();

        future_into_py(py, async move {
            let decoded = inner
                .async_retrieve_chunk_opt::<DecodedArray>(chunk_indices.as_ref(), &codec_options)
                .await
                .map_err(ZarristaError::from)?;
            Ok(decoded)
        })
    }

    fn retrieve_encoded_chunk<'py>(
        &self,
        py: Python<'py>,
        chunk_indices: PyChunkIndices,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        future_into_py(py, async move {
            let encoded = inner
                .async_retrieve_encoded_chunk(chunk_indices.as_ref())
                .await
                .map_err(ZarristaError::from)?;
            Ok(encoded.map(PyBytes::new))
        })
    }

    #[pyo3(signature = (chunk_indices, decoded_chunk, **codec_options))]
    fn store_chunk<'py>(
        &self,
        py: Python<'py>,
        chunk_indices: PyChunkIndices,
        decoded_chunk: PyArrayBytes,
        codec_options: Option<PyCodecOptions>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let codec_options = codec_options
            .map(|opts| opts.into_inner())
            .unwrap_or_default();

        future_into_py(py, async move {
            inner
                .async_store_chunk_opt(
                    chunk_indices.as_ref(),
                    decoded_chunk.as_array_bytes()?,
                    &codec_options,
                )
                .await
                .map_err(ZarristaError::from)?;
            Ok(())
        })
    }

    fn store_encoded_chunk<'py>(
        &self,
        py: Python<'py>,
        chunk_indices: PyChunkIndices,
        encoded_chunk: PyBytes,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        future_into_py(py, async move {
            // Safety:
            // The responsibility is on the caller to ensure the chunk is encoded correctly
            unsafe {
                inner
                    .async_store_encoded_chunk(chunk_indices.as_ref(), encoded_chunk.into_inner())
                    .await
                    .map_err(ZarristaError::from)?;
            }
            Ok(())
        })
    }
}

impl From<Array<dyn AsyncReadableWritableListableStorageTraits>> for PyAsyncArray {
    fn from(inner: Array<dyn AsyncReadableWritableListableStorageTraits>) -> Self {
        Self::new(Arc::new(inner))
    }
}

impl From<Arc<Array<dyn AsyncReadableWritableListableStorageTraits>>> for PyAsyncArray {
    fn from(inner: Arc<Array<dyn AsyncReadableWritableListableStorageTraits>>) -> Self {
        Self::new(inner)
    }
}
