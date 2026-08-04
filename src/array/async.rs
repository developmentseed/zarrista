//! The `AsyncArray` Python class: metadata accessors and numpy-style reads.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use pyo3_bytes::PyBytes;
use zarrs::array::{
    Array, ArrayShardedExt, ArraySubset, AsyncArrayShardedReadableExt,
    AsyncArrayShardedReadableExtCache,
};
use zarrs::storage::{AsyncReadableStorageTraits, AsyncReadableWritableListableStorageTraits};

use crate::array::selection::PySelection;
use crate::array::shared::shared_array_methods;
use crate::array::{PyChunkIndices, PyEncodedChunk};
use crate::array_bytes::PyArrayBytes;
use crate::codec::{CodecChainSubchunkExt, PyCodecOptions};
use crate::data::{DecodedArray, PyDataInput};
use crate::error::{ZarristaError, ZarristaResult};
use crate::metadata::PyArrayMetadata;
use crate::node::PyNodePath;
use crate::storage::{AsyncReadOnlyStorageAdapter, PyAsyncStorage};

/// A Zarr array.
#[pyclass(module = "zarrista", frozen, name = "AsyncArray", from_py_object)]
#[derive(Clone)]
pub struct PyAsyncArray {
    pub(crate) inner: Arc<Array<dyn AsyncReadableWritableListableStorageTraits>>,
    store: PyAsyncStorage,
}

impl PyAsyncArray {
    pub(crate) fn new(
        inner: Arc<Array<dyn AsyncReadableWritableListableStorageTraits>>,
        store: PyAsyncStorage,
    ) -> Self {
        Self { inner, store }
    }

    pub fn inner(&self) -> &Arc<Array<dyn AsyncReadableWritableListableStorageTraits>> {
        &self.inner
    }

    /// Resolve a selection against the array's shape
    fn array_subset(&self, selection: &PySelection) -> ZarristaResult<ArraySubset> {
        selection.to_array_subset(self.inner.shape())
    }

    /// Resolve a selection against the array's **chunk grid** shape
    fn chunk_grid_subset(&self, selection: &PySelection) -> ZarristaResult<ArraySubset> {
        selection.to_array_subset(self.inner.chunk_grid_shape())
    }
}

// Metadata accessors shared with `PyArray`; see `array/shared.rs`.
shared_array_methods!(PyAsyncArray);

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
        let inner = Array::new_with_metadata(store.inner(), path.as_str(), metadata.into_inner())?;
        Ok(Self::new(Arc::new(inner), store))
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
        future_into_py(py, async move {
            let inner = Array::async_open(store.inner(), path.as_str())
                .await
                .map_err(ZarristaError::from)?;
            Ok(Self::new(Arc::new(inner), store))
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
                .async_compact_chunk(&chunk_indices, &codec_options)
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
                .async_erase_chunk(&chunk_indices)
                .await
                .map_err(ZarristaError::from)?;
            Ok(())
        })
    }

    fn erase_chunks<'py>(
        &self,
        py: Python<'py>,
        chunks: PySelection,
    ) -> PyResult<Bound<'py, PyAny>> {
        let chunks = self.chunk_grid_subset(&chunks)?;
        let inner = self.inner.clone();
        future_into_py(py, async move {
            inner
                .async_erase_chunks(&chunks)
                .await
                .map_err(ZarristaError::from)?;
            Ok(())
        })
    }

    /// Return a read-only view of this array; writes raise at runtime.
    fn read_only(&self) -> Self {
        let read_list_storage = self.inner.storage().readable_listable();
        let storage = Arc::new(AsyncReadOnlyStorageAdapter::new(read_list_storage));
        Self::new(
            Arc::new(self.inner.with_storage(storage)),
            self.store.clone(),
        )
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
        let array_subset = self.array_subset(&selection)?;

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
                .async_retrieve_chunk_opt::<DecodedArray>(&chunk_indices, &codec_options)
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
                .async_retrieve_encoded_chunk(&chunk_indices)
                .await
                .map_err(ZarristaError::from)?;
            let chunk_shape = inner
                .chunk_shape(&chunk_indices)
                .map_err(ZarristaError::from)?;
            Ok(encoded.map(|buf| {
                PyEncodedChunk::new(
                    buf,
                    inner.codecs(),
                    inner.data_type().clone(),
                    inner.fill_value().clone(),
                    chunk_shape,
                )
            }))
        })
    }

    #[pyo3(signature = (subchunk_indices, *, subchunk_cache = None))]
    fn retrieve_encoded_subchunk<'py>(
        &self,
        py: Python<'py>,
        subchunk_indices: PyChunkIndices,
        subchunk_cache: Option<&PyAsyncShardCache>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let subchunk_cache = subchunk_cache
            .cloned()
            .unwrap_or_else(|| PyAsyncShardCache::new(&self.inner));

        let inner = self.inner.clone();
        future_into_py(py, async move {
            let Some(encoded) = inner
                .async_retrieve_encoded_subchunk(subchunk_cache.as_ref(), &subchunk_indices)
                .await
                .map_err(ZarristaError::from)?
            else {
                return Ok(None);
            };

            let subchunk_codec_chain = inner.codecs().subchunk_chain()?.expect(
                "zarrs accepts only an exclusively sharded array, so the serializer shards",
            );

            let subchunk_shape = inner
                .effective_subchunk_shape()
                .expect("an exclusively sharded array has no outer array-to-array codecs");

            Ok(Some(PyEncodedChunk::new(
                encoded.into(),
                subchunk_codec_chain,
                inner.data_type().clone(),
                inner.fill_value().clone(),
                subchunk_shape,
            )))
        })
    }

    #[pyo3(signature = (subchunk_indices, *, subchunk_cache = None, **codec_options))]
    fn retrieve_subchunk<'py>(
        &self,
        py: Python<'py>,
        subchunk_indices: PyChunkIndices,
        subchunk_cache: Option<&PyAsyncShardCache>,
        codec_options: Option<PyCodecOptions>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let subchunk_cache = subchunk_cache
            .cloned()
            .unwrap_or_else(|| PyAsyncShardCache::new(&self.inner));
        let codec_options = codec_options
            .map(|opts| opts.into_inner())
            .unwrap_or_default();

        let inner = self.inner.clone();
        future_into_py(py, async move {
            let decoded = inner
                .async_retrieve_subchunk_opt::<DecodedArray>(
                    subchunk_cache.as_ref(),
                    &subchunk_indices,
                    &codec_options,
                )
                .await
                .map_err(ZarristaError::from)?;
            Ok(decoded)
        })
    }

    #[getter]
    fn store(&self) -> &PyAsyncStorage {
        &self.store
    }

    #[pyo3(signature = (selection, data, **codec_options))]
    fn store_array_subset<'py>(
        &self,
        py: Python<'py>,
        selection: PySelection,
        data: PyDataInput,
        codec_options: Option<PyCodecOptions>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let codec_options = codec_options
            .map(|opts| opts.into_inner())
            .unwrap_or_default();

        let array_subset = self.array_subset(&selection)?;

        future_into_py(py, async move {
            let subset_data = data.as_array_bytes(inner.data_type(), array_subset.shape())?;

            inner
                .async_store_array_subset_opt(&array_subset, subset_data, &codec_options)
                .await
                .map_err(ZarristaError::from)?;

            Ok(())
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
                    &chunk_indices,
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
                    .async_store_encoded_chunk(&chunk_indices, encoded_chunk.into_inner())
                    .await
                    .map_err(ZarristaError::from)?;
            }
            Ok(())
        })
    }

    /// Write the array metadata to the store.
    fn store_metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        future_into_py(py, async move {
            inner
                .async_store_metadata()
                .await
                .map_err(ZarristaError::from)?;
            Ok(())
        })
    }
}

#[pyclass(
    module = "zarrista",
    frozen,
    name = "AsyncShardCache",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyAsyncShardCache {
    inner: Arc<AsyncArrayShardedReadableExtCache>,
}

impl PyAsyncShardCache {
    fn new<TStorage: ?Sized + AsyncReadableStorageTraits>(array: &Array<TStorage>) -> Self {
        Self {
            inner: Arc::new(AsyncArrayShardedReadableExtCache::new(array)),
        }
    }
}

#[pymethods]
impl PyAsyncShardCache {
    fn clear<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        future_into_py(py, async move {
            inner.clear().await;
            Ok(())
        })
    }

    fn is_empty<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        future_into_py(py, async move { Ok(inner.is_empty().await) })
    }

    fn size<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        future_into_py(py, async move { Ok(inner.len().await) })
    }
}

impl AsRef<AsyncArrayShardedReadableExtCache> for PyAsyncShardCache {
    fn as_ref(&self) -> &AsyncArrayShardedReadableExtCache {
        &self.inner
    }
}
