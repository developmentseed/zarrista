//! The `Array` Python class: metadata accessors and numpy-style reads.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_bytes::PyBytes;
use zarrs::array::{
    Array, ArrayShardedExt, ArrayShardedReadableExt, ArrayShardedReadableExtCache, ArraySubset,
};
use zarrs::storage::ReadableWritableListableStorageTraits;

use crate::array::selection::PySelection;
use crate::array::shared::shared_array_methods;
use crate::array::{PyChunkIndices, PyEncodedChunk};
use crate::array_bytes::PyArrayBytes;
use crate::codec::{CodecChainSubchunkExt, PyCodecOptions};
use crate::data::{PyDataInput, PyTensor};
use crate::error::ZarristaResult;
use crate::metadata::PyArrayMetadata;
use crate::node::PyNodePath;
use crate::repr::array_repr;
use crate::storage::{PySyncStorage, ReadOnlyStorageAdapter};

/// A Zarr array.
#[pyclass(module = "zarrista", frozen, name = "Array", skip_from_py_object)]
#[derive(Clone)]
pub struct PyArray {
    pub(crate) inner: Arc<Array<dyn ReadableWritableListableStorageTraits>>,
    pub(crate) store: PySyncStorage,
}

crate::wasm_send_sync!(PyArray);

impl PyArray {
    pub(crate) fn new(
        inner: Arc<Array<dyn ReadableWritableListableStorageTraits>>,
        store: PySyncStorage,
    ) -> Self {
        Self { inner, store }
    }

    pub fn inner(&self) -> &Arc<Array<dyn ReadableWritableListableStorageTraits>> {
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

// Metadata accessors shared with `PyAsyncArray`; see `array/shared.rs`.
shared_array_methods!(PyArray);

#[pymethods]
impl PyArray {
    /// Read a region with numpy-style basic indexing, e.g. `arr[0:10, :, 5]`.
    fn __getitem__(&self, py: Python, selection: PySelection) -> ZarristaResult<PyTensor> {
        self.retrieve_array_subset(py, selection)
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        let path = PyNodePath::from(self.inner.path().clone());
        array_repr(py, "Array", &path, self.inner.shape(), &self.dtype())
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
        store: PySyncStorage,
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
    fn open(py: Python, store: PySyncStorage, path: PyNodePath) -> ZarristaResult<Self> {
        let inner = crate::py::detach(py, || Array::open(store.inner(), path.as_str()))?;
        Ok(Self::new(Arc::new(inner), store))
    }

    #[pyo3(signature = (chunk_indices, /, **codec_options))]
    fn compact_chunk(
        &self,
        py: Python,
        chunk_indices: PyChunkIndices,
        codec_options: Option<PyCodecOptions>,
    ) -> ZarristaResult<bool> {
        crate::py::detach(py, move || {
            let codec_options = codec_options
                .map(|opts| opts.into_inner())
                .unwrap_or_default();
            Ok(self.inner.compact_chunk(&chunk_indices, &codec_options)?)
        })
    }

    #[pyo3(signature = (chunk_indices, /))]
    fn erase_chunk(&self, py: Python, chunk_indices: PyChunkIndices) -> ZarristaResult<()> {
        crate::py::detach(py, || {
            self.inner.erase_chunk(&chunk_indices)?;
            Ok(())
        })
    }

    #[pyo3(signature = (chunks, /))]
    fn erase_chunks(&self, py: Python, chunks: PySelection) -> ZarristaResult<()> {
        crate::py::detach(py, move || {
            let chunks = self.chunk_grid_subset(&chunks)?;
            self.inner.erase_chunks(&chunks)?;
            Ok(())
        })
    }

    fn erase_metadata(&self, py: Python) -> ZarristaResult<()> {
        crate::py::detach(py, || {
            self.inner.erase_metadata()?;
            Ok(())
        })
    }

    fn read_only(&self) -> Self {
        let read_list_storage = self.inner.storage().readable_listable();
        let storage = Arc::new(ReadOnlyStorageAdapter::new(read_list_storage));
        Self::new(
            Arc::new(self.inner.with_storage(storage)),
            self.store.clone(),
        )
    }

    /// Read a region of the array, using numpy-style basic indexing.
    ///
    /// Returns one of the decoded result classes (`FixedLengthTensor`,
    /// `VariableLengthTensor`, `OptionalFixedLengthTensor`,
    /// `OptionalVariableLengthTensor`) depending on the dtype layout.
    #[pyo3(signature = (selection, /))]
    fn retrieve_array_subset(
        &self,
        py: Python,
        selection: PySelection,
    ) -> ZarristaResult<PyTensor> {
        crate::py::detach(py, move || {
            let array_subset = self.array_subset(&selection)?;
            Ok(self.inner.retrieve_array_subset(&array_subset)?)
        })
    }

    #[pyo3(signature = (chunk_indices, /, **codec_options))]
    fn retrieve_chunk(
        &self,
        py: Python,
        chunk_indices: PyChunkIndices,
        codec_options: Option<PyCodecOptions>,
    ) -> ZarristaResult<PyTensor> {
        crate::py::detach(py, move || {
            let codec_options = codec_options
                .map(|opts| opts.into_inner())
                .unwrap_or_default();
            Ok(self
                .inner
                .retrieve_chunk_opt(&chunk_indices, &codec_options)?)
        })
    }

    #[pyo3(signature = (chunk_indices, /))]
    fn retrieve_encoded_chunk(
        &self,
        py: Python,
        chunk_indices: PyChunkIndices,
    ) -> ZarristaResult<Option<PyEncodedChunk>> {
        crate::py::detach(py, move || {
            let encoded = self.inner.retrieve_encoded_chunk(&chunk_indices)?;
            let chunk_shape = self.inner.chunk_shape(&chunk_indices)?;
            Ok(encoded.map(|buf| {
                PyEncodedChunk::new(
                    buf.into(),
                    self.inner.codecs(),
                    self.inner.data_type().clone(),
                    self.inner.fill_value().clone(),
                    chunk_shape,
                )
            }))
        })
    }

    #[pyo3(signature = (subchunk_indices, /, *, shard_cache = None))]
    fn retrieve_encoded_subchunk(
        &self,
        py: Python,
        subchunk_indices: PyChunkIndices,
        shard_cache: Option<&PyShardCache>,
    ) -> ZarristaResult<Option<PyEncodedChunk>> {
        crate::py::detach(py, move || {
            let shard_cache = shard_cache.cloned().unwrap_or_else(|| self.shard_cache());
            let Some(encoded) = self
                .inner
                .retrieve_encoded_subchunk(shard_cache.as_ref(), &subchunk_indices)?
            else {
                return Ok(None);
            };

            let subchunk_codec_chain = self.inner.codecs().subchunk_chain()?.expect(
                "zarrs accepts only an exclusively sharded array, so the serializer shards",
            );

            let subchunk_shape = self
                .inner
                .effective_subchunk_shape()
                .expect("an exclusively sharded array has no outer array-to-array codecs");

            Ok(Some(PyEncodedChunk::new(
                encoded.into(),
                subchunk_codec_chain,
                self.inner.data_type().clone(),
                self.inner.fill_value().clone(),
                subchunk_shape,
            )))
        })
    }

    #[pyo3(signature = (subchunk_indices, /, *, shard_cache = None, **codec_options))]
    fn retrieve_subchunk(
        &self,
        py: Python,
        subchunk_indices: PyChunkIndices,
        shard_cache: Option<&PyShardCache>,
        codec_options: Option<PyCodecOptions>,
    ) -> ZarristaResult<PyTensor> {
        crate::py::detach(py, move || {
            let shard_cache = shard_cache.cloned().unwrap_or_else(|| self.shard_cache());
            let codec_options = codec_options
                .map(|opts| opts.into_inner())
                .unwrap_or_default();

            Ok(self.inner.retrieve_subchunk_opt(
                shard_cache.as_ref(),
                &subchunk_indices,
                &codec_options,
            )?)
        })
    }

    #[getter]
    fn storage(&self) -> PySyncStorage {
        self.store.clone()
    }

    #[pyo3(signature = (selection, data, /, **codec_options))]
    fn store_array_subset(
        &self,
        py: Python,
        selection: PySelection,
        data: PyDataInput,
        codec_options: Option<PyCodecOptions>,
    ) -> ZarristaResult<()> {
        crate::py::detach(py, move || {
            let codec_options = codec_options
                .map(|opts| opts.into_inner())
                .unwrap_or_default();
            let array_subset = self.array_subset(&selection)?;
            let subset_data = data.as_array_bytes(self.inner.data_type(), array_subset.shape())?;
            self.inner
                .store_array_subset_opt(&array_subset, subset_data, &codec_options)?;
            Ok(())
        })
    }

    #[pyo3(signature = (chunk_indices, decoded_chunk, /, **codec_options))]
    fn store_chunk(
        &self,
        py: Python,
        chunk_indices: PyChunkIndices,
        decoded_chunk: &PyArrayBytes,
        codec_options: Option<PyCodecOptions>,
    ) -> ZarristaResult<()> {
        crate::py::detach(py, move || {
            let codec_options = codec_options
                .map(|opts| opts.into_inner())
                .unwrap_or_default();
            self.inner.store_chunk_opt(
                &chunk_indices,
                decoded_chunk.as_array_bytes()?,
                &codec_options,
            )?;
            Ok(())
        })
    }

    #[pyo3(signature = (chunks, data, /, **codec_options))]
    fn store_chunks(
        &self,
        py: Python,
        chunks: PySelection,
        data: PyDataInput,
        codec_options: Option<PyCodecOptions>,
    ) -> ZarristaResult<()> {
        crate::py::detach(py, move || {
            let codec_options = codec_options
                .map(|opts| opts.into_inner())
                .unwrap_or_default();
            let chunk_subset = self.chunk_grid_subset(&chunks)?;
            // `chunk_subset` counts chunks, but `data` covers the elements that
            // those chunks span, which is what `store_chunks_opt` validates.
            let array_subset = self.inner.chunks_subset(&chunk_subset)?;
            let subset_data = data.as_array_bytes(self.inner.data_type(), array_subset.shape())?;
            self.inner
                .store_chunks_opt(&chunk_subset, subset_data, &codec_options)?;
            Ok(())
        })
    }

    #[pyo3(signature = (chunk_indices, encoded_chunk, /))]
    fn store_encoded_chunk(
        &self,
        py: Python,
        chunk_indices: PyChunkIndices,
        encoded_chunk: PyBytes,
    ) -> ZarristaResult<()> {
        crate::py::detach(py, move || {
            // Safety:
            // The responsibility is on the caller to ensure the chunk is encoded correctly
            unsafe {
                self.inner
                    .store_encoded_chunk(&chunk_indices, encoded_chunk.into_inner())?;
            }
            Ok(())
        })
    }

    /// Write the array metadata to the store.
    fn store_metadata(&self, py: Python) -> ZarristaResult<()> {
        crate::py::detach(py, || {
            self.inner.store_metadata()?;
            Ok(())
        })
    }

    /// Create an empty shard index cache for this array.
    fn shard_cache(&self) -> PyShardCache {
        PyShardCache::new(self.clone())
    }
}

/// A cache of the shard indexes of one array.
#[pyclass(module = "zarrista", frozen, name = "ShardCache", skip_from_py_object)]
#[derive(Clone)]
pub struct PyShardCache {
    cache: Arc<ArrayShardedReadableExtCache>,
    array: PyArray,
}

crate::wasm_send_sync!(PyShardCache);

impl PyShardCache {
    fn new(array: PyArray) -> Self {
        Self {
            cache: Arc::new(ArrayShardedReadableExtCache::new(&array.inner)),
            array,
        }
    }
}

#[pymethods]
impl PyShardCache {
    fn __repr__(&self, py: Python) -> PyResult<String> {
        Ok(format!("ShardCache(array={})", self.array.__repr__(py)?))
    }

    /// Remove every shard index from the cache.
    fn clear(&self, py: Python) {
        crate::py::detach(py, || self.cache.clear());
    }

    /// Return whether the cache holds no shard index.
    fn is_empty(&self, py: Python) -> bool {
        crate::py::detach(py, || self.cache.is_empty())
    }

    /// Return the number of shard indexes in the cache.
    fn size(&self, py: Python) -> usize {
        crate::py::detach(py, || self.cache.len())
    }
}

impl AsRef<ArrayShardedReadableExtCache> for PyShardCache {
    fn as_ref(&self) -> &ArrayShardedReadableExtCache {
        &self.cache
    }
}
