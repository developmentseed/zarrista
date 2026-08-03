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
use crate::data::DecodedArray;
use crate::error::ZarristaResult;
use crate::metadata::PyArrayMetadata;
use crate::node::PyNodePath;
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
    fn __getitem__(&self, selection: PySelection) -> ZarristaResult<DecodedArray> {
        self.retrieve_array_subset(selection)
    }

    fn __repr__(&self) -> String {
        format!(
            "Array(shape={:?}, dtype={:?})",
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
    fn open(store: PySyncStorage, path: PyNodePath) -> ZarristaResult<Self> {
        let inner = Array::open(store.inner(), path.as_str())?;
        Ok(Self::new(Arc::new(inner), store))
    }

    #[pyo3(signature = (chunk_indices, **codec_options))]
    fn compact_chunk(
        &self,
        chunk_indices: PyChunkIndices,
        codec_options: Option<PyCodecOptions>,
    ) -> ZarristaResult<bool> {
        let codec_options = codec_options
            .map(|opts| opts.into_inner())
            .unwrap_or_default();
        Ok(self.inner.compact_chunk(&chunk_indices, &codec_options)?)
    }

    fn erase_chunk(&self, chunk_indices: PyChunkIndices) -> ZarristaResult<()> {
        self.inner.erase_chunk(&chunk_indices)?;
        Ok(())
    }

    fn erase_chunks(&self, chunks: PySelection) -> ZarristaResult<()> {
        let chunks = self.chunk_grid_subset(&chunks)?;
        self.inner.erase_chunks(&chunks)?;
        Ok(())
    }

    fn erase_metadata(&self) -> ZarristaResult<()> {
        self.inner.erase_metadata()?;
        Ok(())
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
    /// Returns one of the decoded result classes (`Tensor`, `VariableArray`,
    /// `MaskedTensor`, `MaskedVariableArray`) depending on the dtype layout.
    fn retrieve_array_subset(&self, selection: PySelection) -> ZarristaResult<DecodedArray> {
        let array_subset = self.array_subset(&selection)?;
        Ok(self.inner.retrieve_array_subset(&array_subset)?)
    }

    #[pyo3(signature = (chunk_indices, **codec_options))]
    fn retrieve_chunk(
        &self,
        chunk_indices: PyChunkIndices,
        codec_options: Option<PyCodecOptions>,
    ) -> ZarristaResult<DecodedArray> {
        let codec_options = codec_options
            .map(|opts| opts.into_inner())
            .unwrap_or_default();
        Ok(self
            .inner
            .retrieve_chunk_opt(&chunk_indices, &codec_options)?)
    }

    fn retrieve_encoded_chunk(
        &self,
        chunk_indices: PyChunkIndices,
    ) -> ZarristaResult<Option<PyEncodedChunk>> {
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
    }

    fn retrieve_encoded_subchunk(
        &self,
        subchunk_indices: PyChunkIndices,
    ) -> ZarristaResult<Option<PyEncodedChunk>> {
        // TODO: allow user to manage shard cache
        let subchunk_cache = ArrayShardedReadableExtCache::new(&self.inner);

        let Some(encoded) = self
            .inner
            .retrieve_encoded_subchunk(&subchunk_cache, &subchunk_indices)?
        else {
            return Ok(None);
        };

        let subchunk_codec_chain = self
            .inner
            .codecs()
            .subchunk_chain()?
            .expect("zarrs already validated that the array is an exclusively sharded array");

        let subchunk_shape = self
            .inner
            .effective_subchunk_shape()
            .expect("zarrs already validated that the array is an exclusively sharded array");

        Ok(Some(PyEncodedChunk::new(
            encoded.into(),
            subchunk_codec_chain,
            self.inner.data_type().clone(),
            self.inner.fill_value().clone(),
            subchunk_shape,
        )))
    }

    #[pyo3(signature = (subchunk_indices, **codec_options))]
    fn retrieve_subchunk(
        &self,
        subchunk_indices: PyChunkIndices,
        codec_options: Option<PyCodecOptions>,
    ) -> ZarristaResult<DecodedArray> {
        let codec_options = codec_options
            .map(|opts| opts.into_inner())
            .unwrap_or_default();

        // TODO: allow user to manage shard cache
        let subchunk_cache = ArrayShardedReadableExtCache::new(&self.inner);

        Ok(self
            .inner
            .retrieve_subchunk_opt(&subchunk_cache, &subchunk_indices, &codec_options)?)
    }

    #[getter]
    fn store(&self) -> PySyncStorage {
        self.store.clone()
    }

    #[pyo3(signature = (chunk_indices, decoded_chunk, **codec_options))]
    fn store_chunk(
        &self,
        chunk_indices: PyChunkIndices,
        decoded_chunk: &PyArrayBytes,
        codec_options: Option<PyCodecOptions>,
    ) -> ZarristaResult<()> {
        let codec_options = codec_options
            .map(|opts| opts.into_inner())
            .unwrap_or_default();
        self.inner.store_chunk_opt(
            &chunk_indices,
            decoded_chunk.as_array_bytes()?,
            &codec_options,
        )?;
        Ok(())
    }

    fn store_encoded_chunk(
        &self,
        chunk_indices: PyChunkIndices,
        encoded_chunk: PyBytes,
    ) -> ZarristaResult<()> {
        // Safety:
        // The responsibility is on the caller to ensure the chunk is encoded correctly
        unsafe {
            self.inner
                .store_encoded_chunk(&chunk_indices, encoded_chunk.into_inner())?;
        }
        Ok(())
    }

    /// Write the array metadata to the store.
    fn store_metadata(&self) -> ZarristaResult<()> {
        self.inner.store_metadata()?;
        Ok(())
    }
}
