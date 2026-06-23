//! The `Array` Python class: metadata accessors and numpy-style reads.

use std::sync::Arc;

use crate::array::selection::PySelection;
use crate::array::shared::array_metadata_accessors;
use crate::array::PyChunkIndices;
use crate::array_bytes::PyArrayBytes;
use crate::codec::PyCodecOptions;
use crate::decoded_array::DecodedArray;
use crate::error::ZarristaResult;
use crate::node::PyNodePath;
use crate::storage::PySyncStorage;
use pyo3::prelude::*;
use pyo3_bytes::PyBytes;
use zarrs::array::Array;
use zarrs::storage::ReadableWritableListableStorageTraits;

/// A Zarr array.
#[pyclass(module = "zarrista", frozen, name = "Array")]
pub struct PyArray {
    pub(crate) inner: Arc<Array<dyn ReadableWritableListableStorageTraits>>,
}

impl PyArray {
    pub(crate) fn new(inner: Arc<Array<dyn ReadableWritableListableStorageTraits>>) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> &Arc<Array<dyn ReadableWritableListableStorageTraits>> {
        &self.inner
    }
}

// Metadata accessors shared with `PyAsyncArray`; see `array/shared.rs`.
array_metadata_accessors!(PyArray);

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

    /// Open the array stored at `path` in `store`.
    #[staticmethod]
    #[pyo3(
        signature = (store, path = PyNodePath::root()),
        text_signature = "(store, path='/')"
    )]
    fn open(store: PySyncStorage, path: PyNodePath) -> ZarristaResult<Self> {
        let inner = Array::open(store.into(), path.as_str())?;
        Ok(Self::new(Arc::new(inner)))
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
        Ok(self
            .inner
            .compact_chunk(chunk_indices.as_ref(), &codec_options)?)
    }

    fn erase_chunk(&self, chunk_indices: PyChunkIndices) -> ZarristaResult<()> {
        self.inner.erase_chunk(chunk_indices.as_ref())?;
        Ok(())
    }

    fn erase_metadata(&self) -> ZarristaResult<()> {
        self.inner.erase_metadata()?;
        Ok(())
    }

    /// Read a region of the array, using numpy-style basic indexing.
    ///
    /// Returns one of the decoded result classes (`Tensor`, `VariableArray`,
    /// `MaskedTensor`, `MaskedVariableArray`) depending on the dtype layout.
    fn retrieve_array_subset(&self, selection: PySelection) -> ZarristaResult<DecodedArray> {
        let array_subset = selection.to_array_subset(self.inner.shape())?;
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
            .retrieve_chunk_opt(chunk_indices.as_ref(), &codec_options)?)
    }

    fn retrieve_encoded_chunk(
        &self,
        chunk_indices: PyChunkIndices,
    ) -> ZarristaResult<Option<PyBytes>> {
        let encoded = self.inner.retrieve_encoded_chunk(chunk_indices.as_ref())?;
        Ok(encoded.map(|buf| PyBytes::new(buf.into())))
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
            chunk_indices.as_ref(),
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
                .store_encoded_chunk(chunk_indices.as_ref(), encoded_chunk.into_inner())?;
        }
        Ok(())
    }
}

impl From<Array<dyn ReadableWritableListableStorageTraits>> for PyArray {
    fn from(inner: Array<dyn ReadableWritableListableStorageTraits>) -> Self {
        Self::new(Arc::new(inner))
    }
}

impl From<Arc<Array<dyn ReadableWritableListableStorageTraits>>> for PyArray {
    fn from(inner: Arc<Array<dyn ReadableWritableListableStorageTraits>>) -> Self {
        Self::new(inner)
    }
}
