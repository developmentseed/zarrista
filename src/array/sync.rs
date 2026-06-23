//! The `Array` Python class: metadata accessors and numpy-style reads.

use std::sync::Arc;

use crate::array::selection::PySelection;
use crate::array::shared::array_metadata_accessors;
use crate::array::util::PyChunkIndices;
use crate::chunks::PyChunkGrid;
use crate::codec::{PyArrayToArrayCodec, PyBytesToBytesCodec, PyCodecChain, PyCodecOptions};
use crate::decoded_array::DecodedArray;
use crate::dtype::PyDataType;
use crate::error::ZarristaResult;
use crate::fill_value::PyFillValue;
use crate::node::PyNodePath;
use crate::storage::PySyncStorage;
use pyo3::prelude::*;
use pyo3_bytes::PyBytes;
use zarrs::array::{Array, ArrayBuilder};
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

    /// Create a new array
    #[staticmethod]
    #[pyo3(
        signature = (store, dtype, chunk_grid, fill_value, *, path="/", subchunk_shape=None, array_to_array_codecs=None, bytes_to_bytes_codecs=None),
        text_signature = "(store, dtype, chunk_grid, fill_value, *, path='/', subchunk_shape=None, array_to_array_codecs=None, bytes_to_bytes_codecs=None)"
    )]
    #[expect(clippy::too_many_arguments)]
    fn create(
        store: PySyncStorage,
        dtype: PyDataType,
        chunk_grid: PyChunkGrid,
        fill_value: PyFillValue,
        path: &str,
        subchunk_shape: Option<Vec<u64>>,
        array_to_array_codecs: Option<Vec<PyArrayToArrayCodec>>,
        bytes_to_bytes_codecs: Option<Vec<PyBytesToBytesCodec>>,
    ) -> ZarristaResult<Self> {
        let store = store.into_inner();
        let mut builder = ArrayBuilder::new_with_chunk_grid(
            chunk_grid,
            dtype.into_inner(),
            fill_value.into_inner(),
        );

        if let Some(subchunk_shape) = subchunk_shape {
            builder.subchunk_shape(subchunk_shape);
        }
        if let Some(array_to_array_codecs) = array_to_array_codecs {
            builder.array_to_array_codecs(
                array_to_array_codecs
                    .into_iter()
                    .map(|c| c.into_inner())
                    .collect(),
            );
        }
        if let Some(bytes_to_bytes_codecs) = bytes_to_bytes_codecs {
            builder.bytes_to_bytes_codecs(
                bytes_to_bytes_codecs
                    .into_iter()
                    .map(|c| c.into_inner())
                    .collect(),
            );
        }

        Ok(Self {
            inner: builder.build(store, path)?,
        })
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
