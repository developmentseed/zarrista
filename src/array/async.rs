//! The `AsyncArray` Python class: metadata accessors and numpy-style reads.

use std::sync::Arc;

use crate::array::selection::PySelection;
use crate::array::shared::array_metadata_accessors;
use crate::array::util::PyChunkIndices;
use crate::codec::PyCodecOptions;
use crate::decoded_array::DecodedArray;
use crate::error::ZarristaError;
use crate::node::PyNodePath;
use crate::storage::PyAsyncStorage;
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
}

impl From<Array<dyn AsyncReadableWritableListableStorageTraits>> for PyAsyncArray {
    fn from(inner: Array<dyn AsyncReadableWritableListableStorageTraits>) -> Self {
        Self::new(Arc::new(inner))
    }
}
