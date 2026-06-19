//! The `AsyncArray` Python class: metadata accessors and numpy-style reads.

use std::sync::Arc;

use crate::array::selection::PySelection;
use crate::array::util::PyChunkIndices;
use crate::chunks::PyChunkGrid;
use crate::codec::{PyCodecChain, PyCodecOptions};
use crate::data::Decoded;
use crate::dtype::PyDataType;
use crate::error::ZarristaError;
use crate::node::PyNodePath;
use crate::storage::PyAsyncStorage;
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use pyo3_bytes::PyBytes;
use pythonize::pythonize;
use pythonize::Result as PythonizeResult;
use zarrs::array::Array;
use zarrs::storage::AsyncReadableWritableListableStorageTraits;

/// A Zarr array.
#[pyclass(module = "zarrista", frozen, name = "AsyncArray")]
pub struct PyAsyncArray {
    pub(crate) inner: Arc<Array<dyn AsyncReadableWritableListableStorageTraits>>,
}

impl PyAsyncArray {
    pub(crate) fn new(inner: Arc<Array<dyn AsyncReadableWritableListableStorageTraits>>) -> Self {
        Self { inner }
    }
}

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

    /// The array's user attributes as a dict.
    #[getter]
    fn attrs<'py>(&self, py: Python<'py>) -> PythonizeResult<Bound<'py, PyAny>> {
        pythonize(py, self.inner.attributes())
    }

    #[getter]
    fn chunk_grid(&self) -> PyChunkGrid {
        self.inner.chunk_grid().clone().into()
    }

    #[getter]
    fn codecs(&self) -> PyCodecChain {
        self.inner.codecs().into()
    }

    /// The dimension names, if any were specified.
    #[getter]
    fn dimension_names(&self) -> &Option<Vec<Option<String>>> {
        self.inner.dimension_names()
    }

    /// The Zarr data-type
    #[getter]
    fn dtype(&self) -> PyDataType {
        self.inner.data_type().clone().into()
    }

    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        pythonize(py, self.inner.metadata()).unwrap()
    }

    /// The number of dimensions.
    #[getter]
    fn ndim(&self) -> usize {
        self.inner.dimensionality()
    }

    /// The array's path in the store.
    #[getter]
    fn path(&self) -> &str {
        self.inner.path().as_str()
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
                .async_retrieve_array_subset::<Decoded>(&array_subset)
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
                .async_retrieve_chunk_opt::<Decoded>(chunk_indices.as_ref(), &codec_options)
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

    /// The array shape.
    #[getter]
    fn shape(&self) -> &[u64] {
        self.inner.shape()
    }
}
