//! The `Array` Python class: metadata accessors and numpy-style reads.

use crate::array::selection::PySelection;
use crate::array::util::PyChunkIndices;
use crate::chunks::PyChunkGrid;
use crate::codec::{PyCodecChain, PyCodecOptions};
use crate::decoded_array::DecodedArray;
use crate::dtype::PyDataType;
use crate::error::ZarristaResult;
use crate::node::PyNodePath;
use crate::storage::PySyncStorage;
use pyo3::prelude::*;
use pyo3_bytes::PyBytes;
use pythonize::pythonize;
use pythonize::Result as PythonizeResult;
use zarrs::array::Array;
use zarrs::storage::ReadableWritableListableStorageTraits;

/// A Zarr array.
#[pyclass(module = "zarrista", frozen, name = "Array")]
pub struct PyArray {
    pub(crate) inner: Array<dyn ReadableWritableListableStorageTraits>,
}

impl PyArray {
    pub(crate) fn new(inner: Array<dyn ReadableWritableListableStorageTraits>) -> Self {
        Self { inner }
    }
}

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
        Ok(Self::new(inner))
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

    /// The array shape.
    #[getter]
    fn shape(&self) -> &[u64] {
        self.inner.shape()
    }
}
