//! The `Array` Python class: metadata accessors and numpy-style reads.

use crate::convert::attributes_to_py;
use crate::error::to_py_err;
use crate::store::extract_storage;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice};
use std::borrow::Cow;
use zarrs::array::Array;
use zarrs::storage::ReadableListableStorageTraits;

/// A read-only Zarr array.
#[pyclass(module = "zarrsita", frozen, name = "Array")]
pub struct PyArray {
    pub(crate) inner: Array<dyn ReadableListableStorageTraits>,
}

impl PyArray {
    pub(crate) fn new(inner: Array<dyn ReadableListableStorageTraits>) -> Self {
        Self { inner }
    }
}

/// A single-axis selector after normalization.
enum Sel {
    /// An integer index; drops the axis.
    Index(i64),
    /// A `slice` object to resolve against the axis length.
    Slice(Py<PySlice>),
    /// The whole axis (`:`, trailing axes, or an ellipsis fill).
    Full,
}

#[pymethods]
impl PyArray {
    /// Open the array stored at `path` in `store`.
    #[staticmethod]
    #[pyo3(signature = (store, path = "/"))]
    fn open(store: &Bound<'_, PyAny>, path: &str) -> PyResult<Self> {
        let storage = extract_storage(store)?;
        let inner = Array::open(storage, path).map_err(to_py_err)?;
        Ok(Self::new(inner))
    }

    /// The array shape.
    #[getter]
    fn shape(&self) -> &[u64] {
        self.inner.shape()
    }

    /// The number of dimensions.
    #[getter]
    fn ndim(&self) -> usize {
        self.inner.shape().len()
    }

    // /// The chunk shape (size of a chunk along each dimension).
    // #[getter]
    // fn chunks(&self) -> PyResult<Vec<u64>> {
    //     // TODO: review
    //     let origin = vec![0u64; self.inner.shape().len()];
    //     let chunk_shape = self.inner.chunk_shape(&origin).map_err(to_py_err)?;
    //     Ok(chunk_shape.iter().map(|n| n.get()).collect())
    // }

    /// The Zarr V3 data-type name (e.g. `"float32"`).
    #[getter]
    fn dtype(&self) -> Option<Cow<'static, str>> {
        self.inner.data_type().name_v3()
    }

    /// The dimension names, if any were specified.
    #[getter]
    fn dimension_names(&self) -> &Option<Vec<Option<String>>> {
        self.inner.dimension_names()
    }

    /// The array's user attributes as a dict.
    #[getter]
    fn attrs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        attributes_to_py(py, self.inner.attributes())
    }

    // /// The fill value as a Python scalar (or `None` if not interpretable).
    // #[getter]
    // fn fill_value(&self, py: Python<'_>) -> PyResult<PyObject> {
    //     dtype::fill_value_to_py(
    //         py,
    //         self.inner.data_type(),
    //         self.inner.fill_value().as_ne_bytes(),
    //     )
    // }

    // /// Read a region with numpy-style basic indexing.
    // fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    //     let shape = self.inner.shape().to_vec();
    //     let (ranges, out_shape) = parse_index(py, key, &shape)?;
    //     let subset = ArraySubset::new_with_ranges(&ranges);
    //     dtype::read_region(py, &self.inner, &Region::Subset(&subset), &out_shape)
    // }

    // /// Read a single chunk by its chunk coordinates.
    // fn get_chunk(&self, py: Python<'_>, chunk_coords: Vec<u64>) -> PyResult<PyObject> {
    //     let chunk_shape = self.inner.chunk_shape(&chunk_coords).map_err(to_py_err)?;
    //     let out_shape: Vec<usize> = chunk_shape.iter().map(|n| n.get() as usize).collect();
    //     dtype::read_region(py, &self.inner, &Region::Chunk(&chunk_coords), &out_shape)
    // }

    fn __repr__(&self) -> String {
        format!(
            "Array(shape={:?}, dtype={:?})",
            self.inner.shape(),
            self.dtype().unwrap_or_else(|| "<unknown>".into())
        )
    }
}
