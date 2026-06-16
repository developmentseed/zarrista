//! The `AsyncArray` Python class: metadata accessors and numpy-style reads.

use std::sync::Arc;

use crate::dtype::PyDataType;
use crate::error::to_py_err;
use crate::node::PyNodePath;
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use pyo3_object_store::AnyObjectStore;
use pythonize::pythonize;
use pythonize::Result as PythonizeResult;
use zarrs::array::Array;
use zarrs::storage::AsyncReadableListableStorageTraits;
use zarrs_object_store::AsyncObjectStore;

/// A read-only Zarr array.
#[pyclass(module = "zarrsita", frozen, name = "AsyncArray")]
pub struct PyAsyncArray {
    pub(crate) inner: Arc<Array<dyn AsyncReadableListableStorageTraits>>,
}

impl PyAsyncArray {
    pub(crate) fn new(inner: Arc<Array<dyn AsyncReadableListableStorageTraits>>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyAsyncArray {
    /// Open the array stored at `path` in `store`.
    #[staticmethod]
    #[pyo3(signature = (store, path))]
    fn open_async<'py>(
        py: Python<'py>,
        store: AnyObjectStore,
        path: PyNodePath,
    ) -> PyResult<Bound<'py, PyAny>> {
        let storage: Arc<dyn AsyncReadableListableStorageTraits> =
            Arc::new(AsyncObjectStore::new(store.into_dyn()));
        future_into_py(py, async move {
            let inner = Array::async_open(storage, path.as_str())
                .await
                .map_err(to_py_err)?;
            Ok(Self::new(Arc::new(inner)))
        })
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

    /// The Zarr data-type
    #[getter]
    fn dtype(&self) -> PyDataType {
        self.inner.data_type().clone().into()
    }

    /// The dimension names, if any were specified.
    #[getter]
    fn dimension_names(&self) -> &Option<Vec<Option<String>>> {
        self.inner.dimension_names()
    }

    /// The array's user attributes as a dict.
    #[getter]
    fn attrs<'py>(&self, py: Python<'py>) -> PythonizeResult<Bound<'py, PyAny>> {
        pythonize(py, self.inner.attributes())
    }

    fn __repr__(&self) -> String {
        format!(
            "AsyncArray(shape={:?}, dtype={:?})",
            self.inner.shape(),
            self.dtype().__repr__()
        )
    }
}
