//! The `AsyncArray` Python class: metadata accessors and numpy-style reads.

use std::sync::Arc;

use crate::chunks::PyChunkGrid;
use crate::codec::PyCodecChain;
use crate::data::{DataInner, PyData};
use crate::dtype::PyDataType;
use crate::error::ZarrsitaError;
use crate::node::PyNodePath;
use ndarray::ArrayD;
use pyo3::exceptions::PyNotImplementedError;
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
        store: AnyObjectStore,
        path: PyNodePath,
    ) -> PyResult<Bound<'py, PyAny>> {
        let storage: Arc<dyn AsyncReadableListableStorageTraits> =
            Arc::new(AsyncObjectStore::new(store.into_dyn()));
        future_into_py(py, async move {
            let inner = Array::async_open(storage, path.as_str())
                .await
                .map_err(ZarrsitaError::from)?;
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

    fn retrieve_chunk<'py>(
        &self,
        py: Python<'py>,
        chunk_indices: Vec<u64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        use zarrs::array::data_type::*;

        let inner = self.inner.clone();

        future_into_py(py, async move {
            let dtype = inner.data_type();

            macro_rules! retrieve {
                ($dtype:ty, $variant:ident, $elem:ty) => {
                    if dtype.is::<$dtype>() {
                        let chunk = inner
                            .async_retrieve_chunk::<ArrayD<$elem>>(&chunk_indices)
                            .await
                            .map_err(ZarrsitaError::from)?;
                        return Ok(PyData::from(DataInner::$variant(chunk)));
                    }
                };
            }

            retrieve!(BoolDataType, Bool, bool);
            retrieve!(Int8DataType, Int8, i8);
            retrieve!(Int16DataType, Int16, i16);
            retrieve!(Int32DataType, Int32, i32);
            retrieve!(Int64DataType, Int64, i64);
            retrieve!(UInt8DataType, Uint8, u8);
            retrieve!(UInt16DataType, Uint16, u16);
            retrieve!(UInt32DataType, Uint32, u32);
            retrieve!(UInt64DataType, Uint64, u64);
            retrieve!(Float16DataType, Float16, half::f16);
            retrieve!(Float32DataType, Float32, f32);
            retrieve!(Float64DataType, Float64, f64);

            Err(PyNotImplementedError::new_err(format!(
                "reading data type {dtype} is not supported yet"
            )))
        })
    }

    /// The array shape.
    #[getter]
    fn shape(&self) -> &[u64] {
        self.inner.shape()
    }
}
