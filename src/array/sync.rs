//! The `Array` Python class: metadata accessors and numpy-style reads.

use crate::array::selection::PySelection;
use crate::array::util::PyChunkIndices;
use crate::chunks::PyChunkGrid;
use crate::codec::{PyCodecChain, PyCodecOptions};
use crate::data::{for_each_dtype, DataInner, PyData};
use crate::dtype::PyDataType;
use crate::error::ZarristaResult;
use crate::node::PyNodePath;
use crate::storage::PySyncStorage;
use ndarray::ArrayD;
use pyo3::exceptions::PyNotImplementedError;
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
    fn __getitem__(&self, selection: PySelection) -> ZarristaResult<PyData> {
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

    /// Read a region of the array as `Data`, using numpy-style basic indexing.
    fn retrieve_array_subset(&self, selection: PySelection) -> ZarristaResult<PyData> {
        use zarrs::array::data_type::*;

        let array_subset = selection.to_array_subset(self.inner.shape())?;
        let dtype = self.inner.data_type();

        macro_rules! arm {
            ($dtype:ty, $variant:ident, $elem:ty) => {
                if dtype.is::<$dtype>() {
                    let data = self
                        .inner
                        .retrieve_array_subset::<ArrayD<$elem>>(&array_subset)?;
                    return Ok(PyData::from(DataInner::$variant(data)));
                }
            };
        }
        for_each_dtype!(arm);

        Err(PyNotImplementedError::new_err(format!(
            "reading data type {dtype} is not supported yet"
        ))
        .into())
    }

    #[pyo3(signature = (chunk_indices, **codec_options))]
    fn retrieve_chunk(
        &self,
        chunk_indices: PyChunkIndices,
        codec_options: Option<PyCodecOptions>,
    ) -> ZarristaResult<PyData> {
        use zarrs::array::data_type::*;

        let dtype = self.inner.data_type();
        let codec_options = codec_options
            .map(|opts| opts.into_inner())
            .unwrap_or_default();

        macro_rules! arm {
            ($dtype:ty, $variant:ident, $elem:ty) => {
                if dtype.is::<$dtype>() {
                    let chunk = self.inner.retrieve_chunk_opt::<ArrayD<$elem>>(
                        chunk_indices.as_ref(),
                        &codec_options,
                    )?;
                    return Ok(PyData::from(DataInner::$variant(chunk)));
                }
            };
        }
        for_each_dtype!(arm);

        Err(PyNotImplementedError::new_err(format!(
            "reading data type {dtype} is not supported yet"
        ))
        .into())
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

    // /// The chunk shape (size of a chunk along each dimension).
    // #[getter]
    // fn chunks(&self) -> PyResult<Vec<u64>> {
    //     // TODO: review
    //     let origin = vec![0u64; self.inner.shape().len()];
    //     let chunk_shape = self.inner.chunk_shape(&origin).map_err(to_py_err)?;
    //     Ok(chunk_shape.iter().map(|n| n.get()).collect())
    // }

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
}
