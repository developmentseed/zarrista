use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use zarrs::array::ArrayBuilder;

use crate::array::type_wrappers::PyDimensionName;
use crate::array::{PyArray, PyArrayShape, PyChunkGrid, PyChunkKeyEncoding, PyFillValue};
use crate::codec::{PyArrayToArrayCodec, PyArrayToBytesCodec, PyBytesToBytesCodec};
use crate::dtype::PyDataType;
use crate::error::ZarristaError;
use crate::error::ZarristaResult;
use crate::metadata::{PyArrayMetadataV3, PyAttributes};
use crate::storage::PySyncStorage;

#[pyclass(module = "zarrista.array", frozen, name = "ArrayBuilder")]
pub struct PyArrayBuilder(ArrayBuilder);

impl PyArrayBuilder {
    fn with(&self, f: impl FnOnce(&mut ArrayBuilder)) -> Self {
        let mut b = self.0.clone();
        f(&mut b);
        Self(b)
    }
}

#[pymethods]
impl PyArrayBuilder {
    #[new]
    fn py_new(chunk_grid: PyChunkGrid, dtype: PyDataType, fill_value: PyFillValue) -> Self {
        Self(ArrayBuilder::new_with_chunk_grid(
            chunk_grid.into_inner(),
            dtype.into_inner(),
            fill_value.into_inner(),
        ))
    }

    #[staticmethod]
    fn like<'py>(array: Bound<'py, PyAny>) -> ZarristaResult<Self> {
        if let Ok(array) = array.cast::<PyArray>() {
            Ok(Self(ArrayBuilder::from_array(array.get().inner())))
        } else {
            #[cfg(feature = "async")]
            if let Ok(array) = array.cast::<crate::array::PyAsyncArray>() {
                return Ok(Self(ArrayBuilder::from_array(array.get().inner())));
            }

            Err(PyTypeError::new_err(format!(
                "expected an Array or AsyncArray, got {}",
                array.get_type().name()?
            ))
            .into())
        }
    }

    fn attrs(&self, attrs: PyAttributes) -> PyResult<Self> {
        Ok(self.with(|builder| {
            builder.attributes(attrs.into_inner());
        }))
    }

    fn chunk_grid(&self, chunk_grid: PyChunkGrid) -> Self {
        self.with(|builder| {
            builder.chunk_grid(chunk_grid.into_inner());
        })
    }

    fn chunk_key_encoding(&self, chunk_key_encoding: PyChunkKeyEncoding) -> Self {
        self.with(|builder| {
            builder.chunk_key_encoding(chunk_key_encoding.into_inner());
        })
    }

    // TODO:
    // fn codec_options

    fn compressors(&self, compressors: Vec<PyBytesToBytesCodec>) -> Self {
        self.with(|builder| {
            builder
                .bytes_to_bytes_codecs(compressors.into_iter().map(|c| c.into_inner()).collect());
        })
    }

    fn create(&self, store: PySyncStorage, path: &str) -> ZarristaResult<PyArray> {
        let array = self.0.build_arc(store.into_inner(), path)?;
        array.store_metadata()?;
        Ok(array.into())
    }

    #[cfg(feature = "async")]
    fn create_async<'py>(
        &self,
        py: Python<'py>,
        store: crate::storage::PyAsyncStorage,
        path: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let array = self
            .0
            .build_arc(store.into_inner(), path)
            .map_err(ZarristaError::from)?;

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            array
                .async_store_metadata()
                .await
                .map_err(ZarristaError::from)?;
            Ok(crate::array::PyAsyncArray::from(array))
        })
    }

    fn create_metadata(&self) -> ZarristaResult<PyArrayMetadataV3> {
        Ok(self.0.build_metadata()?.into())
    }

    /// Set the data type of the array to be built.
    fn data_type(&self, data_type: PyDataType) -> Self {
        self.with(|builder| {
            builder.data_type(data_type.into_inner());
        })
    }

    fn dimension_names(&self, dimension_names: Option<Vec<PyDimensionName>>) -> Self {
        self.with(|builder| {
            builder.dimension_names(dimension_names);
        })
    }

    fn filters(&self, filters: Vec<PyArrayToArrayCodec>) -> Self {
        self.with(|builder| {
            builder.array_to_array_codecs(filters.into_iter().map(|f| f.into_inner()).collect());
        })
    }

    fn serializer(&self, serializer: PyArrayToBytesCodec) -> Self {
        self.with(|builder| {
            builder.array_to_bytes_codec(serializer.into_inner());
        })
    }

    /// Set the shape of the array to be built.
    fn shape(&self, shape: PyArrayShape) -> Self {
        self.with(|builder| {
            builder.shape(shape);
        })
    }

    fn subchunk_shape(&self, subchunk_shape: Option<PyArrayShape>) -> Self {
        self.with(|builder| {
            builder.subchunk_shape(subchunk_shape);
        })
    }

    // TODO:
    // fn storage_transformers
}
