//! Data-type handling: zarrs `DataType` names, reading regions into numpy
//! arrays, and converting fill values into Python scalars.

use std::borrow::Cow;

use crate::error::ZarristaResult;
use crate::metadata::PyMetadataV3;
use pyo3::prelude::*;
use zarrs::array::{DataType, DataTypeSize};
use zarrs::metadata::v3::MetadataV3;

#[derive(Debug, Clone)]
#[pyclass(module = "zarrista", frozen, name = "DataType", skip_from_py_object)]
pub struct PyDataType {
    inner: DataType,
}

impl PyDataType {
    pub(crate) fn inner(&self) -> &DataType {
        &self.inner
    }
}

#[pymethods]
impl PyDataType {
    /// Construct a data type from its Zarr v3 metadata.
    #[staticmethod]
    fn from_metadata(metadata: PyMetadataV3) -> ZarristaResult<Self> {
        let data_type = DataType::from_metadata(&metadata.into_inner())?;
        Ok(Self { inner: data_type })
    }

    /// Construct a data type from its Zarr v3 name (e.g. `"float32"`).
    #[staticmethod]
    fn from_string(name: &str) -> ZarristaResult<Self> {
        let metadata = MetadataV3::new(name);
        let data_type = DataType::from_metadata(&metadata)?;
        Ok(Self { inner: data_type })
    }

    #[getter]
    fn name(&self) -> Option<Cow<'static, str>> {
        self.inner.name_v3()
    }

    #[getter]
    fn size(&self) -> Option<usize> {
        match self.inner.size() {
            DataTypeSize::Fixed(n) => Some(n),
            DataTypeSize::Variable => None,
        }
    }

    fn __eq__(&self, other: &Bound<PyAny>) -> bool {
        if let Ok(other) = other.cast::<Self>() {
            self.inner == other.get().inner
        } else {
            false
        }
    }

    pub(crate) fn __repr__(&self) -> String {
        format!("DataType({})", self.inner)
    }
}

impl From<DataType> for PyDataType {
    fn from(data_type: DataType) -> Self {
        PyDataType { inner: data_type }
    }
}

impl From<PyDataType> for DataType {
    fn from(py_data_type: PyDataType) -> Self {
        py_data_type.inner
    }
}
