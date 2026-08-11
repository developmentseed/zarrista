//! Data-type handling: zarrs `DataType` names, reading regions into numpy
//! arrays, and converting fill values into Python scalars.

use std::borrow::Cow;

use pyo3::prelude::*;
use zarrs::array::{DataType, DataTypeSize};
use zarrs::metadata::v3::MetadataV3;

use crate::error::ZarristaResult;
use crate::metadata::PyMetadataV3;

#[derive(Debug, Clone)]
#[pyclass(module = "zarrista", frozen, name = "DataType", from_py_object)]
pub struct PyDataType {
    inner: DataType,
}

crate::wasm_send_sync!(PyDataType);

impl PyDataType {
    pub(crate) fn inner(&self) -> &DataType {
        &self.inner
    }

    pub fn into_inner(self) -> DataType {
        self.inner
    }
}

#[pymethods]
impl PyDataType {
    /// Construct a data type from its Zarr v3 metadata.
    #[staticmethod]
    #[pyo3(signature = (metadata, /))]
    fn from_metadata(metadata: PyMetadataV3) -> ZarristaResult<Self> {
        let data_type = DataType::from_metadata(&metadata.into_inner())?;
        Ok(Self { inner: data_type })
    }

    /// Construct a data type from its Zarr v3 name (e.g. `"float32"`).
    #[staticmethod]
    #[pyo3(signature = (name, /))]
    fn from_string(name: &str) -> ZarristaResult<Self> {
        let metadata = MetadataV3::new(name);
        let data_type = DataType::from_metadata(&metadata)?;
        Ok(Self { inner: data_type })
    }

    #[getter]
    pub(crate) fn name(&self) -> Option<Cow<'static, str>> {
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

    pub(crate) fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        // Render the Zarr v3 name
        let name = self.inner.name_v3().into_pyobject(py)?.repr()?;
        Ok(format!("DataType({name})"))
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
