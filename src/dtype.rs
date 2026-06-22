//! Data-type handling: zarrs `DataType` names, reading regions into numpy
//! arrays, and converting fill values into Python scalars.

use std::borrow::Cow;

use crate::error::ZarristaError;
use crate::metadata::PyMetadataV3;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::PyString;
use zarrs::array::{ArrayCreateError, DataType, DataTypeSize};
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
    #[new]
    fn py_new(metadata: PyMetadataV3) -> Self {
        let data_type = DataType::from_metadata(&metadata.into_inner()).unwrap();
        PyDataType { inner: data_type }
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

impl FromPyObject<'_, '_> for PyDataType {
    type Error = ZarristaError;

    // Taken from https://github.com/zarrs/zarrs/blob/38a7be3e51c0b7f2f6a88ba0859714ab07878cb4/zarrs/src/array/builder/array_builder_data_type.rs#L36-L52
    fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(slf) = obj.cast::<Self>() {
            return Ok(slf.get().clone());
        }

        let metadata = if obj.is_instance_of::<PyString>() {
            let string_type = obj.extract::<PyBackedStr>()?;
            // assume the metadata corresponds to a "name" if it cannot be parsed as MetadataV3
            // this makes "float32" work for example, where normally r#""float32""# would be required
            MetadataV3::try_from(string_type.as_str())
                .unwrap_or(MetadataV3::new(string_type.as_str()))
        } else {
            obj.extract::<PyMetadataV3>()?.into_inner()
        };

        Ok(DataType::from_metadata(&metadata)
            .map_err(ArrayCreateError::DataTypeCreateError)?
            .into())
    }
}
