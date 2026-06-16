use pyo3::prelude::*;
use pythonize::{depythonize, pythonize, PythonizeError};
use zarrs::array::ArrayMetadataV3;
use zarrs::group::GroupMetadataV3;
use zarrs::metadata::v3::MetadataV3;

pub struct PyMetadataV3(MetadataV3);

impl PyMetadataV3 {
    pub fn into_inner(self) -> MetadataV3 {
        self.0
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyMetadataV3 {
    type Error = PythonizeError;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        Ok(PyMetadataV3(depythonize(&obj)?))
    }
}

impl<'py> IntoPyObject<'py> for PyMetadataV3 {
    type Target = PyAny;
    type Error = PythonizeError;
    type Output = Bound<'py, Self::Target>;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        pythonize(py, &self.0)
    }
}

impl AsRef<MetadataV3> for PyMetadataV3 {
    fn as_ref(&self) -> &MetadataV3 {
        &self.0
    }
}

impl From<MetadataV3> for PyMetadataV3 {
    fn from(metadata: MetadataV3) -> Self {
        PyMetadataV3(metadata)
    }
}

impl From<PyMetadataV3> for MetadataV3 {
    fn from(py_metadata: PyMetadataV3) -> Self {
        py_metadata.0
    }
}

pub struct PyGroupMetadataV3(GroupMetadataV3);

impl<'a, 'py> FromPyObject<'a, 'py> for PyGroupMetadataV3 {
    type Error = PythonizeError;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        Ok(PyGroupMetadataV3(depythonize(&obj)?))
    }
}

impl<'py> IntoPyObject<'py> for PyGroupMetadataV3 {
    type Target = PyAny;
    type Error = PythonizeError;
    type Output = Bound<'py, Self::Target>;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        pythonize(py, &self.0)
    }
}

impl AsRef<GroupMetadataV3> for PyGroupMetadataV3 {
    fn as_ref(&self) -> &GroupMetadataV3 {
        &self.0
    }
}

impl From<GroupMetadataV3> for PyGroupMetadataV3 {
    fn from(group_metadata: GroupMetadataV3) -> Self {
        PyGroupMetadataV3(group_metadata)
    }
}

impl From<PyGroupMetadataV3> for GroupMetadataV3 {
    fn from(py_group_metadata: PyGroupMetadataV3) -> Self {
        py_group_metadata.0
    }
}

pub struct PyArrayMetadataV3(ArrayMetadataV3);

impl<'a, 'py> FromPyObject<'a, 'py> for PyArrayMetadataV3 {
    type Error = PythonizeError;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        Ok(PyArrayMetadataV3(depythonize(&obj)?))
    }
}

impl<'py> IntoPyObject<'py> for PyArrayMetadataV3 {
    type Target = PyAny;
    type Error = PythonizeError;
    type Output = Bound<'py, Self::Target>;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        pythonize(py, &self.0)
    }
}

impl AsRef<ArrayMetadataV3> for PyArrayMetadataV3 {
    fn as_ref(&self) -> &ArrayMetadataV3 {
        &self.0
    }
}

impl From<ArrayMetadataV3> for PyArrayMetadataV3 {
    fn from(array_metadata: ArrayMetadataV3) -> Self {
        PyArrayMetadataV3(array_metadata)
    }
}

impl From<PyArrayMetadataV3> for ArrayMetadataV3 {
    fn from(py_array_metadata: PyArrayMetadataV3) -> Self {
        py_array_metadata.0
    }
}
