use pyo3::prelude::*;
use pythonize::{depythonize, pythonize, PythonizeError};
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
