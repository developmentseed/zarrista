use pyo3::prelude::*;
use pyo3_bytes::PyBytes;
use zarrs::array::FillValue;

#[derive(Debug, Clone)]
#[pyclass(module = "zarrista", frozen, name = "FillValue", from_py_object)]
pub struct PyFillValue(FillValue);

impl PyFillValue {
    pub(crate) fn inner(&self) -> &FillValue {
        &self.0
    }

    pub fn into_inner(self) -> FillValue {
        self.0
    }
}

#[pymethods]
impl PyFillValue {
    #[new]
    fn new(bytes: Vec<u8>) -> Self {
        Self(FillValue::new(bytes))
    }

    #[getter]
    fn size(&self) -> usize {
        self.0.size()
    }

    fn as_bytes(&self) -> &[u8] {
        self.0.as_ne_bytes()
    }

    fn equals_all(&self, other: PyBytes) -> bool {
        self.0.equals_all(other.as_ref())
    }
}

impl From<FillValue> for PyFillValue {
    fn from(fill_value: FillValue) -> Self {
        PyFillValue(fill_value)
    }
}

impl From<PyFillValue> for FillValue {
    fn from(py_fill_value: PyFillValue) -> Self {
        py_fill_value.0
    }
}
