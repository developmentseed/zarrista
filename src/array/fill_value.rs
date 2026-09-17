use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::{PyBackedBytes, PyBackedStr};
use pyo3_bytes::PyBytes;
use zarrs::array::{DataType, FillValue};

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
    #[pyo3(signature = (value, /))]
    fn new(value: Vec<u8>) -> Self {
        Self(FillValue::new(value))
    }

    #[getter]
    fn size(&self) -> usize {
        self.0.size()
    }

    fn as_bytes(&self) -> &[u8] {
        self.0.as_ne_bytes()
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        // Use the Python bytes type, not our PyBytes adapter, to create the repr
        let bytes = pyo3::types::PyBytes::new(py, self.0.as_ne_bytes()).repr()?;
        Ok(format!("FillValue({bytes})"))
    }

    #[pyo3(signature = (other, /))]
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

#[derive(Debug, Clone, FromPyObject)]
pub struct PyFillValueInput<'py>(Bound<'py, PyAny>);

impl PyFillValueInput<'_> {
    pub fn resolve(&self, dtype: &DataType) -> PyResult<FillValue> {
        use zarrs::array::data_type::*;

        let fill_value = if dtype.is::<BoolDataType>() {
            FillValue::from(self.0.extract::<bool>()?)
        } else if dtype.is::<UInt8DataType>() {
            FillValue::from(self.0.extract::<u8>()?)
        } else if dtype.is::<UInt16DataType>() {
            FillValue::from(self.0.extract::<u16>()?)
        } else if dtype.is::<UInt32DataType>() {
            FillValue::from(self.0.extract::<u32>()?)
        } else if dtype.is::<UInt64DataType>() {
            FillValue::from(self.0.extract::<u64>()?)
        } else if dtype.is::<Int8DataType>() {
            FillValue::from(self.0.extract::<i8>()?)
        } else if dtype.is::<Int16DataType>() {
            FillValue::from(self.0.extract::<i16>()?)
        } else if dtype.is::<Int32DataType>() {
            FillValue::from(self.0.extract::<i32>()?)
        } else if dtype.is::<Int64DataType>() {
            FillValue::from(self.0.extract::<i64>()?)
        } else if dtype.is::<BFloat16DataType>() {
            FillValue::from(half::bf16::from_f64(self.0.extract()?))
        } else if dtype.is::<Float16DataType>() {
            FillValue::from(half::f16::from_f64(self.0.extract()?))
        } else if dtype.is::<Float32DataType>() {
            FillValue::from(self.0.extract::<f32>()?)
        } else if dtype.is::<Float64DataType>() {
            FillValue::from(self.0.extract::<f64>()?)
        } else if dtype.is::<BytesDataType>() {
            FillValue::from(self.0.extract::<PyBackedBytes>()?.as_ref())
        } else if dtype.is::<StringDataType>() {
            FillValue::from(self.0.extract::<PyBackedStr>()?.as_str())
        } else {
            // Last, try to extract as bytes
            if let Ok(bytes) = self.0.extract::<Vec<u8>>() {
                return Ok(FillValue::new(bytes));
            }

            return Err(PyValueError::new_err(format!(
                "cannot resolve fill value for data type {}",
                dtype.name_v3().unwrap_or_else(|| "<unknown>".into())
            )));
        };

        Ok(fill_value)
    }
}
