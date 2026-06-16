use ndarray::ArrayD;
use numpy::PyArray;
use pyo3::prelude::*;

pub enum DataInner {
    Bool(ArrayD<bool>),
    Float16(ArrayD<half::f16>),
    Float32(ArrayD<f32>),
    Float64(ArrayD<f64>),
    Int16(ArrayD<i16>),
    Int32(ArrayD<i32>),
    Int64(ArrayD<i64>),
    Int8(ArrayD<i8>),
    Uint16(ArrayD<u16>),
    Uint32(ArrayD<u32>),
    Uint64(ArrayD<u64>),
    Uint8(ArrayD<u8>),
}

#[pyclass(module = "zarrsita", frozen, name = "Data")]
pub struct PyData {
    inner: DataInner,
}

#[pymethods]
impl PyData {
    /// Copy the decoded chunk into a NumPy array.
    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        use DataInner::*;

        // Matching exhaustively means adding a `DataInner` variant is a compile
        // error here until it is handled.
        match &self.inner {
            Bool(array) => PyArray::from_array(py, array).into_any(),
            Float16(array) => PyArray::from_array(py, array).into_any(),
            Float32(array) => PyArray::from_array(py, array).into_any(),
            Float64(array) => PyArray::from_array(py, array).into_any(),
            Int16(array) => PyArray::from_array(py, array).into_any(),
            Int32(array) => PyArray::from_array(py, array).into_any(),
            Int64(array) => PyArray::from_array(py, array).into_any(),
            Int8(array) => PyArray::from_array(py, array).into_any(),
            Uint16(array) => PyArray::from_array(py, array).into_any(),
            Uint32(array) => PyArray::from_array(py, array).into_any(),
            Uint64(array) => PyArray::from_array(py, array).into_any(),
            Uint8(array) => PyArray::from_array(py, array).into_any(),
        }
    }
}

impl From<DataInner> for PyData {
    fn from(inner: DataInner) -> Self {
        Self { inner }
    }
}
