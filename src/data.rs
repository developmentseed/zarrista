use ndarray::ArrayD;
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

impl From<DataInner> for PyData {
    fn from(inner: DataInner) -> Self {
        Self { inner }
    }
}
