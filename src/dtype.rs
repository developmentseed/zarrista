//! Data-type handling: zarrs `DataType` names, reading regions into numpy
//! arrays, and converting fill values into Python scalars.

use std::borrow::Cow;

use crate::error::to_py_err;
use crate::metadata::PyMetadataV3;
use numpy::prelude::*;
use numpy::IntoPyArray;
use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;
use zarrs::array::{Array, ArraySubset};
use zarrs::array::{ArrayError, ElementOwned};
use zarrs::array::{DataType, DataTypeSize};
use zarrs::storage::ReadableListableStorageTraits;

#[pyclass(module = "zarrsita", frozen, name = "DataType")]
pub struct PyDataType {
    pub(crate) inner: DataType,
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

/// The store trait object backing every zarrsita array/group.
pub(crate) type DynStorage = dyn ReadableListableStorageTraits;

/// A region of an array to read: either an explicit subset or a whole chunk.
pub(crate) enum Region<'a> {
    Subset(&'a ArraySubset),
    Chunk(&'a [u64]),
}

fn retrieve_vec<T: ElementOwned>(
    array: &Array<DynStorage>,
    region: &Region<'_>,
) -> Result<Vec<T>, ArrayError> {
    match region {
        Region::Subset(subset) => array.retrieve_array_subset(*subset),
        Region::Chunk(indices) => array.retrieve_chunk(indices),
    }
}

fn vec_to_numpy<T: numpy::Element>(
    py: Python<'_>,
    data: Vec<T>,
    shape: &[usize],
) -> PyResult<Py<PyAny>> {
    let array = data.into_pyarray(py);
    let reshaped = array.reshape(shape.to_vec())?;
    Ok(reshaped.into_any().unbind())
}

/// Read a region of `array` into a C-order numpy array of the given output
/// shape. Only fixed-length numeric and boolean dtypes are supported so far.
pub(crate) fn read_region(
    py: Python<'_>,
    array: &Array<DynStorage>,
    region: &Region<'_>,
    out_shape: &[usize],
) -> PyResult<Py<PyAny>> {
    let name = array.data_type().name_v3();

    macro_rules! arm {
        ($t:ty) => {{
            let data: Vec<$t> = retrieve_vec(array, region).map_err(to_py_err)?;
            vec_to_numpy(py, data, out_shape)
        }};
    }

    match name.as_deref() {
        Some("bool") => arm!(bool),
        Some("int8") => arm!(i8),
        Some("int16") => arm!(i16),
        Some("int32") => arm!(i32),
        Some("int64") => arm!(i64),
        Some("uint8") => arm!(u8),
        Some("uint16") => arm!(u16),
        Some("uint32") => arm!(u32),
        Some("uint64") => arm!(u64),
        Some("float16") => arm!(half::f16),
        Some("float32") => arm!(f32),
        Some("float64") => arm!(f64),
        other => Err(PyNotImplementedError::new_err(format!(
            "reading dtype {:?} is not supported yet",
            other.unwrap_or("<unknown>")
        ))),
    }
}

/// Convert a fill value (native-endian bytes) into a Python scalar, returning
/// `None` for dtypes we do not yet interpret.
pub(crate) fn fill_value_to_py(
    py: Python<'_>,
    data_type: &DataType,
    bytes: &[u8],
) -> PyResult<Py<PyAny>> {
    macro_rules! scalar {
        ($t:ty) => {{
            const N: usize = std::mem::size_of::<$t>();
            match <[u8; N]>::try_from(bytes) {
                Ok(arr) => <$t>::from_ne_bytes(arr).into_bound_py_any(py)?.unbind(),
                Err(_) => py.None(),
            }
        }};
    }

    let dtype_name = data_type.name_v3();
    let value = match dtype_name.as_deref() {
        Some("bool") => (!bytes.is_empty() && bytes[0] != 0)
            .into_bound_py_any(py)?
            .unbind(),
        Some("int8") => scalar!(i8),
        Some("int16") => scalar!(i16),
        Some("int32") => scalar!(i32),
        Some("int64") => scalar!(i64),
        Some("uint8") => scalar!(u8),
        Some("uint16") => scalar!(u16),
        Some("uint32") => scalar!(u32),
        Some("uint64") => scalar!(u64),
        Some("float16") => match <[u8; 2]>::try_from(bytes) {
            Ok(arr) => half::f16::from_ne_bytes(arr)
                .to_f32()
                .into_bound_py_any(py)?
                .unbind(),
            Err(_) => py.None(),
        },
        Some("float32") => scalar!(f32),
        Some("float64") => scalar!(f64),
        _ => py.None(),
    };
    Ok(value)
}
