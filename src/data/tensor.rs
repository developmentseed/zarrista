use std::borrow::Cow;
use std::ffi::c_int;
use std::num::NonZeroU32;
use std::sync::Arc;

use bytes::Bytes;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3_bytes::PyBytes;
use zarrs::array::data_type::{NumpyDateTime64DataType, NumpyTimeDelta64DataType, NumpyTimeUnit};
use zarrs::array::{ArrayError, DataType, DataTypeSize};

use crate::data::buffer_protocol::PyTensorBuffer;
use crate::dtype::PyDataType;

/// Fixed-width, dense decoded data.
///
/// We don't use the upstream `Tensor` type because its bytes are not reference counted, and thus
/// don't play nicely with buffer protocol export
#[derive(Clone)]
#[pyclass(module = "zarrista", frozen, name = "Tensor", skip_from_py_object)]
pub struct PyTensor {
    pub(super) bytes: Bytes,
    pub(super) data_type: DataType,
    pub(super) shape: Arc<[u64]>,
}

crate::wasm_send_sync!(PyTensor);

impl PyTensor {
    /// Construct a new PyTensor from the given bytes, data type, and shape.
    ///
    /// # Errors
    ///
    /// - If `data_type` is not fixed-size
    /// - If `bytes` does not hold exactly `product(shape) * item_size` bytes.
    pub fn new(bytes: Bytes, data_type: DataType, shape: Arc<[u64]>) -> Result<Self, ArrayError> {
        let DataTypeSize::Fixed(item_size) = data_type.size() else {
            return Err(ArrayError::Other(format!(
                "Tensor requires a fixed-size data type, but {data_type} is variable-size"
            )));
        };

        let expected = shape
            .iter()
            .try_fold(item_size, |acc, &dim| {
                usize::try_from(dim)
                    .ok()
                    .and_then(|dim| acc.checked_mul(dim))
            })
            .ok_or_else(|| {
                ArrayError::Other(format!(
                    "Tensor shape {shape:?} * item size {item_size} overflows usize"
                ))
            })?;

        if bytes.len() != expected {
            return Err(ArrayError::UnexpectedChunkDecodedSize(
                bytes.len(),
                expected,
            ));
        }

        Ok(Self {
            bytes,
            data_type,
            shape,
        })
    }

    pub fn into_inner(self) -> (Bytes, DataType, Arc<[u64]>) {
        (self.bytes, self.data_type, self.shape)
    }
}

#[pymethods]
impl PyTensor {
    #[getter]
    fn shape(&self) -> &[u64] {
        &self.shape
    }

    #[getter]
    fn dtype(&self) -> PyDataType {
        self.data_type.clone().into()
    }

    /// The raw decoded bytes, zero-copy, as a buffer-protocol object.
    fn buffer(&self) -> PyBytes {
        PyBytes::new(self.bytes.clone())
    }

    /// Reinterpret the raw bytes as a numpy array of this dtype and shape.
    ///
    /// Zero-copy view (`np.frombuffer`) — numpy tolerates an unaligned buffer.
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let name = numpy_dtype_name(&self.data_type)?;
        let np = py.import("numpy")?;
        let flat = np.call_method1("frombuffer", (self.buffer(), name.as_ref()))?;
        flat.call_method1("reshape", (&*self.shape,))
    }

    /// NumPy array-coercion protocol backing `np.asarray(tensor)` /
    /// `np.array(tensor)`.
    #[pyo3(signature = (dtype=None, copy=None))]
    fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<Bound<'py, PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let arr = self.to_numpy(py)?;

        // copy=False demands zero-copy, so we can only hand back our own view.
        if copy == Some(false) {
            if let Some(dt) = &dtype {
                let requested = py.import("numpy")?.call_method1("dtype", (dt,))?;
                let current = arr.getattr("dtype")?;
                if !requested.eq(&current)? {
                    return Err(PyValueError::new_err(
                        "cannot return a zero-copy array with a different dtype",
                    ));
                }
            }
            return Ok(arr);
        }

        if let Some(dt) = dtype {
            arr.call_method1("astype", (dt,))
        } else if copy == Some(true) {
            arr.call_method0("copy")
        } else {
            Ok(arr)
        }
    }

    /// Export as a PEP 3118 buffer: an N-dimensional, typed, read-only,
    /// zero-copy view. Powers `memoryview(tensor)` and `np.asarray(tensor)`.
    ///
    /// Raises `BufferError` if a writable buffer is requested (the data is
    /// immutable) or if the dtype has no standard format code (e.g. bfloat16,
    /// complex — use `to_numpy` or `__dlpack__` for those).
    unsafe fn __getbuffer__(
        slf: PyRef<Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let tensor_buffer = PyTensorBuffer::try_from(slf.clone())?;
        let py_tensor_buffer = Bound::new(slf.py(), tensor_buffer)?;
        unsafe { PyTensorBuffer::__getbuffer__(py_tensor_buffer.borrow(), view, flags) }
    }

    /// Free the `format` string and the `shape`/`strides` arrays allocated by
    /// `__getbuffer__`. The `obj` reference is released by `PyBuffer_Release`.
    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {}
}

/// The NumPy data type name for a Zarr data type.
///
/// Two families need a rename.
///
/// **Temporal.** Zarr keeps the unit and the scale factor in the configuration,
/// while NumPy puts them in the name: `numpy.datetime64` with unit `s` and scale
/// factor 10 is `datetime64[10s]`.
///
/// **Complex.** Zarr names these two ways, and the number counts something
/// different in each. `complex64` counts the bits of the whole value, which is
/// also what NumPy counts. `complex_float32` names the component type instead.
/// Both describe a pair of 32-bit floats.
///
/// | Zarr name                        | Component | Total bits | NumPy name   |
/// | -------------------------------- | --------- | ---------- | ------------ |
/// | `complex64` / `complex_float32`  | `float32` | 64         | `complex64`  |
/// | `complex128` / `complex_float64` | `float64` | 128        | `complex128` |
fn numpy_dtype_name(data_type: &DataType) -> PyResult<Cow<'static, str>> {
    // Cast temporal data types to their NumPy names
    if let Some(dt) = data_type.downcast_ref::<NumpyDateTime64DataType>() {
        return Ok(numpy_temporal_name("datetime64", dt.unit, dt.scale_factor).into());
    }
    if let Some(dt) = data_type.downcast_ref::<NumpyTimeDelta64DataType>() {
        return Ok(numpy_temporal_name("timedelta64", dt.unit, dt.scale_factor).into());
    }

    let name = data_type.name_v3().ok_or_else(|| {
        PyNotImplementedError::new_err(format!(
            "data type {data_type} has no zarr v3 name / numpy mapping"
        ))
    })?;
    Ok(match name.as_ref() {
        "complex_float32" => Cow::Borrowed("complex64"),
        "complex_float64" => Cow::Borrowed("complex128"),
        _ => name,
    })
}

/// A NumPy temporal data type name, such as `datetime64[10s]`.
///
/// NumPy reads a scale factor of 1 as no scale factor, so `datetime64[1s]` and
/// `datetime64[s]` give the same data type. Therefore this always writes the
/// scale factor.
fn numpy_temporal_name(kind: &str, unit: NumpyTimeUnit, scale_factor: NonZeroU32) -> String {
    // The generic unit has no code. NumPy spells it as the bare name.
    let Some(code) = numpy_time_unit_code(unit) else {
        return kind.to_string();
    };
    format!("{kind}[{scale_factor}{code}]")
}

/// The NumPy code for a temporal unit, or `None` for the generic unit.
fn numpy_time_unit_code(unit: NumpyTimeUnit) -> Option<&'static str> {
    let s = match unit {
        NumpyTimeUnit::Generic => return None,
        NumpyTimeUnit::Year => "Y",
        NumpyTimeUnit::Month => "M",
        NumpyTimeUnit::Week => "W",
        NumpyTimeUnit::Day => "D",
        NumpyTimeUnit::Hour => "h",
        NumpyTimeUnit::Minute => "m",
        NumpyTimeUnit::Second => "s",
        NumpyTimeUnit::Millisecond => "ms",
        NumpyTimeUnit::Microsecond => "us",
        NumpyTimeUnit::Nanosecond => "ns",
        NumpyTimeUnit::Picosecond => "ps",
        NumpyTimeUnit::Femtosecond => "fs",
        NumpyTimeUnit::Attosecond => "as",
    };
    Some(s)
}

/// Fixed-width data with a validity mask. Skeleton.
#[pyclass(module = "zarrista", frozen, name = "MaskedTensor")]
pub struct PyMaskedTensor {
    data: PyTensor,
    /// The mask is 1 byte per element where 0 = invalid/missing, non-zero = valid/present.
    mask: PyTensor,
}

crate::wasm_send_sync!(PyMaskedTensor);

impl PyMaskedTensor {
    /// Construct a new PyMaskedTensor from the given bytes, mask, data type, and shape.
    pub fn new(data: PyTensor, mask: PyTensor) -> Self {
        Self { data, mask }
    }
}

#[pymethods]
impl PyMaskedTensor {
    #[getter]
    fn shape(&self) -> &[u64] {
        &self.data.shape
    }

    #[getter]
    fn data(&self) -> PyTensor {
        self.data.clone()
    }

    #[getter]
    fn dtype(&self) -> PyDataType {
        self.data.data_type.clone().into()
    }

    #[getter]
    fn mask(&self) -> PyTensor {
        self.mask.clone()
    }

    /// Convert to a NumPy masked array (`numpy.ma.MaskedArray`).
    ///
    /// The data and mask are NumPy views over the underlying Rust memory. NumPy's
    /// masked-array convention is the inverse of ours — `True` marks *masked*
    /// (missing) elements — so our validity mask (non-zero = valid) is negated.
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let data = self.data.to_numpy(py)?;
        let valid = self.mask.to_numpy(py)?;
        let np = py.import("numpy")?;
        let masked = np.call_method1("logical_not", (valid,))?;
        np.getattr("ma")?
            .call_method1("masked_array", (data, masked))
    }

    /// NumPy array-coercion protocol backing `np.asarray(tensor)` /
    /// `np.array(tensor)`. Returns a `numpy.ma.MaskedArray`.
    #[pyo3(signature = (dtype=None, copy=None))]
    fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<Bound<'py, PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let arr = self.to_numpy(py)?;

        // copy=False demands zero-copy, so we can only hand back our own view.
        if copy == Some(false) {
            if let Some(dt) = &dtype {
                let requested = py.import("numpy")?.call_method1("dtype", (dt,))?;
                let current = arr.getattr("dtype")?;
                if !requested.eq(&current)? {
                    return Err(PyValueError::new_err(
                        "cannot return a zero-copy array with a different dtype",
                    ));
                }
            }
            return Ok(arr);
        }

        if let Some(dt) = dtype {
            arr.call_method1("astype", (dt,))
        } else if copy == Some(true) {
            arr.call_method0("copy")
        } else {
            Ok(arr)
        }
    }
}

#[cfg(test)]
mod tests {
    use zarrs::array::data_type;

    use super::*;

    #[test]
    fn new_accepts_matching_length() {
        // uint8 (1 byte) × shape [4] = 4 bytes.
        let tensor = PyTensor::new(
            Bytes::from(vec![0u8; 4]),
            data_type::uint8(),
            Arc::from([4u64]),
        );
        assert!(tensor.is_ok());
    }

    #[test]
    fn new_rejects_mismatched_length() {
        // float32 (4 bytes) × shape [2] needs 8 bytes; supply 7.
        let result = PyTensor::new(
            Bytes::from(vec![0u8; 7]),
            data_type::float32(),
            Arc::from([2u64]),
        );
        assert!(matches!(
            result,
            Err(ArrayError::UnexpectedChunkDecodedSize(7, 8))
        ));
    }
}
