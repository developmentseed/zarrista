use std::ffi::{CStr, c_int};
use std::ptr;

use bytes::Bytes;
use pyo3::exceptions::{PyBufferError, PyOverflowError};
use pyo3::ffi;
use pyo3::prelude::*;
use zarrs::array::DataType;

use crate::data::PyFixedLengthTensor;

/// Internal TensorBuffer that implements the buffer protocol
#[pyclass(
    module = "zarrista",
    frozen,
    name = "TensorBuffer",
    skip_from_py_object
)]
pub struct PyTensorBuffer {
    bytes: Bytes,
    format: &'static CStr,
    itemsize: isize,
    len: isize,
    ndim: c_int,
    shape: Box<[isize]>,
    strides: Box<[isize]>,
}

impl TryFrom<PyFixedLengthTensor> for PyTensorBuffer {
    type Error = PyErr;

    fn try_from(tensor: PyFixedLengthTensor) -> Result<Self, Self::Error> {
        let (bytes, data_type, shape) = tensor.into_inner();
        let (format, itemsize) = data_type_to_buffer_protocol_format(&data_type)?;

        let shape = shape
            .iter()
            .map(|&s| {
                isize::try_from(s).map_err(|err| {
                    PyOverflowError::new_err(format!("shape {s} overflows isize: {err}"))
                })
            })
            .collect::<PyResult<Vec<isize>>>()?;

        let mut strides = vec![0isize; shape.len()];
        let mut acc = itemsize;
        for i in (0..shape.len()).rev() {
            strides[i] = acc;
            acc *= shape[i];
        }

        let ndim = c_int::try_from(shape.len()).map_err(|err| {
            PyOverflowError::new_err(format!("ndim {} overflows c_int: {err}", shape.len()))
        })?;
        let len = isize::try_from(bytes.len()).map_err(|err| {
            PyOverflowError::new_err(format!("len {} overflows isize: {err}", bytes.len()))
        })?;

        Ok(Self {
            bytes,
            format,
            itemsize,
            len,
            ndim,
            shape: shape.into_boxed_slice(),
            strides: strides.into_boxed_slice(),
        })
    }
}

// #[pymethods]
impl PyTensorBuffer {
    /// Export as a PEP 3118 buffer: an N-dimensional, typed, read-only,
    /// zero-copy view. Powers `memoryview(tensor)` and `np.asarray(tensor)`.
    ///
    /// Raises `BufferError` if a writable buffer is requested (the data is
    /// immutable) or if the dtype has no standard format code (e.g. bfloat16,
    /// complex — use `to_numpy` or `__dlpack__` for those).
    pub unsafe fn __getbuffer__(
        slf: PyRef<Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("View is null"));
        }
        if (flags & ffi::PyBUF_WRITABLE) == ffi::PyBUF_WRITABLE {
            return Err(PyBufferError::new_err(
                "FixedLengthTensor buffer is read-only",
            ));
        }

        unsafe {
            // Fill in the Py_buffer struct fields
            // SAFETY: view is a valid pointer provided by Python's buffer protocol machinery
            (*view).buf = slf.bytes.as_ptr() as _;

            (*view).len = slf.len;
            (*view).itemsize = slf.itemsize;
            (*view).readonly = 1;
            (*view).ndim = slf.ndim;

            // Only provide format string if requested (PyBUF_FORMAT flag)
            (*view).format = if (flags & ffi::PyBUF_FORMAT) == ffi::PyBUF_FORMAT {
                slf.format.as_ptr() as _
            } else {
                ptr::null_mut()
            };

            // SAFETY: shape and strides are Box<[isize; 3]> owned by self.
            // The Py_INCREF below keeps self alive for the lifetime of this view.
            (*view).shape = slf.shape.as_ptr() as _;
            (*view).strides = slf.strides.as_ptr() as _;

            // We don't use indirect addressing (PIL-style)
            (*view).suboffsets = ptr::null_mut();

            // Reserved for internal use by the exporter
            (*view).internal = ptr::null_mut();

            // CRITICAL: Increment reference count to keep PyTensorBuffer alive while buffer
            // exists.
            //
            // Python will call Py_DECREF after __releasebuffer__ returns.
            (*view).obj = slf.as_ptr();
            ffi::Py_INCREF((*view).obj);
        }
        Ok(())
    }

    /// Called when a buffer view is released.
    ///
    /// For our implementation, this is a no-op because:
    /// - `shape` and `strides` are owned by the PyTensorBuffer struct (not allocated per-view)
    /// - Python handles the `Py_DECREF` on `view.obj` automatically
    ///
    /// We still need to implement this method because PyO3 requires it when
    /// `__getbuffer__` is implemented.
    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {
        // Nothing to clean up - all memory is owned by the PyTensorBuffer struct
    }
}

/// Map a zarrs [`DataType`] to a native-endian PEP 3118 struct format code and
/// its element size in bytes.
///
/// # Errors
///
/// Returns a `BufferError` if the data type has no standard buffer-protocol
/// format code
fn data_type_to_buffer_protocol_format(data_type: &DataType) -> PyResult<(&'static CStr, isize)> {
    use zarrs::array::data_type::*;

    if data_type.is::<BoolDataType>() {
        Ok((c"?", 1))
    } else if data_type.is::<Int8DataType>() {
        Ok((c"b", 1))
    } else if data_type.is::<Int16DataType>() {
        Ok((c"h", 2))
    } else if data_type.is::<Int32DataType>() {
        Ok((c"i", 4))
    } else if data_type.is::<Int64DataType>() {
        Ok((c"q", 8))
    } else if data_type.is::<UInt8DataType>() {
        Ok((c"B", 1))
    } else if data_type.is::<UInt16DataType>() {
        Ok((c"H", 2))
    } else if data_type.is::<UInt32DataType>() {
        Ok((c"I", 4))
    } else if data_type.is::<UInt64DataType>() {
        Ok((c"Q", 8))
    } else if data_type.is::<Float16DataType>() {
        Ok((c"e", 2))
    } else if data_type.is::<Float32DataType>() {
        Ok((c"f", 4))
    } else if data_type.is::<Float64DataType>() {
        Ok((c"d", 8))
    } else {
        Err(PyBufferError::new_err(format!(
            "data type {data_type} has no buffer-protocol format code"
        )))
    }
}
