use std::ffi::c_int;
use std::sync::Arc;

use bytes::Bytes;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3_bytes::PyBytes;
use zarrs::array::DataType;

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
    pub fn new(bytes: Bytes, data_type: DataType, shape: Arc<[u64]>) -> Self {
        Self {
            bytes,
            data_type,
            shape,
        }
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
        // TODO: will the name_v3 always be understood by numpy?
        let name = self.data_type.name_v3().ok_or_else(|| {
            PyNotImplementedError::new_err(format!(
                "data type {} has no zarr v3 name / numpy mapping",
                self.data_type
            ))
        })?;
        let np = py.import("numpy")?;
        let flat = np.call_method1("frombuffer", (self.buffer(), name.into_owned()))?;
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
