use std::ffi::{c_int, c_void};
use std::sync::Arc;

use bytes::Bytes;
use dlpark::SafeManagedTensor;
use dlpark::ffi::{Device, DeviceType};
use dlpark::traits::{RowMajorCompactLayout, TensorLike};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_bytes::PyBytes;
use zarrs::array::DataType;

use crate::data::buffer_protocol::PyTensorBuffer;
use crate::dtype::PyDataType;
use crate::error::{ZarristaError, ZarristaResult};

/// Fixed-width, dense decoded data.
///
/// We don't use the upstream `Tensor` type because its bytes are not reference counted, and thus
/// don't play nicely with buffer protocol export
#[derive(Clone)]
#[pyclass(module = "zarrista", frozen, name = "Tensor", skip_from_py_object)]
pub struct PyTensor {
    bytes: Bytes,
    data_type: DataType,
    shape: Arc<[u64]>,
}

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

    /// Export via the DLPack protocol so consumers (e.g. `np.from_dlpack`) can
    /// import this data zero-copy.
    #[pyo3(signature = (**_kwargs))]
    fn __dlpack__<'py>(
        &self,
        _kwargs: Option<Bound<'py, PyDict>>,
    ) -> ZarristaResult<SafeManagedTensor> {
        SafeManagedTensor::new(self.clone())
    }

    /// The DLPack device this data lives on: `(device_type, device_id)`. Always CPU.
    fn __dlpack_device__(&self) -> (i32, i32) {
        (DeviceType::Cpu as i32, 0)
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

impl TensorLike<RowMajorCompactLayout> for PyTensor {
    type Error = ZarristaError;

    fn data_ptr(&self) -> *mut c_void {
        self.bytes.as_ptr().cast::<c_void>().cast_mut()
    }

    fn memory_layout(&self) -> RowMajorCompactLayout {
        let shape = self
            .shape()
            .iter()
            .map(|s| i64::try_from(*s).expect("overflow converting shape to i64"))
            .collect();
        RowMajorCompactLayout::new(shape)
    }

    fn byte_offset(&self) -> u64 {
        0
    }

    fn device(&self) -> Result<Device, Self::Error> {
        Ok(Device::CPU)
    }

    fn data_type(&self) -> Result<dlpark::ffi::DataType, Self::Error> {
        use zarrs::array::data_type::*;

        let dtype = &self.data_type;
        if dtype.is::<BoolDataType>() {
            Ok(dlpark::ffi::DataType::BOOL)
        } else if dtype.is::<Int8DataType>() {
            Ok(dlpark::ffi::DataType::I8)
        } else if dtype.is::<Int16DataType>() {
            Ok(dlpark::ffi::DataType::I16)
        } else if dtype.is::<Int32DataType>() {
            Ok(dlpark::ffi::DataType::I32)
        } else if dtype.is::<Int64DataType>() {
            Ok(dlpark::ffi::DataType::I64)
        } else if dtype.is::<UInt8DataType>() {
            Ok(dlpark::ffi::DataType::U8)
        } else if dtype.is::<UInt16DataType>() {
            Ok(dlpark::ffi::DataType::U16)
        } else if dtype.is::<UInt32DataType>() {
            Ok(dlpark::ffi::DataType::U32)
        } else if dtype.is::<UInt64DataType>() {
            Ok(dlpark::ffi::DataType::U64)
        } else if dtype.is::<Float16DataType>() {
            Ok(dlpark::ffi::DataType::F16)
        } else if dtype.is::<Float32DataType>() {
            Ok(dlpark::ffi::DataType::F32)
        } else if dtype.is::<Float64DataType>() {
            Ok(dlpark::ffi::DataType::F64)
        } else if dtype.is::<BFloat16DataType>() {
            Ok(dlpark::ffi::DataType::BF16)
        } else {
            Err(PyValueError::new_err("Unsupported data type in dlpack").into())
        }
    }
}

/// Fixed-width data with a validity mask. Skeleton.
#[pyclass(module = "zarrista", frozen, name = "MaskedTensor")]
pub struct PyMaskedTensor {
    data: PyTensor,
    /// The mask is 1 byte per element where 0 = invalid/missing, non-zero = valid/present.
    mask: PyTensor,
}

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
}
