use std::ffi::c_void;

use bytes::Bytes;
use dlpark::SafeManagedTensor;
use dlpark::ffi::{Device, DeviceType};
use dlpark::traits::{RowMajorCompactLayout, TensorLike};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_bytes::PyBytes;
use zarrs::array::DataType;

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
    shape: Vec<u64>,
}

impl PyTensor {
    /// Construct a new PyTensor from the given bytes, data type, and shape.
    pub fn new(bytes: Bytes, data_type: DataType, shape: Vec<u64>) -> Self {
        Self {
            bytes,
            data_type,
            shape,
        }
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
    // TODO: it might be nice for the Tensor itself to implement the buffer protocol, instead of
    // having to call the buffer method? :shrug:
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
        flat.call_method1("reshape", (&self.shape,))
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
}

/// Convert a zarrs [`DataType`] to a [`dlpark::ffi::DataType`].
///
/// # Errors
/// Returns [`TensorError::UnsupportedDataType`] if the data type is not supported.
fn data_type_to_dlpack(data_type: &DataType) -> ZarristaResult<dlpark::ffi::DataType> {
    use zarrs::array::data_type::*;

    if data_type.is::<BoolDataType>() {
        Ok(dlpark::ffi::DataType::BOOL)
    } else if data_type.is::<Int8DataType>() {
        Ok(dlpark::ffi::DataType::I8)
    } else if data_type.is::<Int16DataType>() {
        Ok(dlpark::ffi::DataType::I16)
    } else if data_type.is::<Int32DataType>() {
        Ok(dlpark::ffi::DataType::I32)
    } else if data_type.is::<Int64DataType>() {
        Ok(dlpark::ffi::DataType::I64)
    } else if data_type.is::<UInt8DataType>() {
        Ok(dlpark::ffi::DataType::U8)
    } else if data_type.is::<UInt16DataType>() {
        Ok(dlpark::ffi::DataType::U16)
    } else if data_type.is::<UInt32DataType>() {
        Ok(dlpark::ffi::DataType::U32)
    } else if data_type.is::<UInt64DataType>() {
        Ok(dlpark::ffi::DataType::U64)
    } else if data_type.is::<Float16DataType>() {
        Ok(dlpark::ffi::DataType::F16)
    } else if data_type.is::<Float32DataType>() {
        Ok(dlpark::ffi::DataType::F32)
    } else if data_type.is::<Float64DataType>() {
        Ok(dlpark::ffi::DataType::F64)
    } else if data_type.is::<BFloat16DataType>() {
        Ok(dlpark::ffi::DataType::BF16)
    } else {
        Err(PyValueError::new_err("Unsupported data type in dlpack").into())
    }
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
        data_type_to_dlpack(&self.data_type)
    }
}

/// Fixed-width data with a validity mask. Skeleton.
#[pyclass(module = "zarrista", frozen, name = "MaskedTensor")]
pub struct PyMaskedTensor {
    #[expect(dead_code)]
    bytes: Bytes,
    /// The mask is 1 byte per element where 0 = invalid/missing, non-zero = valid/present.
    #[expect(dead_code)]
    mask: Bytes,
    data_type: DataType,
    shape: Vec<u64>,
}

impl PyMaskedTensor {
    /// Construct a new PyMaskedTensor from the given bytes, mask, data type, and shape.
    pub fn new(bytes: Bytes, mask: Bytes, data_type: DataType, shape: Vec<u64>) -> Self {
        Self {
            bytes,
            mask,
            data_type,
            shape,
        }
    }
}

#[pymethods]
impl PyMaskedTensor {
    #[getter]
    fn shape(&self) -> &[u64] {
        &self.shape
    }

    #[getter]
    fn dtype(&self) -> PyDataType {
        self.data_type.clone().into()
    }
}
