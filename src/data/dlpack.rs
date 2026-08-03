use std::ffi::c_void;

use dlpark::ManagedBox;
use dlpark::ffi::{
    DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLManagedTensorVersioned,
    DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION, DLTensor,
};
use dlpark::metadata::CopiedSlice;
use dlpark::python::device::dlpack_device;
use dlpark::{Builder, legacy};
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use zarrs::array::DataType;

use crate::data::PyTensor;
use crate::error::ZarristaResult;

#[pymethods]
impl PyTensor {
    /// Export via the DLPack protocol so consumers (e.g. `np.from_dlpack`) can
    /// import this data zero-copy.
    #[pyo3(signature = (**_kwargs))]
    fn __dlpack__<'py>(
        &self,
        _kwargs: Option<Bound<'py, PyDict>>,
    ) -> ZarristaResult<legacy::Dlpack> {
        let dlpack_dtype = DLDataType::from_zarrs_data_type(&self.data_type)?;

        let shape = self
            .shape
            .iter()
            .map(|s| i64::try_from(*s).expect("overflow converting shape to i64"))
            .collect::<Vec<_>>();
        let strides = row_major_compact_strides(&shape);

        // The boxed `Bytes` handed to the builder is what keeps the buffer alive: dlpark stores it
        // as the managed tensor's `manager_ctx` and drops it from the deleter, which a consumer
        // may run on any thread and long after this `PyTensor` is gone.
        let builder = Builder::new(
            Box::new(self.bytes.clone()),
            CopiedSlice::new(shape, strides),
        );
        let data = self.bytes.as_ptr().cast::<c_void>().cast_mut();

        // SAFETY:
        // `data` is the start of the `Bytes` allocation whose refcount is held by the
        // `ctx` clone above, so the pointer stays valid until the deleter drops that clone — which
        // a consumer may do on another thread, long after this `PyTensor` is gone.
        //
        // `PyTensor::new` guarantees `bytes.len() == product(shape) * item_size`, so the shape,
        // row-major strides, and dtype describe exactly the initialized, in-bounds elements.
        //
        // Alignment is not a concern: the DLTensor contract explicitly tells CPU consumers not to
        // rely on the data pointer being aligned, so an unaligned `Bytes` buffer is within spec.
        let builder = unsafe { builder.data(data) };

        Ok(builder
            .device(DLDevice::CPU)
            .dtype(dlpack_dtype)
            .try_build()
            .map_err(|err| PyValueError::new_err(err.to_string()))?)
    }

    /// The DLPack device this data lives on: `(device_type, device_id)`. Always CPU.
    fn __dlpack_device__(&self) -> (u32, i32) {
        (DLDeviceType::CPU.0, 0)
    }
}

/// Strides, in elements, for a row-major contiguous buffer of the given shape.
fn row_major_compact_strides(shape: &[i64]) -> Vec<i64> {
    let mut strides = vec![1; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    strides
}

pub trait DLPackDataTypeExt {
    /// Convert from DLPack data type to Zarr data type.
    fn zarrs_data_type(&self) -> ZarristaResult<DataType>;

    // Convert from Zarr data type to DLPack data type
    fn from_zarrs_data_type(data_type: &DataType) -> ZarristaResult<Self>
    where
        Self: Sized;
}

impl DLPackDataTypeExt for DLDataType {
    fn zarrs_data_type(&self) -> ZarristaResult<DataType> {
        use zarrs::array::data_type::*;

        if self.lanes != 1 {
            return Err(PyValueError::new_err(format!(
                "the data has {} lanes per element, but only scalar elements are supported",
                self.lanes
            ))
            .into());
        }

        let data_type = match (self.code, self.bits) {
            (DLDataTypeCode::BOOL, 8) => DataType::new(BoolDataType),
            (DLDataTypeCode::INT, 8) => DataType::new(Int8DataType),
            (DLDataTypeCode::INT, 16) => DataType::new(Int16DataType),
            (DLDataTypeCode::INT, 32) => DataType::new(Int32DataType),
            (DLDataTypeCode::INT, 64) => DataType::new(Int64DataType),
            (DLDataTypeCode::UINT, 8) => DataType::new(UInt8DataType),
            (DLDataTypeCode::UINT, 16) => DataType::new(UInt16DataType),
            (DLDataTypeCode::UINT, 32) => DataType::new(UInt32DataType),
            (DLDataTypeCode::UINT, 64) => DataType::new(UInt64DataType),
            (DLDataTypeCode::FLOAT, 16) => DataType::new(Float16DataType),
            (DLDataTypeCode::FLOAT, 32) => DataType::new(Float32DataType),
            (DLDataTypeCode::FLOAT, 64) => DataType::new(Float64DataType),
            (DLDataTypeCode::BFLOAT, 16) => DataType::new(BFloat16DataType),
            (code, bits) => {
                return Err(PyValueError::new_err(format!(
                    "the data has DLPack type code {} at {bits} bits, \
                 which has no Zarr data type",
                    code.0
                ))
                .into());
            }
        };

        Ok(data_type)
    }

    fn from_zarrs_data_type(data_type: &DataType) -> ZarristaResult<Self>
    where
        Self: Sized,
    {
        use zarrs::array::data_type::*;

        let (code, bits) = if data_type.is::<BoolDataType>() {
            (DLDataTypeCode::BOOL, 8)
        } else if data_type.is::<Int8DataType>() {
            (DLDataTypeCode::INT, 8)
        } else if data_type.is::<Int16DataType>() {
            (DLDataTypeCode::INT, 16)
        } else if data_type.is::<Int32DataType>() {
            (DLDataTypeCode::INT, 32)
        } else if data_type.is::<Int64DataType>() {
            (DLDataTypeCode::INT, 64)
        } else if data_type.is::<UInt8DataType>() {
            (DLDataTypeCode::UINT, 8)
        } else if data_type.is::<UInt16DataType>() {
            (DLDataTypeCode::UINT, 16)
        } else if data_type.is::<UInt32DataType>() {
            (DLDataTypeCode::UINT, 32)
        } else if data_type.is::<UInt64DataType>() {
            (DLDataTypeCode::UINT, 64)
        } else if data_type.is::<Float16DataType>() {
            (DLDataTypeCode::FLOAT, 16)
        } else if data_type.is::<Float32DataType>() {
            (DLDataTypeCode::FLOAT, 32)
        } else if data_type.is::<Float64DataType>() {
            (DLDataTypeCode::FLOAT, 64)
        } else if data_type.is::<BFloat16DataType>() {
            (DLDataTypeCode::BFLOAT, 16)
        } else {
            return Err(PyValueError::new_err("Unsupported data type in dlpack").into());
        };

        Ok(DLDataType::scalar(code, bits))
    }
}

/// A DLPack tensor from a Python producer
///
/// We customize this to run `Drop` with the GIL held, since the producer may need to change Python
/// state.
pub struct PyManagedTensor(Option<ManagedBox<DLManagedTensorVersioned>>);

// SAFETY: the tensor's buffer is plain memory that stays valid until the
// deleter runs, so reading it needs no GIL and is sound from any thread. The
// deleter is the only operation that touches the interpreter, and `Drop` always
// runs it under `Python::try_attach`.
unsafe impl Send for PyManagedTensor {}
impl Drop for PyManagedTensor {
    fn drop(&mut self) {
        // Try to drop the tensor with the GIL held.
        let drop_successful = Python::try_attach(|_py| drop(self.0.take())).is_some();

        // If the GIL could not be acquired, the interpreter is shutting down and the deleter
        // cannot run safely.
        //
        // We leak the tensor instead of running the deleter. The process is ending, so the memory
        // returns to the operating system regardless.
        //
        // Numpy appears to do the same. If the interpreter is shutting down, it does not run the
        // deleter, and the memory leaks.
        // <https://github.com/numpy/numpy/blob/4978efe1055ddda49a26524bca5ae6a2d01e6802/numpy/_core/src/multiarray/dlpack.c#L91-L115>
        if !drop_successful {
            std::mem::forget(self.0.take());
        }
    }
}
