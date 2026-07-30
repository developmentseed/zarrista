use std::ffi::c_void;

use dlpark::ffi::{DLDataType, DLDataTypeCode, DLDevice, DLDeviceType};
use dlpark::metadata::CopiedSlice;
use dlpark::{Builder, legacy};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

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
        let dlpack_dtype = self.dlpack_data_type()?;
        let shape = self
            .shape
            .iter()
            .map(|s| i64::try_from(*s).expect("overflow converting shape to i64"))
            .collect::<Vec<_>>();
        let strides = row_major_compact_strides(&shape);

        // The boxed `Bytes` handed to the builder is what keeps the buffer alive: dlpark stores it
        // as the managed tensor's `manager_ctx` and drops it from the deleter, which a consumer may
        // run on any thread and long after this `PyTensor` is gone.
        let data = self.bytes.as_ptr().cast::<c_void>().cast_mut();
        let builder = Builder::new(
            Box::new(self.bytes.clone()),
            CopiedSlice::new(shape, strides),
        );

        // SAFETY: `data` points at the start of the `Bytes` allocation moved into the context, so
        // it stays valid until the deleter runs. The shape is the tensor's own shape, the strides
        // are row-major compact, and `dtype` is the element type those bytes were decoded as, so
        // together they describe exactly the initialized elements of the buffer.
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

impl PyTensor {
    /// The DLPack element descriptor for this tensor's zarr data type.
    fn dlpack_data_type(&self) -> ZarristaResult<DLDataType> {
        use zarrs::array::data_type::*;

        let dtype = &self.data_type;
        let (code, bits) = if dtype.is::<BoolDataType>() {
            (DLDataTypeCode::BOOL, 8)
        } else if dtype.is::<Int8DataType>() {
            (DLDataTypeCode::INT, 8)
        } else if dtype.is::<Int16DataType>() {
            (DLDataTypeCode::INT, 16)
        } else if dtype.is::<Int32DataType>() {
            (DLDataTypeCode::INT, 32)
        } else if dtype.is::<Int64DataType>() {
            (DLDataTypeCode::INT, 64)
        } else if dtype.is::<UInt8DataType>() {
            (DLDataTypeCode::UINT, 8)
        } else if dtype.is::<UInt16DataType>() {
            (DLDataTypeCode::UINT, 16)
        } else if dtype.is::<UInt32DataType>() {
            (DLDataTypeCode::UINT, 32)
        } else if dtype.is::<UInt64DataType>() {
            (DLDataTypeCode::UINT, 64)
        } else if dtype.is::<Float16DataType>() {
            (DLDataTypeCode::FLOAT, 16)
        } else if dtype.is::<Float32DataType>() {
            (DLDataTypeCode::FLOAT, 32)
        } else if dtype.is::<Float64DataType>() {
            (DLDataTypeCode::FLOAT, 64)
        } else if dtype.is::<BFloat16DataType>() {
            (DLDataTypeCode::BFLOAT, 16)
        } else {
            return Err(PyValueError::new_err("Unsupported data type in dlpack").into());
        };
        Ok(DLDataType::scalar(code, bits))
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
