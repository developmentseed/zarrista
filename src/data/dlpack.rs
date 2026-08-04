use std::ffi::c_void;

use dlpark::ffi::{
    DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLManagedTensorVersioned,
    DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION, DLTensor,
};
use dlpark::metadata::CopiedSlice;
use dlpark::python::device::dlpack_device;
use dlpark::{Builder, ManagedBox, legacy};
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

impl PyManagedTensor {
    /// The DLPack tensor description: data pointer, device, shape, and strides.
    pub fn tensor(&self) -> &DLTensor {
        self.0
            .as_ref()
            .expect("the tensor is taken only while dropping")
            .tensor()
    }

    /// The tensor's bytes, borrowed for as long as this owns them.
    ///
    /// This returns an `Err` for a tensor that is not on the host, or whose strides are not
    /// compact.
    pub fn as_bytes(&self) -> ZarristaResult<&[u8]> {
        let tensor = self.tensor();

        // Every call below needs the same thing: that the producer's `shape`,
        // `strides`, and `data` pointers describe what the DLPack ABI says they
        // do. `Self` owns the tensor, so the deleter has not run and none of
        // them dangle. dlpark reports a null or negative `ndim` as an error
        // rather than reading through it, so only a producer that lies about
        // its own metadata could break these.

        // SAFETY: the shape and strides pointers are the producer's own.
        let is_compact = unsafe { tensor.is_compact() }.map_err(dlpack_import_error)?;
        if !is_compact {
            // Load-bearing, not just a convenience: `num_bytes` below describes
            // the logical tensor, which only matches the bytes at the data
            // pointer when the strides are compact.
            return Err(PyValueError::new_err(
                "the data is not C-contiguous. Make it contiguous first, \
                 for example with `numpy.ascontiguousarray`.",
            )
            .into());
        }

        // SAFETY: the shape pointer is the producer's own.
        let len = unsafe { tensor.num_bytes() }.map_err(dlpack_import_error)?;
        if len == 0 {
            return Ok(&[]);
        }

        // SAFETY: the shape and data pointers are the producer's own.
        //
        // This rejects a tensor on another device, so a device pointer can never reach the slice
        // below, and it rejects a null pointer for a tensor that is not empty.
        let ptr = unsafe { tensor.cpu_data_ptr_bytes() }.map_err(dlpack_import_error)?;

        // SAFETY: `ptr` is non-null, and `u8` needs no alignment.
        //
        // The tensor is compact, so `len` bytes from `ptr` are exactly its initialized elements,
        // inside the producer's allocation.
        //
        // The DLPack contract gives a consumer read access until it runs the deleter, which `Self`
        // owns and has not run. The lifetime is the same as `&self`, so the slice cannot outlive
        // the tensor that backs it.
        Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
    }

    /// Access the zarr data type
    pub fn data_type(&self) -> ZarristaResult<DataType> {
        self.tensor().dtype.zarrs_data_type()
    }

    /// The tensor's shape, in elements along each dimension.
    pub fn shape(&self) -> ZarristaResult<Vec<u64>> {
        // SAFETY: `Self` owns the tensor, so the producer's shape, strides, and
        // data pointer stay valid and readable for this borrow.
        let shape = unsafe { self.tensor().shape().map_err(dlpack_import_error)? };
        shape
            .iter()
            .map(|dim| u64::try_from(*dim))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| PyValueError::new_err("the data has a negative dimension").into())
    }
}

impl FromPyObject<'_, '_> for PyManagedTensor {
    type Error = PyErr;

    /// Import a DLPack tensor, asking the provider to copy it to CPU memory if necessary.
    fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        let py = obj.py();
        let kwargs = PyDict::new(py);
        kwargs.set_item(
            intern!(py, "max_version"),
            (DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION),
        )?;
        kwargs.set_item(intern!(py, "dl_device"), (DLDeviceType::CPU.0, 0))?;

        let capsule = obj
            .call_method(intern!(py, "__dlpack__"), (), Some(&kwargs))
            .map_err(|err| {
                let device = dlpack_device(obj).map_or_else(
                    |_| "an unknown device".to_string(),
                    |device| format!("device {:?}", device.device_type),
                );
                PyValueError::new_err(format!(
                    "The data is on {device}, and the producer could not move it to the host: {err}."
                ))
            })?;

        // dlpark reads a capsule directly, so this does not call `__dlpack__`
        // a second time.
        Ok(Self(Some(
            capsule.extract::<ManagedBox<DLManagedTensorVersioned>>()?,
        )))
    }
}

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

/// Add DLPack error prefix text
fn dlpack_import_error(err: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(format!("Error in DLPack import: {err}"))
}
