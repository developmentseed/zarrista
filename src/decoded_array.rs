//! Decoded array data exposed to Python.
//!
//! SPIKE: instead of decoding into a typed `ndarray::ArrayD<T>` (which copies
//! every buffer via `bytemuck::pod_collect_to_vec`), we retrieve the raw
//! post-codec [`ArrayBytes`] and wrap it zero-copy. Retrieval is a single
//! generic call (`retrieve::<DecodedArray>`) — no per-dtype macro — because we
//! implement [`FromArrayBytes`] on our own [`DecodedArray`] type.
//!
//! `ArrayBytes` has three layouts (`Fixed`, `Variable`, `Optional`), which we
//! surface as four concrete Python classes so each exposes exactly the faces it
//! can support:
//!
//! - [`PyTensor`] — fixed-width, dense. Buffer protocol + `to_numpy`.
//! - [`PyVariableArray`] — variable-length (string/bytes). (skeleton)
//! - [`PyMaskedTensor`] — fixed-width with a validity mask. (skeleton)
//! - [`PyMaskedVariableArray`] — variable-length with a validity mask. (skeleton)
//!
//! Buffers are **not** aligned: numpy's `frombuffer` tolerates unaligned data
//! (it sets `aligned=False`), and any consumer that materializes an owned array
//! pays the alignment copy as part of the copy it was already doing.

use std::borrow::Cow;
use std::ffi::c_void;

use bytes::Bytes;
use dlpark::ffi::{Device, DeviceType};
use dlpark::traits::{RowMajorCompactLayout, TensorLike};
use dlpark::SafeManagedTensor;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::IntoPyObjectExt;
use pyo3_bytes::PyBytes;
use zarrs::array::{ArrayBytes, ArrayError, DataType, FromArrayBytes};

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

/// Variable-length data (string/bytes). Skeleton: carries metadata only for now.
#[pyclass(module = "zarrista", frozen, name = "VariableArray")]
pub struct PyVariableArray {
    #[expect(dead_code)]
    bytes: Bytes,
    #[expect(dead_code)]
    offsets: Vec<usize>,
    data_type: DataType,
    shape: Vec<u64>,
}

#[pymethods]
impl PyVariableArray {
    #[getter]
    fn shape(&self) -> &[u64] {
        &self.shape
    }

    #[getter]
    fn dtype(&self) -> PyDataType {
        self.data_type.clone().into()
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

/// Variable-length data with a validity mask. Skeleton.
#[pyclass(module = "zarrista", frozen, name = "MaskedVariableArray")]
pub struct PyMaskedVariableArray {
    #[expect(dead_code)]
    bytes: Bytes,
    #[expect(dead_code)]
    offsets: Vec<usize>,
    /// The mask is 1 byte per element where 0 = invalid/missing, non-zero = valid/present.
    #[expect(dead_code)]
    mask: Bytes,
    data_type: DataType,
    shape: Vec<u64>,
}

#[pymethods]
impl PyMaskedVariableArray {
    #[getter]
    fn shape(&self) -> &[u64] {
        &self.shape
    }

    #[getter]
    fn dtype(&self) -> PyDataType {
        self.data_type.clone().into()
    }
}

/// Internal decoded result, produced by our [`FromArrayBytes`] impl. Carries the
/// post-codec bytes (zero-copy), the data type, and the region shape (which
/// zarrs hands us, so we never have to re-derive it).
pub enum DecodedArray {
    Tensor(PyTensor),
    Variable(PyVariableArray),
    MaskedTensor(PyMaskedTensor),
    MaskedVariable(PyMaskedVariableArray),
}

impl FromArrayBytes for DecodedArray {
    fn from_array_bytes(
        bytes: ArrayBytes<'static>,
        shape: &[u64],
        data_type: &DataType,
    ) -> Result<Self, ArrayError> {
        let shape = shape.to_vec();
        let data_type = data_type.clone();
        Ok(match bytes {
            ArrayBytes::Fixed(bytes) => DecodedArray::Tensor(PyTensor {
                bytes: cow_to_bytes(bytes),
                data_type,
                shape,
            }),
            ArrayBytes::Variable(v) => {
                let (buf, offsets) = v.into_parts();
                DecodedArray::Variable(PyVariableArray {
                    bytes: cow_to_bytes(buf),
                    // Ideally in the future we'll avoid a copy:
                    // https://github.com/zarrs/zarrs/issues/406
                    offsets: offsets.to_vec(),
                    data_type,
                    shape,
                })
            }
            ArrayBytes::Optional(optional) => {
                let (data, mask) = optional.into_parts();
                match *data {
                    ArrayBytes::Fixed(fixed) => DecodedArray::MaskedTensor(PyMaskedTensor {
                        bytes: cow_to_bytes(fixed),
                        mask: cow_to_bytes(mask),
                        data_type,
                        shape,
                    }),
                    ArrayBytes::Variable(variable) => {
                        let (buf, offsets) = variable.into_parts();
                        DecodedArray::MaskedVariable(PyMaskedVariableArray {
                            bytes: cow_to_bytes(buf),
                            // Ideally in the future we'll avoid a copy:
                            // https://github.com/zarrs/zarrs/issues/406
                            offsets: offsets.to_vec(),
                            mask: cow_to_bytes(mask),
                            data_type,
                            shape,
                        })
                    }
                    ArrayBytes::Optional(_) => {
                        unreachable!("nested optional is not a valid layout")
                    }
                }
            }
        })
    }
}

impl<'py> IntoPyObject<'py> for DecodedArray {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            DecodedArray::Tensor(py_tensor) => py_tensor.into_bound_py_any(py),
            DecodedArray::Variable(py_variable_array) => py_variable_array.into_bound_py_any(py),
            DecodedArray::MaskedTensor(py_masked_tensor) => py_masked_tensor.into_bound_py_any(py),
            DecodedArray::MaskedVariable(py_masked_variable_array) => {
                py_masked_variable_array.into_bound_py_any(py)
            }
        }
    }
}

/// Move a `'static` `Cow<[u8]>` into `bytes::Bytes`. Owned is a zero-copy move;
/// borrowed (rare for retrieval) copies.
fn cow_to_bytes(cow: Cow<'static, [u8]>) -> Bytes {
    match cow {
        Cow::Owned(v) => Bytes::from(v),
        Cow::Borrowed(b) => Bytes::copy_from_slice(b),
    }
}
