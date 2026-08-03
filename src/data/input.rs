use std::borrow::Cow;

use bytes::Bytes;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_bytes::PyBytes;
use zarrs::array::{ArrayBytes, DataType};

use crate::array_bytes::PyArrayBytes;
use crate::error::ZarristaResult;

pub enum PyDataInput {
    /// Raw bytes. No type information, so length is all we can check.
    Bytes(PyBytes),
    /// A typed buffer from `__dlpack__` or the buffer protocol,
    /// normalised to C-contiguous native-order bytes at extraction.
    Typed {
        bytes: Bytes,
        input_dtype: DataType,
        input_shape: Vec<u64>,
    },
    /// The explicit `ArrayBytes` class: variable-length and masked data.
    ArrayBytes(PyArrayBytes),
}

impl PyDataInput {
    pub fn as_array_bytes(
        &self,
        data_type: &DataType,
        shape: &[u64],
    ) -> ZarristaResult<ArrayBytes<'_>> {
        match self {
            PyDataInput::Bytes(bytes) => Ok(ArrayBytes::Fixed(Cow::Borrowed(bytes.as_ref()))),
            PyDataInput::Typed {
                bytes,
                input_dtype,
                input_shape,
            } => {
                if input_dtype != data_type {
                    // A cast can lose data, so the user has to ask for it.
                    let input_name = data_type_display(input_dtype);
                    let target_name = data_type_display(data_type);
                    return Err(PyTypeError::new_err(format!(
                        "the data has type {input_name}, but the array has type {target_name}."
                    ))
                    .into());
                }
                if input_shape.as_slice() != shape {
                    return Err(PyValueError::new_err(format!(
                        "the data has shape {input_shape:?}, but the destination has shape {shape:?}."
                    ))
                    .into());
                }
                Ok(ArrayBytes::Fixed(Cow::Borrowed(bytes)))
            }
            PyDataInput::ArrayBytes(array_bytes) => Ok(array_bytes.as_array_bytes()?),
        }
    }
}

impl FromPyObject<'_, '_> for PyDataInput {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        // 1. Check for literal PyArrayBytes pyclass
        if let Ok(array_bytes) = obj.cast::<PyArrayBytes>() {
            return Ok(Self::ArrayBytes(array_bytes.get().clone()));
        }

        // 2. DLPack extraction
        if obj.hasattr(intern!(obj.py(), "__dlpack__"))? {
            return import_dlpack(obj);
        }

        // Anything else that exposes bytes. There is no type information here,
        // so the caller has opted out of the data type and shape checks.
        Ok(Self::Bytes(obj.extract::<PyBytes>()?))
    }
}

}

/// The Zarr v3 name of `data_type`, for use in a user-facing message.
///
/// `DataType`'s `Display` renders as `int32 / <i4`, which is informative but is
/// not something a user can paste into `astype`. Fall back to it only for a data
/// type that has no Zarr v3 name.
fn data_type_display(data_type: &DataType) -> Cow<'_, str> {
    data_type
        .name_v3()
        .unwrap_or_else(|| Cow::Owned(data_type.to_string()))
}
