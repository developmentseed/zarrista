//! Holds [PyDataInput] and manages conversion from Python in-memory array-like objects into
//! Rust-accessible data

use std::borrow::Cow;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3_bytes::PyBytes;
use zarrs::array::{ArrayBytes, DataType};

use crate::array_bytes::PyArrayBytes;
use crate::data::dlpack::PyManagedTensor;
use crate::error::ZarristaResult;

pub enum PyDataInput {
    /// Raw bytes. No type information, so length is all we can check.
    Bytes(PyBytes),
    /// A DLPack tensor
    DLPack(PyManagedTensor),
    /// The explicit `ArrayBytes` class: variable-length and masked data.
    ArrayBytes(PyArrayBytes),
}

impl PyDataInput {
    pub fn as_array_bytes(
        &self,
        array_data_type: &DataType,
        array_shape: &[u64],
    ) -> ZarristaResult<ArrayBytes<'_>> {
        match self {
            PyDataInput::Bytes(bytes) => Ok(ArrayBytes::Fixed(Cow::Borrowed(bytes.as_ref()))),
            PyDataInput::DLPack(tensor) => {
                let data_data_type = &tensor.data_type()?;
                if data_data_type != array_data_type {
                    // A cast can lose data, so the user has to ask for it.
                    let input_name = data_type_display(data_data_type);
                    let target_name = data_type_display(array_data_type);
                    return Err(PyTypeError::new_err(format!(
                        "the data has type {input_name}, but the array has type {target_name}."
                    ))
                    .into());
                }

                let data_shape = tensor.shape()?;
                if data_shape.as_slice() != array_shape {
                    return Err(PyValueError::new_err(format!(
                        "the data has shape {data_shape:?}, but the destination has shape {array_shape:?}."
                    ))
                    .into());
                }

                Ok(ArrayBytes::Fixed(Cow::Borrowed(tensor.as_bytes()?)))
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
            return Ok(Self::DLPack(obj.extract()?));
        }

        // Anything else that exposes bytes. There is no type information here,
        // so the caller has opted out of the data type and shape checks.
        if let Ok(buf) = obj.extract::<PyBytes>() {
            return Ok(Self::Bytes(buf));
        }

        let type_name = obj.get_type().name()?;
        Err(PyTypeError::new_err(format!(
            "Expected one of `PyArrayBytes`, a DLPack tensor, or a buffer, but got {type_name}."
        )))
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
