//! Decoded array data exposed to Python.
//!
//! Instead of choosing one preferred representation of decoded array data, such as numpy, or
//! arrow, or dlpack, we expose the raw post-codec bytes and let consumers choose how to
//! materialize them.
//!
//! This ensures that users always have a some way to zero-copy access the resulting data.
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

mod tensor;
mod variable;

pub use tensor::{PyMaskedTensor, PyTensor};
pub use variable::{PyMaskedVariableArray, PyVariableArray};

use std::borrow::Cow;

use bytes::Bytes;
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use zarrs::array::{ArrayBytes, ArrayError, DataType, FromArrayBytes};

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
            ArrayBytes::Fixed(bytes) => {
                DecodedArray::Tensor(PyTensor::new(cow_to_bytes(bytes), data_type, shape))
            }
            ArrayBytes::Variable(v) => {
                let (buf, offsets) = v.into_parts();
                DecodedArray::Variable(PyVariableArray::new(
                    cow_to_bytes(buf),
                    offsets.to_vec(),
                    data_type,
                    shape,
                ))
            }
            ArrayBytes::Optional(optional) => {
                let (data, mask) = optional.into_parts();
                match *data {
                    ArrayBytes::Fixed(fixed) => DecodedArray::MaskedTensor(PyMaskedTensor::new(
                        cow_to_bytes(fixed),
                        cow_to_bytes(mask),
                        data_type,
                        shape,
                    )),
                    ArrayBytes::Variable(variable) => {
                        let (buf, offsets) = variable.into_parts();
                        DecodedArray::MaskedVariable(PyMaskedVariableArray::new(
                            cow_to_bytes(buf),
                            offsets.to_vec(),
                            cow_to_bytes(mask),
                            data_type,
                            shape,
                        ))
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
