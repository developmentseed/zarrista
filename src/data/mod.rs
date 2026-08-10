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
//! - [`PyFixedLengthTensor`] — fixed-width, dense. Buffer protocol + `to_numpy`.
//! - [`PyVariableLengthTensor`] — variable-length (string/bytes). (skeleton)
//! - [`PyOptionalFixedLengthTensor`] — fixed-width with a validity mask. (skeleton)
//! - [`PyOptionalVariableLengthTensor`] — variable-length with a validity mask. (skeleton)
//!
//! Buffers are **not** aligned: numpy's `frombuffer` tolerates unaligned data
//! (it sets `aligned=False`), and any consumer that materializes an owned array
//! pays the alignment copy as part of the copy it was already doing.

mod buffer_protocol;
mod dlpack;
mod fixed;
mod input;
mod variable;

use std::borrow::Cow;
use std::sync::Arc;

use bytes::Bytes;
pub use fixed::{PyFixedLengthTensor, PyOptionalFixedLengthTensor};
pub use input::PyDataInput;
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
pub use variable::{PyOptionalVariableLengthTensor, PyVariableLengthTensor};
use zarrs::array::{ArrayBytes, ArrayError, DataType, FromArrayBytes, data_type};

/// Internal decoded result, produced by our [`FromArrayBytes`] impl. Carries the
/// post-codec bytes (zero-copy), the data type, and the region shape (which
/// zarrs hands us, so we never have to re-derive it).
///
/// The variants mirror the `ArrayBytes` layouts that produce them.
pub enum Tensor {
    Fixed(PyFixedLengthTensor),
    Variable(PyVariableLengthTensor),
    OptionalFixed(PyOptionalFixedLengthTensor),
    OptionalVariable(PyOptionalVariableLengthTensor),
}

impl FromArrayBytes for Tensor {
    fn from_array_bytes(
        bytes: ArrayBytes<'static>,
        shape: &[u64],
        data_type: &DataType,
    ) -> Result<Self, ArrayError> {
        let shape = Arc::from(shape);
        let data_type = data_type.clone();
        Ok(match bytes {
            ArrayBytes::Fixed(bytes) => Tensor::Fixed(PyFixedLengthTensor::new(
                cow_to_bytes(bytes),
                data_type,
                shape,
            )?),
            ArrayBytes::Variable(v) => {
                let (buf, offsets) = v.into_parts();
                Tensor::Variable(PyVariableLengthTensor::new(
                    cow_to_bytes(buf),
                    offsets.to_vec(),
                    data_type,
                    shape,
                ))
            }
            ArrayBytes::Optional(optional) => {
                let (data, mask) = optional.into_parts();
                match *data {
                    ArrayBytes::Fixed(fixed) => {
                        Tensor::OptionalFixed(PyOptionalFixedLengthTensor::new(
                            PyFixedLengthTensor::new(
                                cow_to_bytes(fixed),
                                data_type,
                                shape.clone(),
                            )?,
                            PyFixedLengthTensor::new(cow_to_bytes(mask), data_type::bool(), shape)?,
                        ))
                    }
                    ArrayBytes::Variable(variable) => {
                        let (buf, offsets) = variable.into_parts();
                        Tensor::OptionalVariable(PyOptionalVariableLengthTensor::new(
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

impl<'py> IntoPyObject<'py> for Tensor {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            Tensor::Fixed(tensor) => tensor.into_bound_py_any(py),
            Tensor::Variable(tensor) => tensor.into_bound_py_any(py),
            Tensor::OptionalFixed(tensor) => tensor.into_bound_py_any(py),
            Tensor::OptionalVariable(tensor) => tensor.into_bound_py_any(py),
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
