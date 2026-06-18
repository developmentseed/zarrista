use std::borrow::Cow;

use pyo3::prelude::*;
use pyo3_bytes::PyBytes;
use zarrs::array::{ArrayBytes, ArrayBytesVariableLength};

// pub struct PyArrayBytes2<'a>(ArrayBytes<'a>);

#[derive(Debug, FromPyObject)]
pub struct PyFixedArrayBytes(PyBytes);

impl PyFixedArrayBytes {
    fn as_array_bytes(&self) -> ArrayBytes<'_> {
        ArrayBytes::Fixed(Cow::Borrowed(self.0.as_ref()))
    }
}

#[derive(Debug, FromPyObject)]
pub struct PyVariableArrayBytes {
    bytes: PyBytes,
    offsets: Vec<usize>,
}

impl PyVariableArrayBytes {
    fn as_array_bytes(&self) -> ArrayBytes<'_> {
        ArrayBytes::Variable(ArrayBytesVariableLength::new(
            self.bytes.as_ref(),
            &self.offsets,
        ))
    }
}

pub struct PyArrayBytesOptional {
    data: Box<PyArrayBytes>,
    mask: PyBytes,
}

pub enum PyArrayBytes {
    Fixed(PyFixedArrayBytes),
    Variable(PyVariableArrayBytes),
    Optional(PyArrayBytesOptional),
}
