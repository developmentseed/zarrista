//! Python `ArrayBytes`: an owned, zero-copy-friendly holder for the data that
//! crosses the Python/codec boundary.
//!
//! The class owns its underlying buffers (`PyBytes` keeps the Python buffer
//! alive via the buffer-protocol export), so a borrowing [`ArrayBytes<'_>`] can
//! be produced for the duration of a codec call without copying, and codec
//! output can be handed back to Python without copying either.

use std::borrow::Cow;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_bytes::PyBytes;
use zarrs::array::{ArrayBytes, ArrayBytesOffsets, ArrayBytesOptional, ArrayBytesVariableLength};

/// Bytes for a chunk as they cross the Python boundary.
#[pyclass(module = "zarrista", frozen, name = "ArrayBytes")]
pub struct PyArrayBytes(ArrayBytesOwned);

#[pymethods]
impl PyArrayBytes {
    #[new]
    #[pyo3(signature = (bytes, *, mask=None, offsets=None))]
    fn py_new(bytes: PyBytes, mask: Option<PyBytes>, offsets: Option<Vec<usize>>) -> Self {
        let data = match offsets {
            Some(offsets) => ArrayBytesOwned::Variable { bytes, offsets },
            None => ArrayBytesOwned::Fixed(bytes),
        };
        let repr = match mask {
            Some(mask) => ArrayBytesOwned::Optional {
                data: Box::new(data),
                mask,
            },
            None => data,
        };
        PyArrayBytes(repr)
    }

    /// The underlying element bytes (the data buffer for optional bytes).
    #[getter]
    fn bytes(&self) -> PyBytes {
        self.0.element_bytes().clone()
    }

    /// Element byte offsets, or `None` for fixed-length data.
    #[getter]
    fn offsets(&self) -> Option<Vec<usize>> {
        self.0.element_offsets().map(<[usize]>::to_vec)
    }

    /// The validity mask (1 byte per element), or `None` if not optional.
    #[getter]
    fn mask(&self) -> Option<PyBytes> {
        self.0.validity_mask().cloned()
    }

    fn __repr__(&self) -> String {
        let kind = match &self.0 {
            ArrayBytesOwned::Fixed(_) => "fixed",
            ArrayBytesOwned::Variable { .. } => "variable",
            ArrayBytesOwned::Optional { .. } => "optional",
        };
        format!(
            "ArrayBytes(<{kind}>, {} bytes)",
            self.0.element_bytes().as_slice().len()
        )
    }
}

impl PyArrayBytes {
    /// Borrow a zarrs [`ArrayBytes`] out of `self` for the duration of a codec
    /// call. Zero-copy: every buffer is borrowed from the owned `PyBytes`.
    ///
    /// # Errors
    /// Returns a `ValueError` if the offsets are not monotonically increasing or
    /// the final offset is out of bounds of the bytes buffer.
    pub fn as_array_bytes(&self) -> PyResult<ArrayBytes<'_>> {
        self.0.as_array_bytes()
    }

    /// Take ownership of codec output for handing back to Python.
    #[must_use]
    pub fn from_zarrs(bytes: ArrayBytes<'_>) -> Self {
        Self(bytes.into())
    }
}

/// The owned representation, mirroring zarrs' [`ArrayBytes`] sum type.
///
/// - Element buffers (fixed/variable payloads and the optional mask) are stored
///   as [`PyBytes`] and borrowed zero-copy.
/// - Offsets are copied into a `Vec<usize>` (they need a per-element cast from
///   Python ints and are tiny metadata relative to the payload).
enum ArrayBytesOwned {
    Fixed(PyBytes),
    Variable {
        bytes: PyBytes,
        offsets: Vec<usize>,
    },
    Optional {
        data: Box<ArrayBytesOwned>,
        mask: PyBytes,
    },
}

impl ArrayBytesOwned {
    fn as_array_bytes(&self) -> PyResult<ArrayBytes<'_>> {
        Ok(match self {
            ArrayBytesOwned::Fixed(bytes) => ArrayBytes::Fixed(Cow::Borrowed(bytes.as_slice())),
            ArrayBytesOwned::Variable { bytes, offsets } => {
                let offsets = ArrayBytesOffsets::new(Cow::Borrowed(offsets.as_slice()))
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let variable =
                    ArrayBytesVariableLength::new(Cow::Borrowed(bytes.as_slice()), offsets)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                ArrayBytes::Variable(variable)
            }
            ArrayBytesOwned::Optional { data, mask } => {
                let data = data.as_array_bytes()?;
                ArrayBytes::Optional(ArrayBytesOptional::new(
                    data,
                    Cow::Borrowed(mask.as_slice()),
                ))
            }
        })
    }

    /// The element bytes buffer, descending into optional data.
    fn element_bytes(&self) -> &PyBytes {
        match self {
            ArrayBytesOwned::Fixed(bytes) | ArrayBytesOwned::Variable { bytes, .. } => bytes,
            ArrayBytesOwned::Optional { data, .. } => data.element_bytes(),
        }
    }

    /// The element offsets, descending into optional data.
    fn element_offsets(&self) -> Option<&[usize]> {
        match self {
            ArrayBytesOwned::Fixed(_) => None,
            ArrayBytesOwned::Variable { offsets, .. } => Some(offsets),
            ArrayBytesOwned::Optional { data, .. } => data.element_offsets(),
        }
    }

    fn validity_mask(&self) -> Option<&PyBytes> {
        match self {
            ArrayBytesOwned::Optional { mask, .. } => Some(mask),
            ArrayBytesOwned::Fixed(_) | ArrayBytesOwned::Variable { .. } => None,
        }
    }
}

impl From<ArrayBytes<'_>> for ArrayBytesOwned {
    fn from(bytes: ArrayBytes<'_>) -> Self {
        match bytes {
            ArrayBytes::Fixed(bytes) => ArrayBytesOwned::Fixed(cow_to_pybytes(bytes)),
            ArrayBytes::Variable(variable) => {
                let (bytes, offsets) = variable.into_parts();
                ArrayBytesOwned::Variable {
                    bytes: cow_to_pybytes(bytes),
                    offsets: offsets.to_vec(),
                }
            }
            ArrayBytes::Optional(optional) => {
                let (data, mask) = optional.into_parts();
                ArrayBytesOwned::Optional {
                    data: Box::new(ArrayBytesOwned::from(*data)),
                    mask: cow_to_pybytes(mask),
                }
            }
        }
    }
}

/// Wrap a `Cow<[u8]>` as `PyBytes`, moving the allocation when already owned.
///
/// A `Cow::Borrowed` is copied once into a fresh allocation.
pub(crate) fn cow_to_pybytes(bytes: Cow<'_, [u8]>) -> PyBytes {
    PyBytes::from(bytes.into_owned())
}
