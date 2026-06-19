//! Decoded array data exposed to Python.
//!
//! SPIKE: instead of decoding into a typed `ndarray::ArrayD<T>` (which copies
//! every buffer via `bytemuck::pod_collect_to_vec`), we retrieve the raw
//! post-codec [`ArrayBytes`] and wrap it zero-copy. Retrieval is a single
//! generic call (`retrieve::<Decoded>`) — no per-dtype macro — because we
//! implement [`FromArrayBytes`] on our own [`Decoded`] type.
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

use bytes::Bytes;
use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;
use pyo3_bytes::PyBytes;
use zarrs::array::{ArrayBytes, ArrayError, DataType, FromArrayBytes};

use crate::dtype::PyDataType;

/// Fixed-width, dense decoded data.
///
/// We don't use the upstream `Tensor` type because its bytes are not reference counted, and thus
/// don't play nicely with buffer protocol export
#[pyclass(module = "zarrista", frozen, name = "Tensor")]
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
}

/// Internal decoded result, produced by our [`FromArrayBytes`] impl. Carries the
/// post-codec bytes (zero-copy), the data type, and the region shape (which
/// zarrs hands us, so we never have to re-derive it).
pub enum Decoded {
    Tensor(PyTensor),
    Variable {
        bytes: ArrayBytes<'static>,
        data_type: DataType,
        shape: Vec<u64>,
    },
    MaskedTensor {
        bytes: ArrayBytes<'static>,
        data_type: DataType,
        shape: Vec<u64>,
    },
    MaskedVariable {
        bytes: ArrayBytes<'static>,
        data_type: DataType,
        shape: Vec<u64>,
    },
}

/// Move a `'static` `Cow<[u8]>` into `bytes::Bytes`. Owned is a zero-copy move;
/// borrowed (rare for retrieval) copies.
fn cow_to_bytes(cow: Cow<'static, [u8]>) -> Bytes {
    match cow {
        Cow::Owned(v) => Bytes::from(v),
        Cow::Borrowed(b) => Bytes::copy_from_slice(b),
    }
}

impl FromArrayBytes for Decoded {
    fn from_array_bytes(
        bytes: ArrayBytes<'static>,
        shape: &[u64],
        data_type: &DataType,
    ) -> Result<Self, ArrayError> {
        let shape = shape.to_vec();
        let data_type = data_type.clone();
        Ok(match bytes {
            ArrayBytes::Fixed(b) => Decoded::Tensor(PyTensor {
                bytes: cow_to_bytes(b),
                data_type,
                shape,
            }),
            ArrayBytes::Variable(v) => Decoded::Variable {
                bytes: ArrayBytes::Variable(v),
                data_type,
                shape,
            },
            ArrayBytes::Optional(o) => {
                // Peek at the inner layout to pick the masked class, then move
                // the value back into an owned `ArrayBytes`.
                let inner_is_variable = matches!(o.data(), ArrayBytes::Variable(_));
                let bytes = ArrayBytes::Optional(o);
                if inner_is_variable {
                    Decoded::MaskedVariable {
                        bytes,
                        data_type,
                        shape,
                    }
                } else {
                    Decoded::MaskedTensor {
                        bytes,
                        data_type,
                        shape,
                    }
                }
            }
        })
    }
}

/// Convert into the appropriate concrete Python result class. Implemented as
/// `IntoPyObject` so both the sync and async retrieve paths can simply return
/// `Decoded` and let pyo3 build the Python object once the GIL is held.
impl<'py> IntoPyObject<'py> for Decoded {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let obj = match self {
            Decoded::Tensor(py_tensor) => return py_tensor.into_bound_py_any(py),
            Decoded::Variable {
                data_type, shape, ..
            } => Bound::new(py, PyVariableArray { data_type, shape })?.into_any(),
            Decoded::MaskedTensor {
                data_type, shape, ..
            } => Bound::new(py, PyMaskedTensor { data_type, shape })?.into_any(),
            Decoded::MaskedVariable {
                data_type, shape, ..
            } => Bound::new(py, PyMaskedVariableArray { data_type, shape })?.into_any(),
        };
        Ok(obj)
    }
}

/// Variable-length data (string/bytes). Skeleton: carries metadata only for now.
#[pyclass(module = "zarrista", frozen, name = "VariableArray")]
pub struct PyVariableArray {
    data_type: DataType,
    shape: Vec<u64>,
}

#[pymethods]
impl PyVariableArray {
    #[getter]
    fn shape(&self) -> Vec<u64> {
        self.shape.clone()
    }

    #[getter]
    fn dtype(&self) -> PyDataType {
        self.data_type.clone().into()
    }

    fn to_numpy(&self) -> PyResult<()> {
        Err(PyNotImplementedError::new_err(
            "variable-length data is not yet exposed to numpy",
        ))
    }
}

/// Fixed-width data with a validity mask. Skeleton.
#[pyclass(module = "zarrista", frozen, name = "MaskedTensor")]
pub struct PyMaskedTensor {
    data_type: DataType,
    shape: Vec<u64>,
}

#[pymethods]
impl PyMaskedTensor {
    #[getter]
    fn shape(&self) -> Vec<u64> {
        self.shape.clone()
    }

    #[getter]
    fn dtype(&self) -> PyDataType {
        self.data_type.clone().into()
    }

    fn to_numpy(&self) -> PyResult<()> {
        Err(PyNotImplementedError::new_err(
            "masked data is not yet exposed to numpy",
        ))
    }
}

/// Variable-length data with a validity mask. Skeleton.
#[pyclass(module = "zarrista", frozen, name = "MaskedVariableArray")]
pub struct PyMaskedVariableArray {
    data_type: DataType,
    shape: Vec<u64>,
}

#[pymethods]
impl PyMaskedVariableArray {
    #[getter]
    fn shape(&self) -> Vec<u64> {
        self.shape.clone()
    }

    #[getter]
    fn dtype(&self) -> PyDataType {
        self.data_type.clone().into()
    }

    fn to_numpy(&self) -> PyResult<()> {
        Err(PyNotImplementedError::new_err(
            "masked variable-length data is not yet exposed to numpy",
        ))
    }
}
