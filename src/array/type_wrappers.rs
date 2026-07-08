//! Newtype wrappers of upstream types to implement FromPyObject and IntoPyObject
//!
//! These wrappers are **not** standalone Python classes; they only define serde
//!
//! We can use `pub type` type aliases instead of newtype wrappers whenever upstream types are
//! implemented as type aliases and whenever the underlying type already has FromPyObject and
//! IntoPyObject implemented.
//!
//! Keep alphabetical ordering.

use pyo3::prelude::*;
use pyo3::types::{PySlice, PyTuple};
use zarrs::array::{ArrayIndices, ArrayShape, ArraySubset, ChunkShape, DimensionName};

/// An ND index to an element in an array or chunk.
pub type PyArrayIndices = ArrayIndices;

/// An array shape. Dimensions may be zero.
pub type PyArrayShape = ArrayShape;

/// An array subset.
#[derive(Clone, Debug)]
pub struct PyArraySubset(ArraySubset);

impl<'py> IntoPyObject<'py> for PyArraySubset {
    type Target = PyTuple;
    type Error = PyErr;
    type Output = Bound<'py, Self::Target>;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let ranges = self.0.to_ranges();
        let slices = ranges.into_iter().map(|range| {
            PySlice::new(
                py,
                range.start.try_into().expect("u64 to i64 overflow"),
                range.end.try_into().expect("u64 to i64 overflow"),
                1,
            )
        });
        PyTuple::new(py, slices)
    }
}

impl From<PyArraySubset> for ArraySubset {
    fn from(subset: PyArraySubset) -> Self {
        subset.0
    }
}

impl From<ArraySubset> for PyArraySubset {
    fn from(subset: ArraySubset) -> Self {
        Self(subset)
    }
}

impl AsRef<ArraySubset> for PyArraySubset {
    fn as_ref(&self) -> &ArraySubset {
        &self.0
    }
}

/// Chunk indices
pub type PyChunkIndices = Vec<u64>;

/// A chunk shape. Dimensions must be non-zero.
pub type PyChunkShape = ChunkShape;

/// A dimension name.
pub type PyDimensionName = DimensionName;
