//! Newtype wrappers of upstream types to implement FromPyObject and IntoPyObject
//!
//! These wrappers are **not** standalone Python classes; they only define serde

use pyo3::prelude::*;
use pyo3::types::{PySlice, PyTuple};
use zarrs::array::{ArrayIndices, ArrayShape, ArraySubset, ChunkShape};

/// An ND index to an element in an array or chunk.
#[derive(Debug, Clone, PartialEq, Eq, Hash, FromPyObject, IntoPyObject, IntoPyObjectRef)]
pub struct PyArrayIndices(ArrayIndices);

impl AsRef<ArrayIndices> for PyArrayIndices {
    fn as_ref(&self) -> &ArrayIndices {
        &self.0
    }
}

impl From<PyArrayIndices> for ArrayIndices {
    fn from(py_key: PyArrayIndices) -> Self {
        py_key.0
    }
}

impl From<ArrayIndices> for PyArrayIndices {
    fn from(key: ArrayIndices) -> Self {
        Self(key)
    }
}

#[derive(IntoPyObject, FromPyObject, Clone, Debug)]
pub struct PyArrayShape(ArrayShape);

impl PyArrayShape {
    pub fn into_inner(self) -> ArrayShape {
        self.0
    }
}

impl From<ArrayShape> for PyArrayShape {
    fn from(shape: ArrayShape) -> Self {
        Self(shape)
    }
}

impl From<PyArrayShape> for ArrayShape {
    fn from(shape: PyArrayShape) -> Self {
        shape.0
    }
}

impl AsRef<[u64]> for PyArrayShape {
    fn as_ref(&self) -> &[u64] {
        &self.0
    }
}

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

#[derive(IntoPyObject, FromPyObject, Clone, Debug)]
pub struct PyChunkIndices(Vec<u64>);

impl AsRef<[u64]> for PyChunkIndices {
    fn as_ref(&self) -> &[u64] {
        &self.0
    }
}

#[derive(IntoPyObject, FromPyObject, Clone, Debug)]
pub struct PyChunkShape(ChunkShape);

impl PyChunkShape {
    pub fn into_inner(self) -> ChunkShape {
        self.0
    }
}

impl From<ChunkShape> for PyChunkShape {
    fn from(shape: ChunkShape) -> Self {
        Self(shape)
    }
}

impl From<PyChunkShape> for ChunkShape {
    fn from(shape: PyChunkShape) -> Self {
        shape.0
    }
}

impl AsRef<ChunkShape> for PyChunkShape {
    fn as_ref(&self) -> &ChunkShape {
        &self.0
    }
}
