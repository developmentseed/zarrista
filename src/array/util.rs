use pyo3::prelude::*;

#[derive(IntoPyObject, FromPyObject, Clone, Debug)]
pub struct PyChunkIndices(Vec<u64>);

impl AsRef<[u64]> for PyChunkIndices {
    fn as_ref(&self) -> &[u64] {
        &self.0
    }
}

#[derive(IntoPyObject, FromPyObject, Clone, Debug)]
pub struct PyArrayShape(Vec<u64>);

impl From<Vec<u64>> for PyArrayShape {
    fn from(shape: Vec<u64>) -> Self {
        Self(shape)
    }
}

impl From<PyArrayShape> for Vec<u64> {
    fn from(shape: PyArrayShape) -> Self {
        shape.0
    }
}

impl AsRef<[u64]> for PyArrayShape {
    fn as_ref(&self) -> &[u64] {
        &self.0
    }
}
