use pyo3::prelude::*;

#[derive(IntoPyObject, FromPyObject, Clone, Debug)]
pub struct PyChunkIndices(Vec<u64>);

impl AsRef<[u64]> for PyChunkIndices {
    fn as_ref(&self) -> &[u64] {
        &self.0
    }
}
