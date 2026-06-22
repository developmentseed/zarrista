use pyo3::prelude::*;
use zarrs::array::ChunkGrid;

use crate::error::ZarristaResult;
use crate::metadata::PyMetadataV3;

#[derive(Debug, Clone)]
#[pyclass(module = "zarrista", frozen, name = "ChunkGrid", from_py_object)]
pub struct PyChunkGrid(ChunkGrid);

#[pymethods]
impl PyChunkGrid {
    #[staticmethod]
    fn from_metadata(metadata: PyMetadataV3, shape: Vec<u64>) -> ZarristaResult<Self> {
        Ok(Self(ChunkGrid::from_metadata(metadata.as_ref(), &shape)?))
    }

    #[getter]
    fn metadata(&self) -> PyMetadataV3 {
        self.0.metadata().into()
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.0.dimensionality()
    }

    #[getter]
    fn array_shape(&self) -> &[u64] {
        self.0.array_shape()
    }

    #[getter]
    fn grid_shape(&self) -> &[u64] {
        self.0.grid_shape()
    }
}

impl From<ChunkGrid> for PyChunkGrid {
    fn from(chunk_grid: ChunkGrid) -> Self {
        PyChunkGrid(chunk_grid)
    }
}

impl From<PyChunkGrid> for ChunkGrid {
    fn from(py_chunk_grid: PyChunkGrid) -> Self {
        py_chunk_grid.0
    }
}
