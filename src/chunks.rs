use pyo3::prelude::*;
use zarrs::array::ChunkGrid;

#[pyclass(module = "zarrsita", frozen, name = "ChunkGrid")]
pub struct PyChunkGrid(ChunkGrid);

#[pymethods]
impl PyChunkGrid {
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
