use std::sync::Arc;

use pyo3::prelude::*;
use zarrs::array::chunk_grid::{RectilinearChunkGrid, RegularBoundedChunkGrid};
use zarrs::array::ChunkGrid;

use crate::array::{PyArrayShape, PyChunkShape};
use crate::error::ZarristaResult;
use crate::metadata::PyMetadataV3;

#[derive(Debug, Clone)]
#[pyclass(module = "zarrista", frozen, name = "ChunkGrid", from_py_object)]
pub struct PyChunkGrid(ChunkGrid);

impl PyChunkGrid {
    pub fn new(chunk_grid: ChunkGrid) -> Self {
        Self(chunk_grid)
    }

    pub fn into_inner(self) -> ChunkGrid {
        self.0
    }
}

#[pymethods]
impl PyChunkGrid {
    #[staticmethod]
    fn regular_bounded(
        array_shape: PyArrayShape,
        chunk_shape: PyChunkShape,
    ) -> ZarristaResult<Self> {
        let chunk_grid =
            RegularBoundedChunkGrid::new(array_shape.into_inner(), chunk_shape.into_inner())?;
        Ok(Self(Arc::new(chunk_grid).into()))
    }

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
