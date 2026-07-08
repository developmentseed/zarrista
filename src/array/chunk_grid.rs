use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyInt;
use zarrs::array::chunk_grid::{
    ChunkEdgeLengths, RectilinearChunkGrid, RegularBoundedChunkGrid, RegularChunkGrid,
    RunLengthElement,
};
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
    fn rectilinear(
        array_shape: PyArrayShape,
        chunk_shapes: Vec<PyChunkEdgeLengths>,
    ) -> ZarristaResult<Self> {
        let chunk_shapes = chunk_shapes.into_iter().map(|c| c.0).collect::<Vec<_>>();
        let chunk_grid = RectilinearChunkGrid::new(array_shape, &chunk_shapes)?;
        Ok(Self(Arc::new(chunk_grid).into()))
    }

    #[staticmethod]
    fn regular(array_shape: PyArrayShape, chunk_shape: PyChunkShape) -> ZarristaResult<Self> {
        let chunk_grid = RegularChunkGrid::new(array_shape, chunk_shape)?;
        Ok(Self(Arc::new(chunk_grid).into()))
    }

    /// This chunk grid is experimental and may be incompatible with other Zarr V3 implementations.
    #[staticmethod]
    fn regular_bounded(
        array_shape: PyArrayShape,
        chunk_shape: PyChunkShape,
    ) -> ZarristaResult<Self> {
        let chunk_grid = RegularBoundedChunkGrid::new(array_shape, chunk_shape)?;
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

pub struct PyChunkEdgeLengths(ChunkEdgeLengths);

impl FromPyObject<'_, '_> for PyChunkEdgeLengths {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        if obj.is_instance_of::<PyInt>() {
            Ok(Self(ChunkEdgeLengths::Scalar(obj.extract()?)))
        } else {
            let elements = obj.extract::<Vec<PyRunLengthElement>>()?;
            Ok(Self(ChunkEdgeLengths::Varying(
                elements.into_iter().map(|e| e.0).collect(),
            )))
        }
    }
}

pub struct PyRunLengthElement(RunLengthElement);

impl FromPyObject<'_, '_> for PyRunLengthElement {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        if obj.is_instance_of::<PyInt>() {
            Ok(Self(RunLengthElement::Single(obj.extract()?)))
        } else {
            Ok(Self(RunLengthElement::Repeated(obj.extract()?)))
        }
    }
}
