use std::borrow::Cow;

use pyo3::prelude::*;
use zarrs::array::ChunkKeyEncoding;

use crate::error::ZarristaResult;
use crate::metadata::PyMetadataV3;

#[derive(Debug, Clone)]
#[pyclass(module = "zarrista", frozen, name = "ChunkKeyEncoding", from_py_object)]
pub struct PyChunkKeyEncoding(ChunkKeyEncoding);

impl PyChunkKeyEncoding {
    pub fn into_inner(self) -> ChunkKeyEncoding {
        self.0
    }

    pub fn new(encoding: ChunkKeyEncoding) -> Self {
        Self(encoding)
    }
}

#[pymethods]
impl PyChunkKeyEncoding {
    fn __repr__(&self) -> String {
        format!("ChunkKeyEncoding({:?})", self.0)
    }

    #[staticmethod]
    fn from_metadata(metadata: PyMetadataV3) -> ZarristaResult<Self> {
        Ok(Self::new(ChunkKeyEncoding::from_metadata(
            metadata.as_ref(),
        )?))
    }

    /// The codec's Zarr v3 metadata
    #[getter]
    fn metadata(&self) -> PyMetadataV3 {
        self.0.metadata().into()
    }

    /// The codec's Zarr v3 name if it has one.
    #[getter]
    fn name(&self) -> Option<Cow<'static, str>> {
        self.0.name_v3()
    }
}
