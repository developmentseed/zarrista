use std::borrow::Cow;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use zarrs::array::chunk_key_encoding::DefaultChunkKeyEncoding;
use zarrs::array::{ChunkKeyEncoding, ChunkKeySeparator};

use crate::error::ZarristaResult;
use crate::metadata::PyMetadataV3;

#[derive(Debug, Clone)]
#[pyclass(module = "zarrista", frozen, name = "ChunkKeyEncoding", from_py_object)]
pub struct PyChunkKeyEncoding(ChunkKeyEncoding);

crate::wasm_send_sync!(PyChunkKeyEncoding);

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
    fn __repr__(&self, py: Python) -> PyResult<String> {
        let metadata = self.0.metadata();
        crate::repr::named_config_repr(
            py,
            "ChunkKeyEncoding",
            Some(Cow::Borrowed(metadata.name())),
            metadata.configuration().cloned().map(Into::into),
        )
    }

    // TODO: not sure whether we want constructors as classmethods or as free functions.
    #[staticmethod]
    fn default(sep: PyChunkKeySeparator) -> Self {
        let encoding = DefaultChunkKeyEncoding::new(sep.0);
        Self(Arc::new(encoding).into())
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

impl From<PyChunkKeyEncoding> for ChunkKeyEncoding {
    fn from(encoding: PyChunkKeyEncoding) -> Self {
        encoding.0
    }
}

impl From<ChunkKeyEncoding> for PyChunkKeyEncoding {
    fn from(encoding: ChunkKeyEncoding) -> Self {
        Self(encoding)
    }
}

#[derive(Debug, Clone)]
pub struct PyChunkKeySeparator(ChunkKeySeparator);

impl FromPyObject<'_, '_> for PyChunkKeySeparator {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        let s = obj.extract::<PyBackedStr>()?;
        match s.to_ascii_lowercase().as_str() {
            "." => Ok(Self(ChunkKeySeparator::Dot)),
            "/" => Ok(Self(ChunkKeySeparator::Slash)),
            _ => Err(PyValueError::new_err(format!(
                "Invalid chunk key separator: {}",
                s
            ))),
        }
    }
}
