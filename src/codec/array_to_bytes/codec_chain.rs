use std::sync::Arc;

pub use pyo3::prelude::*;
use zarrs::array::CodecChain;

use crate::codec::{PyArrayToArrayCodec, PyArrayToBytesCodec, PyBytesToBytesCodec};

#[pyclass(module = "zarrista.codec", frozen, name = "CodecChain")]
pub struct PyCodecChain(Arc<CodecChain>);

#[pymethods]
impl PyCodecChain {
    /// The bytes-to-bytes codecs ("compressors").
    #[getter]
    fn compressors(&self) -> Vec<PyBytesToBytesCodec> {
        self.0
            .bytes_to_bytes_codecs()
            .iter()
            .map(|c| PyBytesToBytesCodec::new(c.clone()))
            .collect()
    }

    /// The array-to-array codecs ("filters").
    #[getter]
    fn filters(&self) -> Vec<PyArrayToArrayCodec> {
        self.0
            .array_to_array_codecs()
            .iter()
            .map(|f| PyArrayToArrayCodec::new(f.clone()))
            .collect()
    }

    /// The array-to-bytes codec ("serializer").
    #[getter]
    fn serializer(&self) -> PyArrayToBytesCodec {
        PyArrayToBytesCodec::new(self.0.array_to_bytes_codec().clone())
    }
}

impl From<PyCodecChain> for Arc<CodecChain> {
    fn from(chain: PyCodecChain) -> Self {
        chain.0
    }
}

impl From<Arc<CodecChain>> for PyCodecChain {
    fn from(chain: Arc<CodecChain>) -> Self {
        Self(chain)
    }
}
