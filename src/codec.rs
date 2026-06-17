use std::sync::Arc;

use pyo3::prelude::*;
use zarrs::array::CodecChain;

use crate::metadata::PyMetadataV3;

#[pyclass(module = "zarrista", frozen, name = "CodecChain")]
pub struct PyCodecChain(Arc<CodecChain>);

#[pymethods]
impl PyCodecChain {
    #[new]
    fn new(metadatas: Vec<PyMetadataV3>) -> Self {
        let metadatas = metadatas
            .into_iter()
            .map(|m| m.into_inner())
            .collect::<Vec<_>>();
        let codec_chain = CodecChain::from_metadata(&metadatas).unwrap();
        PyCodecChain(Arc::new(codec_chain))
    }

    fn create_metadatas(&self) -> Vec<PyMetadataV3> {
        self.0
            .create_metadatas(&Default::default())
            .into_iter()
            .map(|m| m.into())
            .collect()
    }
}

impl From<Arc<CodecChain>> for PyCodecChain {
    fn from(codec_chain: Arc<CodecChain>) -> Self {
        PyCodecChain(codec_chain)
    }
}
