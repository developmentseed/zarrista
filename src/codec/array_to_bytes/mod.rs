//! Array to bytes codecs, or "serializers".

use std::sync::Arc;

use pyo3::prelude::*;
use zarrs::array::ArrayToBytesCodecTraits;

#[pyclass(module = "zarrista.codec", frozen, name = "ArrayToBytesCodec")]
pub struct PyArrayToBytesCodec(Arc<dyn ArrayToBytesCodecTraits>);

impl PyArrayToBytesCodec {
    pub fn new(codec: Arc<dyn ArrayToBytesCodecTraits>) -> Self {
        Self(codec)
    }
}
