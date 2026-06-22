pub(super) mod blosc;
pub(super) mod crc32c;
pub(super) mod gzip;
pub(super) mod zstd;

use std::borrow::Cow;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_bytes::PyBytes;
use zarrs::array::{BytesToBytesCodecTraits, CodecOptions};

use crate::error::ZarristaResult;

#[derive(Debug, Clone)]
#[pyclass(
    module = "zarrista.codec",
    subclass,
    frozen,
    name = "BytesToBytesCodec",
    from_py_object
)]
pub struct PyBytesToBytesCodec(Arc<dyn BytesToBytesCodecTraits>);

impl PyBytesToBytesCodec {
    pub fn new(codec: Arc<dyn BytesToBytesCodecTraits>) -> Self {
        Self(codec)
    }

    pub fn into_inner(self) -> Arc<dyn BytesToBytesCodecTraits> {
        self.0
    }
}

#[pymethods]
impl PyBytesToBytesCodec {
    fn encode(&self, decoded_value: PyBytes) -> ZarristaResult<PyBytes> {
        let encoded = self.0.encode(
            Cow::Borrowed(decoded_value.as_ref()),
            &CodecOptions::default(),
        )?;
        Ok(PyBytes::new(encoded.into_owned().into()))
    }
}
