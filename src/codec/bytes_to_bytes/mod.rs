pub(super) mod blosc;

use std::borrow::Cow;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_bytes::PyBytes;
use zarrs::array::{BytesToBytesCodecTraits, CodecOptions};

use crate::error::ZarristaResult;

#[pyclass(module = "zarrista.codec", subclass, frozen, name = "BytesToBytesCodec")]
pub struct PyBytesToBytesCodec(Arc<dyn BytesToBytesCodecTraits>);

impl PyBytesToBytesCodec {
    pub fn new(codec: Arc<dyn BytesToBytesCodecTraits>) -> Self {
        Self(codec)
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
