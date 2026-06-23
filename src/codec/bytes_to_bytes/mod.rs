//! Bytes to bytes codecs, or "compressors".

pub(super) mod blosc;
pub(super) mod crc32c;
pub(super) mod gzip;
pub(super) mod zstd;

use std::borrow::Cow;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_bytes::PyBytes;
use zarrs::array::codec::api::CodecMetadata;
use zarrs::array::{BytesToBytesCodecTraits, Codec, CodecOptions};

use crate::error::ZarristaResult;
use crate::metadata::{PyConfiguration, PyMetadataV3};

#[pyclass(module = "zarrista.codec", frozen, name = "BytesToBytesCodec")]
pub struct PyBytesToBytesCodec(Arc<dyn BytesToBytesCodecTraits>);

impl PyBytesToBytesCodec {
    pub fn new(codec: Arc<dyn BytesToBytesCodecTraits>) -> Self {
        Self(codec)
    }
}

#[pymethods]
impl PyBytesToBytesCodec {
    fn __repr__(&self) -> String {
        format!("BytesToBytesCodec({:?})", self.0)
    }

    /// Build a codec from its Zarr v3 metadata,
    #[staticmethod]
    fn from_config(metadata: PyMetadataV3) -> ZarristaResult<Self> {
        let codec = Codec::from_metadata(CodecMetadata::V3(metadata.as_ref()))?;
        match codec {
            Codec::BytesToBytes(c) => Ok(Self::new(c)),
            _ => {
                Err(PyValueError::new_err("metadata does not describe a BytesToBytes codec").into())
            }
        }
    }

    /// The codec's Zarr v3 configuration
    #[getter]
    fn config(&self) -> Option<PyConfiguration> {
        self.0
            .configuration_v3(&Default::default())
            .map(|config| config.into())
    }

    fn encode(&self, decoded_value: PyBytes) -> ZarristaResult<PyBytes> {
        let encoded = self.0.encode(
            Cow::Borrowed(decoded_value.as_ref()),
            &CodecOptions::default(),
        )?;
        Ok(PyBytes::new(encoded.into_owned().into()))
    }

    /// The codec's Zarr v3 name if it has one.
    #[getter]
    fn name(&self) -> Option<Cow<'static, str>> {
        self.0.name_v3()
    }
}
