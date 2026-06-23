//! Array to bytes codecs, or "serializers".

use std::borrow::Cow;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zarrs::array::codec::api::CodecMetadata;
use zarrs::array::{ArrayToBytesCodecTraits, Codec};

use crate::error::ZarristaResult;
use crate::metadata::{PyConfiguration, PyMetadataV3};

#[pyclass(module = "zarrista.codec", frozen, name = "ArrayToBytesCodec")]
pub struct PyArrayToBytesCodec(Arc<dyn ArrayToBytesCodecTraits>);

impl PyArrayToBytesCodec {
    pub fn new(codec: Arc<dyn ArrayToBytesCodecTraits>) -> Self {
        Self(codec)
    }
}

#[pymethods]
impl PyArrayToBytesCodec {
    fn __repr__(&self) -> String {
        format!("ArrayToBytesCodec({:?})", self.0)
    }

    /// Build a codec from its Zarr v3 metadata,
    #[staticmethod]
    fn from_config(metadata: PyMetadataV3) -> ZarristaResult<Self> {
        let codec = Codec::from_metadata(CodecMetadata::V3(metadata.as_ref()))?;
        match codec {
            Codec::ArrayToBytes(c) => Ok(Self::new(c)),
            _ => Err(
                PyValueError::new_err("metadata does not describe an ArrayToBytes codec").into(),
            ),
        }
    }

    /// The codec's Zarr v3 configuration
    #[getter]
    fn config(&self) -> Option<PyConfiguration> {
        self.0
            .configuration_v3(&Default::default())
            .map(|config| config.into())
    }

    /// The codec's Zarr v3 name if it has one.
    #[getter]
    fn name(&self) -> Option<Cow<'static, str>> {
        self.0.name_v3()
    }
}
