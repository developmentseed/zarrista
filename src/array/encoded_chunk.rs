use std::borrow::Cow;
use std::num::NonZeroU64;
use std::sync::Arc;

use bytes::Bytes;
pub use pyo3::prelude::*;
use pyo3_bytes::PyBytes;
use zarrs::array::{ArrayToBytesCodecTraits, CodecChain, DataType, FillValue, FromArrayBytes};

use crate::array::PyFillValue;
use crate::codec::{PyCodecChain, PyCodecOptions};
use crate::data::DecodedArray;
use crate::dtype::PyDataType;
use crate::error::ZarristaResult;

#[pyclass(module = "zarrista", frozen, name = "EncodedChunk")]
pub struct PyEncodedChunk {
    bytes: Bytes,
    codecs: Arc<CodecChain>,
    data_type: DataType,
    fill_value: FillValue,
    shape: Vec<NonZeroU64>,
}

impl PyEncodedChunk {
    pub fn new(
        bytes: Bytes,
        codecs: Arc<CodecChain>,
        data_type: DataType,
        fill_value: FillValue,
        shape: Vec<NonZeroU64>,
    ) -> Self {
        Self {
            bytes,
            codecs,
            data_type,
            fill_value,
            shape,
        }
    }
}

#[pymethods]
impl PyEncodedChunk {
    #[getter]
    fn buffer(&self) -> PyBytes {
        PyBytes::new(self.bytes.clone())
    }

    #[getter]
    fn codecs(&self) -> PyCodecChain {
        self.codecs.clone().into()
    }

    #[getter]
    fn data_type(&self) -> PyDataType {
        self.data_type.clone().into()
    }

    #[pyo3(signature = (**codec_options))]
    fn decode(&self, codec_options: Option<PyCodecOptions>) -> ZarristaResult<DecodedArray> {
        let codec_options = codec_options
            .map(|opts| opts.into_inner())
            .unwrap_or_default();

        let bytes = self.codecs.decode(
            Cow::Borrowed(&self.bytes),
            &self.shape,
            &self.data_type,
            &self.fill_value,
            &codec_options,
        )?;
        let shape = self.shape.iter().map(|v| v.get()).collect::<Vec<_>>();
        Ok(DecodedArray::from_array_bytes(
            bytes.into_owned(),
            &shape,
            &self.data_type,
        )?)
    }

    #[getter]
    fn fill_value(&self) -> PyFillValue {
        self.fill_value.clone().into()
    }

    #[getter]
    fn shape(&self) -> Vec<NonZeroU64> {
        self.shape.clone()
    }
}
