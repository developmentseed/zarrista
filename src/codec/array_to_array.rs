use std::borrow::Cow;
use std::num::NonZeroU64;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zarrs::array::codec::api::CodecMetadata;
use zarrs::array::codec::{BitroundCodec, TransposeCodec, TransposeOrder};
use zarrs::array::{ArrayToArrayCodecTraits, Codec, CodecOptions};

use crate::array::PyFillValue;
use crate::array_bytes::PyArrayBytes;
use crate::dtype::PyDataType;
use crate::error::ZarristaResult;
use crate::metadata::{PyConfiguration, PyMetadataV3};
use crate::repr::named_config_repr;

#[pyfunction]
pub fn transpose(order: Vec<usize>) -> ZarristaResult<PyArrayToArrayCodec> {
    let codec = TransposeCodec::new(TransposeOrder::new(&order)?);
    Ok(PyArrayToArrayCodec(Arc::new(codec)))
}

#[pyfunction]
pub fn bitround(keepbits: u32) -> PyArrayToArrayCodec {
    let codec = BitroundCodec::new(keepbits);
    PyArrayToArrayCodec(Arc::new(codec))
}

#[derive(Debug, Clone)]
#[pyclass(
    module = "zarrista.codec",
    frozen,
    name = "ArrayToArrayCodec",
    from_py_object
)]
pub struct PyArrayToArrayCodec(Arc<dyn ArrayToArrayCodecTraits>);

crate::wasm_send_sync!(PyArrayToArrayCodec);

impl PyArrayToArrayCodec {
    pub fn into_inner(self) -> Arc<dyn ArrayToArrayCodecTraits> {
        self.0
    }

    pub fn new(codec: Arc<dyn ArrayToArrayCodecTraits>) -> Self {
        Self(codec)
    }
}

#[pymethods]
impl PyArrayToArrayCodec {
    fn __repr__(&self, py: Python) -> PyResult<String> {
        named_config_repr(py, "ArrayToArrayCodec", self.0.name_v3(), self.config())
    }

    /// Build a codec from its Zarr v3 metadata,
    #[staticmethod]
    #[pyo3(signature = (metadata, /))]
    fn from_config(metadata: PyMetadataV3) -> ZarristaResult<Self> {
        let codec = Codec::from_metadata(CodecMetadata::V3(metadata.as_ref()))?;
        match codec {
            Codec::ArrayToArray(c) => Ok(Self::new(c)),
            _ => Err(
                PyValueError::new_err("metadata does not describe an ArrayToArray codec").into(),
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

    fn encoded_data_type(&self, decoded_data_type: &PyDataType) -> ZarristaResult<PyDataType> {
        Ok(self.0.encoded_data_type(decoded_data_type.inner())?.into())
    }

    fn encoded_fill_value(
        &self,
        decoded_data_type: &PyDataType,
        decoded_fill_value: &PyFillValue,
    ) -> ZarristaResult<PyFillValue> {
        Ok(self
            .0
            .encoded_fill_value(decoded_data_type.inner(), decoded_fill_value.inner())?
            .into())
    }

    fn encode(
        &self,
        py: Python,
        bytes: &PyArrayBytes,
        shape: Vec<NonZeroU64>,
        data_type: &PyDataType,
        fill_value: &PyFillValue,
    ) -> ZarristaResult<PyArrayBytes> {
        crate::py::detach(py, || {
            let encoded = self.0.encode(
                bytes.as_array_bytes()?,
                &shape,
                data_type.inner(),
                fill_value.inner(),
                &CodecOptions::default(),
            )?;
            Ok(PyArrayBytes::from_zarrs(encoded))
        })
    }

    fn decode(
        &self,
        py: Python,
        bytes: &PyArrayBytes,
        shape: Vec<NonZeroU64>,
        data_type: &PyDataType,
        fill_value: &PyFillValue,
    ) -> ZarristaResult<PyArrayBytes> {
        crate::py::detach(py, || {
            let decoded = self.0.decode(
                bytes.as_array_bytes()?,
                &shape,
                data_type.inner(),
                fill_value.inner(),
                &CodecOptions::default(),
            )?;
            Ok(PyArrayBytes::from_zarrs(decoded))
        })
    }

    fn encoded_shape(&self, decoded_shape: Vec<NonZeroU64>) -> ZarristaResult<Vec<NonZeroU64>> {
        Ok(self.0.encoded_shape(&decoded_shape)?)
    }

    fn decoded_shape(
        &self,
        encoded_shape: Vec<NonZeroU64>,
    ) -> ZarristaResult<Option<Vec<NonZeroU64>>> {
        Ok(self.0.decoded_shape(&encoded_shape)?)
    }

    /// The codec's Zarr v3 name if it has one.
    #[getter]
    fn name(&self) -> Option<Cow<'static, str>> {
        self.0.name_v3()
    }
}
