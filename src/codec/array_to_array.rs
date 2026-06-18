use std::num::NonZeroU64;
use std::sync::Arc;

use pyo3::prelude::*;
use zarrs::array::codec::{BitroundCodec, TransposeCodec, TransposeOrder};
use zarrs::array::{ArrayToArrayCodecTraits, CodecOptions};

use crate::array_bytes::PyArrayBytes;
use crate::dtype::PyDataType;
use crate::error::ZarristaResult;
use crate::fill_value::PyFillValue;

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

#[pyclass(module = "zarrista", frozen, name = "ArrayToArrayCodec")]
pub struct PyArrayToArrayCodec(Arc<dyn ArrayToArrayCodecTraits>);

#[pymethods]
impl PyArrayToArrayCodec {
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
        bytes: &PyArrayBytes,
        shape: Vec<NonZeroU64>,
        data_type: &PyDataType,
        fill_value: &PyFillValue,
    ) -> ZarristaResult<PyArrayBytes> {
        let encoded = self.0.encode(
            bytes.as_array_bytes()?,
            &shape,
            data_type.inner(),
            fill_value.inner(),
            &CodecOptions::default(),
        )?;
        Ok(PyArrayBytes::from_zarrs(encoded))
    }

    fn decode(
        &self,
        bytes: &PyArrayBytes,
        shape: Vec<NonZeroU64>,
        data_type: &PyDataType,
        fill_value: &PyFillValue,
    ) -> ZarristaResult<PyArrayBytes> {
        let decoded = self.0.decode(
            bytes.as_array_bytes()?,
            &shape,
            data_type.inner(),
            fill_value.inner(),
            &CodecOptions::default(),
        )?;
        Ok(PyArrayBytes::from_zarrs(decoded))
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

    fn __repr__(&self) -> String {
        format!("ArrayToArrayCodec({:?})", self.0)
    }
}
