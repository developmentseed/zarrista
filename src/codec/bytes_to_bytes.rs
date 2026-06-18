use std::num::NonZeroU64;
use std::sync::Arc;

use pyo3::prelude::*;
use zarrs::array::codec::{BitroundCodec, TransposeCodec, TransposeOrder};
use zarrs::array::{ArrayToArrayCodecTraits, BytesToBytesCodecTraits, CodecOptions};

use crate::array_bytes::PyArrayBytes;
use crate::dtype::PyDataType;
use crate::error::ZarristaResult;
use crate::fill_value::PyFillValue;

#[pyfunction]
pub fn blosc(order: Vec<usize>) -> ZarristaResult<PyBytesToBytesCodec> {
    let codec = TransposeCodec::new(TransposeOrder::new(&order)?);
    Ok(PyBytesToBytesCodec(Arc::new(codec)))
}

#[pyclass(module = "zarrista.codec", frozen, name = "BytesToBytesCodec")]
pub struct PyBytesToBytesCodec(Arc<dyn BytesToBytesCodecTraits>);
