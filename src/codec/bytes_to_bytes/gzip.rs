use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zarrs::array::codec::GzipCodec;

use crate::codec::PyBytesToBytesCodec;
use crate::error::ZarristaResult;

/// Create a `gzip` codec.
///
/// `level` is the compression level, an integer from 0 (no compression) to 9
/// (most compression).
#[pyfunction]
pub fn gzip(level: u32) -> ZarristaResult<PyBytesToBytesCodec> {
    let codec = GzipCodec::new(level).map_err(|_| {
        PyValueError::new_err(format!(
            "invalid gzip compression level {level}; must be between 0 and 9"
        ))
    })?;
    Ok(PyBytesToBytesCodec::new(Arc::new(codec)))
}
