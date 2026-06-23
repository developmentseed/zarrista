use std::sync::Arc;

use pyo3::prelude::*;
use zarrs::array::codec::Crc32cCodec;

use crate::codec::PyBytesToBytesCodec;

/// Create a `crc32c` codec, which appends a CRC32C checksum to the encoded
/// bytestream.
#[pyfunction]
pub fn crc32c() -> PyBytesToBytesCodec {
    PyBytesToBytesCodec::new(Arc::new(Crc32cCodec::new()))
}
