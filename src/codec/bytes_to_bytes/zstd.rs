use std::sync::Arc;

use pyo3::prelude::*;
use zarrs::array::codec::ZstdCodec;

use crate::codec::PyBytesToBytesCodec;

/// Create a `zstd` codec.
///
/// `level` is the compression level. When `checksum` is true, a checksum is
/// written to (and verified on decode from) the encoded bytestream.
#[pyfunction]
pub fn zstd(level: i32, checksum: bool) -> PyBytesToBytesCodec {
    PyBytesToBytesCodec::new(Arc::new(ZstdCodec::new(level, checksum)))
}
