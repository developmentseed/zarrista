use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::{Borrowed, FromPyObject};
use zarrs::array::codec::{BloscCodec, BloscCompressionLevel, BloscCompressor, BloscShuffleMode};

use crate::codec::PyBytesToBytesCodec;
use crate::error::ZarristaResult;

/// The `blosc` compressor.
///
/// Extracted from a Python string: one of `"blosclz"`, `"lz4"`, `"lz4hc"`,
/// `"snappy"`, `"zlib"`, or `"zstd"` (case-insensitive).
pub struct PyBloscCompressor(BloscCompressor);

impl FromPyObject<'_, '_> for PyBloscCompressor {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        let name = ob.extract::<PyBackedStr>()?;
        let compressor = match name.to_ascii_lowercase().as_str() {
            "blosclz" => BloscCompressor::BloscLZ,
            "lz4" => BloscCompressor::LZ4,
            "lz4hc" => BloscCompressor::LZ4HC,
            "snappy" => BloscCompressor::Snappy,
            "zlib" => BloscCompressor::Zlib,
            "zstd" => BloscCompressor::Zstd,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown blosc compressor {other:?}; expected one of \
                     'blosclz', 'lz4', 'lz4hc', 'snappy', 'zlib', 'zstd'"
                )))
            }
        };
        Ok(Self(compressor))
    }
}

/// The `blosc` compression level.
///
/// Extracted from a Python integer between 0 and 9 (inclusive). A level of 0
/// disables compression; 1 is fastest and 9 produces the most compression.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PyBloscCompressionLevel(BloscCompressionLevel);

impl FromPyObject<'_, '_> for PyBloscCompressionLevel {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        let compression_level = ob.extract::<u8>()?.try_into().map_err(|_| {
            PyValueError::new_err("blosc compression level must be between 0 and 9")
        })?;
        Ok(Self(compression_level))
    }
}

/// The `blosc` shuffle mode.
///
/// Extracted from a Python string: one of `"noshuffle"`, `"shuffle"`, or
/// `"bitshuffle"` (case-insensitive).
pub struct PyBloscShuffleMode(BloscShuffleMode);

impl FromPyObject<'_, '_> for PyBloscShuffleMode {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        let mode = ob.extract::<PyBackedStr>()?;
        let shuffle = match mode.to_ascii_lowercase().as_str() {
            "noshuffle" => BloscShuffleMode::NoShuffle,
            "shuffle" => BloscShuffleMode::Shuffle,
            "bitshuffle" => BloscShuffleMode::BitShuffle,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown blosc shuffle mode {other:?}; expected one of \
                     'noshuffle', 'shuffle', 'bitshuffle'"
                )))
            }
        };
        Ok(Self(shuffle))
    }
}

/// Create a `blosc` codec from its parameters.
///
/// `typesize` is required (a positive integer) whenever `shuffle_mode` is not
/// `"noshuffle"`. The block size is chosen automatically when `blocksize` is
/// `None` or `0`.
#[pyfunction]
#[pyo3(signature = (
    cname,
    clevel,
    shuffle_mode,
    *,
    blocksize = None,
    typesize = None,
))]
pub fn blosc(
    cname: PyBloscCompressor,
    clevel: PyBloscCompressionLevel,
    shuffle_mode: PyBloscShuffleMode,
    blocksize: Option<usize>,
    typesize: Option<usize>,
) -> ZarristaResult<PyBytesToBytesCodec> {
    let codec = BloscCodec::new(cname.0, clevel.0, blocksize, shuffle_mode.0, typesize)?;
    Ok(PyBytesToBytesCodec::new(Arc::new(codec)))
}
