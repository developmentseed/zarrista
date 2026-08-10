//! zarrista: a small, read-only, zarrita-flavored Python binding to zarrs.

#![cfg_attr(not(test), deny(unused_crate_dependencies))]
#![warn(missing_docs)]

mod array;
mod array_bytes;
mod codec;
mod data;
mod dtype;
mod error;
mod exceptions;
mod group;
mod metadata;
mod node;
mod py;
mod repr;
mod storage;
mod thread_pool;
mod wasm;

use pyo3::prelude::*;
// In cargo.toml only to add decompression support to zarrs_zip
use rc_zip as _;

use crate::array::{
    PyArray, PyArrayBuilder, PyChunkGrid, PyChunkKeyEncoding, PyEncodedChunk, PyFillValue,
    PyShardCache,
};
use crate::array_bytes::PyArrayBytes;
use crate::codec::register_codec_module;
use crate::data::{
    PyFixedLengthTensor, PyOptionalFixedLengthTensor, PyOptionalVariableLengthTensor,
    PyVariableLengthTensor,
};
use crate::dtype::PyDataType;
use crate::exceptions::register_exceptions_module;
use crate::group::PyGroup;
use crate::storage::{PyFilesystemStore, PyMemoryStore, PyZipStore};
use crate::thread_pool::PyThreadPool;

/// The compiled core of zarrista, imported as `zarrista._zarrista`.
#[pymodule]
fn _zarrista(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_class::<PyArray>()?;
    m.add_class::<PyArrayBuilder>()?;
    m.add_class::<PyArrayBytes>()?;
    #[cfg(feature = "async")]
    m.add_class::<crate::array::PyAsyncArray>()?;
    #[cfg(feature = "async")]
    m.add_class::<crate::group::PyAsyncGroup>()?;
    #[cfg(feature = "async")]
    m.add_class::<crate::array::PyAsyncShardCache>()?;
    #[cfg(feature = "async")]
    m.add_class::<crate::storage::PyAsyncZipStore>()?;
    m.add_class::<PyChunkGrid>()?;
    m.add_class::<PyChunkKeyEncoding>()?;
    m.add_class::<PyDataType>()?;
    m.add_class::<PyEncodedChunk>()?;
    m.add_class::<PyFilesystemStore>()?;
    m.add_class::<PyFillValue>()?;
    m.add_class::<PyFixedLengthTensor>()?;
    m.add_class::<PyGroup>()?;
    m.add_class::<PyMemoryStore>()?;
    m.add_class::<PyOptionalFixedLengthTensor>()?;
    m.add_class::<PyOptionalVariableLengthTensor>()?;
    m.add_class::<PyShardCache>()?;
    m.add_class::<PyThreadPool>()?;
    m.add_class::<PyVariableLengthTensor>()?;
    m.add_class::<PyZipStore>()?;

    register_codec_module(m)?;
    register_exceptions_module(m)?;

    Ok(())
}
