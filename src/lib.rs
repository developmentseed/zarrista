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
mod storage;
mod wasm;

use pyo3::prelude::*;

use crate::array::{PyArray, PyArrayBuilder, PyChunkGrid, PyChunkKeyEncoding, PyFillValue};
use crate::array_bytes::PyArrayBytes;
use crate::codec::register_codec_module;
use crate::data::{PyMaskedTensor, PyMaskedVariableArray, PyTensor, PyVariableArray};
use crate::dtype::PyDataType;
use crate::exceptions::register_exceptions_module;
use crate::group::PyGroup;
use crate::storage::{PyFilesystemStore, PyMemoryStore};

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
    m.add_class::<PyChunkGrid>()?;
    m.add_class::<PyChunkKeyEncoding>()?;
    m.add_class::<PyTensor>()?;
    m.add_class::<PyVariableArray>()?;
    m.add_class::<PyMaskedTensor>()?;
    m.add_class::<PyMaskedVariableArray>()?;
    m.add_class::<PyDataType>()?;
    m.add_class::<PyFillValue>()?;
    m.add_class::<PyFilesystemStore>()?;
    m.add_class::<PyGroup>()?;
    m.add_class::<PyMemoryStore>()?;

    register_codec_module(m)?;
    register_exceptions_module(m)?;

    Ok(())
}
