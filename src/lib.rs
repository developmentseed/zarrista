//! zarrista: a small, read-only, zarrita-flavored Python binding to zarrs.

mod array;
mod array_bytes;
mod chunks;
mod codec;
mod decoded_array;
mod dtype;
mod error;
mod exceptions;
mod fill_value;
mod group;
mod metadata;
mod node;
mod storage;

use pyo3::prelude::*;

use crate::array::{PyArray, PyAsyncArray};
use crate::array_bytes::PyArrayBytes;
use crate::chunks::PyChunkGrid;
use crate::codec::register_codec_module;
use crate::decoded_array::{PyMaskedTensor, PyMaskedVariableArray, PyTensor, PyVariableArray};
use crate::dtype::PyDataType;
use crate::exceptions::register_exceptions_module;
use crate::fill_value::PyFillValue;
use crate::group::{PyAsyncGroup, PyGroup};
use crate::storage::{PyFilesystemStore, PyMemoryStore};

/// The compiled core of zarrista, imported as `zarrista._zarrista`.
#[pymodule]
fn _zarrista(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_class::<PyArray>()?;
    m.add_class::<PyArrayBytes>()?;
    m.add_class::<PyAsyncArray>()?;
    m.add_class::<PyAsyncGroup>()?;
    m.add_class::<PyChunkGrid>()?;
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
