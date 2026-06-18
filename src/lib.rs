//! zarrista: a small, read-only, zarrita-flavored Python binding to zarrs.

mod array;
mod array_bytes;
mod chunks;
mod codec;
mod data;
mod dtype;
mod error;
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
use crate::data::PyData;
use crate::dtype::PyDataType;
use crate::error::{NotFoundError, ZarristaException};
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
    m.add_class::<PyData>()?;
    m.add_class::<PyDataType>()?;
    m.add_class::<PyFilesystemStore>()?;
    m.add_class::<PyGroup>()?;
    m.add_class::<PyMemoryStore>()?;

    register_codec_module(m)?;

    m.add("ZarristaError", m.py().get_type::<ZarristaException>())?;
    m.add("NotFoundError", m.py().get_type::<NotFoundError>())?;

    Ok(())
}
