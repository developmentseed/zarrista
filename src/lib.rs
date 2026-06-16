//! zarrsita: a small, read-only, zarrita-flavored Python binding to zarrs.

mod array;
mod chunks;
mod codec;
mod data;
mod dtype;
mod error;
mod group;
mod metadata;
mod node;
mod storage;

use pyo3::prelude::*;

use crate::array::{PyArray, PyAsyncArray};
use crate::chunks::PyChunkGrid;
use crate::codec::PyCodecChain;
use crate::data::PyData;
use crate::dtype::PyDataType;
use crate::error::{NotFoundError, ZarrsitaException};
use crate::group::{PyAsyncGroup, PyGroup};
use crate::storage::{PyFilesystemStore, PyMemoryStore};

/// The compiled core of zarrsita, imported as `zarrsita._zarrsita`.
#[pymodule]
fn _zarrsita(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_class::<PyArray>()?;
    m.add_class::<PyAsyncArray>()?;
    m.add_class::<PyAsyncGroup>()?;
    m.add_class::<PyChunkGrid>()?;
    m.add_class::<PyCodecChain>()?;
    m.add_class::<PyData>()?;
    m.add_class::<PyDataType>()?;
    m.add_class::<PyFilesystemStore>()?;
    m.add_class::<PyGroup>()?;
    m.add_class::<PyMemoryStore>()?;

    m.add("ZarrsitaError", m.py().get_type::<ZarrsitaException>())?;
    m.add("NotFoundError", m.py().get_type::<NotFoundError>())?;

    Ok(())
}
