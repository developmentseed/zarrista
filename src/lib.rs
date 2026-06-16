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
mod store;

use pyo3::prelude::*;

use crate::array::{PyArray, PyAsyncArray};
use crate::chunks::PyChunkGrid;
use crate::codec::PyCodecChain;
use crate::data::PyData;
use crate::dtype::PyDataType;
use crate::error::{NotFoundError, ZarrsitaException};
use crate::group::{PyAsyncGroup, PyGroup};
use crate::store::{FilesystemStore, MemoryStore};

// /// Open a Zarr array or group from a store.
// ///
// /// With `kind` omitted, the node kind is auto-detected (array first, then
// /// group). Pass `kind="array"` or `kind="group"` to require a specific kind.
// #[pyfunction]
// #[pyo3(signature = (store, path = "/", *, kind = None))]
// fn open(
//     py: Python<'_>,
//     store: &Bound<'_, PyAny>,
//     path: &str,
//     kind: Option<&str>,
// ) -> PyResult<PyObject> {
//     let storage = extract_storage(store)?;
//     match kind {
//         None => open_node(py, storage, path),
//         Some("array") => {
//             let inner = ZarrsArray::open(storage, path).map_err(to_py_err)?;
//             Ok(Py::new(py, Array::new(inner))?.into_any())
//         }
//         Some("group") => {
//             let inner = ZarrsGroup::open(storage.clone(), path).map_err(to_py_err)?;
//             Ok(Py::new(py, PyGroup::new(storage, path.to_string(), inner))?.into_any())
//         }
//         Some(other) => Err(PyValueError::new_err(format!(
//             "kind must be 'array', 'group', or None, got {other:?}"
//         ))),
//     }
// }

/// The compiled core of zarrsita, imported as `zarrsita._zarrsita`.
#[pymodule]
fn _zarrsita(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_class::<FilesystemStore>()?;
    m.add_class::<MemoryStore>()?;
    m.add_class::<PyArray>()?;
    m.add_class::<PyAsyncArray>()?;
    m.add_class::<PyAsyncGroup>()?;
    m.add_class::<PyChunkGrid>()?;
    m.add_class::<PyCodecChain>()?;
    m.add_class::<PyData>()?;
    m.add_class::<PyDataType>()?;
    m.add_class::<PyGroup>()?;
    // m.add_function(wrap_pyfunction!(open, m)?)?;

    m.add("ZarrsitaError", m.py().get_type::<ZarrsitaException>())?;
    m.add("NotFoundError", m.py().get_type::<NotFoundError>())?;

    Ok(())
}
