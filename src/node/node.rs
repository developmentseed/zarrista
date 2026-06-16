//! Opening a node (array or group) at a path, with auto-detection.

use crate::array::PyArray;
use crate::error::not_found;
use crate::group::PyGroup;
use crate::store::Storage;
use pyo3::prelude::*;
use zarrs::array::Array;
use zarrs::group::Group;
use zarrs::node::NodePath;

/// Open the node at `path`, trying an array first, then a group.
///
/// Returns a Python `Array` or `Group`, or raises `NotFoundError` if neither
/// exists at the path.
pub(crate) fn open_node(py: Python<'_>, storage: Storage, path: NodePath) -> PyResult<Py<PyAny>> {
    if let Ok(inner) = Array::open(storage.clone(), path.as_str()) {
        return Ok(Py::new(py, PyArray::new(inner))?.into_any());
    }
    if let Ok(inner) = Group::open(storage.clone(), path.as_str()) {
        return Ok(Py::new(py, PyGroup::new(storage, path.into(), inner))?.into_any());
    }

    Err(not_found(path.as_str()))
}
