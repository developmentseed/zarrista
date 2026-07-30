//! Opening a node (array or group) at a path, with auto-detection.

use std::sync::Arc;

use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use zarrs::array::Array;
use zarrs::group::Group;
use zarrs::node::{Node, NodeMetadata};

use crate::array::PyArray;
use crate::error::ZarristaError;
use crate::group::PyGroup;
use crate::storage::PySyncStorage;

/// An opened node: either an array or a group.
pub(crate) struct PyNode {
    node: Node,
    storage: PySyncStorage,
}

impl PyNode {
    pub fn new(node: Node, storage: PySyncStorage) -> Self {
        Self { node, storage }
    }
}

// TODO: remove this if we make `Node` a full pyclass
impl<'py> IntoPyObject<'py> for PyNode {
    type Target = PyAny;
    type Error = ZarristaError;
    type Output = Bound<'py, Self::Target>;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let store = self.storage;
        let path = self.node.path().clone();
        let node_metadata = NodeMetadata::from(self.node);
        match node_metadata {
            NodeMetadata::Array(array_metadata) => {
                let array = Array::new_with_metadata(store.inner(), path.as_str(), array_metadata)?;
                Ok(PyArray::new(Arc::new(array), store).into_bound_py_any(py)?)
            }
            NodeMetadata::Group(group_metadata) => {
                let group = Group::new_with_metadata(store.inner(), path.as_str(), group_metadata)?;
                Ok(PyGroup::new(Arc::new(group), store).into_bound_py_any(py)?)
            }
        }
    }
}

#[derive(IntoPyObject)]
pub enum PyArrayOrGroup {
    Array(PyArray),
    Group(PyGroup),
}

impl From<PyArray> for PyArrayOrGroup {
    fn from(array: PyArray) -> Self {
        Self::Array(array)
    }
}

impl From<PyGroup> for PyArrayOrGroup {
    fn from(group: PyGroup) -> Self {
        Self::Group(group)
    }
}
