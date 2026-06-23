//! Opening a node (array or group) at a path, with auto-detection.

use std::sync::Arc;

use crate::array::{PyArray, PyAsyncArray};
use crate::error::{ZarristaError, ZarristaResult};
use crate::group::{PyAsyncGroup, PyGroup};
use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;
use zarrs::array::Array;
use zarrs::group::Group;
use zarrs::node::{Node, NodeMetadata};
use zarrs::storage::{
    AsyncReadableWritableListableStorageTraits, ReadableWritableListableStorageTraits,
};

/// An opened node: either an array or a group.
pub(crate) struct PyNode {
    node: Node,
    storage: Arc<dyn ReadableWritableListableStorageTraits>,
}

impl PyNode {
    pub fn new(
        node: Node,
        storage: Arc<dyn ReadableWritableListableStorageTraits>,
    ) -> ZarristaResult<Self> {
        Ok(Self { node, storage })
    }
}

impl<'py> IntoPyObject<'py> for PyNode {
    type Target = PyAny;
    type Error = ZarristaError;
    type Output = Bound<'py, Self::Target>;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let storage = self.storage;
        let path = self.node.path().clone();
        let node_metadata = NodeMetadata::from(self.node);
        match node_metadata {
            NodeMetadata::Array(array_metadata) => {
                let array = Array::new_with_metadata(storage, path.as_str(), array_metadata)?;
                Ok(PyArray::new(array).into_bound_py_any(py)?)
            }
            NodeMetadata::Group(group_metadata) => {
                let group = Group::new_with_metadata(storage, path.as_str(), group_metadata)?;
                Ok(PyGroup::new(group).into_bound_py_any(py)?)
            }
        }
    }
}

/// An opened node from an async store: either an array or a group.
pub(crate) struct PyAsyncNode {
    node: Node,
    storage: Arc<dyn AsyncReadableWritableListableStorageTraits>,
}

impl PyAsyncNode {
    pub fn new(
        node: Node,
        storage: Arc<dyn AsyncReadableWritableListableStorageTraits>,
    ) -> ZarristaResult<Self> {
        Ok(Self { node, storage })
    }
}

impl<'py> IntoPyObject<'py> for PyAsyncNode {
    type Target = PyAny;
    type Error = ZarristaError;
    type Output = Bound<'py, Self::Target>;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let storage = self.storage;
        let path = self.node.path().clone();
        let node_metadata = NodeMetadata::from(self.node);
        match node_metadata {
            NodeMetadata::Array(array_metadata) => {
                let array = Array::new_with_metadata(storage, path.as_str(), array_metadata)?;
                Ok(PyAsyncArray::new(Arc::new(array)).into_bound_py_any(py)?)
            }
            NodeMetadata::Group(group_metadata) => {
                let group = Group::new_with_metadata(storage, path.as_str(), group_metadata)?;
                Ok(PyAsyncGroup::new(Arc::new(group)).into_bound_py_any(py)?)
            }
        }
    }
}
