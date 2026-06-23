//! Opening a node (array or group) at a path, with auto-detection.

use std::sync::Arc;

use crate::array::{PyArray, PyAsyncArray};
use crate::error::ZarristaResult;
use crate::group::{PyAsyncGroup, PyGroup};
use pyo3::prelude::*;
use zarrs::node::Node;
use zarrs::storage::{
    AsyncReadableWritableListableStorageTraits, ReadableWritableListableStorageTraits,
};

/// An opened node: either an array or a group.
pub(crate) struct PyNode {
    node: Node,
    storage: Arc<dyn ReadableWritableListableStorageTraits>,
}

impl PyNode {
    pub fn new(node: Node, storage: Arc<dyn ReadableWritableListableStorageTraits>) -> Self {
        Self { node, storage }
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

#[derive(IntoPyObject)]
pub enum PyAsyncArrayOrGroup {
    Array(PyAsyncArray),
    Group(PyAsyncGroup),
}

impl From<PyAsyncArray> for PyAsyncArrayOrGroup {
    fn from(array: PyAsyncArray) -> Self {
        Self::Array(array)
    }
}

impl From<PyAsyncGroup> for PyAsyncArrayOrGroup {
    fn from(group: PyAsyncGroup) -> Self {
        Self::Group(group)
    }
}
