//! Opening a node (array or group) at a path, with auto-detection.

use std::sync::Arc;

use crate::array::{PyArray, PyAsyncArray};
use crate::error::{ZarrsitaError, ZarrsitaResult};
use crate::group::{PyAsyncGroup, PyGroup};
use pyo3::prelude::*;
use zarrs::array::Array;
use zarrs::group::Group;
use zarrs::node::NodePath;
use zarrs::storage::{AsyncReadableListableStorageTraits, ReadableListableStorageTraits};

/// An opened node: either an array or a group.
#[derive(IntoPyObject)]
// TODO: fix this
#[allow(clippy::large_enum_variant)]
pub(crate) enum Node {
    Array(PyArray),
    Group(PyGroup),
}

/// Open the node at `path`, trying an array first, then a group.
///
/// Returns a [`Node`] (Python `Array` or `Group`), or raises `NotFoundError`
/// if neither exists at the path.
pub(crate) fn open_node(
    storage: Arc<dyn ReadableListableStorageTraits>,
    path: NodePath,
) -> ZarrsitaResult<Node> {
    if let Ok(inner) = Array::open(storage.clone(), path.as_str()) {
        return Ok(Node::Array(PyArray::new(inner)));
    }
    if let Ok(inner) = Group::open(storage.clone(), path.as_str()) {
        return Ok(Node::Group(PyGroup::new(storage, path, inner)));
    }

    Err(ZarrsitaError::not_found(path.as_str()))
}

/// An opened async node: either an array or a group.
#[derive(IntoPyObject)]
pub(crate) enum AsyncNode {
    Array(PyAsyncArray),
    Group(PyAsyncGroup),
}

/// Async variant of [`open_node`], trying an array first, then a group.
///
/// Returns an [`AsyncNode`] (Python `AsyncArray` or `AsyncGroup`), or raises
/// `NotFoundError` if neither exists at the path.
pub(crate) async fn open_node_async(
    storage: Arc<dyn AsyncReadableListableStorageTraits>,
    path: NodePath,
) -> ZarrsitaResult<AsyncNode> {
    if let Ok(inner) = Array::async_open(storage.clone(), path.as_str()).await {
        return Ok(AsyncNode::Array(PyAsyncArray::new(Arc::new(inner))));
    }
    if let Ok(inner) = Group::async_open(storage.clone(), path.as_str()).await {
        return Ok(AsyncNode::Group(PyAsyncGroup::new(
            storage,
            path,
            Arc::new(inner),
        )));
    }

    Err(ZarrsitaError::not_found(path.as_str()))
}
