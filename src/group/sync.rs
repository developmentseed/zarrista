//! The `Group` Python class: attributes and child navigation.

use std::sync::Arc;

use super::last_segment;
use crate::error::ZarristaResult;
use crate::group::shared::group_metadata_accessors;
use crate::node::{open_node, Node, PyNodePath};
use crate::storage::PySyncStorage;
use pyo3::prelude::*;
use zarrs::group::Group;
use zarrs::node::NodePath;
use zarrs::storage::ReadableWritableListableStorageTraits;

/// A Zarr group.
#[pyclass(module = "zarrista", frozen, name = "Group")]
pub struct PyGroup {
    pub(crate) path: NodePath,
    pub(crate) inner: Group<dyn ReadableWritableListableStorageTraits>,
}

impl PyGroup {
    pub(crate) fn new(
        path: NodePath,
        inner: Group<dyn ReadableWritableListableStorageTraits>,
    ) -> Self {
        Self { path, inner }
    }

    fn storage(&self) -> Arc<dyn ReadableWritableListableStorageTraits> {
        self.inner.storage()
    }
}

// Metadata accessors shared with `PyAsyncGroup`; see `group/shared.rs`.
group_metadata_accessors!(PyGroup);

#[pymethods]
impl PyGroup {
    /// Open the group stored at `path` in `store`.
    #[staticmethod]
    #[pyo3(
        signature = (store, path = PyNodePath::root()),
        text_signature = "(store, path='/')"
    )]
    fn open(store: PySyncStorage, path: PyNodePath) -> ZarristaResult<Self> {
        let store = store.into_inner();
        let inner = Group::open(store, path.as_str())?;
        Ok(Self::new(path.into(), inner))
    }

    /// Names of the direct child arrays.
    fn array_keys(&self) -> ZarristaResult<Vec<String>> {
        let paths = self.inner.child_array_paths()?;
        Ok(paths.iter().map(|p| last_segment(p.as_str())).collect())
    }

    /// Names of the direct child groups.
    fn group_keys(&self) -> ZarristaResult<Vec<String>> {
        let paths = self.inner.child_group_paths()?;
        Ok(paths.iter().map(|p| last_segment(p.as_str())).collect())
    }

    /// Open a direct child array or group by name.
    fn __getitem__(&self, name: &str) -> ZarristaResult<Node> {
        open_node(self.storage.clone(), self.path.join(name)?)
    }

    fn __repr__(&self) -> String {
        format!("Group(path={:?})", self.path)
    }
}
