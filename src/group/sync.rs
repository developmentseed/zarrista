//! The `Group` Python class: attributes and child navigation.

use std::sync::Arc;

use super::last_segment;
use crate::array::PyArray;
use crate::error::ZarristaResult;
use crate::group::shared::group_metadata_accessors;
use crate::node::{PyArrayOrGroup, PyNode, PyNodePath};
use crate::storage::PySyncStorage;
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use zarrs::array::Array;
use zarrs::group::Group;
use zarrs::node::NodeMetadata;
use zarrs::storage::ReadableWritableListableStorageTraits;

/// A Zarr group.
#[pyclass(module = "zarrista", frozen, name = "Group")]
pub struct PyGroup {
    pub(crate) inner: Arc<Group<dyn ReadableWritableListableStorageTraits>>,
}

impl PyGroup {
    pub(crate) fn new(inner: Arc<Group<dyn ReadableWritableListableStorageTraits>>) -> Self {
        Self { inner }
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
        Ok(Self::new(Arc::new(inner)))
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
    fn __getitem__(&self, name: PyBackedStr) -> ZarristaResult<PyNode> {
        self.child(name)
    }

    fn child(&self, name: PyBackedStr) -> ZarristaResult<PyNode> {
        let children = self.inner.children(false)?;
        let selected_child = children
            .into_iter()
            .find(|child| child.name().as_str() == name.as_str())
            .ok_or(PyKeyError::new_err(format!("child {name} not found")))?;
        Ok(PyNode::new(selected_child, self.storage()))
    }

    /// Every node under the group, recursively, as `(path, metadata)` pairs.
    fn traverse(&self) -> ZarristaResult<Vec<PyArrayOrGroup>> {
        self.inner
            .traverse()?
            .into_iter()
            .map(|(path, metadata)| {
                let storage = self.storage();
                match metadata {
                    NodeMetadata::Array(array_metadata) => {
                        let array =
                            Array::new_with_metadata(storage, path.as_str(), array_metadata)?;
                        Ok(PyArray::new(Arc::new(array)).into())
                    }
                    NodeMetadata::Group(group_metadata) => {
                        let group =
                            Group::new_with_metadata(storage, path.as_str(), group_metadata)?;
                        Ok(PyGroup::new(Arc::new(group)).into())
                    }
                }
            })
            .collect()
    }

    /// The direct child arrays of the group.
    fn child_arrays(&self) -> ZarristaResult<Vec<PyArray>> {
        Ok(self
            .inner
            .child_arrays()?
            .into_iter()
            .map(|array| array.into())
            .collect())
    }

    /// The direct child groups of the group.
    fn child_groups(&self) -> ZarristaResult<Vec<PyGroup>> {
        Ok(self
            .inner
            .child_groups()?
            .into_iter()
            .map(Self::from)
            .collect())
    }

    /// The full paths of the group's direct children.
    fn child_paths(&self) -> ZarristaResult<Vec<PyNodePath>> {
        Ok(self
            .inner
            .child_paths()?
            .into_iter()
            .map(|p| p.into())
            .collect())
    }

    /// The full paths of the group's direct child arrays.
    fn child_array_paths(&self) -> ZarristaResult<Vec<PyNodePath>> {
        Ok(self
            .inner
            .child_array_paths()?
            .into_iter()
            .map(|p| p.into())
            .collect())
    }

    /// The full paths of the group's direct child groups.
    fn child_group_paths(&self) -> ZarristaResult<Vec<PyNodePath>> {
        Ok(self
            .inner
            .child_group_paths()?
            .into_iter()
            .map(|p| p.into())
            .collect())
    }

    /// Write the group metadata to the store.
    fn store_metadata(&self) -> ZarristaResult<()> {
        self.inner.store_metadata()?;
        Ok(())
    }

    /// Erase the group metadata from the store. Succeeds if it does not exist.
    fn erase_metadata(&self) -> ZarristaResult<()> {
        self.inner.erase_metadata()?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!("Group(path={:?})", self.inner.path().as_str())
    }
}

impl From<Group<dyn ReadableWritableListableStorageTraits>> for PyGroup {
    fn from(inner: Group<dyn ReadableWritableListableStorageTraits>) -> Self {
        Self::new(Arc::new(inner))
    }
}

impl From<Arc<Group<dyn ReadableWritableListableStorageTraits>>> for PyGroup {
    fn from(inner: Arc<Group<dyn ReadableWritableListableStorageTraits>>) -> Self {
        Self::new(inner)
    }
}
