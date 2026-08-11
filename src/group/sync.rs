//! The `Group` Python class: attributes and child navigation.

use std::sync::Arc;

use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use zarrs::array::Array;
use zarrs::group::Group;
use zarrs::node::NodeMetadata;
use zarrs::storage::ReadableWritableListableStorageTraits;

use super::last_segment;
use crate::array::PyArray;
use crate::error::ZarristaResult;
use crate::group::shared::group_metadata_accessors;
use crate::node::{PyArrayOrGroup, PyNode, PyNodePath};
use crate::storage::PySyncStorage;

/// A Zarr group.
#[pyclass(module = "zarrista", frozen, name = "Group")]
pub struct PyGroup {
    pub(crate) inner: Arc<Group<dyn ReadableWritableListableStorageTraits>>,
    store: PySyncStorage,
}

crate::wasm_send_sync!(PyGroup);

impl PyGroup {
    pub(crate) fn new(
        inner: Arc<Group<dyn ReadableWritableListableStorageTraits>>,
        store: PySyncStorage,
    ) -> Self {
        Self { inner, store }
    }

    /// The `zarrs` storage behind the group, as opposed to the Python store wrapper
    /// that the `storage` getter returns.
    fn inner_storage(&self) -> Arc<dyn ReadableWritableListableStorageTraits> {
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
    fn open(py: Python, store: PySyncStorage, path: PyNodePath) -> ZarristaResult<Self> {
        let inner = crate::py::detach(py, || Group::open(store.inner(), path.as_str()))?;
        Ok(Self::new(Arc::new(inner), store))
    }

    /// Names of the direct child arrays.
    fn array_keys(&self, py: Python) -> ZarristaResult<Vec<String>> {
        crate::py::detach(py, || {
            let paths = self.inner.child_array_paths()?;
            Ok(paths.iter().map(|p| last_segment(p.as_str())).collect())
        })
    }

    /// Names of the direct child groups.
    fn group_keys(&self, py: Python) -> ZarristaResult<Vec<String>> {
        crate::py::detach(py, || {
            let paths = self.inner.child_group_paths()?;
            Ok(paths.iter().map(|p| last_segment(p.as_str())).collect())
        })
    }

    /// Open a direct child array or group by name.
    fn __getitem__(&self, py: Python, name: PyBackedStr) -> ZarristaResult<PyNode> {
        self.child(py, name)
    }

    #[pyo3(signature = (name, /))]
    fn child(&self, py: Python, name: PyBackedStr) -> ZarristaResult<PyNode> {
        crate::py::detach(py, || {
            let children = self.inner.children(false)?;
            let selected_child = children
                .into_iter()
                .find(|child| child.name().as_str() == name)
                .ok_or(PyKeyError::new_err(format!("child {name} not found")))?;
            Ok(PyNode::new(selected_child, self.store.clone()))
        })
    }

    /// Every node under the group, recursively, as `Array`/`Group` objects.
    fn traverse(&self, py: Python) -> ZarristaResult<Vec<PyArrayOrGroup>> {
        crate::py::detach(py, || {
            self.inner
                .traverse()?
                .into_iter()
                .map(|(path, metadata)| {
                    let storage = self.inner_storage();
                    match metadata {
                        NodeMetadata::Array(array_metadata) => {
                            let array =
                                Array::new_with_metadata(storage, path.as_str(), array_metadata)?;
                            Ok(PyArray::new(Arc::new(array), self.store.clone()).into())
                        }
                        NodeMetadata::Group(group_metadata) => {
                            let group =
                                Group::new_with_metadata(storage, path.as_str(), group_metadata)?;
                            Ok(PyGroup::new(Arc::new(group), self.store.clone()).into())
                        }
                    }
                })
                .collect()
        })
    }

    /// The direct child arrays of the group.
    fn child_arrays(&self, py: Python) -> ZarristaResult<Vec<PyArray>> {
        crate::py::detach(py, || {
            Ok(self
                .inner
                .child_arrays()?
                .into_iter()
                .map(|array| PyArray::new(Arc::new(array), self.store.clone()))
                .collect())
        })
    }

    /// The direct child groups of the group.
    fn child_groups(&self, py: Python) -> ZarristaResult<Vec<PyGroup>> {
        crate::py::detach(py, || {
            Ok(self
                .inner
                .child_groups()?
                .into_iter()
                .map(|group| PyGroup::new(Arc::new(group), self.store.clone()))
                .collect())
        })
    }

    /// The full paths of the group's direct children.
    fn child_paths(&self, py: Python) -> ZarristaResult<Vec<PyNodePath>> {
        crate::py::detach(py, || {
            Ok(self
                .inner
                .child_paths()?
                .into_iter()
                .map(|p| p.into())
                .collect())
        })
    }

    /// The full paths of the group's direct child arrays.
    fn child_array_paths(&self, py: Python) -> ZarristaResult<Vec<PyNodePath>> {
        crate::py::detach(py, || {
            Ok(self
                .inner
                .child_array_paths()?
                .into_iter()
                .map(|p| p.into())
                .collect())
        })
    }

    /// The full paths of the group's direct child groups.
    fn child_group_paths(&self, py: Python) -> ZarristaResult<Vec<PyNodePath>> {
        crate::py::detach(py, || {
            Ok(self
                .inner
                .child_group_paths()?
                .into_iter()
                .map(|p| p.into())
                .collect())
        })
    }

    /// Erase the group metadata from the store. Succeeds if it does not exist.
    fn erase_metadata(&self, py: Python) -> ZarristaResult<()> {
        crate::py::detach(py, || {
            self.inner.erase_metadata()?;
            Ok(())
        })
    }

    #[getter]
    fn storage(&self) -> PySyncStorage {
        self.store.clone()
    }

    /// Write the group metadata to the store.
    fn store_metadata(&self, py: Python) -> ZarristaResult<()> {
        crate::py::detach(py, || {
            self.inner.store_metadata()?;
            Ok(())
        })
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        let path = PyNodePath::from(self.inner.path().clone());
        Ok(format!("Group(path={})", path.repr(py)?))
    }
}
