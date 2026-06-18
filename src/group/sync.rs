//! The `Group` Python class: attributes and child navigation.

use std::sync::Arc;

use super::last_segment;
use crate::error::ZarristaResult;
use crate::node::{open_node, Node, PyNodePath};
use crate::storage::PySyncStorage;
use pyo3::prelude::*;
use pythonize::pythonize;
use pythonize::Result as PythonizeResult;
use zarrs::group::Group;
use zarrs::node::NodePath;
use zarrs::storage::ReadableWritableListableStorageTraits;

/// A Zarr group.
#[pyclass(module = "zarrista", frozen, name = "Group")]
pub struct PyGroup {
    pub(crate) storage: Arc<dyn ReadableWritableListableStorageTraits>,
    pub(crate) path: NodePath,
    pub(crate) inner: Group<dyn ReadableWritableListableStorageTraits>,
}

impl PyGroup {
    pub(crate) fn new(
        storage: Arc<dyn ReadableWritableListableStorageTraits>,
        path: NodePath,
        inner: Group<dyn ReadableWritableListableStorageTraits>,
    ) -> Self {
        Self {
            storage,
            path,
            inner,
        }
    }
}

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
        let inner = Group::open(store.clone(), path.as_str())?;
        Ok(Self::new(store, path.into(), inner))
    }

    /// The group's user attributes as a dict.
    #[getter]
    fn attrs<'py>(&self, py: Python<'py>) -> PythonizeResult<Bound<'py, PyAny>> {
        pythonize(py, self.inner.attributes())
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
