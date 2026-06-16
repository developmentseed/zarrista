//! The `Group` Python class: attributes and child navigation.

use std::sync::Arc;

use super::last_segment;
use crate::error::to_py_err;
use crate::node::{open_node, Node, PyNodePath};
use crate::store::extract_storage;
use pyo3::prelude::*;
use pythonize::pythonize;
use pythonize::Result as PythonizeResult;
use zarrs::group::Group;
use zarrs::node::NodePath;
use zarrs::storage::ReadableListableStorageTraits;

/// A read-only Zarr group.
#[pyclass(module = "zarrsita", frozen, name = "Group")]
pub struct PyGroup {
    pub(crate) storage: Arc<dyn ReadableListableStorageTraits>,
    pub(crate) path: NodePath,
    pub(crate) inner: Group<dyn ReadableListableStorageTraits>,
}

impl PyGroup {
    pub(crate) fn new(
        storage: Arc<dyn ReadableListableStorageTraits>,
        path: NodePath,
        inner: Group<dyn ReadableListableStorageTraits>,
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
    fn open(store: &Bound<'_, PyAny>, path: PyNodePath) -> PyResult<Self> {
        let storage = extract_storage(store)?;
        let inner = Group::open(storage.clone(), path.as_str()).map_err(to_py_err)?;
        Ok(Self::new(storage, path.into(), inner))
    }

    /// The group's user attributes as a dict.
    #[getter]
    fn attrs<'py>(&self, py: Python<'py>) -> PythonizeResult<Bound<'py, PyAny>> {
        pythonize(py, self.inner.attributes())
    }

    /// Names of the direct child arrays.
    fn array_keys(&self) -> PyResult<Vec<String>> {
        let paths = self.inner.child_array_paths().map_err(to_py_err)?;
        Ok(paths.iter().map(|p| last_segment(p.as_str())).collect())
    }

    /// Names of the direct child groups.
    fn group_keys(&self) -> PyResult<Vec<String>> {
        let paths = self.inner.child_group_paths().map_err(to_py_err)?;
        Ok(paths.iter().map(|p| last_segment(p.as_str())).collect())
    }

    /// Open a direct child array or group by name.
    fn __getitem__(&self, name: &str) -> PyResult<Node> {
        open_node(self.storage.clone(), self.path.join(name).unwrap())
    }

    fn __repr__(&self) -> String {
        format!("Group(path={:?})", self.path)
    }
}
