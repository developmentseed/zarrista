use std::sync::Arc;

use super::last_segment;
use crate::error::to_py_err;
use crate::node::{open_node_async, PyNodePath};
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use pyo3_object_store::AnyObjectStore;
use pythonize::pythonize;
use pythonize::Result as PythonizeResult;
use zarrs::group::Group;
use zarrs::node::NodePath;
use zarrs::storage::AsyncReadableListableStorageTraits;
use zarrs_object_store::AsyncObjectStore;

/// A read-only Zarr group.
#[pyclass(module = "zarrsita", frozen, name = "AsyncGroup")]
pub struct PyAsyncGroup {
    pub(crate) storage: Arc<dyn AsyncReadableListableStorageTraits>,
    pub(crate) path: NodePath,
    pub(crate) inner: Arc<Group<dyn AsyncReadableListableStorageTraits>>,
}

impl PyAsyncGroup {
    pub(crate) fn new(
        storage: Arc<dyn AsyncReadableListableStorageTraits>,
        path: NodePath,
        inner: Arc<Group<dyn AsyncReadableListableStorageTraits>>,
    ) -> Self {
        Self {
            storage,
            path,
            inner,
        }
    }
}

#[pymethods]
impl PyAsyncGroup {
    /// Open the group stored at `path` in `store`.
    #[staticmethod]
    #[pyo3(
        signature = (store, path = PyNodePath::root()),
        text_signature = "(store, path='/')"
    )]
    fn open_async<'py>(
        py: Python<'py>,
        store: AnyObjectStore,
        path: PyNodePath,
    ) -> PyResult<Bound<'py, PyAny>> {
        let storage: Arc<dyn AsyncReadableListableStorageTraits> =
            Arc::new(AsyncObjectStore::new(store.into_dyn()));
        future_into_py(py, async move {
            let inner = Group::async_open(storage.clone(), path.as_str())
                .await
                .map_err(to_py_err)?;
            Ok(Self::new(storage, path.into(), Arc::new(inner)))
        })
    }

    /// The group's user attributes as a dict.
    #[getter]
    fn attrs<'py>(&self, py: Python<'py>) -> PythonizeResult<Bound<'py, PyAny>> {
        pythonize(py, self.inner.attributes())
    }

    /// Names of the direct child arrays.
    fn array_keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        future_into_py(py, async move {
            let paths = inner.async_child_array_paths().await.unwrap();
            Ok(paths
                .iter()
                .map(|p| last_segment(p.as_str()))
                .collect::<Vec<_>>())
        })
    }

    /// Names of the direct child groups.
    fn group_keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        future_into_py(py, async move {
            let paths = inner.async_child_group_paths().await.unwrap();
            Ok(paths
                .iter()
                .map(|p| last_segment(p.as_str()))
                .collect::<Vec<_>>())
        })
    }

    /// Open a direct child array or group by name.
    fn open_child_async<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        let storage = self.storage.clone();
        let path = self.path.join(name).map_err(to_py_err)?;
        future_into_py(py, async move { open_node_async(storage, path).await })
    }

    fn __repr__(&self) -> String {
        format!("AsyncGroup(path={:?})", self.path)
    }
}
