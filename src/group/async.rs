use std::sync::Arc;

use super::last_segment;
use crate::error::ZarristaError;
use crate::group::shared::group_metadata_accessors;
use crate::node::{open_node_async, PyNodePath};
use crate::storage::PyAsyncStorage;
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use zarrs::group::Group;
use zarrs::node::NodePath;
use zarrs::storage::AsyncReadableWritableListableStorageTraits;

/// A Zarr group.
#[derive(Clone)]
#[pyclass(module = "zarrista", frozen, name = "AsyncGroup", from_py_object)]
pub struct PyAsyncGroup {
    pub(crate) inner: Arc<Group<dyn AsyncReadableWritableListableStorageTraits>>,
}

impl PyAsyncGroup {
    pub(crate) fn new(
        path: NodePath,
        inner: Arc<Group<dyn AsyncReadableWritableListableStorageTraits>>,
    ) -> Self {
        Self { path, inner }
    }

    fn storage(&self) -> Arc<dyn AsyncReadableWritableListableStorageTraits> {
        self.inner.storage()
    }
}

// Metadata accessors shared with `PyGroup`; see `group/shared.rs`.
group_metadata_accessors!(PyAsyncGroup);

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
        store: PyAsyncStorage,
        path: PyNodePath,
    ) -> PyResult<Bound<'py, PyAny>> {
        let storage = store.into_inner();
        future_into_py(py, async move {
            let inner = Group::async_open(storage.clone(), path.as_str())
                .await
                .map_err(ZarristaError::from)?;
            Ok(Self::new(path.into(), Arc::new(inner)))
        })
    }

    /// Names of the direct child arrays.
    fn array_keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        future_into_py(py, async move {
            let paths = inner
                .async_child_array_paths()
                .await
                .map_err(ZarristaError::from)?;
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
            let paths = inner
                .async_child_group_paths()
                .await
                .map_err(ZarristaError::from)?;
            Ok(paths
                .iter()
                .map(|p| last_segment(p.as_str()))
                .collect::<Vec<_>>())
        })
    }

    /// Open a direct child array or group by name.
    fn open_child_async<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        let storage = self.storage();
        let path = self.path.join(name).map_err(ZarristaError::from)?;
        future_into_py(py, async move {
            open_node_async(storage, path).await.map_err(PyErr::from)
        })
    }

    fn __repr__(&self) -> String {
        format!("AsyncGroup(path={:?})", self.path)
    }
}
