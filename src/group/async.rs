use std::sync::Arc;

use super::last_segment;
use crate::array::PyAsyncArray;
use crate::error::{ZarristaError, ZarristaResult};
use crate::group::shared::group_metadata_accessors;
use crate::node::{flatten_nodes, PyAsyncNode, PyNodePath};
use crate::storage::PyAsyncStorage;
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use zarrs::group::Group;
use zarrs::storage::AsyncReadableWritableListableStorageTraits;

/// A Zarr group.
#[derive(Clone)]
#[pyclass(module = "zarrista", frozen, name = "AsyncGroup", from_py_object)]
pub struct PyAsyncGroup {
    pub(crate) inner: Arc<Group<dyn AsyncReadableWritableListableStorageTraits>>,
}

impl PyAsyncGroup {
    pub(crate) fn new(inner: Arc<Group<dyn AsyncReadableWritableListableStorageTraits>>) -> Self {
        Self { inner }
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
            Ok(Self::new(Arc::new(inner)))
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
    fn open_child_async<'py>(&self, py: Python<'py>, name: String) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let storage = self.storage();
        future_into_py(py, async move {
            let children = inner
                .async_children(false)
                .await
                .map_err(ZarristaError::from)?;
            let selected_child = children
                .into_iter()
                .find(|child| child.name().as_str() == name.as_str())
                .ok_or_else(|| PyKeyError::new_err(format!("child {name} not found")))?;
            PyAsyncNode::new(selected_child, storage).map_err(PyErr::from)
        })
    }

    /// The direct children of the group as `AsyncArray` and `AsyncGroup` objects.
    ///
    /// If `recursive` is true, descendants are included as well (flattened).
    #[pyo3(signature = (recursive = false))]
    fn children<'py>(&self, py: Python<'py>, recursive: bool) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let storage = self.storage();
        future_into_py(py, async move {
            let nodes = inner
                .async_children(recursive)
                .await
                .map_err(ZarristaError::from)?;
            flatten_nodes(nodes)
                .into_iter()
                .map(|node| PyAsyncNode::new(node, storage.clone()))
                .collect::<ZarristaResult<Vec<_>>>()
                .map_err(PyErr::from)
        })
    }

    /// The direct child arrays of the group.
    fn child_arrays<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        future_into_py(py, async move {
            let arrays = inner
                .async_child_arrays()
                .await
                .map_err(ZarristaError::from)?;
            Ok(arrays
                .into_iter()
                .map(|array| PyAsyncArray::new(Arc::new(array)))
                .collect::<Vec<_>>())
        })
    }

    /// The direct child groups of the group.
    fn child_groups<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        future_into_py(py, async move {
            let groups = inner
                .async_child_groups()
                .await
                .map_err(ZarristaError::from)?;
            Ok(groups.into_iter().map(Self::from).collect::<Vec<_>>())
        })
    }

    /// The full paths of the group's direct children.
    fn child_paths<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        future_into_py(py, async move {
            let paths = inner
                .async_child_paths()
                .await
                .map_err(ZarristaError::from)?;
            Ok(paths.into_iter().map(PyNodePath::from).collect::<Vec<_>>())
        })
    }

    /// The full paths of the group's direct child arrays.
    fn child_array_paths<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        future_into_py(py, async move {
            let paths = inner
                .async_child_array_paths()
                .await
                .map_err(ZarristaError::from)?;
            Ok(paths.into_iter().map(PyNodePath::from).collect::<Vec<_>>())
        })
    }

    /// The full paths of the group's direct child groups.
    fn child_group_paths<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        future_into_py(py, async move {
            let paths = inner
                .async_child_group_paths()
                .await
                .map_err(ZarristaError::from)?;
            Ok(paths.into_iter().map(PyNodePath::from).collect::<Vec<_>>())
        })
    }

    /// Write the group metadata to the store.
    fn store_metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        future_into_py(py, async move {
            inner
                .async_store_metadata()
                .await
                .map_err(ZarristaError::from)?;
            Ok(())
        })
    }

    /// Erase the group metadata from the store. Succeeds if it does not exist.
    fn erase_metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        future_into_py(py, async move {
            inner
                .async_erase_metadata()
                .await
                .map_err(ZarristaError::from)?;
            Ok(())
        })
    }

    fn __repr__(&self) -> String {
        format!("AsyncGroup(path={:?})", self.inner.path().as_str())
    }
}

impl From<Group<dyn AsyncReadableWritableListableStorageTraits>> for PyAsyncGroup {
    fn from(inner: Group<dyn AsyncReadableWritableListableStorageTraits>) -> Self {
        Self::new(Arc::new(inner))
    }
}
