use std::sync::Arc;

use super::last_segment;
use crate::array::PyAsyncArray;
use crate::error::{ZarristaError, ZarristaResult};
use crate::group::shared::group_metadata_accessors;
use crate::node::{PyAsyncArrayOrGroup, PyAsyncNode, PyNodePath};
use crate::storage::PyAsyncStorage;
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use zarrs::array::Array;
use zarrs::group::Group;
use zarrs::node::NodeMetadata;
use zarrs::storage::AsyncReadableWritableListableStorageTraits;

/// A Zarr group.
#[derive(Clone)]
#[pyclass(module = "zarrista", frozen, name = "AsyncGroup", from_py_object)]
pub struct PyAsyncGroup {
    pub(crate) inner: Arc<Group<dyn AsyncReadableWritableListableStorageTraits>>,
    store: PyAsyncStorage,
}

impl PyAsyncGroup {
    pub(crate) fn new(
        inner: Arc<Group<dyn AsyncReadableWritableListableStorageTraits>>,
        store: PyAsyncStorage,
    ) -> Self {
        Self { inner, store }
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
        future_into_py(py, async move {
            let inner = Group::async_open(store.inner(), path.as_str())
                .await
                .map_err(ZarristaError::from)?;
            Ok(Self::new(Arc::new(inner), store))
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
        let storage = self.store.clone();
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

    /// Every node under the group, recursively, as `AsyncArray`/`AsyncGroup` objects.
    fn traverse<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let storage = self.store.clone();
        let inner = self.inner.clone();
        future_into_py(py, async move {
            let nodes = inner.async_traverse().await.map_err(ZarristaError::from)?;
            nodes
                .into_iter()
                .map(|(path, metadata)| {
                    let storage = storage.clone();
                    match metadata {
                        NodeMetadata::Array(array_metadata) => {
                            let array = Array::new_with_metadata(
                                storage.inner(),
                                path.as_str(),
                                array_metadata,
                            )?;
                            Ok(PyAsyncArray::new(Arc::new(array), storage.clone()).into())
                        }
                        NodeMetadata::Group(group_metadata) => {
                            let group = Group::new_with_metadata(
                                storage.inner(),
                                path.as_str(),
                                group_metadata,
                            )?;
                            Ok(PyAsyncGroup::new(Arc::new(group), storage.clone()).into())
                        }
                    }
                })
                .collect::<ZarristaResult<Vec<PyAsyncArrayOrGroup>>>()
                .map_err(PyErr::from)
        })
    }

    /// The direct child arrays of the group.
    fn child_arrays<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let store = self.store.clone();
        future_into_py(py, async move {
            let arrays = inner
                .async_child_arrays()
                .await
                .map_err(ZarristaError::from)?;
            Ok(arrays
                .into_iter()
                .map(|array| PyAsyncArray::new(Arc::new(array), store.clone()))
                .collect::<Vec<_>>())
        })
    }

    /// The direct child groups of the group.
    fn child_groups<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let store = self.store.clone();
        future_into_py(py, async move {
            let groups = inner
                .async_child_groups()
                .await
                .map_err(ZarristaError::from)?;
            Ok(groups
                .into_iter()
                .map(|group| PyAsyncGroup::new(Arc::new(group), store.clone()))
                .collect::<Vec<_>>())
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

    #[getter]
    fn store(&self) -> &PyAsyncStorage {
        &self.store
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

    fn __repr__(&self) -> String {
        format!("AsyncGroup(path={:?})", self.inner.path().as_str())
    }
}
