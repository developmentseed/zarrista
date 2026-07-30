use crate::error::ZarristaResult;
use pyo3::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;
use zarrs::filesystem::FilesystemStore;

/// A store backed by a local directory.
#[pyclass(
    module = "zarrista",
    frozen,
    name = "FilesystemStore",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyFilesystemStore {
    pub(crate) storage: Arc<FilesystemStore>,
    path: PathBuf,
}

crate::wasm_send_sync!(PyFilesystemStore);

#[pymethods]
impl PyFilesystemStore {
    /// Open a filesystem store rooted at `path`.
    #[new]
    fn new(path: PathBuf) -> ZarristaResult<Self> {
        let store = FilesystemStore::new(&path)?;
        Ok(Self {
            storage: Arc::new(store),
            path,
        })
    }

    fn __repr__(&self) -> String {
        format!("FilesystemStore({})", self.path.display())
    }
}
