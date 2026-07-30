use pyo3::prelude::*;
use std::sync::Arc;
use zarrs::storage::store::MemoryStore;

/// An in-memory store, primarily useful for testing.
#[pyclass(module = "zarrista", frozen, name = "MemoryStore", skip_from_py_object)]
#[derive(Clone)]
pub struct PyMemoryStore(pub(super) Arc<MemoryStore>);

crate::wasm_send_sync!(PyMemoryStore);

#[pymethods]
impl PyMemoryStore {
    #[new]
    fn new() -> Self {
        Self(Arc::new(MemoryStore::new()))
    }

    fn __repr__(&self) -> String {
        "MemoryStore()".to_string()
    }
}
