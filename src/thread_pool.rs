use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

#[pyclass(name = "ThreadPool", frozen, module = "zarrista")]
pub(crate) struct PyThreadPool(Arc<ThreadPool>);

#[pymethods]
impl PyThreadPool {
    #[new]
    fn new(num_threads: usize) -> PyResult<Self> {
        let pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|err| {
                PyValueError::new_err(format!("Could not create rayon threadpool. {err}"))
            })?;
        Ok(Self(Arc::new(pool)))
    }

    fn __repr__(&self) -> String {
        format!("ThreadPool(num_threads={})", self.0.current_num_threads())
    }
}

impl PyThreadPool {
    pub fn inner(&self) -> &Arc<ThreadPool> {
        &self.0
    }
}

impl AsRef<ThreadPool> for PyThreadPool {
    fn as_ref(&self) -> &ThreadPool {
        &self.0
    }
}
