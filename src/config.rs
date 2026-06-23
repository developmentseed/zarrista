use pyo3::prelude::*;
use zarrs::config::{global_config, global_config_mut};

#[pyclass(module = "zarrista", name = "Config")]
pub struct PyConfig;

#[pymethods]
impl PyConfig {
    #[getter]
    fn chunk_concurrent_minimum(&self) -> usize {
        let config = global_config();
        config.chunk_concurrent_minimum()
    }

    #[setter]
    fn set_validate_checksums(&mut self, value: bool) {
        let mut config = global_config_mut();
        config.set_validate_checksums(value);
    }

    #[getter]
    fn validate_checksums(&self) -> bool {
        let config = global_config();
        config.validate_checksums()
    }
}
