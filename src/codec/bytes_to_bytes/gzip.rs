use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pythonize::depythonize;
use zarrs::array::codec::GzipCodec;

use crate::error::ZarristaResult;

pub use sealed::PyGzip;

/// `PyGzip` lives in a private module with a private `()` field, so it can only
/// be constructed via [`PyGzip::new`], enforcing correct submodule instantiation
mod sealed {
    use std::sync::Arc;

    use pyo3::prelude::*;
    use pyo3::PyClassInitializer;
    use zarrs::array::codec::GzipCodec;

    use crate::codec::PyBytesToBytesCodec;

    /// The `gzip` bytes-to-bytes codec.
    ///
    /// A subclass of `BytesToBytesCodec`, so it inherits the codec methods (e.g.
    /// `encode`) while adding `gzip`-specific constructors.
    //
    // See https://pyo3.rs/v0.29.0/class.html#inheritance for docs on subclassing in pyo3
    #[pyclass(module = "zarrista.codec", extends = PyBytesToBytesCodec, frozen, name = "Gzip")]
    pub struct PyGzip(());

    impl PyGzip {
        /// Wrap a [`GzipCodec`] as an initializer for the `PyGzip` subclass: the
        /// codec is stored in the [`PyBytesToBytesCodec`] base, with `PyGzip`
        /// itself carrying no extra state.
        pub(super) fn new(codec: GzipCodec) -> PyClassInitializer<Self> {
            PyClassInitializer::from(PyBytesToBytesCodec::new(Arc::new(codec)))
                .add_subclass(PyGzip(()))
        }
    }
}

#[pymethods]
impl PyGzip {
    /// Create a `gzip` codec.
    ///
    /// `level` is the compression level, an integer from 0 (no compression) to
    /// 9 (most compression).
    #[new]
    fn py_new(level: u32) -> ZarristaResult<PyClassInitializer<Self>> {
        let codec = GzipCodec::new(level).map_err(|_| {
            PyValueError::new_err(format!(
                "invalid gzip compression level {level}; must be between 0 and 9"
            ))
        })?;
        Ok(Self::new(codec))
    }

    /// Create a `gzip` codec from a configuration mapping, e.g. `{"level": 5}`.
    #[staticmethod]
    fn from_config(config: &Bound<'_, PyAny>) -> ZarristaResult<Py<Self>> {
        let codec = GzipCodec::new_with_configuration(&depythonize(config)?)?;
        Ok(Py::new(config.py(), Self::new(codec))?)
    }
}
