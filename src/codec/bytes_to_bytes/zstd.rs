use pyo3::prelude::*;
use pythonize::depythonize;
use zarrs::array::codec::ZstdCodec;

use crate::error::ZarristaResult;

pub use sealed::PyZstd;

/// `PyZstd` lives in a private module with a private `()` field, so it can only
/// be constructed via [`PyZstd::new`], enforcing correct submodule instantiation
mod sealed {
    use std::sync::Arc;

    use pyo3::prelude::*;
    use pyo3::PyClassInitializer;
    use zarrs::array::codec::ZstdCodec;

    use crate::codec::PyBytesToBytesCodec;

    /// The `zstd` bytes-to-bytes codec.
    ///
    /// A subclass of `BytesToBytesCodec`, so it inherits the codec methods (e.g.
    /// `encode`) while adding `zstd`-specific constructors.
    //
    // See https://pyo3.rs/v0.29.0/class.html#inheritance for docs on subclassing in pyo3
    #[pyclass(module = "zarrista.codec", extends = PyBytesToBytesCodec, frozen, name = "Zstd")]
    pub struct PyZstd(());

    impl PyZstd {
        /// Wrap a [`ZstdCodec`] as an initializer for the `PyZstd` subclass: the
        /// codec is stored in the [`PyBytesToBytesCodec`] base, with `PyZstd`
        /// itself carrying no extra state.
        pub(super) fn new(codec: ZstdCodec) -> PyClassInitializer<Self> {
            PyClassInitializer::from(PyBytesToBytesCodec::new(Arc::new(codec)))
                .add_subclass(PyZstd(()))
        }
    }
}

#[pymethods]
impl PyZstd {
    /// Create a `zstd` codec.
    ///
    /// `level` is the compression level. When `checksum` is true, a checksum is
    /// written to (and verified on decode from) the encoded bytestream.
    #[new]
    fn py_new(level: i32, checksum: bool) -> PyClassInitializer<Self> {
        Self::new(ZstdCodec::new(level, checksum))
    }

    /// Create a `zstd` codec from a configuration mapping, e.g.
    /// `{"level": 5, "checksum": false}`.
    #[staticmethod]
    fn from_config(config: &Bound<'_, PyAny>) -> ZarristaResult<Py<Self>> {
        let codec = ZstdCodec::new_with_configuration(&depythonize(config)?)?;
        Ok(Py::new(config.py(), Self::new(codec))?)
    }
}
