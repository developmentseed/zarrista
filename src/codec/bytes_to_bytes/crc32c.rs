use pyo3::prelude::*;
use pythonize::depythonize;
use zarrs::array::codec::Crc32cCodec;

use crate::error::ZarristaResult;

pub use sealed::PyCrc32c;

/// `PyCrc32c` lives in a private module with a private `()` field, so it can only
/// be constructed via [`PyCrc32c::new`], enforcing correct submodule instantiation
mod sealed {
    use std::sync::Arc;

    use pyo3::prelude::*;
    use pyo3::PyClassInitializer;
    use zarrs::array::codec::Crc32cCodec;

    use crate::codec::PyBytesToBytesCodec;

    /// The `crc32c` bytes-to-bytes codec.
    ///
    /// A subclass of `BytesToBytesCodec`, so it inherits the codec methods (e.g.
    /// `encode`) while adding `crc32c`-specific constructors.
    //
    // See https://pyo3.rs/v0.29.0/class.html#inheritance for docs on subclassing in pyo3
    #[pyclass(module = "zarrista.codec", extends = PyBytesToBytesCodec, frozen, name = "Crc32c")]
    pub struct PyCrc32c(());

    impl PyCrc32c {
        /// Wrap a [`Crc32cCodec`] as an initializer for the `PyCrc32c` subclass:
        /// the codec is stored in the [`PyBytesToBytesCodec`] base, with
        /// `PyCrc32c` itself carrying no extra state.
        pub(super) fn new(codec: Crc32cCodec) -> PyClassInitializer<Self> {
            PyClassInitializer::from(PyBytesToBytesCodec::new(Arc::new(codec)))
                .add_subclass(PyCrc32c(()))
        }
    }
}

#[pymethods]
impl PyCrc32c {
    /// Create a `crc32c` codec, which appends a CRC32C checksum to the encoded
    /// bytestream.
    #[new]
    fn py_new() -> PyClassInitializer<Self> {
        Self::new(Crc32cCodec::new())
    }

    /// Create a `crc32c` codec from a configuration mapping, e.g. `{}`.
    #[staticmethod]
    fn from_config(config: &Bound<'_, PyAny>) -> ZarristaResult<Py<Self>> {
        let codec = Crc32cCodec::new_with_configuration(&depythonize(config)?);
        Ok(Py::new(config.py(), Self::new(codec))?)
    }
}
