use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyMapping, PyMappingMethods};
use pyo3::{Borrowed, FromPyObject};
use zarrs::array::CodecOptions;

/// Per-operation [`CodecOptions`], extractable from any mapping
#[derive(Debug, Clone, Default)]
pub struct PyCodecOptions(CodecOptions);

impl PyCodecOptions {
    pub fn into_inner(self) -> CodecOptions {
        self.0
    }
}

impl AsRef<CodecOptions> for PyCodecOptions {
    fn as_ref(&self) -> &CodecOptions {
        &self.0
    }
}

impl From<PyCodecOptions> for CodecOptions {
    fn from(options: PyCodecOptions) -> Self {
        options.0
    }
}

impl FromPyObject<'_, '_> for PyCodecOptions {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        let mut options = CodecOptions::default();
        for item in obj.cast::<PyMapping>()?.items()?.iter() {
            let (key, value) = item.extract::<(PyBackedStr, Bound<'_, PyAny>)>()?;
            match &*key {
                "validate_checksums" => {
                    options.set_validate_checksums(value.extract()?);
                }
                "store_empty_chunks" => {
                    options.set_store_empty_chunks(value.extract()?);
                }
                "concurrent_target" => {
                    options.set_concurrent_target(value.extract()?);
                }
                "chunk_concurrent_minimum" => {
                    options.set_chunk_concurrent_minimum(value.extract()?);
                }
                "experimental_partial_encoding" => {
                    options.set_experimental_partial_encoding(value.extract()?);
                }
                other => {
                    return Err(PyTypeError::new_err(format!(
                        "unexpected keyword argument {other:?}"
                    )));
                }
            }
        }
        Ok(Self(options))
    }
}
