//! Shared `#[pymethods]` for `PyArray` / `PyAsyncArray`.
//!
//! These accessors only read from `self.inner` and perform no I/O, so they are
//! identical between the sync and async variants. The macro emits a separate
//! `#[pymethods]` block (requires the `multiple-pymethods` pyo3 feature).
macro_rules! array_metadata_accessors {
    ($ty:ty) => {
        #[::pyo3::pymethods]
        impl $ty {
            /// The array's user attributes as a dict.
            #[getter]
            fn attrs<'py>(
                &self,
                py: ::pyo3::Python<'py>,
            ) -> ::pythonize::Result<::pyo3::Bound<'py, ::pyo3::PyAny>> {
                ::pythonize::pythonize(py, self.inner.attributes())
            }

            #[getter]
            fn chunk_grid(&self) -> $crate::chunks::PyChunkGrid {
                self.inner.chunk_grid().clone().into()
            }

            /// The dimension names, if any were specified.
            #[getter]
            fn dimension_names(&self) -> &Option<Vec<Option<String>>> {
                self.inner.dimension_names()
            }

            /// The Zarr data-type
            #[getter]
            fn dtype(&self) -> $crate::dtype::PyDataType {
                self.inner.data_type().clone().into()
            }

            #[getter]
            fn metadata(&self) -> $crate::metadata::PyArrayMetadata {
                self.inner.metadata().clone().into()
            }

            /// The number of dimensions.
            #[getter]
            fn ndim(&self) -> usize {
                self.inner.dimensionality()
            }

            /// The array's path in the store.
            #[getter]
            fn path(&self) -> &str {
                self.inner.path().as_str()
            }

            /// The array shape.
            #[getter]
            fn shape(&self) -> &[u64] {
                self.inner.shape()
            }
        }
    };
}

pub(crate) use array_metadata_accessors;

use pyo3::prelude::*;

use crate::array::PyArray;
use crate::codec::{PyArrayToArrayCodec, PyArrayToBytesCodec, PyBytesToBytesCodec};
use crate::error::ZarristaResult;

#[pymethods]
impl PyArray {
    /// filter codecs
    #[getter]
    fn filters(&self) -> Vec<PyArrayToArrayCodec> {
        let codecs = self.inner.codecs();
        codecs
            .array_to_array_codecs()
            .iter()
            .map(|f| PyArrayToArrayCodec::new(f.clone()))
            .collect()
    }

    #[getter]
    fn serializer(&self) -> PyArrayToBytesCodec {
        let codecs = self.inner.codecs();
        PyArrayToBytesCodec::new(codecs.array_to_bytes_codec().clone())
    }

    #[getter]
    fn compressors(&self) -> Vec<PyBytesToBytesCodec> {
        let codecs = self.inner.codecs();
        codecs
            .bytes_to_bytes_codecs()
            .iter()
            .map(|c| PyBytesToBytesCodec::new(c.clone()))
            .collect()
    }
}
