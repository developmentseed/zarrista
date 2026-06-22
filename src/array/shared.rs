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

            #[getter]
            fn codecs(&self) -> $crate::codec::PyCodecChain {
                self.inner.codecs().into()
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
            fn metadata<'py>(&self, py: ::pyo3::Python<'py>) -> ::pyo3::Bound<'py, ::pyo3::PyAny> {
                ::pythonize::pythonize(py, self.inner.metadata()).unwrap()
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
