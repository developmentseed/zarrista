//! Shared `#[pymethods]` for `PyGroup` / `PyAsyncGroup`.
//!
//! These accessors only read from `self.inner` and perform no I/O, so they are
//! identical between the sync and async variants. The macro emits a separate
//! `#[pymethods]` block (requires the `multiple-pymethods` pyo3 feature).
macro_rules! group_metadata_accessors {
    ($ty:ty) => {
        #[::pyo3::pymethods]
        impl $ty {
            /// The group's user attributes as a dict.
            #[getter]
            fn attrs<'py>(
                &self,
                py: ::pyo3::Python<'py>,
            ) -> ::pythonize::Result<::pyo3::Bound<'py, ::pyo3::PyAny>> {
                ::pythonize::pythonize(py, self.inner.attributes())
            }

            #[getter]
            fn metadata(&self) -> $crate::metadata::PyGroupMetadata {
                self.inner.metadata().clone().into()
            }

            /// The group's path in the store.
            #[getter]
            fn path(&self) -> &str {
                self.inner.path().as_str()
            }
        }
    };
}

pub(crate) use group_metadata_accessors;
