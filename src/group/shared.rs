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
            fn attrs(&self) -> $crate::metadata::PyAttributes {
                self.inner.attributes().clone().into()
            }

            /// The group's metadata, always exported as Zarr V3.
            #[getter]
            fn metadata(&self) -> $crate::metadata::PyGroupMetadata {
                let options = ::zarrs::group::GroupMetadataOptions::default()
                    .with_metadata_convert_version(::zarrs::config::MetadataConvertVersion::V3);
                self.inner.metadata_opt(&options).into()
            }

            /// The consolidated metadata, if present in the group metadata.
            #[getter]
            fn consolidated_metadata(&self) -> Option<$crate::metadata::PyConsolidatedMetadata> {
                self.inner.consolidated_metadata().map(Into::into)
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
