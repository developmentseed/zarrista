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

            /// Return a new group reference with `attrs`, leaving this one unchanged.
            fn with_attrs(
                &self,
                attrs: $crate::metadata::PyAttributes,
            ) -> $crate::error::ZarristaResult<Self> {
                // Workaround for missing Clone
                // Clone added in
                // https://github.com/zarrs/zarrs/pull/441
                // and available in zarrs 0.24
                let mut updated = ::zarrs::group::Group::new_with_metadata(
                    self.inner.storage(),
                    self.inner.path().as_str(),
                    self.inner.metadata().clone(),
                )?;
                *updated.attributes_mut() = attrs.into_inner();
                Ok(Self::new(
                    ::std::sync::Arc::new(updated),
                    self.store.clone(),
                ))
            }
        }
    };
}

pub(crate) use group_metadata_accessors;
