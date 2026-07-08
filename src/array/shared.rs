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
            fn attrs(&self) -> $crate::metadata::PyAttributes {
                self.inner.attributes().clone().into()
            }

            #[getter]
            fn chunk_grid(&self) -> $crate::array::PyChunkGrid {
                self.inner.chunk_grid().clone().into()
            }

            #[getter]
            fn chunk_grid_shape(&self) -> &[u64] {
                self.inner.chunk_grid_shape()
            }

            fn chunk_key(
                &self,
                chunk_indices: $crate::array::PyChunkIndices,
            ) -> $crate::storage::PyStoreKey {
                self.inner.chunk_key(chunk_indices.as_ref()).into()
            }

            #[getter]
            fn chunk_key_encoding(&self) -> $crate::array::PyChunkKeyEncoding {
                self.inner.chunk_key_encoding().clone().into()
            }

            fn chunk_origin(
                &self,
                chunk_indices: $crate::array::PyChunkIndices,
            ) -> $crate::error::ZarristaResult<$crate::array::PyArrayIndices> {
                Ok(self.inner.chunk_origin(chunk_indices.as_ref())?.into())
            }

            fn chunk_shape(
                &self,
                chunk_indices: $crate::array::PyChunkIndices,
            ) -> $crate::error::ZarristaResult<$crate::array::PyChunkShape> {
                Ok(self.inner.chunk_shape(chunk_indices.as_ref())?.into())
            }

            fn chunk_subset(
                &self,
                chunk_indices: $crate::array::PyChunkIndices,
            ) -> $crate::error::ZarristaResult<$crate::array::PyArraySubset> {
                Ok(self.inner.chunk_subset(chunk_indices.as_ref())?.into())
            }

            /// The bytes-to-bytes codecs ("compressors").
            #[getter]
            fn compressors(&self) -> Vec<$crate::codec::PyBytesToBytesCodec> {
                let codecs = self.inner.codecs();
                codecs
                    .bytes_to_bytes_codecs()
                    .iter()
                    .map(|c| $crate::codec::PyBytesToBytesCodec::new(c.clone()))
                    .collect()
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
            fn fill_value(&self) -> $crate::array::PyFillValue {
                self.inner.fill_value().clone().into()
            }

            /// The array-to-array codecs ("filters").
            #[getter]
            fn filters(&self) -> Vec<$crate::codec::PyArrayToArrayCodec> {
                let codecs = self.inner.codecs();
                codecs
                    .array_to_array_codecs()
                    .iter()
                    .map(|f| $crate::codec::PyArrayToArrayCodec::new(f.clone()))
                    .collect()
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
            fn path(&self) -> $crate::node::PyNodePath {
                self.inner.path().clone().into()
            }

            /// The array-to-bytes codec ("serializer").
            #[getter]
            fn serializer(&self) -> $crate::codec::PyArrayToBytesCodec {
                let codecs = self.inner.codecs();
                $crate::codec::PyArrayToBytesCodec::new(codecs.array_to_bytes_codec().clone())
            }

            /// The array shape.
            #[getter]
            fn shape(&self) -> &[u64] {
                self.inner.shape()
            }

            #[getter]
            fn subset_all(&self) -> $crate::array::PyArraySubset {
                self.inner.subset_all().into()
            }
        }
    };
}

pub(crate) use array_metadata_accessors;
