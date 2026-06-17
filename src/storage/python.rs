// use pyo3::prelude::*;
// use zarrs::storage::{ReadableStorage, ReadableStorageTraits, StorageError, StoreKey};

// use crate::storage::key::PyStoreKey;

// /// A Python backend for making requests that conforms to the GetRangeAsync and GetRangesAsync
// /// protocols defined by obspec.
// /// https://developmentseed.org/obspec/latest/api/get/#obspec.GetRangeAsync
// /// https://developmentseed.org/obspec/latest/api/get/#obspec.GetRangesAsync
// #[derive(Debug)]
// pub(crate) struct PyReadable(Py<PyAny>);

// impl ReadableStorageTraits for PyReadable {
//     fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
//         let key = PyStoreKey::from(key.clone());
//     }
// }
