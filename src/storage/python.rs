//! A custom, duck-typed Python object adapted to the `zarrs` sync storage traits.
//!
//! The Python object declares capabilities via `@property` predicates and
//! implements a small set of methods. [`PyDuckStore`] reads the capability flags
//! once at construction and adapts the object to [`ReadableStorageTraits`]
//! (+ [`ListableStorageTraits`], added in a later task).

use pyo3::call::PyCallArgs;
use pyo3::prelude::*;
use zarrs::storage::byte_range::{ByteRange, ByteRangeIterator};
use zarrs::storage::{
    Bytes, ListableStorageTraits, MaybeBytes, MaybeBytesIterator, ReadableStorageTraits,
    StorageError, StoreKey, StoreKeys, StoreKeysPrefixes, StorePrefix, StorePrefixes,
};

/// A Python object adapted to the `zarrs` sync storage traits.
#[derive(Debug)]
pub(crate) struct PyDuckStore {
    obj: Py<PyAny>,
    supports_get_partial: bool,
    supports_listing: bool,
}

impl PyDuckStore {
    /// Wrap a duck-typed Python store object, reading its capability flags now.
    pub(crate) fn new(obj: &Bound<'_, PyAny>) -> Self {
        Self {
            obj: obj.clone().unbind(),
            supports_get_partial: read_bool_property(obj, "supports_get_partial"),
            supports_listing: read_bool_property(obj, "supports_listing"),
        }
    }

    /// Call the Python `get(key)` method, returning the full value or `None`.
    fn py_get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        Python::attach(|py| {
            let result = self
                .obj
                .bind(py)
                .call_method1("get", (key.as_str(),))
                .map_err(py_to_storage_error)?;
            if result.is_none() {
                return Ok(None);
            }
            let bytes = result.extract::<Vec<u8>>().map_err(py_to_storage_error)?;
            Ok(Some(Bytes::from(bytes)))
        })
    }
}

/// Read a boolean `@property`; a missing or non-bool property is `false`.
fn read_bool_property(obj: &Bound<'_, PyAny>, name: &str) -> bool {
    obj.getattr(name)
        .ok()
        .and_then(|v| v.extract::<bool>().ok())
        .unwrap_or(false)
}

/// Map a Python exception to a `zarrs` `StorageError`.
///
/// `zarrs`'s `StorageError` carries only strings, so the Python exception type
/// and traceback are flattened to the exception's display string here.
fn py_to_storage_error(err: PyErr) -> StorageError {
    StorageError::Other(err.to_string())
}

/// Encode a `ByteRange` as the Python `(kind, offset, length)` triple.
///
/// `kind` is `"start"` (read `length` bytes from `offset`, or to the end when
/// `length` is `None`) or `"suffix"` (read the last `length` bytes; `offset`
/// is unused and reported as `0`).
fn byte_range_to_py(byte_range: &ByteRange) -> (&'static str, u64, Option<u64>) {
    match byte_range {
        ByteRange::FromStart(offset, length) => ("start", *offset, *length),
        ByteRange::Suffix(length) => ("suffix", 0, Some(*length)),
    }
}

impl ReadableStorageTraits for PyDuckStore {
    fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        self.py_get(key)
    }

    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        // When the store declares partial support, delegate to its
        // `get_partial_many` instead of fetching the whole value.
        if self.supports_get_partial {
            let encoded: Vec<(&'static str, u64, Option<u64>)> =
                byte_ranges.map(|br| byte_range_to_py(&br)).collect();
            let bytes: Option<Vec<Bytes>> =
                Python::attach(|py| -> Result<Option<Vec<Bytes>>, StorageError> {
                    let result = self
                        .obj
                        .bind(py)
                        .call_method1("get_partial_many", (key.as_str(), encoded))
                        .map_err(py_to_storage_error)?;
                    if result.is_none() {
                        return Ok(None);
                    }
                    let raw = result
                        .extract::<Vec<Vec<u8>>>()
                        .map_err(py_to_storage_error)?;
                    Ok(Some(raw.into_iter().map(Bytes::from).collect()))
                })?;
            return Ok(bytes.map(|v| {
                Box::new(v.into_iter().map(Ok))
                    as Box<dyn Iterator<Item = Result<Bytes, StorageError>>>
            }));
        }

        // Fallback: fetch the full value and slice each range (mirrors MemoryStore).
        let Some(data) = self.py_get(key)? else {
            return Ok(None);
        };
        let out = byte_ranges.map(move |byte_range: ByteRange| {
            let len = data.len() as u64;
            let start = byte_range.start(len) as usize;
            let end = byte_range.end(len) as usize;
            if end > data.len() {
                Err(StorageError::Other(format!(
                    "byte range {byte_range:?} out of bounds for value of length {}",
                    data.len()
                )))
            } else {
                Ok(data.slice(start..end))
            }
        });
        Ok(Some(Box::new(out)))
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        // Prefer an explicit `size_key`; else fall back to len(get(key)).
        let has_size_key = Python::attach(|py| {
            self.obj
                .bind(py)
                .hasattr("size_key")
                .map_err(py_to_storage_error)
        })?;
        if has_size_key {
            return Python::attach(|py| {
                let result = self
                    .obj
                    .bind(py)
                    .call_method1("size_key", (key.as_str(),))
                    .map_err(py_to_storage_error)?;
                if result.is_none() {
                    return Ok(None);
                }
                result
                    .extract::<u64>()
                    .map(Some)
                    .map_err(py_to_storage_error)
            });
        }
        Ok(self.py_get(key)?.map(|b| b.len() as u64))
    }

    fn supports_get_partial(&self) -> bool {
        self.supports_get_partial
    }
}

impl PyDuckStore {
    /// Error returned by every listable method when listing is not declared.
    fn require_listing(&self) -> Result<(), StorageError> {
        if self.supports_listing {
            Ok(())
        } else {
            Err(StorageError::Unsupported(
                "store does not support listing".to_string(),
            ))
        }
    }

    /// Call a Python method returning a list of key strings.
    fn py_list_keys<'py, A: PyCallArgs<'py>>(
        &self,
        py: Python<'py>,
        method: &str,
        args: A,
    ) -> Result<StoreKeys, StorageError> {
        let result = self
            .obj
            .bind(py)
            .call_method1(method, args)
            .map_err(py_to_storage_error)?;
        let raw = result
            .extract::<Vec<String>>()
            .map_err(py_to_storage_error)?;
        raw.into_iter()
            .map(|k| StoreKey::new(k).map_err(|e| StorageError::Other(e.to_string())))
            .collect()
    }
}

impl ListableStorageTraits for PyDuckStore {
    fn list(&self) -> Result<StoreKeys, StorageError> {
        self.require_listing()?;
        Python::attach(|py| self.py_list_keys(py, "list", ()))
    }

    fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        self.require_listing()?;
        Python::attach(|py| self.py_list_keys(py, "list_prefix", (prefix.as_str(),)))
    }

    fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        self.require_listing()?;
        Python::attach(|py| {
            let result = self
                .obj
                .bind(py)
                .call_method1("list_dir", (prefix.as_str(),))
                .map_err(py_to_storage_error)?;
            let keys_obj = result.get_item("keys").map_err(py_to_storage_error)?;
            let prefixes_obj = result.get_item("prefixes").map_err(py_to_storage_error)?;
            let keys = keys_obj
                .extract::<Vec<String>>()
                .map_err(py_to_storage_error)?
                .into_iter()
                .map(|k| StoreKey::new(k).map_err(|e| StorageError::Other(e.to_string())))
                .collect::<Result<StoreKeys, _>>()?;
            let prefixes = prefixes_obj
                .extract::<Vec<String>>()
                .map_err(py_to_storage_error)?
                .into_iter()
                .map(|p| StorePrefix::new(p).map_err(|e| StorageError::Other(e.to_string())))
                .collect::<Result<StorePrefixes, _>>()?;
            Ok(StoreKeysPrefixes::new(keys, prefixes))
        })
    }

    fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        self.require_listing()?;
        Python::attach(|py| {
            self.obj
                .bind(py)
                .call_method1("size_prefix", (prefix.as_str(),))
                .map_err(py_to_storage_error)?
                .extract::<u64>()
                .map_err(py_to_storage_error)
        })
    }
}
