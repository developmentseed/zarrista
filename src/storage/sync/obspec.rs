use std::sync::Arc;

use pyo3::exceptions::{PyFileNotFoundError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use pyo3_bytes::PyBytes;
use zarrs::storage::byte_range::{ByteRange, ByteRangeIterator};
use zarrs::storage::{
    Bytes, ListableStorageTraits, MaybeBytesIterator, OffsetBytesIterator, ReadableStorageTraits,
    StorageError, StoreKey, StoreKeys, StoreKeysPrefixes, StorePrefix, WritableStorageTraits,
};

/// An object store based on an arbitrary Python object that implements the obspec protocol.
#[derive(Debug, Clone)]
pub struct PyObspecStore(pub(super) Arc<Py<PyAny>>);

crate::wasm_send_sync!(PyObspecStore);

impl FromPyObject<'_, '_> for PyObspecStore {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        if !obj.hasattr("head")? || !obj.hasattr("get")? || !obj.hasattr("get_ranges")? {
            return Err(PyTypeError::new_err(
                "expected an object implementing the obspec protocol",
            ));
        }

        Ok(Self(Arc::new(obj.into())))
    }
}

impl<'py> IntoPyObject<'py> for PyObspecStore {
    type Error = PyErr;
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.0.into_bound(py))
    }
}

/// Convert a Python exception into a [`StorageError`].
fn map_py_err(err: PyErr) -> StorageError {
    StorageError::Other(err.to_string())
}

/// Translate the result of an obspec call, mapping a missing object to `None`.
///
/// obspec implementations signal that a key is absent by raising `FileNotFoundError`,
/// which for `zarrs` is the `Ok(None)` case rather than an error.
fn missing_as_none<T>(py: Python<'_>, result: PyResult<T>) -> Result<Option<T>, StorageError> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(err) if err.is_instance_of::<PyFileNotFoundError>(py) => Ok(None),
        Err(err) => Err(map_py_err(err)),
    }
}

/// A byte range with a known start and exclusive end.
struct BoundedRange {
    /// Position of this range in the original request.
    index: usize,
    start: u64,
    end: u64,
}

/// Byte ranges that `get_ranges` can serve, because both ends are known.
#[derive(Default)]
struct BoundedRanges(Vec<BoundedRange>);

impl BoundedRanges {
    /// Fetch every bounded range in a single `get_ranges` call.
    fn execute(
        &self,
        store: &Bound<'_, PyAny>,
        key: &StoreKey,
    ) -> Result<Option<Vec<(usize, Bytes)>>, StorageError> {
        if self.0.is_empty() {
            return Ok(Some(Vec::new()));
        }

        let py = store.py();
        let starts = self.0.iter().map(|range| range.start).collect::<Vec<_>>();
        let ends = self.0.iter().map(|range| range.end).collect::<Vec<_>>();
        let kwargs = [("starts", starts), ("ends", ends)]
            .into_py_dict(py)
            .map_err(map_py_err)?;

        let result = store.call_method("get_ranges", (key.as_str(),), Some(&kwargs));
        let Some(buffers) = missing_as_none(py, result)? else {
            return Ok(None);
        };

        let buffers = buffers.extract::<Vec<PyBytes>>().map_err(map_py_err)?;
        if buffers.len() != self.0.len() {
            return Err(StorageError::Other(format!(
                "obspec get_ranges returned {} buffers for {} requested byte ranges",
                buffers.len(),
                self.0.len()
            )));
        }

        Ok(Some(
            self.0
                .iter()
                .zip(buffers)
                .map(|(range, buffer)| (range.index, buffer.into_inner()))
                .collect(),
        ))
    }
}

/// A byte range that runs to the end of the object.
struct OpenEndedRange {
    /// Position of this range in the original request.
    index: usize,
    /// The offset for an `OffsetRange`, or the length for a `SuffixRange`.
    value: u64,
}

/// Byte ranges that only the range option of `get` can express, one request each.
///
/// [`ByteRange::FromStart`] with no length and [`ByteRange::Suffix`] differ only in the
/// obspec range option they map to, so they share this representation and the caller
/// supplies the option name at execution time.
#[derive(Default)]
struct OpenEndedRanges(Vec<OpenEndedRange>);

impl OpenEndedRanges {
    /// Fetch each range with its own `get` call.
    ///
    /// `option` is `"offset"` or `"suffix"`, matching obspec's `OffsetRange` and `SuffixRange`.
    fn execute(
        &self,
        store: &Bound<'_, PyAny>,
        key: &StoreKey,
        option: &str,
    ) -> Result<Option<Vec<(usize, Bytes)>>, StorageError> {
        let mut buffers = Vec::with_capacity(self.0.len());
        for range in &self.0 {
            let Some(buffer) = execute_get(store, key, option, range.value)? else {
                return Ok(None);
            };
            buffers.push((range.index, buffer));
        }
        Ok(Some(buffers))
    }
}

/// Fetch one open-ended range via the range option of `get`.
fn execute_get(
    store: &Bound<'_, PyAny>,
    key: &StoreKey,
    option: &str,
    value: u64,
) -> Result<Option<Bytes>, StorageError> {
    let py = store.py();
    let range = [(option, value)].into_py_dict(py).map_err(map_py_err)?;
    let options = [("range", range)].into_py_dict(py).map_err(map_py_err)?;
    let kwargs = [("options", options)]
        .into_py_dict(py)
        .map_err(map_py_err)?;

    let result = store.call_method("get", (key.as_str(),), Some(&kwargs));
    let Some(get_result) = missing_as_none(py, result)? else {
        return Ok(None);
    };

    // `buffer` is the obspec spelling; obstore also exposes it as `bytes`.
    let buffer = get_result
        .call_method0("buffer")
        .and_then(|buffer| buffer.extract::<PyBytes>())
        .map_err(map_py_err)?;

    Ok(Some(buffer.into_inner()))
}

/// The obspec requests needed to serve a set of requested byte ranges.
///
/// `get_ranges` is the only bulk obspec read, and it requires a known start and
/// exclusive end for every range.
///
/// The two open-ended [`ByteRange`] shapes — [`ByteRange::FromStart`] with `(_, None)` and
/// [`ByteRange::Suffix`] — can only be expressed through the range option of `get`, so each costs
/// its own request.
///
/// Partitioning them out keeps every bounded range in the single `get_ranges` call, which is what
/// coalesces neighbouring reads.
///
/// Every entry carries its index into the original byte range sequence so that the
/// responses can be restored to the caller's order.
#[derive(Default)]
struct ReadRanges {
    bounded: BoundedRanges,
    offset: OpenEndedRanges,
    suffix: OpenEndedRanges,
}

impl ReadRanges {
    /// Construct a plan for the given byte ranges.
    fn new(byte_ranges: ByteRangeIterator<'_>) -> Self {
        let mut plan = Self::default();
        for (index, byte_range) in byte_ranges.enumerate() {
            match byte_range {
                ByteRange::FromStart(start, Some(length)) => plan.bounded.0.push(BoundedRange {
                    index,
                    start,
                    end: start + length,
                }),
                ByteRange::FromStart(start, None) => plan.offset.0.push(OpenEndedRange {
                    index,
                    value: start,
                }),
                ByteRange::Suffix(length) => plan.suffix.0.push(OpenEndedRange {
                    index,
                    value: length,
                }),
            }
        }
        plan
    }

    /// The number of byte ranges the plan covers.
    fn len(&self) -> usize {
        self.bounded.0.len() + self.offset.0.len() + self.suffix.0.len()
    }

    /// Issue the planned requests and collect the responses in the caller's order.
    ///
    /// Returns `Ok(None)` if `key` is absent from the store.
    fn execute(
        &self,
        store: &Bound<'_, PyAny>,
        key: &StoreKey,
    ) -> Result<Option<Vec<Bytes>>, StorageError> {
        let Some(bounded) = self.bounded.execute(store, key)? else {
            return Ok(None);
        };
        let Some(offset) = self.offset.execute(store, key, "offset")? else {
            return Ok(None);
        };
        let Some(suffix) = self.suffix.execute(store, key, "suffix")? else {
            return Ok(None);
        };

        // The three collections partition `0..len`, so every slot is assigned exactly once
        // below and no placeholder survives.
        let mut bytes = vec![Bytes::new(); self.len()];
        for (index, buffer) in bounded.into_iter().chain(offset).chain(suffix) {
            bytes[index] = buffer;
        }

        Ok(Some(bytes))
    }
}

impl ReadableStorageTraits for PyObspecStore {
    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        let plan = ReadRanges::new(byte_ranges);
        let bytes = Python::attach(|py| plan.execute(self.0.bind(py), key))?;
        Ok(bytes.map(|bytes| Box::new(bytes.into_iter().map(Ok)) as _))
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        Python::attach(|py| {
            let store = self.0.bind(py);
            let result = store.call_method1("head", (key.as_str(),));
            let Some(object_meta) = missing_as_none(py, result)? else {
                return Ok(None);
            };
            let size = object_meta
                .getattr("size")
                .and_then(|size| size.extract())
                .map_err(map_py_err)?;
            Ok(Some(size))
        })
    }

    fn supports_get_partial(&self) -> bool {
        true
    }
}

impl ListableStorageTraits for PyObspecStore {
    fn list(&self) -> Result<StoreKeys, StorageError> {
        todo!()
    }

    fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        todo!()
    }

    fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        todo!()
    }

    fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        todo!()
    }
}

impl WritableStorageTraits for PyObspecStore {
    fn erase(&self, key: &StoreKey) -> Result<(), StorageError> {
        todo!()
    }

    fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError> {
        todo!()
    }

    fn set(&self, key: &StoreKey, bytes: Bytes) -> Result<(), StorageError> {
        todo!()
    }

    fn set_partial_many(
        &self,
        key: &StoreKey,
        offset_values: OffsetBytesIterator,
    ) -> Result<(), StorageError> {
        todo!()
    }

    fn supports_set_partial(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(byte_ranges: Vec<ByteRange>) -> ReadRanges {
        ReadRanges::new(Box::new(byte_ranges.into_iter()))
    }

    #[test]
    fn partitions_each_byte_range_shape() {
        let plan = plan(vec![
            ByteRange::FromStart(10, Some(5)),
            ByteRange::Suffix(8),
            ByteRange::FromStart(20, None),
            ByteRange::FromStart(0, Some(4)),
        ]);

        assert_eq!(plan.bounded, vec![(0, 10, 15), (3, 0, 4)]);
        assert_eq!(plan.offset, vec![(2, 20)]);
        assert_eq!(plan.suffix, vec![(1, 8)]);
    }

    #[test]
    fn indices_partition_the_requested_ranges() {
        let plan = plan(vec![
            ByteRange::Suffix(8),
            ByteRange::FromStart(20, None),
            ByteRange::FromStart(10, Some(5)),
        ]);

        let mut indices = plan
            .bounded
            .iter()
            .map(|(index, ..)| *index)
            .chain(plan.offset.iter().map(|(index, _)| *index))
            .chain(plan.suffix.iter().map(|(index, _)| *index))
            .collect::<Vec<_>>();
        indices.sort_unstable();

        assert_eq!(plan.len(), 3);
        assert_eq!(indices, vec![0, 1, 2]);
    }
}
