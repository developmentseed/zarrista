//! A storage adapter that reads and lists transparently but rejects all writes
//! at runtime. It satisfies the maximal `ReadableWritableListableStorageTraits`
//! (via zarrs' blanket impl) so a read-only array stays the same `Array` type as
//! a writable one; mutation attempts surface as `StorageError::ReadOnly`.

use std::sync::Arc;

use zarrs::storage::byte_range::{ByteRange, ByteRangeIterator};
use zarrs::storage::{
    AsyncListableStorageTraits, AsyncMaybeBytesIterator, AsyncReadableListableStorageTraits,
    AsyncReadableStorageTraits, AsyncWritableStorageTraits, Bytes, ListableStorageTraits,
    MaybeBytes, MaybeBytesIterator, OffsetBytesIterator, ReadableListableStorageTraits,
    ReadableStorageTraits, StorageError, StoreKey, StoreKeys, StoreKeysPrefixes, StorePrefix,
    WritableStorageTraits,
};

/// Wraps a readable + listable store, rejecting every write with
/// [`StorageError::ReadOnly`].
pub struct ReadOnly(Arc<dyn ReadableListableStorageTraits>);

impl ReadOnly {
    pub fn new(inner: Arc<dyn ReadableListableStorageTraits>) -> Self {
        Self(inner)
    }
}

impl ReadableStorageTraits for ReadOnly {
    fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        self.0.get(key)
    }

    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        self.0.get_partial_many(key, byte_ranges)
    }

    fn get_partial(
        &self,
        key: &StoreKey,
        byte_range: ByteRange,
    ) -> Result<MaybeBytes, StorageError> {
        self.0.get_partial(key, byte_range)
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.0.size_key(key)
    }

    fn supports_get_partial(&self) -> bool {
        self.0.supports_get_partial()
    }
}

impl ListableStorageTraits for ReadOnly {
    fn list(&self) -> Result<StoreKeys, StorageError> {
        self.0.list()
    }

    fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        self.0.list_prefix(prefix)
    }

    fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        self.0.list_dir(prefix)
    }

    fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        self.0.size_prefix(prefix)
    }
}

impl WritableStorageTraits for ReadOnly {
    fn set(&self, _key: &StoreKey, _value: Bytes) -> Result<(), StorageError> {
        Err(StorageError::ReadOnly)
    }

    fn set_partial_many(
        &self,
        _key: &StoreKey,
        _offset_values: OffsetBytesIterator,
    ) -> Result<(), StorageError> {
        Err(StorageError::ReadOnly)
    }

    fn erase(&self, _key: &StoreKey) -> Result<(), StorageError> {
        Err(StorageError::ReadOnly)
    }

    fn erase_prefix(&self, _prefix: &StorePrefix) -> Result<(), StorageError> {
        Err(StorageError::ReadOnly)
    }

    fn supports_set_partial(&self) -> bool {
        false
    }
}

/// Async twin of [`ReadOnly`]: reads and lists transparently, rejects writes.
pub struct AsyncReadOnly(Arc<dyn AsyncReadableListableStorageTraits>);

impl AsyncReadOnly {
    pub fn new(inner: Arc<dyn AsyncReadableListableStorageTraits>) -> Self {
        Self(inner)
    }
}

#[async_trait::async_trait]
impl AsyncReadableStorageTraits for AsyncReadOnly {
    async fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        self.0.get(key).await
    }

    async fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<AsyncMaybeBytesIterator<'a>, StorageError> {
        self.0.get_partial_many(key, byte_ranges).await
    }

    async fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.0.size_key(key).await
    }

    fn supports_get_partial(&self) -> bool {
        self.0.supports_get_partial()
    }
}

#[async_trait::async_trait]
impl AsyncListableStorageTraits for AsyncReadOnly {
    async fn list(&self) -> Result<StoreKeys, StorageError> {
        self.0.list().await
    }

    async fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        self.0.list_prefix(prefix).await
    }

    async fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        self.0.list_dir(prefix).await
    }

    async fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        self.0.size_prefix(prefix).await
    }
}

#[async_trait::async_trait]
impl AsyncWritableStorageTraits for AsyncReadOnly {
    async fn set(&self, _key: &StoreKey, _value: Bytes) -> Result<(), StorageError> {
        Err(StorageError::ReadOnly)
    }

    async fn set_partial_many<'a>(
        &'a self,
        _key: &StoreKey,
        _offset_values: OffsetBytesIterator<'a>,
    ) -> Result<(), StorageError> {
        Err(StorageError::ReadOnly)
    }

    async fn erase(&self, _key: &StoreKey) -> Result<(), StorageError> {
        Err(StorageError::ReadOnly)
    }

    async fn erase_prefix(&self, _prefix: &StorePrefix) -> Result<(), StorageError> {
        Err(StorageError::ReadOnly)
    }

    fn supports_set_partial(&self) -> bool {
        false
    }
}
