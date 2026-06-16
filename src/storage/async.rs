use std::sync::Arc;

use zarrs::storage::{
    AsyncReadableStorageTraits, AsyncReadableWritableStorageTraits, AsyncWritableStorageTraits,
};

#[allow(dead_code)]
pub enum AsyncStorage {
    Readable(Arc<dyn AsyncReadableStorageTraits>),
    Writable(Arc<dyn AsyncWritableStorageTraits>),
    ReadableWritable(Arc<dyn AsyncReadableWritableStorageTraits>),
}
