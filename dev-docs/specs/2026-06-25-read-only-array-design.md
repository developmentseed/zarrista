# Read-only array via a `ReadOnly` storage adapter

**Date:** 2026-06-25
**Status:** Approved, ready for implementation plan

## Problem

`PyArray` wraps `Array<dyn ReadableWritableListableStorageTraits>` — the maximal
zarrs storage trait — and all write methods (`store_chunk`, `store_encoded_chunk`,
`erase_chunk`, `erase_metadata`, `compact_chunk`) live directly on it.

We want a `read_only()` method that returns an array which **raises at runtime** if
the user attempts to mutate it. The motivation is guarding against *accidental*
mutation of an array to which the user actually has write access.

zarrs' built-in `Array::readable()` downgrades to `Array<dyn ReadableStorageTraits>`,
a strictly less-capable *type* that `PyArray::new` will not accept (compile error:
`expected trait ReadableWritableListableStorageTraits, found trait
ReadableStorageTraits`). A compile-time-distinct read-only type is also the wrong
model for our future needs: stores like read-only S3 (via `ObjectStore`) or future
Python-protocol-backed stores only know they are read-only **at runtime**, not at
compile time.

## Decision

Introduce a `ReadOnly<T>` storage **adapter** that wraps a readable + listable store
and **satisfies the maximal `ReadableWritableListableStorageTraits`** by implementing
its write methods as runtime errors. This keeps the entire existing stack (one
`PyArray` type over one storage trait) unchanged — no new pyclass, no type surgery —
while making writes fail at runtime instead of compile time.

This works because zarrs provides a blanket impl: any `T: Readable + Writable +
Listable + 'static` automatically implements `ReadableWritableListableStorageTraits`
(`zarrs_storage/src/storage_sync.rs:254`). So an adapter that *delegates* reads and
lists and *errors* on writes transparently fills the maximal-trait slot.

Rejected alternative — a separate `ReadOnlyArray` pyclass over
`Array<dyn ReadableStorageTraits>`: gives compile-time guarantees but (a) requires
duplicating/factoring all read + metadata methods into shared macros for a second
type, and (b) cannot represent stores whose read-only-ness is only known at runtime,
which is the more important real-world case.

## Components

### 1. `ReadOnly<T>` — sync adapter

New file: `src/storage/read_only.rs`.

```rust
pub struct ReadOnly<T: ?Sized>(Arc<T>);

impl<T: ?Sized> ReadOnly<T> {
    pub fn new(inner: Arc<T>) -> Self { Self(inner) }
}
```

Trait impls (over `T: ?Sized + ReadableListableStorageTraits` as appropriate):

- `ReadableStorageTraits` — delegate every method to `self.0`
  (`get_partial_many`, `size_key`, `supports_get_partial`).
- `ListableStorageTraits` — delegate every method to `self.0`
  (`list`, `list_prefix`, `list_dir`, `size_prefix`).
- `WritableStorageTraits` — **every** method returns
  `Err(StorageError::ReadOnly)`:
  - `set` → `Err(StorageError::ReadOnly)`
  - `set_partial_many` → `Err(StorageError::ReadOnly)`
  - `erase` → `Err(StorageError::ReadOnly)`
  - `erase_prefix` → `Err(StorageError::ReadOnly)`
  - `supports_set_partial` → `false`

`StorageError::ReadOnly` already exists with the message "a write operation was
attempted on a read only store" (`zarrs_storage/src/lib.rs:170`). It flows through
the existing `ZarristaError` conversion and surfaces as a Python exception.

The blanket impl then yields `ReadableWritableListableStorageTraits` for free.

### 2. `AsyncReadOnly<T>` — async adapter

Same file (or `src/storage/read_only.rs` shared). Mirrors `ReadOnly` against the
async traits (`AsyncReadableStorageTraits`, `AsyncListableStorageTraits`,
`AsyncWritableStorageTraits`), whose method set is identical with `async fn`. Write
methods return `Err(StorageError::ReadOnly)`. The async blanket impl
(`storage_async.rs`) yields `AsyncReadableWritableListableStorageTraits`.

### 3. `PyArray::read_only` (sync)

Replace the current non-compiling body in `src/array/sync.rs`:

```rust
fn read_only(&self) -> Self {
    let inner = self.inner.storage().readable_listable(); // RWL -> RL
    let storage = Arc::new(ReadOnly::new(inner));          // RL  -> faked RWL
    Self::new(Arc::new(self.inner.with_storage(storage)))
}
```

- `Array::storage()` (`zarrs/src/array.rs:710`) → `Arc<dyn RWL>`.
- `.readable_listable()` (`storage_sync.rs:248`) downgrades to `Arc<dyn RL>`.
- `Array::with_storage(storage)` (`zarrs/src/array.rs:420`) rebuilds the `Array`
  with the same metadata over the new storage.

### 4. `PyAsyncArray::read_only` (async)

Same shape in `src/array/async.rs`, using `AsyncReadOnly` and the async
`readable_listable()`.

## Data flow

- Reads (`__getitem__`, `retrieve_array_subset`, `retrieve_chunk`,
  `retrieve_encoded_chunk`) and metadata accessors: pass through the adapter
  untouched.
- Writes (`store_chunk`, `store_encoded_chunk`, `erase_chunk`, `erase_metadata`,
  `compact_chunk`): hit the adapter's erroring write methods and raise
  `StorageError::ReadOnly` → Python exception.

## Out of scope (YAGNI for this pass)

- A `read_only` boolean / introspection property on the array. Future direction:
  expose `array.store`, and put `is_read_only` on that store object instead.
- Wrapping genuinely read-only stores at the `PySyncStorage`/`PyAsyncStorage`
  boundary (S3, Python-protocol stores). The `ReadOnly` adapter built here is the
  reusable mechanism that will enable it; the boundary wiring is a separate change.

## Testing

- `read_only()` returns an array that still reads correctly (round-trip a subset
  read equals the source array's read).
- Each write method on a read-only array raises (Python-level assertion that the
  expected exception type is raised) for both sync and async.
- A normal (non-read-only) array still writes successfully — no regression.
```