# `Array.with_shape` — resizing existing arrays

Design for [#111](https://github.com/developmentseed/zarrista/issues/111).

## Problem

zarrista has no way to change the shape of an existing array. zarrs supplies the
underlying operation, but only the metadata half of it.

## What zarrs provides

`Array::set_shape(&mut self, ArrayShape) -> Result<&mut Self, ArrayCreateError>`
rebuilds the chunk grid from the array's existing chunk-grid metadata against the
new shape, then patches `metadata.shape` for V3 or V2. It errors when the new
shape is incompatible with the chunk grid.

It is purely in-memory. Persisting is a separate `store_metadata()` /
`async_store_metadata()`, both of which zarrista already wraps.

zarrs does *not* delete chunks that fall outside a shrunken array's bounds, and
has no reload/refresh on `Array` — `Array::open` is the only way to re-read
metadata, and it constructs a fresh object.

## The model

zarrista's `Array` is a cached snapshot of `zarr.json` plus the objects derived
from it (chunk grid, codecs, data type, fill value), taken at `open()` and never
revisited. That cache is load-bearing: it is why `arr.shape` is a field read
rather than a network round-trip.

The design question is therefore not "mutable or immutable" but **what the sync
points are, and whether the API has one in each direction**:

| Direction | Cause | Answer |
| --- | --- | --- |
| Local ahead of store | `from_metadata()`, `with_shape()` | `store_metadata()` |
| Store ahead of local | another process, another handle, an Icechunk commit | none today |

The two are not symmetric. Local-ahead is *authored* — deliberate, with a fix to
hand. Store-ahead is *ambient* — nothing local causes it and there is no local
signal it happened. Closing that second gap is out of scope here; see
[Deferred](#deferred).

## Decision: return a new array

`with_shape` returns a new `Array` and leaves the receiver untouched. It does not
mutate in place.

Rebinding is the intended usage, and under it no stale handle exists:

```python
arr = Array.open(store)
arr = arr.with_shape([8, 8])
arr.store_metadata()
```

This follows established Python convention (`Path.with_suffix`,
`datetime.replace`) and matches `read_only()`, which already derives a new
`Array` from an existing one in this same class.

### Why not mutate in place

Mutation would match zarr-python's `Array.resize()`, and would keep a
non-rebound handle correct. It was rejected because:

- `#[pyclass(frozen)]` would need interior mutability
  (`RwLock<Arc<Array<…>>>`), touching all 57 `self.inner` sites across
  `sync.rs`, `async.rs`, and `shared.rs`.
- `shape()`, `chunk_grid_shape()`, and `dimension_names()` currently return
  borrowed data, which is impossible when borrowing from a temporary guard.
  They would have to return owned values.
- Locking is per-accessor, so a Python-level sequence such as
  `(arr.shape, arr.chunk_grid_shape)` could straddle a mutation and observe a
  mix of before and after. There is no way to hold the lock across that from
  Python. An immutable object is internally consistent for its whole lifetime.

Stale handles held elsewhere in a program are possible under either design, so
mutation does not solve that problem — it only shifts which object is correct.

## Implementation

`with_shape` performs no I/O and is byte-identical for `PyArray` and
`PyAsyncArray`, so it belongs in the `shared_array_methods!` macro in
`src/array/shared.rs` alongside the metadata accessors — not in `sync.rs` /
`async.rs`. Both classes expose a `pub(crate) fn new(Arc<Array<…>>, store)` and
a `store` field, and the macro expands in each class's own module, so the same
body compiles for both.

```rust
/// Return a new array with `shape`, leaving this one unchanged.
fn with_shape(
    &self,
    shape: $crate::array::PyArrayShape,
) -> $crate::error::ZarristaResult<Self> {
    let mut resized = self.inner.with_storage(self.inner.storage());
    resized.set_shape(shape)?;
    Ok(Self::new(::std::sync::Arc::new(resized), self.store.clone()))
}
```

`Array` is not `Clone`; `with_storage` supplies the deep copy, and passing the
array's own storage back leaves the storage type parameter unchanged.

`AsyncArray.with_shape` is a **sync** method — there is nothing to await. The
stub must say so, since every other `AsyncArray` method that touches the store is
a coroutine.

The macro's module doc comment currently reads "These accessors only read from
`self.inner` and perform no I/O". `with_shape` is not an accessor, so the wording
needs a light revision; the no-I/O invariant still holds.

## Error handling

`set_shape` returns `ArrayCreateError` when the shape does not fit the chunk
grid — wrong dimensionality, or a `rectangular` grid whose explicit boundaries do
not sum to the new extent. `ZarristaError` already converts from
`ArrayCreateError` (`Array::open` relies on it), so this needs no new error type.

There is no manual validation in the function body, per the project's
"no manual validation in function bodies" rule: zarrs is the validator. Because
the error propagates before `Self::new`, a rejected shape never yields an object.

## Documented semantics

The `.pyi` prose must be explicit about three points, each of which a user could
reasonably assume otherwise:

1. Nothing is written. `store_metadata()` is the push, as after
   `from_metadata()`.
2. The receiver is unchanged and remains valid. Rebinding is the intended usage.
3. Shrinking leaves chunks outside the new bounds in the store. Reclaiming them
   will be a separate vacuum call.

## Testing

`tests/array/test_with_shape.py`, following the `tests/array/test_store_metadata.py`
layout — a `_metadata()` helper, sync tests, then an async section.

- returns a new array at the new shape; the original still reports the old shape
- grow → `store_metadata()` → reopen reports the new shape
- shrink → `store_metadata()` → reopen reports the new shape
- data written before a grow reads back at the same coordinates afterward
- wrong dimensionality raises `ZarristaError`
- write a chunk, shrink so that chunk falls outside the new bounds, persist —
  the chunk is still present in the store afterward. Pins current behavior so the
  future vacuum call has something to flip.
- async: grow → `await store_metadata()` → reopen reports the new shape

## Deferred

Out of scope for #111, each warranting its own issue:

- **Vacuum.** A call that erases chunks outside the array's current bounds.
  zarr-python folds this into `resize()`; here it must be separate, because
  `with_shape` does no I/O at all. zarrs supplies `erase_chunks(&dyn
  ArraySubsetTraits)` (and `async_erase_chunks`) to build on.
- **A pull sync point.** A `reopen()` returning a fresh handle from a re-read of
  `zarr.json`, closing the store-ahead-of-local gap.
- **Optimistic concurrency.** Track an ETag at open and use a conditional PUT in
  `store_metadata()`, raising on mismatch rather than silently clobbering a
  concurrent writer. This is the real fix for lost updates, and obstore supports
  conditional put, but it has its own semantics to design.
