# Obspec store as an explicit Python class

**Date:** 2026-07-22
**Status:** Approved, not yet implemented

## Problem

`ObspecStore` implements the `zarrs` sync storage traits on top of a Python object
following the obspec protocol. It reaches Python today only through a duck-typed
fallback in `PySyncStorage::extract`: any object carrying `head`, `get`, and
`get_ranges` is silently adopted as a store.

That has three costs:

- A user cannot signal intent. Passing an object that happens to have those three
  attributes works by accident; passing one that is missing an attribute produces a
  confusing "expected a FilesystemStore, MemoryStore, or ObspecStore" error rather than
  naming what is wrong.
- The protocol check lives in a `FromPyObject` impl rather than in a constructor, so it
  is a second parse point alongside the built-in stores' `cast::<PyClass>()` path.
- Nothing verifies the read path works. `ObspecStore` is the only store whose trait
  impl is code in this repo rather than upstream `zarrs`, and no test exercises it.

## Goals

1. Expose `ObspecStore` as a real `#[pyclass]`, constructed explicitly.
2. Remove the duck-typed fallback so there is exactly one way in.
3. Override `ReadableStorageTraits::get` so whole-key reads use obspec's `get`.
4. Export the class and give it a smoke test.

## Non-goals

- **Implementing write and list.** The eight `todo!()` bodies on
  `WritableStorageTraits` and `ListableStorageTraits` stay as they are. They panic if
  reached; converting them to `StorageError::ReadOnly` / `Unsupported` and then
  implementing them against `obstore.put` / `delete` / `list` /
  `list_with_delimiter` is a separate piece of work.
- **The async counterpart.** An `AsyncObspecStore` driving `get_async` /
  `get_ranges_async` is wanted eventually. Nothing here should block it — see
  "Forward compatibility".
- **Any broader store read API.** Stores stay opaque handles, exposing only a
  constructor and `__repr__`, as `FilesystemStore` and `MemoryStore` do.

## Design

### 1. `PyObspecStore` becomes a pyclass

The existing two-type split is already the right shape and does not change:
`ObspecStore` carries the trait impls, `PyObspecStore` is the `Arc`-holding wrapper.
The wrapper simply becomes a pyclass, following `PyFilesystemStore` and
`PyMemoryStore` exactly:

```rust
#[pyclass(module = "zarrista", frozen, name = "ObspecStore", skip_from_py_object)]
#[derive(Clone)]
pub struct PyObspecStore(pub(super) Arc<ObspecStore>);

#[pymethods]
impl PyObspecStore {
    /// Wrap a Python object implementing the obspec protocol.
    #[new]
    fn new(obj: Py<PyAny>) -> PyResult<Self> { ... }

    fn __repr__(&self) -> String { ... }
}
```

Rust reads `ObspecStore` (the store) against `PyObspecStore` (the Python wrapper);
Python sees only `ObspecStore`. This is the `Py`-prefix convention in CLAUDE.md.

The `hasattr("head" | "get" | "get_ranges")` check moves from `FromPyObject::extract`
into `#[new]`, where a failure can name the missing attribute. The hand-written
`FromPyObject` and `IntoPyObject` impls are deleted: `skip_from_py_object` plus
`obj.cast::<PyObspecStore>()` in `PySyncStorage::extract` is how the other two stores
already work.

`__repr__` should include the wrapped object's own repr, since the wrapper is
otherwise indistinguishable between backends.

**Behaviour change on the way out.** `Array.store` and `Group.store` return
`PySyncStorage` to Python. The hand-written `IntoPyObject` currently hands back the
*original* obspec object, so `Array.open(obstore_store).store is obstore_store` holds
today.

This is forced by the explicit-only decision rather than incidental. Keeping a custom
`IntoPyObject` that returns the raw object would break the round-trip:

```python
arr = Array.open(ObspecStore(s))
Array.open(arr.store)   # TypeError: raw obstore objects are no longer accepted
```

`Array.open(arr.store)` and `Group.open(arr.store, path)` are the natural way to reach
a sibling node from an existing handle, and they work today. Letting the derived
conversion return an `ObspecStore` wrapper keeps them working, and matches
`FilesystemStore` / `MemoryStore`, which already return the zarrista wrapper.

The cost is that `arr.store` is no longer the obstore object, so passing it to obstore
functions (`obstore.get(arr.store, ...)`) stops working.

**`ObspecStore.obj` restores that access.** A read-only getter returns the wrapped
Python object, so the underlying store stays reachable without making `arr.store`
itself the raw object:

```python
arr = Array.open(ObspecStore(s))
arr.store          # ObspecStore  — accepted by Array.open / Group.open
arr.store.obj is s # True — usable with obstore functions directly
```

This keeps the two roles separate: `arr.store` is what zarrista accepts back, `.obj` is
the escape hatch to the backend.

**Required stub updates.** `python/zarrista/_array.pyi` and `python/zarrista/_group.pyi`
type the `store` getter as `FilesystemStore | MemoryStore`; both need `ObspecStore`
added.

### 2. Override `get`

`zarrs` provides a default:

```rust
fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
    self.get_partial(key, ByteRange::FromStart(0, None))
}
```

`FromStart(0, None)` lands in the `OpenEndedRanges` branch, so **every** whole-object
read — every `zarr.json`, every unsharded chunk — becomes
`get(path, options={"range": {"offset": 0}})`, sending `Range: bytes=0-` and taking a
206 response, instead of a plain `get(path)`.

`ObspecStore` overrides `get` to call obspec's `get` with no `options`.

The override's body is nearly identical to `execute_get`, so both collapse onto one
helper:

```rust
/// Fetch a whole object, or one open-ended range of it, via obspec `get`.
fn call_get(
    store: &Bound<'_, PyAny>,
    key: &StoreKey,
    range: Option<(&str, u64)>,   // None -> no `options` kwarg at all
) -> Result<Option<Bytes>, StorageError>
```

`get` passes `None`; `OpenEndedRanges::execute` passes `Some(("offset", offset))` or
`Some(("suffix", length))`. This removes a duplicated body rather than adding one.

### 3. Exports

- `python/zarrista/store.py`: import `ObspecStore` from `._zarrista`, add to `__all__`.
- `python/zarrista/_store.pyi`: the class stub. Per CLAUDE.md the `.pyi` is the single
  source of truth for user-facing docs, so the prose lives here and the Rust `///`
  comment stays a one-line summary. Cover: which obspec methods are required
  (`head`, `get`, `get_ranges`), that the store is currently read-only and write or
  list operations are unsupported, and an obstore example.

### 4. Tests

`tests/test_store_input.py` is currently failing: `test_open_rejects_non_store`
asserts on `"FilesystemStore or MemoryStore"`, but the message is now
`"expected a FilesystemStore, MemoryStore, or ObspecStore"`. Update the regex.

Add one smoke test to the same file: open a zarr array through
`ObspecStore(obstore.store.MemoryStore())` and assert chunk contents, reusing the
existing `array_path` fixture pattern. This is deliberately minimal — no recording
fake, no assertions on which obspec calls were issued, no new API surface. Its purpose
is to establish that the read path executes at all, which nothing does today.

Also add a test that a non-obspec object passed to `ObspecStore(...)` raises
`TypeError`, since the protocol check has moved into the constructor.

**Known risk:** with the `todo!()` bodies retained, any test that reaches a list
operation aborts rather than failing cleanly. If the smoke test panics, that is a
finding — it means the array read path touches `ListableStorageTraits`, and the stubs
need addressing sooner than planned.

## Forward compatibility

`ObspecStore` holds a plain `Py<PyAny>`, not an obstore-specific type, so an
`AsyncObspecStore` can wrap the same Python object and call `get_async` /
`get_ranges_async` without restructuring. The range-partitioning types (`ReadRanges`,
`BoundedRanges`, `OpenEndedRanges`) are pure and contain no sync-specific logic, so an
async implementation reuses them directly; only the three `execute` methods need async
counterparts.

## Testing gap accepted

The ordering invariant — that a single `get_partial_many` mixing bounded, offset, and
suffix ranges returns bytes in the caller's order — is covered only by the existing
Rust unit tests on `ReadRanges` partitioning, not end-to-end. Driving mixed shapes
through `Array.open` deterministically is not currently possible, because store methods
are not exposed to Python and `cargo test` embeds the Homebrew interpreter rather than
the venv, so Rust tests cannot import `obstore` without a machine-specific
`PYTHONPATH`. Accepted for now; revisit if the read path proves fragile.
