# Custom Python store protocol (sync)

**Date:** 2026-06-16
**Status:** Approved — ready for implementation plan

## Goal

Let a user pass an arbitrary, duck-typed Python object as a store, instead of
only the built-in `FilesystemStore` / `MemoryStore`. The object declares its
capabilities (partial reads, listing) and implements a small set of methods;
zarrista adapts it to the `zarrs` storage traits so it works anywhere the
built-in stores do.

This pulls forward roadmap Tier-3 item 7b ("a Python-implementable `Store`
protocol for custom backends, mirroring zarr-python's `Store` ABC"). See
[2026-06-16-api-roadmap.md](2026-06-16-api-roadmap.md).

### In scope

- **Sync only.** Implement `zarrs` `ReadableStorageTraits` +
  `ListableStorageTraits` over a Python object.
- **Readable required, listable optional**, detected via capability predicates.

### Out of scope (deferred, not foreclosed)

- **Async** custom stores (awaiting Python coroutines via
  `pyo3-async-runtimes`). The async/obstore path already exists and remains
  where real I/O concurrency lives.
- **Writable / deletable** custom stores. Writing is out of scope project-wide
  today; when it lands it forces a different `zarrs` static type
  (`ReadableWritableStorageTraits`) and is the right moment to introduce a
  storage enum or split path. We do **not** build that now.

## Why mirror `zarrs`, with an obspec adapter in Python

The Rust↔Python boundary mirrors the methods `zarrs` actually needs, smoothed
into Pythonic shapes. The compiled core stays decoupled from any external
spec; an obspec/obstore adapter can live in pure Python (iterate without
recompiling), and obstore users still get a one-liner. Method names mirror
`zarrs` (which mostly coincides with zarr-python's `Store` ABC); the one
borrowed-from-zarr-python concept is the `supports_listing` predicate, because
`zarrs` expresses listability through its *type system* rather than a method —
something we cannot replicate dynamically.

## Architecture

The entire codebase is statically typed on a single trait object,
`Arc<dyn ReadableListableStorageTraits>`. `Array::open` and `Group::open` both
take exactly that, and `Group::array_keys()` / `group_keys()` go through
`child_array_paths()`, which requires listing.

Rather than refactor `Array`/`Group` onto a capability enum (which fights
`zarrs`'s static generics and forces parallel `Array`/`Group` instantiations
per variant), use **a single Rust wrapper that always implements
`ReadableListableStorageTraits` and degrades at runtime**:

```rust
// src/storage/python.rs
pub(crate) struct PyStore(Py<PyAny>);   // Py<PyAny> is Send + Sync

impl ReadableStorageTraits for PyStore { /* calls Python get / get_partial* / size_key */ }

impl ListableStorageTraits for PyStore {
    fn list(&self) -> Result<StoreKeys, StorageError> {
        // supports_listing == false  ->  Err(StorageError::Unsupported(...))
        // else call the Python list method
    }
    // list_prefix / list_dir / size_prefix likewise
}
// blanket impl => ReadableListableStorageTraits for free
```

`extract_storage` grows one arm: anything that isn't a `FilesystemStore` /
`MemoryStore` is wrapped as `PyStore` and returned as
`Arc<dyn ReadableListableStorageTraits>`. **No changes to `Array`, `Group`,
`node`, or any call site.** A readable-only Python store opens arrays fine;
`group.array_keys()` on it raises a clear runtime error.

This replaces the unused `SyncStorage` enum in `src/storage/sync.rs` for now;
the enum returns when writing is implemented and genuinely needs it.

## The Python protocol (sync)

A duck-typed object. Capabilities are declared via `@property` (zarr-python
style); methods provide the bytes/keys. Tiered so a trivial store (e.g. a dict)
needs almost nothing, while a real backend opts into efficient partial reads.

### Capability predicates (`@property`, authoritative)

| Property                 | Type   | Meaning                                              |
| ------------------------ | ------ | ---------------------------------------------------- |
| `supports_get_partial`   | `bool` | drives `zarrs` `supports_get_partial()`; gates the partial-read methods |
| `supports_listing`       | `bool` | gates the listable methods                           |

A missing property is treated as `False`.

### Readable

| Method                                                     | Required?                       | Returns                       |
| --------------------------------------------------------- | ------------------------------- | ----------------------------- |
| `get(key: str)`                                           | **yes**                         | `bytes \| None` (None = absent) |
| `get_partial(key: str, byte_range: ByteRange)`            | optional (`supports_get_partial`) | `bytes \| None`             |
| `get_partial_many(key: str, byte_ranges: list[ByteRange])`| optional (`supports_get_partial`) | `list[bytes] \| None`       |
| `size_key(key: str)`                                      | optional                        | `int \| None`                 |

- `get` alone yields a correct, working store.
- When `supports_get_partial` is `False`, Rust synthesizes partial reads by
  fetching the full value via `get` and slicing. Efficiency is opt-in;
  correctness is free.
- `get_partial_many` is the one method `zarrs` strictly requires
  (`get_partial_many` on the trait). If the Python object provides only
  `get_partial`, Rust loops it; if it provides neither, Rust falls back to
  `get` + slice.
- `size_key` absent → Rust falls back to `len(get(key))`.

### Listable (only consulted when `supports_listing`)

| Method                      | Returns                                       |
| --------------------------- | --------------------------------------------- |
| `list()`                    | `list[str]`                                   |
| `list_prefix(prefix: str)`  | `list[str]`                                   |
| `list_dir(prefix: str)`     | `{"keys": list[str], "prefixes": list[str]}`  |
| `size_prefix(prefix: str)`  | `int`                                         |

`list_dir` mirrors `zarrs`'s `list_dir`, which returns both the keys and the
child prefixes directly under `prefix`.

### `ByteRange` representation

`zarrs`'s `ByteRange` is an enum: `FromStart(offset: u64, Option<length: u64>)`
**or** `Suffix(length: u64)` (a read of the last `length` bytes, which sharding
uses heavily). The Python representation must express both — a start-only tuple
would make suffix reads inexpressible.

Representation: a small object/tuple carrying `(kind, offset, length)` where
`kind` is `"start"` or `"suffix"` and, for `"start"`, `length` may be `None`
(read to the end of the value). For `"suffix"`, `offset` is unused and `length`
is the suffix size.
The exact concrete form (lightweight dataclass vs. tuple vs. exported pyclass)
is an implementation choice for the plan; the constraint is that both anchors
and an optional length round-trip.

## Error handling

Mirror zarr-python's categories as closely as the bridge allows:

- **Missing key** → the Python method returns `None` → `zarrs` `None`
  (not-found). No exception. Matches zarr-python `get` semantics.
- **Unsupported capability** (e.g. listing when `supports_listing` is `False`)
  → `StorageError::Unsupported("store does not support listing")`, short-
  circuited before any Python call. Surfaced to Python as a clear,
  zarr-python-like "operation not supported" error.
- **Exception raised inside a store method** → caught and wrapped as
  `StorageError::Other(<exception repr>)`, preserving the message.
  **Known limitation:** `zarrs`'s `StorageError` carries only strings, so the
  original Python exception *type* and traceback are flattened to a message on
  the way back out. Acceptable for this milestone; revisit only if it bites.

## GIL / threading

`Py<PyAny>` is `Send + Sync`, so `PyStore` satisfies the trait bounds. Every
method enters via `Python::attach` to take the GIL. Consequence, stated
plainly and accepted: `zarrs` may call a sync store from multiple rayon
threads, but the GIL serializes the Python-side calls — a sync custom Python
store does **not** get true I/O parallelism. That is inherent to sync-first;
the async/obstore path remains where concurrency lives.

## Testing

1. A pure-Python dict-backed store implementing the protocol, used to open an
   array/group and read a chunk — proves the boundary end-to-end.
2. A readable-only store (no `supports_listing`) asserting `group.array_keys()`
   raises the clear "not supported" error.
3. A Rust unit test under `auto-initialize` defining a tiny store class inline.

Round-trip tests vs. zarr-python can come with the broader harness the roadmap
already calls for.

## Files touched

- `src/storage/python.rs` — new `PyStore` wrapper + trait impls (replaces the
  current commented stub).
- `src/storage/sync.rs` — `extract_storage` grows the `PyStore` arm; remove the
  unused `SyncStorage` enum (returns with writing).
- `src/storage/mod.rs` — export wiring as needed.
- `src/error.rs` — map `StorageError::Unsupported` to a clear Python exception
  if the default `ZarristaException` message isn't sufficient.
- Tests under `tests/` (Python) and an inline Rust unit test.

