# zarrista — read-only zarrs API design

**Date:** 2026-06-16
**Status:** Approved

## Goal

Draft a simple, read-only Rust/pyo3 API over the [`zarrs`](https://github.com/zarrs/zarrs)
crate for opening and reading Zarr hierarchies. The API is modeled after the
simplicity of [zarrita.js](https://github.com/manzt/zarrita.js): construct a
store, `open` a node, read array regions with numpy-style indexing.

This is the first functional binding step after the hello-world scaffold. An
official `zarrs-python` binding exists; zarrista is a deliberately leaner,
zarrita-flavored take.

## Decisions

- **Sync** read path (zarrs default features: `filesystem`, `ndarray`, codecs).
  Async/cloud deferred.
- **pyo3 classes directly** — `Array`, `Group`, and the stores are `#[pyclass]`
  from the start (no separate pure-Rust core layer yet).
- **Filesystem + memory** stores only. No HTTP/cloud yet.
- Read results are returned as **numpy arrays** (via the `numpy` crate).

## Dependencies (Cargo.toml)

- `zarrs` — default features (`filesystem`, `ndarray`, common codecs).
- `zarrs_storage` — for `MemoryStore`.
- `numpy = "0.29"` (rust-numpy, pyo3 0.29-compatible).
- `pyo3 = "0.29"` (existing).

## Module layout

```
src/
  lib.rs      # #[pymodule] registration + free open() function
  error.rs    # zarrs errors -> Python exceptions
  store.rs    # FilesystemStore, MemoryStore (#[pyclass])
  array.rs    # Array (#[pyclass])
  group.rs    # Group (#[pyclass])
  dtype.rs    # zarrs DataType.name() <-> numpy dtype mapping + read dispatch
```

Each store holds an `Arc<dyn ReadableListableStorageTraits>` internally.

## API surface (modeled on zarrita.js)

### Stores — constructed directly

```python
store = zarrista.FilesystemStore("/data/example.zarr")
store = zarrista.MemoryStore()
```

### Opening — free `open()`, auto-detecting array vs group

```python
node = zarrista.open(store, "/temp")  # -> Array | Group
arr = zarrista.open(store, "/temp", kind="array")
grp = zarrista.open(store)  # path defaults to "/"
```

Backed by `Array::open` / `Group::open`. `kind` ("array" | "group") narrows and
raises on mismatch. Auto-detect tries array, then group.

### Array

Properties (zarrita names): `.shape`, `.chunks`, `.dtype` (string, e.g.
`"float32"`), `.attrs` (dict), `.dimension_names`, `.fill_value`, `.ndim`.

### Reading → numpy

Numpy-style basic indexing via `__getitem__`, returning a C-order
`numpy.ndarray`:

```python
arr[:]  # whole array
arr[0:10, :]  # sub-region
arr[5, 0:4]  # integer index drops that axis
arr[...]  # ellipsis / implicit trailing full-axis
```

- Supported index elements: `int`, `slice` with **step 1 / None**, full `:`,
  `Ellipsis`, and fewer indices than `ndim` (trailing axes implied full).
- Mapped to a contiguous `ArraySubset` (`ArraySubset::new_with_ranges`) →
  `retrieve_array_subset::<Vec<u8>>` → numpy array typed by dtype dispatch.
- `step != 1`, negative indices, and fancy/boolean indexing raise `IndexError`
  / `NotImplementedError` for now.
- Low-level escape hatch: `arr.get_chunk((0, 0)) -> np.ndarray`
  (maps `retrieve_chunk`).

**dtype scope:** fixed numeric types (int/uint 8–64, float16/32/64, bool).
Variable-length (`string` / `bytes`) raise `NotImplementedError` in this draft.

### Group navigation

zarrs filesystem + memory stores are listable, so:

```python
child = grp["temperature"]  # __getitem__ -> Array | Group
grp.attrs  # dict
grp.array_keys() / grp.group_keys()  # children names
```

## Error handling

`error.rs` defines a `ZarristaError` base exception, with a `NotFoundError`
subclass for missing nodes. zarrs `ArrayCreateError` / `GroupCreateError` /
`StorageError` / `ArrayError` map onto these.

## Testing

- Rust unit tests: build a `MemoryStore` fixture (write a small array with zarrs),
  assert open / metadata / subset reads round-trip.
- pytest smoke tests: the same flow through Python (open, check shape/dtype,
  read a region, compare to expected numpy values).

## Success criteria

- `maturin develop` builds with the new deps.
- Python can open a memory/filesystem Zarr array, read `.shape`/`.dtype`/`.attrs`,
  and `arr[...]` returns a correct numpy array.
- Group `__getitem__` navigates to child arrays/groups.
- Rust `cargo test` and `pytest` pass.

## Out of scope (deferred)

Writing; async / cloud / HTTP stores; strided (step≠1), negative, and fancy
indexing; variable-length (string/bytes) dtypes; consolidated metadata;
explicit v2-vs-v3 pinning (rely on zarrs auto-detect).
