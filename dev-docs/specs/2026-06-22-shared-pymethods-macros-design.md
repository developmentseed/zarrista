# Shared `#[pymethods]` macros for sync/async Array & Group

**Date:** 2026-06-22
**Status:** Approved

## Problem

`PyArray`/`PyAsyncArray` (and `PyGroup`/`PyAsyncGroup`) duplicate a set of
metadata accessors that are byte-for-byte identical between the sync and async
variants. These accessors only read from `self.inner` and perform no I/O, so
they do not differ across the sync/async split — yet they are copy-pasted into
both `#[pymethods]` blocks. As more upstream `zarrs` `Group` methods get
exposed, this duplication will grow.

## Constraints / why not the obvious alternatives

- **No generic Rust base pyclass.** `#[pyclass]` cannot be generic; pyo3 must
  register one concrete type with the interpreter. `#[pyclass] struct
  PyBaseArray<S: Storage>` will not compile.
- **pyo3 inheritance does not share logic here.** A base `#[pymethods]` block
  can only access *base* fields, but the storage-typed `inner` lives in the
  subclass. A Rust base could provide `isinstance` but could not implement
  `shape`/`dtype`/etc.
- **Python API stays unchanged.** Goal is internal Rust DRY only. A
  Python-visible base class / Protocol is explicitly *not* a requirement and is
  out of scope for this change (could be added later as a pure-Python layer).

## Design

Use a `macro_rules!` macro per node type that expands to the shared accessor
*items*. The macro is invoked inside each concrete type's existing
`#[pymethods]` block, so it re-expands against each concrete type and sidesteps
the generics limitation. This works for both sync and async because async's
`inner: Arc<Array<…>>` derefs to `Array<…>`, so `self.inner.shape()` etc.
compile identically.

### `src/array/shared.rs`

Defines `array_metadata_accessors!` emitting these 9 accessors (all `#[getter]`
except where noted):

- `attrs`, `chunk_grid`, `codecs`, `dimension_names`, `dtype`, `metadata`,
  `ndim`, `path`, `shape`

Exported via `pub(crate) use array_metadata_accessors;`. Uses `$crate::`-rooted
paths for referenced types (`PyChunkGrid`, `PyCodecChain`, `PyDataType`) and
fully-qualified `::pythonize::` calls so the macro is hygienic regardless of the
caller's imports.

### `src/group/shared.rs`

Defines `group_metadata_accessors!`. Starts with `attrs`. Grows as more
storage-agnostic `zarrs` `Group` methods are exposed.

### Wiring

- `src/array/mod.rs` and `src/group/mod.rs` each gain `mod shared;`.
- `sync.rs` / `async.rs` for both array and group `use` the macro and invoke it
  at the top of their `#[pymethods]` block; the duplicated accessors are
  deleted from both files.

## Out of scope (stays per-type)

- `__repr__` — differs by type-name string (`"Array"` vs `"AsyncArray"`).
- All I/O methods — `open`/`open_async`, `retrieve_*`, `__getitem__`,
  `array_keys`/`group_keys` — different signatures and bodies (sync returns a
  value; async takes `py` and returns `future_into_py(...)`).
- Any Python-visible base class / Protocol.

## Success criteria

- `cargo build` succeeds; the macro-generated methods are present on all four
  Python classes (verified via existing behavior / a quick attribute check).
- No change to the Python-visible API surface.
- The 9 array accessors and group `attrs` exist in exactly one place each.
