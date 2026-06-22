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

Use a `macro_rules!` macro per node type that expands to a **complete, separate
`#[pymethods] impl $ty` block**. The macro is parameterized over the concrete
type (`$ty:ty`) and invoked at module level (not inside an existing
`#[pymethods]` block), so it re-expands against each concrete type and sidesteps
the generics limitation. This works for both sync and async because async's
`inner: Arc<Array<…>>` derefs to `Array<…>`, so `self.inner.shape()` etc.
compile identically.

### Why a whole `#[pymethods]` block (and `multiple-pymethods`)

A macro invocation *inside* a `#[pymethods]` block does not work: pyo3's
proc-macro processes the impl block before the `macro_rules!` invocation is
expanded, so it never sees the generated methods (`error: macros cannot be used
as items in #[pymethods] impl blocks`). The macro must therefore emit its own
`#[::pyo3::pymethods] impl $ty { … }` block.

Having two `#[pymethods]` blocks for one type (the macro-generated one plus the
hand-written type-specific one) requires the **`multiple-pymethods`** pyo3
feature. This is enabled in `Cargo.toml`. It is verified compatible with the
crate's `abi3-py311` feature, and pulls in `inventory`, which is already a
transitive dependency via pyo3 — so no meaningful new dependency.

### `src/array/shared.rs`

Defines `array_metadata_accessors!($ty)` emitting a `#[pymethods] impl $ty`
block with these 9 `#[getter]` accessors:

- `attrs`, `chunk_grid`, `codecs`, `dimension_names`, `dtype`, `metadata`,
  `ndim`, `path`, `shape`

`metadata` returns `PyArrayMetadata` (the metadata newtype handles
pythonization via its `IntoPyObject` impl) rather than calling `pythonize`
inline, mirroring the group accessor.

Exported via `pub(crate) use array_metadata_accessors;`. Uses `$crate::`-rooted
paths for referenced types (`PyChunkGrid`, `PyCodecChain`, `PyDataType`,
`PyArrayMetadata`) and fully-qualified `::pyo3::` / `::pythonize::` paths so the
macro is hygienic regardless of the caller's imports.

### `src/group/shared.rs`

Defines `group_metadata_accessors!($ty)` emitting a `#[pymethods] impl $ty`
block with the storage-agnostic, no-I/O `Group` accessors:

- `attrs` (`#[getter]`, pythonized attributes map)
- `metadata` (`#[getter]`, returns `PyGroupMetadata` — the newtype's
  `IntoPyObject` impl handles pythonization, so the getter just clones
  `self.inner.metadata()` and `.into()`s it; no inline `pythonize`)
- `consolidated_metadata` (`#[getter]`, returns
  `Option<PyConsolidatedMetadata>` — `self.inner.consolidated_metadata()`
  already returns an owned `Option`, so it just `.map(Into::into)`s)
- `path` (`#[getter]`, `self.inner.path().as_str()`)

`PyConsolidatedMetadata` is added to `src/metadata.rs` via the existing
`pythonized_metadata!` macro (wrapping
`zarrs::metadata_ext::group::consolidated_metadata::ConsolidatedMetadata`,
which derives `Serialize`/`Deserialize`/`Clone`).

Grows further as more storage-agnostic `zarrs` `Group` methods are exposed.
I/O methods (`array_keys`/`group_keys`, child navigation) stay per-type.

### Wiring

- `src/array/mod.rs` and `src/group/mod.rs` each gain `mod shared;`.
- `sync.rs` / `async.rs` for both array and group `use` the macro and invoke it
  at module level (between the struct's inherent `impl` and the type-specific
  `#[pymethods]` block), with a one-line comment pointing at `shared.rs`. The
  duplicated accessors are deleted from both files.

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
