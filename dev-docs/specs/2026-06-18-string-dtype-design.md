# zarrista — variable-length string dtype

**Date:** 2026-06-18
**Status:** Approved

## Goal

Let zarrista read Zarr arrays whose dtype is **variable-length UTF-8 string**,
returning them as a numpy 2 `StringDType` array via `Data.to_numpy()`. This is the
first slice of Tier 1, item 3 of the [API roadmap](2026-06-16-api-roadmap.md)
("complete the dtype story"). Fixed-width raw bytes, complex, and fixed UTF-32 are
explicitly deferred to follow-up tasks.

## Guiding principle (settled in brainstorming)

**Rust produces the decoded payload; Python owns numpy-dtype construction.** The
buffer protocol carries *bytes*, not type semantics, for anything richer than a
buffer-native scalar. For variable-length strings this is not just a preference —
it is forced: rust-numpy has no safe API for numpy 2 `StringDType`
([PyO3/rust-numpy#505](https://github.com/PyO3/rust-numpy/issues/505)). So the
numpy array is always built on the Python side, and Rust never touches
`StringDType`.

## Dtype categories

This frames the broader "dtype story" so the string work slots in cleanly. Only
**category 3** is implemented in this round.

| Category | Members | Held as | Buffer protocol | `to_numpy` |
|---|---|---|---|---|
| 1. Buffer-native | existing 12 numerics | `ArrayD<T>` | typed, strided (unchanged) | `np.asarray(self)` — zero-copy typed view |
| 2. Raw fixed-width | `r*N`, complex, UTF-32 (future) | bytes + numpy-dtype string + shape | flat `B` buffer | `np.frombuffer(self, dt).reshape(shape)` |
| 3. Variable-length | **`string` (this round)**, bytes (future) | `ArrayD<String>` | none | build `list[str]` → `np.array(lst, StringDType()).reshape(shape)` |

Category 2 infrastructure is **not** built in this round — string needs none of it.

## What zarrs gives us

- `DataType::String` is `StringDataType`; `dtype.is::<StringDataType>()` identifies it.
- `String: ElementOwned` — zarrs does the UTF-8 decode for us.
- With the `ndarray` feature (already enabled), `retrieve_array_subset_ndarray::<String>()`
  and `retrieve_chunk_ndarray::<String>()` return `ArrayD<String>`.
- The `vlen_utf8` codec (and `vlen`, `vlen_v2`) is implemented in zarrs, so it can
  decode what zarr-python writes for a string array.
- Edge/missing chunks are filled with the array's string fill value automatically
  during retrieval — no special handling here.

## Changes

### `src/data.rs` — `DataInner` and `PyData`

1. **Add a variant:** `DataInner::String(ArrayD<String>)`.
2. **`with_array!` gains a `String($a) => $body` arm.** This compiles because the
   only remaining `with_array!` call-site bodies are `a.shape()`, `a.strides()`, and
   `a.as_ptr()` — all valid for `ArrayD<String>`. (See deletion below for why no
   body needs `numpy::Element`.)
3. **Delete `to_numpy_with_copy`.** It is the `buffer_format == None` fallback and is
   currently dead (all 12 numerics have a format). String is handled by an explicit
   branch in `to_numpy` instead, so the fallback — the one body that needed
   `PyArray::from_array` (and thus `numpy::Element`, which `String` is not) — is
   removed. This is what lets `with_array!` accept a `String` arm.
4. **`buffer_format`:** add `String(_) => None`. `__getbuffer__` already rejects the
   `None` case with `PyBufferError` ("no buffer-protocol representation") — strings
   are not buffer-exportable, by design.
5. **`itemsize` / `data_ptr`:** add `String(_)` arms. These are only reached from the
   buffer-protocol path, which `String` never enters, so the arms exist solely for
   match exhaustiveness (`itemsize` may return the size of a `String` element;
   `data_ptr` returns the `ArrayD<String>` pointer). The stored `strides` are unused
   for strings.
6. **`to_numpy` becomes a 2-way branch:**
   - `DataInner::String(arr)` → build the numpy array (below).
   - everything else → `np.asarray(self)` (zero-copy typed view, unchanged).

   The `StringDType` builder: import `numpy`, construct `np.dtypes.StringDType()`,
   build a flat `list[str]` by iterating `arr` in C-order (`arr.iter()` follows
   zarrs's C-order layout), then
   `np.array(flat_list, dtype=string_dtype).reshape(arr.shape())`. Empty arrays
   (a zero-length axis) reshape correctly from an empty list.

### `src/array/sync.rs` and `src/array/async.rs` — read dispatch

In both `retrieve_array_subset` and `retrieve_chunk`, after the existing
`for_each_dtype!` macro loop, add an explicit string arm:

```rust
if dtype.is::<StringDataType>() {
    let data = self.inner.retrieve_array_subset_ndarray::<String>(&array_subset)?;
    return Ok(PyData::from(DataInner::String(data)));
}
```

(and the `retrieve_chunk_ndarray::<String>` analogue for `retrieve_chunk`). On
`AsyncArray` the same arm uses the async method names —
`async_retrieve_array_subset_ndarray::<String>` and
`async_retrieve_chunk_ndarray::<String>` — matching how the existing numeric arms
call `async_retrieve_array_subset::<ArrayD<$elem>>`. The trailing
`NotImplementedError` for unsupported dtypes stays as the fallback for everything
still unimplemented.

## Testing

- **Python round-trip vs. zarr-python** (extends the harness seeded by the indexing
  work): write variable-length string arrays with zarr-python across a few
  shapes / chunkings, including a partial-edge-chunk case, read them back with
  zarrista's `FilesystemStore`, and assert `Data.to_numpy()` equals the zarr-python
  array. Cover both `retrieve_array_subset`/`__getitem__` and `retrieve_chunk`.
  Include an array with an empty selection (zero-length axis).
- **Rust unit test:** a small `ArrayD<String>` → `DataInner::String` → `to_numpy`
  check is hard without a Python interpreter; rely on the Python round-trip for
  end-to-end coverage and keep any Rust-side test limited to construction.
- **Tooling:** rebuild with `maturin develop` after Rust changes; run tests with
  `uv run --no-project pytest` so uv does not rebuild on every invocation.

## Risks

- **zarr-python's string encoding.** The round-trip assumes zarr-python emits a
  string array zarrs can decode via `vlen_utf8`/`vlen`. If zarr-python uses an
  encoding zarrs does not recognize, the round-trip test will surface it; resolving
  any codec mismatch is part of this task.

## Out of scope (deferred)

- **Fixed-width raw bytes** (`r*N` → `|V<n>`) and the category-2 `frombuffer`
  infrastructure — next task.
- **complex64/128 and fixed UTF-32** — the "fold in" follow-up task.
- **bfloat16** (drags in an `ml_dtypes` runtime dependency) and **variable-length
  bytes** (numpy object array, different semantics).
- **String fill-value scalar exposure** (`fill_value_to_py` / `Array.fill_value`) —
  the separate roadmap item 2.
- **Writing** string arrays.
