# Tensor ND buffer protocol design

**Date:** 2026-07-08
**Status:** Approved, not yet implemented

## Goal

Make `Tensor` (`PyTensor`) itself a first-class, N-dimensional, typed
buffer-protocol (PEP 3118) object. Today the raw decoded bytes are only reachable
as a flat `u8` buffer via `.buffer()`, and a shaped view requires the
`to_numpy()` helper (`np.frombuffer(...).reshape(...)`). After this change,
`memoryview(tensor)`, `np.asarray(tensor)`, and any other buffer consumer get the
correctly shaped and dtyped, zero-copy, read-only view directly.

Non-goals: writable buffers, changing `MaskedTensor`/`VariableArray`, changing
the DLPack export.

## Current state

`PyTensor` (`src/data/tensor.rs`) holds:

- `bytes: Bytes` — `Arc`-backed, so a zero-copy export can keep the allocation
  alive by holding a reference.
- `data_type: DataType`
- `shape: Vec<u64>`

Existing surface:

- `.buffer() -> PyBytes` — flat `u8` buffer-protocol object (via `pyo3_bytes`).
- `.to_numpy()` — `np.frombuffer(self.buffer(), name_v3).reshape(shape)`.
- `__dlpack__` / `__dlpack_device__` — DLPack export (supports bf16).
- `data_type_to_dlpack(&DataType) -> ZarristaResult<dlpack::ffi::DataType>` — the
  existing dtype mapping used by DLPack.

## Design

### 1. New mapping: `data_type_to_format`

A function mirroring `data_type_to_dlpack`, returning a native-endian PEP 3118
struct format code (`&'static str`). Bare codes denote native byte order, which
matches the assumption `np.frombuffer` already makes today.

| dtype    | code | dtype    | code |
| -------- | ---- | -------- | ---- |
| bool     | `?`  | uint8    | `B`  |
| int8     | `b`  | uint16   | `H`  |
| int16    | `h`  | uint32   | `I`  |
| int32    | `i`  | uint64   | `Q`  |
| int64    | `q`  | float16  | `e`  |
| float32  | `f`  | float64  | `d`  |
| bfloat16 | — (unsupported) | | |

Any dtype not in the table (notably `bfloat16`, and any complex/extension type)
→ `BufferError`. This keeps the buffer view honest: it never misrepresents the
element type.

`itemsize` comes from `data_type.fixed_size()`.

### 2. `__getbuffer__` / `__releasebuffer__`

Implement pyo3's buffer-protocol hooks on `PyTensor`:

```rust
unsafe fn __getbuffer__(
    slf: Bound<'_, Self>,
    view: *mut ffi::Py_buffer,
    flags: c_int,
) -> PyResult<()>;

unsafe fn __releasebuffer__(&self, view: *mut ffi::Py_buffer);
```

**Behavior (strict / spec-compliant):**

- **Writability:** if `flags & PyBUF_WRITABLE`, raise `BufferError` (the backing
  `Bytes` is immutable). Always export `readonly = 1`.
- **Data / lifetime:** `buf` = `bytes.as_ptr()`; `len` = `bytes.len()`;
  `itemsize` = element size. Set `view.obj` to a new owned reference to `slf`
  (`Py_INCREF` semantics) so the `Arc`-backed allocation outlives the view.
  `PyBuffer_Release` decrements it and triggers `__releasebuffer__`.
- **Format:** only fill `format` when `flags & PyBUF_FORMAT`; the format string
  comes from `data_type_to_format` (→ `BufferError` if unsupported). Resolve the
  format *before* taking ownership / allocating so an unsupported dtype fails
  cleanly with nothing to unwind.
- **Shape / strides:** only fill `shape`/`strides` when requested by `flags`.
  `ndim = shape.len()`. Strides are C-contiguous row-major, in bytes:
  `strides[ndim-1] = itemsize`, `strides[i] = strides[i+1] * shape[i+1]`.
  Handle the 0-d (scalar) case: `ndim = 0`, `shape`/`strides` null.
- **suboffsets:** null.

**Memory management for `shape`/`strides`:** `Py_buffer.shape` and
`Py_buffer.strides` must point at `Py_ssize_t` (`isize`) arrays that stay valid
until `__releasebuffer__`. `self.shape` is `Vec<u64>`, so it cannot be pointed at
directly. Allocate owned `isize` arrays in `__getbuffer__`, stash them behind a
single boxed struct stored in `view.internal`, point `view.shape`/`view.strides`
at them, and in `__releasebuffer__` reconstruct the `Box` from `view.internal`
and drop it.

### 3. `to_numpy` stays as-is (buffer protocol is purely additive)

`to_numpy()` is **not** changed. It keeps its `name_v3` / `frombuffer` / `reshape`
path.

Rationale: numpy's dtype system does not line up 1:1 with PEP 3118 format codes.
numpy natively represents dtypes that have no (or only numpy-specific) buffer
format code — notably `complex64` / `complex128`, which currently round-trip
through `to_numpy()` and work. Re-routing `to_numpy()` through the buffer protocol
would regress those dtypes to a `BufferError`. The two paths serve genuinely
different consumers (a numpy-name mapping vs. a universal byte-view format code),
so they stay as separate methods.

Consequence: two dtype→string mappings coexist (`name_v3` for numpy,
`data_type_to_format` for the buffer protocol). This is intentional, not
duplication to eliminate.

`.buffer()` (flat `u8`) and `__dlpack__` are likewise unchanged.

## Error handling

- Unsupported dtype → `BufferError`.
- Writable request (`PyBUF_WRITABLE`) → `BufferError`.
- Consumers needing bytes for an unsupported dtype fall back to `.buffer()` (raw
  `u8`); numpy access always remains available via `to_numpy()` (which handles
  the wider numpy dtype set, e.g. complex), and bf16 via `__dlpack__`.

## Testing (Python)

- `memoryview(tensor)` reports correct `shape`, `format`, `itemsize`, `ndim`,
  `readonly`, and byte contents.
- Multi-dimensional tensor: strides are correct C-contiguous row-major.
- `np.asarray(tensor)` round-trips to the expected array for a representative set
  of dtypes (int/uint widths, float16/32/64, bool).
- Unsupported-for-buffer dtype (`complex64`): `memoryview(tensor)` raises
  `BufferError`, while `to_numpy()` still returns the correct `complex64` array
  (proving the buffer protocol is additive and did not regress numpy access).
  `complex64` is written natively by numpy/zarr, so no `ml_dtypes` dependency is
  needed.
- Requesting a writable buffer raises `BufferError`.
- `.buffer()` still returns a flat `u8` buffer of the right length.
- `to_numpy()` still returns the correct shaped array for supported dtypes.

## Docs

- `python/zarrista/_decoded_array.pyi`: update the `Tensor` class docstring to
  note it directly supports the buffer protocol (usable with `memoryview` /
  `np.asarray`); update the `to_numpy` docstring to reference `np.asarray`.
