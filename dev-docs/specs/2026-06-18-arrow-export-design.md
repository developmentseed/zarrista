# zarrista — Arrow export face for `Data`

**Date:** 2026-06-18
**Status:** Approved

## Framing: `Data` is format-neutral

`Data` is a **format-neutral decoded payload**. Consumers reach it through thin,
**co-equal export faces** — today the buffer protocol (`to_numpy`), with this spec
adding **Arrow** (the PyCapsule interface), and **DLPack** a likely future face. No
core decision about `Data` is made to serve one format; each face adapts to `Data`,
not the reverse.

This matters because **no single interchange format covers all Zarr dtypes** (see
the coverage matrix below). Arrow uniquely expresses variable-length and nested data
the buffer protocol can't; DLPack uniquely expresses exotic numerics (complex,
bfloat16) Arrow can't; the buffer protocol is the universal fixed-width baseline. So
several faces are a necessity, not a luxury — and none is privileged.

Arrow specifically is an **additive, exploratory** face: it is not a strong pull in
the zarr-nd community, and it is not the organizing principle of the library. Its
concrete future payoff is the **zero-copy variable-length** path (strings/bytes); for
the current fixed-width dtypes it is a convenience alongside the buffer protocol.

## Goal

Implement the Arrow PyCapsule interface (`__arrow_c_array__` / `__arrow_c_schema__`)
on `Data` so a decoded chunk/subset can be handed to pyarrow, polars, arro3,
datafusion, etc. This round covers the **fixed-width numeric dtypes `Data` already
supports**, and designs — without building — the variable-length path so strings and
bytes can re-enter as its first consumers later.

## Coverage matrix (why several faces)

The full zarrs dtype set is: bool; int 8/16/32/64 + int2/int4; uint 8/16/32/64 +
uint2/uint4; float 16/32/64 + subfloat (float8); complex64/128; raw_bits;
fixed_length_utf32; string; bytes; datetime64 / timedelta64. Notably there is **no
struct/compound and no dictionary/categorical dtype** — Zarr v2's compound dtypes are
only a v3 *extension* point (zarrs models the closest, opaque case as `raw_bits` =
numpy `void`), and "categorical" in Zarr is a *codec/filter* (`numcodecs.Categorize`)
that yields an integer array, not a dtype. So Arrow's nested/`Struct`/`Dictionary`
strengths have nothing to map *from* in Zarr today.

| Zarr dtype | buffer protocol | Arrow | DLPack (future) |
|---|---|---|---|
| int / uint / float32-64 | ✅ | ✅ | ✅ |
| float16 | ✅ | ✅ `Float16` | ✅ |
| bool | ✅ (1 byte) | ✅ `arrow.bool8` | ✅ |
| datetime64 / timedelta64 | ✅ (as int64) | ⚠️ `Timestamp`/`Duration` — zero-copy only for s/ms/us/ns without NaT (see note) | ✅ |
| variable-length string/bytes | ❌ | ✅ `String`/`Binary` | ❌ |
| complex64/128 | ✅ (2 floats) | ❌ no native complex | ✅ |
| subfloat (float8) / bfloat16 | ⚠️ no PEP-3118 code | ❌ | ✅ |
| int2/4, uint2/4 (sub-byte) | ❌ | ❌ | ❌ (awkward everywhere) |

For Zarr's actual dtypes, Arrow's unique value is **variable-length string/bytes**,
plus a **semantic** bonus on the temporal types; DLPack uniquely covers the exotic
numerics (complex, float8/bfloat16); the buffer protocol is the universal fixed-width
baseline. This spec implements the Arrow column for the fixed-width numeric rows.

## Decisions (settled in brainstorming)

- **Side-channel shape.** Arrow arrays are logically 1-D. `Data` exposes its data as
  a **flat, length-`prod(shape)` Arrow array**, plus a public **`Data.shape`**
  property carrying the N-D shape. Consumers reshape if they care. We deliberately do
  **not** use the `arrow.fixed_shape_tensor` extension type — its semantics are
  "batch of tensors" (leading dim = batch), an awkward fit for a single chunk.
- **pyo3-arrow.** Use [`pyo3-arrow`](https://crates.io/crates/pyo3-arrow) to build the
  Arrow arrays and expose the PyCapsules. No pyarrow dependency in Rust.
- **`bool` via `arrow.bool8`, zero-copy.** Use the
  [`arrow.bool8` canonical extension](https://arrow.apache.org/docs/format/CanonicalExtensions.html#bit-boolean)
  (int8 storage, 0=false / nonzero=true) — exactly our 1-byte-per-bool in-memory
  layout. So bool is **zero-copy** like every other fixed-width dtype; no bit-packing,
  no exception. Consumers that don't understand the extension degrade gracefully to
  the int8 storage.
- **Contiguity is NOT a core invariant** of `Data` (see below).
- **Strings on hold.** The variable-length dtype work is paused; it re-enters as the
  first consumer of the Arrow variable-length path designed below.
- **DLPack is a future, co-equal face** (see below).

## Strides and contiguity — not a `Data` constraint

Arrow primitive arrays have no strides: a buffer is contiguous or it is copied. But
**requiring `Data` to be contiguous would privilege Arrow** and could collide with a
future `retrieve_*_into` that decodes directly into a strided destination (a dask
block, a sub-region of a larger output). So `Data` makes **no contiguity guarantee**.

Instead, each face adapts:

- **Buffer protocol** emits strides (already does) — handles any layout.
- **DLPack** (future) carries shape + strides natively — handles any layout.
- **Arrow** is the only stride-intolerant face, so **Arrow alone** compacts to
  contiguous *when needed*, paying that cost itself: in `__arrow_c_array__`, if the
  backing array `is_standard_layout()` wrap it zero-copy; otherwise materialize a
  contiguous copy (`as_standard_layout()`) for the export.

Today every `Data` is a fresh, owned, C-order `retrieve_*_ndarray`, so the contiguous
branch always hits and Arrow export is **zero-copy in practice today** — but the type
does not forbid strided `Data`, so we are not boxed in.

(Native endianness: zarrs decodes to native endianness; Arrow's C Data Interface
requires little-endian, which coincides on all targets (x86-64, aarch64). Big-endian
hosts are out of scope.)

## Zero-copy mechanism

For a contiguous fixed-width dtype, build an arrow-rs array over `Data`'s existing
buffer without copying via `Buffer::from_custom_allocation`: wrap the raw
`(ptr, len)` with a release callback owning a `Py<Data>` reference. This is the same
lifetime trick `__getbuffer__` already uses — the Arrow array's release callback
drops the `Py<Data>` when the consumer is done, keeping the buffer alive exactly as
long as the exported array. pyo3-arrow wraps the result in the PyCapsule.

## Dtype mapping (fixed-width, this round)

| `Data` dtype | Arrow type |
|---|---|
| int8/16/32/64 | `Int8/16/32/64` |
| uint8/16/32/64 | `UInt8/16/32/64` |
| float32/64 | `Float32/64` |
| float16 | `Float16` |
| bool | `Int8` + `arrow.bool8` extension |

All zero-copy when contiguous (the current reality).

## API surface

On `Data`:

```python
data.__arrow_c_array__(requested_schema=None) -> (schema_capsule, array_capsule)
data.__arrow_c_schema__() -> schema_capsule
data.shape -> tuple[int, ...]   # N-D shape; the Arrow array is flat length prod(shape)

# zero-copy introspection (per face)
data.contiguous -> bool             # backing buffer is C-contiguous (strided = not this)
data.arrow_copy -> bool             # will Arrow export copy the bulk data?
data.buffer_protocol_copy -> bool   # will to_numpy / buffer protocol copy?
```

`pa.array(data)`, `pl.Series(data)`, `arro3.core.Array.from_arrow(data)` all work via
the capsule protocol. To recover N-D structure, a consumer combines the flat Arrow
array with `data.shape`. `Data.shape` is promoted from the internal `shape` field
(already stored for the buffer protocol) to a public, documented property.

### Zero-copy introspection

Each face's copy cost is made *visible* rather than folklore, so a consumer can check
before a large export:

| getter | meaning | this round (fixed-width) |
|---|---|---|
| `contiguous` | backing buffer is C-contiguous | usually `True` (fresh retrievals) |
| `arrow_copy` | Arrow export copies the bulk data | `not contiguous` |
| `buffer_protocol_copy` | `to_numpy` / buffer protocol copies | always `False` |

`buffer_protocol_copy` becomes `True` only for the future variable-length dtypes that
have no buffer-protocol representation. For variable-length dtypes, `arrow_copy`
reflects the **bulk/values** data (zero-copy); the small offsets array is always
copied regardless.

## Variable-length, designed not built

The motivating future win; must not be foreclosed. zarrs hands back variable-length
data as **(values blob, offsets)** — exactly Arrow `String`/`LargeString` (UTF-8) and
`Binary`/`LargeBinary`. Future path:

- New `DataInner` variant(s) holding `(values: bytes, offsets, shape)` from a
  variable-length `retrieve_*::<ArrayBytes>()`.
- `__arrow_c_array__` wraps the values buffer **zero-copy** and the offsets buffer
  (converting zarrs `usize` offsets to Arrow `i32`/`i64` — a cheap copy of the small
  offsets array, not the data).
- These variants have **no buffer-protocol representation**, so Arrow becomes their
  primary zero-copy face — the whole point.

Keep the dtype dispatch and `DataInner` open to non-`ArrayD<T>` variants so this
slots in without restructuring.

## DLPack (future, co-equal face)

`__dlpack__` / `__dlpack_device__` is the other zero-copy face worth adding, and a
co-equal one — not subordinate to Arrow. It carries **ND shape and strides natively**
(so it needs no `shape` side-channel and tolerates non-contiguous `Data`) and has
**dtype codes Arrow lacks** (complex, bfloat16), making it the natural face for the
exotic-numeric and GPU/torch/jax interchange cases. It is lower-level C-struct work
(`DLManagedTensor`) with no pyo3-arrow-equivalent ergonomics, so it is deferred and
noted here only to keep the multi-face design coherent.

## Testing

- **Round-trip vs. zarr-python** (extends the existing harness): write numeric arrays
  with zarr-python, read with zarrista, assert the Arrow export matches — e.g.
  `pa.array(data)` (or arro3) reshaped via `data.shape` equals the zarr-python numpy
  array. Cover every numeric dtype and `bool` (verifying `arrow.bool8` round-trips),
  plus a multi-dim chunk to exercise flat-array + `shape`.
- **Zero-copy assertion:** confirm the Arrow buffer pointer equals the `Data` buffer
  pointer for a non-bool contiguous dtype, and that keeping the Arrow array alive
  after dropping the Python `Data` reference is safe (release-callback lifetime
  holds).
- **Tooling:** `maturin develop` after Rust changes; `uv run --no-project pytest`.

## Out of scope (deferred)

- Variable-length `String`/`Binary` export (designed above; lands with the
  string/bytes dtype work).
- DLPack export.
- Arrow *import* / writing.
- complex / float8 / bfloat16 dtypes (no Arrow representation — DLPack territory).
- temporal dtypes (datetime64 / timedelta64 → Arrow `Timestamp`/`Duration`) — a
  natural Arrow follow-up, but not part of this fixed-width-numeric round, and **not
  uniformly zero-copy**: both are int64/LE/epoch-based so the values buffer matches,
  but (1) Arrow supports only s/ms/us/ns — calendar/other units (D/h/m/W/M/Y/sub-ns)
  need a unit cast and W/M/Y have no faithful Arrow representation; and (2) numpy
  encodes NaT as the `INT64_MIN` sentinel while Arrow uses a validity bitmap, so any
  NaT (a valid Zarr fill value) forces an O(n) scan to build a bitmap. Zero-copy only
  holds for unit ∈ {s, ms, us, ns} with no NaT.

## Risks

- **Buffer alignment.** Arrow *recommends* (does not require) 64-byte buffer
  alignment; the C Data Interface treats it as advisory with an ~8-byte floor. Our
  buffer is an ndarray `Vec<T>`, aligned only to `align_of::<T>()` (8 for `f64`), so
  `from_custom_allocation` hands Arrow a buffer that meets the floor but is almost
  never 64-byte aligned. **Correctness is unaffected**; the cost is that a SIMD-heavy
  kernel or strict validator (some pyarrow paths) may silently **re-copy to realign**,
  turning our zero-copy export into one consumer-side copy. `to_numpy` is unaffected
  (the buffer protocol has no alignment requirement). Fallback if it bites: allocate
  the decode buffer 64-byte aligned up front rather than reusing the `Vec`.
- **`arrow.bool8` consumer support.** It is a relatively recent canonical extension;
  older consumers see the int8 storage rather than a boolean. Acceptable (graceful
  degradation), but worth noting.
- **pyo3-arrow / arrow-rs version coupling.** Pin deliberately.
