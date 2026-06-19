# zarrista — Arrow export faces for the decoded result types

**Date:** 2026-06-18
**Status:** Approved (architecture realigned to the
[decoded result types](2026-06-18-data-result-types-design.md) spike)

## Framing: Arrow is one face on the result types

Reads no longer return a single `Data` object. Per
[decoded result types](2026-06-18-data-result-types-design.md), a read returns one
of **four concrete classes** built from the raw post-codec `ArrayBytes`, each
holding `bytes::Bytes` (refcounted, zero-copy):

| class | `ArrayBytes` layout | payload held |
|---|---|---|
| `Tensor` | `Fixed` | values |
| `VariableArray` | `Variable` | values + offsets |
| `MaskedTensor` | `Optional(Fixed)` | values + mask |
| `MaskedVariableArray` | `Optional(Variable)` | values + offsets + mask |

These are **format-neutral payloads**; consumers reach them through thin,
**co-equal export faces** — today the buffer protocol (`to_numpy`/`buffer()` on
`Tensor`), with this spec adding **Arrow** (the PyCapsule interface), and **DLPack**
a likely future face. No decision about the result types is made to serve one
format; each face adapts to the payload, not the reverse.

This matters because **no single interchange format covers all Zarr dtypes** (see
the coverage matrix). Arrow uniquely expresses the variable-length and masked
layouts the buffer protocol can't; DLPack uniquely expresses exotic numerics
(complex, bfloat16) Arrow can't; the buffer protocol is the universal fixed-width
baseline. Several faces are a necessity, not a luxury — and none is privileged.

Arrow specifically is **additive and exploratory**: not a strong pull in the
zarr-nd community, not the organizing principle of the library. Its concrete payoff
is the **zero-copy variable-length / masked** path, where the buffer protocol simply
cannot represent the data.

## Goal

Implement the Arrow PyCapsule interface (`__arrow_c_array__` / `__arrow_c_schema__`)
on the result classes so a decoded chunk/subset can be handed to pyarrow, polars,
arro3, datafusion, etc. **`Tensor` first** (the dtypes that already round-trip);
`VariableArray` and the masked classes follow as their dtypes land, but their Arrow
mapping is designed here because they're where Arrow earns its keep.

## Coverage matrix (why several faces)

The full zarrs dtype set is: bool; int 8/16/32/64 + int2/int4; uint 8/16/32/64 +
uint2/uint4; float 16/32/64 + subfloat (float8); complex64/128; raw_bits;
fixed_length_utf32; string; bytes; datetime64 / timedelta64. There is **no
struct/compound and no dictionary/categorical dtype** — Zarr v2's compound dtypes are
only a v3 *extension* point (zarrs models the closest, opaque case as `raw_bits` =
numpy `void`), and "categorical" in Zarr is a *codec/filter* (`numcodecs.Categorize`)
yielding an integer array, not a dtype. So Arrow's nested/`Struct`/`Dictionary`
strengths have nothing to map *from* in Zarr today.

| Zarr dtype | buffer protocol | Arrow | DLPack (future) |
|---|---|---|---|
| int / uint / float32-64 | ✅ | ✅ | ✅ |
| float16 | ✅ | ✅ `Float16` | ✅ |
| bool | ✅ (1 byte) | ✅ `arrow.bool8` | ✅ |
| datetime64 / timedelta64 | ✅ (as int64) | ⚠️ `Timestamp`/`Duration` — zero-copy only for s/ms/us/ns without NaT (see note) | ✅ |
| variable-length string/bytes | ❌ | ✅ `Utf8`/`Binary` | ❌ |
| complex64/128 | ✅ (2 floats) | ❌ no native complex | ✅ |
| subfloat (float8) / bfloat16 | ⚠️ no PEP-3118 code | ❌ | ✅ |
| int2/4, uint2/4 (sub-byte) | ❌ | ❌ | ❌ (awkward everywhere) |

For Zarr's actual dtypes, Arrow's unique value is **variable-length string/bytes** and
**masked** data, plus a **semantic** bonus on the temporal types; DLPack uniquely
covers the exotic numerics; the buffer protocol is the universal fixed-width baseline.

## Per-class Arrow mapping

The result class (set by the `ArrayBytes` layout) determines the Arrow array kind;
the dtype refines it. Values are always **zero-copy** (a `bytes::Bytes` clone);
only the small side buffers (offsets, validity) are ever copied.

| class | Arrow array | values | offsets | validity |
|---|---|---|---|---|
| `Tensor` | primitive (+ `arrow.bool8` for bool) | zero-copy | — | — |
| `VariableArray` | `Utf8`/`LargeUtf8` (string) or `Binary`/`LargeBinary` (bytes) | zero-copy | `usize` → `i32`/`i64` (copy) | — |
| `MaskedTensor` | primitive **+ validity bitmap** | zero-copy | — | byte-mask → bitmap (copy) |
| `MaskedVariableArray` | `Utf8`/`Binary` **+ validity bitmap** | zero-copy | `usize` → `i32`/`i64` (copy) | byte-mask → bitmap (copy) |

Two layout conversions are intrinsic to Arrow and unavoidable (but cheap — side
buffers, not bulk data):

- **Offsets.** zarrs hands variable-length offsets as `usize` (currently materialized
  to `Vec<usize>`; zarrs#406 tracks avoiding that). Arrow needs `i32` (`Utf8`/`Binary`)
  or `i64` (`LargeUtf8`/`LargeBinary`). Pick the `Large` variant when the last offset
  exceeds `i32::MAX`, else `i32`; either way it's a copy of the small offsets array.
- **Validity.** zarrs `Optional` carries a mask of **1 byte per element** (0 = invalid,
  non-zero = valid). Arrow validity is **1 bit per element** (1 = valid). So masked
  export **bit-packs** the byte-mask into a validity bitmap — `bit[i] = mask[i] != 0` —
  a small copy, and derives `null_count`. (This is the inverse of the `bool8` case,
  where Arrow happens to match our byte layout and *no* packing is needed.)

### `bool` via `arrow.bool8` (zero-copy)

`Tensor` of `bool` uses the
[`arrow.bool8` canonical extension](https://arrow.apache.org/docs/format/CanonicalExtensions.html#bit-boolean)
(int8 storage, 0=false / non-zero=true) — exactly our 1-byte-per-bool layout. So bool
values export **zero-copy**, no bit-packing. Consumers that don't understand the
extension degrade gracefully to the int8 storage. (Note the asymmetry: a `bool`
*value* column needs no packing via `bool8`, but a *validity mask* always bit-packs,
because Arrow validity is defined as a bitmap.)

### `Tensor` dtype mapping (first round)

| dtype | Arrow type |
|---|---|
| int8/16/32/64 | `Int8/16/32/64` |
| uint8/16/32/64 | `UInt8/16/32/64` |
| float32/64 | `Float32/64` |
| float16 | `Float16` |
| bool | `Int8` + `arrow.bool8` |

## Zero-copy mechanism (simpler now, thanks to `bytes::Bytes`)

Because the payload is `bytes::Bytes` (refcounted), the values buffer needs **no
`Py_INCREF`/release-callback dance**. Build the arrow-rs `Buffer` over the bytes via
`Buffer::from_custom_allocation(ptr, len, owner)`, where `owner` is a cloned
`bytes::Bytes` (an `Arc`-like handle) — the Arrow buffer keeps the allocation alive by
holding its own refcount, dropped when the consumer releases the Arrow array.
`pyo3-arrow` wraps the resulting array in the PyCapsule. One allocation backs the
buffer protocol, `to_numpy`, and Arrow simultaneously via cheap `Bytes` clones.

## Contiguity & alignment

The values payload is a single contiguous `bytes::Bytes` blob in C-order — so unlike
the old `ArrayD<T>` design there is **no stride/non-contiguity case to handle**;
Arrow's contiguity requirement is always satisfied for the values buffer.

The live concern is **alignment**, and the decision (from the result-types spec) is
**stay unaligned**: the bytes come straight from the decode allocator, contractually
1-byte aligned (de-facto 16 from malloc). Consequences for Arrow:

- **Correctness:** unaffected. Arrow's C Data Interface treats 64-byte alignment as
  *advisory* (≈8-byte floor).
- **Performance:** a SIMD kernel or strict validator (some pyarrow paths) may silently
  **re-copy to realign**, turning a zero-copy export into one consumer-side copy.
- **Owning consumers** (`pa.array(...)` that materializes, compute kernels) pay any
  realign copy as part of work they were already doing — we never pay a *dedicated*
  alignment copy.

If it ever bites, expose an explicit aligned-copy path rather than aligning every
read (the result-types spec's lazy-alignment stance).

## Side-channel shape

Arrow arrays are logically 1-D. Each result class exposes its data as a **flat,
length-`prod(shape)` Arrow array** plus its existing **`shape`** property; consumers
reshape if they care. We deliberately do **not** use `arrow.fixed_shape_tensor` (its
semantics are "batch of tensors", leading dim = batch — wrong for a single chunk).

## API surface

On each result class (`Tensor` first):

```python
obj.__arrow_c_array__(requested_schema=None) -> (schema_capsule, array_capsule)
obj.__arrow_c_schema__() -> schema_capsule
obj.shape  # already present; the Arrow array is flat length prod(shape)
```

`pa.array(obj)`, `pl.Series(obj)`, `arro3.core.Array.from_arrow(obj)` all work via the
capsule protocol; recover N-D structure by combining with `obj.shape`.

**Optional introspection** (revisit when implementing): a per-class `arrow_copy: bool`
could advertise whether an export copies bulk data. With the `bytes::Bytes` design the
values are always zero-copy, so this would only ever flag alignment realign risk or
the (always-copied) side buffers — lower value than under the old design, so it's
deferred rather than specified.

## Tooling

Use [`pyo3-arrow`](https://crates.io/crates/pyo3-arrow) to build the Arrow arrays and
expose the capsules — no pyarrow dependency in Rust. (Separately, `dlpark`'s pyo3
feature pins pyo3 0.25 vs our 0.29 — see the result-types spec — but that's a DLPack
concern, not Arrow.)

## Testing

- **Round-trip vs. zarr-python** (extends the existing harness): write numeric arrays,
  read with zarrista, assert the Arrow export of `Tensor` matches — e.g.
  `pa.array(tensor)` (or arro3) reshaped via `tensor.shape` equals the zarr-python
  numpy array. Cover every numeric dtype and `bool` (verifying `arrow.bool8`), plus a
  multi-dim chunk for flat-array + `shape`.
- **Zero-copy assertion:** the Arrow values buffer pointer equals `tensor.buffer()`'s
  pointer (no copy), and the Arrow array outlives a dropped Python reference (the
  `Bytes` refcount holds).
- **When variable/masked land:** assert `VariableArray` → `Utf8`/`Binary` with correct
  offsets, and masked → correct validity bitmap + `null_count` from the byte-mask.
- **Tooling:** `maturin develop`; `uv run --no-project pytest`.

## Out of scope (deferred)

- `VariableArray` / masked Arrow export (designed above; lands with the
  variable-length and nullable dtype work).
- DLPack export (co-equal future face; covers complex/bfloat16 Arrow can't).
- Arrow *import* / writing.
- complex / float8 / bfloat16 dtypes (no Arrow representation — DLPack territory).
- temporal dtypes (datetime64 / timedelta64 → `Timestamp`/`Duration`) — a natural
  Arrow follow-up, but **not uniformly zero-copy**: both are int64/LE/epoch-based so
  the values buffer matches, but (1) Arrow supports only s/ms/us/ns — calendar/other
  units (D/h/m/W/M/Y/sub-ns) need a unit cast and W/M/Y have no faithful Arrow
  representation; and (2) numpy encodes NaT as the `INT64_MIN` sentinel while Arrow
  uses a validity bitmap, so any NaT (a valid Zarr fill value) forces an O(n) scan to
  build a bitmap. Zero-copy only holds for unit ∈ {s, ms, us, ns} with no NaT.

## Risks

- **Buffer alignment.** As above: correctness fine, possible silent consumer-side
  realign copy; `to_numpy` unaffected (buffer protocol has no alignment requirement).
- **`arrow.bool8` consumer support.** A relatively recent canonical extension; older
  consumers see int8 storage rather than boolean. Acceptable (graceful degradation).
- **`pyo3-arrow` / arrow-rs version coupling.** Pin deliberately.
