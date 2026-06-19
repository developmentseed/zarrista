# zarrista — decoded result types (ArrayBytes-backed `Data`)

**Date:** 2026-06-18
**Status:** Prototyping (spike on branch `kyle/dlpack-tensor-spike`)

## Goal

Reorient what a read returns. Today a read decodes into a typed
`ndarray::ArrayD<T>` wrapped in a single `Data` class, dispatched by a 12-arm
per-dtype macro. Replace that with a **format-neutral, zero-copy** payload built
from the raw post-codec [`ArrayBytes`], surfaced as **four concrete Python result
classes**. This collapses the dispatch to a single generic retrieval and sets up
the multi-protocol faces (buffer protocol now; Arrow, DLPack later — see
[arrow-export-design](2026-06-18-arrow-export-design.md)).

## Why (key findings that drove this)

- **The typed path copies *every* buffer.** `retrieve::<ArrayD<T>>` /
  `::<Vec<T>>` go through `convert_from_bytes_slice` →
  `bytemuck::allocation::pod_collect_to_vec`, which **unconditionally** allocates
  a fresh aligned `Vec<T>` and `copy_from_slice`s into it — no alignment fast
  path. So today's `Data` pays a full copy per read. Retrieving raw `ArrayBytes`
  avoids it.
- **`FromArrayBytes` is post-codec.** All retrieval funnels through
  `retrieve_array_subset_opt`, which runs the *entire* codec pipeline and then
  calls `T::from_array_bytes(bytes, shape, data_type)`. Implementing our own
  `FromArrayBytes` customizes only the final "bytes → our type" step; it does
  **not** skip codecs. zarrs also passes us the region shape, so we never
  re-derive it.
- **`ArrayBytes` has three layouts, and `Tensor`/`into_fixed` only handles one.**
  `ArrayBytes` is `Fixed | Variable | Optional` (the last is data + a 1-byte/elem
  validity mask). zarrs's own `Tensor` does `bytes.into_fixed()`, which errors for
  both `Variable` and `Optional`. So `Tensor` is fixed-dense-only; we need our own
  type to represent all three.
- **`ArrayBytes` does no alignment handling.** It's `Cow<'static, [u8]>` straight
  from the decode allocation — contractually 1-byte aligned (de-facto 16 from
  system malloc; a borrowed shard sub-slice could be less). The alignment we have
  *today* is purely the side effect of the typed-path copy above. (See
  [Alignment](#alignment).)

## Architecture

### One Rust type in, four Python types out

We implement `FromArrayBytes` for an internal `Decoded` enum, retrieve
`::<Decoded>` (one call, no macro), and convert to the matching Python class:

| `ArrayBytes` variant | Python class | exposes |
|---|---|---|
| `Fixed` | **`Tensor`** | buffer protocol + `to_numpy` (+ Arrow/DLPack later) |
| `Variable` | **`VariableArray`** | (skeleton; Arrow later) |
| `Optional(Fixed)` | **`MaskedTensor`** | (skeleton; Arrow-with-validity later) |
| `Optional(Variable)` | **`MaskedVariableArray`** | (skeleton) |

**Separate classes, not one union class with conditionally-erroring methods.**
The type *is* the information the consumer needs (`isinstance`), each class
implements exactly the faces it can support, and it mirrors numpy (`MaskedArray`
is its own type) and Arrow (typed array classes). The "union" lives only as the
throwaway `Decoded` enum at the FFI seam, surfaced via `IntoPyObject` so both the
sync and async retrieve paths just return `Decoded`.

### Zero-copy payload

`Tensor` holds the decoded bytes as `bytes::Bytes` (refcounted, cheaply cloned),
obtained by moving the `Fixed` `Cow::Owned(Vec<u8>)` into `Bytes` (zero-copy).
Every face takes a cheap `Bytes` clone:

- **buffer protocol:** `pyo3_bytes::PyBytes::new(bytes.clone())` — zero-copy,
  already implements the buffer protocol.
- **`to_numpy`:** `np.frombuffer(PyBytes, dtype).reshape(shape)`, where `dtype` is
  the zarr v3 dtype name (numpy accepts the same names for the fixed numerics) and
  the bytes are native-endian (matching numpy's default).

This is the "Rust hands over bytes, Python interprets the dtype" model: numpy owns
the reinterpretation, so we never do a Rust-side `Vec<u8>→Vec<T>` cast (which would
require alignment — the exact pain hit in
[async-tiff#165](https://github.com/developmentseed/async-tiff/pull/165), which
fell back to `bytemuck::try_cast_vec` + copy).

## Alignment

**Decision: do not align up front.** Buffers are whatever the decode allocator
produced (unaligned in the worst case).

- **Buffer protocol:** no alignment requirement; numpy handles unaligned.
- **`np.frombuffer` / view consumers:** numpy views unaligned bytes (sets
  `aligned=False`); correct, and fine on x86-64/aarch64 where unaligned loads are
  cheap. **No copy.**
- **Owning consumers** (`np.array(data)`, `.copy()`, most ops): numpy allocates
  its own *aligned* buffer and copies into it — so alignment is **fused into the
  copy they were already doing**. We never pay a *dedicated* alignment copy.
- **Arrow:** recommends 64-byte but the C Data Interface relaxes to advisory;
  pyarrow mostly tolerates, occasionally realign-copies in strict kernels.

Rejected alternatives: changing zarrs to allocate aligned output (not a one-liner;
allocations spread across codecs, and borrowed shard sub-slices can't align without
a copy — worth an upstream feature request, not a local fix); and
`retrieve_*_into` with our own aligned buffer (doesn't support variable-length and
needs `unsafe` — not adopting yet). If a consumer ever needs guaranteed alignment,
expose an explicit aligned-copy method rather than copying on every read.

## Dispatch collapse (the payoff)

Sync and async `retrieve_array_subset` / `retrieve_chunk` each become a single
`retrieve::<Decoded>(…)` call — deleting the 12-arm `for_each_dtype!` macro, the
`DataInner` enum, and the hand-rolled `unsafe` buffer-protocol code in `data.rs`.
`Decoded: IntoPyObject` builds the right class.

## Faces: now vs. later

- **Now:** buffer protocol + `to_numpy` on `Tensor`. The other three classes are
  skeletons carrying `shape`/`dtype` (piped through so the type hierarchy exists),
  with `to_numpy` raising `NotImplementedError`.
- **Later (own specs):** Arrow `__arrow_c_array__` on all four (variable-length and
  masked are where Arrow earns its keep — zero-copy `String`/`Binary` + validity
  bitmaps); DLPack `__dlpack__` on `Tensor`/`MaskedTensor`.
- **Zero-copy introspection** (`contiguous`, `arrow_copy`, `buffer_protocol_copy`)
  from the Arrow spec applies here too.

## DLPack note (deferred, with a real blocker)

zarrs has a `dlpack` feature wiring its `Tensor` to the `dlpark` crate
(`SafeManagedTensorVersioned`). But **`dlpark` 0.6's `pyo3` feature pins pyo3
0.25, and we're on pyo3 0.29** — two pyo3 versions can't share one extension
module. So we cannot use `dlpark`'s capsule helper directly; DLPack would mean
hand-rolling the `PyCapsule` (name `dltensor`/`dltensor_versioned` + a deleter
calling the managed tensor's `deleter`) over `dlpark`'s core (which *is* pyo3-free
and usable), or waiting for a `dlpark` pyo3 bump. DLPack also only covers the plain
numerics + bf16 (no complex, no var-length), so it's one face among several, not a
replacement. Deferred.

## Naming

Provisional: `Tensor`, `VariableArray`, `MaskedTensor`, `MaskedVariableArray`.
("Tensor" deliberately names only the fixed-dense class, not the umbrella — there
is no single umbrella class by design.)

## Testing

- Round-trip vs. zarr-python: write fixed numeric arrays, read with zarrista,
  assert `Tensor.to_numpy()` (and `np.frombuffer(tensor.buffer())`) equal the
  zarr-python array across dtypes/shapes, including a multi-dim chunk.
- Assert a variable-length / masked array returns the correct class (not `Tensor`).
- `maturin develop`; `uv run --no-project pytest`.

## Out of scope (this spike)

Exposing data from `VariableArray`/`MaskedTensor`/`MaskedVariableArray`; Arrow and
DLPack faces; complex/temporal/sub-byte dtype numpy mappings; writing.
