# State of Zarrista

_Prepared for the Zarr prioritization call — 2026-06-23. One week of work._

## TL;DR

Zarrista is a small, Python-first Zarr library that binds the Rust **`zarrs`**
crate directly via PyO3. In ~one week it has gone from nothing to a working,
type-hinted package that can:

- **Read** Zarr v3 arrays and groups (sync + async), with NumPy / Arrow / DLPack
  zero-copy exchange, validated round-trip against zarr-python.
- **Create** arrays via a fluent `ArrayBuilder`, and **write** individual chunks.
- Talk to **local FS, object stores (S3/GCS/Azure via obstore), and Icechunk**.

It is explicitly an **evaluation prototype** — not production-ready, and not yet
benchmarked. The open question for this call is whether to invest further.

This document is a factual snapshot of what exists today, derived directly from
the source tree (`python/zarrista/*.pyi`, `src/`, `tests/`) at commit `f7af44f`.

---

## What is zarrista, and how does it relate to zarrs-python?

There are two ways to put `zarrs` under Python:

| | **zarrs-python** (existing) | **zarrista** (this project) |
|---|---|---|
| Goal | Drop-in accelerator for **zarr-python**'s codec pipeline | A standalone, low-level Zarr API in the shape of zarrita.js |
| Surface | Implements zarr-python's store/codec hooks | Its own `Array` / `Group` / `ArrayBuilder` classes |
| Audience | Existing zarr-python users (transparent) | New callers wanting a thin, explicit, typed Zarr API |
| Maturity | Established | One week old, prototype |

These are **complementary, not competing**. zarrs-python makes today's
zarr-python faster from the inside; zarrista explores what a clean, Rust-native
Python Zarr API could look like, and whether zarr-python could one day lean on it
for object modelling (not just codecs). Both sit on the same `zarrs` core.

---

## Could zarr-python depend on this?

**Not today, but the architecture is deliberately compatible.** Relevant facts:

- **Pure Zarr v3, spec-faithful metadata.** `ArrayBuilder.create_metadata()`
  emits standard v3 JSON; arrays written by zarrista are read back correctly by
  **zarr-python**, and vice versa (verified both directions — see the notebook
  and `tests/test_indexing.py`).
- **Stores are pluggable at the Rust boundary**, but there is currently **no
  Python-side store protocol** — you cannot hand zarrista an arbitrary
  zarr-python store. It accepts concrete stores only (filesystem, memory,
  obstore, Icechunk). This is the single biggest blocker to zarr-python depending
  on it, and it is not yet designed.
- **Read coverage is solid; write coverage is minimal** (single chunks only — no
  multi-chunk region writes, no group creation). zarr-python would need both.

**Realistic dependency paths**, in increasing ambition:
1. zarr-python uses zarrista's **metadata / codec** machinery as a fast helper.
2. zarr-python delegates **whole-array read/write** to zarrista behind its store
   abstraction (requires the Python store protocol + region writes).
3. zarr-python adopts zarrista objects directly (largest change; furthest off).

None of these are close yet, but nothing in the design precludes them.

---

## Expected performance gains

**Honest status: not yet benchmarked.** No numbers should be quoted in the call.
What we *can* say is where gains are structurally expected, because the work moves
out of Python and into `zarrs`:

- **Codec pipeline in native code.** Decompression, sharding, transpose, etc. run
  in Rust with Rayon-based parallelism (`concurrent_target` /
  `chunk_concurrent_minimum` are exposed as codec options), instead of per-chunk
  Python overhead. This is the same thesis that motivates zarrs-python's measured
  speedups.
- **Zero-copy hand-off.** Decoded buffers cross into NumPy (buffer protocol /
  `np.frombuffer`), Arrow (C Data Interface), and DLPack **without a copy**. The
  Rust allocation *is* the array's backing memory.
- **No Python-level chunk orchestration** for reads — selection → chunk fetch →
  decode → assemble happens once, in Rust.

**Recommendation:** the highest-value next deliverable for *this* audience is a
small, reproducible benchmark (zarrista vs zarr-python vs zarrs-python on a
representative read) so the perf claim is evidence, not architecture.

---

## How much of the zarrs API have we wrapped?

Coverage by area (✅ done · ⚠️ partial · ❌ absent). Method names below are the
**actual live surface** at `f7af44f`.

### Reading — ✅ mature
- `Array.open` / `AsyncArray.open_async`
- `arr[selection]` and `retrieve_array_subset(selection)` — NumPy-style **basic**
  indexing (int, step-1 slice, ellipsis, negative indices). `step != 1`, newaxis,
  boolean/fancy indexing raise `NotImplementedError`/`IndexError`.
- `retrieve_chunk(idx)`, `retrieve_encoded_chunk(idx)` (pre-codec bytes)
- Metadata accessors: `shape`, `ndim`, `dtype`, `attrs`, `dimension_names`,
  `chunk_grid`, `filters`, `serializer`, `compressors`, `metadata`

### Groups — ✅ read · ❌ write
- `Group.open` / `AsyncGroup.open_async`
- `array_keys`, `group_keys`, `traverse`, `child_arrays`, `child_groups`,
  `child_paths` (+ array/group variants), `child(name)` / `grp[name]`
- `attrs`, `metadata`, `consolidated_metadata` (read; consolidated metadata is
  **read-only** — not written)
- `store_metadata` / `erase_metadata` exist, but there is **no group creation
  builder**.

### Creating arrays — ✅ via `ArrayBuilder`
- Immutable, chainable: `shape`, `chunk_grid`, `chunk_key_encoding`, `data_type`,
  `dimension_names`, `attrs`, `filters`, `compressors`, `serializer`,
  `subchunk_shape` (enables sharding), plus `like(array)` to clone config.
- Materialize: `create` / `create_async` (auto-writes metadata, commit `f7af44f`)
  and `create_metadata()` (metadata only, no store touch).

### Writing data — ⚠️ minimal
- `store_chunk(idx, ArrayBytes)`, `store_encoded_chunk(idx, bytes)`,
  `compact_chunk(idx)`, `erase_chunk(idx)`, `erase_metadata()` (sync + async).
- **Absent:** multi-chunk region writes (`store_array_subset`), array resize,
  in-place attribute/metadata updates, partial encoding.

### Codecs — ⚠️ thin but extensible
- Convenience constructors: `transpose`, `bitround` (array→array); `gzip`, `zstd`,
  `blosc`, `crc32c` (bytes→bytes); sharding via `serializer`/`subchunk_shape`.
- **Any** zarrs codec is still usable via `from_config({...v3 metadata...})`;
  most simply lack a typed Python constructor.

### Data types — ✅ read path
- All v3 fixed/variable dtypes decode; result dispatches to `Tensor`,
  `VariableArray`, `MaskedTensor`, or `MaskedVariableArray`.
- `MaskedTensor` / `MaskedVariableArray` carry data but **do not yet expose
  `to_numpy()`**. Complex dtypes not surfaced to NumPy.

### Stores — ✅ broad
- Sync: `FilesystemStore`, `MemoryStore`.
- Async: obstore (`ObjectStore` → S3/GCS/Azure/local/HTTP), Icechunk (`Session`).
- **Absent:** Python-side custom store protocol.

### Data exchange — ✅ multiple zero-copy faces
- Buffer protocol + `Tensor.to_numpy()`; Arrow C Data Interface on `Tensor` /
  `VariableArray`; DLPack on `Tensor`.

---

## How much *more* should we wrap? (suggested priorities)

Ordered by leverage for the stated goals (zarr-python interop + write workloads):

1. **Benchmark harness** — turn the perf thesis into measured numbers. _(Small,
   high signal for funding conversations.)_
2. **Multi-chunk region writes** (`store_array_subset`) — the obvious missing half
   of the write story; today only single chunks can be written.
3. **Python store protocol** — the gateway to any real zarr-python integration.
4. **Group creation / metadata writes** — needed for end-to-end dataset authoring.
5. **`to_numpy()` for masked/variable results** — completes the read story for
   string/variable and nullable data.
6. **Codec breadth** — typed constructors for the rest of the zarrs codec set
   (and consolidated-metadata writing).

Lower priority / known debt (from `src/` TODOs): selection parsing cleanup
(`src/array/selection.rs`), richer node `Path` type, `Node` as a full `#[pyclass]`,
DLPack version pin on `dlpark`.

---

## Engineering signals

- **Type-driven, "parse-don't-validate"** design at the PyO3 boundary (see
  `CLAUDE.md`): inputs are parsed into already-valid typed forms; no scattered
  runtime validation.
- **Full type hints** (`.pyi` stubs) for the entire surface; `py.typed` shipped.
- **Tests** round-trip against zarr-python, pyarrow, and Icechunk
  (`tests/test_indexing.py`, `test_builder.py`, `test_group.py`, `test_arrow.py`,
  `test_icechunk.py`, `test_exceptions.py`, `test_store_input.py`).
- **Structured exception hierarchy** under `zarrista.exceptions` (13 classes).
- **Docs site** (mkdocs-material + mkdocstrings) published with versioning (mike).
  This audit added the previously-missing reference pages (builder, exceptions,
  `ChunkKeyEncoding`, `FillValue`).
- **Distribution**: single abi3 wheel, Python 3.11+; CI builds wheels + runs tests.

---

## Bottom line for the call

- **Progress in one week is substantial**: a coherent, typed, sync+async Zarr
  read API with zero-copy exchange, plus array creation and chunk writes, all
  interoperable with zarr-python.
- **It is a prototype**: write support is minimal, there's no Python store
  protocol, and there are **no benchmarks yet** — so the central "is it faster?"
  claim is currently unproven.
- **The cheapest, most decision-relevant next step is a benchmark**; the most
  strategically important is the Python store protocol if zarr-python integration
  is the goal.
