# zarrista — API roadmap & next areas

**Date:** 2026-06-16
**Status:** Draft

## Positioning

zarrista is **not** the same kind of project as the official
[`zarrs-python`](https://github.com/zarrs/zarrs-python). That binding is small and
narrow: it injects the `zarrs` codec pipeline *behind* zarr-python, accelerating
encode/decode while zarr-python owns the API, stores, indexing, and metadata.

zarrista is being explored as a **low-level Zarr API in its own right** — one that
could *replace* zarr-python, or that zarr-python could *depend on* for its core in
the medium term. That ambition sets the design constraints below.

### Design mindset vs. shipping order

These are deliberately different:

- **Design mindset: zarr-python replacement.** Every API decision should be made as
  if this library will eventually need writing, full indexing semantics, a
  pluggable store abstraction, groups-with-creation, and consolidated metadata.
  Don't paint ourselves into a read-only or numerics-only corner with the type
  signatures, class hierarchy, or store traits.
- **Shipping order: fast standalone cloud reader first.** The immediate goal is to
  get something real working end-to-end so we can **benchmark** against zarr-python
  on cloud reads. The reader path (async + obstore) is where zarrs should already
  beat zarr-python, so it's both the fastest path to a demo and the most
  differentiated.

The litmus test for any near-term decision: *does it move us toward a benchmarkable
cloud reader, without foreclosing the replacement-grade API later?*

## Current surface (as of this doc)

Read-only, async-first metadata + raw-chunk reader:

- `Array` / `AsyncArray`: metadata properties (`shape`, `dtype`, `ndim`, `attrs`,
  `metadata`, `chunk_grid`, `codecs`, `dimension_names`, `path`) plus
  `retrieve_chunk(chunk_indices)`.
- `Group` / `AsyncGroup`: `attrs`, `array_keys()`, `group_keys()`, child navigation.
- `Data`: zero-copy numpy via the buffer protocol.
- Stores: sync `FilesystemStore` / `MemoryStore`; async = any `obstore.ObjectStore`.
- Dtypes: fixed-width numerics only (bool, int/uint 8–64, float16/32/64).
- No writing, no array-coordinate indexing, no fill values, no var-length dtypes.

The key gap: `retrieve_chunk` is a *chunk-coordinate* primitive. Users think in
*array coordinates*. Closing that gap is what turns this from a chunk inspector into
an array library.

## Tier 1 — makes it usable (do first)

1. **Array indexing / `__getitem__`.** Map Python `slice`/`int`/`Ellipsis`/`None`
   to `zarrs::array_subset::ArraySubset`, call `retrieve_array_subset_opt`, return
   an ndarray. Start with **basic indexing** (slices + ints + ellipsis); defer
   orthogonal/vectorized/boolean to a later pass (mirror zarr-python's `.oindex` /
   `.vindex` split). The `retrieve_array_subset` path is already stubbed/commented
   in `array/sync.rs`. This is the single highest-impact change.

2. **Fill values + edge chunks.** Indexing forces this: subsets spanning the array
   boundary or hitting missing chunks need the fill value. Extraction code already
   exists commented-out in `dtype.rs`. Expose as `Array.fill_value` *and* wire into
   the subset read path — without it, partial-edge-chunk reads are wrong.

3. **Complete the dtype story.** Add **variable-length strings** and **fixed-width
   bytes** (target numpy 2 `StringDType`). Structured/complex dtypes can wait.

## Tier 2 — where zarrista should beat zarr-python

The differentiation, and the thing to benchmark:

4. **Parallel multi-chunk / subset reads.** Lean on zarrs's concurrent codec+I/O
   pipeline. Expose `retrieve_chunks(list_of_indices)` and make `__getitem__` over a
   multi-chunk region fan out internally with a configurable concurrency limit
   (`zarrs` `CodecOptions`/concurrency knobs). Headline: a single
   `await arr[big_slice]` pulling hundreds of chunks concurrently from S3.

5. **`retrieve_*_into` / preallocated output.** Decode into a caller-provided
   buffer to avoid an allocation and integrate cleanly with xarray/dask block
   fetching. Builds on the existing `Data` buffer-protocol work.

## Tier 3 — larger projects (replacement-grade, sequence deliberately)

6. **Writing.** New axis: writable stores (obstore PUT), `store_chunk` /
   `store_array_subset`, `create_array` / `create_group`, resize. Required for the
   replacement goal; not required for the first benchmark.

7. **Store extensibility.** (a) Let obstore back the **sync** path too via
   `block_on`, so we don't maintain two store worlds. (b) A Python-implementable
   `Store` protocol for custom backends, mirroring zarr-python's `Store` ABC.

8. **Consolidated metadata + group creation** — replacement-grade parity items.

## Cross-cutting: testing

Currently ~one smoke test. Before Tier 1 lands, stand up **round-trip tests against
zarr-python**: write with zarr-python, read with zarrista, assert equality across
dtypes/codecs/sharding. Cheapest way to buy correctness confidence and a prerequisite
for trustworthy benchmarks. Do this in parallel with Tier 1.

## Recommended sequence

1. Round-trip test harness vs. zarr-python (parallel, ongoing).
2. Tier 1: indexing → fill values/dtypes.
3. Tier 2: parallel bulk reads → benchmark vs. zarr-python on a real cloud dataset.
4. Reassess with benchmark numbers in hand before committing to Tier 3 (writing).

## Out of scope (for now)

Writing; full fancy/boolean indexing; consolidated metadata; group creation; custom
Python stores. All are in-scope for the *design* (don't foreclose them) but not for
the first benchmarkable milestone.
