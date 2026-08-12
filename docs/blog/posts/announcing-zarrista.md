---
draft: false
date: 2026-08-13
categories:
  - Release
  - Feature
authors:
  - kylebarron
  - d-v-b
# links:
#     # TODO: update changelog link
#   - CHANGELOG.md#0130-2025-11-05
---

# Zarrista: Faster Zarr Interface for Python

_This blog post was fully written by humans._

[Zarr] is the pre-eminent open data format for storing chunked N-dimensional arrays.

We're releasing Zarrista, a new high-performance Python library for interfacing with [Zarr] data, powered by Rust and the [Zarrs] library.

[Zarr]: https://zarr.dev/
[Zarrs]: https://zarrs.dev/

<!-- more -->

## A new library, but planned integration into Zarr-Python

Zarrista is developed as an independent library, but we plan to update Zarr-Python to support using Zarrista under the hood.

### Improved performance with compiled code

[Zarr-Python], the current canonical Python library for working with Zarr, is primarily _written in Python_. Aside from a few compiled dependencies for working with codecs, all of Zarr-Python's own source code is Python.

In contrast, Zarrista is _fully compiled_. Only the user interface is Python; everything else is compiled Rust code. When you read data from a Zarr [`Array`][zarrista.Array] object, the entire sequence of operations is happening in compiled code.

This allows for significant performance improvements compared to Zarr-Python. See the [Benchmarks](#benchmarks) section for more information.

[Zarr-Python]: https://zarr.readthedocs.io/en/stable/

### Planned integration into Zarr-Python

However, to avoid fracturing the ecosystem, we **plan to integrate Zarrista as a native backend into Zarr-Python**.

We hope to bring most of the performance potential into Zarr-Python directly, so that existing users can get speedups without learning a new API. The Zarr-Python PR [#4064](https://github.com/zarr-developers/zarr-python/pull/4064) prototypes generic backends to allow opting in to a Zarrista driver.

### Standalone library for lower-level APIs

For users who are not tied to Zarr-Python, it may be possible to eke out the best possible performance using Zarrista directly.

For example, Zarrista allows users to separate IO-bound network operations and CPU-bound codec operations, for optimal scheduling.

Zarrista also offers full async and sync counterparts for users to choose what works best for them.

### Keeping Zarr-Python maintainable without Rust source code

Zarrista began after seeing Davis [exploring bringing Rust into Zarr-Python directly](https://github.com/zarr-developers/zarr-python/pull/4064).

Building


Kyle

In terms of build system, testing, etc. It's easier to have the underlying Rust


### Lower-level APIs

## Performance

### Benchmarks

In [our PR](https://github.com/zarrs/zarr_benchmarks/pull/12) to [`zarr_benchmarks`](https://github.com/zarrs/zarr_benchmarks), Zarrista is the fastest Python chunked array library, in line with Google's [Tensorstore]. It's surpassed only by the Rust [Zarrs] library, which Zarrista uses internally.

These benchmarks only use the synchronous local file system APIs. We'd like to benchmark remote object store performance in the future.

[Zarrs]: https://zarrs.dev/
[Tensorstore]: https://github.com/google/tensorstore

#### Read All

The minimum time and peak memory usage to read an entire dataset into memory.

![](../../assets/benchmark_read_all.svg)

#### Read Chunk-By-Chunk

The minimum time and peak memory usage to read a dataset chunk-by-chunk into memory.

![](../../assets/benchmark_read_chunks.svg)

#### Read Subchunk-By-Subchunk

The minimum time and peak memory usage to read a dataset subchunk-by-subchunk into memory.

![](../../assets/benchmark_read_subchunks.svg)

### Zero-copy data exchange

## Integrations

### Obstore

### Icechunk

## Usage Example

## AI Usage

Zarrista's source code is _minimally_ vibe-coded.

Most of the library code was written by hand by Kyle, sometimes resulting from a conversation with Claude.

Documentation and type stubs are mixed. Much documentation is written by hand but Python type stubs are partially kept up to date via Claude.

Almost all current tests were written by Claude.

## Future Work

