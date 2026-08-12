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

We're releasing Zarrista, a new high-performance Python library for working with [Zarr] data, powered by Rust.

[Zarr]: https://zarr.dev/

<!-- more -->

## Why a new library?

### Focus on performance

Zarrista wraps the Zarrs Rust library.

### Planned integration into Zarr-Python



## Performance

### Benchmarks

In [our PR](https://github.com/zarrs/zarr_benchmarks/pull/12) to [`zarr_benchmarks`](https://github.com/zarrs/zarr_benchmarks), Zarrista is the fastest Python chunked array library, in line with Google's [Tensorstore]. It's surpassed only by the Rust [Zarrs] library, which Zarrista uses internally.

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



## Future Work

