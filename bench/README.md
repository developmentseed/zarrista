# Read benchmarks

These scripts answer one question: **is the native zarrs binding faster than
zarr-python?**

They are for evaluation. There is no CI integration and no regression
tracking. Run them by hand, and read the numbers as evidence, not as a
contract.

## Build zarrista in release mode first

**The numbers mean nothing unless zarrista is built with `--release`.** The
normal development loop builds in debug mode, which turns off optimization in
the Rust code. A debug build is many times slower than a release build.

Python cannot tell the two builds apart at run time, so the benchmark cannot
check this for you.

```bash
uv sync --group bench --no-install-package zarrista
uv run --no-project maturin develop --uv --release
```

When you go back to normal development, build a debug version again:

```bash
uv run --no-project maturin develop --uv
```

## Run

Benchmark a sharded array:

```bash
uv run --no-project python bench/bench_read.py --shards 512,512
```

Benchmark a plain chunked array. Omit `--shards`:

```bash
uv run --no-project python bench/bench_read.py
```

A shape is one comma-separated value, such as `--shards 512,512`.

| Option | Default | Meaning |
| --- | --- | --- |
| `--shape` | `2048,2048` | The array shape. |
| `--chunks` | `64,64` | The chunk shape. This is the inner chunk shape when sharded. |
| `--shards` | none | The shard shape. Omit it for a plain chunked array. |
| `--dtype` | `uint16` | The NumPy data type name. |
| `--iterations` | `10` | The number of timed reads for each implementation. |
| `--compressor` | `zstd` | A blosc `cname`, or `none` for no compression. |
| `--threads` | CPU count | The thread count, applied to every implementation. |

## What it measures

The benchmark writes one array with zarr-python. Every implementation then
reads those same bytes from a local filesystem store.

| Row | What it is |
| --- | --- |
| `zarr-python` | Stock zarr-python with the pure-Python codec pipeline. |
| `zarr-python+zarrs` | zarr-python with the `zarrs` Rust codec pipeline plugin. |
| `zarrista` | zarrista's `array[...]`, then `.to_numpy()`. |

The middle row matters. Without it, you cannot tell whether a speed increase
comes from Rust codecs or from a native end-to-end binding.

Each implementation does one read that is checked against the source data,
then one warm-up read, then the timed reads. The `.to_numpy()` call is inside
zarrista's timed region, because the other two rows already return a NumPy
array.

## Example output

Measured on an Apple M-series laptop, with 10 threads and a release build.
Your numbers will differ. Read the ratios, not the absolute times.

Sharded, `--shards 512,512`:

```
array: shape=(2048, 2048) dtype=uint16 chunks=(64, 64) shards=(512, 512) compressor=zstd
size: 8.4 MB logical, 10 iterations, FilesystemStore, 10 threads
correctness: all implementations match the source data
note: these numbers are valid only if zarrista was built with --release

implementation           best (ms)   median (ms)   median MB/s    vs zarr-python
zarr-python                  81.97         83.49           100             1.00x
zarr-python+zarrs             2.60          2.66          3157            31.42x
zarrista                      2.34          2.40          3502            34.86x
```

Plain chunked:

```
array: shape=(2048, 2048) dtype=uint16 chunks=(64, 64) shards=None compressor=zstd
size: 8.4 MB logical, 10 iterations, FilesystemStore, 10 threads
correctness: all implementations match the source data
note: these numbers are valid only if zarrista was built with --release

implementation           best (ms)   median (ms)   median MB/s    vs zarr-python
zarr-python                 193.82        197.89            42             1.00x
zarr-python+zarrs            21.28         21.61           388             9.16x
zarrista                     10.78         11.19           749            17.68x
```

Both Rust rows are far ahead of the pure-Python row. The two Rust rows are
close on the sharded array, but zarrista is about twice as fast as the `zarrs`
codec pipeline on the plain chunked array. This is the result that the middle
row exists to show: on this shape, the gain comes from more than the codecs
alone.

Treat one pair of runs as a starting point, not as a conclusion. Vary the
shape, the chunk shape, the data type, and the thread count before you trust a
ratio.

## What is not controlled

- **Store request concurrency.** zarr-python's `async.concurrency` setting
  stays at its default of 10. It limits concurrent store requests, which is an
  IO axis and not a CPU axis. `--threads` does not change it.
- **The page cache.** The fixture is written and then read immediately, so the
  data is usually warm in the operating system page cache. The benchmark
  measures decode speed much more than it measures disk speed.
- **Other work on the machine.** Close other programs before you trust a
  result.

## Not covered yet

- An in-memory store. `zarrista.MemoryStore()` has no API that accepts
  external bytes, so the two libraries cannot read one set of bytes from
  memory. Each would have to write its own copy.
- Partial and strided region reads.
- Writes.
- Remote object stores.
