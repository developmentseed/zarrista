"""Compare full-array read speed across three Zarr implementations.

The benchmark writes one array with zarr-python. It then reads the whole array
repeatedly with each implementation, and prints a comparison table.

Build zarrista in release mode first. A debug build is many times slower, and
it makes the result meaningless:

    uv sync --group bench --no-install-package zarrista
    uv run --no-project maturin develop --uv --release

Then run the benchmark. Omit `--shards` to benchmark a plain chunked array:

    uv run --no-project python bench/bench_read.py
    uv run --no-project python bench/bench_read.py --shards 512 512
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

ARRAY_NAME = "bench"
"""The name of the array inside the fixture store."""


def build_data(shape: tuple[int, ...], dtype: str) -> np.ndarray:
    """Build the deterministic source array that the benchmark writes and checks.

    Args:
        shape: The array shape.
        dtype: The NumPy data type name.

    Returns:
        An array of the given shape and data type.
    """
    count = int(np.prod(shape))
    return (np.arange(count) % 251).astype(dtype).reshape(shape)


def write_fixture(
    root: Path,
    data: np.ndarray,
    chunks: tuple[int, ...],
    shards: tuple[int, ...] | None,
    compressor: str,
) -> None:
    """Write the benchmark array once, with zarr-python.

    Every implementation then reads these same bytes.

    Args:
        root: The directory that holds the store.
        data: The data to write.
        chunks: The chunk shape. This is the inner chunk shape when sharded.
        shards: The shard shape, or `None` for a plain chunked array.
        compressor: A blosc `cname`, or `"none"` for no compression.
    """
    import zarr
    from zarr.codecs import BloscCodec

    compressors = (BloscCodec(cname=compressor),) if compressor != "none" else None
    array = zarr.create_array(
        store=str(root),
        name=ARRAY_NAME,
        shape=data.shape,
        chunks=chunks,
        shards=shards,
        dtype=data.dtype.name,
        compressors=compressors,
    )
    array[...] = data


def time_reads(
    read: Callable[[], np.ndarray],
    iterations: int,
) -> tuple[np.ndarray, list[float]]:
    """Run one warm-up read, then time `iterations` more reads.

    Args:
        read: A function that reads the whole array and returns it.
        iterations: The number of timed reads.

    Returns:
        The data from the last read, and the elapsed seconds of each timed read.
    """
    out = read()
    times: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        out = read()
        times.append(time.perf_counter() - start)
    return out, times


def run_zarr(
    root: Path,
    iterations: int,
    overrides: dict[str, object],
) -> tuple[np.ndarray, list[float]]:
    """Time full-array reads through zarr-python, under the given configuration.

    The array is opened inside the configuration scope, so that zarr-python
    picks up the selected codec pipeline.

    Args:
        root: The directory that holds the store.
        iterations: The number of timed reads.
        overrides: The `zarr.config` settings to apply for this run.

    Returns:
        The data from the last read, and the elapsed seconds of each timed read.
    """
    import zarr

    with zarr.config.set(overrides):
        array = zarr.open_array(store=str(root), path=ARRAY_NAME)

        def read() -> np.ndarray:
            return array[...]

        return time_reads(read, iterations)


def report(
    results: list[tuple[str, list[float]]],
    nbytes: int,
    header: list[str],
) -> None:
    """Print the header lines and the comparison table.

    The final column compares each implementation against the first one.

    Args:
        results: The name and the timings of each implementation, in order.
        nbytes: The logical size of the array in bytes.
        header: The lines to print above the table.
    """
    for line in header:
        print(line)
    print()
    megabytes = nbytes / 1e6
    baseline = statistics.median(results[0][1])
    baseline_label = f"vs {results[0][0]}"
    print(
        f"{'implementation':<22}{'best (ms)':>12}{'median (ms)':>14}"
        f"{'median MB/s':>14}{baseline_label:>18}",
    )
    for name, times in results:
        best = min(times)
        median = statistics.median(times)
        print(
            f"{name:<22}{best * 1e3:>12.2f}{median * 1e3:>14.2f}"
            f"{megabytes / median:>14.0f}{baseline / median:>17.2f}x",
        )


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Compare full-array Zarr read speed.")
    parser.add_argument("--shape", type=int, nargs="+", default=[2048, 2048])
    parser.add_argument(
        "--chunks",
        type=int,
        nargs="+",
        default=[64, 64],
        help="chunk shape; the inner chunk shape when --shards is given",
    )
    parser.add_argument(
        "--shards",
        type=int,
        nargs="+",
        default=None,
        help="shard shape; omit for a plain chunked array",
    )
    parser.add_argument("--dtype", default="uint16")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--compressor",
        default="zstd",
        choices=["zstd", "lz4", "none"],
        help="blosc cname, or none for no compression",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Write the fixture, time every implementation, and print the table.

    Args:
        args: The parsed command-line arguments.
    """
    shape = tuple(args.shape)
    chunks = tuple(args.chunks)
    shards = tuple(args.shards) if args.shards else None
    data = build_data(shape, args.dtype)

    root = Path(tempfile.mkdtemp(prefix="zarrista-bench-"))
    try:
        write_fixture(root, data, chunks, shards, args.compressor)

        results: list[tuple[str, list[float]]] = []
        out, times = run_zarr(root, args.iterations, {})
        np.testing.assert_array_equal(out, data)
        results.append(("zarr-python", times))

        out, times = run_zarr(
            root,
            args.iterations,
            {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"},
        )
        np.testing.assert_array_equal(out, data)
        results.append(("zarr-python+zarrs", times))

        header = [
            (
                f"array: shape={shape} dtype={args.dtype} chunks={chunks} "
                f"shards={shards} compressor={args.compressor}"
            ),
            (
                f"size: {data.nbytes / 1e6:.1f} MB logical,"
                f" {args.iterations} iterations, FilesystemStore"
            ),
            "correctness: all implementations match the source data",
            "note: these numbers are valid only if zarrista was built with --release",
        ]
        report(results, data.nbytes, header)
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main(parse_args())
