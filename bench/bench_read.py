"""Compare full-array read speed across three Zarr implementations.

The benchmark writes one array with zarr-python. It then reads the whole array
repeatedly with each implementation, and prints a comparison table.

Build zarrista in release mode first. A debug build is many times slower, and
it makes the result meaningless:

    uv sync --group bench --no-install-package zarrista
    uv run --no-project maturin develop --uv --release

Then run the benchmark. Omit `--shards` to benchmark a plain chunked array:

    uv run --no-project python bench/bench_read.py
    uv run --no-project python bench/bench_read.py --shards 512,512
"""

from __future__ import annotations

import os
import shutil
import statistics
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import click
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

ARRAY_NAME = "bench"
"""The name of the array inside the fixture store."""

Compressor = Literal["zstd", "lz4", "none"]
"""The compression choices that the benchmark accepts."""


class ShapeParam(click.ParamType):
    """A click parameter that holds a comma-separated list of sizes.

    The parameter converts `"512,512"` into `(512, 512)`. It rejects any value
    that is not a list of positive integers, so that the benchmark receives an
    already-valid shape.
    """

    name = "sizes"

    def convert(
        self,
        value: str | tuple[int, ...],
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> tuple[int, ...]:
        """Convert a comma-separated string into a tuple of sizes.

        Args:
            value: The text to convert, or an already-converted tuple.
            param: The parameter that the value belongs to.
            ctx: The click context.

        Returns:
            The sizes along each dimension.
        """
        if isinstance(value, tuple):
            return value
        try:
            sizes = tuple(int(part) for part in value.split(","))
        except ValueError:
            message = f"{value!r} is not a comma-separated list of integers"
            self.fail(message, param, ctx)
        if not sizes or any(size < 1 for size in sizes):
            self.fail(f"{value!r} must hold only positive integers", param, ctx)
        return sizes


SIZES = ShapeParam()
"""The shared instance of the comma-separated size parameter."""


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
    compressor: Compressor,
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
            # A full basic selection always gives an array, never a scalar, but
            # the declared return type also covers the scalar case.
            return cast("np.ndarray", array[...])

        return time_reads(read, iterations)


def run_zarrista(root: Path, iterations: int) -> tuple[np.ndarray, list[float]]:
    """Time full-array reads through zarrista.

    The `.to_numpy()` call is inside the timed region. zarrista returns a
    `Tensor`, and the other implementations return a NumPy array, so this makes
    every implementation produce the same result type.

    Args:
        root: The directory that holds the store.
        iterations: The number of timed reads.

    Returns:
        The data from the last read, and the elapsed seconds of each timed read.
    """
    import zarrista
    from zarrista.store import FilesystemStore

    array = zarrista.Array.open(FilesystemStore(root), path=f"/{ARRAY_NAME}")

    def read() -> np.ndarray:
        # The benchmark only uses fixed-width numeric data types, which always
        # decode to a `Tensor`. The other members of `DecodedArray` cannot
        # occur here, and one of them has no `to_numpy` method.
        tensor = cast("zarrista.Tensor", array[...])
        return tensor.to_numpy()

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


@click.command()
@click.option(
    "--shape",
    type=SIZES,
    default="2048,2048",
    show_default=True,
    help="The array shape, as comma-separated sizes.",
)
@click.option(
    "--chunks",
    type=SIZES,
    default="64,64",
    show_default=True,
    help="The chunk shape. This is the inner chunk shape when --shards is given.",
)
@click.option(
    "--shards",
    type=SIZES,
    default=None,
    help="The shard shape. Omit it for a plain chunked array.",
)
@click.option(
    "--dtype",
    default="uint16",
    show_default=True,
    help="The NumPy data type name.",
)
@click.option(
    "--iterations",
    type=int,
    default=10,
    show_default=True,
    help="The number of timed reads for each implementation.",
)
@click.option(
    "--compressor",
    type=click.Choice(["zstd", "lz4", "none"]),
    default="zstd",
    show_default=True,
    help="A blosc cname, or none for no compression.",
)
@click.option(
    "--threads",
    type=int,
    default=os.cpu_count() or 1,
    show_default="CPU count",
    help="The thread count, applied to every implementation.",
)
def main(  # noqa: PLR0913
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    shards: tuple[int, ...] | None,
    dtype: str,
    iterations: int,
    compressor: Compressor,
    threads: int,
) -> None:
    """Compare full-array read speed across three Zarr implementations."""
    # Rayon reads RAYON_NUM_THREADS once, when it first builds its global
    # thread pool. zarrista and zarrs each hold a separate pool. Set the
    # variable before either extension is imported, or --threads does nothing.
    # This is why every extension import in this file is function-local.
    os.environ["RAYON_NUM_THREADS"] = str(threads)

    data = build_data(shape, dtype)
    root = Path(tempfile.mkdtemp(prefix="zarrista-bench-"))
    try:
        write_fixture(root, data, chunks, shards, compressor)

        max_workers = {"threading.max_workers": threads}

        results: list[tuple[str, list[float]]] = []
        out, times = run_zarr(root, iterations, max_workers)
        np.testing.assert_array_equal(out, data)
        results.append(("zarr-python", times))

        out, times = run_zarr(
            root,
            iterations,
            {**max_workers, "codec_pipeline.path": "zarrs.ZarrsCodecPipeline"},
        )
        np.testing.assert_array_equal(out, data)
        results.append(("zarr-python+zarrs", times))

        out, times = run_zarrista(root, iterations)
        np.testing.assert_array_equal(out, data)
        results.append(("zarrista", times))

        header = [
            (
                f"array: shape={shape} dtype={dtype} chunks={chunks} "
                f"shards={shards} compressor={compressor}"
            ),
            (
                f"size: {data.nbytes / 1e6:.1f} MB logical,"
                f" {iterations} iterations, FilesystemStore,"
                f" {threads} threads"
            ),
            "correctness: all implementations match the source data",
            "note: these numbers are valid only if zarrista was built with --release",
        ]
        report(results, data.nbytes, header)
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
