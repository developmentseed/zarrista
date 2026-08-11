"""`erase_chunks` deletes a region of the chunk grid.

Its argument is a numpy-style selection in **chunk-grid** coordinates, not
element coordinates, so it is resolved against `chunk_grid_shape` rather than
`shape`. Both spaces are plain integer tuples and nothing but the shape used to
resolve them tells the two apart, so these tests pin the distinction down.
"""

from pathlib import Path

import numpy as np
import pytest
import zarr
from numpy.typing import NDArray
from obstore.store import LocalStore

from zarrista import (
    Array,
    ArrayBuilder,
    ArrayBytes,
    AsyncArray,
    ChunkGrid,
    DataType,
    FillValue,
)
from zarrista.store import MemoryStore

# The contents of the 4x4 fixture: chunk (i, j) is filled with `1 + 2i + j`.
FULL = np.array(
    [
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4],
    ],
    dtype="int32",
)
EMPTY = np.zeros((4, 4), dtype="int32")


def _chunked_array() -> Array:
    """A 4x4 int32 array: a 2x2 grid of 2x2 chunks, each filled with `1 + 2i + j`."""
    array = ArrayBuilder(
        ChunkGrid.regular([4, 4], chunk_shape=[2, 2]),
        DataType.from_string("int32"),
        FillValue(b"\x00\x00\x00\x00"),
    ).create(MemoryStore(), "/a")
    for i in (0, 1):
        for j in (0, 1):
            block = np.full((2, 2), 1 + 2 * i + j, dtype="int32")
            array.store_chunk([i, j], ArrayBytes(block.tobytes()))
    return array


def _sharded_array() -> Array:
    """An 8x8 int32 array: a 2x2 grid of 4x4 shards, each split into 2x2 subchunks."""
    array = (
        ArrayBuilder(
            ChunkGrid.regular([8, 8], chunk_shape=[4, 4]),
            DataType.from_string("int32"),
            FillValue(b"\x00\x00\x00\x00"),
        )
        .subchunk_shape([2, 2])
        .create(MemoryStore(), "/a")
    )
    for i in (0, 1):
        for j in (0, 1):
            block = np.full((4, 4), 1 + 2 * i + j, dtype="int32")
            array.store_chunk([i, j], ArrayBytes(block.tobytes()))
    return array


def test_erases_a_row_of_the_chunk_grid():
    """Chunk index 0 covers elements 0:2, not element 0."""
    array = _chunked_array()

    array.erase_chunks((0, slice(None)))

    expected = FULL.copy()
    expected[0:2, :] = 0
    np.testing.assert_array_equal(array[:, :].to_numpy(), expected)


def test_erases_a_single_chunk():
    array = _chunked_array()

    array.erase_chunks((0, 1))

    expected = FULL.copy()
    expected[0:2, 2:4] = 0
    np.testing.assert_array_equal(array[:, :].to_numpy(), expected)


def test_negative_index_wraps_against_the_chunk_grid():
    """-1 is the last *chunk* (elements 2:4); against the shape it would be a no-op."""
    array = _chunked_array()

    array.erase_chunks((-1, slice(None)))

    expected = FULL.copy()
    expected[2:4, :] = 0
    np.testing.assert_array_equal(array[:, :].to_numpy(), expected)


def test_ellipsis_erases_every_chunk():
    array = _chunked_array()

    array.erase_chunks(...)

    np.testing.assert_array_equal(array[:, :].to_numpy(), EMPTY)


def test_empty_tuple_erases_every_chunk():
    array = _chunked_array()

    array.erase_chunks(())

    np.testing.assert_array_equal(array[:, :].to_numpy(), EMPTY)


def test_erasing_absent_chunks_is_a_noop():
    array = _chunked_array()
    array.erase_chunks(...)

    array.erase_chunks(...)

    np.testing.assert_array_equal(array[:, :].to_numpy(), EMPTY)


def test_index_beyond_the_chunk_grid_raises():
    """3 is a valid element index for the (4, 4) array, but the grid is (2, 2)."""
    array = _chunked_array()

    with pytest.raises(IndexError, match="axis 0 with size 2"):
        array.erase_chunks((3, 0))


def test_a_rejected_selection_erases_nothing():
    """The selection resolves before any chunk is deleted."""
    array = _chunked_array()

    with pytest.raises(IndexError):
        array.erase_chunks((3, 0))

    np.testing.assert_array_equal(array[:, :].to_numpy(), FULL)


def test_strided_selection_is_rejected():
    array = _chunked_array()

    with pytest.raises(NotImplementedError, match="step other than 1"):
        array.erase_chunks((slice(0, 2, 2), 0))


def test_sharded_erases_a_whole_shard():
    """The chunk grid of a sharded array is the shard grid."""
    array = _sharded_array()

    array.erase_chunks((0, 0))

    np.testing.assert_array_equal(
        array.retrieve_chunk([0, 0]).to_numpy(),
        np.zeros((4, 4), dtype="int32"),
    )


def test_sharded_leaves_other_shards_intact():
    array = _sharded_array()

    array.erase_chunks((0, 0))

    np.testing.assert_array_equal(
        array.retrieve_chunk([0, 1]).to_numpy(),
        np.full((4, 4), 2, dtype="int32"),
    )


def test_subchunk_index_beyond_the_shard_grid_raises():
    """(0, 3) addresses the (4, 4) subchunk grid; the shard grid is (2, 2)."""
    array = _sharded_array()

    with pytest.raises(IndexError, match="axis 1 with size 2"):
        array.erase_chunks((0, 3))


# --- async ---------------------------------------------------------------


async def _async_chunked_array(tmp_path: Path) -> AsyncArray:
    """The same 4x4 / 2x2 fixture, written with zarr-python and opened async."""
    path = tmp_path / "a.zarr"
    z = zarr.create_array(
        store=str(path),
        shape=(4, 4),
        chunks=(2, 2),
        dtype="int32",
        fill_value=0,
    )
    z[:] = FULL
    return await AsyncArray.open(LocalStore(str(path)))


async def _read_all(array: AsyncArray) -> NDArray[np.int32]:
    return (await array[:, :]).to_numpy()


async def test_async_erases_a_row_of_the_chunk_grid(tmp_path: Path):
    array = await _async_chunked_array(tmp_path)

    await array.erase_chunks((0, slice(None)))

    expected = FULL.copy()
    expected[0:2, :] = 0
    np.testing.assert_array_equal(await _read_all(array), expected)


async def test_async_ellipsis_erases_every_chunk(tmp_path: Path):
    array = await _async_chunked_array(tmp_path)

    await array.erase_chunks(...)

    np.testing.assert_array_equal(await _read_all(array), EMPTY)


async def test_async_index_beyond_the_chunk_grid_raises(tmp_path: Path):
    array = await _async_chunked_array(tmp_path)

    with pytest.raises(IndexError, match="axis 0 with size 2"):
        await array.erase_chunks((3, 0))
