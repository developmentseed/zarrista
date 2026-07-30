"""`Array.with_chunk_grid()` returns a new array with a different chunk grid."""

import numpy as np

from zarrista import Array, ArrayBuilder, ArrayBytes, ChunkGrid, DataType, FillValue
from zarrista.store import MemoryStore


def _array(store: MemoryStore) -> Array:
    """A 4x4 int8 array in `store` at `/a`, chunked 2x2, fill 0.

    `ArrayBuilder.create` writes the metadata, so the array is openable.
    """
    return ArrayBuilder(
        ChunkGrid.regular([4, 4], [2, 2]),
        DataType.from_string("int8"),
        FillValue(b"\x00"),
    ).create(store, "/a")


def test_returns_new_array_and_leaves_original() -> None:
    array = _array(MemoryStore())

    regridded = array.with_chunk_grid(ChunkGrid.regular([8, 8], [4, 4]))

    assert regridded.shape == [8, 8]
    assert regridded.chunk_grid.metadata == ChunkGrid.regular([8, 8], [4, 4]).metadata
    assert array.shape == [4, 4]
    assert array.chunk_grid.metadata == ChunkGrid.regular([4, 4], [2, 2]).metadata


def test_does_not_write() -> None:
    store = MemoryStore()
    array = _array(store)

    array.with_chunk_grid(ChunkGrid.regular([8, 8], [4, 4]))

    # The stored metadata is untouched until `store_metadata` is called.
    reopened = Array.open(store, "/a")
    assert reopened.shape == [4, 4]
    assert reopened.chunk_grid.metadata == ChunkGrid.regular([4, 4], [2, 2]).metadata


def test_chunk_shape_change_persists() -> None:
    store = MemoryStore()
    _array(store).with_chunk_grid(ChunkGrid.regular([4, 4], [4, 4])).store_metadata()

    reopened = Array.open(store, "/a")
    assert reopened.shape == [4, 4]
    assert reopened.chunk_grid.metadata == ChunkGrid.regular([4, 4], [4, 4]).metadata


def test_shape_may_change_too() -> None:
    store = MemoryStore()
    _array(store).with_chunk_grid(ChunkGrid.regular([8, 8], [2, 2])).store_metadata()

    assert Array.open(store, "/a").shape == [8, 8]


def test_rectilinear_grid_accepted() -> None:
    array = _array(MemoryStore())
    grid = ChunkGrid.rectilinear([4, 4], [2, 2])

    regridded = array.with_chunk_grid(grid)

    assert regridded.chunk_grid.metadata == grid.metadata


def test_existing_chunks_are_not_migrated() -> None:
    # Pins the hazard the docs describe: regridding rewrites metadata only, so
    # bytes already in the store stay exactly as they were, under keys that no
    # longer describe the same region.
    store = MemoryStore()
    array = _array(store)
    # One 2x2 int8 chunk is 4 bytes.
    array.store_chunk([0, 0], ArrayBytes(np.arange(4, dtype="int8").tobytes()))

    # A 4x4 chunk would be 16 bytes.
    array.with_chunk_grid(ChunkGrid.regular([4, 4], [4, 4])).store_metadata()

    # `array` still describes the 2x2 grid, so it can still address the old chunk.
    # It is untouched: still 4 bytes, not re-encoded to 16.
    assert len(array.retrieve_encoded_chunk([0, 0])) == 4
