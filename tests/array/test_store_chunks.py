"""`store_chunks` writes a region of the chunk grid.

Its selection is in **chunk-grid** coordinates, like `erase_chunks`, but `data`
covers the *elements* that those chunks span. These tests pin that distinction:
selecting one chunk of a 4x4 array chunked 2x2 needs 2x2 elements of data, not
1x1.
"""

import numpy as np
import pytest
from obstore.store import LocalStore

from zarrista import Array, ArrayBuilder, ChunkGrid, DataType, FillValue
from zarrista.store import MemoryStore

DATA = np.arange(16, dtype="int32").reshape(4, 4)


def _array() -> Array:
    """A 4x4 int32 array chunked 2x2, so the chunk grid is 2x2."""
    return ArrayBuilder(
        ChunkGrid.regular([4, 4], [2, 2]),
        DataType.from_string("int32"),
        FillValue(b"\x00\x00\x00\x00"),
    ).create(MemoryStore(), "/a")


def test_writes_the_whole_chunk_grid() -> None:
    array = _array()

    array.store_chunks((slice(None), slice(None)), DATA)

    np.testing.assert_array_equal(array[:, :].to_numpy(), DATA)


def test_writes_a_single_chunk_into_its_element_region() -> None:
    """Chunk (0, 1) covers elements [0:2, 2:4], so it takes 2x2 elements."""
    array = _array()

    # A sliced view is not C-contiguous, which `store_chunks` rejects.
    array.store_chunks((0, 1), np.ascontiguousarray(DATA[0:2, 2:4]))

    expected = np.zeros((4, 4), dtype="int32")
    expected[0:2, 2:4] = DATA[0:2, 2:4]
    np.testing.assert_array_equal(array[:, :].to_numpy(), expected)


def test_data_shaped_like_the_chunk_grid_raises() -> None:
    """The destination shape is in elements, so 2x2 chunks want 4x4 elements."""
    array = _array()

    with pytest.raises(ValueError, match=r"destination has shape \[4, 4\]"):
        array.store_chunks((slice(None), slice(None)), np.zeros((2, 2), dtype="int32"))


async def test_async_writes_a_single_chunk_into_its_element_region(tmp_path) -> None:
    store = LocalStore(str(tmp_path))
    array = await ArrayBuilder(
        ChunkGrid.regular([4, 4], [2, 2]),
        DataType.from_string("int32"),
        FillValue(b"\x00\x00\x00\x00"),
    ).create_async(store, "/a")

    await array.store_chunks((0, 1), np.ascontiguousarray(DATA[0:2, 2:4]))

    expected = np.zeros((4, 4), dtype="int32")
    expected[0:2, 2:4] = DATA[0:2, 2:4]
    np.testing.assert_array_equal((await array[:, :]).to_numpy(), expected)
