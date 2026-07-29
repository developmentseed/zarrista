"""`Array.read_only()` returns an array that reads normally but raises on writes."""

import numpy as np
import pytest

from zarrista import Array, ArrayBuilder, ArrayBytes, ChunkGrid, DataType, FillValue
from zarrista.exceptions import ZarristaError
from zarrista.store import MemoryStore


def _writable_array() -> Array:
    """A 4x4 int8 array (single 4x4 chunk, fill 0) created in a MemoryStore."""
    return ArrayBuilder(
        ChunkGrid.regular([4, 4], [4, 4]),
        DataType.from_string("int8"),
        FillValue(b"\x00"),
    ).create(MemoryStore(), "/a")


def test_read_only_still_reads():
    array = _writable_array()
    array.store_chunk([0, 0], ArrayBytes(np.arange(16, dtype="int8").tobytes()))

    ro = array.read_only()

    np.testing.assert_array_equal(
        ro.retrieve_chunk([0, 0]).to_numpy(),
        np.arange(16, dtype="int8").reshape(4, 4),
    )
    assert ro.shape == [4, 4]


def test_read_only_store_chunk_raises():
    # The underlying StorageError::ReadOnly may be wrapped by the Array layer
    # (e.g. into an ArrayError); assert the read-only failure surfaces, not the
    # exact wrapper class.
    ro = _writable_array().read_only()
    with pytest.raises(ZarristaError, match="read only"):
        ro.store_chunk([0, 0], ArrayBytes(np.zeros(16, dtype="int8").tobytes()))


def test_read_only_erase_metadata_raises():
    # `ArrayBuilder.create` already stored metadata, so erasing it is a real write.
    ro = _writable_array().read_only()
    with pytest.raises(ZarristaError, match="read only"):
        ro.erase_metadata()


def test_writable_array_still_writes():
    """The original array (and any non-read-only array) writes without error."""
    array = _writable_array()
    array.store_chunk([0, 0], ArrayBytes(np.ones(16, dtype="int8").tobytes()))
    np.testing.assert_array_equal(
        array.retrieve_chunk([0, 0]).to_numpy(),
        np.ones((4, 4), dtype="int8"),
    )


# --- async ---------------------------------------------------------------

import zarr  # noqa: E402
from obstore.store import LocalStore  # noqa: E402

from zarrista import AsyncArray  # noqa: E402


async def _async_writable_array(tmp_path) -> AsyncArray:
    """A 4x4 int8 array written with zarr-python, opened async over obstore."""
    path = tmp_path / "a.zarr"
    z = zarr.create_array(store=str(path), shape=(4, 4), chunks=(4, 4), dtype="int8")
    z[:] = np.arange(16, dtype="int8").reshape(4, 4)
    return await AsyncArray.open_async(LocalStore(str(path)))


async def test_async_read_only_still_reads(tmp_path):
    arr = await _async_writable_array(tmp_path)
    ro = arr.read_only()
    chunk = await ro.retrieve_chunk([0, 0])
    np.testing.assert_array_equal(
        chunk.to_numpy(),
        np.arange(16, dtype="int8").reshape(4, 4),
    )


async def test_async_read_only_store_chunk_raises(tmp_path):
    arr = await _async_writable_array(tmp_path)
    ro = arr.read_only()
    with pytest.raises(ZarristaError, match="read only"):
        await ro.store_chunk([0, 0], ArrayBytes(np.zeros(16, dtype="int8").tobytes()))
