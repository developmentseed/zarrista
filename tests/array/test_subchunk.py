"""Reading individual subchunks (inner chunks) of a sharded array.

`retrieve_subchunk` decodes one inner chunk of a shard; `retrieve_encoded_subchunk`
returns its raw stored bytes. Subchunk indices address the subchunk grid, which
spans the whole array (see `subchunk_grid_shape`).
"""

import numpy as np

from zarrista import (
    Array,
    ArrayBuilder,
    ArrayBytes,
    ChunkGrid,
    DataType,
    FillValue,
    Tensor,
)
from zarrista.store import MemoryStore


def _sharded_array(store: MemoryStore | None = None) -> Array:
    """A 4x4 int32 array: one 4x4 shard split into 2x2 subchunks (a 2x2 grid)."""
    return (
        ArrayBuilder(
            ChunkGrid.regular([4, 4], [4, 4]),
            DataType.from_string("int32"),
            FillValue(b"\x00\x00\x00\x00"),
        )
        .subchunk_shape([2, 2])
        .create(store if store is not None else MemoryStore(), "/a")
    )


def test_retrieve_subchunk_decodes_each_inner_chunk():
    arr = _sharded_array()
    data = np.arange(16, dtype="int32").reshape(4, 4)
    arr.store_chunk([0, 0], ArrayBytes(data.tobytes()))

    assert arr.is_sharded
    assert arr.subchunk_grid_shape == [2, 2]

    for r in range(2):
        for c in range(2):
            sub = arr.retrieve_subchunk([r, c])
            assert isinstance(sub, Tensor)
            expected = data[r * 2 : r * 2 + 2, c * 2 : c * 2 + 2]
            np.testing.assert_array_equal(sub.to_numpy(), expected)


def test_retrieve_encoded_subchunk_returns_raw_bytes():
    arr = _sharded_array()
    data = np.arange(16, dtype="int32").reshape(4, 4)
    arr.store_chunk([0, 0], ArrayBytes(data.tobytes()))

    # No compressor, so the encoded inner chunk is the raw little-endian data.
    encoded = arr.retrieve_encoded_subchunk([1, 0])
    assert encoded is not None
    decoded = np.frombuffer(bytes(encoded), dtype="int32").reshape(2, 2)
    np.testing.assert_array_equal(decoded, data[2:4, 0:2])


def test_retrieve_encoded_subchunk_absent_is_none():
    arr = _sharded_array()  # nothing written
    assert arr.retrieve_encoded_subchunk([0, 0]) is None


# --- async ---------------------------------------------------------------

from obstore.store import LocalStore  # noqa: E402

from zarrista import AsyncArray  # noqa: E402


async def _async_sharded_array(tmp_path) -> AsyncArray:
    return await (
        ArrayBuilder(
            ChunkGrid.regular([4, 4], [4, 4]),
            DataType.from_string("int32"),
            FillValue(b"\x00\x00\x00\x00"),
        )
        .subchunk_shape([2, 2])
        .create_async(LocalStore(str(tmp_path)), "/a")
    )


async def test_async_retrieve_subchunk(tmp_path):
    arr = await _async_sharded_array(tmp_path)
    data = np.arange(16, dtype="int32").reshape(4, 4)
    await arr.store_chunk([0, 0], ArrayBytes(data.tobytes()))

    sub = await arr.retrieve_subchunk([1, 1])
    assert isinstance(sub, Tensor)
    np.testing.assert_array_equal(sub.to_numpy(), data[2:4, 2:4])

    encoded = await arr.retrieve_encoded_subchunk([1, 1])
    assert encoded is not None
    decoded = np.frombuffer(bytes(encoded), dtype="int32").reshape(2, 2)
    np.testing.assert_array_equal(decoded, data[2:4, 2:4])
