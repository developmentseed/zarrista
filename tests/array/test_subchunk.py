"""Reading individual subchunks (inner chunks) of a sharded array.

`retrieve_subchunk` decodes one inner chunk of a shard. `retrieve_encoded_subchunk`
returns its stored bytes as an `EncodedChunk`, paired with the codec chain that
decodes them. Subchunk indices address the subchunk grid, which spans the whole
array (see `subchunk_grid_shape`).
"""

import copy

import numpy as np
import pytest
from obstore.store import LocalStore

from zarrista import (
    Array,
    ArrayBuilder,
    ArrayBytes,
    AsyncArray,
    ChunkGrid,
    DataType,
    EncodedChunk,
    FillValue,
    FixedLengthTensor,
    codec,
)
from zarrista.exceptions import ArrayError
from zarrista.store import MemoryStore

DATA = np.arange(16, dtype="int32").reshape(4, 4)


def _builder(*, compressed: bool = False) -> ArrayBuilder:
    """A 4x4 int32 array: one 4x4 shard split into 2x2 subchunks (a 2x2 grid)."""
    builder = ArrayBuilder(
        ChunkGrid.regular([4, 4], [4, 4]),
        DataType.from_string("int32"),
        FillValue(b"\x00\x00\x00\x00"),
    ).subchunk_shape([2, 2])
    if compressed:
        # The builder nests codecs inside the sharding codec, so this compresses
        # each subchunk and leaves the array exclusively sharded.
        builder = builder.compressors([codec.zstd(3, checksum=False)])
    return builder


def _sharded_array(*, compressed: bool = False, written: bool = True) -> Array:
    arr = _builder(compressed=compressed).create(MemoryStore(), "/a")
    if written:
        arr.store_chunk([0, 0], ArrayBytes(DATA.tobytes()))
    return arr


def _encoded(arr: Array, subchunk_indices: list[int]) -> EncodedChunk:
    """The encoded subchunk, which the caller knows is present."""
    chunk = arr.retrieve_encoded_subchunk(subchunk_indices)
    assert chunk is not None
    return chunk


def _expected(r: int, c: int) -> np.ndarray:
    return DATA[r * 2 : r * 2 + 2, c * 2 : c * 2 + 2]


def test_retrieve_subchunk_decodes_each_inner_chunk():
    arr = _sharded_array()

    assert arr.is_sharded
    assert arr.subchunk_grid_shape == [2, 2]

    for r in range(2):
        for c in range(2):
            sub = arr.retrieve_subchunk([r, c])
            assert isinstance(sub, FixedLengthTensor)
            np.testing.assert_array_equal(sub.to_numpy(), _expected(r, c))


def test_retrieve_encoded_subchunk_returns_an_encoded_chunk():
    chunk = _encoded(_sharded_array(), [1, 0])

    assert isinstance(chunk, EncodedChunk)
    # The subchunk shape, not the shard shape.
    assert chunk.shape == [2, 2]
    assert chunk.data_type == DataType.from_string("int32")
    # No compressor, so the encoded inner chunk is the raw little-endian data.
    assert len(chunk.buffer) == 16


def test_decode_matches_retrieve_subchunk():
    arr = _sharded_array()

    for r in range(2):
        for c in range(2):
            decoded = _encoded(arr, [r, c]).decode()
            np.testing.assert_array_equal(np.asarray(decoded), _expected(r, c))
            np.testing.assert_array_equal(
                np.asarray(decoded),
                np.asarray(arr.retrieve_subchunk([r, c])),
            )


def test_decode_runs_the_subchunk_codec_chain():
    """A compressed subchunk decodes correctly, so the inner chain is applied."""
    arr = _sharded_array(compressed=True)
    chunk = _encoded(arr, [1, 1])

    # zstd changes the size, so these are not the raw 16 bytes.
    assert len(chunk.buffer) != 16
    np.testing.assert_array_equal(np.asarray(chunk.decode()), _expected(1, 1))


def test_codecs_is_the_subchunk_chain_not_the_array_chain():
    """The chain decodes one subchunk, so it is the sharding codec's inner chain."""
    arr = _sharded_array(compressed=True)

    assert arr.codecs.serializer.name == "sharding_indexed"

    chunk = _encoded(arr, [0, 0])
    assert chunk.codecs.serializer.name == "bytes"
    assert [c.name for c in chunk.codecs.compressors] == ["zstd"]


def test_absent_subchunk_is_none():
    arr = _sharded_array(written=False)

    assert arr.retrieve_encoded_subchunk([0, 0]) is None


def test_decode_accepts_codec_options():
    chunk = _encoded(_sharded_array(compressed=True), [0, 1])

    decoded = chunk.decode(validate_checksums=False)

    np.testing.assert_array_equal(np.asarray(decoded), _expected(0, 1))


def _array_with_outer_codec(codecs: list) -> Array:
    """An array whose sharding codec is not the only codec.

    `ArrayBuilder` nests codecs inside the sharding codec, so it cannot build
    one of these. Write the metadata by hand instead.
    """
    metadata = copy.deepcopy(_sharded_array(written=False).metadata)
    metadata["codecs"] = codecs
    arr = Array.from_metadata(metadata, MemoryStore(), "/a")
    arr.store_metadata()
    arr.store_chunk([0, 0], ArrayBytes(DATA.tobytes()))
    return arr


@pytest.mark.parametrize("position", ["filter", "compressor"])
def test_not_exclusively_sharded_raises(position: str):
    """Only an exclusively sharded array has byte ranges that hold one subchunk.

    A codec outside the sharding codec re-encodes the whole shard, so no byte
    range of the shard holds the bytes of one subchunk.
    """
    sharding = copy.deepcopy(_sharded_array(written=False).metadata["codecs"])
    codecs = (
        [{"name": "transpose", "configuration": {"order": [1, 0]}}, *sharding]
        if position == "filter"
        else [*sharding, {"name": "crc32c"}]
    )
    arr = _array_with_outer_codec(codecs)

    assert arr.is_sharded
    with pytest.raises(ArrayError, match="not exclusively sharded"):
        arr.retrieve_encoded_subchunk([0, 0])


# --- async ---------------------------------------------------------------


async def _async_sharded_array(tmp_path) -> AsyncArray:
    arr = await _builder().create_async(LocalStore(str(tmp_path)), "/a")
    await arr.store_chunk([0, 0], ArrayBytes(DATA.tobytes()))
    return arr


async def test_async_retrieve_subchunk(tmp_path):
    arr = await _async_sharded_array(tmp_path)

    sub = await arr.retrieve_subchunk([1, 1])
    assert isinstance(sub, FixedLengthTensor)
    np.testing.assert_array_equal(sub.to_numpy(), _expected(1, 1))


async def test_async_retrieve_encoded_subchunk(tmp_path):
    arr = await _async_sharded_array(tmp_path)

    chunk = await arr.retrieve_encoded_subchunk([1, 1])

    assert isinstance(chunk, EncodedChunk)
    assert chunk.shape == [2, 2]
    np.testing.assert_array_equal(
        np.asarray(await chunk.decode_async()),
        _expected(1, 1),
    )


async def test_async_absent_subchunk_is_none(tmp_path):
    arr = await _builder().create_async(LocalStore(str(tmp_path)), "/a")

    assert await arr.retrieve_encoded_subchunk([0, 0]) is None
