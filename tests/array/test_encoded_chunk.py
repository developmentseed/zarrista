"""Reading a chunk's encoded bytes and decoding them separately.

`retrieve_encoded_chunk` returns an `EncodedChunk`: the raw stored bytes plus the
codec chain, data type, fill value, and shape that decode them. Reading the bytes
is IO-bound and decoding them is CPU-bound, so the two steps can run in different
places.
"""

from pathlib import Path

import numpy as np
import pytest
from obstore.store import LocalStore

from zarrista import (
    Array,
    ArrayBuilder,
    ArrayBytes,
    ChunkGrid,
    DataType,
    EncodedChunk,
    FillValue,
    Tensor,
    ThreadPool,
    codec,
)
from zarrista.store import MemoryStore

DATA = np.arange(16, dtype="int32").reshape(4, 4)


def _array(*, compressed: bool = False) -> Array:
    """A 4x4 int32 array in one 4x4 chunk."""
    builder = ArrayBuilder(
        ChunkGrid.regular([4, 4], [4, 4]),
        DataType.from_string("int32"),
        FillValue(b"\x00\x00\x00\x00"),
    )
    if compressed:
        builder = builder.compressors([codec.zstd(3, checksum=False)])
    return builder.create(MemoryStore(), "/a")


def _written(*, compressed: bool = False) -> Array:
    arr = _array(compressed=compressed)
    arr.store_chunk([0, 0], ArrayBytes(DATA.tobytes()))
    return arr


def _encoded(arr: Array) -> EncodedChunk:
    """The encoded chunk at `[0, 0]`, which the caller knows is present.

    `retrieve_encoded_chunk` returns `None` for an absent chunk, because an
    absent chunk is a normal state that means "all fill value". These tests
    write the chunk first, so narrow the type once here.
    """
    chunk = arr.retrieve_encoded_chunk([0, 0])
    assert chunk is not None
    return chunk


def test_retrieve_encoded_chunk_returns_an_encoded_chunk():
    chunk = _encoded(_written())

    assert isinstance(chunk, EncodedChunk)
    assert chunk.shape == [4, 4]
    assert chunk.data_type == DataType.from_string("int32")
    # Uncompressed int32: 16 elements of 4 bytes.
    assert len(chunk.buffer) == 64


def test_decode_matches_retrieve_chunk():
    arr = _written()

    decoded = _encoded(arr).decode()

    assert isinstance(decoded, Tensor)
    np.testing.assert_array_equal(np.asarray(decoded), DATA)
    np.testing.assert_array_equal(
        np.asarray(decoded),
        np.asarray(arr.retrieve_chunk([0, 0])),
    )


def test_decode_runs_the_full_codec_chain():
    """A compressed chunk decodes correctly, so the chain is really applied."""
    arr = _written(compressed=True)
    chunk = _encoded(arr)

    # zstd shrinks this data, so the stored bytes are not the raw 64.
    assert len(chunk.buffer) != 64
    assert [c.name for c in chunk.codecs.compressors] == ["zstd"]

    np.testing.assert_array_equal(np.asarray(chunk.decode()), DATA)


def test_codecs_describes_the_chain_that_decodes_it():
    chunk = _encoded(_written(compressed=True))

    assert chunk.codecs.serializer.name == "bytes"
    assert [f.name for f in chunk.codecs.filters] == []


def test_absent_chunk_is_none():
    assert _array().retrieve_encoded_chunk([0, 0]) is None


def test_decode_accepts_codec_options():
    chunk = _encoded(_written(compressed=True))

    decoded = chunk.decode(validate_checksums=False)

    np.testing.assert_array_equal(np.asarray(decoded), DATA)


def test_decode_rejects_an_unknown_codec_option():
    chunk = _encoded(_written())

    with pytest.raises(TypeError):
        chunk.decode(not_a_codec_option=True)


def test_encoded_chunk_survives_the_array_going_away():
    """The chunk holds no reference to the array or the store."""

    def read() -> EncodedChunk:
        return _encoded(_written())

    np.testing.assert_array_equal(np.asarray(read().decode()), DATA)


async def test_decode_async_matches_decode():
    chunk = _encoded(_written(compressed=True))

    decoded = await chunk.decode_async()

    np.testing.assert_array_equal(np.asarray(decoded), DATA)


async def test_decode_async_on_a_dedicated_thread_pool():
    chunk = _encoded(_written(compressed=True))

    decoded = await chunk.decode_async(pool=ThreadPool(2))

    np.testing.assert_array_equal(np.asarray(decoded), DATA)


async def test_decode_async_accepts_codec_options():
    chunk = _encoded(_written())

    decoded = await chunk.decode_async(concurrent_target=1)

    np.testing.assert_array_equal(np.asarray(decoded), DATA)


async def test_async_array_returns_an_encoded_chunk(tmp_path: Path):
    """`AsyncArray.retrieve_encoded_chunk` mirrors the sync method."""
    arr = await (
        ArrayBuilder(
            ChunkGrid.regular([4, 4], [4, 4]),
            DataType.from_string("int32"),
            FillValue(b"\x00\x00\x00\x00"),
        )
        .compressors([codec.zstd(3, checksum=False)])
        .create_async(LocalStore(str(tmp_path)), "/a")
    )
    await arr.store_chunk([0, 0], ArrayBytes(DATA.tobytes()))

    chunk = await arr.retrieve_encoded_chunk([0, 0])

    assert isinstance(chunk, EncodedChunk)
    assert chunk.shape == [4, 4]
    np.testing.assert_array_equal(np.asarray(await chunk.decode_async()), DATA)


async def test_async_array_absent_chunk_is_none(tmp_path: Path):
    arr = await ArrayBuilder(
        ChunkGrid.regular([4, 4], [4, 4]),
        DataType.from_string("int32"),
        FillValue(b"\x00\x00\x00\x00"),
    ).create_async(LocalStore(str(tmp_path)), "/a")

    assert await arr.retrieve_encoded_chunk([0, 0]) is None
