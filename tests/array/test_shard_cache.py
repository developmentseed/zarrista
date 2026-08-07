"""Caching the shard indexes of a sharded array.

To read one subchunk, zarrista first reads the index of the shard that holds it.
A `ShardCache` keeps that index, so later reads of the same shard do not read it
again. `Array.shard_cache` constructs a cache, and `retrieve_subchunk` and
`retrieve_encoded_subchunk` accept one through `shard_cache`.
"""

import numpy as np
import pytest
from obstore.store import LocalStore

from zarrista import (
    Array,
    ArrayBuilder,
    ArrayBytes,
    AsyncArray,
    AsyncShardCache,
    ChunkGrid,
    DataType,
    FillValue,
    ShardCache,
)
from zarrista.store import MemoryStore

DATA = np.arange(32, dtype="int32").reshape(8, 4)


def _builder() -> ArrayBuilder:
    """An 8x4 int32 array: two 4x4 shards, each split into 2x2 subchunks.

    The subchunk grid is 4x2. Subchunk rows 0 and 1 live in shard [0, 0], and
    subchunk rows 2 and 3 live in shard [1, 0].
    """
    return ArrayBuilder(
        ChunkGrid.regular([8, 4], [4, 4]),
        DataType.from_string("int32"),
        FillValue(b"\x00\x00\x00\x00"),
    ).subchunk_shape([2, 2])


def _sharded_array() -> Array:
    arr = _builder().create(MemoryStore(), "/a")
    arr.store_chunk([0, 0], ArrayBytes(DATA[:4].tobytes()))
    arr.store_chunk([1, 0], ArrayBytes(DATA[4:].tobytes()))
    return arr


def _expected(r: int, c: int) -> np.ndarray:
    return DATA[r * 2 : r * 2 + 2, c * 2 : c * 2 + 2]


def test_a_new_cache_is_empty():
    cache = _sharded_array().shard_cache()

    assert isinstance(cache, ShardCache)
    assert cache.is_empty()
    assert cache.size() == 0


def test_the_cache_holds_one_index_for_each_shard():
    arr = _sharded_array()
    cache = arr.shard_cache()

    arr.retrieve_subchunk([0, 0], shard_cache=cache)
    assert cache.size() == 1
    assert not cache.is_empty()

    # A different subchunk of the same shard reuses the cached shard index.
    arr.retrieve_subchunk([1, 1], shard_cache=cache)
    assert cache.size() == 1

    # A subchunk of the other shard adds the index of that shard.
    arr.retrieve_subchunk([2, 0], shard_cache=cache)
    assert cache.size() == 2


def test_a_cached_read_returns_the_same_data():
    arr = _sharded_array()
    cache = arr.shard_cache()

    for r in range(4):
        for c in range(2):
            cached = arr.retrieve_subchunk([r, c], shard_cache=cache)
            np.testing.assert_array_equal(np.asarray(cached), _expected(r, c))
            np.testing.assert_array_equal(
                np.asarray(cached),
                np.asarray(arr.retrieve_subchunk([r, c])),
            )


def test_retrieve_encoded_subchunk_accepts_a_cache():
    arr = _sharded_array()
    cache = arr.shard_cache()

    chunk = arr.retrieve_encoded_subchunk([3, 1], shard_cache=cache)

    assert chunk is not None
    assert cache.size() == 1
    np.testing.assert_array_equal(np.asarray(chunk.decode()), _expected(3, 1))


def test_clear_empties_the_cache():
    arr = _sharded_array()
    cache = arr.shard_cache()
    arr.retrieve_subchunk([0, 0], shard_cache=cache)

    cache.clear()

    assert cache.is_empty()
    assert cache.size() == 0
    # The cache stays usable after it is cleared.
    arr.retrieve_subchunk([0, 0], shard_cache=cache)
    assert cache.size() == 1


def test_each_array_constructs_its_own_cache():
    arr = _sharded_array()

    assert arr.shard_cache() is not arr.shard_cache()


def test_repr_names_the_array_that_created_the_cache():
    arr = _sharded_array()

    assert repr(arr.shard_cache()) == f"ShardCache(array={arr!r})"


def test_a_value_that_is_not_a_shard_cache_raises():
    arr = _sharded_array()

    with pytest.raises(TypeError, match="ShardCache"):
        arr.retrieve_subchunk([0, 0], shard_cache="not a cache")


# --- async ---------------------------------------------------------------


async def _async_sharded_array(tmp_path) -> AsyncArray:
    arr = await _builder().create_async(LocalStore(str(tmp_path)), "/a")
    await arr.store_chunk([0, 0], ArrayBytes(DATA[:4].tobytes()))
    await arr.store_chunk([1, 0], ArrayBytes(DATA[4:].tobytes()))
    return arr


async def test_async_a_new_cache_is_empty(tmp_path):
    arr = await _async_sharded_array(tmp_path)

    cache = arr.shard_cache()

    assert isinstance(cache, AsyncShardCache)
    assert await cache.is_empty()
    assert await cache.size() == 0


async def test_async_the_cache_holds_one_index_for_each_shard(tmp_path):
    arr = await _async_sharded_array(tmp_path)
    cache = arr.shard_cache()

    await arr.retrieve_subchunk([0, 0], shard_cache=cache)
    assert await cache.size() == 1

    await arr.retrieve_subchunk([1, 1], shard_cache=cache)
    assert await cache.size() == 1

    await arr.retrieve_encoded_subchunk([2, 0], shard_cache=cache)
    assert await cache.size() == 2


async def test_async_a_cached_read_returns_the_same_data(tmp_path):
    arr = await _async_sharded_array(tmp_path)
    cache = arr.shard_cache()

    sub = await arr.retrieve_subchunk([3, 1], shard_cache=cache)

    np.testing.assert_array_equal(np.asarray(sub), _expected(3, 1))


async def test_async_clear_empties_the_cache(tmp_path):
    arr = await _async_sharded_array(tmp_path)
    cache = arr.shard_cache()
    await arr.retrieve_subchunk([0, 0], shard_cache=cache)

    await cache.clear()

    assert await cache.is_empty()


async def test_async_repr_names_the_array_that_created_the_cache(tmp_path):
    arr = await _async_sharded_array(tmp_path)

    assert repr(arr.shard_cache()) == f"AsyncShardCache(array={arr!r})"


async def test_async_a_value_that_is_not_a_shard_cache_raises(tmp_path):
    arr = await _async_sharded_array(tmp_path)

    with pytest.raises(TypeError, match="AsyncShardCache"):
        await arr.retrieve_subchunk([0, 0], shard_cache="not a cache")
