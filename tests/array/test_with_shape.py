"""`Array.with_shape()` returns a new array at a new shape, without writing."""

import numpy as np
import pytest

from zarrista import Array, ArrayBuilder, ArrayBytes, ChunkGrid, DataType, FillValue
from zarrista.exceptions import ZarristaError
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


def test_with_shape_returns_new_array_and_leaves_original() -> None:
    array = _array(MemoryStore())

    resized = array.with_shape([8, 8])

    assert resized.shape == [8, 8]
    assert array.shape == [4, 4]


def test_with_shape_does_not_write() -> None:
    store = MemoryStore()
    array = _array(store)

    array.with_shape([8, 8])

    # The stored metadata is untouched until `store_metadata` is called.
    assert Array.open(store, "/a").shape == [4, 4]


def test_grow_then_store_metadata_persists() -> None:
    store = MemoryStore()
    _array(store).with_shape([8, 8]).store_metadata()

    assert Array.open(store, "/a").shape == [8, 8]


def test_shrink_then_store_metadata_persists() -> None:
    store = MemoryStore()
    _array(store).with_shape([2, 2]).store_metadata()

    assert Array.open(store, "/a").shape == [2, 2]


def test_data_survives_a_grow() -> None:
    store = MemoryStore()
    array = _array(store)
    chunk = np.arange(4, dtype="int8").reshape(2, 2)
    array.store_chunk([0, 0], ArrayBytes(chunk.tobytes()))

    array.with_shape([8, 8]).store_metadata()

    grown = Array.open(store, "/a")
    np.testing.assert_array_equal(grown.retrieve_chunk([0, 0]).to_numpy(), chunk)


def test_shrink_leaves_out_of_bounds_chunks() -> None:
    # Pins current behavior: reclaiming these chunks is a future vacuum call.
    store = MemoryStore()
    array = _array(store)
    array.store_chunk([1, 1], ArrayBytes(np.ones(4, dtype="int8").tobytes()))

    array.with_shape([2, 2]).store_metadata()

    # `array` still describes the 4x4 shape, so it can still address chunk [1, 1].
    assert array.retrieve_encoded_chunk([1, 1]) is not None


def test_wrong_dimensionality_raises() -> None:
    array = _array(MemoryStore())

    with pytest.raises(ZarristaError):
        array.with_shape([8, 8, 8])
