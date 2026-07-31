"""`Array.with_shape()` returns a new array at a new shape, without writing."""

from pathlib import Path

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
    FillValue,
)
from zarrista.exceptions import ArrayCreateError
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


@pytest.mark.parametrize("shape", [[8, 8, 8], [8]])
def test_wrong_dimensionality_raises(shape: list[int]) -> None:
    array = _array(MemoryStore())

    with pytest.raises(ArrayCreateError, match="inconsistent dimensionality"):
        array.with_shape(shape)


# --- async --------------------------------------------------------------------


async def test_async_with_shape_returns_new_array(tmp_path: Path) -> None:
    array = await ArrayBuilder(
        ChunkGrid.regular([4, 4], [2, 2]),
        DataType.from_string("int8"),
        FillValue(b"\x00"),
    ).create_async(LocalStore(str(tmp_path)), "/a")

    # `with_shape` is sync even on `AsyncArray`: it performs no I/O.
    resized = array.with_shape([8, 8])
    await resized.store_metadata()

    assert array.shape == [4, 4]
    reopened = await AsyncArray.open_async(LocalStore(str(tmp_path)), "/a")
    assert reopened.shape == [8, 8]
