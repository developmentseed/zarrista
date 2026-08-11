"""`Array.with_attrs()` returns a new array with new attributes, without writing."""

import pytest
from obstore.store import LocalStore

from zarrista import Array, ArrayBuilder, AsyncArray, ChunkGrid, DataType, FillValue
from zarrista.store import MemoryStore


def _builder() -> ArrayBuilder:
    """A 4x4 int8 array, chunked 2x2, fill 0, with two user attributes."""
    return ArrayBuilder(
        ChunkGrid.regular([4, 4], chunk_shape=[2, 2]),
        DataType.from_string("int8"),
        FillValue(b"\x00"),
    ).attrs({"units": "m", "long_name": "height"})


def test_with_attrs_replaces_attrs_and_leaves_the_original() -> None:
    array = _builder().create(MemoryStore(), "/a")

    updated = array.with_attrs({"units": "km"})

    # `long_name` is absent, so the new attributes replace rather than merge.
    assert updated.attrs == {"units": "km"}
    assert array.attrs == {"units": "m", "long_name": "height"}
    assert updated.shape == array.shape


def test_nothing_is_written_until_store_metadata() -> None:
    store = MemoryStore()
    updated = _builder().create(store, "/a").with_attrs({"units": "km"})
    assert Array.open(store, "/a").attrs["units"] == "m"

    updated.store_metadata()

    stored = Array.open(store, "/a").attrs
    assert stored["units"] == "km"
    assert "long_name" not in stored
    # `store_metadata` also records the zarrs version, which zarrista cannot
    # yet suppress. Remove this once `ArrayMetadataOptions` is exposed.
    assert "_zarrs" in stored


def test_a_value_that_is_not_json_serializable_raises() -> None:
    array = _builder().create(MemoryStore(), "/a")

    with pytest.raises(TypeError):
        array.with_attrs({"bad": object()})


async def test_async_with_attrs(tmp_path) -> None:
    store = LocalStore(str(tmp_path))
    array = await _builder().create_async(store, "/a")

    updated = array.with_attrs({"units": "km"})
    await updated.store_metadata()

    assert updated.attrs == {"units": "km"}
    assert array.attrs == {"units": "m", "long_name": "height"}
    assert (await AsyncArray.open(store, "/a")).attrs["units"] == "km"
