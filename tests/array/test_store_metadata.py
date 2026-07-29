"""`store_metadata()` writes the array's in-memory metadata to the store."""

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from obstore.store import LocalStore

from zarrista import Array, AsyncArray
from zarrista.exceptions import ZarristaError
from zarrista.store import FilesystemStore, MemoryStore

if TYPE_CHECKING:
    from zarr_metadata import ZarrV3ArrayMetadataJSON


def _metadata(data_type: str = "int16") -> "ZarrV3ArrayMetadataJSON":
    return {
        "zarr_format": 3,
        "attributes": {},
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (2, 2)}},
        "data_type": data_type,
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0,
        "node_type": "array",
        "shape": (4, 4),
        "codecs": ({"name": "bytes"},),
    }


def test_from_metadata_does_not_write() -> None:
    store = MemoryStore()
    Array.from_metadata(_metadata(), store)

    # Nothing was written, so there is no array to open.
    with pytest.raises(ZarristaError):
        Array.open(store)


def test_store_metadata_makes_array_openable() -> None:
    store = MemoryStore()
    array = Array.from_metadata(_metadata(), store)
    array.store_metadata()

    reopened = Array.open(store)
    assert reopened.shape == [4, 4]
    assert reopened.dtype.name == "int16"


def test_store_metadata_overwrites(tmp_path: Path) -> None:
    store = FilesystemStore(tmp_path)
    Array.from_metadata(_metadata("int16"), store).store_metadata()
    Array.from_metadata(_metadata("float64"), store).store_metadata()

    assert Array.open(FilesystemStore(tmp_path)).dtype.name == "float64"


def test_read_only_store_metadata_raises() -> None:
    store = MemoryStore()
    read_only = Array.from_metadata(_metadata(), store).read_only()
    with pytest.raises(ZarristaError, match="read only"):
        read_only.store_metadata()


# --- async --------------------------------------------------------------------


async def test_async_store_metadata_makes_array_openable(tmp_path: Path) -> None:
    store = LocalStore(str(tmp_path))
    array = AsyncArray.from_metadata(_metadata(), store)
    await array.store_metadata()

    reopened = await AsyncArray.open_async(LocalStore(str(tmp_path)))
    assert reopened.shape == [4, 4]
    assert reopened.dtype.name == "int16"


async def test_async_read_only_store_metadata_raises(tmp_path: Path) -> None:
    store = LocalStore(str(tmp_path))
    read_only = AsyncArray.from_metadata(_metadata(), store).read_only()
    with pytest.raises(ZarristaError, match="read only"):
        await read_only.store_metadata()
