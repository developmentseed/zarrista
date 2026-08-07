"""Reading a Zarr store held inside a zip file.

A tiny int32 array is written with zarr-python, packed into a zip file, and then
read back through `ZipStore`. The store is read-only: every write must fail.
"""

import zipfile
from pathlib import Path

import numpy as np
import pytest
import zarr
from obstore.store import LocalStore

from zarrista import Array, AsyncArray
from zarrista.exceptions import StorageError
from zarrista.store import AsyncZipStore, FilesystemStore, ZipStore

EXPECTED = np.arange(4, dtype="int32")


def _zip_zarr(
    tmp_path: Path,
    name: str,
    compression: int = zipfile.ZIP_STORED,
    prefix: str = "",
) -> Path:
    """Write a tiny array and pack it into `name`; returns the containing dir."""
    zarr_dir = tmp_path / "src" / "a.zarr"
    z = zarr.create_array(store=str(zarr_dir), shape=(4,), chunks=(2,), dtype="int32")
    z[:] = EXPECTED
    with zipfile.ZipFile(tmp_path / name, "w", compression) as zf:
        for entry in sorted(zarr_dir.rglob("*")):
            if entry.is_file():
                zf.write(entry, prefix + str(entry.relative_to(zarr_dir)))
    return tmp_path


@pytest.fixture
def zipped(tmp_path: Path) -> Path:
    """A stored (uncompressed) zip of a Zarr array; returns the containing dir."""
    return _zip_zarr(tmp_path, "a.zip")


@pytest.mark.parametrize(
    "compression",
    [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED],
    ids=["stored", "deflated"],
)
def test_reads_through_zip(tmp_path: Path, compression: int):
    root = _zip_zarr(tmp_path, "a.zip", compression)
    array = Array.open(ZipStore(FilesystemStore(root), "a.zip"))
    np.testing.assert_array_equal(array[:].to_numpy(), EXPECTED)


def test_open_matches_constructor(zipped: Path):
    store = ZipStore.open(FilesystemStore(zipped), "a.zip")
    np.testing.assert_array_equal(Array.open(store)[:].to_numpy(), EXPECTED)


def test_repr(zipped: Path):
    assert repr(ZipStore(FilesystemStore(zipped), "a.zip")) == "ZipStore(a.zip)"


def test_array_storage_round_trips(zipped: Path):
    store = ZipStore(FilesystemStore(zipped), "a.zip")
    assert isinstance(Array.open(store).storage, ZipStore)


def test_writes_are_rejected(zipped: Path):
    array = Array.open(ZipStore(FilesystemStore(zipped), "a.zip"))
    with pytest.raises(StorageError, match="read only store"):
        array.store_metadata()


@pytest.mark.parametrize("path", ["nested/", "nested", "/nested/", "//nested"])
def test_path_selects_a_directory_in_the_zip(tmp_path: Path, path: str):
    """Leading and trailing slashes are normalized, so every spelling agrees."""
    root = _zip_zarr(tmp_path, "a.zip", prefix="nested/")
    store = ZipStore(FilesystemStore(root), "a.zip", path=path)
    np.testing.assert_array_equal(Array.open(store)[:].to_numpy(), EXPECTED)


def test_path_is_keyword_only(zipped: Path):
    with pytest.raises(TypeError):
        ZipStore(FilesystemStore(zipped), "a.zip", "nested/")


def test_zip_inside_a_zip(zipped: Path):
    with zipfile.ZipFile(zipped / "outer.zip", "w") as zf:
        zf.write(zipped / "a.zip", "a.zip")

    outer = ZipStore(FilesystemStore(zipped), "outer.zip")
    inner = ZipStore(outer, "a.zip")
    np.testing.assert_array_equal(Array.open(inner)[:].to_numpy(), EXPECTED)


def test_missing_key_is_rejected(zipped: Path):
    with pytest.raises(StorageError):
        ZipStore(FilesystemStore(zipped), "absent.zip")


def test_non_zip_value_is_rejected(tmp_path: Path):
    (tmp_path / "a.zip").write_bytes(b"not a zip file")
    with pytest.raises(StorageError):
        ZipStore(FilesystemStore(tmp_path), "a.zip")


def test_invalid_key_is_rejected(zipped: Path):
    with pytest.raises(ValueError, match="invalid store key"):
        ZipStore(FilesystemStore(zipped), "/leading-slash")


# --- async --------------------------------------------------------------------


async def test_async_reads_through_zip(zipped: Path):
    store = await AsyncZipStore.open(LocalStore(str(zipped)), "a.zip")
    array = await AsyncArray.open(store)
    np.testing.assert_array_equal((await array[:]).to_numpy(), EXPECTED)


async def test_async_writes_are_rejected(zipped: Path):
    store = await AsyncZipStore.open(LocalStore(str(zipped)), "a.zip")
    array = await AsyncArray.open(store)
    with pytest.raises(StorageError, match="read only store"):
        await array.store_metadata()
