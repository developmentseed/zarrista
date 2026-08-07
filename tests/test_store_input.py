"""Store input acceptance: built-in stores open; bad objects are rejected.

Verifies that widening the wrapped storage to the writable trait object did not
change which Python objects `Array.open` / `Group.open` accept.
"""

from pathlib import Path

import numpy as np
import pytest
import zarr

from zarrista import Array, Group
from zarrista.store import FilesystemStore


@pytest.fixture
def array_path(tmp_path: Path) -> Path:
    """A tiny int32 array written with zarr-python; returns the store path."""
    path = tmp_path / "a.zarr"
    z = zarr.create_array(store=str(path), shape=(4,), chunks=(2,), dtype="int32")
    z[:] = np.arange(4, dtype="int32")
    return path


def test_filesystem_store_opens_and_reads(array_path: Path):
    array = Array.open(FilesystemStore(str(array_path)))
    np.testing.assert_array_equal(
        array.retrieve_chunk([0]).to_numpy(),
        np.array([0, 1], dtype="int32"),
    )


def test_open_rejects_non_store(array_path: Path):
    with pytest.raises(TypeError, match="FilesystemStore, MemoryStore, or ZipStore"):
        Array.open(object())
    with pytest.raises(TypeError, match="FilesystemStore, MemoryStore, or ZipStore"):
        Group.open(object())
