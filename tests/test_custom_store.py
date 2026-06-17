import os

import numpy as np
import pytest
import zarr

from zarrista import Array


@pytest.fixture
def zarr_bytes(tmp_path):
    """Write a tiny v3 array with zarr-python, return {key: bytes}."""
    store = zarr.storage.LocalStore(str(tmp_path))
    root = zarr.create_group(store=store)
    arr = root.create_array("a", shape=(4,), chunks=(2,), dtype="int32")
    arr[:] = np.arange(4, dtype="int32")

    mapping: dict[str, bytes] = {}
    for dirpath, _dirs, files in os.walk(tmp_path):
        for name in files:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, tmp_path).replace(os.sep, "/")
            with open(full, "rb") as fh:
                mapping[rel] = fh.read()
    return mapping


class ReadOnlyDictStore:
    """Minimal readable store: implements only `get`."""

    supports_get_partial = False
    supports_listing = False

    def __init__(self, mapping: dict[str, bytes]):
        self._mapping = mapping

    def get(self, key: str) -> bytes | None:
        return self._mapping.get(key)


def test_open_array_and_read_chunk_from_custom_store(zarr_bytes):
    store = ReadOnlyDictStore(zarr_bytes)
    array = Array.open(store, "/a")
    data = array.retrieve_chunk([0])
    assert array.shape == [4]
    np.testing.assert_array_equal(data.to_numpy(), np.array([0, 1], dtype="int32"))
