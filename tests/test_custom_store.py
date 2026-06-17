from __future__ import annotations

import os

import numpy as np
import pytest
import zarr
from zarrista import Array, Group


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


class DictStore(ReadOnlyDictStore):
    """Readable + listable dict store."""

    supports_listing = True

    def list(self) -> list[str]:
        return list(self._mapping)

    def list_prefix(self, prefix: str) -> list[str]:
        return [k for k in self._mapping if k.startswith(prefix)]

    def list_dir(self, prefix: str) -> dict[str, list[str]]:
        keys: list[str] = []
        prefixes: set[str] = set()
        for k in self._mapping:
            if not k.startswith(prefix):
                continue
            rest = k[len(prefix):]
            if "/" in rest:
                prefixes.add(prefix + rest.split("/", 1)[0] + "/")
            else:
                keys.append(k)
        return {"keys": keys, "prefixes": sorted(prefixes)}

    def size_prefix(self, prefix: str) -> int:
        return sum(len(v) for k, v in self._mapping.items() if k.startswith(prefix))


def test_listing_works_when_supported(zarr_bytes):
    group = Group.open(DictStore(zarr_bytes), "/")
    assert group.array_keys() == ["a"]


def test_listing_raises_when_unsupported(zarr_bytes):
    group = Group.open(ReadOnlyDictStore(zarr_bytes), "/")
    with pytest.raises(Exception, match="does not support listing"):
        group.array_keys()
