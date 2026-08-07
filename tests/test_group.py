"""Group navigation and metadata methods, mirroring upstream `zarrs::group::Group`.

A small hierarchy is written with zarr-python and read back with zarrista:

    /                (root group, attrs={"title": "root"})
    ├── a0           (array)
    └── g1           (group)
        └── inner    (array)
"""

from pathlib import Path

import pytest
import zarr
from obstore.store import LocalStore

from zarrista import Array, AsyncArray, AsyncGroup, Group
from zarrista.exceptions import ZarristaError
from zarrista.store import FilesystemStore


@pytest.fixture
def hierarchy(tmp_path: Path) -> Path:
    """Write the hierarchy above with zarr-python; returns the store path."""
    path = tmp_path / "h.zarr"
    root = zarr.open_group(store=str(path), mode="w")
    root.attrs["title"] = "root"
    root.create_array("a0", shape=(4,), chunks=(2,), dtype="int32")
    g1 = root.create_group("g1")
    g1.create_array("inner", shape=(2,), chunks=(2,), dtype="int32")
    return path


# --- sync ---------------------------------------------------------------------


def test_array_and_group_keys(hierarchy: Path):
    group = Group.open(FilesystemStore(hierarchy))
    assert group.array_keys() == ["a0"]
    assert group.group_keys() == ["g1"]


def test_traverse(hierarchy: Path):
    group = Group.open(FilesystemStore(hierarchy))
    by_path = {node.path: node for node in group.traverse()}
    assert sorted(by_path) == ["/a0", "/g1", "/g1/inner"]
    assert isinstance(by_path["/a0"], Array)
    assert isinstance(by_path["/g1"], Group)
    assert isinstance(by_path["/g1/inner"], Array)


def test_child_arrays_and_groups(hierarchy: Path):
    group = Group.open(FilesystemStore(hierarchy))

    arrays = group.child_arrays()
    assert [a.path for a in arrays] == ["/a0"]
    assert all(isinstance(a, Array) for a in arrays)

    groups = group.child_groups()
    assert [g.path for g in groups] == ["/g1"]
    assert all(isinstance(g, Group) for g in groups)


def test_child_paths(hierarchy: Path):
    group = Group.open(FilesystemStore(hierarchy))
    assert sorted(group.child_paths()) == ["/a0", "/g1"]
    assert group.child_array_paths() == ["/a0"]
    assert group.child_group_paths() == ["/g1"]


def test_child_lookup(hierarchy: Path):
    group = Group.open(FilesystemStore(hierarchy))

    array = group.child("a0")
    assert isinstance(array, Array)
    assert array.path == "/a0"

    subgroup = group["g1"]
    assert isinstance(subgroup, Group)
    assert subgroup.path == "/g1"

    with pytest.raises(KeyError):
        group.child("missing")


def test_metadata_is_v3(hierarchy: Path):
    group = Group.open(FilesystemStore(hierarchy))
    assert group.metadata["zarr_format"] == 3
    assert group.metadata["node_type"] == "group"


def test_store_and_erase_metadata(hierarchy: Path):
    group = Group.open(FilesystemStore(hierarchy))

    # Re-storing the metadata is a no-op round-trip; the group stays openable.
    group.store_metadata()
    reopened = Group.open(FilesystemStore(hierarchy))
    assert reopened.attrs == {"title": "root"}

    # After erasing, the group can no longer be opened.
    group.erase_metadata()
    with pytest.raises(ZarristaError):
        Group.open(FilesystemStore(hierarchy))


# --- async --------------------------------------------------------------------


async def test_async_traverse(hierarchy: Path):
    group = await AsyncGroup.open(LocalStore(str(hierarchy)))
    by_path = {node.path: node for node in await group.traverse()}
    assert sorted(by_path) == ["/a0", "/g1", "/g1/inner"]
    assert isinstance(by_path["/a0"], AsyncArray)
    assert isinstance(by_path["/g1"], AsyncGroup)
    assert isinstance(by_path["/g1/inner"], AsyncArray)


async def test_async_child_arrays_and_groups(hierarchy: Path):
    group = await AsyncGroup.open(LocalStore(str(hierarchy)))

    arrays = await group.child_arrays()
    assert [a.path for a in arrays] == ["/a0"]
    assert all(isinstance(a, AsyncArray) for a in arrays)

    groups = await group.child_groups()
    assert [g.path for g in groups] == ["/g1"]
    assert all(isinstance(g, AsyncGroup) for g in groups)


async def test_async_child_paths(hierarchy: Path):
    group = await AsyncGroup.open(LocalStore(str(hierarchy)))
    assert sorted(await group.child_paths()) == ["/a0", "/g1"]
    assert await group.child_array_paths() == ["/a0"]
    assert await group.child_group_paths() == ["/g1"]


async def test_async_open_child(hierarchy: Path):
    group = await AsyncGroup.open(LocalStore(str(hierarchy)))

    array = await group.child("a0")
    assert isinstance(array, AsyncArray)
    assert array.path == "/a0"

    subgroup = await group.child("g1")
    assert isinstance(subgroup, AsyncGroup)
    assert subgroup.path == "/g1"

    with pytest.raises(KeyError):
        await group.child("missing")


async def test_async_metadata_is_v3(hierarchy: Path):
    group = await AsyncGroup.open(LocalStore(str(hierarchy)))
    assert group.metadata["zarr_format"] == 3
