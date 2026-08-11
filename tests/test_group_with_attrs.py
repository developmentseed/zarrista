"""`Group.with_attrs()` returns a new group with new attributes, without writing."""

from pathlib import Path

import pytest
import zarr
from obstore.store import LocalStore

from zarrista import AsyncGroup, Group
from zarrista.store import FilesystemStore


@pytest.fixture
def hierarchy(tmp_path: Path) -> Path:
    """A root group with two user attributes; returns the store path."""
    path = tmp_path / "h.zarr"
    root = zarr.open_group(store=str(path), mode="w")
    root.attrs["title"] = "root"
    root.attrs["institution"] = "example"
    return path


def test_with_attrs_replaces_attrs_and_leaves_the_original(hierarchy: Path):
    group = Group.open(FilesystemStore(hierarchy))

    updated = group.with_attrs({"title": "renamed"})

    # `institution` is absent, so the new attributes replace rather than merge.
    assert updated.attrs == {"title": "renamed"}
    assert group.attrs == {"title": "root", "institution": "example"}
    assert updated.path == group.path


def test_nothing_is_written_until_store_metadata(hierarchy: Path):
    updated = Group.open(FilesystemStore(hierarchy)).with_attrs({"title": "renamed"})
    assert Group.open(FilesystemStore(hierarchy)).attrs["title"] == "root"

    updated.store_metadata()

    stored = Group.open(FilesystemStore(hierarchy)).attrs
    assert stored["title"] == "renamed"
    assert "institution" not in stored


def test_a_value_that_is_not_json_serializable_raises(hierarchy: Path):
    group = Group.open(FilesystemStore(hierarchy))

    with pytest.raises(TypeError):
        group.with_attrs({"bad": object()})


async def test_async_with_attrs(hierarchy: Path):
    group = await AsyncGroup.open(LocalStore(str(hierarchy)))

    updated = group.with_attrs({"title": "renamed"})
    await updated.store_metadata()

    assert updated.attrs == {"title": "renamed"}
    assert group.attrs == {"title": "root", "institution": "example"}
    reopened = await AsyncGroup.open(LocalStore(str(hierarchy)))
    assert reopened.attrs["title"] == "renamed"
