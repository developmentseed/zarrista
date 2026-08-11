"""`Group.with_consolidated_metadata()` sets or clears the block, without writing."""

from pathlib import Path
from typing import Any

import pytest
import zarr
from obstore.store import LocalStore

from zarrista import Array, AsyncGroup, Group
from zarrista.store import FilesystemStore


@pytest.fixture
def hierarchy(tmp_path: Path) -> Path:
    """A root group with one child array `a0`; returns the store path."""
    path = tmp_path / "h.zarr"
    root = zarr.open_group(store=str(path), mode="w")
    root.create_array("a0", shape=(4,), chunks=(2,), dtype="int32")
    return path


@pytest.fixture
def consolidated(hierarchy: Path) -> dict[str, Any]:
    """A consolidated metadata block that describes the `a0` array."""
    array = Array.open(FilesystemStore(hierarchy), "/a0")
    return {
        "kind": "inline",
        "must_understand": False,
        "metadata": {"a0": array.metadata},
    }


def test_with_consolidated_metadata_leaves_the_original(
    hierarchy: Path,
    consolidated: dict[str, Any],
):
    group = Group.open(FilesystemStore(hierarchy))

    updated = group.with_consolidated_metadata(consolidated)

    assert updated.consolidated_metadata is not None
    assert updated.consolidated_metadata["kind"] == "inline"
    assert list(updated.consolidated_metadata["metadata"]) == ["a0"]
    assert group.consolidated_metadata is None


def test_none_clears_the_block(hierarchy: Path, consolidated: dict[str, Any]):
    group = Group.open(FilesystemStore(hierarchy)).with_consolidated_metadata(
        consolidated,
    )

    assert group.with_consolidated_metadata(None).consolidated_metadata is None


def test_nothing_is_written_until_store_metadata(
    hierarchy: Path,
    consolidated: dict[str, Any],
):
    updated = Group.open(FilesystemStore(hierarchy)).with_consolidated_metadata(
        consolidated,
    )
    assert Group.open(FilesystemStore(hierarchy)).consolidated_metadata is None

    updated.store_metadata()

    stored = Group.open(FilesystemStore(hierarchy)).consolidated_metadata
    assert stored is not None
    assert stored["metadata"]["a0"]["shape"] == [4]


def test_a_zarr_v2_group_raises(tmp_path: Path, consolidated: dict[str, Any]):
    path = tmp_path / "v2.zarr"
    zarr.open_group(store=str(path), mode="w", zarr_format=2)
    group = Group.open(FilesystemStore(path))

    with pytest.raises(ValueError, match="Zarr V2"):
        group.with_consolidated_metadata(consolidated)


async def test_async_with_consolidated_metadata(
    hierarchy: Path,
    consolidated: dict[str, Any],
):
    group = await AsyncGroup.open(LocalStore(str(hierarchy)))

    updated = group.with_consolidated_metadata(consolidated)
    await updated.store_metadata()

    assert group.consolidated_metadata is None
    reopened = await AsyncGroup.open(LocalStore(str(hierarchy)))
    assert reopened.consolidated_metadata is not None
