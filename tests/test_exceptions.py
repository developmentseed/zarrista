"""The zarrista.exceptions hierarchy: a base class with one subclass per
underlying zarrs error category, all importable from zarrista.exceptions."""

from pathlib import Path

import pytest
import zarr

from zarrista import FilesystemStore, Group
from zarrista import exceptions as exc

LEAF_EXCEPTIONS = [
    "NotFoundError",
    "ArrayCreateError",
    "ArrayError",
    "GroupCreateError",
    "NodeCreateError",
    "NodePathError",
    "StorageError",
    "CodecError",
    "TransposeOrderError",
    "PluginCreateError",
    "SerializationError",
]


def test_base_is_an_exception():
    assert issubclass(exc.ZarristaError, Exception)
    assert exc.ZarristaError.__name__ == "ZarristaError"
    assert exc.ZarristaError.__module__ == "zarrista.exceptions"


@pytest.mark.parametrize("name", LEAF_EXCEPTIONS)
def test_leaf_subclasses_base(name):
    cls = getattr(exc, name)
    assert issubclass(cls, exc.ZarristaError)
    assert cls is not exc.ZarristaError
    assert cls.__module__ == "zarrista.exceptions"


def test_importable_by_name():
    from zarrista.exceptions import (  # noqa: F401
        CodecError,
        NotFoundError,
        ZarristaError,
    )


def test_filesystem_store_create_folds_into_storage_error():
    # Filesystem-store creation failures surface as StorageError; there is no
    # dedicated FilesystemStoreCreateError class.
    assert not hasattr(exc, "FilesystemStoreCreateError")
    assert "FilesystemStoreCreateError" not in exc.__all__


def _make_group(tmp_path: Path) -> Path:
    """Create an empty Zarr v3 group with zarr-python; return its store path."""
    path = tmp_path / "g.zarr"
    zarr.open_group(str(path), mode="w")
    return path


def test_missing_child_raises_not_found(tmp_path: Path):
    group = Group.open(FilesystemStore(str(_make_group(tmp_path))))
    with pytest.raises(exc.NotFoundError):
        group["does_not_exist"]


def test_base_catches_subclass(tmp_path: Path):
    group = Group.open(FilesystemStore(str(_make_group(tmp_path))))
    with pytest.raises(exc.ZarristaError):
        group["does_not_exist"]
