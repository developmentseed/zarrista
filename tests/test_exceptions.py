"""The zarrista.exceptions hierarchy: a base class with one subclass per
underlying zarrs error category, all importable from zarrista.exceptions."""

import pytest
from zarrista import exceptions as exc

LEAF_EXCEPTIONS = [
    "NotFoundError",
    "ArrayCreateError",
    "ArrayError",
    "GroupCreateError",
    "NodeCreateError",
    "NodePathError",
    "StorageError",
    "FilesystemStoreCreateError",
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
