"""`__repr__` shows Zarr v3 names and Python syntax, never Rust internals.

These types wrap a zarrs struct. Formatting that struct with Rust's `Debug`
would print its Rust type name and `{ field: value }` syntax, which are
implementation details of the extension. Each repr is built from the Zarr v3
name and configuration instead.

Python renders every value, so the quoting and the literals are Python's own:
strings use `'`, and a boolean reads `False` rather than the JSON `false`.
"""

import re
from pathlib import Path

import numpy as np
import pytest
import zarr

from zarrista import Array, ChunkKeyEncoding, codec
from zarrista.store import FilesystemStore

# `Debug` output looks like `ZstdCodec { compression: 3 }`: a Rust type name in
# UpperCamelCase, then a brace. Python's dict repr never matches this.
RUST_DEBUG_STRUCT = re.compile(r"[A-Z]\w+ \{")


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        (codec.crc32c(), "BytesToBytesCodec('crc32c')"),
        (
            codec.zstd(3, checksum=False),
            "BytesToBytesCodec('zstd', config={'level': 3, 'checksum': False})",
        ),
        (
            codec.transpose([1, 0]),
            "ArrayToArrayCodec('transpose', config={'order': [1, 0]})",
        ),
        (
            ChunkKeyEncoding.default("/"),
            "ChunkKeyEncoding('default', config={'separator': '/'})",
        ),
    ],
)
def test_repr_uses_the_zarr_name_and_python_syntax(obj: object, expected: str) -> None:
    assert repr(obj) == expected


def test_repr_never_shows_rust_struct_syntax() -> None:
    """A guard for every type whose repr is built from a zarrs struct."""
    for obj in (
        codec.crc32c(),
        codec.zstd(3, checksum=False),
        codec.gzip(5),
        codec.blosc("zstd", 5, "noshuffle"),
        codec.transpose([1, 0]),
        ChunkKeyEncoding.default("."),
    ):
        assert not RUST_DEBUG_STRUCT.search(repr(obj)), repr(obj)


def test_empty_configuration_is_omitted() -> None:
    """`crc32c` takes no options, so a `config=` entry would say nothing."""
    assert "config=" not in repr(codec.crc32c())


def test_tensor_repr(tmp_path: Path) -> None:
    array = zarr.create_array(
        store=str(tmp_path),
        shape=(4, 4),
        chunks=(2, 2),
        dtype="int32",
    )
    array[:] = np.zeros((4, 4), dtype="int32")

    tensor = Array.open(FilesystemStore(tmp_path))[:, :]

    assert repr(tensor) == "Tensor(shape=(4, 4), dtype='int32')"


def test_variable_array_repr(tmp_path: Path) -> None:
    """The data type shows its Zarr v3 name, not the numpy descr."""
    array = zarr.create_array(store=str(tmp_path), shape=(3,), chunks=(3,), dtype=str)
    array[:] = np.array(["a", "bb", "ccc"], dtype=object)

    variable = Array.open(FilesystemStore(tmp_path))[:]

    assert repr(variable) == "VariableArray(shape=(3,), dtype='string')"

