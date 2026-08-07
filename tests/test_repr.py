"""`__repr__` shows Zarr v3 names and Python syntax, never Rust internals.

These types wrap a zarrs struct. Formatting that struct with Rust's `Debug`
would print its Rust type name and `{ field: value }` syntax, which are
implementation details of the extension. Each repr is built from the Zarr v3
name and configuration instead.
"""

import re

import pytest

from zarrista import ChunkKeyEncoding, codec

# `Debug` output looks like `ZstdCodec { compression: 3 }`: a Rust type name in
# UpperCamelCase, then a brace. Python's dict repr never matches this.
RUST_DEBUG_STRUCT = re.compile(r"[A-Z]\w+ \{")


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        (codec.crc32c(), 'BytesToBytesCodec("crc32c")'),
        (
            codec.zstd(3, checksum=False),
            "BytesToBytesCodec(\"zstd\", config={'level': 3, 'checksum': False})",
        ),
        (
            codec.transpose([1, 0]),
            "ArrayToArrayCodec(\"transpose\", config={'order': [1, 0]})",
        ),
        (
            ChunkKeyEncoding.default("/"),
            "ChunkKeyEncoding(\"default\", config={'separator': '/'})",
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
