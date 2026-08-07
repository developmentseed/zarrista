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

from zarrista import (
    Array,
    ArrayBuilder,
    ArrayBytes,
    ChunkGrid,
    ChunkKeyEncoding,
    DataType,
    FillValue,
    Group,
    ThreadPool,
    codec,
)
from zarrista.store import FilesystemStore, MemoryStore

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


def test_array_and_group_repr(tmp_path: Path) -> None:
    """The path comes first, because it tells two arrays of one store apart."""
    zarr.open_group(store=str(tmp_path), mode="w")
    array = zarr.create_array(
        store=str(tmp_path),
        name="temperature",
        shape=(4, 4),
        chunks=(2, 2),
        dtype="float32",
    )
    array[:] = np.zeros((4, 4), dtype="float32")
    store = FilesystemStore(tmp_path)

    assert (
        repr(Array.open(store, "/temperature"))
        == "Array(path='/temperature', shape=(4, 4), dtype='float32')"
    )
    assert repr(Group.open(store)) == "Group(path='/')"


def test_shard_cache_repr_nests_the_array(tmp_path: Path) -> None:
    array = zarr.create_array(
        store=str(tmp_path),
        shape=(4, 4),
        chunks=(2, 2),
        dtype="int32",
    )
    array[:] = np.zeros((4, 4), dtype="int32")

    cache = Array.open(FilesystemStore(tmp_path)).shard_cache()

    assert (
        repr(cache) == "ShardCache(array=Array(path='/', shape=(4, 4), dtype='int32'))"
    )


def test_store_repr_names_its_argument() -> None:
    assert repr(FilesystemStore("/data")) == "FilesystemStore(path='/data')"
    assert repr(MemoryStore()) == "MemoryStore()"


def test_chunk_grid_repr() -> None:
    """Reuses the named-configuration form, so it reads like the codecs."""
    grid = ChunkGrid.regular([4, 4], [2, 2])

    assert repr(grid) == "ChunkGrid('regular', config={'chunk_shape': [2, 2]})"


def test_fill_value_repr() -> None:
    assert repr(FillValue(b"\x00\x00\x00\x00")) == r"FillValue(b'\x00\x00\x00\x00')"


def test_thread_pool_repr() -> None:
    assert repr(ThreadPool(4)) == "ThreadPool(num_threads=4)"


def test_array_bytes_repr() -> None:
    data = np.arange(4, dtype="int32").tobytes()

    assert repr(ArrayBytes(data)) == "ArrayBytes(layout='fixed', nbytes=16)"


def test_array_builder_repr() -> None:
    builder = ArrayBuilder(
        ChunkGrid.regular([4, 4], [2, 2]),
        DataType.from_string("int32"),
        FillValue(b"\x00\x00\x00\x00"),
    )

    # The builder has no accessors of its own, so the metadata it would write
    # is the whole description of it.
    assert repr(builder) == (
        "ArrayBuilder(metadata={"
        "'zarr_format': 3, 'node_type': 'array', 'shape': [4, 4], "
        "'data_type': 'int32', "
        "'chunk_grid': {'name': 'regular', 'configuration': {'chunk_shape': [2, 2]}}, "
        "'chunk_key_encoding': "
        "{'name': 'default', 'configuration': {'separator': '/'}}, "
        "'fill_value': 0, "
        "'codecs': [{'name': 'bytes', 'configuration': {'endian': 'little'}}]"
        "})"
    )


def test_encoded_chunk_repr(tmp_path: Path) -> None:
    """Shows the encoded size, which is what the decoded chunk does not have."""
    array = zarr.create_array(
        store=str(tmp_path),
        shape=(4, 4),
        chunks=(2, 2),
        dtype="int32",
        compressors=None,
    )
    array[:] = np.arange(16, dtype="int32").reshape(4, 4)

    chunk = Array.open(FilesystemStore(tmp_path)).retrieve_encoded_chunk([0, 0])

    assert repr(chunk) == "EncodedChunk(shape=(2, 2), dtype='int32', nbytes=16)"


def test_codec_chain_repr_nests_each_codec(tmp_path: Path) -> None:
    array = zarr.create_array(
        store=str(tmp_path),
        shape=(4, 4),
        chunks=(2, 2),
        dtype="int32",
        compressors=None,
    )
    array[:] = np.zeros((4, 4), dtype="int32")

    chain = Array.open(FilesystemStore(tmp_path)).codecs

    assert repr(chain) == (
        "CodecChain(filters=[], "
        "serializer=ArrayToBytesCodec('bytes', config={'endian': 'little'}), "
        "compressors=[])"
    )
