"""Tests for the chained `ArrayBuilder` array-creation API."""

import pytest

from zarrista import (
    ArrayBuilder,
    ChunkGrid,
    ChunkKeyEncoding,
    DataType,
    FillValue,
    codec,
)
from zarrista.codec import ArrayToBytesCodec
from zarrista.exceptions import ChunkGridCreateError, ZarristaError
from zarrista.store import MemoryStore


def _builder() -> ArrayBuilder:
    """A minimal int8 builder: 8x8 array, 4x4 regular chunks, fill value 0."""
    return ArrayBuilder(
        ChunkGrid.regular([8, 8], [4, 4]),
        DataType.from_string("int8"),
        FillValue(b"\x00"),
    )


def test_create_metadata_without_store():
    """`create_metadata` produces v3 metadata without touching a store."""
    meta = _builder().create_metadata()

    assert meta["zarr_format"] == 3
    assert meta["node_type"] == "array"
    assert meta["shape"] == [8, 8]
    assert meta["data_type"] == "int8"
    assert meta["chunk_grid"] == {
        "name": "regular",
        "configuration": {"chunk_shape": [4, 4]},
    }
    assert meta["fill_value"] == 0
    # No serializer set -> default `bytes` codec.
    assert meta["codecs"] == [{"name": "bytes", "configuration": {"endian": "little"}}]


def test_setters_return_new_instances():
    """Each setter returns a new builder and leaves the receiver unchanged."""
    base = _builder()
    modified = base.shape([16, 16])

    assert modified is not base
    assert base.create_metadata()["shape"] == [8, 8]
    assert modified.create_metadata()["shape"] == [16, 16]


def test_create_returns_configured_array():
    """`create` returns an array reflecting the builder's configuration."""
    array = (
        _builder()
        .shape([16, 16])
        .dimension_names(["y", "x"])
        .create(MemoryStore(), "/a")
    )

    assert array.shape == [16, 16]
    assert array.dtype == DataType.from_string("int8")
    assert array.dimension_names == ["y", "x"]


def test_dimension_names_can_be_cleared():
    meta = (
        _builder().dimension_names(["y", "x"]).dimension_names(None).create_metadata()
    )
    assert "dimension_names" not in meta or meta["dimension_names"] is None


def test_filters_and_compressors():
    array = (
        _builder()
        .filters([codec.transpose([1, 0])])
        .compressors([codec.zstd(3, checksum=False)])
        .create(MemoryStore(), "/a")
    )

    assert [f.name for f in array.filters] == ["transpose"]
    assert [c.name for c in array.compressors] == ["zstd"]


def test_serializer():
    array = (
        _builder()
        .serializer(
            ArrayToBytesCodec.from_config(
                {"name": "bytes", "configuration": {"endian": "big"}},
            ),
        )
        .create(MemoryStore(), "/a")
    )

    assert array.serializer.name == "bytes"
    assert array.serializer.config == {"endian": "big"}


def test_subchunk_shape_enables_sharding():
    """Setting a subchunk shape selects the sharding serializer."""
    meta = _builder().subchunk_shape([2, 2]).create_metadata()
    assert meta["codecs"][0]["name"] == "sharding_indexed"


def test_chunk_key_encoding():
    cke = ChunkKeyEncoding.default(".")
    meta = _builder().chunk_key_encoding(cke).create_metadata()
    assert meta["chunk_key_encoding"] == {
        "name": "default",
        "configuration": {"separator": "."},
    }


def test_attrs():
    meta = _builder().attrs({"units": "m", "scale": 2}).create_metadata()
    assert meta["attributes"] == {"units": "m", "scale": 2}


def test_like_copies_configuration():
    """`like` reproduces an existing array's metadata."""
    source = _builder().dimension_names(["y", "x"]).create(MemoryStore(), "/a")
    copied = ArrayBuilder.like(source).create_metadata()
    assert copied == source.metadata


def test_like_with_override():
    """`like` followed by a setter overrides only that field."""
    source = (
        _builder()
        .compressors([codec.zstd(3, checksum=False)])
        .create(MemoryStore(), "/a")
    )
    overridden = (
        ArrayBuilder.like(source)
        .compressors([codec.gzip(5)])
        .create(MemoryStore(), "/b")
    )

    assert [c.name for c in source.compressors] == ["zstd"]
    assert [c.name for c in overridden.compressors] == ["gzip"]


def test_chunk_grid_dimension_mismatch_raises():
    """A chunk shape with the wrong dimensionality is rejected at parse time."""
    with pytest.raises(ChunkGridCreateError):
        ChunkGrid.regular([8, 8], [4, 4, 4])


def test_chunk_grid_create_error_is_zarrista_error():
    assert issubclass(ChunkGridCreateError, ZarristaError)
