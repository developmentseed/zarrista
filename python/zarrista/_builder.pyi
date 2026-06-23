from collections.abc import Mapping, Sequence

from zarr_metadata import ArrayMetadataV3, JSONValue

from zarrista.codec import (
    ArrayToArrayCodec,
    ArrayToBytesCodec,
    BytesToBytesCodec,
)

from ._array import Array, AsyncArray
from ._chunk_key_encoding import ChunkKeyEncoding
from ._chunks import ChunkGrid
from ._dtype import DataType
from ._fill_value import FillValue
from ._store import AsyncStore, FilesystemStore, MemoryStore

class ArrayBuilder:
    """A chained, immutable builder for creating Zarr arrays.

    Every setter returns a *new* `ArrayBuilder` and leaves the receiver
    unchanged, so a builder can be safely shared and specialized. Seed one with
    the constructor or [`ArrayBuilder.like`][zarrista.ArrayBuilder.like], chain
    setters to configure it, then materialize the array with
    [`create`][zarrista.ArrayBuilder.create] /
    [`create_async`][zarrista.ArrayBuilder.create_async], or produce only its
    metadata with
    [`create_metadata`][zarrista.ArrayBuilder.create_metadata].
    """

    def __init__(
        self,
        chunk_grid: ChunkGrid,
        dtype: DataType,
        fill_value: FillValue,
    ) -> None:
        """Create a builder from a chunk grid, data type, and fill value."""
    @staticmethod
    def like(array: Array | AsyncArray) -> ArrayBuilder:
        """Create a builder copying the configuration of an existing array."""
    def attrs(self, attrs: Mapping[str, JSONValue]) -> ArrayBuilder:
        """Return a new builder with the given user attributes set."""
    def chunk_grid(self, chunk_grid: ChunkGrid) -> ArrayBuilder:
        """Return a new builder with the chunk grid set.

        This may also change the array shape, since the grid carries one.
        """
    def chunk_key_encoding(
        self,
        chunk_key_encoding: ChunkKeyEncoding,
    ) -> ArrayBuilder:
        """Return a new builder with the chunk key encoding set."""
    def compressors(
        self,
        compressors: Sequence[BytesToBytesCodec],
    ) -> ArrayBuilder:
        """Return a new builder with the bytes-to-bytes codecs ("compressors") set."""
    def data_type(self, data_type: DataType) -> ArrayBuilder:
        """Return a new builder with the data type set."""
    def dimension_names(
        self,
        dimension_names: Sequence[str | None] | None,
    ) -> ArrayBuilder:
        """Return a new builder with the dimension names set (or cleared)."""
    def filters(self, filters: Sequence[ArrayToArrayCodec]) -> ArrayBuilder:
        """Return a new builder with the array-to-array codecs ("filters") set."""
    def serializer(self, serializer: ArrayToBytesCodec) -> ArrayBuilder:
        """Return a new builder with the array-to-bytes codec ("serializer") set.

        Sharding is itself an array-to-bytes codec, so a sharding serializer is
        passed here too.
        """
    def shape(self, shape: Sequence[int]) -> ArrayBuilder:
        """Return a new builder with the array shape set."""
    def subchunk_shape(
        self,
        subchunk_shape: Sequence[int] | None,
    ) -> ArrayBuilder:
        """Return a new builder with the inner (subchunk) shape, enabling sharding."""
    def create(self, store: FilesystemStore | MemoryStore, path: str) -> Array:
        """Build the array in `store` at `path` and return it."""
    async def create_async(self, store: AsyncStore, path: str) -> AsyncArray:
        """Build the array in an async `store` at `path` and return it."""
    def create_metadata(self) -> ArrayMetadataV3:
        """Build the array's Zarr v3 metadata without touching a store."""
