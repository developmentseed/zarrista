from collections.abc import Mapping, Sequence

from zarr_metadata import JSONValue, ZarrV3ArrayMetadataJSON

from zarrista.codec import (
    ArrayToArrayCodec,
    ArrayToBytesCodec,
    BytesToBytesCodec,
)
from zarrista.store import AsyncStore, SyncStore

from ._array import Array, AsyncArray
from ._chunk_key_encoding import ChunkKeyEncoding
from ._chunks import ChunkGrid
from ._dtype import DataType
from ._fill_value import FillValue

class ArrayBuilder:
    """A chained, immutable builder for creating Zarr arrays.

    Every setter returns a *new* `ArrayBuilder` and leaves the receiver
    unchanged. Start
    with the constructor or [`ArrayBuilder.like`][zarrista.ArrayBuilder.like],
    chain setters to configure it, then make the array with
    [`create`][zarrista.ArrayBuilder.create] or
    [`create_async`][zarrista.ArrayBuilder.create_async]. To make only the
    metadata, use
    [`create_metadata`][zarrista.ArrayBuilder.create_metadata].

    Examples:
        Create a 1024x1024 `int32` array with 256x256 chunks and zstd
        compression:

        ```py
        from zarrista import ArrayBuilder, ChunkGrid, DataType, FillValue, codec
        from zarrista.store import FilesystemStore

        grid = ChunkGrid.regular([1024, 1024], [256, 256])
        dtype = DataType.from_string("int32")
        fill_value = FillValue((0).to_bytes(4, "little"))

        array = (
            ArrayBuilder(grid, dtype, fill_value)
            .dimension_names(["y", "x"])
            .compressors([codec.zstd(3, checksum=False)])
            .create(FilesystemStore("data"), "/temperature")
        )
        ```

        `create` writes the metadata to the store. Each setter returns a new
        builder, so you can keep a partly configured builder and reuse it:

        ```py
        base = ArrayBuilder(grid, dtype, fill_value).dimension_names(["y", "x"])
        uncompressed = base.create(store, "/raw")
        compressed = base.compressors([codec.zstd(3, checksum=False)]).create(
            store, "/compressed"
        )
        ```
    """

    def __init__(
        self,
        chunk_grid: ChunkGrid,
        dtype: DataType,
        fill_value: FillValue,
    ) -> None:
        """Create a builder from a chunk grid, data type, and fill value.

        Args:
            chunk_grid: The chunk grid of the array. The grid also gives the
                array shape.
            dtype: The data type of the array.
            fill_value: The fill value of the array. This must match `dtype`.
        """
    @staticmethod
    def like(array: Array | AsyncArray) -> ArrayBuilder:
        """Create a builder that copies the configuration of an existing array.

        Args:
            array: The array to copy the configuration from.

        Returns:
            A new builder with the same configuration as `array`.
        """
    def attrs(self, attrs: Mapping[str, JSONValue]) -> ArrayBuilder:
        """Return a new builder with the given user attributes set.

        Args:
            attrs: The user attributes of the array.

        Returns:
            A new builder with the user attributes set.
        """
    def chunk_grid(self, chunk_grid: ChunkGrid) -> ArrayBuilder:
        """Return a new builder with the chunk grid set.

        This can also change the array shape, because the grid carries one.

        Args:
            chunk_grid: The chunk grid of the array.

        Returns:
            A new builder with the chunk grid set.
        """
    def chunk_key_encoding(
        self,
        chunk_key_encoding: ChunkKeyEncoding,
    ) -> ArrayBuilder:
        """Return a new builder with the chunk key encoding set.

        Args:
            chunk_key_encoding: How the array maps chunk grid indices to store
                keys.

        Returns:
            A new builder with the chunk key encoding set.
        """
    def compressors(
        self,
        compressors: Sequence[BytesToBytesCodec],
    ) -> ArrayBuilder:
        """Return a new builder with the bytes-to-bytes codecs ("compressors") set.

        Args:
            compressors: The bytes-to-bytes codecs, in pipeline order.

        Returns:
            A new builder with the compressors set.
        """
    def data_type(self, data_type: DataType) -> ArrayBuilder:
        """Return a new builder with the data type set.

        Args:
            data_type: The data type of the array. The fill value must match
                this data type.

        Returns:
            A new builder with the data type set.
        """
    def dimension_names(
        self,
        dimension_names: Sequence[str | None] | None,
    ) -> ArrayBuilder:
        """Return a new builder with the dimension names set (or cleared).

        Args:
            dimension_names: One name per dimension. A name can be `None` to
                leave that dimension unnamed. Give `None` to clear all the
                names.

        Returns:
            A new builder with the dimension names set.
        """
    def filters(self, filters: Sequence[ArrayToArrayCodec]) -> ArrayBuilder:
        """Return a new builder with the array-to-array codecs ("filters") set.

        Args:
            filters: The array-to-array codecs, in pipeline order.

        Returns:
            A new builder with the filters set.
        """
    def serializer(self, serializer: ArrayToBytesCodec) -> ArrayBuilder:
        """Return a new builder with the array-to-bytes codec ("serializer") set.

        Sharding is itself an array-to-bytes codec. Therefore you pass a
        sharding serializer here too.

        Args:
            serializer: The array-to-bytes codec of the array.

        Returns:
            A new builder with the serializer set.
        """
    def shape(self, shape: Sequence[int]) -> ArrayBuilder:
        """Return a new builder with the array shape set.

        This shape replaces the shape that the chunk grid gives.

        Args:
            shape: The shape of the array, in elements along each dimension.

        Returns:
            A new builder with the array shape set.
        """
    def subchunk_shape(
        self,
        subchunk_shape: Sequence[int] | None,
    ) -> ArrayBuilder:
        """Return a new builder with the inner (subchunk) shape, enabling sharding.

        Args:
            subchunk_shape: The shape of an inner chunk, in elements along each
                dimension. Give `None` to disable sharding.

        Returns:
            A new builder with the subchunk shape set.
        """
    def create(self, store: SyncStore, path: str) -> Array:
        """Build the array in `store` at `path` and return it.

        This **does** write to the store. It stores the array's metadata at
        `path`, and it overwrites any metadata already there. Use
        [`ArrayBuilder.create_metadata`][zarrista.ArrayBuilder.create_metadata]
        to build metadata without a store.

        Args:
            store: The store to write the array metadata to.
            path: The absolute path of the array in the store.

        Returns:
            The new array.

        Raises:
            ArrayCreateError: If `path` is not a valid absolute node path, if
                the fill value does not match the data type, or if the number
                of dimension names does not match the number of dimensions.
        """
    async def create_async(self, store: AsyncStore, path: str) -> AsyncArray:
        """Build the array in an async `store` at `path` and return it.

        This **does** write to the store. It stores the array's metadata at
        `path`, and it overwrites any metadata already there. Use
        [`ArrayBuilder.create_metadata`][zarrista.ArrayBuilder.create_metadata]
        to build metadata without a store.

        Args:
            store: The async store to write the array metadata to.
            path: The absolute path of the array in the store.

        Returns:
            The new array.

        Raises:
            ArrayCreateError: If `path` is not a valid absolute node path, if
                the fill value does not match the data type, or if the number
                of dimension names does not match the number of dimensions.
        """
    def create_metadata(self) -> ZarrV3ArrayMetadataJSON:
        """Build the array's Zarr v3 metadata without a store.

        This writes nothing. Pass the result to
        [`Array.from_metadata`][zarrista.Array.from_metadata] to open an
        in-memory array. To build and write in one step, use
        [`ArrayBuilder.create`][zarrista.ArrayBuilder.create].

        Returns:
            The Zarr v3 metadata of the array.

        Raises:
            ArrayCreateError: If the fill value does not match the data type,
                or if the number of dimension names does not match the number
                of dimensions.
        """
