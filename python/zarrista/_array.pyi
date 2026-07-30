from collections.abc import Buffer
from types import EllipsisType
from typing import TypeAlias, Unpack

from zarr_metadata import JSONValue, ZarrV3ArrayMetadataJSON

from zarrista.codec import (
    ArrayToArrayCodec,
    ArrayToBytesCodec,
    BytesToBytesCodec,
    CodecOptions,
)

from ._array_bytes import ArrayBytes
from ._chunk_key_encoding import ChunkKeyEncoding
from ._chunks import ChunkGrid
from ._decoded_array import DecodedArray
from ._dtype import DataType
from ._fill_value import FillValue
from ._store import AsyncStore, FilesystemStore, MemoryStore

_AxisSelector: TypeAlias = int | slice | EllipsisType
Selection: TypeAlias = _AxisSelector | tuple[_AxisSelector, ...]
"""A numpy-style basic-indexing selection: what you would write inside `[]`.

Supports integers, step-1 slices, `Ellipsis`, and tuples of those (with fewer
entries than `ndim` implying full trailing axes). Negative indices and slice
bounds are normalized. `step != 1`, `None`/newaxis, boolean, and fancy/array
indexing are not supported.
"""

class Array:
    """A Zarr array."""

    @staticmethod
    def open(store: FilesystemStore | MemoryStore, path: str = "/") -> Array:
        """Open the array stored at `path` in `store`."""
    @staticmethod
    def from_metadata(
        metadata: ZarrV3ArrayMetadataJSON,
        store: FilesystemStore | MemoryStore,
        path: str = "/",
    ) -> Array:
        """Use the provided metadata to open a new array at `path` in `store`.

        This does **not** write the metadata to the store; use
        [`Array.store_metadata`][zarrista.Array.store_metadata] for that.
        """
    @property
    def attrs(self) -> dict[str, JSONValue]:
        """The array's user attributes as a dict."""
    @property
    def chunk_grid(self) -> ChunkGrid:
        """The chunk grid of the array."""
    @property
    def chunk_grid_shape(self) -> list[int]:
        """The shape of the chunk grid (i.e. the number of chunks per dimension)."""
    def chunk_key(self, chunk_indices: list[int]) -> str:
        """Return the store key of the chunk at `chunk_indices`."""
    @property
    def chunk_key_encoding(self) -> ChunkKeyEncoding:
        """The chunk key encoding, mapping chunk grid indices to store keys."""
    def chunk_origin(self, chunk_indices: list[int]) -> list[int]:
        """Return the origin of the chunk at `chunk_indices`.

        Raises if `chunk_indices` are incompatible with the chunk grid.
        """
    def chunk_shape(self, chunk_indices: list[int]) -> list[int]:
        """Return the shape of the chunk at `chunk_indices`.

        Raises if `chunk_indices` are incompatible with the chunk grid.
        """
    def chunk_subset(self, chunk_indices: list[int]) -> tuple[slice, ...]:
        """Return the array subset spanned by the chunk at `chunk_indices`.

        Returned as a tuple of slices, one per dimension.

        Raises if `chunk_indices` are incompatible with the chunk grid.
        """
    @property
    def compressors(self) -> list[BytesToBytesCodec]:
        """The bytes-to-bytes codecs ("compressors")."""
    @property
    def filters(self) -> list[ArrayToArrayCodec]:
        """The array-to-array codecs ("filters")."""
    @property
    def serializer(self) -> ArrayToBytesCodec:
        """The array-to-bytes codec ("serializer")."""
    @property
    def dimension_names(self) -> list[str | None] | None:
        """The dimension names, if any were specified."""
    @property
    def dtype(self) -> DataType:
        """The Zarr data type."""
    @property
    def fill_value(self) -> FillValue:
        """The array's fill value."""
    @property
    def metadata(self) -> ZarrV3ArrayMetadataJSON:
        """The array's full Zarr v3 metadata."""
    @property
    def ndim(self) -> int:
        """The number of dimensions."""
    @property
    def path(self) -> str:
        """The array's path in the store."""
    def retrieve_array_subset(
        self,
        selection: Selection,
        **codec_options: Unpack[CodecOptions],
    ) -> DecodedArray:
        """Read and decode an array region selected with numpy-style basic indexing.

        The result is ndim-preserving (consistent with a zarrs `ArraySubset`): an
        integer selects a length-1 range and that axis is retained.

        Keyword arguments are passed as [`CodecOptions`][zarrista.codec.CodecOptions].
        """
    def retrieve_chunk(
        self,
        chunk_indices: list[int],
        **codec_options: Unpack[CodecOptions],
    ) -> DecodedArray:
        """Read and decode the chunk at the given chunk grid indices.

        Keyword arguments are passed as [`CodecOptions`][zarrista.codec.CodecOptions].
        """
    def retrieve_encoded_chunk(self, chunk_indices: list[int]) -> Buffer | None:
        """Read the raw, still-encoded bytes of the chunk at `chunk_indices`.

        The bytes are returned verbatim, without running the codec pipeline.
        Returns `None` if the chunk is absent from the store.
        """
    def retrieve_subchunk(
        self,
        subchunk_indices: list[int],
        **codec_options: Unpack[CodecOptions],
    ) -> DecodedArray:
        """Read and decode a single subchunk (inner chunk) of a sharded array.

        `subchunk_indices` index the subchunk grid (see `subchunk_grid_shape`). For
        an unsharded array a subchunk is a whole chunk. Only the addressed inner
        chunk is read from its shard rather than the entire shard.

        Keyword arguments are passed as [`CodecOptions`][zarrista.codec.CodecOptions].
        """
    def retrieve_encoded_subchunk(self, subchunk_indices: list[int]) -> Buffer | None:
        """Read the raw, still-encoded bytes of a subchunk of a sharded array.

        The bytes are returned verbatim, without running the codec pipeline.
        Returns `None` if the subchunk is absent from the store.
        """
    @property
    def store(self) -> FilesystemStore | MemoryStore:
        """Retrieve the store backing this array."""
    def store_chunk(
        self,
        chunk_indices: list[int],
        decoded_chunk: ArrayBytes,
        **codec_options: Unpack[CodecOptions],
    ) -> None:
        """Encode `decoded_chunk` and write it as the chunk at `chunk_indices`.

        `decoded_chunk` holds the decoded chunk data; the array's codec pipeline
        encodes it before it is written. If the data equals the fill value and
        `store_empty_chunks` is `False`, the chunk is erased instead.

        Keyword arguments are passed as [`CodecOptions`][zarrista.codec.CodecOptions].
        """
    def store_encoded_chunk(
        self,
        chunk_indices: list[int],
        encoded_chunk: Buffer,
    ) -> None:
        """Write already-encoded bytes directly as the chunk at `chunk_indices`.

        The bytes are stored verbatim with no encoding. The caller is
        responsible for ensuring they match the array's codec pipeline; invalid
        bytes produce a chunk that cannot be decoded.
        """
    def store_metadata(self) -> None:
        """Write the array's metadata to the store.

        This is the write counterpart to
        [`Array.from_metadata`][zarrista.Array.from_metadata], which only constructs an
        in-memory array, without writing to the store.

        Any existing metadata at the array's path is overwritten.
        """
    def compact_chunk(
        self,
        chunk_indices: list[int],
        **codec_options: Unpack[CodecOptions],
    ) -> bool:
        """Re-encode the stored chunk in place, returning whether it was rewritten.

        Reads the encoded chunk, attempts to produce a more compact encoding,
        and rewrites it if that succeeds. Returns `True` if the chunk was
        rewritten, `False` if it was absent or already optimal.

        Keyword arguments are passed as [`CodecOptions`][zarrista.codec.CodecOptions].
        """
    def erase_chunk(self, chunk_indices: list[int]) -> None:
        """Delete the chunk at `chunk_indices` from the store.

        Erasing an absent chunk is a no-op.
        """
    def erase_metadata(self) -> None:
        """Delete the array's metadata from the store."""
    def read_only(self) -> Array:
        """Return a read-only view of this array.

        Reads behave identically, but any write (`store_chunk`, `erase_chunk`,
        `erase_metadata`, ...) raises at runtime.
        """
    @property
    def is_sharded(self) -> bool:
        """Whether the array's array-to-bytes codec is `sharding_indexed`."""
    @property
    def subchunk_shape(self) -> list[int] | None:
        """The inner-chunk shape from the `sharding_indexed` codec metadata.

        `None` if the array is not sharded.
        """
    @property
    def effective_subchunk_shape(self) -> list[int] | None:
        """The subchunk shape's effective "read granularity".

        Accounts for array-to-array codecs (e.g. `transpose`) that precede the
        sharding codec and reshape the subset spanned by one subchunk. `None` if
        the array is not sharded or the effective shape is indeterminate.
        """
    @property
    def subchunk_grid(self) -> ChunkGrid:
        """The subchunk grid.

        Built from the effective subchunk shape so that reading one subchunk reads
        a single contiguous byte range. For an unsharded array this is the normal
        chunk grid.
        """
    @property
    def subchunk_grid_shape(self) -> list[int]:
        """The shape of the subchunk grid (the number of subchunks per dimension).

        For an unsharded array this is the normal chunk grid shape.
        """
    @property
    def shape(self) -> list[int]:
        """The array shape."""
    @property
    def subset_all(self) -> tuple[slice, ...]:
        """The array subset that spans the entire array, as a tuple of slices."""
    def __getitem__(self, selection: Selection) -> DecodedArray:
        """Read a region with numpy-style basic indexing, e.g. `arr[0:10, :, 5]`.

        Sugar for `retrieve_array_subset`.
        """

class AsyncArray:
    """A Zarr array backed by an async store."""

    @staticmethod
    async def open_async(store: AsyncStore, path: str = "/") -> AsyncArray:
        """Open the array stored at `path` in `store`.

        `store` may be an obstore `ObjectStore` or an icechunk `Session`.
        """
    @staticmethod
    def from_metadata(
        metadata: ZarrV3ArrayMetadataJSON,
        store: AsyncStore,
        path: str = "/",
    ) -> AsyncArray:
        """Use the provided metadata to open a new array at `path` in `store`.

        This does **not** write the metadata to the store; use
        [`AsyncArray.store_metadata`][zarrista.AsyncArray.store_metadata] for
        that.
        """
    @property
    def attrs(self) -> dict[str, JSONValue]:
        """The array's user attributes as a dict."""
    @property
    def chunk_grid(self) -> ChunkGrid:
        """The chunk grid of the array."""
    @property
    def chunk_grid_shape(self) -> list[int]:
        """The shape of the chunk grid (i.e. the number of chunks per dimension)."""
    def chunk_key(self, chunk_indices: list[int]) -> str:
        """Return the store key of the chunk at `chunk_indices`."""
    @property
    def chunk_key_encoding(self) -> ChunkKeyEncoding:
        """The chunk key encoding, mapping chunk grid indices to store keys."""
    def chunk_origin(self, chunk_indices: list[int]) -> list[int]:
        """Return the origin of the chunk at `chunk_indices`.

        Raises if `chunk_indices` are incompatible with the chunk grid.
        """
    def chunk_shape(self, chunk_indices: list[int]) -> list[int]:
        """Return the shape of the chunk at `chunk_indices`.

        Raises if `chunk_indices` are incompatible with the chunk grid.
        """
    def chunk_subset(self, chunk_indices: list[int]) -> tuple[slice, ...]:
        """Return the array subset spanned by the chunk at `chunk_indices`.

        Returned as a tuple of slices, one per dimension.

        Raises if `chunk_indices` are incompatible with the chunk grid.
        """
    @property
    def compressors(self) -> list[BytesToBytesCodec]:
        """The bytes-to-bytes codecs ("compressors")."""
    @property
    def filters(self) -> list[ArrayToArrayCodec]:
        """The array-to-array codecs ("filters")."""
    @property
    def serializer(self) -> ArrayToBytesCodec:
        """The array-to-bytes codec ("serializer")."""
    @property
    def dimension_names(self) -> list[str | None] | None:
        """The dimension names, if any were specified."""
    @property
    def dtype(self) -> DataType:
        """The Zarr data type."""
    @property
    def fill_value(self) -> FillValue:
        """The array's fill value."""
    @property
    def metadata(self) -> ZarrV3ArrayMetadataJSON:
        """The array's full Zarr v3 metadata."""
    @property
    def ndim(self) -> int:
        """The number of dimensions."""
    @property
    def path(self) -> str:
        """The array's path in the store."""
    async def retrieve_array_subset(
        self,
        selection: Selection,
        **codec_options: Unpack[CodecOptions],
    ) -> DecodedArray:
        """Read and decode an array region selected with numpy-style basic indexing.

        The result is ndim-preserving (consistent with a zarrs `ArraySubset`): an
        integer selects a length-1 range and that axis is retained.

        Keyword arguments are passed as [`CodecOptions`][zarrista.codec.CodecOptions].
        """
    async def retrieve_chunk(
        self,
        chunk_indices: list[int],
        **codec_options: Unpack[CodecOptions],
    ) -> DecodedArray:
        """Read and decode the chunk at the given chunk grid indices.

        Keyword arguments are passed as [`CodecOptions`][zarrista.codec.CodecOptions].
        """
    async def retrieve_encoded_chunk(self, chunk_indices: list[int]) -> Buffer | None:
        """Read the raw, still-encoded bytes of the chunk at `chunk_indices`.

        The bytes are returned verbatim, without running the codec pipeline.
        Returns `None` if the chunk is absent from the store.
        """
    async def retrieve_subchunk(
        self,
        subchunk_indices: list[int],
        **codec_options: Unpack[CodecOptions],
    ) -> DecodedArray:
        """Read and decode a single subchunk (inner chunk) of a sharded array.

        `subchunk_indices` index the subchunk grid (see `subchunk_grid_shape`). For
        an unsharded array a subchunk is a whole chunk. Only the addressed inner
        chunk is read from its shard rather than the entire shard.

        Keyword arguments are passed as [`CodecOptions`][zarrista.codec.CodecOptions].
        """
    async def retrieve_encoded_subchunk(
        self,
        subchunk_indices: list[int],
    ) -> Buffer | None:
        """Read the raw, still-encoded bytes of a subchunk of a sharded array.

        The bytes are returned verbatim, without running the codec pipeline.
        Returns `None` if the subchunk is absent from the store.
        """
    @property
    def store(self) -> AsyncStore:
        """Retrieve the store backing this array."""
    async def store_chunk(
        self,
        chunk_indices: list[int],
        decoded_chunk: ArrayBytes,
        **codec_options: Unpack[CodecOptions],
    ) -> None:
        """Encode `decoded_chunk` and write it as the chunk at `chunk_indices`.

        `decoded_chunk` holds the decoded chunk data; the array's codec pipeline
        encodes it before it is written. If the data equals the fill value and
        `store_empty_chunks` is `False`, the chunk is erased instead.

        Keyword arguments are passed as [`CodecOptions`][zarrista.codec.CodecOptions].
        """
    async def store_encoded_chunk(
        self,
        chunk_indices: list[int],
        encoded_chunk: Buffer,
    ) -> None:
        """Write already-encoded bytes directly as the chunk at `chunk_indices`.

        The bytes are stored verbatim with no encoding. The caller is
        responsible for ensuring they match the array's codec pipeline; invalid
        bytes produce a chunk that cannot be decoded.
        """
    async def store_metadata(self) -> None:
        """Write the array's metadata to the store.

        This is the write counterpart to
        [`AsyncArray.from_metadata`][zarrista.AsyncArray.from_metadata], which only
        constructs an in-memory array, without writing to the store.

        Any existing metadata at the array's path is overwritten.
        """
    async def compact_chunk(
        self,
        chunk_indices: list[int],
        **codec_options: Unpack[CodecOptions],
    ) -> bool:
        """Re-encode the stored chunk in place, returning whether it was rewritten.

        Reads the encoded chunk, attempts to produce a more compact encoding,
        and rewrites it if that succeeds. Returns `True` if the chunk was
        rewritten, `False` if it was absent or already optimal.

        Keyword arguments are passed as [`CodecOptions`][zarrista.codec.CodecOptions].
        """
    async def erase_chunk(self, chunk_indices: list[int]) -> None:
        """Delete the chunk at `chunk_indices` from the store.

        Erasing an absent chunk is a no-op.
        """
    async def erase_metadata(self) -> None:
        """Delete the array's metadata from the store."""
    def read_only(self) -> AsyncArray:
        """Return a read-only view of this array.

        Reads behave identically, but any write (`store_chunk`, `erase_chunk`,
        `erase_metadata`, ...) raises at runtime.
        """
    @property
    def is_sharded(self) -> bool:
        """Whether the array's array-to-bytes codec is `sharding_indexed`."""
    @property
    def subchunk_shape(self) -> list[int] | None:
        """The inner-chunk shape from the `sharding_indexed` codec metadata.

        `None` if the array is not sharded.
        """
    @property
    def effective_subchunk_shape(self) -> list[int] | None:
        """The subchunk shape's effective "read granularity".

        Accounts for array-to-array codecs (e.g. `transpose`) that precede the
        sharding codec and reshape the subset spanned by one subchunk. `None` if
        the array is not sharded or the effective shape is indeterminate.
        """
    @property
    def subchunk_grid(self) -> ChunkGrid:
        """The subchunk grid.

        Built from the effective subchunk shape so that reading one subchunk reads
        a single contiguous byte range. For an unsharded array this is the normal
        chunk grid.
        """
    @property
    def subchunk_grid_shape(self) -> list[int]:
        """The shape of the subchunk grid (the number of subchunks per dimension).

        For an unsharded array this is the normal chunk grid shape.
        """
    @property
    def shape(self) -> list[int]:
        """The array shape."""
    @property
    def subset_all(self) -> tuple[slice, ...]:
        """The array subset that spans the entire array, as a tuple of slices."""
    async def __getitem__(self, selection: Selection) -> DecodedArray:
        """Read a region with numpy-style basic indexing: `await arr[0:10, :, 5]`.

        Sugar for `retrieve_array_subset`.
        """
