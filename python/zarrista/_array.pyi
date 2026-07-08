from collections.abc import Buffer
from types import EllipsisType
from typing import TypeAlias, Unpack

from zarr_metadata import ArrayMetadataV3, JSONValue

from zarrista.codec import (
    ArrayToArrayCodec,
    ArrayToBytesCodec,
    BytesToBytesCodec,
    CodecOptions,
)

from ._array_bytes import ArrayBytes
from ._chunks import ChunkGrid
from ._decoded_array import DecodedArray
from ._dtype import DataType
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
        metadata: ArrayMetadataV3,
        store: FilesystemStore | MemoryStore,
        path: str = "/",
    ) -> Array:
        """Use the provided metadata to open a new array at `path` in `store`.

        This does **not** write the metadata to the store.
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
    def metadata(self) -> ArrayMetadataV3:
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
    def shape(self) -> list[int]:
        """The array shape."""
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
        metadata: ArrayMetadataV3,
        store: AsyncStore,
        path: str = "/",
    ) -> AsyncArray:
        """Use the provided metadata to open a new array at `path` in `store`.

        This does **not** write the metadata to the store.
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
    def metadata(self) -> ArrayMetadataV3:
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
    def shape(self) -> list[int]:
        """The array shape."""
    async def __getitem__(self, selection: Selection) -> DecodedArray:
        """Read a region with numpy-style basic indexing: `await arr[0:10, :, 5]`.

        Sugar for `retrieve_array_subset`.
        """
