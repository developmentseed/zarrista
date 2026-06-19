from types import EllipsisType
from typing import Any, Unpack

from zarrista.codec import CodecOptions

from ._chunks import ChunkGrid
from ._codec import CodecChain
from ._data import DataArray
from ._dtype import DataType
from ._store import AsyncStore, FilesystemStore, MemoryStore

_AxisSelector = int | slice | EllipsisType
Selection = _AxisSelector | tuple[_AxisSelector, ...]
"""A numpy-style basic-indexing selection: what you would write inside `[]`.

Supports integers, step-1 slices, `Ellipsis`, and tuples of those (with fewer
entries than `ndim` implying full trailing axes). Negative indices and slice
bounds are normalized. `step != 1`, `None`/newaxis, boolean, and fancy/array
indexing are not supported.
"""

class Array:
    """A read-only Zarr array."""

    @staticmethod
    def open(store: FilesystemStore | MemoryStore, path: str = "/") -> Array:
        """Open the array stored at `path` in `store`."""
    @property
    def attrs(self) -> dict[str, Any]:
        """The array's user attributes as a dict."""
    @property
    def chunk_grid(self) -> ChunkGrid:
        """The chunk grid of the array."""
    @property
    def codecs(self) -> CodecChain:
        """The codec chain used to encode and decode the array's chunks."""
    @property
    def dimension_names(self) -> list[str | None] | None:
        """The dimension names, if any were specified."""
    @property
    def dtype(self) -> DataType:
        """The Zarr data type."""
    @property
    def metadata(self) -> dict[str, Any]:
        """The array's full Zarr v3 metadata as a dict."""
    @property
    def ndim(self) -> int:
        """The number of dimensions."""
    @property
    def path(self) -> str:
        """The array's path in the store."""
    def retrieve_array_subset(
        self, selection: Selection, **codec_options: Unpack[CodecOptions]
    ) -> DataArray:
        """Read and decode an array region selected with numpy-style basic indexing.

        The result is ndim-preserving (consistent with a zarrs `ArraySubset`): an
        integer selects a length-1 range and that axis is retained.

        Keyword arguments are passed as [`CodecOptions`][zarrista.codec.CodecOptions].
        """
    def retrieve_chunk(
        self, chunk_indices: list[int], **codec_options: Unpack[CodecOptions]
    ) -> DataArray:
        """Read and decode the chunk at the given chunk grid indices.

        Keyword arguments are passed as [`CodecOptions`][zarrista.codec.CodecOptions].
        """
    @property
    def shape(self) -> list[int]:
        """The array shape."""
    def __getitem__(self, selection: Selection) -> DataArray:
        """Read a region with numpy-style basic indexing, e.g. `arr[0:10, :, 5]`.

        Sugar for `retrieve_array_subset`.
        """
    def __repr__(self) -> str: ...

class AsyncArray:
    """A read-only Zarr array backed by an async store."""

    @staticmethod
    async def open_async(store: AsyncStore, path: str = "/") -> AsyncArray:
        """Open the array stored at `path` in `store`.

        `store` may be an obstore `ObjectStore` or an icechunk `Session`.
        """
    @property
    def attrs(self) -> dict[str, Any]:
        """The array's user attributes as a dict."""
    @property
    def chunk_grid(self) -> ChunkGrid:
        """The chunk grid of the array."""
    @property
    def codecs(self) -> CodecChain:
        """The codec chain used to encode and decode the array's chunks."""
    @property
    def dimension_names(self) -> list[str | None] | None:
        """The dimension names, if any were specified."""
    @property
    def dtype(self) -> DataType:
        """The Zarr data type."""
    @property
    def metadata(self) -> dict[str, Any]:
        """The array's full Zarr v3 metadata as a dict."""
    @property
    def ndim(self) -> int:
        """The number of dimensions."""
    @property
    def path(self) -> str:
        """The array's path in the store."""
    async def retrieve_array_subset(
        self, selection: Selection, **codec_options: Unpack[CodecOptions]
    ) -> DataArray:
        """Read and decode an array region selected with numpy-style basic indexing.

        The result is ndim-preserving (consistent with a zarrs `ArraySubset`): an
        integer selects a length-1 range and that axis is retained.

        Keyword arguments are passed as [`CodecOptions`][zarrista.codec.CodecOptions].
        """
    async def retrieve_chunk(
        self, chunk_indices: list[int], **codec_options: Unpack[CodecOptions]
    ) -> DataArray:
        """Read and decode the chunk at the given chunk grid indices.

        Keyword arguments are passed as [`CodecOptions`][zarrista.codec.CodecOptions].
        """
    @property
    def shape(self) -> list[int]:
        """The array shape."""
    async def __getitem__(self, selection: Selection) -> DataArray:
        """Read a region with numpy-style basic indexing: `await arr[0:10, :, 5]`.

        Sugar for `retrieve_array_subset`.
        """
    def __repr__(self) -> str: ...
