from typing import Any

from obstore.store import ObjectStore

from ._chunks import ChunkGrid
from ._codec import CodecChain
from ._data import Data
from ._dtype import DataType
from ._store import FilesystemStore, MemoryStore

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
    def retrieve_chunk(self, chunk_indices: list[int]) -> Data:
        """Read and decode the chunk at the given chunk grid indices."""
    @property
    def shape(self) -> list[int]:
        """The array shape."""
    def __repr__(self) -> str: ...

class AsyncArray:
    """A read-only Zarr array backed by an async store."""

    @staticmethod
    async def open_async(store: ObjectStore, path: str = "/") -> AsyncArray:
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
    async def retrieve_chunk(self, chunk_indices: list[int]) -> Data:
        """Read and decode the chunk at the given chunk grid indices."""
    @property
    def shape(self) -> list[int]:
        """The array shape."""
    def __repr__(self) -> str: ...
