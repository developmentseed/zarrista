from typing import Any

from ._array import Array, AsyncArray
from ._store import AsyncStore, FilesystemStore, MemoryStore

class Group:
    """A read-only Zarr group."""

    @staticmethod
    def open(store: FilesystemStore | MemoryStore, path: str = "/") -> Group:
        """Open the group stored at `path` in `store`."""
    @property
    def attrs(self) -> dict[str, Any]:
        """The group's user attributes as a dict."""
    def array_keys(self) -> list[str]:
        """Names of the direct child arrays."""
    def group_keys(self) -> list[str]:
        """Names of the direct child groups."""
    def __getitem__(self, name: str) -> Array | Group:
        """Open a direct child array or group by name."""

class AsyncGroup:
    """A read-only Zarr group backed by an async store."""

    @staticmethod
    async def open_async(store: AsyncStore, path: str = "/") -> AsyncGroup:
        """Open the group stored at `path` in `store`.

        `store` may be an obstore `ObjectStore` or an icechunk `Session`.
        """
    @property
    def attrs(self) -> dict[str, Any]:
        """The group's user attributes as a dict."""
    async def array_keys(self) -> list[str]:
        """Names of the direct child arrays."""
    async def group_keys(self) -> list[str]:
        """Names of the direct child groups."""
    async def open_child_async(self, name: str) -> AsyncArray | AsyncGroup:
        """Open a direct child array or group by name."""
