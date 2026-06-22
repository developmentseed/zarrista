from zarr_metadata import ConsolidatedMetadataV3, GroupMetadataV3, JSONValue

from ._array import Array, AsyncArray
from ._store import AsyncStore, FilesystemStore, MemoryStore

class Group:
    """A read-only Zarr group."""

    @staticmethod
    def open(store: FilesystemStore | MemoryStore, path: str = "/") -> Group:
        """Open the group stored at `path` in `store`."""
    @property
    def attrs(self) -> dict[str, JSONValue]:
        """The group's user attributes as a dict."""
    @property
    def metadata(self) -> GroupMetadataV3:
        """The group's full Zarr v3 metadata."""
    @property
    def consolidated_metadata(self) -> ConsolidatedMetadataV3 | None:
        """The consolidated metadata, if present in the group metadata."""
    @property
    def path(self) -> str:
        """The group's path in the store."""
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
    def attrs(self) -> dict[str, JSONValue]:
        """The group's user attributes as a dict."""
    @property
    def metadata(self) -> GroupMetadataV3:
        """The group's full Zarr v3 metadata."""
    @property
    def consolidated_metadata(self) -> ConsolidatedMetadataV3 | None:
        """The consolidated metadata, if present in the group metadata."""
    @property
    def path(self) -> str:
        """The group's path in the store."""
    async def array_keys(self) -> list[str]:
        """Names of the direct child arrays."""
    async def group_keys(self) -> list[str]:
        """Names of the direct child groups."""
    async def open_child_async(self, name: str) -> AsyncArray | AsyncGroup:
        """Open a direct child array or group by name."""
