from zarr_metadata import ConsolidatedMetadataV3, GroupMetadataV3, JSONValue

from ._array import Array, AsyncArray
from ._store import AsyncStore, FilesystemStore, MemoryStore

class Group:
    """A Zarr group."""

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
    def traverse(self) -> list[Array | Group]:
        """Return every node under the group, recursively."""
    def child_arrays(self) -> list[Array]:
        """Return the direct child arrays of the group."""
    def child_groups(self) -> list[Group]:
        """Return the direct child groups of the group."""
    def child_paths(self) -> list[str]:
        """Return the full paths of the group's direct children."""
    def child_array_paths(self) -> list[str]:
        """Return the full paths of the group's direct child arrays."""
    def child_group_paths(self) -> list[str]:
        """Return the full paths of the group's direct child groups."""
    def erase_metadata(self) -> None:
        """Erase the group metadata from the store. Succeeds if it does not exist."""
    def child(self, name: str) -> Array | Group:
        """Open a direct child array or group by name."""
    def store_metadata(self) -> None:
        """Write the group metadata to the store."""
    @property
    def store(self) -> FilesystemStore | MemoryStore:
        """Retrieve the store backing this group."""
    def __getitem__(self, name: str) -> Array | Group:
        """Open a direct child array or group by name."""

class AsyncGroup:
    """A Zarr group backed by an async store."""

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
    async def traverse(self) -> list[AsyncArray | AsyncGroup]:
        """Return every node under the group, recursively."""
    async def child_arrays(self) -> list[AsyncArray]:
        """Return the direct child arrays of the group."""
    async def child_groups(self) -> list[AsyncGroup]:
        """Return the direct child groups of the group."""
    async def child_paths(self) -> list[str]:
        """Return the full paths of the group's direct children."""
    async def child_array_paths(self) -> list[str]:
        """Return the full paths of the group's direct child arrays."""
    async def child_group_paths(self) -> list[str]:
        """Return the full paths of the group's direct child groups."""
    async def store_metadata(self) -> None:
        """Write the group metadata to the store."""
    async def erase_metadata(self) -> None:
        """Erase the group metadata from the store. Succeeds if it does not exist."""
    async def open_child_async(self, name: str) -> AsyncArray | AsyncGroup:
        """Open a direct child array or group by name."""
    @property
    def store(self) -> AsyncStore:
        """Retrieve the store backing this group."""
