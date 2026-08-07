from zarr_metadata import (
    JSONValue,
    ZarrV3ConsolidatedMetadataJSON,
    ZarrV3GroupMetadataJSON,
)

from ._array import Array, AsyncArray
from ._store import AsyncStore, SyncStore

class Group:
    """A Zarr group."""

    @staticmethod
    def open(store: SyncStore, path: str = "/") -> Group:
        """Open the group stored at `path` in `store`.

        Args:
            store: The store that holds the group.
            path: The absolute path of the group in the store.

        Returns:
            The group at `path`.

        Raises:
            GroupCreateError: If the store holds no group metadata at `path`.
            ValueError: If `path` is not a valid absolute node path.
        """
    @property
    def attrs(self) -> dict[str, JSONValue]:
        """The group's user attributes as a dict."""
    @property
    def metadata(self) -> ZarrV3GroupMetadataJSON:
        """The group's full Zarr v3 metadata."""
    @property
    def consolidated_metadata(self) -> ZarrV3ConsolidatedMetadataJSON | None:
        """The consolidated metadata, if present in the group metadata."""
    @property
    def path(self) -> str:
        """The group's path in the store."""
    def array_keys(self) -> list[str]:
        """Return the names of the direct child arrays.

        Returns:
            The name of each direct child array.
        """
    def group_keys(self) -> list[str]:
        """Return the names of the direct child groups.

        Returns:
            The name of each direct child group.
        """
    def traverse(self) -> list[Array | Group]:
        """Return every node under the group, recursively.

        Returns:
            Each array and group below this group, at any depth.
        """
    def child_arrays(self) -> list[Array]:
        """Return the direct child arrays of the group.

        Returns:
            Each direct child array.
        """
    def child_groups(self) -> list[Group]:
        """Return the direct child groups of the group.

        Returns:
            Each direct child group.
        """
    def child_paths(self) -> list[str]:
        """Return the full paths of the group's direct children.

        Returns:
            The full path of each direct child.
        """
    def child_array_paths(self) -> list[str]:
        """Return the full paths of the group's direct child arrays.

        Returns:
            The full path of each direct child array.
        """
    def child_group_paths(self) -> list[str]:
        """Return the full paths of the group's direct child groups.

        Returns:
            The full path of each direct child group.
        """
    def erase_metadata(self) -> None:
        """Erase the group metadata from the store.

        This succeeds if the metadata does not exist.
        """
    def child(self, name: str) -> Array | Group:
        """Open a direct child array or group by name.

        Args:
            name: The name of the direct child.

        Returns:
            The child array or group.

        Raises:
            KeyError: If the group has no direct child with that name.
        """
    def store_metadata(self) -> None:
        """Write the group metadata to the store.

        This overwrites any metadata that exists at the group's path.
        """
    @property
    def storage(self) -> SyncStore:
        """The store that backs this group."""
    def __getitem__(self, name: str) -> Array | Group:
        """Open a direct child array or group by name.

        Args:
            name: The name of the direct child.

        Returns:
            The child array or group.

        Raises:
            KeyError: If the group has no direct child with that name.
        """

class AsyncGroup:
    """A Zarr group backed by an async store."""

    @staticmethod
    async def open(store: AsyncStore, path: str = "/") -> AsyncGroup:
        """Open the group stored at `path` in `store`.

        Args:
            store: The store that holds the group. This is either an obstore
                `ObjectStore` or an icechunk `Session`.
            path: The absolute path of the group in the store.

        Returns:
            The group at `path`.

        Raises:
            GroupCreateError: If the store holds no group metadata at `path`.
            ValueError: If `path` is not a valid absolute node path.
        """
    @property
    def attrs(self) -> dict[str, JSONValue]:
        """The group's user attributes as a dict."""
    @property
    def metadata(self) -> ZarrV3GroupMetadataJSON:
        """The group's full Zarr v3 metadata."""
    @property
    def consolidated_metadata(self) -> ZarrV3ConsolidatedMetadataJSON | None:
        """The consolidated metadata, if present in the group metadata."""
    @property
    def path(self) -> str:
        """The group's path in the store."""
    async def array_keys(self) -> list[str]:
        """Return the names of the direct child arrays.

        Returns:
            The name of each direct child array.
        """
    async def group_keys(self) -> list[str]:
        """Return the names of the direct child groups.

        Returns:
            The name of each direct child group.
        """
    async def traverse(self) -> list[AsyncArray | AsyncGroup]:
        """Return every node under the group, recursively.

        Returns:
            Each array and group below this group, at any depth.
        """
    async def child_arrays(self) -> list[AsyncArray]:
        """Return the direct child arrays of the group.

        Returns:
            Each direct child array.
        """
    async def child_groups(self) -> list[AsyncGroup]:
        """Return the direct child groups of the group.

        Returns:
            Each direct child group.
        """
    async def child_paths(self) -> list[str]:
        """Return the full paths of the group's direct children.

        Returns:
            The full path of each direct child.
        """
    async def child_array_paths(self) -> list[str]:
        """Return the full paths of the group's direct child arrays.

        Returns:
            The full path of each direct child array.
        """
    async def child_group_paths(self) -> list[str]:
        """Return the full paths of the group's direct child groups.

        Returns:
            The full path of each direct child group.
        """
    async def store_metadata(self) -> None:
        """Write the group metadata to the store.

        This overwrites any metadata that exists at the group's path.
        """
    async def erase_metadata(self) -> None:
        """Erase the group metadata from the store.

        This succeeds if the metadata does not exist.
        """
    async def child(self, name: str) -> AsyncArray | AsyncGroup:
        """Open a direct child array or group by name.

        Args:
            name: The name of the direct child.

        Returns:
            The child array or group.

        Raises:
            KeyError: If the group has no direct child with that name.
        """
    @property
    def storage(self) -> AsyncStore:
        """The store that backs this group."""
