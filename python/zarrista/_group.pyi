from collections.abc import Mapping

from zarr_metadata import (
    JSONValue,
    ZarrV3ConsolidatedMetadataJSON,
    ZarrV3GroupMetadataJSON,
)

from zarrista.store import AsyncStore, SyncStore

from ._array import Array, AsyncArray

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
    def with_attrs(self, attrs: Mapping[str, JSONValue]) -> Group:
        """Return a new group reference with `attrs`, leaving this one unchanged.

        The new attributes replace the old ones. Any key that `attrs` does not
        contain is gone from the new group reference. To keep the existing
        attributes, merge them yourself:

        ```py
        group = group.with_attrs({**group.attrs, "title": "root"})
        ```

        Nothing is persisted to the store. Call
        [`Group.store_metadata`][zarrista.Group.store_metadata] to persist the new
        attributes to the store:

        ```py
        group = group.with_attrs({"title": "root"})
        group.store_metadata()
        ```

        This group is unaffected and remains usable; it simply goes on describing
        the old attributes. Rebinding, as above, is the intended usage.

        Args:
            attrs: The user attributes of the new group reference. Each value
                must be JSON-serializable.

        Returns:
            A new group reference that uses `attrs`.

        Raises:
            TypeError: If a value in `attrs` is not JSON-serializable.
        """
    def with_consolidated_metadata(
        self,
        consolidated_metadata: ZarrV3ConsolidatedMetadataJSON | None,
    ) -> Group:
        """Return a new group reference with `consolidated_metadata`.

        This group is unchanged. Pass `None` to remove the consolidated metadata
        from the new group reference.

        This does not build the consolidated metadata. It stores the block that
        you give it, and it does not check the block against the hierarchy.

        Nothing is persisted to the store. Call
        [`Group.store_metadata`][zarrista.Group.store_metadata] to persist the new
        block to the store:

        ```py
        group = group.with_consolidated_metadata(consolidated_metadata)
        group.store_metadata()
        ```

        Args:
            consolidated_metadata: The consolidated metadata of the new group
                reference, or `None` to remove it. The block has the same shape
                that
                [`Group.consolidated_metadata`][zarrista.Group.consolidated_metadata]
                returns.

        Returns:
            A new group reference that uses `consolidated_metadata`.

        Raises:
            ValueError: If the group holds Zarr V2 metadata. Consolidated
                metadata is a Zarr V3 convention.
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
    def with_attrs(self, attrs: Mapping[str, JSONValue]) -> AsyncGroup:
        """Return a new group reference with `attrs`, leaving this one unchanged.

        The new attributes replace the old ones. Any key that `attrs` does not
        contain is gone from the new group reference. To keep the existing
        attributes, merge them yourself:

        ```py
        group = group.with_attrs({**group.attrs, "title": "root"})
        ```

        Nothing is persisted to the store. Call
        [`AsyncGroup.store_metadata`][zarrista.AsyncGroup.store_metadata] to
        persist the new attributes to the store:

        ```py
        group = group.with_attrs({"title": "root"})
        await group.store_metadata()
        ```

        This group is unaffected and remains usable; it simply goes on describing
        the old attributes. Rebinding, as above, is the intended usage.

        Args:
            attrs: The user attributes of the new group reference. Each value
                must be JSON-serializable.

        Returns:
            A new group reference that uses `attrs`.

        Raises:
            TypeError: If a value in `attrs` is not JSON-serializable.
        """
    def with_consolidated_metadata(
        self,
        consolidated_metadata: ZarrV3ConsolidatedMetadataJSON | None,
    ) -> AsyncGroup:
        """Return a new group reference with `consolidated_metadata`.

        This group is unchanged. Pass `None` to remove the consolidated metadata
        from the new group reference.

        This does not build the consolidated metadata. It stores the block that
        you give it, and it does not check the block against the hierarchy.

        Nothing is persisted to the store. Call
        [`AsyncGroup.store_metadata`][zarrista.AsyncGroup.store_metadata] to
        persist the new block to the store:

        ```py
        group = group.with_consolidated_metadata(consolidated_metadata)
        await group.store_metadata()
        ```

        Args:
            consolidated_metadata: The consolidated metadata of the new group
                reference, or `None` to remove it. The block has the same shape
                that
                [`AsyncGroup.consolidated_metadata`][zarrista.AsyncGroup.consolidated_metadata]
                returns.

        Returns:
            A new group reference that uses `consolidated_metadata`.

        Raises:
            ValueError: If the group holds Zarr V2 metadata. Consolidated
                metadata is a Zarr V3 convention.
        """
    @property
    def storage(self) -> AsyncStore:
        """The store that backs this group."""
