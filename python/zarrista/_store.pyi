from pathlib import Path

from zarrista.store import AsyncStore, SyncStore

class FilesystemStore:
    """A store backed by a local directory."""

    def __init__(self, path: str | Path) -> None:
        """Open a filesystem store rooted at `path`.

        This does not check that `path` exists. The store reports an error only
        when you read from it or write to it.

        Args:
            path: The path to the root directory of the store.
        """

class MemoryStore:
    """An in-memory store, primarily useful for testing."""

    def __init__(self) -> None: ...

class ZipStore:
    """A read-only store backed by a zip file that another store holds.

    The store reads the zip metadata when you open it, and then serves each Zarr
    key from an entry in the zip file.

    This store is read only and does not support writing.

    Examples:
        Open a Zarr array that is held in a local zip file:

        >>> from zarrista import Array
        >>> from zarrista.store import FilesystemStore, ZipStore
        >>> backing = FilesystemStore("/data")
        >>> store = ZipStore(backing, "archive.zip")
        >>> array = Array.open(store)
    """

    def __init__(
        self,
        store: SyncStore,
        key: str,
        *,
        path: str | None = None,
    ) -> None:
        """Open the zip file that is stored at `key` in `store`.

        Args:
            store: The store that holds the zip file.
            key: The store key of the zip file, such as `archive.zip`.
            path: The directory inside the zip file to use as the root of this
                store. A value of `None` uses the whole zip file. The store
                accepts `nested`, `nested/`, and `/nested/` as the same
                directory.

        Raises:
            StorageError: If `store` holds no value at `key`, or if that value
                is not a valid zip file.
            ValueError: If `key` is not a valid store key.
        """

    @staticmethod
    def open(
        store: SyncStore,
        key: str,
        *,
        path: str | None = None,
    ) -> ZipStore:
        """Open the zip file that is stored at `key` in `store`.

        This is an alias for [`ZipStore`][zarrista.store.ZipStore].

        Args:
            store: The store that holds the zip file.
            key: The store key of the zip file, such as `archive.zip`.
            path: The directory inside the zip file to use as the root of this
                store. A value of `None` uses the whole zip file. The store
                accepts `nested`, `nested/`, and `/nested/` as the same
                directory.

        Returns:
            The store for the zip file at `key`.

        Raises:
            StorageError: If `store` holds no value at `key`, or if that value
                is not a valid zip file.
            ValueError: If `key` is not a valid store key.
        """

class AsyncZipStore:
    """A read-only store backed by a zip file that another async store holds.

    This is the async form of [`ZipStore`][zarrista.store.ZipStore]. Use it with
    [`AsyncArray`][zarrista.AsyncArray] and
    [`AsyncGroup`][zarrista.AsyncGroup].

    This store is read only and does not support writing.
    """

    @staticmethod
    async def open(
        store: AsyncStore,
        key: str,
        *,
        path: str | None = None,
    ) -> AsyncZipStore:
        """Open the zip file that is stored at `key` in `store`.

        Args:
            store: The async store that holds the zip file.
            key: The store key of the zip file, such as `archive.zip`.
            path: The directory inside the zip file to use as the root of this
                store. A value of `None` uses the whole zip file. The store
                accepts `nested`, `nested/`, and `/nested/` as the same
                directory.

        Returns:
            The store for the zip file at `key`.

        Raises:
            StorageError: If `store` holds no value at `key`, or if that value
                is not a valid zip file.
            ValueError: If `key` is not a valid store key.
        """
