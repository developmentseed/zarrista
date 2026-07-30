from pathlib import Path
from typing import TypeAlias

from icechunk import Session
from obstore.store import ObjectStore

AsyncStore: TypeAlias = ObjectStore | Session
"""A store that the async API accepts.

This is either an obstore [`ObjectStore`][obstore.store.ObjectStore] or an
icechunk [`Session`][icechunk.Session]. An `ObjectStore` supports any
object-store backend, such as S3, GCS, or the local filesystem. A `Session`
gives a transactional, versioned store.

Note: the Rust extension serializes an icechunk `Session` and reconstructs it as
a separate icechunk instance. That instance must also be able to read the
session's data. Therefore, keep the data in storage such as a local filesystem
or S3. A session from `icechunk.in_memory_storage()` does not work, because the
data exists only in the Python process, and the reads cannot find it.
"""

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
