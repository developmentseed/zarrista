"""Available stores."""

from typing import TYPE_CHECKING, TypeAlias

from ._zarrista import AsyncZipStore, FilesystemStore, MemoryStore, ZipStore

if TYPE_CHECKING:
    from icechunk import Session
    from obstore.store import ObjectStore

SyncStore: TypeAlias = FilesystemStore | MemoryStore | ZipStore
"""A store that the sync API accepts.

This is a [`FilesystemStore`][zarrista.store.FilesystemStore], a
[`MemoryStore`][zarrista.store.MemoryStore], or a
[`ZipStore`][zarrista.store.ZipStore].
"""

# Note: this is a string so that icechunk and obstore can be optional dependencies
AsyncStore: TypeAlias = "ObjectStore | Session | AsyncZipStore"
"""A store that the async API accepts.

This is an obstore [`ObjectStore`][obstore.store.ObjectStore], an icechunk
[`Session`][icechunk.Session], or an
[`AsyncZipStore`][zarrista.store.AsyncZipStore]. An `ObjectStore` supports any
object-store backend, such as S3, GCS, or the local filesystem. A `Session`
gives a transactional, versioned store.

Note: the Rust extension serializes an icechunk `Session` and reconstructs it as
a separate icechunk instance. That instance must also be able to read the
session's data. Therefore, keep the data in storage such as a local filesystem
or S3. A session from `icechunk.in_memory_storage()` does not work, because the
data exists only in the Python process, and the reads cannot find it.
"""


__all__ = [
    "AsyncZipStore",
    "FilesystemStore",
    "MemoryStore",
    "ZipStore",
]
