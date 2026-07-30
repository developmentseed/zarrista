"""Available stores."""

from typing import TYPE_CHECKING, TypeAlias

from ._zarrista import FilesystemStore, MemoryStore

if TYPE_CHECKING:
    from icechunk import Session
    from obstore.store import ObjectStore

# Note: this is a string so that icechunk and obstore can be optional dependencies
AsyncStore: TypeAlias = "ObjectStore | Session"
"""A store accepted by the async API.

Either an obstore [`ObjectStore`][obstore.store.ObjectStore] (any object-store
backend, e.g. S3, GCS, local) or an icechunk [`Session`][icechunk.Session]
(a transactional, versioned store).

Note: an icechunk `Session` is serialized and reconstructed inside the Rust
extension, which runs as a separate icechunk instance. The session's data must
therefore live in storage that instance can also read (local filesystem, S3,
etc.). Sessions backed by `icechunk.in_memory_storage()` will not work, since
that data only exists in the Python process and reads will fail to find it.
"""


__all__ = [
    "FilesystemStore",
    "MemoryStore",
]
