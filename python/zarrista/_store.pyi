from pathlib import Path
from typing import TypeAlias

from icechunk import Session
from obstore.store import ObjectStore

AsyncStore: TypeAlias = ObjectStore | Session
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

class FilesystemStore:
    """A store backed by a local directory."""

    def __init__(self, path: str | Path) -> None:
        """Open a filesystem store rooted at `path`."""

class MemoryStore:
    """An in-memory store, primarily useful for testing."""

    def __init__(self) -> None: ...
