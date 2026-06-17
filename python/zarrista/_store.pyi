from pathlib import Path

from icechunk import Session
from obstore.store import ObjectStore

AsyncStore = ObjectStore | Session
"""A store accepted by the async API.

Either an obstore [`ObjectStore`][obstore.store.ObjectStore] (any object-store
backend, e.g. S3, GCS, local) or an icechunk [`Session`][icechunk.Session]
(a transactional, versioned store).
"""

class FilesystemStore:
    """A store backed by a local directory."""

    def __init__(self, path: str | Path) -> None:
        """Open a filesystem store rooted at `path`."""
    def __repr__(self) -> str: ...

class MemoryStore:
    """An in-memory store, primarily useful for testing."""

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...
