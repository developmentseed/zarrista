from pathlib import Path

class FilesystemStore:
    """A store backed by a local directory."""

    def __init__(self, path: str | Path) -> None:
        """Open a filesystem store rooted at `path`."""
    def __repr__(self) -> str: ...

class MemoryStore:
    """An in-memory store, primarily useful for testing."""

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...
