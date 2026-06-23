class ZarristaError(Exception):
    """Base class for all zarrista exceptions."""

class ArrayCreateError(ZarristaError):
    """Raised when an array cannot be opened or created."""

class ArrayError(ZarristaError):
    """Raised on an error reading from or operating on an array."""

class GroupCreateError(ZarristaError):
    """Raised when a group cannot be opened or created."""

class NodeCreateError(ZarristaError):
    """Raised when a child node cannot be enumerated or created."""

class NodePathError(ZarristaError):
    """Raised when a node path is invalid."""

class StorageError(ZarristaError):
    """Raised on an error from the underlying storage backend.

    This also covers failures opening a filesystem store.
    """

class CodecError(ZarristaError):
    """Raised on a codec encode/decode error."""

class TransposeOrderError(ZarristaError):
    """Raised when a transpose codec order is invalid."""

class PluginCreateError(ZarristaError):
    """Raised when a codec or other plugin cannot be created from its configuration."""

class SerializationError(ZarristaError):
    """Raised when (de)serializing JSON or converting to/from Python objects fails."""

class ChunkGridCreateError(ZarristaError):
    """Raised when a chunk grid cannot be created from the given shapes."""

class IncompatibleDimensionalityError(ZarristaError):
    """Raised when a shape's dimensionality is incompatible with another."""

__all__ = [
    "ArrayCreateError",
    "ArrayError",
    "ChunkGridCreateError",
    "CodecError",
    "GroupCreateError",
    "IncompatibleDimensionalityError",
    "NodeCreateError",
    "NodePathError",
    "PluginCreateError",
    "SerializationError",
    "StorageError",
    "TransposeOrderError",
    "ZarristaError",
]
