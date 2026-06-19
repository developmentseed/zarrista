class ZarristaError(Exception):
    """Base class for all zarrista exceptions."""

class NotFoundError(ZarristaError):
    """Raised when no array or group exists at a path."""

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
    """Raised on an error from the underlying storage backend."""

class FilesystemStoreCreateError(ZarristaError):
    """Raised when a filesystem store cannot be opened."""

class CodecError(ZarristaError):
    """Raised on a codec encode/decode error."""

class TransposeOrderError(ZarristaError):
    """Raised when a transpose codec order is invalid."""

class PluginCreateError(ZarristaError):
    """Raised when a codec or other plugin cannot be created from its configuration."""

class SerializationError(ZarristaError):
    """Raised when (de)serializing JSON or converting to/from Python objects fails."""

__all__ = [
    "ArrayCreateError",
    "ArrayError",
    "CodecError",
    "FilesystemStoreCreateError",
    "GroupCreateError",
    "NodeCreateError",
    "NodePathError",
    "NotFoundError",
    "PluginCreateError",
    "SerializationError",
    "StorageError",
    "TransposeOrderError",
    "ZarristaError",
]
