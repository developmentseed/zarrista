from . import codec, exceptions
from ._zarrista import (
    Array,
    ArrayBytes,
    AsyncArray,
    AsyncGroup,
    ChunkGrid,
    DataType,
    FilesystemStore,
    FillValue,
    Group,
    MaskedTensor,
    MaskedVariableArray,
    MemoryStore,
    Tensor,
    VariableArray,
    __version__,
)

DataArray = Tensor | VariableArray | MaskedTensor | MaskedVariableArray
"""The result of a read: one of the four decoded array layouts."""

__all__ = [
    "Array",
    "ArrayBytes",
    "AsyncArray",
    "AsyncGroup",
    "ChunkGrid",
    "DataType",
    "DataArray",
    "FillValue",
    "FilesystemStore",
    "Group",
    "MaskedTensor",
    "MaskedVariableArray",
    "MemoryStore",
    "Tensor",
    "VariableArray",
    "__version__",
    "codec",
    "exceptions",
]
