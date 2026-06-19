from typing import TypeAlias

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

DecodedArray: TypeAlias = Tensor | VariableArray | MaskedTensor | MaskedVariableArray
"""The result of a read: one of the four decoded array layouts.

Which one is returned depends on the dtype's byte layout (fixed vs. variable, and
whether it carries a validity mask). Use `isinstance` to narrow to a concrete
type before using layout-specific methods.
"""


__all__ = [
    "Array",
    "ArrayBytes",
    "AsyncArray",
    "AsyncGroup",
    "DecodedArray",
    "ChunkGrid",
    "DataType",
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
