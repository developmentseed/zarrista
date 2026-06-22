"""A low-level Zarr API for Python, binding to Rust's Zarrs."""

from typing import Literal, TypeAlias

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

DataTypeName: TypeAlias = Literal[
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "string",
    "bytes",
]
"""The Zarr v3 names of the built-in fixed data types.
"""


__all__ = [
    "Array",
    "ArrayBytes",
    "AsyncArray",
    "AsyncGroup",
    "ChunkGrid",
    "DataType",
    "DataTypeName",
    "DecodedArray",
    "FilesystemStore",
    "FillValue",
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
