"""A low-level Zarr API for Python, binding to Rust's Zarrs."""

from typing import TypeAlias

from . import codec, exceptions, store
from ._zarrista import (
    Array,
    ArrayBuilder,
    ArrayBytes,
    AsyncArray,
    AsyncGroup,
    ChunkGrid,
    ChunkKeyEncoding,
    DataType,
    EncodedChunk,
    FillValue,
    Group,
    MaskedTensor,
    MaskedVariableArray,
    Tensor,
    ThreadPool,
    VariableArray,
    __version__,
)

DecodedArray: TypeAlias = Tensor | VariableArray | MaskedTensor | MaskedVariableArray
"""The result of a read: one of the four decoded array layouts.

The layout depends on the byte layout of the data type. A data type is either
fixed-width or variable-length, and it either carries a validity mask or does
not. Use `isinstance` to narrow to a concrete type before you use a method that
belongs to one layout.
"""


__all__ = [
    "Array",
    "ArrayBuilder",
    "ArrayBytes",
    "AsyncArray",
    "AsyncGroup",
    "ChunkGrid",
    "ChunkKeyEncoding",
    "DataType",
    "DecodedArray",
    "EncodedChunk",
    "FillValue",
    "Group",
    "MaskedTensor",
    "MaskedVariableArray",
    "Tensor",
    "ThreadPool",
    "VariableArray",
    "__version__",
    "codec",
    "exceptions",
    "store",
]
