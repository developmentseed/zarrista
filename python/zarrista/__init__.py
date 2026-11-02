"""A low-level Zarr API for Python, binding to Rust's Zarrs."""

from typing import TypeAlias

from . import codec, exceptions, store
from ._zarrista import (
    Array,
    ArrayBuilder,
    ArrayBytes,
    ChunkGrid,
    ChunkKeyEncoding,
    DataType,
    EncodedChunk,
    FillValue,
    FixedLengthTensor,
    Group,
    OptionalFixedLengthTensor,
    OptionalVariableLengthTensor,
    ShardCache,
    ThreadPool,
    VariableLengthTensor,
    __version__,
)

# The async classes are only registered by the Rust extension when it is
# built with the `async` feature. The pyemscripten wheel is built with
# `--no-default-features`, so importing them unconditionally makes
# `import zarrista` fail on Pyodide. Import them only when they are present.
try:
    from ._zarrista import AsyncArray, AsyncGroup, AsyncShardCache
except ImportError:
    _ASYNC_CLASSES_AVAILABLE = False
else:
    _ASYNC_CLASSES_AVAILABLE = True

Tensor: TypeAlias = (
    FixedLengthTensor
    | VariableLengthTensor
    | OptionalFixedLengthTensor
    | OptionalVariableLengthTensor
)
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
    "AsyncShardCache",
    "ChunkGrid",
    "ChunkKeyEncoding",
    "DataType",
    "EncodedChunk",
    "FillValue",
    "FixedLengthTensor",
    "Group",
    "OptionalFixedLengthTensor",
    "OptionalVariableLengthTensor",
    "ShardCache",
    "Tensor",
    "ThreadPool",
    "VariableLengthTensor",
    "__version__",
    "codec",
    "exceptions",
    "store",
]

if not _ASYNC_CLASSES_AVAILABLE:
    __all__ = [
        name
        for name in __all__
        if name not in ("AsyncArray", "AsyncGroup", "AsyncShardCache")
    ]
