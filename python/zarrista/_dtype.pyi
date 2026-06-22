from typing import Any, Literal, TypeAlias

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

Documents the common names for editor autocompletion; arbitrary strings (e.g.
raw `"r*"` types or extension data types) are still accepted by
[`DataType.from_string`][zarrista.DataType.from_string].
"""

class DataType:
    """A Zarr v3 data type."""

    @staticmethod
    def from_metadata(metadata: dict[str, Any]) -> DataType:
        """Construct a data type from its Zarr v3 metadata."""
    @staticmethod
    def from_string(name: DataTypeName | str) -> DataType:
        """Construct a data type from its Zarr v3 name (e.g. `"float32"`)."""
    @property
    def name(self) -> str | None:
        """The Zarr v3 data-type name (e.g. `"float64"`)."""
    @property
    def size(self) -> int | None:
        """The fixed size in bytes, or `None` for variable-length data types."""
    def __eq__(self, other: object) -> bool: ...
