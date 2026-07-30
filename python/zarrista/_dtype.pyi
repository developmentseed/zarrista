from typing import TypeAlias

from zarr_metadata import ZarrV3NamedConfigJSON
from zarr_metadata.v3.data_type import (
    BoolDataTypeName,
    BytesDataTypeName,
    Complex64DataTypeName,
    Complex128DataTypeName,
    Float16DataTypeName,
    Float32DataTypeName,
    Float64DataTypeName,
    Int8DataTypeName,
    Int16DataTypeName,
    Int32DataTypeName,
    Int64DataTypeName,
    RawBytesDataTypeName,
    StringDataTypeName,
    Uint8DataTypeName,
    Uint16DataTypeName,
    Uint32DataTypeName,
    Uint64DataTypeName,
)

DataTypeName: TypeAlias = (
    BoolDataTypeName
    | Int8DataTypeName
    | Int16DataTypeName
    | Int32DataTypeName
    | Int64DataTypeName
    | Uint8DataTypeName
    | Uint16DataTypeName
    | Uint32DataTypeName
    | Uint64DataTypeName
    | Float16DataTypeName
    | Float32DataTypeName
    | Float64DataTypeName
    | Complex64DataTypeName
    | Complex128DataTypeName
    | StringDataTypeName
    | BytesDataTypeName
    | RawBytesDataTypeName
)
"""The Zarr v3 names of the data types `from_string` can build.

Composed from the per-dtype name literals in `zarr_metadata.v3.data_type`, so
it stays in sync with the spec rather than being hand-maintained here.
"""

class DataType:
    """A Zarr v3 data type."""

    @staticmethod
    def from_metadata(metadata: ZarrV3NamedConfigJSON) -> DataType:
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
