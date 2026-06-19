from typing import Any

class DataType:
    """A Zarr v3 data type."""

    def __init__(self, metadata: dict[str, Any]) -> None:
        """Construct a data type from its Zarr v3 metadata."""
    @property
    def name(self) -> str | None:
        """The Zarr v3 data-type name (e.g. `"float64"`)."""
    @property
    def size(self) -> int | None:
        """The fixed size in bytes, or `None` for variable-length data types."""
    def __eq__(self, other: object) -> bool: ...
