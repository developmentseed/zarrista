from collections.abc import Buffer

class FillValue:
    """An array's fill value, stored as native-endian bytes."""

    def __init__(self, bytes: bytes) -> None:
        """Construct a fill value from its native-endian bytes."""
    @property
    def size(self) -> int:
        """The size of the fill value in bytes."""
    def as_bytes(self) -> bytes:
        """The fill value as native-endian bytes."""
    def equals_all(self, other: Buffer) -> bool:
        """Whether `other` is entirely repetitions of this fill value."""
