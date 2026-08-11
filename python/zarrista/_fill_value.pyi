from collections.abc import Buffer

class FillValue:
    """An array's fill value, stored as native-endian bytes."""

    def __init__(self, bytes: bytes) -> None:
        """Construct a fill value from its native-endian bytes.

        Args:
            bytes: The native-endian bytes of one fill value element.
        """
    @property
    def size(self) -> int:
        """The size of the fill value in bytes."""
    def as_bytes(self) -> bytes:
        """Return the fill value as native-endian bytes.

        Returns:
            The native-endian bytes of one fill value element.
        """
    def equals_all(self, other: Buffer) -> bool:
        """Return whether `other` contains only repetitions of this fill value.

        Args:
            other: The bytes to compare against this fill value.

        Returns:
            `True` if `other` is a whole number of copies of this fill value.
                `False` if the bytes differ, or if the length of `other` is not a
                multiple of `size`.
        """
