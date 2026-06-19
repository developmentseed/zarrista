from collections.abc import Buffer

class ArrayBytes:
    """Chunk bytes as input or output from a codec.

    Wraps a data buffer, plus optional element byte offsets (for variable-length
    data) and an optional validity mask.
    """

    def __init__(
        self,
        bytes: Buffer,
        *,
        mask: Buffer | None = None,
        offsets: list[int] | None = None,
    ) -> None:
        """Construct from a data buffer, with an optional mask and offsets."""
    @property
    def bytes(self) -> Buffer:
        """The underlying element bytes (the data buffer for optional bytes)."""
    @property
    def offsets(self) -> list[int] | None:
        """Element byte offsets, or `None` for fixed-length data."""
    @property
    def mask(self) -> Buffer | None:
        """The validity mask (1 byte per element), or `None` if not optional."""
