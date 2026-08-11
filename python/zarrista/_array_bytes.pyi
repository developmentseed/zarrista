from collections.abc import Buffer

class ArrayBytes:
    """Chunk bytes as input to or output from a codec.

    The object holds a data buffer. It can also hold the element byte offsets
    for variable-length data, and a validity mask.
    """

    def __init__(
        self,
        value: Buffer,
        /,
        *,
        mask: Buffer | None = None,
        offsets: list[int] | None = None,
    ) -> None:
        """Construct from a data buffer, with an optional mask and offsets.

        This does not check `mask` or `offsets` against `value`. The codec
        pipeline reports an error only when it uses the data.

        Args:
            value: The element bytes.

        Keyword Args:
            mask: The validity mask, with one byte per element. Give `None` for
                data that is not optional.
            offsets: The element byte offsets. Give `None` for fixed-length
                data.
        """  # noqa: DOC101, DOC103
    @property
    def bytes(self) -> Buffer:
        """The underlying element bytes (the data buffer for optional bytes)."""
    @property
    def offsets(self) -> list[int] | None:
        """Element byte offsets, or `None` for fixed-length data."""
    @property
    def mask(self) -> Buffer | None:
        """The validity mask (1 byte per element), or `None` if not optional."""
