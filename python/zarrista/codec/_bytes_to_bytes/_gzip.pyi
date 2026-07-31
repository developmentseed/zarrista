from zarrista.codec._bytes_to_bytes import BytesToBytesCodec

def gzip(level: int) -> BytesToBytesCodec:
    """Construct a `gzip` codec.

    Args:
        level: The compression level, from 0 (no compression) to 9 (most
            compression).

    Returns:
        The new codec.

    Raises:
        ValueError: If `level` is more than 9.
        OverflowError: If `level` is negative.
    """
