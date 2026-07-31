from zarrista.codec._bytes_to_bytes import BytesToBytesCodec

def zstd(level: int, checksum: bool) -> BytesToBytesCodec:
    """Construct a `zstd` codec.

    Args:
        level: The compression level. The codec does not check the range, and
            zstd clamps the value to the range that it supports.
        checksum: Whether to write a checksum to the encoded bytestream. The
            codec verifies this checksum when it decodes the bytestream.

    Returns:
        The new codec.
    """
