from zarrista.codec._bytes_to_bytes import BytesToBytesCodec

def crc32c() -> BytesToBytesCodec:
    """Construct a `crc32c` codec.

    The codec appends a CRC32C checksum to the encoded bytestream.

    Returns:
        The new codec.
    """
