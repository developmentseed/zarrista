from zarrista.codec._bytes_to_bytes import BytesToBytesCodec

def crc32c() -> BytesToBytesCodec:
    """Construct a `crc32c` codec.

    Appends a CRC32C checksum to the encoded bytestream.
    """
