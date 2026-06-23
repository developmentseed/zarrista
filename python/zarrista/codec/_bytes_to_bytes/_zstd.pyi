from zarrista.codec._bytes_to_bytes import BytesToBytesCodec

def zstd(level: int, checksum: bool) -> BytesToBytesCodec:
    """Construct a `zstd` codec.

    `level` is the compression level. When `checksum` is true, a checksum is
    written to (and verified on decode from) the encoded bytestream.
    """
