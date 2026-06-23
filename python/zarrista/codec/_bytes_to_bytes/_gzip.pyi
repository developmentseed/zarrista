from zarrista.codec._bytes_to_bytes import BytesToBytesCodec

def gzip(level: int) -> BytesToBytesCodec:
    """Construct a `gzip` codec.

    `level` is the compression level, an integer from 0 (no compression) to 9
    (most compression).
    """
