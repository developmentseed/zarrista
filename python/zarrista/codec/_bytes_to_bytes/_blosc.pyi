from typing import Literal, TypeAlias

from zarrista.codec._bytes_to_bytes import BytesToBytesCodec

BloscCompressor: TypeAlias = Literal[
    "blosclz",
    "lz4",
    "lz4hc",
    "snappy",
    "zlib",
    "zstd",
]
"""A `blosc` compressor name."""

BloscShuffle: TypeAlias = Literal["noshuffle", "shuffle", "bitshuffle"]
"""A `blosc` shuffle mode."""

def blosc(
    cname: BloscCompressor,
    clevel: int,
    shuffle_mode: BloscShuffle,
    *,
    blocksize: int | None = None,
    typesize: int | None = None,
) -> BytesToBytesCodec:
    """Construct a `blosc` codec from its parameters.

    `clevel` is the compression level, an integer from 0 (no compression) to 9
    (most compression). `typesize` is required (a positive integer) whenever
    `shuffle_mode` is not `"noshuffle"`. The block size is chosen automatically
    when `blocksize` is `None` or `0`.
    """
