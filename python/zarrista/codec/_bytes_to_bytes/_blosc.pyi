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

    Args:
        cname: The name of the internal compressor.
        clevel: The compression level, from 0 (no compression) to 9 (most
            compression).
        shuffle_mode: The shuffle mode to apply before compression.

    Keyword Args:
        blocksize: The block size in bytes. Give `None` or `0` to let blosc
            choose the block size.
        typesize: The size of one element in bytes. This must be a positive
            integer if `shuffle_mode` is not `"noshuffle"`.

    Returns:
        The new codec.

    Raises:
        ValueError: If `cname` is not a known compressor, or if `clevel` is
            outside the range 0 to 9.
        PluginCreateError: If `shuffle_mode` is not `"noshuffle"` and
            `typesize` is `None` or `0`.
    """  # noqa: DOC101, DOC103
