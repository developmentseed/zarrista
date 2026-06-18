from typing import Any, Literal

from zarrista.codec._bytes_to_bytes import BytesToBytesCodec

BloscCompressor = Literal["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"]
"""A `blosc` compressor name."""

BloscShuffle = Literal["noshuffle", "shuffle", "bitshuffle"]
"""A `blosc` shuffle mode."""

class Blosc(BytesToBytesCodec):
    """The `blosc` bytes-to-bytes codec."""

    def __init__(
        self,
        cname: BloscCompressor,
        clevel: int,
        shuffle_mode: BloscShuffle,
        *,
        blocksize: int | None = None,
        typesize: int | None = None,
    ) -> None:
        """Construct a `blosc` codec from its parameters.

        `clevel` is the compression level, an integer from 0 (no compression)
        to 9 (most compression). `typesize` is required (a positive integer)
        whenever `shuffle_mode` is not `"noshuffle"`. The block size is chosen
        automatically when `blocksize` is `None` or `0`.
        """
    @staticmethod
    def from_config(config: dict[str, Any]) -> Blosc:
        """Construct a `blosc` codec from a configuration mapping.

        For example `{"cname": "lz4", "clevel": 5, "shuffle": "shuffle",
        "typesize": 4, "blocksize": 0}`.
        """
