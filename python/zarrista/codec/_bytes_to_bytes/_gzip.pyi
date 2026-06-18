from typing import Any

from zarrista.codec._bytes_to_bytes import BytesToBytesCodec

class Gzip(BytesToBytesCodec):
    """The `gzip` bytes-to-bytes codec."""

    def __init__(self, level: int) -> None:
        """Construct a `gzip` codec.

        `level` is the compression level, an integer from 0 (no compression)
        to 9 (most compression).
        """
    @staticmethod
    def from_config(config: dict[str, Any]) -> Gzip:
        """Construct a `gzip` codec from a config mapping, e.g. `{"level": 5}`."""
