from typing import Any

from zarrista.codec._bytes_to_bytes import BytesToBytesCodec

class Crc32c(BytesToBytesCodec):
    """The `crc32c` bytes-to-bytes codec."""

    def __init__(self) -> None:
        """Construct a `crc32c` codec.

        Appends a CRC32C checksum to the encoded bytestream.
        """
    @staticmethod
    def from_config(config: dict[str, Any]) -> Crc32c:
        """Construct a `crc32c` codec from a configuration mapping, e.g. `{}`."""
