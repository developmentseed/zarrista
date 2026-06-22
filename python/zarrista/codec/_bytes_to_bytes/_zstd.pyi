from zarr_metadata.v3.codec.zstd import ZstdCodecConfiguration

from zarrista.codec._bytes_to_bytes import BytesToBytesCodec

class Zstd(BytesToBytesCodec):
    """The `zstd` bytes-to-bytes codec."""

    def __init__(self, level: int, checksum: bool) -> None:
        """Construct a `zstd` codec.

        `level` is the compression level. When `checksum` is true, a checksum
        is written to (and verified on decode from) the encoded bytestream.
        """
    @staticmethod
    def from_config(config: ZstdCodecConfiguration) -> Zstd:
        """Construct a `zstd` codec from a configuration mapping.

        For example, `{"level": 5, "checksum": false}`.
        """
