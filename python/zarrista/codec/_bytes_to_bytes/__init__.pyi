class BytesToBytesCodec:
    """A Zarr v3 bytes-to-bytes codec."""

    def encode(self, decoded_value: bytes) -> bytes:
        """Encode chunk bytes for this codec."""
