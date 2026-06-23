from zarr_metadata import JSONValue

class BytesToBytesCodec:
    """A Zarr v3 bytes-to-bytes codec."""

    @property
    def name(self) -> str | None:
        """The codec's Zarr v3 name (e.g. `"blosc"`), if any."""
    @property
    def config(self) -> JSONValue | None:
        """The codec's Zarr v3 configuration as a dict, if any."""
    @staticmethod
    def from_config(metadata: JSONValue) -> BytesToBytesCodec:
        """Build a codec from its Zarr v3 metadata.

        For example `{"name": "gzip", "configuration": {"level": 5}}`.
        """
    def encode(self, decoded_value: bytes) -> bytes:
        """Encode chunk bytes for this codec."""
