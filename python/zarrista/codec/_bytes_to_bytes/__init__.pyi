from zarr_metadata import JSONValue

# Imported only so that the `Raises:` sections below link to the exception docs.
from zarrista.exceptions import PluginCreateError  # noqa: F401

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
        """Construct a codec from its Zarr v3 metadata.

        Args:
            metadata: The Zarr v3 metadata of the codec, for example
                `{"name": "gzip", "configuration": {"level": 5}}`.

        Returns:
            The new codec.

        Raises:
            PluginCreateError: If the metadata names an unsupported codec, or
                if the configuration is not valid for that codec.
        """
    def encode(self, decoded_value: bytes) -> bytes:
        """Encode chunk bytes with this codec.

        Args:
            decoded_value: The decoded chunk bytes.

        Returns:
            The encoded chunk bytes.
        """
