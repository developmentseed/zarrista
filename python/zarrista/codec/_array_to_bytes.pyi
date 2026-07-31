from zarr_metadata import JSONValue

class ArrayToBytesCodec:
    """A Zarr v3 array-to-bytes codec (the "serializer")."""

    @property
    def name(self) -> str | None:
        """The codec's Zarr v3 name (e.g. `"bytes"`, `"sharding_indexed"`), if any."""
    @property
    def config(self) -> JSONValue | None:
        """The codec's Zarr v3 configuration as a dict, if any."""
    @staticmethod
    def from_config(metadata: JSONValue) -> ArrayToBytesCodec:
        """Construct a codec from its Zarr v3 metadata.

        Args:
            metadata: The Zarr v3 metadata of the codec, for example
                `{"name": "bytes", "configuration": {"endian": "little"}}`.

        Returns:
            The new codec.

        Raises:
            PluginCreateError: If the metadata names an unsupported codec, or
                if the configuration is not valid for that codec.
        """
