from zarr_metadata import JSONValue

class ArrayToBytesCodec:
    """A Zarr v3 array-to-bytes codec (the "serializer")."""

    @property
    def name(self) -> str | None:
        """The codec's Zarr v3 name (e.g. `"bytes"`, `"sharding_indexed"`), if any."""
    @property
    def configuration(self) -> JSONValue | None:
        """The codec's Zarr v3 configuration as a dict, if any."""
    @staticmethod
    def from_config(metadata: JSONValue) -> ArrayToBytesCodec:
        """Build a codec from its Zarr v3 metadata.

        For example `{"name": "bytes", "configuration": {"endian": "little"}}`.
        """
