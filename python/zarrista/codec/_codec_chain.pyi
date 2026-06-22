from zarr_metadata import MetadataV3, NamedConfigV3

class CodecChain:
    """The ordered chain of codecs used to encode and decode an array's chunks."""

    def __init__(self, metadatas: list[MetadataV3]) -> None:
        """Construct a codec chain from a list of Zarr v3 codec metadata."""
    def create_metadatas(self) -> list[NamedConfigV3]:
        """Return the Zarr v3 metadata for each codec in the chain."""
