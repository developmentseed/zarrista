from typing import Any

class CodecChain:
    """The ordered chain of codecs used to encode and decode an array's chunks."""

    def __init__(self, metadatas: list[dict[str, Any]]) -> None:
        """Construct a codec chain from a list of Zarr v3 codec metadata."""
    def create_metadatas(self) -> list[dict[str, Any]]:
        """Return the Zarr v3 metadata for each codec in the chain."""
