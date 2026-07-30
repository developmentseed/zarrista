from typing import Literal

from zarr_metadata import ZarrV3NamedConfigJSON

class ChunkKeyEncoding:
    """How an array maps chunk grid indices to store keys."""

    @staticmethod
    def default(sep: Literal[".", "/"]) -> ChunkKeyEncoding:
        """Construct the `default` chunk key encoding with the given separator."""
    @staticmethod
    def from_metadata(metadata: ZarrV3NamedConfigJSON) -> ChunkKeyEncoding:
        """Build a chunk key encoding from its Zarr v3 metadata."""
    @property
    def metadata(self) -> ZarrV3NamedConfigJSON:
        """The chunk key encoding's Zarr v3 metadata."""
    @property
    def name(self) -> str | None:
        """The chunk key encoding's Zarr v3 name (e.g. `"default"`), if any."""
