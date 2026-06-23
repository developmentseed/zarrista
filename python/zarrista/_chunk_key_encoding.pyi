from typing import Literal

from zarr_metadata import NamedConfigV3

class ChunkKeyEncoding:
    """How an array maps chunk grid indices to store keys."""

    @staticmethod
    def default(sep: Literal[".", "/"]) -> ChunkKeyEncoding:
        """The `default` chunk key encoding with the given separator."""
    @staticmethod
    def from_metadata(metadata: NamedConfigV3) -> ChunkKeyEncoding:
        """Build a chunk key encoding from its Zarr v3 metadata."""
    @property
    def metadata(self) -> NamedConfigV3:
        """The chunk key encoding's Zarr v3 metadata."""
    @property
    def name(self) -> str | None:
        """The chunk key encoding's Zarr v3 name (e.g. `"default"`), if any."""
