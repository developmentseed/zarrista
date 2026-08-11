from typing import Literal

from zarr_metadata import ZarrV3NamedConfigJSON

# Imported only so that the `Raises:` sections below link to the exception docs.
from zarrista.exceptions import PluginCreateError  # noqa: F401

class ChunkKeyEncoding:
    """How an array maps chunk grid indices to store keys."""

    @staticmethod
    def default(sep: Literal[".", "/"]) -> ChunkKeyEncoding:
        """Construct the `default` chunk key encoding with the given separator.

        Args:
            sep: The separator between the parts of a chunk key.

        Returns:
            The new chunk key encoding.

        Raises:
            ValueError: If `sep` is not `"."` or `"/"`.
        """
    @staticmethod
    def from_metadata(metadata: ZarrV3NamedConfigJSON) -> ChunkKeyEncoding:
        """Construct a chunk key encoding from its Zarr v3 metadata.

        Args:
            metadata: The Zarr v3 metadata of the chunk key encoding.

        Returns:
            The new chunk key encoding.

        Raises:
            PluginCreateError: If the metadata names an unsupported chunk key
                encoding, or if the configuration is not valid for it.
        """
    @property
    def metadata(self) -> ZarrV3NamedConfigJSON:
        """The chunk key encoding's Zarr v3 metadata."""
    @property
    def name(self) -> str | None:
        """The chunk key encoding's Zarr v3 name (e.g. `"default"`), if any."""
