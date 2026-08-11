from collections.abc import Sequence
from typing import TypeAlias

from zarr_metadata import ZarrV3NamedConfigJSON

# Imported only so that the `Raises:` sections below link to the exception docs.
from zarrista.exceptions import (  # noqa: F401
    ChunkGridCreateError,
    IncompatibleDimensionalityError,
    PluginCreateError,
)

_RunLength: TypeAlias = int | tuple[int, int]
"""One run of chunks along a rectilinear chunk edge.

A single chunk size, or a `(size, count)` pair. The pair means `count`
adjacent chunks that each have the given size.
"""

_ChunkEdgeLengths: TypeAlias = int | Sequence[_RunLength]
"""The chunk sizes along one dimension.

An integer if the dimension has one chunk size, or a sequence of runs if the
chunk sizes change along the dimension.
"""

class ChunkGrid:
    """The chunk grid of an array.

    The chunk grid shows how the array shape divides into chunks.
    """

    @staticmethod
    def regular(array_shape: Sequence[int], chunk_shape: Sequence[int]) -> ChunkGrid:
        """Construct a regular grid with a fixed chunk shape.

        Args:
            array_shape: The shape of the array, in elements along each dimension.
            chunk_shape: The shape of each chunk, in elements along each dimension.

        Returns:
            The new chunk grid.

        Raises:
            ChunkGridCreateError: If `chunk_shape` is not compatible with
                `array_shape`.
            ValueError: If an element of `chunk_shape` is zero.
        """
    @staticmethod
    def rectilinear(
        array_shape: Sequence[int],
        chunk_shapes: Sequence[_ChunkEdgeLengths],
    ) -> ChunkGrid:
        """Construct a rectilinear grid with different chunk sizes along each dimension.

        Args:
            array_shape: The shape of the array, in elements along each dimension.
            chunk_shapes: The chunk sizes along each dimension.

        Returns:
            The new chunk grid.

        Raises:
            ChunkGridCreateError: If `chunk_shapes` is not compatible with
                `array_shape`.
            ValueError: If a chunk size is zero.
        """
    @staticmethod
    def regular_bounded(
        array_shape: Sequence[int],
        chunk_shape: Sequence[int],
    ) -> ChunkGrid:
        """Construct a regular grid that clips the last chunks to the array bounds.

        This chunk grid is experimental. Other Zarr V3 implementations can be
        incompatible with it.

        Args:
            array_shape: The shape of the array, in elements along each dimension.
            chunk_shape: The maximum shape of a chunk, in elements along each
                dimension. The grid clips the chunks at the array bounds to a
                smaller shape.

        Returns:
            The new chunk grid.

        Raises:
            IncompatibleDimensionalityError: If `chunk_shape` and `array_shape`
                have a different number of dimensions.
            ValueError: If an element of `chunk_shape` is zero.
        """
    @staticmethod
    def from_metadata(
        metadata: ZarrV3NamedConfigJSON,
        shape: Sequence[int],
    ) -> ChunkGrid:
        """Construct a chunk grid from its Zarr v3 metadata and the array shape.

        Args:
            metadata: The Zarr v3 metadata of the chunk grid.
            shape: The shape of the array, in elements along each dimension.

        Returns:
            The new chunk grid.

        Raises:
            PluginCreateError: If the metadata names an unsupported chunk grid,
                or if the metadata is not compatible with `shape`.
        """
    @property
    def metadata(self) -> ZarrV3NamedConfigJSON:
        """The chunk grid's Zarr v3 metadata."""
    @property
    def ndim(self) -> int:
        """The number of dimensions."""
    @property
    def array_shape(self) -> list[int]:
        """The shape of the array, in elements along each dimension."""
    @property
    def grid_shape(self) -> list[int]:
        """The shape of the grid, in number of chunks along each dimension."""
