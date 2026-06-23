from collections.abc import Sequence
from typing import TypeAlias

from zarr_metadata import NamedConfigV3

_RunLength: TypeAlias = int | tuple[int, int]
"""One run of a rectilinear chunk edge: a single chunk size, or a
`(size, count)` pair meaning `count` consecutive chunks of `size`."""

_ChunkEdgeLengths: TypeAlias = int | Sequence[_RunLength]
"""Chunk sizes along one dimension: a scalar (regular along that axis) or a
sequence of runs (varying sizes)."""

class ChunkGrid:
    """The chunk grid of an array: how its shape is partitioned into chunks."""

    @staticmethod
    def regular(array_shape: Sequence[int], chunk_shape: Sequence[int]) -> ChunkGrid:
        """Construct a regular grid with a fixed `chunk_shape` over `array_shape`."""
    @staticmethod
    def rectilinear(
        array_shape: Sequence[int],
        chunk_shapes: Sequence[_ChunkEdgeLengths],
    ) -> ChunkGrid:
        """Construct a rectilinear grid with per-dimension chunk sizes."""
    @staticmethod
    def regular_bounded(
        array_shape: Sequence[int],
        chunk_shape: Sequence[int],
    ) -> ChunkGrid:
        """Construct a regular grid whose final chunks are clipped to the array bounds.

        Experimental and may be incompatible with other Zarr V3 implementations.
        """
    @staticmethod
    def from_metadata(metadata: NamedConfigV3, shape: Sequence[int]) -> ChunkGrid:
        """Build a chunk grid from its Zarr v3 metadata and the array shape."""
    @property
    def metadata(self) -> NamedConfigV3:
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
