class ChunkGrid:
    """The chunk grid of an array: how its shape is partitioned into chunks."""

    @property
    def ndim(self) -> int:
        """The number of dimensions."""
    @property
    def array_shape(self) -> list[int]:
        """The shape of the array, in elements along each dimension."""
    @property
    def grid_shape(self) -> list[int]:
        """The shape of the grid, in number of chunks along each dimension."""
