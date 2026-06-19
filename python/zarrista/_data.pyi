import sys
from typing import Any, TypeAlias

from numpy.typing import NDArray

from ._dtype import DataType

if sys.version_info >= (3, 12):
    from collections.abc import Buffer
else:
    from typing_extensions import Buffer

class Tensor:
    """Fixed-width, dense decoded array data.

    The decoded bytes are held zero-copy. Reinterpret them as a NumPy array with
    `to_numpy()`, or get the raw bytes as a buffer-protocol object via `buffer()`.
    """

    @property
    def shape(self) -> list[int]:
        """The shape of the decoded region."""
    @property
    def dtype(self) -> DataType:
        """The Zarr data type."""
    def buffer(self) -> Buffer:
        """The raw decoded bytes as a zero-copy buffer-protocol object."""
    def to_numpy(self) -> NDArray[Any]:
        """Reinterpret the raw bytes as a NumPy array of this dtype and shape.

        A zero-copy view via `np.frombuffer`; NumPy tolerates the (possibly
        unaligned) buffer.
        """

class VariableArray:
    """Variable-length decoded data (e.g. strings or bytes).

    Not yet exposed to NumPy.
    """

    @property
    def shape(self) -> list[int]:
        """The shape of the decoded region."""
    @property
    def dtype(self) -> DataType:
        """The Zarr data type."""
    def to_numpy(self) -> NDArray[Any]:
        """Not yet implemented: raises `NotImplementedError`."""

class MaskedTensor:
    """Fixed-width decoded data with a validity mask.

    Not yet exposed to NumPy.
    """

    @property
    def shape(self) -> list[int]:
        """The shape of the decoded region."""
    @property
    def dtype(self) -> DataType:
        """The Zarr data type."""
    def to_numpy(self) -> NDArray[Any]:
        """Not yet implemented: raises `NotImplementedError`."""

class MaskedVariableArray:
    """Variable-length decoded data with a validity mask.

    Not yet exposed to NumPy.
    """

    @property
    def shape(self) -> list[int]:
        """The shape of the decoded region."""
    @property
    def dtype(self) -> DataType:
        """The Zarr data type."""
    def to_numpy(self) -> NDArray[Any]:
        """Not yet implemented: raises `NotImplementedError`."""

DataArray: TypeAlias = Tensor | VariableArray | MaskedTensor | MaskedVariableArray
"""The result of a read: one of the four decoded array layouts.

Which one is returned depends on the dtype's byte layout (fixed vs. variable, and
whether it carries a validity mask). Use `isinstance` to narrow to a concrete
type before using layout-specific methods.
"""
