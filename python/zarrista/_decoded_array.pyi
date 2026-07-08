import sys
from typing import Any, TypeAlias

from numpy.typing import NDArray

from ._dtype import DataType

if sys.version_info >= (3, 12):
    from collections.abc import Buffer
else:
    from typing_extensions import Buffer

if sys.version_info >= (3, 13):
    from types import CapsuleType
else:
    from typing_extensions import CapsuleType

class Tensor:
    """Fixed-width, dense decoded array data.

    `Tensor` implements the buffer protocol directly as an N-dimensional, typed,
    read-only view, so for any type supported by the buffer protocol it works with
    `memoryview(tensor)` and `np.asarray(tensor)`.

    Or, use `to_numpy` to get a NumPy array view over the Rust memory. This is
    zero-copy whenever possible.
    """

    @property
    def shape(self) -> list[int]:
        """The shape of the decoded region."""
    @property
    def dtype(self) -> DataType:
        """The Zarr data type."""
    def buffer(self) -> Buffer:
        """Return the raw decoded bytes as a zero-copy buffer-protocol object."""
    def to_numpy(self) -> NDArray[Any]:
        """Access a NumPy array view over Rust memory.

        This is a zero-copy view via `np.frombuffer`. Unlike the buffer protocol,
        this path covers the full NumPy dtype set (e.g. complex).
        """
    def __buffer__(self, flags: int, /) -> memoryview:
        """Export an N-dimensional, typed, read-only PEP 3118 buffer view.

        Raises `BufferError` if a writable buffer is requested, or if the dtype has no
        standard format code.
        """
    def __dlpack__(
        self,
        *,
        stream: int | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> CapsuleType:
        """Export the data as a DLPack capsule (e.g. for `np.from_dlpack`)."""
    def __dlpack_device__(self) -> tuple[int, int]:
        """Return the DLPack device `(device_type, device_id)`. Always CPU."""

class VariableArray:
    """Variable-length decoded data (e.g. strings or bytes).

    Exposes the Arrow PyCapsule interface for zero-copy data exchange for variable
    length string or bytes data.
    """

    @property
    def shape(self) -> list[int]:
        """The shape of the decoded region."""
    @property
    def dtype(self) -> DataType:
        """The Zarr data type."""
    def __arrow_c_schema__(self) -> CapsuleType:
        """Export the Arrow schema as a PyCapsule (Arrow C Data Interface)."""
    def __arrow_c_array__(
        self,
        requested_schema: object | None = None,
    ) -> tuple[CapsuleType, CapsuleType]:
        """Export as an Arrow array: a `(schema_capsule, array_capsule)` pair."""

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

DecodedArray: TypeAlias = Tensor | VariableArray | MaskedTensor | MaskedVariableArray
"""The result of a read: one of the four decoded array layouts.

Which one is returned depends on the dtype's byte layout (fixed vs. variable, and
whether it carries a validity mask). Use `isinstance` to narrow to a concrete
type before using layout-specific methods.
"""
