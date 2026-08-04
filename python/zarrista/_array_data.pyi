import sys
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import DTypeLike, NDArray

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
    read-only view. Therefore it works with `memoryview(tensor)` and
    `np.asarray(tensor)` for any type that the buffer protocol supports.

    Use `to_numpy` to get a NumPy array view over the Rust memory. This is
    zero-copy whenever possible.
    """

    @property
    def shape(self) -> list[int]:
        """The shape of the decoded region."""
    @property
    def dtype(self) -> DataType:
        """The Zarr data type."""
    def buffer(self) -> Buffer:
        """Return the raw decoded bytes as a zero-copy buffer-protocol object.

        Returns:
            A read-only view over the decoded bytes.
        """
    def to_numpy(self) -> NDArray[Any]:
        """Return a NumPy array view over Rust memory.

        This is a zero-copy view through `np.frombuffer`. Unlike the buffer
        protocol, this path covers the full NumPy dtype set (e.g. complex).

        Returns:
            A read-only NumPy array with the same shape and dtype as this
            tensor.
        """
    def __buffer__(self, flags: int, /) -> memoryview:
        """Export an N-dimensional, typed, read-only PEP 3118 buffer view.

        Args:
            flags: The buffer request flags, as in `inspect.BufferFlags`.

        Returns:
            A read-only memoryview over the decoded bytes.

        Raises:
            BufferError: If `flags` request a writable buffer, or if the data
                type has no standard format code.
        """
    def __array__(
        self,
        dtype: DTypeLike | None = None,
        copy: bool | None = None,
    ) -> NDArray[Any]:
        """Return a NumPy array, for `np.asarray` and `np.array`.

        Args:
            dtype: The data type of the result. Give `None` to keep the tensor's
                own data type.
            copy: Whether to copy the data. Give `None` to let NumPy decide.

        Returns:
            A NumPy array with the same shape as this tensor.
        """
    def __dlpack__(
        self,
        *,
        stream: int | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> CapsuleType:
        """Export the data as a DLPack capsule (e.g. for `np.from_dlpack`).

        Keyword Args:
            stream: The stream to synchronize with. The data is always on the
                CPU, so this argument has no effect.
            max_version: The highest DLPack version that the caller supports.
            dl_device: The device that the caller wants the data on.
            copy: Whether to copy the data.

        Returns:
            A capsule that holds the DLPack tensor.
        """  # noqa: DOC101, DOC103
    def __dlpack_device__(self) -> tuple[int, int]:
        """Return the DLPack device `(device_type, device_id)`. Always CPU.

        Returns:
            The pair `(1, 0)`, which is DLPack's identifier for the CPU.
        """

class VariableArray:
    """Variable-length decoded data (e.g. strings or bytes).

    The class exposes the Arrow PyCapsule interface. Therefore variable-length
    string or bytes data moves without a copy.
    """

    @property
    def shape(self) -> list[int]:
        """The shape of the decoded region."""
    @property
    def dtype(self) -> DataType:
        """The Zarr data type."""
    def __arrow_c_schema__(self) -> CapsuleType:
        """Export the Arrow schema as a PyCapsule (Arrow C Data Interface).

        Returns:
            A capsule that holds the `ArrowSchema`.
        """
    def __arrow_c_array__(
        self,
        requested_schema: object | None = None,
    ) -> tuple[CapsuleType, CapsuleType]:
        """Export as an Arrow array: a `(schema_capsule, array_capsule)` pair.

        Args:
            requested_schema: A capsule that holds the `ArrowSchema` that the
                caller wants. Give `None` to accept this array's own schema.

        Returns:
            The pair `(schema_capsule, array_capsule)`.

        Raises:
            TypeError: If `requested_schema` is neither `None` nor a capsule.
        """

class MaskedTensor:
    """Fixed-width decoded data with a validity mask.

    Use `to_numpy` (or `np.asarray`/`np.array`) to get a `numpy.ma.MaskedArray`
    view over the underlying Rust memory.
    """

    @property
    def shape(self) -> list[int]:
        """The shape of the decoded region."""
    @property
    def dtype(self) -> DataType:
        """The Zarr data type."""
    @property
    def data(self) -> Tensor:
        """The values, without the mask applied."""
    @property
    def mask(self) -> Tensor:
        """The validity mask (`bool`, `True` = valid/present)."""
    def to_numpy(self) -> np.ma.MaskedArray:
        """Return a `numpy.ma.MaskedArray` view over Rust memory.

        NumPy's masked-array convention is the inverse of ours: `True` marks a
        *masked* (missing) element. Therefore this method negates the validity
        mask.

        Returns:
            A masked array with the same shape and dtype as this tensor.
        """
    def __array__(
        self,
        dtype: DTypeLike | None = None,
        copy: bool | None = None,
    ) -> np.ma.MaskedArray:
        """Return a masked array, for `np.asarray` and `np.array`.

        Args:
            dtype: The data type of the result. Give `None` to keep the
                tensor's own data type.
            copy: Whether to copy the data. Give `None` to let NumPy decide.

        Returns:
            A masked array with the same shape as this tensor.
        """

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

ArrayData: TypeAlias = Tensor | VariableArray | MaskedTensor | MaskedVariableArray
"""In-memory array data: one of the four layouts a read can return.

The layout depends on the byte layout of the data type. A data type is either
fixed-width or variable-length, and it either carries a validity mask or does
not. Use `isinstance` to narrow to a concrete type before you use a method that
belongs to one layout.
"""
