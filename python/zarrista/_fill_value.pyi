from collections.abc import Buffer
from typing import SupportsFloat, SupportsIndex, TypeAlias

from zarr_metadata import (
    BoolFillValue,
    BytesFillValue,
    Complex64FillValue,
    Complex128FillValue,
    Float64FillValue,
    Int64FillValue,
    NumpyDatetime64FillValue,
    RawBytesFillValue,
    StringFillValue,
)

from zarrista._dtype import DataType, DataTypeInput

# Imported only so that the `Raises:` sections below link to the exception docs.
from zarrista.exceptions import FillValueError  # noqa: F401

FillValueJSON: TypeAlias = (
    BoolFillValue
    | Int64FillValue
    | Float64FillValue
    | Complex64FillValue
    | Complex128FillValue
    | StringFillValue
    | BytesFillValue
    | RawBytesFillValue
    | NumpyDatetime64FillValue
)
"""The Zarr v3 metadata form of a fill value.

This alias comes from the per-dtype fill value types in `zarr_metadata`.
Therefore it stays in agreement with the spec, and nobody must maintain the list
here.
"""

FillValueInput: TypeAlias = (
    FillValue | FillValueJSON | complex | SupportsFloat | SupportsIndex
)
"""A value that a `FillValue` can be constructed from.

This is a Python value of the data type, such as `0`, `1.5`, `True`, `"NaN"`,
`1 + 2j` or `"missing"`. An integer data type reads `__index__`, and a float
data type reads `__float__`. Therefore a NumPy scalar such as
`numpy.float32(1.5)`, and any other object that acts like a number, also works.
It is also a `FillValue` that already has the same data type.
"""

class FillValue:
    """An array's fill value, together with the data type that it belongs to.

    Two fill values of different data types are never equal, even when they hold the
    same bytes.

    Examples:
        Construct a fill value from a Python value and a data type name:

        ```py
        from zarrista import FillValue

        fill_value = FillValue(-9999, dtype="int32")
        fill_value.metadata  # -9999
        ```

        A float data type also takes the Zarr v3 names of the non-finite
        values:

        ```py
        FillValue("NaN", dtype="float32")
        FillValue(float("nan"), dtype="float32")  # the same fill value
        ```
    """

    def __init__(self, value: FillValueInput, /, dtype: DataTypeInput) -> None:
        """Construct a fill value from a Python value and its data type.

        Args:
            value: One fill value element, as a Python value of `dtype`.
            dtype: The data type that `value` belongs to.

        Raises:
            OverflowError: If `value` is outside the range of `dtype`.
            TypeError: If `value` is not a value of `dtype`.
            FillValueError: If `value` is a `FillValue` of another data type, or
                if `dtype` cannot use `value` as a fill value.
        """
    @property
    def dtype(self) -> DataType:
        """The data type that gives these bytes their meaning."""
    @property
    def size(self) -> int:
        """The size of the fill value in bytes."""
    @property
    def metadata(self) -> FillValueJSON:
        """The Zarr v3 metadata form of the fill value.

        This is what the array metadata stores, such as `-9999`, `"NaN"`, or
        `[1.0, 2.0]` for a complex data type.

        Raises:
            FillValueError: If the data type cannot describe this fill value.
        """
    def as_bytes(self) -> bytes:
        """Return the fill value as native-endian bytes.

        Returns:
            The native-endian bytes of one fill value element.
        """
    def equals_all(self, other: Buffer, /) -> bool:
        """Return whether `other` contains only repetitions of this fill value.

        Args:
            other: The bytes to compare against this fill value.

        Returns:
            `True` if `other` is a whole number of copies of this fill value.
                `False` if the bytes differ, or if the length of `other` is not a
                multiple of `size`.
        """
    def __eq__(self, other: object, /) -> bool: ...
