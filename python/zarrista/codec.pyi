from typing import Any

from ._dtype import DataType

class ArrayToArrayCodec:
    """A Zarr v3 array-to-array codec."""

    def encoded_data_type(self, decoded_data_type: DataType) -> DataType:
        """The data type produced by encoding ``decoded_data_type``."""
    def encoded_fill_value(
        self, decoded_data_type: DataType, decoded_fill_value: Any
    ) -> Any:
        """The fill value produced by encoding ``decoded_fill_value``."""
    def encode(
        self,
        bytes: Any,
        shape: list[int],
        data_type: DataType,
        fill_value: Any,
    ) -> Any:
        """Encode chunk bytes (an ``ArrayBytes``) for this codec."""
    def decode(
        self,
        bytes: Any,
        shape: list[int],
        data_type: DataType,
        fill_value: Any,
    ) -> Any:
        """Decode chunk bytes (an ``ArrayBytes``) for this codec."""
    def encoded_shape(self, decoded_shape: list[int]) -> list[int]:
        """The chunk shape produced by encoding ``decoded_shape``."""
    def decoded_shape(self, encoded_shape: list[int]) -> list[int] | None:
        """The chunk shape that decodes to ``encoded_shape``, if determinable."""
    def __repr__(self) -> str: ...

class CodecChain:
    """The ordered chain of codecs used to encode and decode an array's chunks."""

    def __init__(self, metadatas: list[dict[str, Any]]) -> None:
        """Construct a codec chain from a list of Zarr v3 codec metadata."""
    def create_metadatas(self) -> list[dict[str, Any]]:
        """The Zarr v3 metadata for each codec in the chain."""

def transpose(order: list[int]) -> ArrayToArrayCodec:
    """Construct a transpose codec with the given axis order."""

def bitround(keepbits: int) -> ArrayToArrayCodec:
    """Construct a bit-rounding codec keeping ``keepbits`` mantissa bits."""
