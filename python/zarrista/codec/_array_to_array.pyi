from zarr_metadata import JSONValue

from zarrista._array_bytes import ArrayBytes
from zarrista._dtype import DataType
from zarrista._fill_value import FillValue

class ArrayToArrayCodec:
    """A Zarr v3 array-to-array codec."""

    @property
    def name(self) -> str | None:
        """The codec's Zarr v3 name (e.g. `"transpose"`), if any."""
    @property
    def config(self) -> JSONValue | None:
        """The codec's Zarr v3 configuration as a dict, if any."""
    @staticmethod
    def from_config(metadata: JSONValue) -> ArrayToArrayCodec:
        """Build a codec from its Zarr v3 metadata.

        For example `{"name": "transpose", "configuration": {"order": [1, 0]}}`.
        """
    def encoded_data_type(self, decoded_data_type: DataType) -> DataType:
        """Return the data type produced by encoding `decoded_data_type`."""
    def encoded_fill_value(
        self,
        decoded_data_type: DataType,
        decoded_fill_value: FillValue,
    ) -> FillValue:
        """Return the fill value produced by encoding `decoded_fill_value`."""
    def encode(
        self,
        bytes: ArrayBytes,
        shape: list[int],
        data_type: DataType,
        fill_value: FillValue,
    ) -> ArrayBytes:
        """Encode chunk bytes for this codec."""
    def decode(
        self,
        bytes: ArrayBytes,
        shape: list[int],
        data_type: DataType,
        fill_value: FillValue,
    ) -> ArrayBytes:
        """Decode chunk bytes for this codec."""
    def encoded_shape(self, decoded_shape: list[int]) -> list[int]:
        """Return the chunk shape produced by encoding `decoded_shape`."""
    def decoded_shape(self, encoded_shape: list[int]) -> list[int] | None:
        """Return the chunk shape that decodes to `encoded_shape`, if determinable."""

def transpose(order: list[int]) -> ArrayToArrayCodec:
    """Construct a transpose codec with the given axis order."""

def bitround(keepbits: int) -> ArrayToArrayCodec:
    """Construct a bit-rounding codec keeping `keepbits` mantissa bits."""
