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
        """Construct a codec from its Zarr v3 metadata.

        Args:
            metadata: The Zarr v3 metadata of the codec, for example
                `{"name": "transpose", "configuration": {"order": [1, 0]}}`.

        Returns:
            The new codec.

        Raises:
            PluginCreateError: If the metadata names an unsupported codec, or
                if the configuration is not valid for that codec.
        """
    def encoded_data_type(self, decoded_data_type: DataType) -> DataType:
        """Return the data type that this codec produces when it encodes.

        Args:
            decoded_data_type: The data type of the decoded chunk.

        Returns:
            The data type of the encoded chunk.
        """
    def encoded_fill_value(
        self,
        decoded_data_type: DataType,
        decoded_fill_value: FillValue,
    ) -> FillValue:
        """Return the fill value that this codec produces when it encodes.

        Args:
            decoded_data_type: The data type of the decoded chunk.
            decoded_fill_value: The fill value of the decoded chunk.

        Returns:
            The fill value of the encoded chunk.
        """
    def encode(
        self,
        bytes: ArrayBytes,
        shape: list[int],
        data_type: DataType,
        fill_value: FillValue,
    ) -> ArrayBytes:
        """Encode chunk bytes with this codec.

        Args:
            bytes: The decoded chunk bytes.
            shape: The shape of the decoded chunk, in elements along each
                dimension.
            data_type: The data type of the decoded chunk.
            fill_value: The fill value of the decoded chunk.

        Returns:
            The encoded chunk bytes.

        Raises:
            CodecError: If `bytes` does not agree with `shape` and `data_type`,
                or if the codec cannot encode the chunk.
        """
    def decode(
        self,
        bytes: ArrayBytes,
        shape: list[int],
        data_type: DataType,
        fill_value: FillValue,
    ) -> ArrayBytes:
        """Decode chunk bytes with this codec.

        Args:
            bytes: The encoded chunk bytes.
            shape: The shape of the encoded chunk, in elements along each
                dimension.
            data_type: The data type of the encoded chunk.
            fill_value: The fill value of the encoded chunk.

        Returns:
            The decoded chunk bytes.

        Raises:
            CodecError: If `bytes` does not agree with `shape` and `data_type`,
                or if the codec cannot decode the chunk.
        """
    def encoded_shape(self, decoded_shape: list[int]) -> list[int]:
        """Return the chunk shape that this codec produces when it encodes.

        Args:
            decoded_shape: The shape of the decoded chunk, in elements along
                each dimension.

        Returns:
            The shape of the encoded chunk.

        Raises:
            CodecError: If `decoded_shape` has a number of dimensions that the
                codec does not support.
        """
    def decoded_shape(self, encoded_shape: list[int]) -> list[int] | None:
        """Return the chunk shape that decodes to `encoded_shape`.

        Args:
            encoded_shape: The shape of the encoded chunk, in elements along
                each dimension.

        Returns:
            The shape of the decoded chunk, or `None` if the codec cannot
            determine it.
        """

def transpose(order: list[int]) -> ArrayToArrayCodec:
    """Construct a transpose codec with the given axis order.

    Args:
        order: The new order of the axes. This must be a permutation of the
            axis indices, from 0 to one less than the number of dimensions.

    Returns:
        The new codec.

    Raises:
        TransposeOrderError: If `order` is not a permutation of the axis
            indices.
    """

def bitround(keepbits: int) -> ArrayToArrayCodec:
    """Construct a bit-rounding codec that keeps `keepbits` mantissa bits.

    Args:
        keepbits: The number of mantissa bits to keep.

    Returns:
        The new codec.

    Raises:
        OverflowError: If `keepbits` is negative.
    """
