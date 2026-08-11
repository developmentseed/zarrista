from zarrista.codec._array_to_array import ArrayToArrayCodec
from zarrista.codec._array_to_bytes import ArrayToBytesCodec
from zarrista.codec._bytes_to_bytes import BytesToBytesCodec

class CodecChain:
    """A full Zarr v3 codec pipeline.

    A codec chain holds:

    - the array-to-array codecs ("filters")
    - one array-to-bytes codec ("serializer")
    - the bytes-to-bytes codecs ("compressors")

    The chain applies them in that order to encode, and in the reverse order to decode.

    An array carries the codec chain that encodes its chunks. To decode chunk
    bytes with it, use [`EncodedChunk`][zarrista.EncodedChunk], which holds the
    bytes and the chain together.
    """

    @property
    def filters(self) -> list[ArrayToArrayCodec]:
        """The array-to-array codecs ("filters")."""
    @property
    def serializer(self) -> ArrayToBytesCodec:
        """The array-to-bytes codec ("serializer").

        For a sharded array, this is the `sharding_indexed` codec.
        """
    @property
    def compressors(self) -> list[BytesToBytesCodec]:
        """The bytes-to-bytes codecs ("compressors")."""
