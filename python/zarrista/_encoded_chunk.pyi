from collections.abc import Buffer
from typing import Unpack

from zarrista.codec import CodecChain, CodecOptions

from ._decoded_array import DecodedArray
from ._dtype import DataType
from ._fill_value import FillValue
from ._thread_pool import ThreadPool

class EncodedChunk:
    """Encoded chunk bytes, together with the codec chain that decodes them.

    This class splits a chunk read into two steps. The first step reads the
    encoded bytes, which is IO-bound. The second step decodes them, which is
    CPU-bound. You can then run each step where it is best: read the bytes with
    `await`, and decode them on a thread pool.

    [`Array.retrieve_encoded_chunk`][zarrista.Array.retrieve_encoded_chunk]
    returns an `EncodedChunk`. The object holds the data type, the fill value,
    the shape, and the codec chain, so you do not track which codec chain
    belongs to which bytes.

    The object holds no reference to the array or to the store. It is therefore
    safe to send to another thread.

    Examples:
        Read a chunk, then decode it off the main thread:

        ```py
        chunk = array.retrieve_encoded_chunk([0, 0])
        if chunk is not None:
            decoded = await chunk.decode_async()
        ```
    """

    @property
    def buffer(self) -> Buffer:
        """The raw, still-encoded chunk bytes."""
    @property
    def codecs(self) -> CodecChain:
        """The codec chain that decodes the bytes."""
    @property
    def data_type(self) -> DataType:
        """The Zarr data type of the decoded chunk."""
    @property
    def fill_value(self) -> FillValue:
        """The fill value of the decoded chunk."""
    @property
    def shape(self) -> list[int]:
        """The shape of the decoded chunk, in elements along each dimension."""
    def decode(self, **codec_options: Unpack[CodecOptions]) -> DecodedArray:
        """Decode the chunk bytes on the calling thread.

        The method releases the GIL while it decodes. Other Python threads can
        therefore run at the same time.

        Args:
            **codec_options: The codec options, as
                [`CodecOptions`][zarrista.codec.CodecOptions].

        Returns:
            The decoded chunk data.

        Raises:
            TypeError: If a keyword argument is not a known codec option.
        """
    async def decode_async(
        self,
        *,
        pool: ThreadPool | None = None,
        **codec_options: Unpack[CodecOptions],
    ) -> DecodedArray:
        """Decode the chunk bytes on a Rust thread pool.

        The method does the work on a thread pool and does not hold the GIL.
        Use it to decode many chunks at the same time.

        Every decode uses the full CPU by default, because `concurrent_target`
        defaults to the number of threads in the pool. To decode N chunks at
        the same time, pass `concurrent_target=max(1, cores // N)`.

        Args:
            pool: The thread pool that runs the decode. Defaults to the global
                Rust thread pool, which zarrista also uses for its other
                parallel work. Pass a [`ThreadPool`][zarrista.ThreadPool] to
                use a separate pool.
            **codec_options: The codec options, as
                [`CodecOptions`][zarrista.codec.CodecOptions].

        Returns:
            The decoded chunk data.

        Raises:
            TypeError: If a keyword argument is not a known codec option.
        """
