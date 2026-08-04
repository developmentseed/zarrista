from collections.abc import Buffer
from types import EllipsisType
from typing import Protocol, TypeAlias, Unpack

from zarr_metadata import JSONValue, ZarrV3ArrayMetadataJSON

from zarrista.codec import CodecChain, CodecOptions

from ._array_bytes import ArrayBytes
from ._array_data import ArrayData
from ._chunk_key_encoding import ChunkKeyEncoding
from ._chunks import ChunkGrid
from ._dtype import DataType
from ._encoded_chunk import EncodedChunk
from ._fill_value import FillValue
from ._store import AsyncStore, FilesystemStore, MemoryStore

_AxisSelector: TypeAlias = int | slice | EllipsisType
Selection: TypeAlias = _AxisSelector | tuple[_AxisSelector, ...]
"""A numpy-style basic-indexing selection: what you would write inside `[]`.

This supports integers, step-1 slices, `Ellipsis`, and tuples of those. A tuple
with fewer entries than `ndim` selects all of the trailing axes. Negative
indices and slice bounds are normalized.

The following are not supported: a slice with a step that is not 1, `None` (also
called `np.newaxis`), boolean indexing, and fancy or array indexing.
"""

class SupportsDLPack(Protocol):
    """An object that exports its data through the DLPack protocol.

    numpy arrays, PyTorch tensors, JAX arrays, and CuPy arrays all satisfy this.

    !!! warning "Not importable at runtime"

        To use this type hint in your code, import it within a `TYPE_CHECKING`
        block.
    """

    def __dlpack__(self, **kwargs: object) -> object: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...

DataInput: TypeAlias = SupportsDLPack | ArrayBytes | Buffer
"""In-memory array-like data that can be written to a Zarr Array/AsyncArray.

Prefer rich types such as numpy arrays or anything that supports the DLPack interface.
These allow richer validation. Passing a plain buffer will skip any data type or shape
validation.

!!! warning "Not importable at runtime"

    To use this type hint in your code, import it within a `TYPE_CHECKING`
    block.
"""

class Array:
    """A Zarr array.

    The array's data stays in the store, which is usually not in memory. A read
    fetches only the region that you ask for, decodes it, and returns it as
    [`ArrayData`][zarrista.ArrayData].
    """

    @staticmethod
    def open(store: FilesystemStore | MemoryStore, path: str = "/") -> Array:
        """Open the array stored at `path` in `store`.

        Args:
            store: The store that holds the array.
            path: The absolute path of the array in the store.

        Returns:
            The array at `path`.

        Raises:
            ArrayCreateError: If the store holds no array metadata at `path`.
            ValueError: If `path` is not a valid absolute node path.
        """
    @staticmethod
    def from_metadata(
        metadata: ZarrV3ArrayMetadataJSON,
        store: FilesystemStore | MemoryStore,
        path: str = "/",
    ) -> Array:
        """Use the provided metadata to open a new array at `path` in `store`.

        This does **not** write the metadata to the store. Use
        [`Array.store_metadata`][zarrista.Array.store_metadata] to write it.

        Args:
            metadata: The Zarr v3 metadata of the array.
            store: The store to attach the array to.
            path: The absolute path of the array in the store.

        Returns:
            The new array.

        Raises:
            ArrayCreateError: If the metadata is not valid.
            ValueError: If `path` is not a valid absolute node path.
        """
    @property
    def attrs(self) -> dict[str, JSONValue]:
        """The array's user attributes as a dict."""
    @property
    def chunk_grid(self) -> ChunkGrid:
        """The chunk grid of the array."""
    @property
    def chunk_grid_shape(self) -> list[int]:
        """The shape of the chunk grid (i.e. the number of chunks per dimension)."""
    def chunk_key(self, chunk_indices: list[int]) -> str:
        """Return the store key of the chunk at `chunk_indices`.

        This does not check `chunk_indices` against the chunk grid. It gives a
        key for indices that are outside the grid.

        Args:
            chunk_indices: The position of the chunk in the chunk grid.

        Returns:
            The store key of the chunk.
        """
    @property
    def chunk_key_encoding(self) -> ChunkKeyEncoding:
        """The chunk key encoding, mapping chunk grid indices to store keys."""
    def chunk_origin(self, chunk_indices: list[int]) -> list[int]:
        """Return the origin of the chunk at `chunk_indices`.

        Args:
            chunk_indices: The position of the chunk in the chunk grid.

        Returns:
            The index of the first array element in the chunk.

        Raises:
            ArrayError: If `chunk_indices` has a different number of dimensions
                from the chunk grid.
        """
    def chunk_shape(self, chunk_indices: list[int]) -> list[int]:
        """Return the shape of the chunk at `chunk_indices`.

        Args:
            chunk_indices: The position of the chunk in the chunk grid.

        Returns:
            The shape of the chunk, in elements along each dimension.

        Raises:
            ArrayError: If `chunk_indices` has a different number of dimensions
                from the chunk grid.
        """
    def chunk_subset(self, chunk_indices: list[int]) -> tuple[slice, ...]:
        """Return the array subset spanned by the chunk at `chunk_indices`.

        Args:
            chunk_indices: The position of the chunk in the chunk grid.

        Returns:
            One slice per dimension, which together give the array subset of
            the chunk.

        Raises:
            ArrayError: If `chunk_indices` has a different number of dimensions
                from the chunk grid.
        """
    @property
    def codecs(self) -> CodecChain:
        """The codec chain that encodes and decodes the array's chunks.

        Use its `filters`, `serializer`, and `compressors` properties to inspect
        the individual codecs.
        """
    @property
    def dimension_names(self) -> list[str | None] | None:
        """The dimension names, if any were specified."""
    @property
    def dtype(self) -> DataType:
        """The Zarr data type."""
    @property
    def fill_value(self) -> FillValue:
        """The array's fill value."""
    @property
    def metadata(self) -> ZarrV3ArrayMetadataJSON:
        """The array's full Zarr v3 metadata."""
    @property
    def ndim(self) -> int:
        """The number of dimensions."""
    @property
    def path(self) -> str:
        """The array's path in the store."""
    def retrieve_array_subset(
        self,
        selection: Selection,
        **codec_options: Unpack[CodecOptions],
    ) -> ArrayData:
        """Read and decode an array region selected with numpy-style basic indexing.

        The result keeps the number of dimensions, which agrees with a zarrs
        `ArraySubset`. An integer selects a range of length 1, and the method
        keeps that axis.

        Args:
            selection: The region to read, in element coordinates.
            **codec_options: The codec options, as
                [`CodecOptions`][zarrista.codec.CodecOptions].

        Returns:
            The decoded data for the region.

        Raises:
            NotImplementedError: If `selection` uses a slice with a step that is
                not 1, or if it uses `None` (`np.newaxis`).
            IndexError: If `selection` has more entries than the array has
                dimensions.
            TypeError: If a keyword argument is not a known codec option.
        """
    def retrieve_chunk(
        self,
        chunk_indices: list[int],
        **codec_options: Unpack[CodecOptions],
    ) -> ArrayData:
        """Read and decode the chunk at the given chunk grid indices.

        Args:
            chunk_indices: The position of the chunk in the chunk grid.
            **codec_options: The codec options, as
                [`CodecOptions`][zarrista.codec.CodecOptions].

        Returns:
            The decoded chunk data. An absent chunk decodes to the fill value.

        Raises:
            TypeError: If a keyword argument is not a known codec option.
        """
    def retrieve_encoded_chunk(self, chunk_indices: list[int]) -> EncodedChunk | None:
        """Read the raw, still-encoded bytes of the chunk at `chunk_indices`.

        The method reads the bytes and does not run the codec pipeline. To
        decode them, use [`EncodedChunk.decode`][zarrista.EncodedChunk.decode]
        or
        [`EncodedChunk.decode_async`][zarrista.EncodedChunk.decode_async].

        Args:
            chunk_indices: The position of the chunk in the chunk grid.

        Returns:
            The encoded chunk, or `None` if the chunk is absent from the store.
        """
    def retrieve_subchunk(
        self,
        subchunk_indices: list[int],
        **codec_options: Unpack[CodecOptions],
    ) -> ArrayData:
        """Read and decode a single subchunk (inner chunk) of a sharded array.

        `subchunk_indices` index the subchunk grid (see `subchunk_grid_shape`).
        For an array that is not sharded, a subchunk is a whole chunk. The
        method reads only the addressed inner chunk from its shard, not the
        whole shard.

        Args:
            subchunk_indices: The position of the subchunk in the subchunk grid.
            **codec_options: The codec options, as
                [`CodecOptions`][zarrista.codec.CodecOptions].

        Returns:
            The decoded subchunk data.

        Raises:
            TypeError: If a keyword argument is not a known codec option.
        """
    def retrieve_encoded_subchunk(
        self,
        subchunk_indices: list[int],
    ) -> EncodedChunk | None:
        """Read the raw, still-encoded bytes of a subchunk of a sharded array.

        The method reads the bytes and does not run the codec pipeline. To
        decode them, use [`EncodedChunk.decode`][zarrista.EncodedChunk.decode]
        or
        [`EncodedChunk.decode_async`][zarrista.EncodedChunk.decode_async]. The
        returned chunk carries the codec chain for the subchunks, which is the
        inner chain of the `sharding_indexed` codec.

        The array must be **exclusively** sharded: the `sharding_indexed` codec
        must be the only codec. An array-to-array codec before it, or a
        bytes-to-bytes codec after it, re-encodes the shard, so no byte range of
        the shard holds the bytes of one subchunk.

        Args:
            subchunk_indices: The position of the subchunk in the subchunk grid.

        Returns:
            The encoded subchunk, or `None` if the subchunk is absent from the
            store.

        Raises:
            ArrayError: If the array is not exclusively sharded.
        """
    @property
    def store(self) -> FilesystemStore | MemoryStore:
        """Retrieve the store backing this array."""
    def store_array_subset(
        self,
        selection: Selection,
        data: DataInput,
        **codec_options: Unpack[CodecOptions],
    ) -> None:
        """Encode `data` and write it to the region selected by `selection`.

        The region does not have to align with the chunk grid. zarrista reads,
        updates, and rewrites each chunk that the region touches, so a write
        that covers whole chunks is cheaper than one that covers parts of them.

        `data` may be any type allowed by `DataInput`.

        Args:
            selection: The region to write, in element coordinates.
            data: The data to write.
            **codec_options: The codec options, as
                [`CodecOptions`][zarrista.codec.CodecOptions].

        Raises:
            TypeError: If `data` has a different data type from the array, or if
                a keyword argument is not a known codec option.
            ValueError: If `data` has a different shape from the selected
                region, or if it is not C-contiguous.
            NotImplementedError: If `selection` uses a slice with a step that is
                not 1, or if it uses `None` (`np.newaxis`).
            IndexError: If `selection` has more entries than the array has
                dimensions.
            ArrayError: If the array is read-only, or if the size of `data` does
                not match the selected region.
        """
    def store_chunk(
        self,
        chunk_indices: list[int],
        decoded_chunk: ArrayBytes,
        **codec_options: Unpack[CodecOptions],
    ) -> None:
        """Encode `decoded_chunk` and write it as the chunk at `chunk_indices`.

        The array's codec pipeline encodes the data before the method writes it.
        If the data equals the fill value and `store_empty_chunks` is `False`,
        the method erases the chunk instead.

        Args:
            chunk_indices: The position of the chunk in the chunk grid.
            decoded_chunk: The decoded chunk data.
            **codec_options: The codec options, as
                [`CodecOptions`][zarrista.codec.CodecOptions].

        Raises:
            ArrayError: If the size of `decoded_chunk` does not match the chunk,
                or if the array is read-only.
            TypeError: If a keyword argument is not a known codec option.
        """
    def store_encoded_chunk(
        self,
        chunk_indices: list[int],
        encoded_chunk: Buffer,
    ) -> None:
        """Write already-encoded bytes directly as the chunk at `chunk_indices`.

        The method stores the bytes without any change, and it does not encode
        them. You must make sure that they match the array's codec pipeline.
        Bytes that do not match give a chunk that nothing can decode.

        Args:
            chunk_indices: The position of the chunk in the chunk grid.
            encoded_chunk: The encoded chunk bytes.

        Raises:
            StorageError: If the array is read-only.
        """
    def store_metadata(self) -> None:
        """Write the array's metadata to the store.

        This is the write counterpart to
        [`Array.from_metadata`][zarrista.Array.from_metadata], which only
        constructs an in-memory array and writes nothing.

        The method overwrites any metadata at the array's path.

        Raises:
            StorageError: If the array is read-only.
        """
    def compact_chunk(
        self,
        chunk_indices: list[int],
        **codec_options: Unpack[CodecOptions],
    ) -> bool:
        """Re-encode the stored chunk in place, and report whether it changed.

        The method reads the encoded chunk and tries to make a more compact
        encoding. If that succeeds, it rewrites the chunk.

        Args:
            chunk_indices: The position of the chunk in the chunk grid.
            **codec_options: The codec options, as
                [`CodecOptions`][zarrista.codec.CodecOptions].

        Returns:
            `True` if the method rewrote the chunk. `False` if the chunk is
            absent, or if it is already as compact as possible.

        Raises:
            ArrayError: If the stored chunk cannot be decoded, or if the array
                is read-only.
            TypeError: If a keyword argument is not a known codec option.
        """
    def erase_chunk(self, chunk_indices: list[int]) -> None:
        """Delete the chunk at `chunk_indices` from the store.

        To erase a chunk that is absent does nothing.

        Args:
            chunk_indices: The position of the chunk in the chunk grid.

        Raises:
            StorageError: If the array is read-only.
        """
    def erase_chunks(self, chunks: Selection) -> None:
        """Delete the specified chunks from the store.

        `chunks` is a numpy-style selection in **chunk-grid** coordinates, not
        element coordinates.

        On an array with chunk shape `(10, 10)`, `erase_chunks((0, slice(0, 2)))`
        erases the two chunks that cover elements `[0:10, 0:20]`.

        An empty selection (`()` or `...`) erases every chunk. For a sharded
        array the chunk grid is the shard grid, so the method erases whole
        shards. To erase chunks that are absent does nothing.

        Args:
            chunks: The chunks to erase, in chunk-grid coordinates.

        Raises:
            StorageError: If the array is read-only.
        """
    def erase_metadata(self) -> None:
        """Delete the array's metadata from the store.

        This succeeds if the metadata does not exist.

        Raises:
            StorageError: If the array is read-only.
        """
    def read_only(self) -> Array:
        """Return a read-only view of this array.

        Reads behave in the same way. Every write operation raises.

        Returns:
            A read-only view of this array.
        """
    def with_chunk_grid(self, chunk_grid: ChunkGrid) -> Array:
        """Return a new array reference with `chunk_grid`.

        This does not mutate the existing `array`.

        The new array's shape comes from the grid, so this can change the shape
        and the chunking together:

        ```py
        array = array.with_chunk_grid(ChunkGrid.regular([8, 8], [4, 4]))
        array.store_metadata()
        ```

        Nothing is persisted to the store. Call
        [`Array.store_metadata`][zarrista.Array.store_metadata] to persist the new
        grid to the store. This array is unaffected and remains usable; rebinding,
        as above, is the intended usage.

        **Existing chunks are neither migrated nor erased.** If the chunk shape
        changes, chunks already in the store sit at keys that no longer describe
        the same region of the array, so later reads may fail to decode or return
        wrong data. It is the caller's responsibility to ensure the new grid is
        compatible with whatever is already stored. The safe uses are setting the
        grid before any chunks are written, or erasing and rewriting the existing
        chunks yourself.

        If only the array shape is changing, use
        [`Array.with_shape`][zarrista.Array.with_shape] instead — it preserves the
        chunking, so existing chunks stay valid.

        Args:
            chunk_grid: The chunk grid of the new array reference. The grid also
                gives the new array shape, and it can change the number of
                dimensions.

        Returns:
            A new array reference that uses `chunk_grid`.
        """
    def with_shape(self, shape: list[int]) -> Array:
        """Return a new array reference with `shape`, leaving this one unchanged.

        Nothing is persisted to the store. Call
        [`Array.store_metadata`][zarrista.Array.store_metadata] to persist the new
        shape to the store:

        ```py
        array = array.with_shape([8, 8])
        array.store_metadata()
        ```

        This array is unaffected and remains usable; it simply goes on describing
        the old shape. Rebinding, as above, is the intended usage.

        Growing an array leaves the new region reading as the fill value. Shrinking
        leaves any chunks outside the new bounds in the store, where they are no
        longer addressable through this array; reclaiming that space will be a
        separate call.

        Args:
            shape: The shape of the new array reference, in elements along each
                dimension. This must have the same number of dimensions as the
                array's chunk grid.

        Returns:
            A new array reference that uses `shape`.

        Raises:
            ArrayCreateError: If `shape` is not compatible with the array's
                chunk grid, such as a shape of the wrong dimensionality.
            OverflowError: If an element of `shape` is negative.
        """
    @property
    def is_sharded(self) -> bool:
        """Whether the array's array-to-bytes codec is `sharding_indexed`."""
    @property
    def subchunk_shape(self) -> list[int] | None:
        """The inner-chunk shape from the `sharding_indexed` codec metadata.

        `None` if the array is not sharded.
        """
    @property
    def effective_subchunk_shape(self) -> list[int] | None:
        """The subchunk shape's effective "read granularity".

        This accounts for the array-to-array codecs (e.g. `transpose`) that come
        before the sharding codec and change the subset that one subchunk spans.
        `None` if the array is not sharded, or if the effective shape is
        indeterminate.
        """
    @property
    def subchunk_grid(self) -> ChunkGrid:
        """The subchunk grid.

        This grid comes from the effective subchunk shape. Therefore a read of
        one subchunk reads a single adjacent byte range. For an array that is
        not sharded, this is the normal chunk grid.
        """
    @property
    def subchunk_grid_shape(self) -> list[int]:
        """The shape of the subchunk grid (the number of subchunks per dimension).

        For an array that is not sharded, this is the normal chunk grid shape.
        """
    @property
    def shape(self) -> list[int]:
        """The array shape."""
    @property
    def subset_all(self) -> tuple[slice, ...]:
        """The array subset that spans the entire array, as a tuple of slices."""
    def __getitem__(self, selection: Selection) -> ArrayData:
        """Read a region with numpy-style basic indexing, e.g. `arr[0:10, :, 5]`.

        This is sugar for `retrieve_array_subset`.

        Args:
            selection: The region to read, in element coordinates.

        Returns:
            The decoded data for the region.

        Raises:
            NotImplementedError: If `selection` uses a slice with a step that is
                not 1, or if it uses `None` (`np.newaxis`).
            IndexError: If `selection` has more entries than the array has
                dimensions.
        """

class AsyncArray:
    """A Zarr array backed by an async store.

    The array's data stays in the store, which is usually not in memory. A read
    fetches only the region that you ask for, decodes it, and returns it as
    [`ArrayData`][zarrista.ArrayData].
    """

    @staticmethod
    async def open_async(store: AsyncStore, path: str = "/") -> AsyncArray:
        """Open the array stored at `path` in `store`.

        Args:
            store: The store that holds the array. This is either an obstore
                `ObjectStore` or an icechunk `Session`.
            path: The absolute path of the array in the store.

        Returns:
            The array at `path`.

        Raises:
            ArrayCreateError: If the store holds no array metadata at `path`.
            ValueError: If `path` is not a valid absolute node path.
        """
    @staticmethod
    def from_metadata(
        metadata: ZarrV3ArrayMetadataJSON,
        store: AsyncStore,
        path: str = "/",
    ) -> AsyncArray:
        """Use the provided metadata to open a new array at `path` in `store`.

        This does **not** write the metadata to the store. Use
        [`AsyncArray.store_metadata`][zarrista.AsyncArray.store_metadata] to
        write it.

        Args:
            metadata: The Zarr v3 metadata of the array.
            store: The async store to attach the array to.
            path: The absolute path of the array in the store.

        Returns:
            The new array.

        Raises:
            ArrayCreateError: If the metadata is not valid.
            ValueError: If `path` is not a valid absolute node path.
        """
    @property
    def attrs(self) -> dict[str, JSONValue]:
        """The array's user attributes as a dict."""
    @property
    def chunk_grid(self) -> ChunkGrid:
        """The chunk grid of the array."""
    @property
    def chunk_grid_shape(self) -> list[int]:
        """The shape of the chunk grid (i.e. the number of chunks per dimension)."""
    def chunk_key(self, chunk_indices: list[int]) -> str:
        """Return the store key of the chunk at `chunk_indices`.

        This does not check `chunk_indices` against the chunk grid. It gives a
        key for indices that are outside the grid.

        Args:
            chunk_indices: The position of the chunk in the chunk grid.

        Returns:
            The store key of the chunk.
        """
    @property
    def chunk_key_encoding(self) -> ChunkKeyEncoding:
        """The chunk key encoding, mapping chunk grid indices to store keys."""
    def chunk_origin(self, chunk_indices: list[int]) -> list[int]:
        """Return the origin of the chunk at `chunk_indices`.

        Args:
            chunk_indices: The position of the chunk in the chunk grid.

        Returns:
            The index of the first array element in the chunk.

        Raises:
            ArrayError: If `chunk_indices` has a different number of dimensions
                from the chunk grid.
        """
    def chunk_shape(self, chunk_indices: list[int]) -> list[int]:
        """Return the shape of the chunk at `chunk_indices`.

        Args:
            chunk_indices: The position of the chunk in the chunk grid.

        Returns:
            The shape of the chunk, in elements along each dimension.

        Raises:
            ArrayError: If `chunk_indices` has a different number of dimensions
                from the chunk grid.
        """
    def chunk_subset(self, chunk_indices: list[int]) -> tuple[slice, ...]:
        """Return the array subset spanned by the chunk at `chunk_indices`.

        Args:
            chunk_indices: The position of the chunk in the chunk grid.

        Returns:
            One slice per dimension, which together give the array subset of
            the chunk.

        Raises:
            ArrayError: If `chunk_indices` has a different number of dimensions
                from the chunk grid.
        """
    @property
    def codecs(self) -> CodecChain:
        """The codec chain that encodes and decodes the array's chunks.

        Use its `filters`, `serializer`, and `compressors` properties to inspect
        the individual codecs.
        """
    @property
    def dimension_names(self) -> list[str | None] | None:
        """The dimension names, if any were specified."""
    @property
    def dtype(self) -> DataType:
        """The Zarr data type."""
    @property
    def fill_value(self) -> FillValue:
        """The array's fill value."""
    @property
    def metadata(self) -> ZarrV3ArrayMetadataJSON:
        """The array's full Zarr v3 metadata."""
    @property
    def ndim(self) -> int:
        """The number of dimensions."""
    @property
    def path(self) -> str:
        """The array's path in the store."""
    async def retrieve_array_subset(
        self,
        selection: Selection,
        **codec_options: Unpack[CodecOptions],
    ) -> ArrayData:
        """Read and decode an array region selected with numpy-style basic indexing.

        The result keeps the number of dimensions, which agrees with a zarrs
        `ArraySubset`. An integer selects a range of length 1, and the method
        keeps that axis.

        Args:
            selection: The region to read, in element coordinates.
            **codec_options: The codec options, as
                [`CodecOptions`][zarrista.codec.CodecOptions].

        Returns:
            The decoded data for the region.

        Raises:
            NotImplementedError: If `selection` uses a slice with a step that is
                not 1, or if it uses `None` (`np.newaxis`).
            IndexError: If `selection` has more entries than the array has
                dimensions.
            TypeError: If a keyword argument is not a known codec option.
        """
    async def retrieve_chunk(
        self,
        chunk_indices: list[int],
        **codec_options: Unpack[CodecOptions],
    ) -> ArrayData:
        """Read and decode the chunk at the given chunk grid indices.

        Args:
            chunk_indices: The position of the chunk in the chunk grid.
            **codec_options: The codec options, as
                [`CodecOptions`][zarrista.codec.CodecOptions].

        Returns:
            The decoded chunk data. An absent chunk decodes to the fill value.

        Raises:
            TypeError: If a keyword argument is not a known codec option.
        """
    async def retrieve_encoded_chunk(
        self,
        chunk_indices: list[int],
    ) -> EncodedChunk | None:
        """Read the raw, still-encoded bytes of the chunk at `chunk_indices`.

        The method reads the bytes and does not run the codec pipeline. To
        decode them, use [`EncodedChunk.decode`][zarrista.EncodedChunk.decode]
        or
        [`EncodedChunk.decode_async`][zarrista.EncodedChunk.decode_async].

        Args:
            chunk_indices: The position of the chunk in the chunk grid.

        Returns:
            The encoded chunk, or `None` if the chunk is absent from the store.
        """
    async def retrieve_subchunk(
        self,
        subchunk_indices: list[int],
        **codec_options: Unpack[CodecOptions],
    ) -> ArrayData:
        """Read and decode a single subchunk (inner chunk) of a sharded array.

        `subchunk_indices` index the subchunk grid (see `subchunk_grid_shape`).
        For an array that is not sharded, a subchunk is a whole chunk. The
        method reads only the addressed inner chunk from its shard, not the
        whole shard.

        Args:
            subchunk_indices: The position of the subchunk in the subchunk grid.
            **codec_options: The codec options, as
                [`CodecOptions`][zarrista.codec.CodecOptions].

        Returns:
            The decoded subchunk data.

        Raises:
            TypeError: If a keyword argument is not a known codec option.
        """
    async def retrieve_encoded_subchunk(
        self,
        subchunk_indices: list[int],
    ) -> EncodedChunk | None:
        """Read the raw, still-encoded bytes of a subchunk of a sharded array.

        The method reads the bytes and does not run the codec pipeline. To
        decode them, use [`EncodedChunk.decode`][zarrista.EncodedChunk.decode]
        or
        [`EncodedChunk.decode_async`][zarrista.EncodedChunk.decode_async]. The
        returned chunk carries the codec chain for the subchunks, which is the
        inner chain of the `sharding_indexed` codec.

        The array must be **exclusively** sharded: the `sharding_indexed` codec
        must be the only codec. An array-to-array codec before it, or a
        bytes-to-bytes codec after it, re-encodes the shard, so no byte range of
        the shard holds the bytes of one subchunk.

        Args:
            subchunk_indices: The position of the subchunk in the subchunk grid.

        Returns:
            The encoded subchunk, or `None` if the subchunk is absent from the
            store.

        Raises:
            ArrayError: If the array is not exclusively sharded.
        """
    @property
    def store(self) -> AsyncStore:
        """Retrieve the store backing this array."""
    async def store_array_subset(
        self,
        selection: Selection,
        data: DataInput,
        **codec_options: Unpack[CodecOptions],
    ) -> None:
        """Encode `data` and write it to the region selected by `selection`.

        The region does not have to align with the chunk grid. zarrista reads,
        updates, and rewrites each chunk that the region touches, so a write
        that covers whole chunks is cheaper than one that covers parts of them.

        `data` may be any type allowed by `DataInput`.

        Args:
            selection: The region to write, in element coordinates.
            data: The data to write.
            **codec_options: The codec options, as
                [`CodecOptions`][zarrista.codec.CodecOptions].

        Raises:
            TypeError: If `data` has a different data type from the array, or if
                a keyword argument is not a known codec option.
            ValueError: If `data` has a different shape from the selected
                region, or if it is not C-contiguous.
            NotImplementedError: If `selection` uses a slice with a step that is
                not 1, or if it uses `None` (`np.newaxis`).
            IndexError: If `selection` has more entries than the array has
                dimensions.
            ArrayError: If the array is read-only, or if the size of `data` does
                not match the selected region.
        """
    async def store_chunk(
        self,
        chunk_indices: list[int],
        decoded_chunk: ArrayBytes,
        **codec_options: Unpack[CodecOptions],
    ) -> None:
        """Encode `decoded_chunk` and write it as the chunk at `chunk_indices`.

        The array's codec pipeline encodes the data before the method writes it.
        If the data equals the fill value and `store_empty_chunks` is `False`,
        the method erases the chunk instead.

        Args:
            chunk_indices: The position of the chunk in the chunk grid.
            decoded_chunk: The decoded chunk data.
            **codec_options: The codec options, as
                [`CodecOptions`][zarrista.codec.CodecOptions].

        Raises:
            ArrayError: If the size of `decoded_chunk` does not match the chunk,
                or if the array is read-only.
            TypeError: If a keyword argument is not a known codec option.
        """
    async def store_encoded_chunk(
        self,
        chunk_indices: list[int],
        encoded_chunk: Buffer,
    ) -> None:
        """Write already-encoded bytes directly as the chunk at `chunk_indices`.

        The method stores the bytes without any change, and it does not encode
        them. You must make sure that they match the array's codec pipeline.
        Bytes that do not match give a chunk that nothing can decode.

        Args:
            chunk_indices: The position of the chunk in the chunk grid.
            encoded_chunk: The encoded chunk bytes.

        Raises:
            StorageError: If the array is read-only.
        """
    async def store_metadata(self) -> None:
        """Write the array's metadata to the store.

        This is the write counterpart to
        [`AsyncArray.from_metadata`][zarrista.AsyncArray.from_metadata], which
        only constructs an in-memory array and writes nothing.

        The method overwrites any metadata at the array's path.

        Raises:
            StorageError: If the array is read-only.
        """
    async def compact_chunk(
        self,
        chunk_indices: list[int],
        **codec_options: Unpack[CodecOptions],
    ) -> bool:
        """Re-encode the stored chunk in place, and report whether it changed.

        The method reads the encoded chunk and tries to make a more compact
        encoding. If that succeeds, it rewrites the chunk.

        Args:
            chunk_indices: The position of the chunk in the chunk grid.
            **codec_options: The codec options, as
                [`CodecOptions`][zarrista.codec.CodecOptions].

        Returns:
            `True` if the method rewrote the chunk. `False` if the chunk is
            absent, or if it is already as compact as possible.

        Raises:
            ArrayError: If the stored chunk cannot be decoded, or if the array
                is read-only.
            TypeError: If a keyword argument is not a known codec option.
        """
    async def erase_chunk(self, chunk_indices: list[int]) -> None:
        """Delete the chunk at `chunk_indices` from the store.

        To erase a chunk that is absent does nothing.

        Args:
            chunk_indices: The position of the chunk in the chunk grid.

        Raises:
            StorageError: If the array is read-only.
        """
    async def erase_chunks(self, chunks: Selection) -> None:
        """Delete the specified chunks from the store.

        `chunks` is a numpy-style selection in **chunk-grid** coordinates, not
        element coordinates.

        On an array with chunk shape `(10, 10)`, `erase_chunks((0, slice(0, 2)))`
        erases the two chunks that cover elements `[0:10, 0:20]`.

        An empty selection (`()` or `...`) erases every chunk. For a sharded
        array the chunk grid is the shard grid, so the method erases whole
        shards. To erase chunks that are absent does nothing.

        Args:
            chunks: The chunks to erase, in chunk-grid coordinates.

        Raises:
            StorageError: If the array is read-only.
        """
    async def erase_metadata(self) -> None:
        """Delete the array's metadata from the store.

        This succeeds if the metadata does not exist.

        Raises:
            StorageError: If the array is read-only.
        """
    def read_only(self) -> AsyncArray:
        """Return a read-only view of this array.

        Reads behave in the same way. Every write operation raises.

        Returns:
            A read-only view of this array.
        """
    def with_chunk_grid(self, chunk_grid: ChunkGrid) -> AsyncArray:
        """Return a new array reference with `chunk_grid`, leaving this one unchanged.

        This method is synchronous: it performs no I/O. The new array's shape comes
        from the grid, so this can change the shape and the chunking together:

        ```py
        array = array.with_chunk_grid(ChunkGrid.regular([8, 8], [4, 4]))
        await array.store_metadata()
        ```

        Nothing is persisted to the store. Call
        [`AsyncArray.store_metadata`][zarrista.AsyncArray.store_metadata] to persist
        the new grid to the store. This array is unaffected and remains usable;
        rebinding, as above, is the intended usage.

        **Existing chunks are neither migrated nor erased.** If the chunk shape
        changes, chunks already in the store sit at keys that no longer describe
        the same region of the array, so later reads may fail to decode or return
        wrong data. It is the caller's responsibility to ensure the new grid is
        compatible with whatever is already stored. The safe uses are setting the
        grid before any chunks are written, or erasing and rewriting the existing
        chunks yourself.

        If only the array shape is changing, use
        [`AsyncArray.with_shape`][zarrista.AsyncArray.with_shape] instead — it
        preserves the chunking, so existing chunks stay valid.

        Args:
            chunk_grid: The chunk grid of the new array reference. The grid also
                gives the new array shape, and it can change the number of
                dimensions.

        Returns:
            A new array reference that uses `chunk_grid`.
        """
    def with_shape(self, shape: list[int]) -> AsyncArray:
        """Return a new array reference with `shape`, leaving this one unchanged.

        This method is synchronous: it performs no I/O. Nothing is persisted to the
        store. Call
        [`AsyncArray.store_metadata`][zarrista.AsyncArray.store_metadata] to persist
        the new shape to the store:

        ```py
        array = array.with_shape([8, 8])
        await array.store_metadata()
        ```

        This array is unaffected and remains usable; it simply goes on describing
        the old shape. Rebinding, as above, is the intended usage.

        Growing an array leaves the new region reading as the fill value. Shrinking
        leaves any chunks outside the new bounds in the store, where they are no
        longer addressable through this array; reclaiming that space will be a
        separate call.

        Args:
            shape: The shape of the new array reference, in elements along each
                dimension. This must have the same number of dimensions as the
                array's chunk grid.

        Returns:
            A new array reference that uses `shape`.

        Raises:
            ArrayCreateError: If `shape` is not compatible with the array's
                chunk grid, such as a shape of the wrong dimensionality.
            OverflowError: If an element of `shape` is negative.
        """
    @property
    def is_sharded(self) -> bool:
        """Whether the array's array-to-bytes codec is `sharding_indexed`."""
    @property
    def subchunk_shape(self) -> list[int] | None:
        """The inner-chunk shape from the `sharding_indexed` codec metadata.

        `None` if the array is not sharded.
        """
    @property
    def effective_subchunk_shape(self) -> list[int] | None:
        """The subchunk shape's effective "read granularity".

        This accounts for the array-to-array codecs (e.g. `transpose`) that come
        before the sharding codec and change the subset that one subchunk spans.
        `None` if the array is not sharded, or if the effective shape is
        indeterminate.
        """
    @property
    def subchunk_grid(self) -> ChunkGrid:
        """The subchunk grid.

        This grid comes from the effective subchunk shape. Therefore a read of
        one subchunk reads a single adjacent byte range. For an array that is
        not sharded, this is the normal chunk grid.
        """
    @property
    def subchunk_grid_shape(self) -> list[int]:
        """The shape of the subchunk grid (the number of subchunks per dimension).

        For an array that is not sharded, this is the normal chunk grid shape.
        """
    @property
    def shape(self) -> list[int]:
        """The array shape."""
    @property
    def subset_all(self) -> tuple[slice, ...]:
        """The array subset that spans the entire array, as a tuple of slices."""
    async def __getitem__(self, selection: Selection) -> ArrayData:
        """Read a region with numpy-style basic indexing: `await arr[0:10, :, 5]`.

        This is sugar for `retrieve_array_subset`.

        Args:
            selection: The region to read, in element coordinates.

        Returns:
            The decoded data for the region.

        Raises:
            NotImplementedError: If `selection` uses a slice with a step that is
                not 1, or if it uses `None` (`np.newaxis`).
            IndexError: If `selection` has more entries than the array has
                dimensions.
        """
