from typing import TypedDict

class CodecOptions(TypedDict, total=False):
    """Per-operation codec options for encoding and decoding.

    These control runtime behaviour such as concurrency limits and checksum
    validation. They are passed as keyword arguments to the array read/write
    methods, e.g. `arr.retrieve_chunk([0, 0], validate_checksums=False)`. All
    keys are optional; omitted keys fall back to the defaults noted below.

    !!! warning "Not importable at runtime"

        To use this type hint in your code, import it within a `TYPE_CHECKING`
        block:

        ```py
        from __future__ import annotations
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            from zarrista.codec import CodecOptions
        ```
    """

    validate_checksums: bool
    """Whether to validate checksums when decoding. Defaults to `True`."""

    store_empty_chunks: bool
    """Whether to store chunks that are entirely the fill value. Defaults to `False`."""

    concurrent_target: int
    """Preferred number of concurrent operations. Defaults to the number of
    threads available to Rayon."""

    chunk_concurrent_minimum: int
    """Preferred minimum chunk concurrency for multi-chunk operations. The
    concurrency of internal codecs is adjusted to accommodate the chunk
    concurrency in accordance with `concurrent_target`. Defaults to `4`."""

    experimental_partial_encoding: bool
    """Whether to use experimental partial encoding. Defaults to `False`."""
