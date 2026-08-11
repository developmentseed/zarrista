# Codec

::: zarrista.codec

<!--
`CodecOptions` is a type-check-only `TypedDict` with no runtime object, so it is
not in `zarrista.codec.__all__` and the block above does not render it. We render
it explicitly to give the `[CodecOptions][zarrista.codec.CodecOptions]`
cross-references in the array read/write docstrings a target (required under
`mkdocs build --strict`).
-->
::: zarrista.codec.CodecOptions

## Codec types

::: zarrista.codec.ArrayToArrayCodec
    options:
      show_bases: false

::: zarrista.codec.ArrayToBytesCodec
    options:
      show_bases: false

::: zarrista.codec.BytesToBytesCodec
    options:
      show_bases: false

::: zarrista.codec.CodecChain
    options:
      show_bases: false

## Array-to-array codecs

::: zarrista.codec.transpose

::: zarrista.codec.bitround

## Bytes-to-bytes codecs

::: zarrista.codec.blosc

::: zarrista.codec.crc32c

::: zarrista.codec.gzip

::: zarrista.codec.zstd
