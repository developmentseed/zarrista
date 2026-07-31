# Codec

::: zarrista.codec

<!--
The codec members are re-exported into `zarrista.codec` from the compiled
extension (`zarrista._zarrista.codec`). There is no stub for that submodule --
`_zarrista.pyi` is a single-file stub -- so under `allow_inspection: false`
griffe cannot resolve those aliases, and `::: zarrista.codec` renders none of
them. We therefore document each one at the stub module that defines it, and
override `heading`/`toc_label` so the public name is what readers see.

The permalink anchors still carry the private path (e.g.
`#zarrista.codec._bytes_to_bytes._blosc.blosc`). Fixing that needs the
`_zarrista` stub to become a stub package so the aliases resolve.

`CodecOptions` is the exception: it is a type-check-only `TypedDict` with no
runtime object, so its only definition is in the stubs and the alias resolves.
It is not in `zarrista.codec.__all__`, so the block above does not render it,
and we render it explicitly to give the
`[CodecOptions][zarrista.codec.CodecOptions]` cross-references in the array
read/write docstrings a target (required under `mkdocs build --strict`).
-->
::: zarrista.codec.CodecOptions

## Codec types

::: zarrista.codec._array_to_array.ArrayToArrayCodec
    options:
      heading: ArrayToArrayCodec
      toc_label: ArrayToArrayCodec
      show_bases: false

::: zarrista.codec._array_to_bytes.ArrayToBytesCodec
    options:
      heading: ArrayToBytesCodec
      toc_label: ArrayToBytesCodec
      show_bases: false

::: zarrista.codec._bytes_to_bytes.BytesToBytesCodec
    options:
      heading: BytesToBytesCodec
      toc_label: BytesToBytesCodec
      show_bases: false

## Array-to-array codecs

::: zarrista.codec._array_to_array.transpose
    options:
      heading: transpose
      toc_label: transpose

::: zarrista.codec._array_to_array.bitround
    options:
      heading: bitround
      toc_label: bitround

## Bytes-to-bytes codecs

::: zarrista.codec._bytes_to_bytes._blosc.blosc
    options:
      heading: blosc
      toc_label: blosc

::: zarrista.codec._bytes_to_bytes._crc32c.crc32c
    options:
      heading: crc32c
      toc_label: crc32c

::: zarrista.codec._bytes_to_bytes._gzip.gzip
    options:
      heading: gzip
      toc_label: gzip

::: zarrista.codec._bytes_to_bytes._zstd.zstd
    options:
      heading: zstd
      toc_label: zstd
