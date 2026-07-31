# Concurrency

zarrista does CPU-bound work, such as decoding, on a global Rust thread pool.
One pool serves the whole process. To set its size, use the
`RAYON_NUM_THREADS` environment variable.

To run the work on a separate pool instead, construct a
[`ThreadPool`][zarrista.ThreadPool] and pass it to
[`EncodedChunk.decode_async`][zarrista.EncodedChunk.decode_async].

To limit how much of the pool one operation claims, pass `concurrent_target`
as a codec option. See [`CodecOptions`][zarrista.codec.CodecOptions].

::: zarrista.ThreadPool
    options:
      show_bases: false
