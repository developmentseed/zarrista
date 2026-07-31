class ThreadPool:
    """A dedicated Rust thread pool for CPU-bound work.

    You do not need this class for normal use. By default,
    [`EncodedChunk.decode_async`][zarrista.EncodedChunk.decode_async] runs on
    the global Rust thread pool. zarrista shares that same pool for all of its
    internal parallel work, so one pool serves the whole process. To set its
    size, use the `RAYON_NUM_THREADS` environment variable.

    Construct a `ThreadPool` only when you want to isolate decode work from the
    global pool. Each `ThreadPool` adds threads to the process, so two pools
    that each have one thread per core can compete for the CPU.

    Examples:
        Decode on a pool of four threads:

        ```py
        pool = zarrista.ThreadPool(4)
        chunk = array.retrieve_encoded_chunk([0, 0])
        decoded = await chunk.decode_async(pool=pool)
        ```
    """

    def __init__(self, num_threads: int) -> None:
        """Construct a thread pool that has a fixed number of threads.

        Args:
            num_threads: The number of threads in the pool.
        """
