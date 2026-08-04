class ShardCache:
    """A cache of the shard indexes of one array.

    A sharded array holds many subchunks inside one shard. Every shard starts
    with a shard index. The shard index is a table that gives the byte offset
    and the byte length of each subchunk in that shard. To read one subchunk,
    zarrista must first read the shard index of the shard that contains it.

    A `ShardCache` keeps the shard index of each shard that zarrista read
    through it. It does **not** keep the subchunk data, and it does **not**
    keep the shard data. A later read of a different subchunk of the same shard
    reuses the cached shard index. That read gets the subchunk bytes directly,
    and does not read the shard index again.

    Each entry holds two 64-bit integers for every subchunk in the shard. A
    shard of 1024 subchunks therefore costs about 16 KiB.

    Use [`Array.subchunk_cache`][zarrista.Array.subchunk_cache] to create a
    cache. A cache belongs to the array that created it, because it stores
    byte offsets into that array's shards. Do not use a cache with a different
    array. zarrista cannot detect this, and the read returns incorrect data.

    Pass the cache to
    [`Array.retrieve_subchunk`][zarrista.Array.retrieve_subchunk] or
    [`Array.retrieve_encoded_subchunk`][zarrista.Array.retrieve_encoded_subchunk].
    If you do not pass a cache, each call creates a cache, uses it once, and
    discards it. Then every call reads the shard index again.

    A cache is safe to share between threads. Reads through one cache take an
    internal lock, so they do not run at the same time. Give each thread its own
    cache if you want the reads to run in parallel.

    Examples:
        Read every subchunk of a shard, and read the shard index only once:

        ```py
        cache = arr.subchunk_cache()
        for i in range(arr.subchunk_grid_shape[0]):
            sub = arr.retrieve_subchunk([i, 0], subchunk_cache=cache)
        ```
    """

    def clear(self) -> None:
        """Remove every shard index from the cache.

        Use this to release the memory of the cache. The cache stays usable.
        """

    def is_empty(self) -> bool:
        """Return whether the cache holds no shard index.

        Returns:
            `True` if the cache is empty.
        """

    def size(self) -> int:
        """Return the number of shard indexes in the cache.

        Returns:
            The number of shards that the cache holds an index for.
        """

class AsyncShardCache:
    """A cache of the shard indexes of one array, for use with `AsyncArray`.

    This is the `AsyncArray` form of
    [`ShardCache`][zarrista.ShardCache]. The methods are coroutines, because the
    cache uses an asynchronous lock.

    Use [`AsyncArray.subchunk_cache`][zarrista.AsyncArray.subchunk_cache] to
    create a cache. Read
    [`ShardCache`][zarrista.ShardCache] for what the cache holds and for the
    rules that apply to it.

    Examples:
        Read every subchunk of a shard, and read the shard index only once:

        ```py
        cache = arr.subchunk_cache()
        for i in range(arr.subchunk_grid_shape[0]):
            sub = await arr.retrieve_subchunk([i, 0], subchunk_cache=cache)
        ```
    """

    async def clear(self) -> None:
        """Remove every shard index from the cache.

        Use this to release the memory of the cache. The cache stays usable.
        """

    async def is_empty(self) -> bool:
        """Return whether the cache holds no shard index.

        Returns:
            `True` if the cache is empty.
        """

    async def size(self) -> int:
        """Return the number of shard indexes in the cache.

        Returns:
            The number of shards that the cache holds an index for.
        """
