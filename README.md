# zarrista

[![PyPI][pypi_badge]][pypi_link]

[pypi_badge]: https://badge.fury.io/py/zarrista.svg
[pypi_link]: https://pypi.org/project/zarrista/

A low-level [Zarr] API for Python, inspired by [zarrita.js], powered from Rust by [Zarrs]. Serving up Zarr chunks like your favorite barista!

[Zarr]: https://zarr.dev/
[zarrita.js]: https://zarrita.dev/
[Zarrs]: https://zarrs.dev/

This has been _minimally_ vibe-coded: _mostly_ but not fully written by hand. Some areas were prototyped with Claude.

This project is for **evaluation**, to consider whether natively binding to [Zarrs] can provide better performance. It is not yet production ready.

## Documentation

[**Documentation website.**](https://developmentseed.org/zarrista/latest)

## Features

- **Low-level, explicit** Zarr access: open arrays and groups, read chunks, and inspect metadata without hidden machinery.
- **Both sync and async** APIs ([`Array`] / [`AsyncArray`], [`Group`] / [`AsyncGroup`]).
- **Rust core** via [Zarrs] for compiled performance.
- **NumPy integration**: read data into [NumPy] arrays through the buffer protocol for zero-copy sharing between Rust and Python.
- **Variety of data access**
    - AWS S3, Google Cloud Storage, Azure Storage via [Obstore]
    - [Icechunk] integration
- **Full type hinting** for all operations.

[NumPy]: https://numpy.org/
[Icechunk]: https://icechunk.io/
[`Array`]: https://developmentseed.org/zarrista/latest/api/array/#zarrista.Array
[`AsyncArray`]: https://developmentseed.org/zarrista/latest/api/array/#zarrista.AsyncArray
[`Group`]: https://developmentseed.org/zarrista/latest/api/group/#zarrista.Group
[`AsyncGroup`]: https://developmentseed.org/zarrista/latest/api/group/#zarrista.AsyncGroup
[Obstore]: https://github.com/developmentseed/obstore

## Example

Open a store, then open an `Array` from it:

```py
from zarrista import Array, FilesystemStore

store = FilesystemStore("data/example.zarr")
array = Array.open(store, path="/temperature")
```

Inspect the array's metadata:

```py
array.shape
# [720, 1440]

array.dtype
# DataType(float32)

array.dimension_names
# ["lat", "lon"]
```

Read a subset of the array. Indexing returns a `Data` buffer, which converts to a [NumPy] array:

```py
data = array[0:128, 0:128]
arr = data.to_numpy()
arr.shape
# (128, 128)
```

You can also read individual chunks by their grid index:

```py
data = array.retrieve_chunk([0, 0])
```

## Custom stores

Besides the built-in `FilesystemStore` and `MemoryStore`, you can pass any
duck-typed Python object as a (synchronous) store. The minimal contract is a
single `get` method plus two capability properties:

```python
from zarrista import Array


class DictStore:
    supports_get_partial = False  # opt into byte-range reads
    supports_listing = False      # opt into listing keys/prefixes

    def __init__(self, mapping: dict[str, bytes]):
        self._mapping = mapping

    def get(self, key: str) -> bytes | None:
        return self._mapping.get(key)


array = Array.open(DictStore(my_bytes), "/path")
```

Declare `supports_listing = True` and implement `list`, `list_prefix`,
`list_dir`, and `size_prefix` to support operations like `Group.array_keys()`;
calling a listing operation on a store that does not support it raises an error.
Declare `supports_get_partial = True` and implement `get_partial_many` to serve
efficient byte-range reads (otherwise partial reads fall back to fetching the
whole value and slicing). The `zarrista.ReadableStore` and
`zarrista.ListableStore` protocols document the full surface.

> Note: if your store defines a method named `list`, add
> `from __future__ import annotations` to the module so later `list[...]` type
> annotations are not shadowed by the method.

This is sync-only; for async use pass an `obstore.ObjectStore`.

## Development

Requires a Rust toolchain and Python 3.11+. We use
[uv](https://docs.astral.sh/uv/) and [maturin](https://www.maturin.rs/).

```bash
# Create a dev environment and install the dev dependencies
uv sync --no-install-package zarrista

# Build the Rust extension and install it into the environment (debug build)
uv run --no-project maturin develop --uv

# Or, in release mode:
uv run --no-project maturin develop --uv --release

# Run the tests
uv run --no-project pytest
```

The `--no-project` is annoying but unavoidable in our current setup. Otherwise `uv` will try to build the rust library _in release mode, as a dependency of the project_ before reaching `uv sync` or `uv run`.
