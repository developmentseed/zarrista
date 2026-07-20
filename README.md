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
from zarrista import Array
from zarrista.store import FilesystemStore

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
