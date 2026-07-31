# zarrista

[![PyPI][pypi_badge]][pypi_link]

[pypi_badge]: https://badge.fury.io/py/zarrista.svg
[pypi_link]: https://pypi.org/project/zarrista/

A low-level [Zarr] API for Python, inspired by [zarrita.js], powered from Rust by [Zarrs]. Serving up Zarr chunks like your favorite barista!

[Zarr]: https://zarr.dev/
[zarrita.js]: https://zarrita.dev/
[Zarrs]: https://zarrs.dev/

This project is _minimally_ vibe-coded. A person wrote most of the code by hand, and Claude prototyped some areas.

This project is for **evaluation**. It examines whether a native binding to [Zarrs] gives better performance. It is not ready for production.

## Documentation

[**Documentation website.**](https://developmentseed.org/zarrista/latest)

## Features

- **Low-level, explicit** Zarr access. Open arrays and groups, read chunks, and examine metadata. The API hides no machinery.
- **Both sync and async** APIs ([`Array`] / [`AsyncArray`], [`Group`] / [`AsyncGroup`]).
- **Rust core** through [Zarrs], for the performance of compiled code.
- **NumPy integration**. Read data into [NumPy] arrays through the buffer protocol. Rust and Python share the memory without a copy.
- **Variety of data access**
    - AWS S3, Google Cloud Storage, Azure Storage through [Obstore]
    - [Icechunk] integration
- **Full type hinting** for all operations.

[NumPy]: https://numpy.org/
[Icechunk]: https://icechunk.io/
[`Array`]: https://developmentseed.org/zarrista/latest/api/array/#zarrista.Array
[`AsyncArray`]: https://developmentseed.org/zarrista/latest/api/array/#zarrista.AsyncArray
[`Group`]: https://developmentseed.org/zarrista/latest/api/group/#zarrista.Group
[`AsyncGroup`]: https://developmentseed.org/zarrista/latest/api/group/#zarrista.AsyncGroup
[Obstore]: https://github.com/developmentseed/obstore
[`DecodedArray`]: https://developmentseed.org/zarrista/latest/api/decoded_array/#zarrista.DecodedArray

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
# DataType(float32 / <f4)

array.dimension_names
# ["lat", "lon"]
```

Read a subset of the array. Indexing returns a [`DecodedArray`], which converts to a [NumPy] array:

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
