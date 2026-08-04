# Array data

Three types carry array data, and they differ in where the data lives and in
what it knows about itself.

| Type | Where the data lives | Data type and shape |
| --- | --- | --- |
| [`Array`][zarrista.Array] | in the store, usually not in memory | yes, from the metadata |
| [`ArrayBytes`][zarrista.ArrayBytes] | in memory | no |
| `ArrayData` | in memory | yes |

An `Array` addresses a whole Zarr array, which can be larger than memory. An
`ArrayData` is a piece of one, decoded and in memory. An `ArrayBytes` is the
same bytes without a data type or a shape, which is what a codec reads and
writes.

A read from an [`Array`][zarrista.Array] returns an `ArrayData`. This applies to
`retrieve_array_subset`, `retrieve_chunk`, and `[...]`. An `ArrayData` is one of
four concrete result types, and the decoded byte layout of the data type selects
which one. Use `isinstance` to narrow to a concrete type before you call a method
that belongs to one layout.

- [`Tensor`](#zarrista.Tensor) — fixed-width, dense data.
- [`VariableArray`](#zarrista.VariableArray) — variable-length data (e.g. strings or
  bytes).
- [`MaskedTensor`](#zarrista.MaskedTensor) — fixed-width data with a validity mask.
- [`MaskedVariableArray`](#zarrista.MaskedVariableArray) — variable-length data with a
  validity mask.

::: zarrista.ArrayData
    options:
      show_bases: false

::: zarrista.Tensor
    options:
      show_bases: false

::: zarrista.VariableArray
    options:
      show_bases: false

::: zarrista.MaskedTensor
    options:
      show_bases: false

::: zarrista.MaskedVariableArray
    options:
      show_bases: false
