# Array data

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
