# DecodedArray

Reading from an [`Array`][zarrista.Array] (via `retrieve_array_subset`,
`retrieve_chunk`, or `[...]`) returns a `DecodedArray`: one of four concrete result
types, chosen by the decoded byte layout of the dtype. Use `isinstance` to narrow to
a concrete type before calling its layout-specific methods.

- [`Tensor`](#zarrista.Tensor) — fixed-width, dense data.
- [`VariableArray`](#zarrista.VariableArray) — variable-length data (e.g. strings or
  bytes).
- [`MaskedTensor`](#zarrista.MaskedTensor) — fixed-width data with a validity mask.
- [`MaskedVariableArray`](#zarrista.MaskedVariableArray) — variable-length data with a
  validity mask.

::: zarrista.DecodedArray
    options:
      show_bases: false

## Tensor

::: zarrista.Tensor
    options:
      show_bases: false

## VariableArray

::: zarrista.VariableArray
    options:
      show_bases: false

## MaskedTensor

::: zarrista.MaskedTensor
    options:
      show_bases: false

## MaskedVariableArray

::: zarrista.MaskedVariableArray
    options:
      show_bases: false
