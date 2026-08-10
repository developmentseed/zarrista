# Tensor

A read from an [`Array`][zarrista.Array] returns a `Tensor`. This applies to
`retrieve_array_subset`, `retrieve_chunk`, and `[...]`. A `Tensor` is one of
four concrete result types, and the decoded byte layout of the data type selects
which one. Use `isinstance` to narrow to a concrete type before you call a method
that belongs to one layout.

- [`FixedLengthTensor`](#zarrista.FixedLengthTensor) — fixed-width, dense data.
- [`VariableLengthTensor`](#zarrista.VariableLengthTensor) — variable-length data
  (e.g. strings or bytes).
- [`OptionalFixedLengthTensor`](#zarrista.OptionalFixedLengthTensor) — fixed-width
  data with a validity mask.
- [`OptionalVariableLengthTensor`](#zarrista.OptionalVariableLengthTensor) —
  variable-length data with a validity mask.

::: zarrista.Tensor
    options:
      show_bases: false

::: zarrista.FixedLengthTensor
    options:
      show_bases: false

::: zarrista.VariableLengthTensor
    options:
      show_bases: false

::: zarrista.OptionalFixedLengthTensor
    options:
      show_bases: false

::: zarrista.OptionalVariableLengthTensor
    options:
      show_bases: false
