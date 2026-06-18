# zarrista

A small, prototypical zarrita-like Python Zarr implementation on top of `zarrs`,
exposed to Python via `pyo3`.

## Rust / pyo3 conventions

- **Prefix every `#[pyclass]` type with `Py`, and set the macro `name` to the
  unprefixed form.** e.g. `#[pyclass(name = "Blosc")] pub struct PyBlosc(...)`.
  This keeps it clear in Rust what's a Python-facing wrapper vs. an upstream
  type, while Python still sees the clean name (`Blosc`).
- **Elide lifetimes whenever possible.** Prefer `'_` over named lifetime
  parameters when the names are not actually referenced. For example, implement
  `FromPyObject` as `impl FromPyObject<'_, '_> for T` with
  `fn extract(ob: Borrowed<'_, '_, PyAny>)` rather than introducing `<'a, 'py>`.
- **Extract to `PyBackedStr`, not `String`, when you don't need ownership.** When
  a `FromPyObject` impl only inspects the string (e.g. matching against known
  values), extract a `PyBackedStr` instead of an owned `String` to avoid a
  needless allocation. `PyBackedStr` derefs to `str`.
- **Prefer turbofish on `extract`.** Write `let name = ob.extract::<PyBackedStr>()?;`
  rather than annotating the binding (`let name: PyBackedStr = ob.extract()?;`).

## Python conventions

- **Prefer absolute imports over relative imports.** Write
  `from zarrista.codec._array_to_array import ArrayToArrayCodec`, not
  `from ._array_to_array import ...`. The package root is `zarrista` (maturin's
  `python-source = "python"`).
