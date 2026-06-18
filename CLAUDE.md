# zarrista

A small, prototypical zarrita-like Python Zarr implementation on top of `zarrs`,
exposed to Python via `pyo3`.

## Rust / pyo3 conventions

- **Elide lifetimes whenever possible.** Prefer `'_` over named lifetime
  parameters when the names are not actually referenced. For example, implement
  `FromPyObject` as `impl FromPyObject<'_, '_> for T` with
  `fn extract(ob: Borrowed<'_, '_, PyAny>)` rather than introducing `<'a, 'py>`.
- **Extract to `PyBackedStr`, not `String`, when you don't need ownership.** When
  a `FromPyObject` impl only inspects the string (e.g. matching against known
  values), extract `let s: PyBackedStr = ob.extract()?;` instead of an owned
  `String` to avoid a needless allocation. `PyBackedStr` derefs to `str`.
