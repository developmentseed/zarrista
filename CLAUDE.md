# zarrista

A small, prototypical zarrita-like Python Zarr implementation on top of `zarrs`,
exposed to Python via `pyo3`.

## Design philosophy

- **Type-driven design, "parse, don't validate."** Encode invariants in types so
  that illegal states are unrepresentable, rather than accepting loose inputs and
  checking them afterward.
- **The `FromPyObject` extractor is the validator.** Parse each input at the pyo3
  boundary into its final, already-valid typed form (use `#[derive(FromPyObject)]`
  enums / unions for inputs that can take several shapes). The rest of the code
  then handles only well-formed values, and the parsing logic lives once on the
  type and is reused by every entry point.
- **No manual validation in function bodies.** Prefer a richly-typed single
  argument over several nullable, mutually-dependent keywords (which force
  cross-field checks). Example: sharding is an array→bytes codec, so it occupies
  the single `serializer` slot — there is no separate `shards` keyword to
  cross-check against it. Avoid `Option`-everything-then-validate; reserve
  `Option` for genuinely optional settings with meaningful defaults.

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

## Documentation conventions

- **The `.pyi` stubs are the single source of truth for user-facing docs.** The
  docs site renders from them only (`allow_inspection: false` in `mkdocs.yml`),
  so full prose, examples, and mkdocstrings cross-references
  (`` [`Array.store_metadata`][zarrista.Array.store_metadata] ``) belong in the
  `.pyi` and nowhere else.
- **A `///` doc comment on a Python-facing item gets the one-line summary only.**
  On `#[pymethods]`, `#[pyclass]`, and `#[pyfunction]` items the doc comment
  becomes Python's `__doc__` (what `help()` shows), which would otherwise
  duplicate the stub. Don't restate the stub's prose there; a single summary line
  is enough. Everything else in `src/` — plain `impl` blocks, `FromPyObject`
  types, private helpers — is Rust-only and keeps full doc comments.
- **Implementation notes go in `//` comments, not `///`.** Rationale aimed at
  Rust readers (why an upstream type was avoided, alignment caveats, safety
  reasoning) must not leak into a Python docstring.
