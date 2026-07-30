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

## Documentation conventions

- **Write all documentation in the spirit of ASD-STE100 Simplified Technical
  English.** This applies to docstrings, `.pyi` stubs, Rust doc comments,
  Markdown pages, and the README. Use one topic per sentence. Keep sentences
  short: 20 words or fewer for instructions, 25 or fewer for descriptions. Use
  the active voice. Use one term for one concept, and do not use synonyms. Do
  not omit articles or relative pronouns (`that`, `which`) to save space. Keep
  the Zarr domain terms (`codec`, `sharding`, `chunk grid`) as they are.
- **Use the active voice for functions and methods, and a noun phrase for
  everything else.** A function or method docstring starts with an imperative
  verb: "Construct a regular grid...", "Return the store key...". A type alias,
  class, attribute, or property docstring names what the thing *is*: "The chunk
  sizes along one dimension." Do not write "Give a single chunk size" on a type.
- **Write all Python docstrings in Google style.** Use `Args:`, `Returns:`,
  `Raises:`, and `Examples:` sections. Do not use NumPy underlines or Sphinx
  `:param:` fields. Every documented parameter must exist in the signature, and
  types belong in the signature, not in the docstring.
- **Verify each `Raises:` entry before you write it.** Name the concrete
  exception type, and confirm it by calling the built extension. Do not infer
  the type from the Rust source, and do not guess. If you cannot confirm it,
  omit the section.

## Python conventions

- **Prefer absolute imports over relative imports.** Write
  `from zarrista.codec._array_to_array import ArrayToArrayCodec`, not
  `from ._array_to_array import ...`. The package root is `zarrista` (maturin's
  `python-source = "python"`).
