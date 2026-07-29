# Exception hierarchy redesign

**Date**: 2026-06-19
**Status**: Design — awaiting review

## Goal

Replace zarrista's mostly-flat Python exception hierarchy (everything funnels
into a single `ZarristaError`, plus a one-off `NotFoundError`) with a structured
hierarchy: a base exception with one dedicated subclass per underlying `zarrs`
error struct. This lets Python users catch specific failure categories
(`CodecError`, `StorageError`, …) instead of a catch-all.

The hierarchy is intentionally **one level deep** for now: a single top-level
exception per underlying `zarrs` error, with no per-variant explosion (e.g.
`CodecError` stays a single leaf even though the upstream `zarrs` `CodecError`
enum has ~22 variants). The design must leave room to add more specific
subclasses later without breaking existing `except` clauses.

## Exception hierarchy

```
ZarristaError                       (base)
├── NotFoundError
├── ArrayCreateError
├── ArrayError
├── GroupCreateError
├── NodeCreateError
├── NodePathError
├── StorageError
├── CodecError
├── TransposeOrderError
├── PluginCreateError
└── SerializationError
```

- Subclass names mirror the upstream `zarrs` error type names for
  discoverability (`ArrayError`, `StorageError`, `CodecError`, …).
- `NotFoundError` is zarrista-specific (raised when no node exists at a path)
  and is retained.
- `SerializationError` does not mirror a single `zarrs` type — it is the home
  for both `serde_json` (de)serialization failures and `pythonize` Python↔Rust
  conversion failures, which are conceptually the same category.
- The base Python class is named `ZarristaError`. The Rust `#[pyclass]`/
  `create_exception!` type stays named `ZarristaException` to avoid colliding
  with the `ZarristaError` *enum* in `src/error.rs`; Python sees it as
  `ZarristaError` (this is already the case today via the `m.add("ZarristaError",
  …)` line).

## Mapping from the `ZarristaError` Rust enum

The `ZarristaError` enum in `src/error.rs` is unchanged. Only the
`From<ZarristaError> for PyErr` impl changes, mapping each variant to its new
dedicated exception:

| `ZarristaError` variant   | Python exception              |
| ------------------------- | ----------------------------- |
| `NotFound`                | `NotFoundError`               |
| `Py(PyErr)`               | passthrough (returned as-is)  |
| `ArrayCreate`             | `ArrayCreateError`            |
| `Array`                   | `ArrayError`                  |
| `GroupCreate`             | `GroupCreateError`            |
| `NodeCreate`              | `NodeCreateError`             |
| `NodePath`                | `NodePathError`               |
| `Storage`                 | `StorageError`                |
| `FilesystemStoreCreate`   | `StorageError`                |
| `Codec`                   | `CodecError`                  |
| `TransposeOrder`          | `TransposeOrderError`         |
| `PluginCreate`            | `PluginCreateError`           |
| `SerdeJson`               | `SerializationError`          |
| `Pythonize`               | `SerializationError`          |

Each non-passthrough variant maps via `<Exception>::new_err(err.to_string())`,
matching the current pattern. `Py(PyErr)` is returned unchanged so native
Python exception types (e.g. `TypeError`, `ValueError`) are preserved.

Note: `Pythonize` currently maps to a native `PyErr` via `err.into()`; this
changes to `SerializationError::new_err(err.to_string())`.

## Module structure

Exceptions move into a dedicated `zarrista.exceptions` module, mirroring the
existing two-layer `zarrista.codec` pattern.

### Rust: `src/exceptions.rs` (new)

- Defines every exception class via `create_exception!`, with the base
  (`ZarristaException` → Python `ZarristaError`) as the parent of all others.
- Sets the module name to `zarrista.exceptions` so `__module__` is correct.
- Provides `pub fn register_exceptions_module(parent: &Bound<'_, PyModule>) ->
  PyResult<()>`, analogous to `register_codec_module`: builds the `exceptions`
  submodule, adds every exception class, registers it in `sys.modules` as
  `zarrista._zarrista.exceptions`, and attaches it via `parent.add_submodule`.

### Rust: `src/error.rs` (changed)

- Keeps the `ZarristaError` enum, `ZarristaResult`, and the
  `From<ZarristaError> for PyErr` impl.
- Removes the inline `create_exception!` definitions; imports the exception
  types from `crate::exceptions` instead.

### Rust: `src/lib.rs` (changed)

- `mod exceptions;` added.
- Replaces the two top-level `m.add("ZarristaError", …)` / `m.add("NotFoundError",
  …)` lines with a single `register_exceptions_module(m)?;` call. The exceptions
  are no longer exposed at the top level of `zarrista._zarrista`; their single
  home is `zarrista.exceptions`.

### Python: `python/zarrista/exceptions.py` (new)

- Re-exports all exception classes from `zarrista._zarrista.exceptions` with an
  explicit `__all__`, mirroring `python/zarrista/codec/__init__.py`.

### Python: `python/zarrista/__init__.py` (changed)

- Adds `from . import exceptions` and lists `"exceptions"` in `__all__`, for
  parity with how `codec` is exposed.

## Usage examples

```python
from zarrista.exceptions import CodecError, ZarristaError, NotFoundError

try:
    array[...]
except CodecError as e:
    ...  # specific
except ZarristaError as e:
    ...  # catch-all base still works
```

## Future extensibility

Adding a more specific exception later (e.g. splitting out an
`InvalidChecksumError` under `CodecError`) requires only:

1. A new `create_exception!(... , InvalidChecksumError, CodecError, ...)` in
   `src/exceptions.rs`, added to the module registration.
2. A new arm (or refined matching) in `From<ZarristaError> for PyErr`.

Existing `except CodecError` / `except ZarristaError` clauses keep working
because the new class is a subclass. No breaking change.

## Testing

- A Python test asserting the class hierarchy: every leaf is a subclass of
  `ZarristaError`; `NotFoundError` etc. are importable from
  `zarrista.exceptions`.
- A test that triggers a representative error per category and asserts the
  expected exception type is raised (at minimum: `NotFoundError` for a missing
  path; `CodecError` or `ArrayError` for a decode failure if easily triggerable).
- Assert `ZarristaError` (base) still catches subclass instances.

## Out of scope

- Per-variant explosion of any upstream `zarrs` enum (deferred; see Future
  extensibility).
- Changing the Rust `ZarristaError` enum shape or any non-error code paths.
```