---
name: preparing-a-release
description: Use before cutting a zarrista release, and when reviewing any new or changed public API — a new `#[pymethods]` entry, a changed signature, a new `.pyi` class — to check the stub against the extension, settle positional-only and keyword-only markers while they are still free to change, and confirm every `Raises:` entry.
---

# Preparing a release

## Overview

The extension and its type stubs are written by hand in two places, and no tool
in CI compares them. pydoclint checks a docstring against its own stub, mkdocs
checks references, and pytest only exercises what a test calls. So a stub can
promise a parameter that the Rust never accepts, and everything stays green.

A release is the last cheap moment to fix a signature. After it, every parameter
name and every calling convention is a compatibility promise.

## When to use

- Before cutting a release.
- After adding or changing anything in a `#[pymethods]` block.
- After adding a `.pyi` class or method.

## 1. Stub against runtime

```bash
uv run --no-project maturin develop --uv
uv run --no-project python .claude/skills/auditing-api-docs/check_api_parity.py
```

This compares each stub signature with `inspect.signature` of the object the
extension exposes, and reports parameter names, order, and kinds
(positional-only, keyword-only, `**kwargs`). It found `retrieve_array_subset`
advertising `**codec_options` that the Rust did not take, and a `/` added to the
async Rust but missed in the async stub.

It skips dunders on purpose: CPython slot wrappers report the slot's own
parameter names (`key`, `value`), which never match the stub's.

## 2. Calling convention

For each new or changed signature, decide the markers **now**:

| Marker | Use when | Example |
| --- | --- | --- |
| `/` positional-only | the name adds nothing at the call site | `.shape(shape, /)`, `child(name, /)` |
| `*` keyword-only | two adjacent parameters share a type, or the argument is an option | `regular(array_shape, *, chunk_shape)` |
| neither | the caller may reasonably want to name it | `open(store, path="/")` |

Notes that cost time to rediscover:

- pyo3 **rejects** `signature` on magic methods. They are already
  positional-only, so only the stub needs `/`.
- A parameter that shadows a builtin (`bytes`) makes griffe resolve the
  *annotation* to the parameter, which produces a broken docs link. Rename it.
- With `**kwargs` present, a rejected keyword reports `missing 1 required
  positional argument`, not "passed as keyword". Still an error, vaguer message.
- Adding `/` or `*` after a release is breaking; removing either is not. Decide
  early, and prefer the permissive option when genuinely unsure.

## 3. Raises accuracy

Confirm each `Raises:` entry by calling the built extension, per CLAUDE.md. The
types are not always the obvious ones — a read-only array raises `ArrayError`
from `store_chunk` but `StorageError` from `erase_chunk`.

Each stub that documents an exception must also import it, or the docs link
silently degrades to plain text. The docs audit catches that.

## 4. Docs

**REQUIRED SUB-SKILL:** run `auditing-api-docs` for the cross-reference,
anchor, and `Returns:` checks.

## 5. Gates

```bash
cargo clippy --all-targets --all-features -- -D warnings
cargo +nightly-2026-06-01 fmt --all -- --check --unstable-features \
    --config imports_granularity=Module,group_imports=StdExternalCrate
uv run --no-project ruff check . && uv run --no-project ruff format --check .
uv run --no-project pydoclint python $(find python -name '*.pyi')
uv run --no-project pytest -q
uv run --no-project mkdocs build --strict
```

Also build a wheel when the stub tree changed shape, and confirm the new files
ship:

```bash
uv run --no-project maturin build --out dist
python -c "import zipfile; print('\n'.join(zipfile.ZipFile('dist/<wheel>').namelist()))"
```

## Common mistakes

- **Trusting a green suite.** Every failure this skill exists to catch passes
  clippy, ruff, pydoclint, pytest, and `mkdocs --strict`.
- **Verifying a check only against passing input.** A check that reports zero
  may be broken. Feed it a known-bad case and confirm it complains.
- **Changing the Rust and forgetting the async twin, or the stub.** Sync and
  async are separate `#[pymethods]` blocks and separate stub classes; the shared
  macros cover only the metadata accessors.
- **Testing that an option is accepted rather than honored.** `f(x, validate=True)`
  returning without error proves nothing if the option is parsed and dropped.
  Corrupt the input and assert the behavior changes.
