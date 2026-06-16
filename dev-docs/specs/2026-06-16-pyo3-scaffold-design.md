# zarrsita — pyo3 scaffold design

**Date:** 2026-06-16
**Status:** Approved

## Goal

Initialize a minimal pyo3 + maturin project setup for `zarrsita`, a prototypical
zarrita-like Python Zarr implementation that will eventually bind to the
[`zarrs`](https://github.com/LDeakin/zarrs) Rust crate. This first step proves
the build toolchain end-to-end with a hello-world extension module. The `zarrs`
dependency and real bindings are deliberately out of scope here.

Reference pattern: the sibling `obstore` project (Cargo workspace + nested crate,
private compiled module surfaced through a pure-Python package).

## Decisions

- **Layout:** flat single crate at the repo root (not a workspace). Can be
  promoted to a workspace later if sibling crates are needed.
- **Scope:** hello-world only — no `zarrs` dependency yet.
- **Tooling:** minimal — maturin, ruff, pytest. No pre-commit, CI, mypy stubs,
  or docs yet.

## Layout

```
zarrsita/
  Cargo.toml          # single crate, [lib] crate-type=["cdylib"], name="_zarrsita"
  pyproject.toml      # maturin backend + dev deps (pytest, ruff)
  src/lib.rs          # #[pymodule] fn _zarrsita with __version__ + hello()
  python/zarrsita/
    __init__.py       # re-exports from ._zarrsita
  tests/
    test_smoke.py     # import zarrsita; assert version + hello()
  .gitignore          # extend for Rust/Python/maturin
  README.md           # add build/dev instructions
```

## Key choices

- **pyo3 0.29** (matches obstore) with the `extension-module` feature, and
  **abi3-py311** so a single wheel covers Python 3.11+. `requires-python = ">=3.11"`.
- The compiled module is **private** (`_zarrsita`) and surfaced through the
  `python/zarrsita/__init__.py` pure-Python package, so Python-side code and
  stubs can be added later without restructuring.
- maturin config: `module-name = "zarrsita._zarrsita"`, `python-source = "python"`,
  `features = ["pyo3/extension-module"]`.
- `src/lib.rs` exposes `__version__` (from `CARGO_PKG_VERSION`) and a trivial
  `hello() -> str` to prove the round-trip.
- Rust `[profile.release]` with `lto = true` for release builds.

## Success criteria

- `maturin develop` builds the extension.
- `python -c "import zarrsita; print(zarrsita.__version__)"` prints a version.
- `pytest` passes the smoke test (`import zarrsita`, version is a non-empty
  string, `hello()` returns the expected greeting).

## Out of scope (deferred)

- `zarrs` dependency and real array/store bindings.
- `.pyi` type stubs.
- pre-commit hooks, GitHub Actions CI, mkdocs docs, mypy config.
