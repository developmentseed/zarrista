# zarrista

A small, prototypical zarrita-like Python Zarr implementation on top of
[zarrs](https://github.com/LDeakin/zarrs).

> **Status:** early prototype. This is currently a hello-world pyo3 scaffold;
> the `zarrs` bindings are not implemented yet.

## Development

Requires a Rust toolchain and Python 3.11+. We use [uv](https://docs.astral.sh/uv/)
and [maturin](https://www.maturin.rs/).

```bash
# Create a dev environment and install the dev dependencies
uv sync

# Build the Rust extension and install it into the environment (debug build)
uv run maturin develop

# Run the tests
uv run pytest
```

Quick check that it imports:

```bash
uv run python -c "import zarrista; print(zarrista.__version__, zarrista.hello())"
```

## Layout

- `src/lib.rs` — the Rust extension module, compiled to `zarrista._zarrista`.
- `python/zarrista/` — the pure-Python package that re-exports the compiled module.
- `tests/` — pytest smoke tests.
