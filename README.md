# zarrista

A low-level Zarr API for Python, inspired by [zarrita.js], powered from Rust by [Zarrs]. Serving up Zarr chunks like your favorite barista!

[zarrita.js]: https://zarrita.dev/
[Zarrs]: https://zarrs.dev/

This has been _minimally_ vibe-coded: _mostly_ but not fully written by hand. Some areas were prototyped with Claude.

## Development

Requires a Rust toolchain and Python 3.11+. We use
[uv](https://docs.astral.sh/uv/) and [maturin](https://www.maturin.rs/).

```bash
# Create a dev environment and install the dev dependencies
uv sync --no-install-package zarrista

# Build the Rust extension and install it into the environment (debug build)
uv run --no-project maturin develop --uv

# Or, in release mode:
uv run --no-project maturin develop --uv --release

# Run the tests
uv run --no-project pytest
```

The `--no-project` is annoying but unavoidable in our current setup. Otherwise `uv` will try to build the rust library _in release mode, as a dependency of the project_ before reaching `uv sync` or `uv run`.
