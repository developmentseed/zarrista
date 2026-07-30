# Development

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

## Formatting

Rust code is formatted with `rustfmt`. We use unstable import sorting options, so you must use nightly to format code.

```bash
rustup toolchain install nightly-2026-06-01

# Format
cargo +nightly-2026-06-01 fmt --all -- --unstable-features --config imports_granularity=Module,group_imports=StdExternalCrate
```

Python code is formatted and linted with `ruff`:

```bash
uv run --no-project ruff check --fix
uv run --no-project ruff format
```

## Docstrings

Docstrings use Google style. We use pydoclint and `mkdocs build --strict` to
enforce this in CI.

```bash
# Validate docstring sections with pydoclint.
uv run --no-project pydoclint python $(find python -name '*.pyi')

# Documented parameters resolve against the API that mkdocstrings renders.
uv run --group docs mkdocs build --strict
```

## Docs Website

```bash
uv run --group docs mkdocs serve
```

## Emscripten Python wheels

Emscripten wheels (PEP 783) are built once per Python version. The entire
toolchain config (Rust toolchain, Emscripten version, ABI tag, rustflags) is
defined by `pyodide-build` *running under that same Python version* — e.g.
Python 3.14 maps to ABI `2026_0`/Emscripten 5.0.3. Use `uvx -p` to query the
config for a given Python version without touching the project venv:

```bash
PYTHON_VERSION=3.14
# The `pyodide` executable lives in pyodide-cli; most subcommands (config,
# xbuildenv) are plugins provided by pyodide-build, so both packages are
# needed.
pyodide_cmd() {
    uvx -p "$PYTHON_VERSION" --from pyodide-cli --with pyodide-build pyodide "$@"
}
PYODIDE_RUST_TOOLCHAIN=$(pyodide_cmd config get rust_toolchain)
PYODIDE_ABI_VERSION=$(pyodide_cmd config get pyodide_abi_version)
PYODIDE_RUSTFLAGS=$(pyodide_cmd config get rustflags)
PYODIDE_CFLAGS=$(pyodide_cmd config get cflags)

echo "PYODIDE_RUST_TOOLCHAIN: $PYODIDE_RUST_TOOLCHAIN"
echo "PYODIDE_ABI_VERSION:    $PYODIDE_ABI_VERSION"
echo "PYODIDE_RUSTFLAGS:      $PYODIDE_RUSTFLAGS"
echo "PYODIDE_CFLAGS:         $PYODIDE_CFLAGS"
```

Install a Rust toolchain and the wasm target.

Note we do **not** use the toolchain Pyodide pins for 3.14 (rustc 1.93): it is
older than zarrista's MSRV (1.94). Force a newer version instead

```bash
RUST_TOOLCHAIN=1.94
rustup toolchain install $RUST_TOOLCHAIN
rustup target add --toolchain $RUST_TOOLCHAIN wasm32-unknown-emscripten
```

Install Emscripten via the Pyodide cross-build environment rather than a
stock emsdk. This pins the Emscripten version matching the target Pyodide ABI
automatically, and applies [Pyodide's patches to
Emscripten](https://github.com/pyodide/pyodide/tree/main/emsdk/patches) —
several of which affect dynamic linking of Rust side modules:

```bash
export PYODIDE_XBUILDENV_PATH="$HOME/.cache/pyodide-xbuildenv"
pyodide_cmd xbuildenv install
pyodide_cmd xbuildenv install-emscripten
source "$PYODIDE_XBUILDENV_PATH/$(pyodide_cmd xbuildenv version)/emsdk/emsdk_env.sh"
```

Build the wheel. Notes on the environment variables:

- `MATURIN_PYEMSCRIPTEN_PLATFORM_VERSION` is required for the wheel to get the
  PyPI-accepted `pyemscripten_*` platform tag instead of the legacy
  `emscripten_x_y_z` tag PyPI rejects (this also needs a recent maturin, hence
  `uvx maturin` rather than the project venv's maturin).
- `CFLAGS_wasm32_unknown_emscripten` is needed for crates that compile C code
  (e.g. zstd-sys, pulled in by zarrs): Pyodide's cflags include `-fPIC`, without which
  the C objects can't be linked into a `SIDE_MODULE` (errors like "relocation
  R_WASM_MEMORY_ADDR_SLEB cannot be used ...; recompile with -fPIC").
- Always build with `--release`: debug builds are ~10x larger (full DWARF) and
  slow.

```bash
RUSTUP_TOOLCHAIN=$RUST_TOOLCHAIN \
CARGO_TARGET_WASM32_UNKNOWN_EMSCRIPTEN_RUSTFLAGS="$PYODIDE_RUSTFLAGS" \
CFLAGS_wasm32_unknown_emscripten="$PYODIDE_CFLAGS" \
MATURIN_PYEMSCRIPTEN_PLATFORM_VERSION=$PYODIDE_ABI_VERSION \
    uvx maturin build \
    --release \
    --no-default-features \
    -o dist \
    --target wasm32-unknown-emscripten \
    -i python$PYTHON_VERSION
```
