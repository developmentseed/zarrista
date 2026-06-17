# zarrista

A low-level Zarr API for Python, inspired by [zarrita.js], powered from Rust by [Zarrs].

[zarrita.js]: https://zarrita.dev/
[Zarrs]: https://zarrs.dev/

This has been _minimally_ vibe-coded (Claude still writes bad Rust code in my opinion).

## Custom stores

Besides the built-in `FilesystemStore` and `MemoryStore`, you can pass any
duck-typed Python object as a (synchronous) store. The minimal contract is a
single `get` method plus two capability properties:

```python
from zarrista import Array


class DictStore:
    supports_get_partial = False  # opt into byte-range reads
    supports_listing = False      # opt into listing keys/prefixes

    def __init__(self, mapping: dict[str, bytes]):
        self._mapping = mapping

    def get(self, key: str) -> bytes | None:
        return self._mapping.get(key)


array = Array.open(DictStore(my_bytes), "/path")
```

Declare `supports_listing = True` and implement `list`, `list_prefix`,
`list_dir`, and `size_prefix` to support operations like `Group.array_keys()`;
calling a listing operation on a store that does not support it raises an error.
Declare `supports_get_partial = True` and implement `get_partial_many` to serve
efficient byte-range reads (otherwise partial reads fall back to fetching the
whole value and slicing). The `zarrista.ReadableStore` and
`zarrista.ListableStore` protocols document the full surface.

> Note: if your store defines a method named `list`, add
> `from __future__ import annotations` to the module so later `list[...]` type
> annotations are not shadowed by the method.

This is sync-only; for async use pass an `obstore.ObjectStore`.

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
