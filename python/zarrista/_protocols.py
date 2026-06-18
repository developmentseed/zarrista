"""Typing protocols for custom, duck-typed sync stores.

A custom store is any Python object satisfying :class:`ReadableStore` (and
optionally :class:`ListableStore`). Pass one anywhere a built-in store is
accepted, e.g. ``Array.open(my_store, "/path")``.

Capabilities are declared with ``@property`` predicates and read once when the
store is wrapped. ``get`` is the only required method.

Two methods are *optional* and consulted only when ``supports_get_partial`` is
true (they are intentionally not part of the runtime-checkable protocol, so a
minimal ``get``-only store still satisfies :class:`ReadableStore`):

- ``get_partial_many(key, ranges) -> list[bytes] | None`` where each range is a
  ``(kind, offset, length)`` triple. ``kind`` is ``"start"`` (read ``length``
  bytes from ``offset``, or to the end when ``length`` is ``None``) or
  ``"suffix"`` (read the last ``length`` bytes). When absent, partial reads
  fall back to fetching the whole value and slicing.
- ``size_key(key) -> int | None``. When absent, the size falls back to
  ``len(get(key))``.
"""

from __future__ import annotations

import builtins
from typing import Protocol, runtime_checkable


@runtime_checkable
class ReadableStore(Protocol):
    """A duck-typed, readable sync store. ``get`` is the only required method."""

    @property
    def supports_get_partial(self) -> bool: ...

    @property
    def supports_listing(self) -> bool: ...

    def get(self, key: str) -> bytes | None: ...


@runtime_checkable
class ListableStore(ReadableStore, Protocol):
    """A readable store that also supports listing keys and prefixes."""

    def list(self) -> builtins.list[str]: ...

    def list_prefix(self, prefix: str) -> builtins.list[str]: ...

    def list_dir(self, prefix: str) -> dict[str, builtins.list[str]]: ...

    def size_prefix(self, prefix: str) -> int: ...
