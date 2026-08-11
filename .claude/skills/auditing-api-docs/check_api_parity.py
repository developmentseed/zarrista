#!/usr/bin/env python3
"""Compare every `.pyi` signature against the signature the extension exposes.

Nothing else catches a drift between the two: pydoclint checks a docstring
against its own stub, mkdocs checks references, and pytest only covers what the
tests call. A stub can therefore promise a parameter the Rust never accepts.

Usage:
    uv run --no-project python .claude/skills/auditing-api-docs/check_api_parity.py
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib
import sys

MODULES = ["zarrista", "zarrista.store", "zarrista.codec", "zarrista.exceptions"]
STUBS = pathlib.Path("python/zarrista")

KIND = {
    inspect.Parameter.POSITIONAL_ONLY: "pos-only",
    inspect.Parameter.POSITIONAL_OR_KEYWORD: "pos-or-kw",
    inspect.Parameter.KEYWORD_ONLY: "kw-only",
    inspect.Parameter.VAR_POSITIONAL: "*args",
    inspect.Parameter.VAR_KEYWORD: "**kwargs",
}


def public_objects() -> dict[str, object]:
    """Map every public name in the package to the object it names."""
    found: dict[str, object] = {}
    for name in MODULES:
        module = importlib.import_module(name)
        for attribute in dir(module):
            if not attribute.startswith("_"):
                found.setdefault(attribute, getattr(module, attribute))
    return found


def stub_parameters(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[tuple[str, str]]:
    """Return `(name, kind)` for each parameter the stub declares."""
    args = fn.args
    params = [(p.arg, "pos-only") for p in args.posonlyargs]
    params += [(p.arg, "pos-or-kw") for p in args.args]
    if args.vararg:
        params.append((args.vararg.arg, "*args"))
    params += [(p.arg, "kw-only") for p in args.kwonlyargs]
    if args.kwarg:
        params.append((args.kwarg.arg, "**kwargs"))
    return [p for p in params if p[0] != "self"]


def runtime_parameters(member: object) -> list[tuple[str, str]] | None:
    """Return `(name, kind)` per parameter, or `None` when pyo3 exposes none."""
    try:
        signature = inspect.signature(member)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    params = [(n, KIND[p.kind]) for n, p in signature.parameters.items() if n != "self"]
    if params and params[0][0] in {"args", "kwargs"}:
        return None  # an opaque `(*args, **kwargs)` carries no information
    return params


def compare_class(cls: ast.ClassDef, target: object) -> int:
    """Report each method of `cls` whose stub disagrees with `target`."""
    issues = 0
    for fn in cls.body:
        if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        decorators = fn.decorator_list
        if any(isinstance(d, ast.Name) and d.id == "property" for d in decorators):
            continue
        # Slot wrappers report the slot's own parameter names (`key`, `value`),
        # which never match ours, and `__dlpack__` deliberately spells out
        # keywords that Rust takes as `**kwargs`.
        if fn.name.startswith("__") and fn.name != "__init__":
            continue
        attribute = "__new__" if fn.name == "__init__" else fn.name
        member = getattr(target, attribute, None)
        if member is None:
            print(f"MISSING  {cls.name}.{fn.name}: in the stub, absent at runtime")
            issues += 1
            continue
        runtime = runtime_parameters(member)
        if runtime is None:
            continue
        stub = stub_parameters(fn)
        if stub != runtime:
            print(f"DIFF     {cls.name}.{fn.name}")
            print(f"           stub:    {stub}")
            print(f"           runtime: {runtime}")
            issues += 1
    return issues


def main() -> int:
    """Compare every stub signature with its runtime counterpart."""
    objects = public_objects()
    issues = 0
    for path in sorted(STUBS.rglob("*.pyi")):
        tree = ast.parse(path.read_text())
        for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            target = objects.get(cls.name)
            if target is not None:
                issues += compare_class(cls, target)
    print(f"signature mismatches: {issues}")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
