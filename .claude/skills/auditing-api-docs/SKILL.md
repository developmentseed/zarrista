---
name: auditing-api-docs
description: Use when preparing a zarrista release, after editing `.pyi` docstrings, or after changing `docs/api/*.md` or the mkdocstrings config — covers type names that render as plain text instead of links, `Returns:` descriptions that repeat their type on every line, `Raises:` entries that do not link, and dead anchors.
---

# Auditing the API docs

## Overview

`mkdocs build --strict` passes even when the docs are visibly broken. It checks
that pages and explicit `[text][target]` references resolve. It does **not**
check the *signature* cross-references that mkdocstrings generates, so a type
that fails to resolve silently degrades from a link to plain text.

The audit therefore reads the built HTML, not the build log.

## When to use

- Before a release.
- After adding or renaming a `.pyi` module, a class, or a type alias.
- After changing a `:::` directive path, `mkdocs.yml`, or the `_zarrista` stubs.
- When a reader reports that a type in a signature is not clickable.

## The audit

```bash
uv run --no-project mkdocs build --strict
uv run --no-project python .claude/skills/auditing-api-docs/check_docs.py
uv run --no-project python .claude/skills/auditing-api-docs/check_docs.py --urls  # network
```

The script exits non-zero if anything fails, and prints a diagnosis per finding.
It runs four checks:

| Check | Reads | Finds |
| --- | --- | --- |
| unresolved annotations | `site/` | `<span title="X">` where a link belongs |
| broken anchors | `site/` | `href="...#frag"` with no matching `id` |
| under-indented `Returns:` | `python/` | wrapped descriptions griffe splits into items |
| external URLs (`--urls`) | docs, stubs, README | non-200 responses |

## Fixing an unresolved annotation

Every failure is a **path mismatch**: griffe resolved the name to path A, and the
`:::` directive created an anchor at path B. Find the two paths, then make them
agree. The four causes, in the order they usually appear:

**1. The symbol is documented at a different path.** Example: annotations
resolved to `zarrista._tensor.Tensor` while tensor.md documented
`zarrista.Tensor`. Fix by removing the duplication — define the symbol once and
import it everywhere else. `SyncStore` and `AsyncStore` were defined verbatim in
both `store.py` and `_store.pyi`; deleting the stub copies and importing from
`zarrista.store` fixed 32 references and removed a docstring that could drift.

**2. Nothing documents the symbol.** Add a `:::` block. If the symbol is
stub-only (`Selection`, `DataTypeName`), the directive must name the private
path, so add `heading:` and `toc_label:` overrides to show the public name.

**3. A bare name in a `Raises:` or `Returns:` section.** Griffe resolves
docstring names in the *enclosing module's* scope. `Raises: ArrayError:` in
`_array.pyi` resolves only if that stub imports `ArrayError`:

```python
# Imported only so that the `Raises:` sections below link to the exception docs.
from zarrista.exceptions import (  # noqa: F401
    ArrayCreateError,
    ArrayError,
    StorageError,
)
```

The `noqa` is required: `select = ["ALL"]` flags the unused import. Do not
qualify the name in the docstring instead — the rendered text would show the
whole dotted path.

`from zarrista.exceptions import *  # noqa: F403` also resolves, and needs no
edit when a `Raises:` entry changes. We keep the explicit list anyway: a star
import in a stub re-exports every name, and the list doubles as a record of
which exceptions the module raises. The audit, not hand-maintenance, is what
keeps the list correct.

**4. An external type with no inventory.** Add that project's `objects.inv` to
`inventories:` in mkdocs.yml. `typing_extensions.CapsuleType` needed
`https://typing-extensions.readthedocs.io/en/latest/objects.inv`.

### When a public alias cannot resolve

Griffe resolves aliases statically, and `allow_inspection: false` forbids the
import fallback. So a name re-exported from the compiled extension resolves only
if the stub tree spells out every hop. `zarrista.codec.CodecChain` →
`zarrista._zarrista.codec.CodecChain` → dead end, until `_zarrista` became a stub
*package* (`_zarrista/__init__.pyi` plus `_zarrista/codec/__init__.pyi`).

A new compiled submodule needs a matching stub module, or nothing in it can be
documented at the path users import. A stub-only directory beside the `.so` is
safe: the extension loader wins over the namespace-package portion, the wheel
ships both, and pyright reads the stubs.

## Fixing a `Returns:` section

Griffe's Google parser starts a new item at each line indented to the body level,
so a wrapped description renders its type prefix once per line. Indent
continuation lines one more level:

```python
Returns:
    The shape of the decoded chunk, or `None` if the codec cannot
        determine it.
```

`docstring_options: {returns_multiple_items: false}` in mkdocs.yml would disable
the splitting globally, at the cost of never being able to document a tuple's
elements separately.

## Verifying a change

```bash
uv run --no-project mkdocs build --strict   # 0 errors
uv run --no-project python .claude/skills/auditing-api-docs/check_docs.py
uv run --no-project pydoclint python $(find python -name '*.pyi')
uv run --no-project ruff check python && uv run --no-project ruff format --check python
uv run --no-project pytest -q                # stub edits can break the runtime package
uvx pyright --pythonpath .venv/bin/python <a file that imports the changed types>
```

Prefer a scratch build when trying an idea: copy `docs/` and `python/` to a
temporary directory, point a copied `mkdocs.yml` at them (`docs_dir`, `paths`),
and build with `-d`. That keeps the repo clean while you compare renderings.

## Common mistakes

- **Trusting a green `--strict` build.** It does not check generated signature
  cross-references. Read the HTML.
- **Assuming a check works because it reports zero.** Inject a known break and
  confirm the check reports it. The anchor check in this skill's script returned
  a clean 0 for exactly this reason before a path-comparison bug was fixed.
- **Aliasing the anchor instead of removing the duplication.** An explicit
  `[](){#some.path}` anchor before a heading does work, and it is the only option
  when you cannot change the code. It leaves two sources of truth.
- **Adding anchors to an auto-rendered module page.** With one `:::` block for a
  whole module there is no per-symbol heading to attach an anchor to, so every
  alias lands at the same place on the page.
- **Letting ruff's isort strand an explanatory comment.** `ruff check --fix`
  moves the import and leaves the comment where it was. Re-read the file.
