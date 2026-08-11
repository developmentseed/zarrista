#!/usr/bin/env python3
"""Audit the rendered API docs for broken cross-references and mis-parsed sections.

Run `mkdocs build` first: three of the four checks read `site/`.

Usage:
    uv run --no-project python .claude/skills/auditing-api-docs/check_docs.py
    uv run --no-project python .claude/skills/auditing-api-docs/check_docs.py --urls
"""

from __future__ import annotations

import argparse
import collections
import itertools
import pathlib
import re
import sys
import urllib.parse

HTTP_OK = 200
SITE = pathlib.Path("site")
SOURCES = pathlib.Path("python")


def unresolved_annotations() -> int:
    """Report annotations that mkdocstrings rendered as text instead of a link.

    mkdocstrings emits `<span title="...">` when it cannot find an anchor for a
    name, and `<a href="...">` when it can.
    """
    counts: collections.Counter[str] = collections.Counter()
    pages = collections.defaultdict(set)
    anchors: collections.defaultdict[str, set[str]] = collections.defaultdict(set)
    for page in SITE.rglob("index.html"):
        text = page.read_text()
        for match in re.finditer(r'<span title="([^"]+)"', text):
            counts[match.group(1)] += 1
            pages[match.group(1)].add(str(page.parent.relative_to(SITE)))
        for anchor in re.findall(r'\sid="(zarrista[^"]+)"', text):
            anchors[anchor.rsplit(".", 1)[-1]].add(anchor)

    if not counts:
        print("unresolved annotations: none")
        return 0

    print(f"unresolved annotations: {sum(counts.values())}")
    for name, count in counts.most_common():
        print(f"  {count:>4}  {name:<52} {', '.join(sorted(pages[name]))}")
        print(f"        -> {_diagnose(name, anchors)}")
    return 1


def _diagnose(name: str, anchors: dict[str, set[str]]) -> str:
    """Name the likely cause of one unresolved annotation.

    The decisive question is whether the symbol is documented *somewhere* under a
    different path, or not documented at all.
    """
    if "." not in name and not anchors.get(name):
        return "bare name in a docstring section, and no page documents it"
    if "." not in name:
        documented = ", ".join(sorted(anchors[name]))
        return f"bare name in a docstring section; import {documented} in the stub"
    if not name.startswith("zarrista"):
        return "external type; add its objects.inv to `inventories` in mkdocs.yml"

    elsewhere = sorted(anchors.get(name.rsplit(".", 1)[-1], set()))
    if elsewhere:
        return f"documented instead at {', '.join(elsewhere)}; make the two paths agree"
    return "no page documents this symbol; add a `:::` block for it"


def broken_anchors() -> int:
    """Report internal links whose `#fragment` has no matching id."""
    html = {p.resolve(): p.read_text() for p in SITE.rglob("*.html")}
    ids = {p: set(re.findall(r'\sid="([^"]+)"', text)) for p, text in html.items()}

    broken: collections.Counter[str] = collections.Counter()
    for page, text in html.items():
        for href in re.findall(r'href="([^"]+)"', text):
            if href.startswith(("http", "mailto:", "data:")) or "#" not in href:
                continue
            path, _, fragment = href.partition("#")
            fragment = urllib.parse.unquote(fragment)
            if not fragment:
                continue
            target = page.parent if path in ("", ".") else (page.parent / path)
            if target.is_dir():
                target = target / "index.html"
            target = target.resolve()
            if target not in ids:
                continue  # a link out of the docs tree, or a generated theme page
            if fragment not in ids[target]:
                rel = target.relative_to(SITE.resolve())
                source = page.relative_to(SITE.resolve())
                broken[f"#{fragment} missing in {rel} (linked from {source})"] += 1

    print(f"broken anchors: {sum(broken.values())}")
    for entry, count in broken.most_common():
        print(f"  {count:>4}  {entry}")
    return 1 if broken else 0


def underindented_returns() -> int:
    """Report `Returns:`/`Yields:` bodies that griffe reads as several items.

    Griffe's Google parser starts a new item at every line indented to the body
    level, so a wrapped description repeats its type prefix once per line.
    Continuation lines need one more indent level.
    """
    hits = []
    for path in sorted(itertools.chain(SOURCES.rglob("*.pyi"), SOURCES.rglob("*.py"))):
        lines = path.read_text().split("\n")
        for index, line in enumerate(lines):
            header = re.match(r"^(\s*)(Returns|Yields):\s*$", line)
            if not header:
                continue
            base = len(header.group(1))
            body_indent = None
            for offset, current in enumerate(lines[index + 1 :], start=index + 2):
                if not current.strip():
                    continue
                indent = len(current) - len(current.lstrip())
                if indent <= base:
                    break
                if body_indent is None:
                    body_indent = indent
                elif indent == body_indent:
                    hits.append(f"{path}:{offset}  {current.strip()[:60]}")

    print(f"under-indented Returns/Yields continuation lines: {len(hits)}")
    for hit in hits:
        print(f"  {hit}")
    return 1 if hits else 0


def external_urls() -> int:
    """Report documentation URLs that do not answer with 200."""
    import urllib.error
    import urllib.request

    urls = set()
    files = itertools.chain(
        pathlib.Path("docs").rglob("*.md"),
        SOURCES.rglob("*.pyi"),
        [pathlib.Path("README.md")],
    )
    for path in files:
        urls |= set(re.findall(r"https?://[^\s)>\"'`]+", path.read_text()))

    bad = []
    for url in sorted(urls):
        headers = {"User-Agent": "zarrista-docs-audit"}
        request = urllib.request.Request(url, headers=headers)  # noqa: S310
        try:
            with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310
                if response.status != HTTP_OK:
                    bad.append(f"{response.status}  {url}")
        except (urllib.error.URLError, OSError, ValueError) as error:
            bad.append(f"ERR   {url}  ({error})")

    print(f"external URLs checked: {len(urls)}, failing: {len(bad)}")
    for entry in bad:
        print(f"  {entry}")
    return 1 if bad else 0


def main() -> int:
    """Run every check and return a process exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--urls",
        action="store_true",
        help="also check external URLs (network)",
    )
    args = parser.parse_args()

    if not SITE.is_dir():
        print("site/ not found: run `uv run --no-project mkdocs build --strict` first")
        return 1

    status = 0
    for check in (unresolved_annotations, broken_anchors, underindented_returns):
        status |= check()
        print()
    if args.urls:
        status |= external_urls()
    return status


if __name__ == "__main__":
    sys.exit(main())
