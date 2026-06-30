#!/usr/bin/env python3
"""Meta-test harness: verify every active cs336 bug definition injects against the reference.

For each bug def under curricula/cs336_a1/modules/*/bugs/*.json (optionally excluding
*_draft / *_v2), resolve the reference source file that defines the def's
`target_function` (preferring developer reference, falling back to the student build
target), run GenericBugInjector, and report whether injection succeeds.

A bug def is HEALTHY iff the injector returns success (target found AND a pattern matched).
This is the Phase-0 "silent bug" guard for the Harden stage.

Usage:
    python scripts/check_bug_defs.py            # active defs only (skip _draft/_v2)
    python scripts/check_bug_defs.py --all      # include drafts/v2
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

# Make `engine` importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.ast_harden.generic_injector import GenericBugInjector  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEV = REPO / "modes" / "developer" / "cs336_basics"
STU = REPO / "modes" / "student" / "cs336_basics"
MODULES = REPO / "curricula" / "cs336_a1" / "modules"


def reference_files() -> list[Path]:
    """Developer reference files first, then student-only files (e.g. generation.py)."""
    ordered = sorted(DEV.glob("*.py"))
    for p in sorted(STU.glob("*.py")):
        if not (DEV / p.name).exists():
            ordered.append(p)
    return ordered


def _defines(src: str, name: str) -> bool:
    """True if src defines a function/method named `name` (methods count via ast.walk)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    return any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
        for n in ast.walk(tree)
    )


def resolve_reference(target: str) -> tuple[Path | None, str | None]:
    for p in reference_files():
        src = p.read_text(encoding="utf-8")
        if _defines(src, target):
            return p, src
    return None, None


def is_active(path: Path) -> bool:
    stem = path.stem
    return not (stem.endswith("_draft") or stem.endswith("_v2") or "_draft" in stem)


def check(path: Path) -> tuple[str, str]:
    """Return (status, detail). status in INJECTED_OK / PATTERN_MISS / NO_TARGET_FN / DEF_ERROR."""
    try:
        bug = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return "DEF_ERROR", f"json: {e}"
    target = bug.get("target_function")
    if not target:
        return "DEF_ERROR", "no target_function"
    ref, src = resolve_reference(target)
    if ref is None:
        return "NO_TARGET_FN", f"'{target}' not defined in any reference file"
    try:
        _, ok = GenericBugInjector(bug).inject(src)
    except Exception as e:  # noqa: BLE001
        return "DEF_ERROR", f"{type(e).__name__}: {e}"
    if not ok:
        return "PATTERN_MISS", f"target '{target}' in {ref.name}, but no node matched"
    return "INJECTED_OK", f"{target} in {ref.name}"


def collect(include_all: bool) -> list[tuple[str, Path]]:
    out = []
    for jp in sorted(MODULES.glob("*/bugs/*.json")):
        if include_all or is_active(jp):
            out.append((jp.parent.parent.name, jp))
    return out


def main(argv: list[str]) -> int:
    include_all = "--all" in argv
    rows = []
    for module, jp in collect(include_all):
        status, detail = check(jp)
        rows.append((module, jp.name, status, detail))

    width = max((len(m) for m, *_ in rows), default=6)
    print(f"{'MODULE':<{width}}  {'BUG FILE':<34}  {'STATUS':<13}  DETAIL")
    print("-" * (width + 90))
    for module, name, status, detail in rows:
        print(f"{module:<{width}}  {name:<34}  {status:<13}  {detail}")
    print("-" * (width + 90))

    from collections import Counter

    tally = Counter(r[2] for r in rows)
    ok = tally.get("INJECTED_OK", 0)
    print(f"SUMMARY: {dict(tally)}  |  healthy={ok}/{len(rows)}")
    # Exit non-zero if any active def is unhealthy.
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
