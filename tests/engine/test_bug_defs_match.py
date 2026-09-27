"""Phase-0 meta-test: every active cs336 Harden bug definition must inject cleanly.

This is the regression guard for the Harden stage. For each active bug definition
(canonical defs; drafts/v2 are excluded), it:

  1. resolves the reference source that defines the def's `target_function`
     (developer reference preferred, student build-target as fallback),
  2. runs GenericBugInjector and asserts the injection SUCCEEDS (target found AND a
     node matched — no NO_TARGET_FN / PATTERN_MISS), and
  3. asserts a matching `<stem>_symptom.txt` exists (required by HardenRunner._select_bug).

If this test fails, the Harden stage is broken for that module: `mastery start-challenge`
would either crash or silently fail to inject a bug.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from engine.ast_harden.generic_injector import GenericBugInjector

REPO = Path(__file__).resolve().parents[2]
DEV = REPO / "modes" / "developer" / "cs336_basics"
STU = REPO / "modes" / "student" / "cs336_basics"
MODULES = REPO / "curricula" / "cs336_a1" / "modules"


def _reference_files() -> list[Path]:
    ordered = sorted(DEV.glob("*.py"))
    for p in sorted(STU.glob("*.py")):
        if not (DEV / p.name).exists():
            ordered.append(p)
    return ordered


def _defines(src: str, name: str) -> bool:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    return any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
        for n in ast.walk(tree)
    )


def _resolve_reference(target: str) -> tuple[Path | None, str | None]:
    for p in _reference_files():
        src = p.read_text(encoding="utf-8")
        if _defines(src, target):
            return p, src
    return None, None


def _is_active(path: Path) -> bool:
    stem = path.stem
    return not (stem.endswith("_v2") or "_draft" in stem)


def _active_bug_defs() -> list[Path]:
    return [p for p in sorted(MODULES.glob("*/bugs/*.json")) if _is_active(p)]


_ACTIVE = _active_bug_defs()


def test_active_bug_defs_discovered():
    """Guard against a glob/path regression silently collecting zero defs."""
    assert len(_ACTIVE) >= 19, f"expected >=19 active bug defs, found {len(_ACTIVE)}"


@pytest.mark.parametrize("bug_path", _ACTIVE, ids=lambda p: p.parent.parent.name)
def test_bug_def_injects(bug_path: Path):
    bug = json.loads(bug_path.read_text(encoding="utf-8"))
    target = bug.get("target_function")
    assert target, f"{bug_path} has no target_function"

    ref_file, src = _resolve_reference(target)
    assert ref_file is not None, (
        f"NO_TARGET_FN: '{target}' is not defined in any reference file "
        f"(module {bug_path.parent.parent.name})"
    )

    buggy, ok = GenericBugInjector(bug).inject(src)
    assert ok, (
        f"PATTERN_MISS: target '{target}' found in {ref_file.name} but no node matched "
        f"(module {bug_path.parent.parent.name})"
    )
    # A successful injection must actually change the code.
    assert ast.dump(ast.parse(buggy)) != ast.dump(ast.parse(src)), (
        f"injection reported success but produced identical code ({bug_path.parent.parent.name})"
    )


@pytest.mark.parametrize("bug_path", _ACTIVE, ids=lambda p: p.parent.parent.name)
def test_bug_def_has_symptom(bug_path: Path):
    symptom = bug_path.parent / f"{bug_path.stem}_symptom.txt"
    assert symptom.exists(), (
        f"HardenRunner._select_bug requires {symptom.name} next to {bug_path.name}"
    )
    assert symptom.read_text(encoding="utf-8").strip(), f"{symptom.name} is empty"
