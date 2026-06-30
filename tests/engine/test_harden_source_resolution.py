"""Guards for Harden source-file resolution.

The CLI used to hardcode the Harden target file as ``{module_id}.py`` (only softmax
was wired correctly), so ``start-challenge`` failed for every other module. The engine
now resolves the file from each module's ``metadata['source_file']``. These tests pin:

  1. ``_harden_source_relpath`` reads the curriculum-declared source file (and falls
     back sensibly for packs that predate the field), and
  2. every active cs336 module's declared ``source_file`` actually DEFINES that module's
     bug ``target_function`` — i.e. the manifest, the bug definition, and the reference
     implementation all agree. If they drift, Harden would inject into the wrong file.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from engine.main import _harden_source_relpath
from engine.schemas import ModuleMetadata

REPO = Path(__file__).resolve().parents[2]
DEV = REPO / "modes" / "developer" / "cs336_basics"
STU = REPO / "modes" / "student" / "cs336_basics"
MODULES = REPO / "curricula" / "cs336_a1" / "modules"
MANIFEST = REPO / "curricula" / "cs336_a1" / "manifest.json"


def test_resolver_uses_declared_source_file():
    mod = ModuleMetadata(id="rmsnorm", name="RMSNorm", path="modules/rmsnorm",
                         metadata={"source_file": "cs336_basics/layers.py"})
    assert _harden_source_relpath(mod) == Path("cs336_basics/layers.py")


def test_resolver_legacy_fallback():
    # Packs without metadata fall back: softmax -> utils.py, others -> <id>.py.
    softmax = ModuleMetadata(id="softmax", name="Softmax", path="modules/softmax")
    other = ModuleMetadata(id="hello", name="Hello", path="modules/hello")
    assert _harden_source_relpath(softmax) == Path("cs336_basics/utils.py")
    assert _harden_source_relpath(other) == Path("hello.py")


def _defines(path: Path, name: str) -> bool:
    if not path.exists():
        return False
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return False
    return any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
        for n in ast.walk(tree)
    )


def _active_modules_with_bug() -> list[str]:
    out = []
    for jp in sorted(MODULES.glob("*/bugs/*.json")):
        if jp.stem.endswith("_v2") or "_draft" in jp.stem:
            continue
        out.append(jp.parent.parent.name)
    return sorted(set(out))


_MANIFEST = json.loads(MANIFEST.read_text(encoding="utf-8"))
_SOURCE_BY_ID = {m["id"]: m.get("metadata", {}).get("source_file") for m in _MANIFEST["modules"]}


@pytest.mark.parametrize("module", _active_modules_with_bug())
def test_manifest_source_file_defines_bug_target(module: str):
    declared = _SOURCE_BY_ID.get(module)
    assert declared, f"module '{module}' has a bug def but no metadata.source_file in the manifest"

    bug = json.loads(next((MODULES / module / "bugs").glob("*.json")).read_text(encoding="utf-8"))
    target = bug["target_function"]

    fname = Path(declared).name
    # Prefer the developer reference; fall back to the student build target (e.g. generation.py).
    candidates = [DEV / fname, STU / fname]
    assert any(_defines(p, target) for p in candidates), (
        f"manifest source_file '{declared}' for module '{module}' does not define its "
        f"bug target_function '{target}' — manifest/bug-def/reference are out of sync"
    )
