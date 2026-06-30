"""Phase-0 EFFECTIVENESS guard: every Harden bug must actually fail its module's test.

`test_bug_defs_match.py` proves each bug *injects and changes code*. That is necessary
but not sufficient: a bug that changes code the test doesn't exercise is a "silent bug"
(the design doc's worst case — it hands the learner an unsolvable/undetectable task).

This test closes that gap empirically. For every active cs336 bug definition it:

  1. builds a throwaway sandbox = developer reference `cs336_basics` + the assignment
     test suite + fixtures (a faithful copy of what the validator runs against),
  2. runs the module's REAL pytest node against the CLEAN reference and asserts it PASSES
     (baseline — proves the node/setup is valid), then
  3. injects the bug, runs the SAME node, and asserts it now FAILS (the bug is detected).

It is non-destructive (operates entirely in a temp dir) and marked `slow` because each
case shells out to a real torch-backed pytest run. Deselect with `-m "not slow"`.

The (module -> real pytest node) map is the source of truth the curriculum validators
should agree with; a mismatch here means a validator points at the wrong/te missing test.
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from engine.ast_harden.generic_injector import GenericBugInjector

pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parents[2]
DEV = REPO / "modes" / "developer" / "cs336_basics"
TESTS = REPO / "tests"
MODULES = REPO / "curricula" / "cs336_a1" / "modules"

# module -> the REAL pytest node that exercises that module's reference function.
# Kept explicit (not derived) so a drifting curriculum validator is caught by comparison.
NODE = {
    "softmax": "test_nn_utils.py::test_softmax_matches_pytorch",
    "cross_entropy": "test_nn_utils.py::test_cross_entropy",
    "gradient_clipping": "test_nn_utils.py::test_gradient_clipping",
    "linear": "test_model.py::test_linear",
    "embedding": "test_model.py::test_embedding",
    "silu": "test_model.py::test_silu_matches_pytorch",
    "rmsnorm": "test_model.py::test_rmsnorm",
    "swiglu": "test_model.py::test_swiglu",
    "attention": "test_model.py::test_scaled_dot_product_attention",
    "rope": "test_model.py::test_rope",
    "multihead_attention": "test_model.py::test_multihead_self_attention_with_rope",
    "transformer_block": "test_model.py::test_transformer_block",
    "transformer_lm": "test_model.py::test_transformer_lm",
    "adamw": "test_optimizer.py::test_adamw",
    "cosine_schedule": "test_optimizer.py::test_get_lr_cosine_schedule",
    "data_loader": "test_data.py::test_get_batch",
    "checkpointing": "test_serialization.py::test_checkpointing",
    "bpe_tokenizer": "test_train_bpe.py::test_train_bpe",
    "tokenizer_class": "test_tokenizer.py::test_overlapping_special_tokens",
}

# Assignment test assets to mirror into the sandbox (NOT tests/engine or tests/e2e,
# which would recurse into this very test).
_TEST_FILES = [
    "__init__.py", "adapters.py", "common.py", "conftest.py", "one_d_probes.py",
    "test_data.py", "test_model.py", "test_nn_utils.py", "test_optimizer.py",
    "test_serialization.py", "test_tokenizer.py", "test_train_bpe.py",
]
_TEST_DIRS = ["_snapshots", "fixtures"]


def _defines(src: str, name: str) -> bool:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    return any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
        for n in ast.walk(tree)
    )


def _reference_file_for(target: str) -> Path:
    for p in sorted(DEV.glob("*.py")):
        if _defines(p.read_text(encoding="utf-8"), target):
            return p
    raise AssertionError(f"no developer reference defines '{target}'")


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory) -> Path:
    """A faithful, throwaway copy of {developer cs336_basics + assignment tests}."""
    root = tmp_path_factory.mktemp("bug_effectiveness")
    shutil.copytree(DEV, root / "cs336_basics")
    tdir = root / "tests"
    tdir.mkdir()
    (tdir / "__init__.py").write_text("", encoding="utf-8")
    for name in _TEST_FILES:
        src = TESTS / name
        if src.exists():
            shutil.copy2(src, tdir / name)
    for name in _TEST_DIRS:
        src = TESTS / name
        if src.exists():
            shutil.copytree(src, tdir / name)
    return root


def _run_node(sandbox: Path, node: str) -> tuple[int, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(sandbox) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", f"tests/{node}",
         "--import-mode=importlib", "-q", "-p", "no:cacheprovider", "--no-header"],
        cwd=sandbox, env=env, capture_output=True, text=True, timeout=600,
    )
    return proc.returncode, proc.stdout + proc.stderr


_CASES = sorted(NODE.items())


def test_node_map_covers_all_active_modules():
    """Every active bug module must have a real pytest node mapped here."""
    active = {
        p.parent.parent.name
        for p in MODULES.glob("*/bugs/*.json")
        if not (p.stem.endswith("_v2") or "_draft" in p.stem)
    }
    missing = active - set(NODE)
    assert not missing, f"active modules with no effectiveness node mapped: {sorted(missing)}"


@pytest.mark.parametrize("module,node", _CASES, ids=[m for m, _ in _CASES])
def test_bug_actually_fails_its_test(sandbox: Path, module: str, node: str):
    bug_path = next((MODULES / module / "bugs").glob("*.json"))
    bug = json.loads(bug_path.read_text(encoding="utf-8"))
    ref_name = _reference_file_for(bug["target_function"]).name
    target = sandbox / "cs336_basics" / ref_name
    clean_src = (DEV / ref_name).read_text(encoding="utf-8")

    try:
        # Baseline: clean reference must PASS (proves node + sandbox are valid).
        target.write_text(clean_src, encoding="utf-8")
        rc_clean, out_clean = _run_node(sandbox, node)
        assert rc_clean == 0, f"clean reference failed {node} (setup invalid):\n{out_clean[-1500:]}"

        # Inject the bug; the SAME node must now FAIL (bug is not silent).
        buggy_src, ok = GenericBugInjector(bug).inject(clean_src)
        assert ok, f"bug failed to inject into {ref_name} for module '{module}'"
        target.write_text(buggy_src, encoding="utf-8")
        rc_bug, _ = _run_node(sandbox, node)
        assert rc_bug != 0, (
            f"SILENT BUG: '{bug.get('id')}' injects but {node} still PASSES — "
            f"the learner would get an undetectable/unsolvable Harden challenge."
        )
    finally:
        target.write_text(clean_src, encoding="utf-8")  # always restore the sandbox file
