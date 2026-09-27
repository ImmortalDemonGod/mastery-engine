"""
RED tests for curricula/cs336_a1/modules/cosine_schedule/validator.sh

Each test names the bug it catches (design-tests contract).
These tests MUST FAIL against the current broken validator and pass after the fix.

Bugs targeted:
  B1 — validator.sh:18 copies optimizer.py instead of utils.py
       (get_lr_cosine_schedule lives in cs336_basics/utils.py, not optimizer.py)
  B2 — validator.sh:33/37/40 uses pytest node ID ::test_lr_cosine_schedule
       but the actual function in tests/test_optimizer.py:52 is test_get_lr_cosine_schedule

See tests/test_cosine_schedule_validator.bug-catalog.md for full reasoning.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
VALIDATOR_PATH = REPO_ROOT / "curricula" / "cs336_a1" / "modules" / "cosine_schedule" / "validator.sh"
DEV_CS336 = REPO_ROOT / "modes" / "developer" / "cs336_basics"


# ── Unit tests: static content inspection of validator.sh ─────────────────────


def test_validator_sh_copies_utils_py_not_optimizer_py():
    """
    Validator BUILD-stage cp must target utils.py — guards against wrong-file-in-cp-command (Bug B1).

    get_lr_cosine_schedule is defined in cs336_basics/utils.py (confirmed: tests/adapters.py:15
    imports it from cs336_basics.utils). The cp on validator.sh:18 currently reads:
        cp cs336_basics/optimizer.py "$SHADOW_WORKTREE/cs336_basics/optimizer.py"
    so the student's utils.py never propagates to the shadow worktree.
    """
    content = VALIDATOR_PATH.read_text()

    # After fix, this line must be present:
    assert 'cp cs336_basics/utils.py "$SHADOW_WORKTREE/cs336_basics/utils.py"' in content, (
        "validator.sh:18 copies optimizer.py instead of utils.py; "
        "get_lr_cosine_schedule is defined in cs336_basics/utils.py (tests/adapters.py:15) "
        "so utils.py must be copied to propagate the student's implementation"
    )


def test_validator_sh_pytest_node_id_matches_test_get_lr_cosine_schedule():
    """
    Validator pytest invocation must reference test_get_lr_cosine_schedule — guards against
    test-name-mismatch-silent-skip (Bug B2).

    tests/test_optimizer.py:52 defines 'def test_get_lr_cosine_schedule():'.
    validator.sh:33/37/40 currently pass node '::test_lr_cosine_schedule' (missing 'get_' prefix).
    pytest finds no matching test and exits with code 4/5 (non-zero), so the validator
    always fails even when the student's implementation is correct.
    """
    content = VALIDATOR_PATH.read_text()

    assert "test_get_lr_cosine_schedule" in content, (
        "validator.sh uses pytest node ID ::test_lr_cosine_schedule, but the actual function "
        "in tests/test_optimizer.py:52 is test_get_lr_cosine_schedule (with 'get_' prefix); "
        "pytest collects 0 tests and exits non-zero"
    )


# ── Integration test: file-propagation check (BUILD stage) ────────────────────


@pytest.fixture()
def shadow_worktree_with_stale_utils(tmp_path):
    """
    Shadow worktree with a stale utils.py.

    Simulates the state before the student's submission is propagated:
    cs336_basics/utils.py is intentionally wrong so we can detect whether
    the validator overwrites it.
    """
    shadow = tmp_path / "shadow"
    cs336_dir = shadow / "cs336_basics"
    cs336_dir.mkdir(parents=True)

    # Stale sentinel — must be overwritten by the validator after the fix
    stale_sentinel = "# STALE — validator must overwrite this with student utils.py\n"
    (cs336_dir / "utils.py").write_text(stale_sentinel)
    (cs336_dir / "__init__.py").write_text("")

    return shadow, stale_sentinel


def test_validator_propagates_utils_py_to_shadow_worktree(shadow_worktree_with_stale_utils):
    """
    Validator BUILD stage must copy cs336_basics/utils.py into SHADOW_WORKTREE — guards against
    wrong-file-in-cp-command (Bug B1, integration layer).

    After the fix: running the validator from the developer reference directory updates the
    shadow worktree's cs336_basics/utils.py (overwriting the stale sentinel).
    Currently: only optimizer.py is copied; utils.py remains stale.
    """
    shadow, stale_sentinel = shadow_worktree_with_stale_utils

    # Run validator from the developer reference directory.
    # modes/developer/cs336_basics/utils.py is the correct reference implementation;
    # this simulates "developer mode" (student has implemented get_lr_cosine_schedule).
    result = subprocess.run(
        [str(VALIDATOR_PATH)],
        cwd=str(DEV_CS336.parent),  # modes/developer/ — cs336_basics/ subdir is accessible
        env={
            **os.environ,
            "SHADOW_WORKTREE": str(shadow),
            "MASTERY_PYTHON": sys.executable,
        },
        capture_output=True,
        text=True,
    )

    # The validator may exit non-zero (e.g., because tests/ is missing in shadow or due to
    # Bug B2's test-name mismatch) — that is expected at this stage. What we assert here is
    # solely that the file-propagation step (the cp command) ran for utils.py.
    actual_utils = (shadow / "cs336_basics" / "utils.py").read_text()
    assert actual_utils != stale_sentinel, (
        "validator.sh must copy cs336_basics/utils.py to $SHADOW_WORKTREE/cs336_basics/utils.py "
        "in the BUILD stage (Bug B1: currently only optimizer.py is copied, leaving utils.py "
        f"stale). validator exit={result.returncode}; stderr={result.stderr[:300]}"
    )

    # Additionally assert that the propagated content matches the developer reference.
    dev_utils_content = (DEV_CS336 / "utils.py").read_text()
    assert actual_utils == dev_utils_content, (
        "After propagation, shadow worktree's utils.py must match the source "
        f"(cs336_basics/utils.py in cwd). Got {len(actual_utils)} chars, "
        f"expected {len(dev_utils_content)} chars."
    )
