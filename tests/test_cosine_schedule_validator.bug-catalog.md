# Bug Catalog: `curricula/cs336_a1/modules/cosine_schedule/validator.sh`

**Session**: design-tests / mastery-cosine-validator-tests  
**Target**: `curricula/cs336_a1/modules/cosine_schedule/validator.sh`  
**Canonical intent**: https://github.com/ImmortalDemonGod/mastery-engine/blob/7f6610a902befcb84fc47e5c82a161e3d3184ce4/audit/02-static-audit.md#L8  
**Adjacent test file**: `tests/test_cosine_schedule_validator.py`

---

## Code Summary

**Public interface**: `validator.sh` is a Bash script invoked by the ValidationSubsystem with
`SHADOW_WORKTREE` and `MASTERY_PYTHON` env vars set. It must exit 0 iff the student's
`get_lr_cosine_schedule` implementation passes the pytest test suite, non-zero otherwise.

**IO boundaries**: filesystem (cp command), subprocess (pytest), env vars (SHADOW_WORKTREE /
MASTERY_PYTHON / VIRTUAL_ENV).

**Branching points**:
- Line 8-11: guard — exit 1 if SHADOW_WORKTREE unset.
- Line 16: dispatch — BUILD STAGE (cwd ≠ SHADOW_WORKTREE) vs HARDEN STAGE (cwd = SHADOW_WORKTREE).
- Lines 30/34/39: dispatch — MASTERY_PYTHON / VIRTUAL_ENV / uv run.

**Load-bearing comments** (lines 14-15): "BUILD STAGE: Copy from main directory to shadow
worktree" — explicitly states the intent to propagate the student's file.

**Type definitions**: `get_lr_cosine_schedule` is defined in `cs336_basics/utils.py` (student
stub: `modes/student/cs336_basics/utils.py:61`; developer reference:
`modes/developer/cs336_basics/utils.py:75`). Imported by `tests/adapters.py:15` as
`from cs336_basics.utils import ... get_lr_cosine_schedule as _get_lr_cosine_schedule_impl`.

**Existing tests**: `tests/test_optimizer.py:52` defines `test_get_lr_cosine_schedule` (note the
`get_` prefix). `tests/test_optimizer.py:4` imports `run_get_lr_cosine_schedule` from `adapters`.

---

## Bug Catalog (ranked by blast radius × plausibility)

### B1 — Wrong file in BUILD-stage `cp` command (CRITICAL)

**The bug**: `validator.sh:18` copies `cs336_basics/optimizer.py` to the shadow worktree; it
never copies `cs336_basics/utils.py`, so the student's `get_lr_cosine_schedule` implementation
never reaches the test environment.

**Blast radius**: Cosine-schedule validation is permanently broken for ALL students, in ALL
submissions. A student with a perfect implementation of `get_lr_cosine_schedule` receives no
credit. A student with a broken one receives no penalty. The validator is silently useless.

**Why it's plausible**: The `cosine_schedule` module is tested alongside AdamW in
`test_optimizer.py`. The validator author plausibly assumed both functions live in `optimizer.py`
(as in some reference implementations), overlooking that `get_lr_cosine_schedule` was placed in
`utils.py`.

**Test types**: contract-pin (unit, static content check) + integration (file-propagation check).

**Self-critique**:
- Fails if bug introduced? YES — the cp command targets the wrong file.
- Passes for wrong-but-stable output? NO — assertion is on the file path in the cp command, which
  is semantically meaningful.
- Fails under non-behavior-changing refactor? NO — asserting the presence of `utils.py` in the cp
  command, not any specific line number or surrounding style.
- Tests observable behavior? YES — the file copied is the observable contract.

---

### B2 — Pytest node ID mismatch: `test_lr_cosine_schedule` vs `test_get_lr_cosine_schedule` (CRITICAL)

**The bug**: `validator.sh` lines 33, 37, and 40 invoke pytest with node ID
`tests/test_optimizer.py::test_lr_cosine_schedule`, but the actual test function in
`tests/test_optimizer.py:52` is named `test_get_lr_cosine_schedule` (with the `get_` prefix). Pytest
finds no matching tests and exits with a non-zero code (exit 5 — "no tests collected"), which
`set -e` propagates as validator failure.

**Blast radius**: Even if B1 were fixed in isolation, the validator would still always exit
non-zero. No cosine-schedule submission can ever be validated; the module is permanently stuck.

**Why it's plausible**: A simple copy-paste error — the author used the module name
`lr_cosine_schedule` as the node suffix but the actual pytest function has the `get_` prefix
that mirrors the implementation function name `get_lr_cosine_schedule`.

**Test types**: contract-pin (unit, static content check).

**Self-critique**:
- Fails if bug introduced? YES — the assertion checks the exact function name present in the file.
- Passes for wrong-but-stable output? NO — checks semantically meaningful identifier.
- Fails under non-behavior-changing refactor? NO — the function name `test_get_lr_cosine_schedule`
  is a public contract item; renaming it would itself be a behavior change.
- Tests observable behavior? YES — the pytest node ID is the observable interface between the
  validator and the test suite.

---

## Skipped Bugs

| Bug considered | Reason skipped |
|---|---|
| HARDEN-stage missing copy | By design: the comment at line 21 says "File already copied by submit-fix." This is an intentional architectural choice, not a bug in this validator. Deferrable. |
| MASTERY_PYTHON / VIRTUAL_ENV / uv dispatch correctness | The three-branch fallback chain (lines 30–40) appears correct and all branches pass PYTHONPATH correctly. No actionable bug found. |
| 5-minute timeout in ValidationSubsystem | This is a system-level configuration concern, not a validator.sh bug. Out of scope for this catalog. |
| `test_adamw` also lives in `test_optimizer.py` but is NOT referenced by this validator | Intentional — this validator is scoped to the cosine schedule module only. |

---

## Evaluation (post-test-run)

### Bugs caught (test failed first run — bug is currently present)

- B1: `test_validator_sh_copies_utils_py_to_shadow_worktree_not_optimizer_py` — FAIL (cp targets
  optimizer.py, not utils.py)
- B1 (integration): `test_validator_propagates_utils_py_to_shadow_worktree` — FAIL (shadow
  worktree's utils.py is not updated by the validator)
- B2: `test_validator_sh_pytest_node_id_matches_test_get_lr_cosine_schedule` — FAIL (node ID is
  `test_lr_cosine_schedule`, not `test_get_lr_cosine_schedule`)

### Bugs characterized (test passed first run — behavior pinned)

None yet; tests were written as RED tests for the fix-stage to verify.

### Bugs discovered during writing not in original catalog

None; both B1 and B2 were already captured in the audit finding and adversarial note.

---

## Investigation Pass (pass+suspect items)

No pass+suspect items; all tests are RED by design (this is the design-tests stage).

**Pre-investigation "0 bugs caught" check**: N/A — this is a known-bug session. The audit finding
already identified both bugs with doubly-confirmed evidence. The tests are written to fail against
the current broken validator and pass after the fix.
