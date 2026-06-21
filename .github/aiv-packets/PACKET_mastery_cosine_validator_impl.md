# AIV Verification Packet (v2.2)

## Identification

| Field | Value |
|-------|-------|
| **Repository** | github.com/ImmortalDemonGod/mastery-engine |
| **Change ID** | mastery-cosine-validator-impl |
| **Commits** | `0bf3e9c` |
| **Head SHA** | `0bf3e9c` |
| **Base SHA** | `c57efea` |
| **Created** | 2026-06-21T02:42:47Z |

## Classification

```yaml
classification:
  risk_tier: R1
  sod_mode: S0
  critical_surfaces: []
  blast_radius: "curricula/cs336_a1/modules/cosine_schedule/validator.sh"
  classification_rationale: >
    Single-file shell-script change fixing two textual bugs: wrong cp source/dest on
    line 18 (optimizer.py → utils.py) and wrong pytest node IDs on lines 33/37/40
    (test_lr_cosine_schedule → test_get_lr_cosine_schedule). No logic changes, no
    new dependencies, no API surface changes. Impact is limited to the cosine_schedule
    validation path in the mastery engine student build stage.
  classified_by: "Claude (write-code stage)"
  classified_at: "2026-06-21T02:42:47Z"
```

## Claims

1. validator.sh BUILD stage copies `cs336_basics/utils.py` (not `optimizer.py`) to shadow worktree — confirmed by static grep and integration test
2. validator.sh pytest invocations reference `test_get_lr_cosine_schedule` (was `test_lr_cosine_schedule`, which collected 0 tests) — confirmed by static grep and pytest `--collect-only`
3. Developer live-fire: validator exits 0 and `test_get_lr_cosine_schedule` PASSES when `cs336_basics/utils.py` contains the correct implementation (CWD=`modes/developer/`)
4. Sentinel live-fire: validator exits non-zero and `test_get_lr_cosine_schedule` FAILS when `cs336_basics/utils.py` returns a wrong constant (99.0 for warmup LR)
5. No existing test files were modified or deleted by this change — confirmed by `git diff HEAD~1 -- tests/`

---

## Evidence References

| # | Evidence File | Commit SHA | Classes |
|---|---------------|------------|---------|
| 1 | EVIDENCE_CURRICULA_CS336_A1_MODULES_COSINE_SCHEDULE_VALIDATOR.SH.md | `0bf3e9c` | A, B, D, E |

---

### Class A — Behavioral / Direct Evidence

**A1 — RED tests (design-tests contract)**

Command: `uv run pytest tests/test_cosine_schedule_validator.py -v --tb=short`

```
platform linux -- Python 3.11.15, pytest-8.4.1
collected 3 items

tests/test_cosine_schedule_validator.py::test_validator_sh_copies_utils_py_not_optimizer_py PASSED
tests/test_cosine_schedule_validator.py::test_validator_sh_pytest_node_id_matches_test_get_lr_cosine_schedule PASSED
tests/test_cosine_schedule_validator.py::test_validator_propagates_utils_py_to_shadow_worktree PASSED

3 passed in 0.38s
```

All 3 RED tests now PASS. Addresses Claim 1 (static content), Claim 2 (static content), and Claim 1 integration (propagation).

**A2 — Developer live-fire (Claim 3 — acceptance criterion #6)**

Setup: shadow worktree at `/tmp/tmp.qBE2kURC00` with `tests/test_optimizer.py`, `tests/adapters.py`, `tests/conftest.py`, `pyproject.toml`, and all `cs336_basics/` files except `utils.py` (stale sentinel). CWD=`modes/developer/` (contains `cs336_basics/utils.py` with correct `get_lr_cosine_schedule` implementation).

Command:
```bash
SHADOW_WORKTREE=/tmp/tmp.qBE2kURC00 \
MASTERY_PYTHON=/root/.cache/aiv-venvs/mastery-engine-7f6610a902be/bin/python \
bash curricula/cs336_a1/modules/cosine_schedule/validator.sh
```

Output:
```
collected 1 item
tests/test_optimizer.py::test_get_lr_cosine_schedule PASSED
1 passed in 0.04s
PERFORMANCE_SECONDS: 2.082698106765747
```

Exit code: 0. Claim 3 VERIFIED.

**A3 — Sentinel live-fire (Claim 4 — acceptance criterion #7)**

Setup: same shadow worktree. `utils.py` patched to return `99.0` for warmup iterations (wrong constant). Validator copies this broken implementation to shadow worktree.

Command: same as A2 but from a tmp dir containing the patched `utils.py`.

Output:
```
collected 1 item
tests/test_optimizer.py::test_get_lr_cosine_schedule FAILED
  AssertionError: Not equal to tolerance rtol=1e-07, atol=0
  Mismatched elements: 7 / 25 (28%)
  Max absolute difference among violations: 99.
1 failed in 0.03s
```

Exit code: 1. Claim 4 VERIFIED.

**A4 — Pre-existing failures confirmed unchanged (acceptance criterion #8)**

Command (before fix, via git stash): `uv run pytest tests/test_optimizer.py -v --tb=line`
Result: `2 failed` (`test_adamw`, `test_get_lr_cosine_schedule`) — both due to `NotImplementedError` stubs in `cs336_basics/optimizer.py:45` and `cs336_basics/utils.py:82`.

Command (after fix): same — still `2 failed`, same reasons.

Conclusion: zero new failures introduced. The stub failures pre-exist this change and are unrelated to `validator.sh`. Claim 5 VERIFIED.

---

### Class B — Referential Evidence (SHA-pinned, line-anchored)

**Scope inventory** — all hunks in commit `0bf3e9c`:

| Line | Before (BROKEN) | After (FIXED) | Source |
|------|-----------------|---------------|--------|
| [`validator.sh:18`](https://github.com/ImmortalDemonGod/mastery-engine/blob/0bf3e9c/curricula/cs336_a1/modules/cosine_schedule/validator.sh#L18) | `cp cs336_basics/optimizer.py "$SHADOW_WORKTREE/cs336_basics/optimizer.py"` | `cp cs336_basics/utils.py "$SHADOW_WORKTREE/cs336_basics/utils.py"` | Ground truth: `softmax/validator.sh:18` |
| [`validator.sh:33`](https://github.com/ImmortalDemonGod/mastery-engine/blob/0bf3e9c/curricula/cs336_a1/modules/cosine_schedule/validator.sh#L33) | `::test_lr_cosine_schedule` | `::test_get_lr_cosine_schedule` | `tests/test_optimizer.py:52` |
| [`validator.sh:37`](https://github.com/ImmortalDemonGod/mastery-engine/blob/0bf3e9c/curricula/cs336_a1/modules/cosine_schedule/validator.sh#L37) | `::test_lr_cosine_schedule` | `::test_get_lr_cosine_schedule` | `tests/test_optimizer.py:52` |
| [`validator.sh:40`](https://github.com/ImmortalDemonGod/mastery-engine/blob/0bf3e9c/curricula/cs336_a1/modules/cosine_schedule/validator.sh#L40) | `::test_lr_cosine_schedule` | `::test_get_lr_cosine_schedule` | `tests/test_optimizer.py:52` |

**Ground-truth reference** — `curricula/cs336_a1/modules/softmax/validator.sh:18` (SHA `66ae5e0`):
```bash
cp cs336_basics/utils.py "$SHADOW_WORKTREE/cs336_basics/utils.py"
```
This is the exact sibling pattern consumed for line 18 — not derived from prose.

**Canonical test function** — `tests/test_optimizer.py:52` (SHA `66ae5e0`):
```python
def test_get_lr_cosine_schedule():
```

**Import chain** — `tests/adapters.py:15` (SHA `66ae5e0`):
```python
from cs336_basics.utils import ... get_lr_cosine_schedule as _get_lr_cosine_schedule_impl
```
Confirms function lives in `cs336_basics.utils`, not `cs336_basics.optimizer`.

---

### Class C — Negative Evidence (what was searched for and NOT found)

**C1 — No bare `test_lr_cosine_schedule` remains in validator.sh (Bug B2 eradication)**

Command: `grep 'test_lr_cosine_schedule' curricula/cs336_a1/modules/cosine_schedule/validator.sh | grep -v 'test_get_lr_cosine_schedule'`
Result: empty (exit 1 = no match). All three occurrences now use `test_get_lr_cosine_schedule`.

**C2 — No `optimizer.py` reference remains in the BUILD cp command**

Command: `grep 'optimizer\.py.*SHADOW_WORKTREE\|SHADOW_WORKTREE.*optimizer\.py' curricula/cs336_a1/modules/cosine_schedule/validator.sh`
Result: empty. The cp command no longer targets `optimizer.py`.

**C3 — Count check**

`grep -c 'test_get_lr_cosine_schedule' curricula/cs336_a1/modules/cosine_schedule/validator.sh` → `3` (exactly the three pytest invocation paths).

**C4 — Bug catalog "Skipped" set**

The bug catalog (`tests/test_cosine_schedule_validator.bug-catalog.md`) lists two bugs: B1 (wrong file) and B2 (wrong test name). Both are addressed. No additional bugs are documented in the catalog that were skipped by this change.

**C5 — No change to test files or Python implementation**

`git diff HEAD~1 -- tests/ cs336_basics/` → empty. Test files and student implementation stubs are unchanged.

---

### Class D — Static Analysis (lint / type / build)

**D1 — Bash syntax validation**

Command: `bash -n curricula/cs336_a1/modules/cosine_schedule/validator.sh`
Result: `BASH SYNTAX OK` (exit 0). The file is syntactically valid bash.

**D2 — ruff (Python linter)**

ruff is configured in `pyproject.toml` under `[tool.ruff]` but is NOT declared as a project dependency. When run against the `.sh` file, ruff generates "invalid-syntax" errors because it attempts to parse bash as Python — this is a tool misconfiguration, not a real code error. These errors are pre-existing (the file was always a bash script; ruff cannot lint it). Python files touched by this change: none.

**D3 — No new dependencies introduced**

This change adds zero new imports, zero new packages, zero new executables. `pyproject.toml` is unchanged.

**D4 — Determinism / pin audit**

The plan required pinning lint tools declared without `==` in pyproject. Audit result: `ruff`, `mypy`, `black`, `isort`, `flake8` appear only in `[tool.X]` config sections — none are declared as project or dev dependencies. `pytest-cov>=7.0.0` is the only dev dependency. No pinning action required.

---

### Class E — Intent Alignment

**Source:** [`audit/02-static-audit.md#L8` @ SHA `7f6610a902befcb84fc47e5c82a161e3d3184ce4`](https://github.com/ImmortalDemonGod/mastery-engine/blob/7f6610a902befcb84fc47e5c82a161e3d3184ce4/audit/02-static-audit.md#L8)

**Recorded defect (read from source):** The audit records at line 8: `cp cs336_basics/optimizer.py "$SHADOW_WORKTREE/cs336_basics/optimizer.py"`. The test suite imports `get_lr_cosine_schedule` from `cs336_basics.utils` (confirmed via `tests/adapters.py:11-15`). The developer reference implementation lives in `modes/developer/cs336_basics/utils.py:75`. The validator copies only `optimizer.py` — it never propagates the student's `utils.py` to the shadow worktree, so `test_optimizer.py::test_lr_cosine_schedule` always runs against the stale/original `utils.py`. The audit also identifies a second defect: the test node ID `test_lr_cosine_schedule` (missing `get_` prefix) means pytest collects 0 tests and exits with a false pass.

**Alignment assessment:** This change directly addresses both defects recorded at the audit source:
- Line 18 is corrected from `optimizer.py` to `utils.py`, consuming the ground-truth pattern from `softmax/validator.sh:18` (the closest sibling validator, confirmed correct).
- Lines 33, 37, 40 are corrected from `test_lr_cosine_schedule` to `test_get_lr_cosine_schedule`, matching the actual function defined at `tests/test_optimizer.py:52`.

The intent is fully satisfied: the validator now propagates the student's `utils.py` to the shadow worktree and runs the correct test node, making cosine-schedule validation functional.

---

### Class F — Provenance (git chain-of-custody of touched test files)

**Test file:** `tests/test_cosine_schedule_validator.py`

Git provenance chain:
```
66ae5e069c275d5ca264ed7e43bca3c550680ae8 2026-06-21 02:35:31 +0000
  design-tests: add RED tests for cosine_schedule validator bugs B1+B2
```

This file was introduced in commit `66ae5e0` as part of the design-tests stage — the stage upstream of write-code. It was not modified by this change (commit `0bf3e9c`). The file contains three tests explicitly named for the bugs they catch (B1, B2) as required by the design-tests contract.

`git diff HEAD~1 -- tests/test_cosine_schedule_validator.py` → empty (file unchanged by this commit).

**Class G — Excluded per operator mandate (cognitive artifacts only).**

---

## Verification Methodology

**Zero-Touch Mandate:** Verifier inspects artifacts only. All evidence above was collected and executed by the write-code agent. No human CI execution required.

Evidence was collected by: static grep, `bash -n` syntax check, `uv run pytest --collect-only` collection check, three RED unit/integration tests, developer live-fire, sentinel live-fire, pre-existing failure baseline, git diff provenance.

---

## Deferred Items (plan §6)

| Item | Classification | Status |
|------|---------------|--------|
| `engine/main.py:1799` — HARDEN stage copies wrong file for `cosine_schedule` module | architectural-correctness | Deferred to follow-up PR; not a blocker for this change |
| Finding `cosine-schedule-wrong-function-name-and-file` (audit line 21) | architectural-correctness | Separate HIGH-severity finding; separate PR |

---

## Summary

Change `mastery-cosine-validator-impl`: 1 commit (`0bf3e9c`), 1 functional file changed (`curricula/cs336_a1/modules/cosine_schedule/validator.sh`, 4 lines). Fixes finding `cosine-validator-wrong-file` recorded at [`audit/02-static-audit.md#L8`](https://github.com/ImmortalDemonGod/mastery-engine/blob/7f6610a902befcb84fc47e5c82a161e3d3184ce4/audit/02-static-audit.md#L8). All 3 RED tests pass. Developer and sentinel live-fire both behave correctly. Zero new test failures.

## Machine-checkable data

```json
{
  "change_id": "mastery-cosine-validator-impl",
  "finding": "cosine-validator-wrong-file",
  "canonical_intent": "https://github.com/ImmortalDemonGod/mastery-engine/blob/7f6610a902befcb84fc47e5c82a161e3d3184ce4/audit/02-static-audit.md#L8",
  "functional_commit": "0bf3e9c",
  "functional_file": "curricula/cs336_a1/modules/cosine_schedule/validator.sh",
  "lines_changed": [18, 33, 37, 40],
  "red_tests": {
    "file": "tests/test_cosine_schedule_validator.py",
    "count": 3,
    "result": "3 passed"
  },
  "live_fire_developer": {"exit_code": 0, "test": "test_get_lr_cosine_schedule", "result": "PASSED"},
  "live_fire_sentinel": {"exit_code": 1, "test": "test_get_lr_cosine_schedule", "result": "FAILED"},
  "pre_existing_failures": ["test_adamw", "test_get_lr_cosine_schedule"],
  "new_failures": [],
  "bash_syntax_valid": true,
  "evidence_classes": ["A", "B", "C", "D", "E", "F"]
}
```
