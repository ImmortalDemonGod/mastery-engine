# AIV Verification Packet (v2.2)

## Identification

| Field | Value |
|-------|-------|
| **Repository** | github.com/ImmortalDemonGod/mastery-engine |
| **Change ID** | mastery-cosine-validator-impl |
| **Commits** | `0bf3e9c`, `b47c691`, `6d76dde` |
| **Head SHA** | `6d76dde` |
| **Base SHA** | `c57efea` (functional base after design-tests) |
| **Created** | 2026-06-21T03:24:00Z |
| **Finding** | `cosine-validator-wrong-file` |

Note: `c87c203` was committed then reverted by `6d76dde` (out-of-scope scope-creep; net effect on tree is zero for harden.py).

## Classification

```yaml
classification:
  risk_tier: R1
  sod_mode: S0
  critical_surfaces: []
  blast_radius: "curricula/cs336_a1/modules/cosine_schedule/validator.sh (4 functional lines + 1 doc comment). engine/stages/harden.py: net-zero change (c87c203 reverted by 6d76dde). No auth/security surface."
  classification_rationale: >
    R1: shell script textual fix — copies the right source file (utils.py not
    optimizer.py) and uses the correct pytest node ID (test_get_lr_cosine_schedule
    not test_lr_cosine_schedule). Limited blast radius: one shell script in
    one curriculum module. The revert commit restores engine/stages/harden.py
    to exactly the baseline state (7f6610a), so its net delta is zero.
  classified_by: "Claude (write-code stage)"
  classified_at: "2026-06-21T03:24:00Z"
```

## Claims

1. `validator.sh` BUILD stage copies `cs336_basics/utils.py` (not `optimizer.py`) to the shadow worktree — `get_lr_cosine_schedule` is imported from `cs336_basics.utils` per `tests/adapters.py:15`
2. `validator.sh` pytest invocations reference `test_get_lr_cosine_schedule` (was `test_lr_cosine_schedule`, which collected 0 tests) — confirmed by grep and `--collect-only`
3. Developer live-fire: `validator.sh` exits 0 and `test_get_lr_cosine_schedule` PASSES with the developer reference `utils.py`
4. Sentinel live-fire: `validator.sh` exits non-zero and `test_get_lr_cosine_schedule` FAILS when `utils.py` returns wrong warmup LR (99.0 sentinel)
5. All 3 RED tests in `tests/test_cosine_schedule_validator.py` pass
6. `engine/stages/harden.py` at HEAD (`6d76dde`) is byte-for-byte identical to baseline `7f6610a` — baseline failure `engine.stages.harden:harden.py:333` is restored; new regression `harden.py:336` is eliminated
7. No test files were modified or deleted by this change set

---

## Evidence References

| # | Evidence File | Commit SHA | Classes |
|---|---------------|------------|---------|
| 1 | EVIDENCE_CURRICULA_CS336_A1_MODULES_COSINE_SCHEDULE_VALIDATOR.SH.md | `0bf3e9c` | A, E |
| 2 | EVIDENCE_TESTS_TEST_COSINE_SCHEDULE_VALIDATOR.md | `0bf3e9c` | A, E |
| 3 | EVIDENCE_ENGINE_STAGES_HARDEN.md | `6d76dde` | A, E |

### Class A (Behavioral / Direct)

`/root/.local/bin/pytest tests/test_cosine_schedule_validator.py -v --tb=short` (foreground run against HEAD `6d76dde`; local execution — no CI configured for this repo):

```
collected 3 items

tests/test_cosine_schedule_validator.py::test_validator_sh_copies_utils_py_not_optimizer_py PASSED
tests/test_cosine_schedule_validator.py::test_validator_sh_pytest_node_id_matches_test_get_lr_cosine_schedule PASSED
tests/test_cosine_schedule_validator.py::test_validator_propagates_utils_py_to_shadow_worktree PASSED

3 passed in 0.38s
```

All 3 RED tests PASS. Developer live-fire (`bash validator.sh` from `modes/developer/` CWD) exits 0: `test_get_lr_cosine_schedule PASSED`, `PERFORMANCE_SECONDS: 2.08`. Sentinel live-fire (same command with `get_lr_cosine_schedule` warmup patched to return `99.0`) exits 1: `test_get_lr_cosine_schedule FAILED`. Revert verification: `git diff 7f6610a..6d76dde -- engine/stages/harden.py` → empty; `harden.py` at HEAD is byte-for-byte identical to the baseline audit commit.

### Class B (Referential Evidence)

**Scope Inventory** (SHA-pinned, line-anchored)

| File | SHA | Lines | Relevance |
|---|---|---|---|
| `curricula/cs336_a1/modules/cosine_schedule/validator.sh` | `0bf3e9c` | L18 | cp source: `optimizer.py` → `utils.py` (ground-truth from softmax sibling) |
| `curricula/cs336_a1/modules/cosine_schedule/validator.sh` | `0bf3e9c` | L33,37,40 | pytest node ID: `test_lr_cosine_schedule` → `test_get_lr_cosine_schedule` |
| `curricula/cs336_a1/modules/cosine_schedule/validator.sh` | `b47c691` | L18 (comment) | doc comment added: confirms utils.py invariant |
| `engine/stages/harden.py` | `6d76dde` | L195-L209 | revert of c87c203 — net: identical to 7f6610a at these lines |
| `curricula/cs336_a1/modules/softmax/validator.sh` | `7f6610a` | L18 | ground-truth sibling: `cp cs336_basics/utils.py "$SHADOW_WORKTREE/cs336_basics/utils.py"` |
| `tests/test_optimizer.py` | `7f6610a` | L52 | `def test_get_lr_cosine_schedule():` — confirms correct test node ID |
| `tests/adapters.py` | `7f6610a` | L11–15 | `from cs336_basics.utils import ... get_lr_cosine_schedule` — confirms function in utils.py |

### Class C (Negative Evidence)

**What we searched for and did NOT find:**

- `cp cs336_basics/optimizer.py` in `validator.sh` → NOT FOUND (cp command changed to `utils.py`; no residual optimizer.py reference in any cp command)
- `::test_lr_cosine_schedule` without the `get_` prefix in `validator.sh` → NOT FOUND (all three pytest invocations now use `::test_get_lr_cosine_schedule`)
- `test_get_lr_cosine_schedule` count other than 3 in `validator.sh` → NOT FOUND (`grep -c` returns exactly `3`)
- Any modification to test files or Python stubs (`tests/`, `cs336_basics/`) in this change set → NOT FOUND (`git diff c57efea..6d76dde -- tests/ cs336_basics/` is empty)
- Any net change to `engine/stages/harden.py` vs audit baseline `7f6610a` → NOT FOUND (`git diff 7f6610a..6d76dde -- engine/stages/harden.py` is empty)

**Bug-catalog Skipped set** (from `tests/test_cosine_schedule_validator.bug-catalog.md`):

| Bug | Reason skipped |
|---|---|
| HARDEN-stage missing copy (`engine/main.py:1799`) | Separate finding; deferred to follow-up PR (architectural-correctness) |
| `cosine-schedule-wrong-function-name-and-file` (build_prompt.txt) | Separate HIGH-severity finding; separate PR |
| `CORR-003` (_select_bug draft-file selection in harden.py) | Separate finding; not in scope of validator.sh correction |

### Class D (Static Analysis)

- **bash syntax**: `bash -n curricula/cs336_a1/modules/cosine_schedule/validator.sh` → exit 0 (syntactically valid)
- **new dependencies**: zero new imports or packages introduced; `pyproject.toml` unchanged
- **determinism / pin audit**: `ruff`, `mypy`, `black`, `isort`, `flake8` appear only in `[tool.X]` config sections — none are declared as project or dev dependencies; no unpinned formatters in CI
- **ruff on validator.sh**: shell script — ruff not applicable; bash syntax checked via `bash -n` above

### Class E (Intent Alignment)

**Canonical intent URL** (SHA-pinned):
`https://github.com/ImmortalDemonGod/mastery-engine/blob/7f6610a902befcb84fc47e5c82a161e3d3184ce4/audit/02-static-audit.md#L8`

**Finding excerpt** (audit/02-static-audit.md L8, read verbatim):
> "Line 18: `cp cs336_basics/optimizer.py "$SHADOW_WORKTREE/cs336_basics/optimizer.py"`. The test suite imports `get_lr_cosine_schedule` from `cs336_basics.utils` (confirmed: `tests/adapters.py:11-15` does `from cs336_basics.utils import ... get_lr_cosine_schedule as _get_lr_cosine_schedule_impl`). The developer reference implementation lives in `modes/developer/cs336_basics/utils.py:75`. The validator copies only `optimizer.py` — it never propagates the student's `utils.py` to the shadow worktree, so `test_optimizer.py::test_lr_cosine_schedule` always runs against the stale/original `utils.py`. Cosine-schedule validation is permanently broken regardless of correct student implementation."

**Adversarial note confirms:** "(1) get_lr_cosine_schedule is defined in cs336_basics/utils.py and imported from cs336_basics.utils at tests/adapters.py:11-15, but validator.sh:18 copies only optimizer.py, so student utils.py never propagates. (2) validator.sh:33/37/40 invoke ::test_lr_cosine_schedule, but tests/test_optimizer.py:52 defines test_get_lr_cosine_schedule, so pytest collects nothing. Critical."

**Alignment assessment:** This change set directly addresses both defects recorded at the audit source:
- `validator.sh:18` corrected from `optimizer.py` to `utils.py`, consuming the ground-truth pattern from `softmax/validator.sh:18`
- Lines 33/37/40 corrected from `test_lr_cosine_schedule` to `test_get_lr_cosine_schedule`, matching `tests/test_optimizer.py:52`
- Revert commit `6d76dde` removes a scope-creeping harden.py change that caused a line-number regression (baseline failure `harden.py:333` shifted to `harden.py:336`); restores baseline failure profile, satisfying plan §15 (no new failures vs baseline)

The intent recorded at the audit source is fully satisfied.

### Class F (Provenance)

**Git chain-of-custody for touched test files:**

| Commit | SHA | File | Action |
|---|---|---|---|
| design-tests stage (upstream) | `66ae5e0` | `tests/test_cosine_schedule_validator.py` | Created (new) — before this change context |

**Justification:** The test file `tests/test_cosine_schedule_validator.py` was created in the upstream design-tests stage (`66ae5e0`) to define RED tests for the two known issues. This change set makes those tests GREEN by correcting `validator.sh` — it does NOT modify the test file itself. `git diff c57efea..6d76dde -- tests/test_cosine_schedule_validator.py` → empty (file unchanged).

No existing test files were modified, renamed, or deleted by this change set.

**Class G — Excluded** (cognitive artifacts only, per operator mandate).

---

## Verification Methodology

**Zero-Touch Mandate:** Verifier inspects artifacts only.
Evidence was collected by `aiv commit` during the change lifecycle.
Packet generated by `aiv close` then amended to cover all three commits.

---

## Known Limitations

- Evidence references point to Layer 1 evidence files at specific commit SHAs.
  Use `git show <sha>:.github/aiv-evidence/<file>` to retrieve.
- No CI configured for this repository; Class A evidence is local execution output.
- E012 warnings from `aiv check` are expected in this environment (same as design-tests packet).

---

## Summary

Change `mastery-cosine-validator-impl`: 3 effective commits. One file net-changed:
1. `curricula/cs336_a1/modules/cosine_schedule/validator.sh` — 4 functional lines + 1 doc comment. Fixes finding `cosine-validator-wrong-file` recorded at `audit/02-static-audit.md#L8` (CRITICAL).

Note: `c87c203` (out-of-scope harden.py change) was reverted by `6d76dde`, leaving `harden.py` identical to baseline `7f6610a`. This eliminates the line-number regression that caused `engine.stages.harden:harden.py:336` to appear as a new failure.

All 3 RED tests pass. Developer live-fire exits 0. Sentinel live-fire exits 1. Zero new test failures vs baseline.

## Machine-checkable data

```json
{
  "change_id": "mastery-cosine-validator-impl",
  "finding": "cosine-validator-wrong-file",
  "canonical_intent": "https://github.com/ImmortalDemonGod/mastery-engine/blob/7f6610a902befcb84fc47e5c82a161e3d3184ce4/audit/02-static-audit.md#L8",
  "functional_commits": ["0bf3e9c", "b47c691", "6d76dde"],
  "net_functional_files": [
    "curricula/cs336_a1/modules/cosine_schedule/validator.sh"
  ],
  "validator_sh_lines_changed": [18, 19, 33, 37, 40],
  "harden_py_net_delta": "zero (c87c203 reverted by 6d76dde; harden.py identical to baseline 7f6610a)",
  "red_tests": {
    "file": "tests/test_cosine_schedule_validator.py",
    "count": 3,
    "result": "3 passed"
  },
  "live_fire_developer": {"exit_code": 0, "test": "test_get_lr_cosine_schedule", "result": "PASSED"},
  "live_fire_sentinel": {"exit_code": 1, "test": "test_get_lr_cosine_schedule", "result": "FAILED"},
  "new_failures": [],
  "baseline_regression_eliminated": "engine.stages.harden:harden.py:336",
  "bash_syntax_valid": true,
  "evidence_classes": ["A", "B", "C", "D", "E", "F"]
}
```
