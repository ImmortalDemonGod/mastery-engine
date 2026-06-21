# AIV Verification Packet (v2.2)

## Identification

| Field | Value |
|-------|-------|
| **Repository** | github.com/ImmortalDemonGod/aiv-protocol |
| **Change ID** | mastery-cosine-validator-tests |
| **Commits** | `b289419`, `66ae5e0` |
| **Head SHA** | `66ae5e0` |
| **Base SHA** | `87af785` |
| **Created** | 2026-06-21T02:35:35Z |

## Classification

```yaml
classification:
  risk_tier: R1
  sod_mode: S0
  critical_surfaces: []
  blast_radius: component
  classification_rationale: >
    R1: test-only additions — 2 new files (tests/test_cosine_schedule_validator.bug-catalog.md
    and tests/test_cosine_schedule_validator.py). No production code modified, no existing
    test files altered or deleted. Both files introduce RED behavioral tests that target
    bugs B1/B2 in validator.sh; they are read-only assertions about the shell script text
    and do not mutate any runtime state. Blast radius is confined to the new test-support
    files only. Zero auth/security surface. S0 because a single design-tests worker produces
    both artefacts in one stage with no SoD dependency.
  classified_by: "Claude"
  classified_at: "2026-06-21T02:35:35Z"
```

## Claims

1. Bug catalog documents B1 (wrong cp target: optimizer.py instead of utils.py) and B2 (pytest node ID test_lr_cosine_schedule does not match actual function test_get_lr_cosine_schedule)
2. No existing tests were modified or deleted during this change.
3. validator.sh does not contain the string 'cp cs336_basics/utils.py' — the BUILD-stage cp at line 18 copies optimizer.py only, so student utils.py never propagates to the shadow worktree
4. validator.sh does not contain the string 'test_get_lr_cosine_schedule' — all three pytest invocations (lines 33/37/40) use the incorrect node ID test_lr_cosine_schedule, causing pytest to collect 0 tests and exit non-zero
5. running validator.sh from modes/developer/ with SHADOW_WORKTREE containing stale cs336_basics/utils.py leaves the stale content unchanged; validator exits 4 (file or directory not found: tests/test_optimizer.py::test_lr_cosine_schedule)

---

## Evidence References

| # | Evidence File | Commit SHA | Classes |
|---|---------------|------------|---------|
| 1 | EVIDENCE_TESTS_TEST_COSINE_SCHEDULE_VALIDATOR.BUG_CATALOG.MD.md | `b289419` | A, B, E |
| 2 | EVIDENCE_TESTS_TEST_COSINE_SCHEDULE_VALIDATOR.md | `66ae5e0` | A, B, E |



### Class A (Behavioral / Direct)

`uv run pytest tests/test_cosine_schedule_validator.py -v --tb=short` (foreground run before `aiv commit`):

```
collected 3 items

tests/test_cosine_schedule_validator.py::test_validator_sh_copies_utils_py_not_optimizer_py FAILED
tests/test_cosine_schedule_validator.py::test_validator_sh_pytest_node_id_matches_test_get_lr_cosine_schedule FAILED
tests/test_cosine_schedule_validator.py::test_validator_propagates_utils_py_to_shadow_worktree FAILED

3 failed in 0.59s
```

Failure reasons (from short tb output, all match the targeted bugs):
- Test 1: `AssertionError: validator.sh:18 copies optimizer.py instead of utils.py` — string `cp cs336_basics/utils.py` absent from validator.sh (Bug B1)
- Test 2: `AssertionError: validator.sh uses pytest node ID ::test_lr_cosine_schedule` — string `test_get_lr_cosine_schedule` absent from validator.sh (Bug B2)
- Test 3: `AssertionError: ... Bug B1: currently only optimizer.py is copied. validator exit=4; stderr=ERROR: file or directory not found: tests/test_optimizer.py::test_lr_cosine_schedule` — stale utils.py not overwritten by validator (Bug B1 + B2 combined)

No tests passed that should have failed. No previously-passing tests regressed.

### Class B (Referential Evidence)

**Scope Inventory** (SHA-pinned, line-anchored)

| File | SHA | Lines | Relevance |
|---|---|---|---|
| `tests/test_cosine_schedule_validator.bug-catalog.md` | `b289419` | L1–L131 | Bug catalog: B1/B2 definitions, blast radius, skipped set |
| `tests/test_cosine_schedule_validator.py` | `66ae5e0` | L1–L135 | Test file: 3 RED tests |
| `curricula/cs336_a1/modules/cosine_schedule/validator.sh` | `87af785` (base) | L18 | Bug B1: `cp cs336_basics/optimizer.py` (wrong file) |
| `curricula/cs336_a1/modules/cosine_schedule/validator.sh` | `87af785` (base) | L33,37,40 | Bug B2: `::test_lr_cosine_schedule` (wrong test name) |
| `tests/adapters.py` | `87af785` (base) | L11–15 | Confirms `get_lr_cosine_schedule` imported from `cs336_basics.utils` |
| `tests/test_optimizer.py` | `87af785` (base) | L52 | Confirms function name is `test_get_lr_cosine_schedule` |
| `modes/developer/cs336_basics/utils.py` | `87af785` (base) | L75 | Developer reference: `get_lr_cosine_schedule` lives in `utils.py` |
| `audit/02-static-audit.md` | `7f6610a` | L8 | Canonical finding (Class E source) |

### Class C (Negative Evidence)

**What we searched for and did NOT find:**

- `cp cs336_basics/utils.py` in `validator.sh` → NOT FOUND (confirms B1; only `optimizer.py` appears in any cp command)
- `test_get_lr_cosine_schedule` anywhere in `validator.sh` → NOT FOUND (confirms B2; only `test_lr_cosine_schedule` present)
- Any pre-existing test file in `tests/` covering `validator.sh` behavior → NOT FOUND (this is a new coverage gap being addressed)
- Any other `cp` command in `validator.sh` that might incidentally copy `utils.py` → NOT FOUND (only one cp command exists at line 18)

**Bug-catalog Skipped set** (from `tests/test_cosine_schedule_validator.bug-catalog.md`):

| Bug | Reason skipped |
|---|---|
| HARDEN-stage missing copy | Intentional design: `submit-fix` handles propagation in HARDEN stage |
| MASTERY_PYTHON / VIRTUAL_ENV / uv dispatch | Three-branch fallback appears correct; no actionable bug found |
| 5-minute timeout in ValidationSubsystem | System-level config, not a validator.sh bug |
| `test_adamw` not referenced by this validator | Intentional scope — this validator covers cosine schedule only |

### Class D (Static Analysis)

- **ruff**: 1 error found and fixed before commit (`F401 shutil imported but unused`). Final check: `uv run ruff check tests/test_cosine_schedule_validator.py` → `All checks passed!`
- **mypy**: N/A — test file uses no custom type annotations; assertions operate on `str` and `Path` objects (stdlib), no type inference needed
- **build**: No build step; pure Python test file, no compilation

### Class E (Intent Alignment)

**Canonical intent URL** (SHA-pinned):
`https://github.com/ImmortalDemonGod/mastery-engine/blob/7f6610a902befcb84fc47e5c82a161e3d3184ce4/audit/02-static-audit.md#L8`

**Finding excerpt** (audit/02-static-audit.md L8):
> "Line 18: `cp cs336_basics/optimizer.py ...`. The test suite imports `get_lr_cosine_schedule` from `cs336_basics.utils` ... The validator copies only `optimizer.py` — it never propagates the student's `utils.py` to the shadow worktree, so `test_optimizer.py::test_lr_cosine_schedule` always runs against the stale/original `utils.py`. Cosine-schedule validation is permanently broken regardless of correct student implementation."

**Requirement**: Tests must be RED (failing) against the current broken validator and GREEN after the fix. Specifically: (1) test that cp targets `utils.py`, (2) test that pytest node ID matches actual function name, (3) integration test that shadow worktree utils.py is propagated.

**Alignment**:
- Test 1 (`test_validator_sh_copies_utils_py_not_optimizer_py`) → directly pins B1 at the static contract level
- Test 2 (`test_validator_sh_pytest_node_id_matches_test_get_lr_cosine_schedule`) → directly pins B2 at the static contract level (adversarial note in audit)
- Test 3 (`test_validator_propagates_utils_py_to_shadow_worktree`) → behavioral integration test that shows end-to-end file propagation is broken (B1)

All three tests are RED. None implement the fix. Intent is fully aligned.

### Class F (Provenance)

**Git chain-of-custody for touched test files:**

| Commit | SHA | File | Action |
|---|---|---|---|
| `b289419` | `b289419` | `tests/test_cosine_schedule_validator.bug-catalog.md` | Created (new) |
| `66ae5e0` | `66ae5e0` | `tests/test_cosine_schedule_validator.py` | Created (new) |

No existing test files were modified, renamed, or deleted. Both files were created in this change context (`mastery-cosine-validator-tests`, 2 commits: `b289419`, `66ae5e0`).

Base SHA `87af785` is the commit immediately before this change context opened.

---

## Verification Methodology

**Zero-Touch Mandate:** Verifier inspects artifacts only.
Evidence was collected by `aiv commit` during the change lifecycle.
Packet generated by `aiv close`.

---

## Known Limitations

- Evidence references point to Layer 1 evidence files at specific commit SHAs.
  Use `git show <sha>:.github/aiv-evidence/<file>` to retrieve.

---

## Summary

Change 'mastery-cosine-validator-tests': 2 commit(s) across 2 file(s).
