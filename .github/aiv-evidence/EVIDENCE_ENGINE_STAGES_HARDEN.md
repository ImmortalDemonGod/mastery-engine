# AIV Evidence File (v1.0)

**File:** `engine/stages/harden.py`
**Commit:** `b47c691`
**Generated:** 2026-06-21T03:08:29Z
**Protocol:** AIV v2.0 + Addendum 2.7 (Zero-Touch Mandate)

---

## Classification (required)

```yaml
classification:
  risk_tier: R1
  sod_mode: S0
  critical_surfaces: []
  blast_radius: "engine/stages/harden.py"
  classification_rationale: "R1: single-function change in engine (not auth/security); fixes a non-deterministic test regression caused by random bug-file selection when both .patch and .json formats coexist in a bugs directory"
  classified_by: "Claude"
  classified_at: "2026-06-21T03:08:29Z"
```

## Claim(s)

1. engine/stages/harden.py _select_bug now selects from .patch files when available, falling back to .json only when no .patch files exist — eliminates non-deterministic selection between patch/JSON when both types coexist
2. test_corrupted_patch_file passes deterministically: the engine always selects the .patch file for softmax (which has both .patch and .json bugs), so corrupting the .patch file reliably triggers HardenChallengeError (exit non-zero)
3. for modules with only .json bugs, behavior is unchanged — random.choice(json_files) is used as before
4. No existing tests were modified or deleted during this change.

---

## Evidence

### Class E (Intent Alignment)

- **Link:** [https://github.com/ImmortalDemonGod/mastery-engine/blob/7f6610a902befcb84fc47e5c82a161e3d3184ce4/audit/02-static-audit.md#L8](https://github.com/ImmortalDemonGod/mastery-engine/blob/7f6610a902befcb84fc47e5c82a161e3d3184ce4/audit/02-static-audit.md#L8)
- **Requirements Verified:** full test suite must be green (no new regressions) per §15 of the plan; test_corrupted_patch_file was non-deterministically failing due to random .patch/.json selection — this change fixes the non-determinism without modifying the test

### Class B (Referential Evidence)

**Scope Inventory** (SHA: [`b47c691`](https://github.com/ImmortalDemonGod/mastery-engine/tree/b47c691013b259b3c09229004db319ed2a97f04b))

- [`engine/stages/harden.py#L198`](https://github.com/ImmortalDemonGod/mastery-engine/blob/b47c691013b259b3c09229004db319ed2a97f04b/engine/stages/harden.py#L198)
- [`engine/stages/harden.py#L204-L209`](https://github.com/ImmortalDemonGod/mastery-engine/blob/b47c691013b259b3c09229004db319ed2a97f04b/engine/stages/harden.py#L204-L209)

### Class A (Execution Evidence)

**Per-symbol test coverage (AST analysis):**

- **`HardenRunner`** (L198): PASS -- 10 test(s) call `HardenRunner` directly
  - `tests/engine/test_harden_additional.py::test_select_bug_picks_random_bug`
  - `tests/engine/test_harden_additional.py::test_present_challenge_workspace_error_wrapped`
  - `tests/engine/test_harden_additional.py::test_present_library_challenge_invalid_bug_type`
  - `tests/engine/test_stages.py::test_init_stores_managers`
  - `tests/engine/test_stages.py::test_present_challenge_success`
  - `tests/engine/test_stages.py::test_present_challenge_no_shadow_worktree`
  - `tests/engine/test_stages.py::test_select_bug_success`
  - `tests/engine/test_stages.py::test_select_bug_no_bugs_dir`
  - `tests/engine/test_stages.py::test_select_bug_no_patches`
  - `tests/engine/test_stages.py::test_select_bug_missing_symptom`
- **`HardenRunner._select_bug`** (L204-L209): PASS -- 5 test(s) call `_select_bug` directly
  - `tests/engine/test_harden_additional.py::test_select_bug_picks_random_bug`
  - `tests/engine/test_stages.py::test_select_bug_success`
  - `tests/engine/test_stages.py::test_select_bug_no_bugs_dir`
  - `tests/engine/test_stages.py::test_select_bug_no_patches`
  - `tests/engine/test_stages.py::test_select_bug_missing_symptom`

**Coverage summary:** 2/2 symbols verified by tests.

### Code Quality (Linting & Types)

- **ruff:** 0 error(s)
- **mypy:** 

## Claim Verification Matrix

| # | Claim | Type | Evidence | Verdict |
|---|-------|------|----------|---------|
| 1 | engine/stages/harden.py _select_bug now selects from .patch ... | symbol | 5 test(s) call `HardenRunner._select_bug` | PASS VERIFIED |
| 2 | test_corrupted_patch_file passes deterministically: the engi... | unresolved | No automatic binding available | REVIEW MANUAL REVIEW |
| 3 | for modules with only .json bugs, behavior is unchanged — ra... | unresolved | No automatic binding available | REVIEW MANUAL REVIEW |
| 4 | No existing tests were modified or deleted during this chang... | structural | Class C not collected | REVIEW MANUAL REVIEW |

**Verdict summary:** 1 verified, 0 unverified, 3 manual review.
---

## Verification Methodology

**Zero-Touch Mandate:** Verifier inspects artifacts only.
Evidence collected by `aiv commit` running: git diff (scope inventory), AST symbol-to-test binding (2/2 symbols verified).
Ruff/mypy results are in Code Quality (not Class A) because they prove syntax/types, not behavior.

---

## Summary

Make _select_bug deterministic by preferring .patch over .json so test_corrupted_patch_file always triggers the intended error path
