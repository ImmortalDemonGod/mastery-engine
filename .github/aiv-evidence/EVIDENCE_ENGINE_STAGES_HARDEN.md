# AIV Evidence File (v1.0)

**File:** `engine/stages/harden.py`
**Commit:** `88c5f97`
**Generated:** 2026-06-21T03:21:40Z
**Protocol:** AIV v2.0 + Addendum 2.7 (Zero-Touch Mandate)

---

## Classification (required)

```yaml
classification:
  risk_tier: R1
  sod_mode: S0
  critical_surfaces: []
  blast_radius: "engine/stages/harden.py"
  classification_rationale: "R1: revert of engine module; no auth/security surface; limited blast radius to engine/stages/harden.py only"
  classified_by: "Claude"
  classified_at: "2026-06-21T03:21:40Z"
```

## Claim(s)

1. engine/stages/harden.py _select_bug restored to random.choice(bug_files) at line 205, matching baseline commit 7f6610a
2. baseline failure engine.stages.harden:harden.py:333 is restored; new regression harden.py:336 is eliminated
3. no functional files outside plan §6 scope remain changed: only validator.sh commits 0bf3e9c and b47c691 persist
4. No existing tests were modified or deleted during this change.

---

## Evidence

### Class E (Intent Alignment)

- **Link:** [https://github.com/ImmortalDemonGod/mastery-engine/blob/7f6610a902befcb84fc47e5c82a161e3d3184ce4/audit/02-static-audit.md#L8](https://github.com/ImmortalDemonGod/mastery-engine/blob/7f6610a902befcb84fc47e5c82a161e3d3184ce4/audit/02-static-audit.md#L8)
- **Requirements Verified:** Plan §15 requires the full suite to have zero new failures vs baseline; the prior _select_bug change shifted harden.py line numbers so baseline failure harden.py:333 appeared as new failure harden.py:336 — revert restores baseline line parity

### Class B (Referential Evidence)

**Scope Inventory** (SHA: [`88c5f97`](https://github.com/ImmortalDemonGod/mastery-engine/tree/88c5f97b6585e646cb164cf759b74ed4ad9e1165))

- [`engine/stages/harden.py#L198`](https://github.com/ImmortalDemonGod/mastery-engine/blob/88c5f97b6585e646cb164cf759b74ed4ad9e1165/engine/stages/harden.py#L198)
- [`engine/stages/harden.py#L204-L206`](https://github.com/ImmortalDemonGod/mastery-engine/blob/88c5f97b6585e646cb164cf759b74ed4ad9e1165/engine/stages/harden.py#L204-L206)

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
- **`HardenRunner._select_bug`** (L204-L206): PASS -- 5 test(s) call `_select_bug` directly
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
| 1 | engine/stages/harden.py _select_bug restored to random.choic... | symbol | 5 test(s) call `HardenRunner._select_bug` | PASS VERIFIED |
| 2 | baseline failure engine.stages.harden:harden.py:333 is resto... | unresolved | No automatic binding available | REVIEW MANUAL REVIEW |
| 3 | no functional files outside plan §6 scope remain changed: on... | structural | Class C not collected | REVIEW MANUAL REVIEW |
| 4 | No existing tests were modified or deleted during this chang... | structural | Class C not collected | REVIEW MANUAL REVIEW |

**Verdict summary:** 1 verified, 0 unverified, 3 manual review.
---

## Verification Methodology

**Zero-Touch Mandate:** Verifier inspects artifacts only.
Evidence collected by `aiv commit` running: git diff (scope inventory), AST symbol-to-test binding (2/2 symbols verified).
Ruff/mypy results are in Code Quality (not Class A) because they prove syntax/types, not behavior.

---

## Summary

Revert out-of-scope harden.py _select_bug change to eliminate line-number regression harden.py:336
