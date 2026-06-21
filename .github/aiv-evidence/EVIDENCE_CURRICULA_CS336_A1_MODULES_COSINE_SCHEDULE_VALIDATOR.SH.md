# AIV Evidence File (v1.0)

**File:** `curricula/cs336_a1/modules/cosine_schedule/validator.sh`
**Commit:** `c57efea`
**Generated:** 2026-06-21T02:41:52Z
**Protocol:** AIV v2.0 + Addendum 2.7 (Zero-Touch Mandate)

---

## Classification (required)

```yaml
classification:
  risk_tier: R1
  sod_mode: S0
  critical_surfaces: []
  blast_radius: "curricula/cs336_a1/modules/cosine_schedule/validator.sh"
  classification_rationale: "Single-file shell script change fixing two textual bugs (wrong filename on line 18, wrong pytest node ID on lines 33/37/40). No logic changes, no new dependencies, no API surface changes. RED tests + live-fire evidence collected."
  classified_by: "Claude"
  classified_at: "2026-06-21T02:41:52Z"
```

## Claim(s)

1. validator.sh BUILD stage copies cs336_basics/utils.py (not optimizer.py) to shadow worktree
2. validator.sh pytest invocations reference test_get_lr_cosine_schedule (was test_lr_cosine_schedule, which collected 0 tests)
3. developer live-fire: validator exits 0 and test_get_lr_cosine_schedule PASSES when utils.py contains a correct implementation
4. sentinel live-fire: validator exits non-zero and test_get_lr_cosine_schedule FAILS when utils.py returns wrong constant
5. No existing tests were modified or deleted during this change.

---

## Evidence

### Class E (Intent Alignment)

- **Link:** [https://github.com/ImmortalDemonGod/mastery-engine/blob/7f6610a902befcb84fc47e5c82a161e3d3184ce4/audit/02-static-audit.md#L8](https://github.com/ImmortalDemonGod/mastery-engine/blob/7f6610a902befcb84fc47e5c82a161e3d3184ce4/audit/02-static-audit.md#L8)
- **Requirements Verified:** audit/02-static-audit.md#L8 records that validator.sh:18 copies optimizer.py instead of utils.py (where get_lr_cosine_schedule lives per tests/adapters.py:15), causing cosine-schedule validation to silently exercise stale code regardless of student implementation

### Class B (Referential Evidence)

**Scope Inventory** (SHA: [`c57efea`](https://github.com/ImmortalDemonGod/mastery-engine/tree/c57efea54073ece68e1598c5f06603b99c7b9823))

- [`curricula/cs336_a1/modules/cosine_schedule/validator.sh#L18`](https://github.com/ImmortalDemonGod/mastery-engine/blob/c57efea54073ece68e1598c5f06603b99c7b9823/curricula/cs336_a1/modules/cosine_schedule/validator.sh#L18)
- [`curricula/cs336_a1/modules/cosine_schedule/validator.sh#L33`](https://github.com/ImmortalDemonGod/mastery-engine/blob/c57efea54073ece68e1598c5f06603b99c7b9823/curricula/cs336_a1/modules/cosine_schedule/validator.sh#L33)
- [`curricula/cs336_a1/modules/cosine_schedule/validator.sh#L37`](https://github.com/ImmortalDemonGod/mastery-engine/blob/c57efea54073ece68e1598c5f06603b99c7b9823/curricula/cs336_a1/modules/cosine_schedule/validator.sh#L37)
- [`curricula/cs336_a1/modules/cosine_schedule/validator.sh#L40`](https://github.com/ImmortalDemonGod/mastery-engine/blob/c57efea54073ece68e1598c5f06603b99c7b9823/curricula/cs336_a1/modules/cosine_schedule/validator.sh#L40)

### Class A (Execution Evidence)

**WARNING:** No tests found that directly import or reference the changed file.
This file has no claim-specific execution evidence.

### Code Quality (Linting & Types)

- **ruff:** 0 error(s)
- **mypy:** 

## Claim Verification Matrix

| # | Claim | Type | Evidence | Verdict |
|---|-------|------|----------|---------|
| 1 | validator.sh BUILD stage copies cs336_basics/utils.py (not o... | unresolved | No automatic binding available | REVIEW MANUAL REVIEW |
| 2 | validator.sh pytest invocations reference test_get_lr_cosine... | unresolved | No automatic binding available | REVIEW MANUAL REVIEW |
| 3 | developer live-fire: validator exits 0 and test_get_lr_cosin... | unresolved | No automatic binding available | REVIEW MANUAL REVIEW |
| 4 | sentinel live-fire: validator exits non-zero and test_get_lr... | unresolved | No automatic binding available | REVIEW MANUAL REVIEW |
| 5 | No existing tests were modified or deleted during this chang... | structural | Class C not collected | REVIEW MANUAL REVIEW |

**Verdict summary:** 0 verified, 0 unverified, 5 manual review.
---

## Verification Methodology

**Zero-Touch Mandate:** Verifier inspects artifacts only.
Evidence collected by `aiv commit` running: git diff (scope inventory), pytest (no claim-specific tests found).
Ruff/mypy results are in Code Quality (not Class A) because they prove syntax/types, not behavior.

---

## Summary

Fix validator.sh to copy utils.py instead of optimizer.py and use correct pytest node ID test_get_lr_cosine_schedule
