# AIV Evidence File (v1.0)

**File:** `curricula/cs336_a1/modules/cosine_schedule/validator.sh`
**Commit:** `8bb7900`
**Previous:** `0bf3e9c`
**Generated:** 2026-06-21T02:56:47Z
**Protocol:** AIV v2.0 + Addendum 2.7 (Zero-Touch Mandate)

---

## Classification (required)

```yaml
classification:
  risk_tier: R1
  sod_mode: S0
  critical_surfaces: []
  blast_radius: "curricula/cs336_a1/modules/cosine_schedule/validator.sh"
  classification_rationale: "R1: shell script comment addition documenting a non-obvious invariant (wrong-file trap); no logic changes"
  classified_by: "Claude"
  classified_at: "2026-06-21T02:56:47Z"
```

## Claim(s)

1. validator.sh BUILD stage copies cs336_basics/utils.py (not optimizer.py) — get_lr_cosine_schedule is defined in cs336_basics.utils per tests/adapters.py:15
2. validator.sh pytest invocations use test_get_lr_cosine_schedule — matches actual test function at tests/test_optimizer.py:52
3. all 3 RED tests in tests/test_cosine_schedule_validator.py pass — static content and propagation verified
4. No existing tests were modified or deleted during this change.

---

## Evidence

### Class E (Intent Alignment)

- **Link:** [https://github.com/ImmortalDemonGod/mastery-engine/blob/7f6610a902befcb84fc47e5c82a161e3d3184ce4/audit/02-static-audit.md#L8](https://github.com/ImmortalDemonGod/mastery-engine/blob/7f6610a902befcb84fc47e5c82a161e3d3184ce4/audit/02-static-audit.md#L8)
- **Requirements Verified:** audit/02-static-audit.md#L8 records that validator.sh copies optimizer.py (wrong) and uses test_lr_cosine_schedule (wrong ID); this change fixes both defects and documents the utils.py invariant

### Class B (Referential Evidence)

**Scope Inventory** (SHA: [`8bb7900`](https://github.com/ImmortalDemonGod/mastery-engine/tree/8bb7900d86a7fdf5755c328cca38ce4ca000fc16))

- [`curricula/cs336_a1/modules/cosine_schedule/validator.sh#L18`](https://github.com/ImmortalDemonGod/mastery-engine/blob/8bb7900d86a7fdf5755c328cca38ce4ca000fc16/curricula/cs336_a1/modules/cosine_schedule/validator.sh#L18)

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
| 2 | validator.sh pytest invocations use test_get_lr_cosine_sched... | unresolved | No automatic binding available | REVIEW MANUAL REVIEW |
| 3 | all 3 RED tests in tests/test_cosine_schedule_validator.py p... | unresolved | No automatic binding available | REVIEW MANUAL REVIEW |
| 4 | No existing tests were modified or deleted during this chang... | structural | Class C not collected | REVIEW MANUAL REVIEW |

**Verdict summary:** 0 verified, 0 unverified, 4 manual review.
---

## Verification Methodology

**Zero-Touch Mandate:** Verifier inspects artifacts only.
Evidence collected by `aiv commit` running: git diff (scope inventory), pytest (no claim-specific tests found).
Ruff/mypy results are in Code Quality (not Class A) because they prove syntax/types, not behavior.

---

## Summary

Document that utils.py must be copied (not optimizer.py) because get_lr_cosine_schedule lives in cs336_basics.utils
