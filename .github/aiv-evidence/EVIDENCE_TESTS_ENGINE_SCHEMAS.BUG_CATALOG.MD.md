# AIV Evidence File (v1.0)

**File:** `tests/engine/schemas.bug-catalog.md`
**Commit:** `119a096`
**Generated:** 2026-06-21T07:00:38Z
**Protocol:** AIV v2.0 + Addendum 2.7 (Zero-Touch Mandate)

---

## Classification (required)

```yaml
classification:
  risk_tier: R0
  sod_mode: S0
  critical_surfaces: []
  blast_radius: "tests/engine/schemas.bug-catalog.md"
  classification_rationale: "Documentation-only artifact — bug catalog markdown file, no logic changes"
  classified_by: "Claude"
  classified_at: "2026-06-21T07:00:38Z"
```

## Claim(s)

1. schemas.bug-catalog.md documents 5 bug classes for UserProgress.mark_stage_complete, maps each to a test type, and records the skipped set
2. No existing tests were modified or deleted during this change.

---

## Evidence

### Class E (Intent Alignment)

- **Link:** [https://github.com/ImmortalDemonGod/mastery-engine/blob/7f6610a902befcb84fc47e5c82a161e3d3184ce4/audit/02-static-audit.md#L17](https://github.com/ImmortalDemonGod/mastery-engine/blob/7f6610a902befcb84fc47e5c82a161e3d3184ce4/audit/02-static-audit.md#L17)
- **Requirements Verified:** CORR-001 requires test design artifacts before coding tests

### Class B (Referential Evidence)

**Scope Inventory** (SHA: [`119a096`](https://github.com/ImmortalDemonGod/mastery-engine/tree/119a096fb513e8ed6bddff73a4e254be6615ff9c))

- [`tests/engine/schemas.bug-catalog.md#L1-L146`](https://github.com/ImmortalDemonGod/mastery-engine/blob/119a096fb513e8ed6bddff73a4e254be6615ff9c/tests/engine/schemas.bug-catalog.md#L1-L146)

### Class A (Execution Evidence)

- Local checks skipped (--skip-checks).
- **Skip reason:** Markdown documentation file; no code to lint/type-check/test


---

## Verification Methodology

**R0 (trivial) -- local checks skipped.**
**Reason:** Markdown documentation file; no code to lint/type-check/test
Only git diff scope inventory was collected. No execution evidence.

---

## Summary

Bug catalog for CORR-001: UserProgress.mark_stage_complete stores synthetic index instead of real module id
