# AIV Evidence File (v1.0)

**File:** `tests/test_cosine_schedule_validator.py`
**Commit:** `b289419`
**Generated:** 2026-06-21T02:35:31Z
**Protocol:** AIV v2.0 + Addendum 2.7 (Zero-Touch Mandate)

---

## Classification (required)

```yaml
classification:
  risk_tier: R0
  sod_mode: S0
  critical_surfaces: []
  blast_radius: "tests/test_cosine_schedule_validator.py"
  classification_rationale: "R0: test-only file; tests are intentionally RED (design-tests stage contract). Claims are about validator.sh content/behavior, verified by 'uv run pytest tests/test_cosine_schedule_validator.py -v' which reported 3 FAILED, 0 passed."
  classified_by: "Claude"
  classified_at: "2026-06-21T02:35:31Z"
```

## Claim(s)

1. validator.sh does not contain the string 'cp cs336_basics/utils.py' — the BUILD-stage cp at line 18 copies optimizer.py only, so student utils.py never propagates to the shadow worktree
2. validator.sh does not contain the string 'test_get_lr_cosine_schedule' — all three pytest invocations (lines 33/37/40) use the incorrect node ID test_lr_cosine_schedule, causing pytest to collect 0 tests and exit non-zero
3. running validator.sh from modes/developer/ with SHADOW_WORKTREE containing stale cs336_basics/utils.py leaves the stale content unchanged; validator exits 4 (file or directory not found: tests/test_optimizer.py::test_lr_cosine_schedule)
4. No existing tests were modified or deleted during this change.

---

## Evidence

### Class E (Intent Alignment)

- **Link:** [https://github.com/ImmortalDemonGod/mastery-engine/blob/7f6610a902befcb84fc47e5c82a161e3d3184ce4/audit/02-static-audit.md#L8](https://github.com/ImmortalDemonGod/mastery-engine/blob/7f6610a902befcb84fc47e5c82a161e3d3184ce4/audit/02-static-audit.md#L8)
- **Requirements Verified:** audit finding cosine-validator-wrong-file: validator must copy utils.py (not optimizer.py) and use correct test node ID test_get_lr_cosine_schedule

### Class B (Referential Evidence)

**Scope Inventory** (SHA: [`b289419`](https://github.com/ImmortalDemonGod/mastery-engine/tree/b2894195465d8aab450d96f1c595cd4a5acca814))

- [`tests/test_cosine_schedule_validator.py#L1-L135`](https://github.com/ImmortalDemonGod/mastery-engine/blob/b2894195465d8aab450d96f1c595cd4a5acca814/tests/test_cosine_schedule_validator.py#L1-L135)

### Class A (Execution Evidence)

- Local checks skipped (--skip-checks).
- **Skip reason:** Changed file IS the test file; tests are intentionally RED (design-tests stage — fix stage will make them GREEN). Evidence: 'uv run pytest tests/test_cosine_schedule_validator.py -v' → 3 FAILED, 0 passed, all for correct reasons (B1: cp optimizer.py not utils.py; B2: test_lr_cosine_schedule not found). ruff: clean (no errors after removing unused shutil import). mypy: N/A for test assertions about file content.


---

## Verification Methodology

**R0 (trivial) -- local checks skipped.**
**Reason:** Changed file IS the test file; tests are intentionally RED (design-tests stage — fix stage will make them GREEN). Evidence: 'uv run pytest tests/test_cosine_schedule_validator.py -v' → 3 FAILED, 0 passed, all for correct reasons (B1: cp optimizer.py not utils.py; B2: test_lr_cosine_schedule not found). ruff: clean (no errors after removing unused shutil import). mypy: N/A for test assertions about file content.
Only git diff scope inventory was collected. No execution evidence.

---

## Summary

Three RED tests pin bugs B1 (wrong cp file) and B2 (wrong pytest node ID) in cosine_schedule validator
