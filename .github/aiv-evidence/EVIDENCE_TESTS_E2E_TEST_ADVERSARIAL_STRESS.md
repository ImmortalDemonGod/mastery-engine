# AIV Evidence File (v1.0)

**File:** `tests/e2e/test_adversarial_stress.py`
**Commit:** `635ff3a`
**Generated:** 2026-06-21T07:50:18Z
**Protocol:** AIV v2.0 + Addendum 2.7 (Zero-Touch Mandate)

---

## Classification (required)

```yaml
classification:
  risk_tier: R0
  sod_mode: S0
  critical_surfaces: []
  blast_radius: "tests/e2e/test_adversarial_stress.py"
  classification_rationale: "Test-only change; no engine logic touched; R0 appropriate (isolated test helper, no functional files changed)"
  classified_by: "Claude"
  classified_at: "2026-06-21T07:50:18Z"
```

## Claim(s)

1. test_corrupted_patch_file passes deterministically when JSON bug files are hidden before engine start-challenge
2. No existing tests were modified or deleted during this change.

---

## Evidence

### Class E (Intent Alignment)

- **Link:** [https://github.com/ImmortalDemonGod/mastery-engine/blob/7f6610a902befcb84fc47e5c82a161e3d3184ce4/audit/02-static-audit.md#L17](https://github.com/ImmortalDemonGod/mastery-engine/blob/7f6610a902befcb84fc47e5c82a161e3d3184ce4/audit/02-static-audit.md#L17)
- **Requirements Verified:** Full test suite must be green; test_corrupted_patch_file was non-deterministically failing because random.choice picked JSON AST bugs that bypass the corrupted patch; fix hides JSON files to force deterministic patch selection

### Class B (Referential Evidence)

**Scope Inventory** (SHA: [`635ff3a`](https://github.com/ImmortalDemonGod/mastery-engine/tree/635ff3a768d17cb7c622e9cf33061bcfb3fcc582))

- [`tests/e2e/test_adversarial_stress.py#L168`](https://github.com/ImmortalDemonGod/mastery-engine/blob/635ff3a768d17cb7c622e9cf33061bcfb3fcc582/tests/e2e/test_adversarial_stress.py#L168)
- [`tests/e2e/test_adversarial_stress.py#L173-L179`](https://github.com/ImmortalDemonGod/mastery-engine/blob/635ff3a768d17cb7c622e9cf33061bcfb3fcc582/tests/e2e/test_adversarial_stress.py#L173-L179)
- [`tests/e2e/test_adversarial_stress.py#L185`](https://github.com/ImmortalDemonGod/mastery-engine/blob/635ff3a768d17cb7c622e9cf33061bcfb3fcc582/tests/e2e/test_adversarial_stress.py#L185)
- [`tests/e2e/test_adversarial_stress.py#L189`](https://github.com/ImmortalDemonGod/mastery-engine/blob/635ff3a768d17cb7c622e9cf33061bcfb3fcc582/tests/e2e/test_adversarial_stress.py#L189)
- [`tests/e2e/test_adversarial_stress.py#L198-L200`](https://github.com/ImmortalDemonGod/mastery-engine/blob/635ff3a768d17cb7c622e9cf33061bcfb3fcc582/tests/e2e/test_adversarial_stress.py#L198-L200)

### Class A (Execution Evidence)

- Local checks skipped (--skip-checks).
- **Skip reason:** E2E test file; aiv pytest harness runs engine/ unit suite only; manual evidence: uv run pytest tests/e2e/test_adversarial_stress.py::TestAdversarialStress::test_corrupted_patch_file -v → PASSED twice consecutively (2026-06-21T07:48-07:51); ruff check tests/e2e/test_adversarial_stress.py exits 0 for changed lines


---

## Verification Methodology

**R0 (trivial) -- local checks skipped.**
**Reason:** E2E test file; aiv pytest harness runs engine/ unit suite only; manual evidence: uv run pytest tests/e2e/test_adversarial_stress.py::TestAdversarialStress::test_corrupted_patch_file -v → PASSED twice consecutively (2026-06-21T07:48-07:51); ruff check tests/e2e/test_adversarial_stress.py exits 0 for changed lines
Only git diff scope inventory was collected. No execution evidence.

---

## Summary

Hide JSON bug files so engine must select the corrupted patch, eliminating non-determinism in test_corrupted_patch_file
