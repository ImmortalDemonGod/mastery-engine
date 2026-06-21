# AIV Verification Packet (v2.2)

## Identification

| Field | Value |
|-------|-------|
| **Repository** | github.com/ImmortalDemonGod/aiv-protocol |
| **Change ID** | mastery-corr-001-tests |
| **Commits** | `828f2c9`, `fb7978b`, `5b46944` |
| **Head SHA** | `5b46944` |
| **Base SHA** | `119a096` |
| **Created** | 2026-06-21T07:02:16Z |

## Classification

```yaml
classification:
  risk_tier: R1
  sod_mode: S0
  critical_surfaces: []
  blast_radius: component
  classification_rationale: "TODO: Describe why this tier was chosen"
  classified_by: "Claude"
  classified_at: "2026-06-21T07:02:16Z"
```

## Claims

1. schemas.bug-catalog.md evaluation section records 5 RED and 5 GREEN tests with root-cause analysis
2. No existing tests were modified or deleted during this change.
3. 5 tests fail with TypeError proving mark_stage_complete lacks module_id parameter (CORR-001)
4. 5 regression tests pass — no regressions: build/justify transitions are not broken

---

## Evidence References

| # | Evidence File | Commit SHA | Classes |
|---|---------------|------------|---------|
| 1 | EVIDENCE_TESTS_ENGINE_SCHEMAS.BUG_CATALOG.MD.md | `828f2c9` | A, B, E |
| 2 | EVIDENCE_TESTS_ENGINE_TEST_SCHEMAS.md | `fb7978b` | A, B, E, F |
| 3 | EVIDENCE_TESTS_ENGINE_SCHEMAS.BUG_CATALOG.MD.md | `5b46944` | A, B, E |



### Class A (Behavioral / Direct Execution)

**Claim 1:** [pytest run confirming 5 RED + 5 GREEN tests at commit fb7978b](https://github.com/ImmortalDemonGod/mastery-engine/commit/fb7978b17a28e831994b4c77af4c1021da727402)

`pytest tests/engine/test_schemas.py -v` (venv `.venv`, Python 3.11.15):

```
FAILED TestMarkStageCompleteHardenRecordsRealId::test_harden_records_caller_supplied_module_id_not_synthetic_index
  TypeError: UserProgress.mark_stage_complete() got an unexpected keyword argument 'module_id'
FAILED TestMarkStageCompleteHardenRecordsRealId::test_harden_does_not_append_synthetic_index
  TypeError: UserProgress.mark_stage_complete() got an unexpected keyword argument 'module_id'
FAILED TestMarkStageCompleteHardenRecordsRealId::test_curriculum_list_lookup_matches_after_mark_complete
  TypeError: UserProgress.mark_stage_complete() got an unexpected keyword argument 'module_id'
FAILED TestMarkStageCompleteHardenRecordsRealId::test_harden_subsequent_module_records_its_own_id
  TypeError: UserProgress.mark_stage_complete() got an unexpected keyword argument 'module_id'
FAILED TestMarkStageCompleteHardenRecordsRealId::test_harden_idempotent_real_module_id
  TypeError: UserProgress.mark_stage_complete() got an unexpected keyword argument 'module_id'
PASSED TestMarkStageCompleteHardenValidation::test_harden_raises_if_module_id_is_none
PASSED TestMarkStageCompleteNonHardenRegressions::test_build_advances_to_justify_without_module_id
PASSED TestMarkStageCompleteNonHardenRegressions::test_justify_advances_to_harden_without_module_id
PASSED TestMarkStageCompleteNonHardenRegressions::test_build_does_not_record_completed_module
PASSED TestMarkStageCompleteNonHardenRegressions::test_justify_does_not_record_completed_module
5 failed, 5 passed in 0.14s
```

Root cause of RED: `engine/schemas.py:156` — `mark_stage_complete(self, stage: str)` has no `module_id` parameter. All 5 RED tests call `mark_stage_complete("harden", module_id="softmax")` and fail immediately with `TypeError`. This is correct: the tests are RED because the bug is present.

---

### Class B (Referential Evidence)

**Claim 3:** [tests/engine/test_schemas.py#L1–L113 at fb7978b](https://github.com/ImmortalDemonGod/mastery-engine/blob/fb7978b17a28e831994b4c77af4c1021da727402/tests/engine/test_schemas.py#L1-L113)

**Scope Inventory** (from 3 file references across evidence files)

- `tests/engine/schemas.bug-catalog.md#L142`
- `tests/engine/schemas.bug-catalog.md#L144-L167`
- `tests/engine/test_schemas.py#L1-L113`

### Class C (Negative / Skipped Set)

**Claim 4:** Does not contain any pre-existing tests for `mark_stage_complete` — absence confirmed by grep.

Searched for any existing tests covering `mark_stage_complete`:
- `grep -r "mark_stage_complete" tests/` → **0 hits** before this change.
- `grep -r "completed_modules" tests/engine/test_state.py` → `test_load_valid_file` at line 38 reads `completed_modules` from a JSON dict but never calls `mark_stage_complete`.
- No test in the suite exercises the harden branch of `mark_stage_complete` or validates the content of `completed_modules` after a harden transition.

Bug-catalog Skipped set (bugs considered but not tested):
- Pydantic serialization of `completed_modules` — trivial, no custom serializer.
- `current_module_index` increment — correct in existing code, out of scope for CORR-001.
- LIBRARY mode (`completed_patterns`, `completed_problems`) — separate finding CORR-002.
- Thread safety — caller concern, not schema concern.
- Invalid `stage` values — no stage enum added by fix, deferred.

---

### Class D (Static Analysis)

`ruff check engine/schemas.py` (run by aiv): no violations.  
`mypy engine/schemas.py --ignore-missing-imports` (run by aiv): no errors.  
Test file `tests/engine/test_schemas.py`: clean import of `engine.schemas.UserProgress`; no syntax errors; passes `ruff check`.

---

### Class E (Intent Alignment)

**Link:** [https://github.com/ImmortalDemonGod/mastery-engine/blob/7f6610a902befcb84fc47e5c82a161e3d3184ce4/audit/02-static-audit.md#L17](https://github.com/ImmortalDemonGod/mastery-engine/blob/7f6610a902befcb84fc47e5c82a161e3d3184ce4/audit/02-static-audit.md#L17)

**Finding CORR-001 as recorded at that URL**:  
> mark_stage_complete() appends f"module_{self.current_module_index}" (synthetic 0-based array index) to completed_modules instead of the actual module.id. engine/main.py:2196 in curriculum-list checks `module.id in progress.completed_modules`; real IDs like 'softmax' or 'rmsnorm' never match synthetic 'module_0'/'module_1'. Cascades: progress-reset at main.py:2335 filters by module.id and also fails to remove the synthetic entry.

**Alignment**: Every RED test in `tests/engine/test_schemas.py` directly exercises the code path described in the finding:
- `engine/schemas.py:168` is the sole location where `completed_modules` is mutated in the harden branch.
- Tests assert on `completed_modules` content after calling `mark_stage_complete("harden", module_id=<real_id>)` — the exact interface the fix must implement.
- The integration-contract test mirrors `engine/main.py:2196`'s `module.id in progress.completed_modules` check.

This design-tests stage does NOT implement the fix; it pins the failure mode so the verify stage can confirm the fix is correct.

---

### Class F (Provenance — Test File Chain of Custody)

**Claim 2:** No existing tests were modified or deleted during this change.

New test file `tests/engine/test_schemas.py`:
- Created in commit [`fb7978b`](https://github.com/ImmortalDemonGod/mastery-engine/commit/fb7978b17a28e831994b4c77af4c1021da727402) by this pipeline stage (design-tests).
- Absence of prior version confirmed: `git log -- tests/engine/test_schemas.py` returns a single entry at `fb7978b` (does not contain any earlier commit).
- Does not touch any previously-existing test logic or modify any existing test file.
- Bug catalog `tests/engine/schemas.bug-catalog.md` created in commit `828f2c9`, evaluation section updated in `5b46944` — no pre-existing file overwritten.

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

Change 'mastery-corr-001-tests': 3 commit(s) across 2 file(s).
