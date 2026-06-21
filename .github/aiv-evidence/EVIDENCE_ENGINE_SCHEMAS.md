# AIV Evidence File (v1.0)

**File:** `engine/schemas.py`
**Commit:** `bd7a5a5`
**Generated:** 2026-06-21T07:19:09Z
**Protocol:** AIV v2.0 + Addendum 2.7 (Zero-Touch Mandate)

---

## Classification (required)

```yaml
classification:
  risk_tier: R1
  sod_mode: S0
  critical_surfaces: []
  blast_radius: "engine/schemas.py"
  classification_rationale: "R1: single-method signature change + branch-body replacement in one model file; no new infrastructure"
  classified_by: "Claude"
  classified_at: "2026-06-21T07:19:09Z"
```

## Claim(s)

1. mark_stage_complete('harden', 'softmax') appends 'softmax' to completed_modules, not 'module_0'
2. mark_stage_complete('harden') with no module_id raises ValueError('module_id is required for harden stage')
3. mark_stage_complete signature has module_id: Optional[str] = None; existing build/justify calls are unaffected
4. No existing tests were modified or deleted during this change.

---

## Evidence

### Class E (Intent Alignment)

- **Link:** [https://github.com/ImmortalDemonGod/mastery-engine/blob/7f6610a902befcb84fc47e5c82a161e3d3184ce4/audit/02-static-audit.md#L17](https://github.com/ImmortalDemonGod/mastery-engine/blob/7f6610a902befcb84fc47e5c82a161e3d3184ce4/audit/02-static-audit.md#L17)
- **Requirements Verified:** audit L17: harden branch must record the caller-supplied real module.id, not a synthetic f-string index

### Class B (Referential Evidence)

**Scope Inventory** (SHA: [`bd7a5a5`](https://github.com/ImmortalDemonGod/mastery-engine/tree/bd7a5a59cb82214b20688e4b30e241e17f13fcff))

- [`engine/schemas.py#L156`](https://github.com/ImmortalDemonGod/mastery-engine/blob/bd7a5a59cb82214b20688e4b30e241e17f13fcff/engine/schemas.py#L156)
- [`engine/schemas.py#L158`](https://github.com/ImmortalDemonGod/mastery-engine/blob/bd7a5a59cb82214b20688e4b30e241e17f13fcff/engine/schemas.py#L158)
- [`engine/schemas.py#L168-L169`](https://github.com/ImmortalDemonGod/mastery-engine/blob/bd7a5a59cb82214b20688e4b30e241e17f13fcff/engine/schemas.py#L168-L169)

### Class A (Execution Evidence)

**Per-symbol test coverage (AST analysis):**

- **`UserProgress`** (L156): PASS -- 87 test(s) call `UserProgress` directly
  - `tests/engine/test_init_cleanup.py::test_init_already_initialized`
  - `tests/engine/test_new_cli_commands.py::test_show_current_module_build_stage`
  - `tests/engine/test_new_cli_commands.py::test_show_current_module_justify_stage`
  - `tests/engine/test_new_cli_commands.py::test_show_current_module_harden_stage_shows_instructions`
  - `tests/engine/test_new_cli_commands.py::test_show_specific_module_by_id`
  - `tests/engine/test_new_cli_commands.py::test_show_nonexistent_module_shows_error`
  - `tests/engine/test_new_cli_commands.py::test_start_challenge_in_harden_stage_succeeds`
  - `tests/engine/test_new_cli_commands.py::test_start_challenge_in_build_stage_shows_error`
  - `tests/engine/test_new_cli_commands.py::test_curriculum_list_shows_all_modules_with_status`
  - `tests/engine/test_new_cli_commands.py::test_progress_reset_completed_module_with_confirmation`
- **`UserProgress.mark_stage_complete`** (L158): PASS -- 13 test(s) call `mark_stage_complete` directly
  - `tests/engine/test_schemas.py::test_harden_records_caller_supplied_module_id_not_synthetic_index`
  - `tests/engine/test_schemas.py::test_harden_does_not_append_synthetic_index`
  - `tests/engine/test_schemas.py::test_curriculum_list_lookup_matches_after_mark_complete`
  - `tests/engine/test_schemas.py::test_harden_subsequent_module_records_its_own_id`
  - `tests/engine/test_schemas.py::test_harden_idempotent_real_module_id`
  - `tests/engine/test_schemas.py::test_harden_raises_if_module_id_is_none`
  - `tests/engine/test_schemas.py::test_build_advances_to_justify_without_module_id`
  - `tests/engine/test_schemas.py::test_justify_advances_to_harden_without_module_id`
  - `tests/engine/test_schemas.py::test_build_does_not_record_completed_module`
  - `tests/engine/test_schemas.py::test_justify_does_not_record_completed_module`

**Coverage summary:** 2/2 symbols verified by tests.

### Code Quality (Linting & Types)

- **ruff:** 0 error(s)
- **mypy:** 

## Claim Verification Matrix

| # | Claim | Type | Evidence | Verdict |
|---|-------|------|----------|---------|
| 1 | mark_stage_complete('harden', 'softmax') appends 'softmax' t... | symbol | 13 test(s) call `UserProgress.mark_stage_complete` | PASS VERIFIED |
| 2 | mark_stage_complete('harden') with no module_id raises Value... | symbol | 13 test(s) call `UserProgress.mark_stage_complete` | PASS VERIFIED |
| 3 | mark_stage_complete signature has module_id: Optional[str] =... | symbol | 13 test(s) call `UserProgress.mark_stage_complete` | PASS VERIFIED |
| 4 | No existing tests were modified or deleted during this chang... | structural | Class C not collected | REVIEW MANUAL REVIEW |

**Verdict summary:** 3 verified, 0 unverified, 1 manual review.
---

## Verification Methodology

**Zero-Touch Mandate:** Verifier inspects artifacts only.
Evidence collected by `aiv commit` running: git diff (scope inventory), AST symbol-to-test binding (2/2 symbols verified).
Ruff/mypy results are in Code Quality (not Class A) because they prove syntax/types, not behavior.

---

## Summary

Fix mark_stage_complete harden branch to record real module ID instead of synthetic index-derived key
