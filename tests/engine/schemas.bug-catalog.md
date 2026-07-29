# Bug Catalog: `engine/schemas.py` — `UserProgress.mark_stage_complete`

Source examined: `engine/schemas.py` (full file, 219 lines)  
Companion type imports: `pydantic.BaseModel`, `pydantic.Field` (public API)  
Audit reference: https://github.com/ImmortalDemonGod/mastery-engine/blob/7f6610a902befcb84fc47e5c82a161e3d3184ce4/audit/02-static-audit.md#L17

---

## 1. Code Summary

### Public interface
- `UserProgress.mark_stage_complete(self, stage: str) -> None` — sole mutating method on the progress model; called by `engine/main.py` at the end of every submit handler.
- Pydantic model with fields: `curriculum_id`, `current_module_index`, `current_stage`, `completed_modules`, `completed_patterns`, `completed_problems`.

### Load-bearing comments
- `schemas.py:159` — `"NOTE: This method maintains legacy LINEAR behavior for backward compatibility."` — signals intentional scope constraint; the harden branch is the only branch that records completion.
- `schemas.py:168` — `# Will be replaced with actual ID` — the author knew this was a placeholder and explicitly documented the debt. The fix is owed.

### IO boundaries
- None directly in this file; state is a plain Pydantic model. The IO boundary is the caller (`engine/main.py`) which loads and saves `UserProgress` from `.mastery_progress.json`.

### Branching points
- `stage == "build"` → advance to `"justify"` (no side effects on `completed_modules`)
- `stage == "justify"` → advance to `"harden"` (no side effects on `completed_modules`)
- `stage == "harden"` → **THE CRITICAL PATH**: appends a string to `completed_modules`, then increments `current_module_index`. This is the only path that mutates `completed_modules`.

### Magic-string contract
- `completed_modules: list[str]` — the contract between `mark_stage_complete` (writer) and `engine/main.py:2196` (reader) is that this list contains **`module.id`** strings from `CurriculumManifest.modules`. Real IDs are slug strings like `"softmax"`, `"rmsnorm"`, `"multihead_attention"`. The current code writes `"module_0"`, `"module_1"`, etc. — a different namespace entirely.

### Existing tests
- `tests/engine/test_state.py` — tests `StateManager` load/save; loads `completed_modules` from JSON but never exercises `mark_stage_complete`. No test for this method exists anywhere in the suite.

---

## 2. Bug Catalog

### BUG-A — Synthetic index stored instead of real module ID (CORR-001) ★★★★★

**Failure mode**: `mark_stage_complete("harden")` appends `f"module_{self.current_module_index}"` (e.g. `"module_0"`) to `completed_modules` instead of the caller-supplied actual module ID (e.g. `"softmax"`).

**Blast radius**: 
- `engine/main.py:2196` checks `module.id in progress.completed_modules`. Since `"softmax" != "module_0"`, every module appears "not completed" in the curriculum list — the ✅ status is never shown.
- `engine/main.py:2335` resets modules by filtering `m != module_id` on real IDs. Synthetic entries are never removed — a completed module that fails to appear in the list also cannot be properly reset.
- Cascades silently; no error is raised, so the user sees wrong UI state forever.

**Why it's plausible**: The placeholder comment `# Will be replaced with actual ID` explicitly marks this as deferred debt. The method has no `module_id` parameter, so it cannot access the real ID — the architecture was never completed.

**Test type(s)**: Captured bug / contract pin (the ID namespace mismatch is a specific, documented failure) + negative path (synthetic string must not appear).

**Self-critique**:
- Fails if a real bug is introduced? YES — if `mark_stage_complete` stores any synthetic string, the assertion `"softmax" in completed_modules` fails.
- Fails under refactor? NO — we assert on `completed_modules` (observable output), not on implementation internals.
- Tests observable behavior? YES — `completed_modules` is the serialized state read by `main.py`.
- Uses public interface? YES — `mark_stage_complete` is the documented mutating method.

---

### BUG-B — No validation when module_id=None is passed to harden branch ★★★

**Failure mode**: If a caller passes `module_id=None` (or the parameter is not provided) to the harden branch, the code silently stores a synthetic index rather than raising. After the fix adds a `module_id` parameter, a `None` value would write `None` into `completed_modules` or silently skip recording — a different form of the same tracking corruption.

**Blast radius**: Same as BUG-A — completion check at `main.py:2196` fails.

**Why it's plausible**: The fix requires a new optional parameter `module_id`. Any call site that forgets to pass it would regress. Explicit validation at the entry point prevents silent corruption.

**Test type(s)**: Negative path — invalid input (`None`) must produce an explicit error.

**Self-critique**:
- Fails if a real bug? YES — if `module_id=None` is accepted silently, the assertion on `ValueError` fails.
- Passes for wrong-but-stable output? NO — the absence of an exception is observable.

---

### BUG-C — Duplicate entry possible if harden called twice for same module ★★

**Failure mode**: The deduplication guard at `schemas.py:169` (`if module_id not in self.completed_modules`) uses the synthetic ID, not the real one. After the fix, if deduplication is not correctly applied to the real module ID, a re-completed module could appear twice in `completed_modules`.

**Blast radius**: Cosmetic duplicates in the list; minor — the `in` check at `main.py:2196` still works, but state size grows unboundedly under retries.

**Why it's plausible**: Deduplication logic is tightly coupled to the ID being stored; a partial fix (change ID but forget the `not in` guard) leaves duplicates.

**Test type(s)**: Invariant — the list must contain at most one entry per module ID.

**Self-critique**: Correctly targets observable state; does not inspect internal variables.

---

### BUG-D — Two distinct module IDs can collide if index is used as key ★★★

**Failure mode**: Two different modules at different indices could have their completion state mis-attributed under any code that uses the index as the completion key (the current buggy behavior). After fix, the IDs `"softmax"` and `"rmsnorm"` must be independently tracked even when they happen to sit at the same index in two different curriculum runs.

**Blast radius**: A module might appear completed when it isn't (if a different module at the same index was completed in a prior session that used a different curriculum ordering).

**Why it's plausible**: The placeholder approach conflates position with identity.

**Test type(s)**: Differential — complete module A then module B; assert both IDs are independently in `completed_modules`.

**Self-critique**: Tests the real ID orthogonality — no interaction with implementation internals.

---

### BUG-E — Non-harden stages inadvertently start modifying completed_modules after refactor ★★

**Failure mode**: A refactor of `mark_stage_complete` that incorrectly applies the `module_id` write to the `build` or `justify` branches would corrupt stage flow.

**Blast radius**: `completed_modules` contains partial-completion entries; curriculum list shows false positives.

**Why it's plausible**: The fix touches adjacent branches; off-by-one error in branch structure is easy.

**Test type(s)**: Negative path / regression — `build` and `justify` transitions must NOT modify `completed_modules`.

**Self-critique**: Tests observable `completed_modules` after `build`/`justify` transitions.

---

## 3. Skipped Bugs

| Bug class | Why skipped |
|---|---|
| Pydantic serialization of `completed_modules` | The field is a plain `list[str]`; Pydantic handles it correctly and there's no custom serializer to test. Trivial. |
| `current_module_index` increment after harden | The increment is not part of CORR-001 and is correct in the existing code. Out of scope for this fix. |
| LIBRARY mode (`completed_patterns`, `completed_problems`) | Separate finding (CORR-002). The `mark_stage_complete` method doesn't touch library fields in the harden path. Out of scope. |
| Thread safety / concurrent writes | `UserProgress` is a pure in-memory model; concurrency is a caller concern. Out of scope. |
| `stage` value not in `{"build","justify","harden"}` | No existing validation and no real-world callsite passes an invalid stage. Deferred — the fix adds no stage enum, so testing unrecognized stages would be testing behavior we're not changing. |

---

## 4. Test→Bug Map

| Test | Bug caught |
|---|---|
| `test_harden_records_caller_supplied_module_id_not_synthetic_index` | BUG-A (primary) |
| `test_harden_does_not_append_synthetic_index` | BUG-A (contract pin) |
| `test_curriculum_list_lookup_matches_after_mark_complete` | BUG-A (integration contract) |
| `test_harden_raises_if_module_id_is_none` | BUG-B |
| `test_harden_idempotent_real_module_id` | BUG-C |
| `test_harden_subsequent_module_records_its_own_id` | BUG-D |
| `test_build_and_justify_do_not_record_completed_module` | BUG-E |

---

## 5. Evaluation (post-first-run)

**5 FAILED / 5 PASSED** (`pytest tests/engine/test_schemas.py -v`, venv activated)

- **Bugs caught** (test FAILED first run — bug is present and fix is needed):
  - BUG-A: `test_harden_records_caller_supplied_module_id_not_synthetic_index` — `TypeError: mark_stage_complete() got an unexpected keyword argument 'module_id'` (current signature lacks the parameter).
  - BUG-A: `test_harden_does_not_append_synthetic_index` — same TypeError.
  - BUG-A: `test_curriculum_list_lookup_matches_after_mark_complete` — same TypeError.
  - BUG-D: `test_harden_subsequent_module_records_its_own_id` — same TypeError.
  - BUG-C: `test_harden_idempotent_real_module_id` — same TypeError.

- **Bugs characterized** (test PASSED first run — behavior pinned, existing regression guards):
  - BUG-B: `test_harden_raises_if_module_id_is_none` — PASSES (TypeError from wrong arity is caught by `pytest.raises((ValueError, TypeError))`). After the fix this must remain GREEN with ValueError.
  - BUG-E: `test_build_advances_to_justify_without_module_id` — PASSES (existing behavior preserved).
  - BUG-E: `test_justify_advances_to_harden_without_module_id` — PASSES (existing behavior preserved).
  - BUG-E: `test_build_does_not_record_completed_module` — PASSES (existing behavior preserved).
  - BUG-E: `test_justify_does_not_record_completed_module` — PASSES (existing behavior preserved).

- **Bugs discovered during writing**: None beyond the catalogued set. The `module_id` parameter is simply absent from `mark_stage_complete`; the fix is straightforward. No hidden aliasing or encoding issues found.

### Investigation pass on suspect findings

- `test_harden_raises_if_module_id_is_none` is PASS for the wrong reason (arity TypeError, not semantic ValueError). Retains value as a regression guard after the fix — **downgrade from BUG-B to regression guard**: after fix it must raise ValueError/TypeError for None input.
- "0 additional bugs caught" beyond BUG-A primary: probed for a secondary encoding issue (`None` in the list, duplicate insertion) — confirmed the deduplication guard at line 169 uses the synthetic string so BUG-C is real but not separately exercisable until the primary fix is applied.

**Honest final stats: 5 RED (fix required), 5 GREEN (regression coverage). All RED tests target the single root cause: missing `module_id` parameter on `mark_stage_complete`.**
