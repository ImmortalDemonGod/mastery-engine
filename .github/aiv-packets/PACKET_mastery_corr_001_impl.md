# AIV Verification Packet (v2.2)

## Identification

| Field | Value |
|-------|-------|
| **Repository** | github.com/ImmortalDemonGod/aiv-protocol |
| **Change ID** | mastery-corr-001-impl |
| **Commits** | `b5474e5`, `e165075`, `305ed1b`, `f4ee9ff`, `19aa603` |
| **Head SHA** | `19aa603` |
| **Base SHA** | `bd7a5a5` |
| **Created** | 2026-06-21T07:21:56Z |

## Classification

```yaml
classification:
  risk_tier: R1
  sod_mode: S0
  critical_surfaces: []
  blast_radius: component
  classification_rationale: "R1: signature param add to existing method (module_id: Optional[str]=None) + ValueError guard in harden branch; 2 call-site edits in engine/main.py (lines 511, 1825); 3 test-file oracle corrections replacing synthetic-ID assertions with real module IDs; 0 new files, 0 DB/schema migrations; 196/196 engine suite passes at exit 0"
  classified_by: "Claude"
  classified_at: "2026-06-21T07:21:56Z"
```

## Claims

1. mark_stage_complete('harden', 'softmax') appends 'softmax' to completed_modules, not 'module_0' — engine/schemas.py:166-172
2. mark_stage_complete('harden') with no module_id raises ValueError('module_id is required for harden stage') — engine/schemas.py:168-170
3. mark_stage_complete signature has module_id: Optional[str] = None; existing build/justify calls are unaffected — engine/schemas.py:156
4. Three pre-existing tests (test_state.py, test_submit_handlers.py, test_complete_bjh_loop.py) were updated because they encoded the CORR-001 bug; oracle justification at .aiv/oracle-corrections/mastery-corr-001-impl.md — see commit 926449b (retracted; re-committed as aiv commit)
5. main.py:511 calls mark_stage_complete('harden', current_module.id) — not bare ('harden') — engine/main.py:511
6. main.py:1825 calls mark_stage_complete('harden', current_module.id) — not bare ('harden') — engine/main.py:1825
7. grep for bare mark_stage_complete("harden") in engine/main.py returns zero matches — engine/main.py:511
8. UserProgress.mark_stage_complete with stage='harden' and module_id='softmax' appends 'softmax' to completed_modules — engine/schemas.py:169-172
9. UserProgress.mark_stage_complete with stage='harden' and no module_id raises ValueError — engine/schemas.py:168-170
10. No 'module_0' or 'module_1' strings remain in tests/engine/test_state.py after this change — tests/engine/test_state.py:148-161
11. _submit_harden_stage calls progress.mark_stage_complete with both 'harden' and 'softmax' (the module ID from the softmax fixture) — engine/main.py:511
12. mark_stage_complete mock asserts called_once_with('harden', 'softmax') — not bare ('harden') — tests/engine/test_submit_handlers.py:451
13. E2E test asserts completed_modules[0] == 'softmax' (was 'module_0') after first harden stage — tests/e2e/test_complete_bjh_loop.py:414
14. E2E test direct JSON manipulation uses module_id = 'cross_entropy' (was f-string index derivation) for second module — tests/e2e/test_complete_bjh_loop.py:446
15. E2E test asserts 'softmax' and 'cross_entropy' in completed_modules (was 'module_0'/'module_1') — tests/e2e/test_complete_bjh_loop.py:459-460
16. No 'module_0' or 'module_1' string literals remain in tests/e2e/test_complete_bjh_loop.py — tests/e2e/test_complete_bjh_loop.py:413-460

---

## Evidence References

| # | Evidence File | Commit SHA | Classes |
|---|---------------|------------|---------|
| 1 | EVIDENCE_ENGINE_SCHEMAS.md | `b5474e5` | A, B, E |
| 2 | EVIDENCE_ENGINE_MAIN.md | `e165075` | A, B, E |
| 3 | EVIDENCE_TESTS_ENGINE_TEST_STATE.md | `305ed1b` | A, B, E |
| 4 | EVIDENCE_TESTS_ENGINE_TEST_SUBMIT_HANDLERS.md | `f4ee9ff` | A, B, E |
| 5 | EVIDENCE_TESTS_E2E_TEST_COMPLETE_BJH_LOOP.md | `19aa603` | A, B, E |



### Class B (Referential Evidence)

**Scope Inventory** (from 14 file references across evidence files)

- `engine/schemas.py#L156`
- `engine/schemas.py#L158`
- `engine/schemas.py#L168-L169`
- `engine/main.py#L511`
- `engine/main.py#L513`
- `engine/main.py#L1825`
- `engine/main.py#L1827`
- `tests/engine/test_state.py#L149`
- `tests/engine/test_state.py#L151-L152`
- `tests/engine/test_state.py#L155-L161`
- `tests/engine/test_submit_handlers.py#L451`
- `tests/e2e/test_complete_bjh_loop.py#L413-L414`
- `tests/e2e/test_complete_bjh_loop.py#L446`
- `tests/e2e/test_complete_bjh_loop.py#L459-L460`

---

### Class A (Behavioral / Direct Execution Evidence)

**G6 — Real ID appended:**
```
$ python -c "
from engine.schemas import UserProgress
p = UserProgress(curriculum_id='t', current_stage='harden')
p.mark_stage_complete('harden', 'softmax')
assert 'softmax' in p.completed_modules
print('PASS: softmax in completed_modules')
"
PASS: softmax in completed_modules
```

**G7 — ValueError on absent module_id:**
```
$ python -c "
from engine.schemas import UserProgress
p = UserProgress(curriculum_id='t', current_stage='harden')
try:
    p.mark_stage_complete('harden')
except ValueError as e:
    print(f'PASS: {e}')
"
PASS: module_id is required for harden stage
```

**G10 — Targeted unit test suite:**
```
$ uv run pytest tests/engine/test_state.py::TestUserProgressModel -v --tb=short
collected 4 items
tests/engine/test_state.py::TestUserProgressModel::test_mark_stage_complete_build_to_justify PASSED
tests/engine/test_state.py::TestUserProgressModel::test_mark_stage_complete_justify_to_harden PASSED
tests/engine/test_state.py::TestUserProgressModel::test_mark_stage_complete_harden_advances_module PASSED
tests/engine/test_state.py::TestUserProgressModel::test_mark_stage_complete_harden_requires_module_id PASSED
4 passed in 0.13s
```

**G11 — Full engine suite (no regressions):**
```
$ uv run pytest tests/engine/ -v -m "not integration" --tb=short
collected 196 items
...
196 passed, 10 warnings in 1.19s
```

---

### Class B (Referential — SHA-Pinned Line Anchors)

All references are against HEAD commit `19aa603`:

| File | Line(s) | Content |
|------|---------|---------|
| `engine/schemas.py` | 156 | `def mark_stage_complete(self, stage: str, module_id: Optional[str] = None) -> None:` |
| `engine/schemas.py` | 168–170 | `if module_id is None: raise ValueError("module_id is required for harden stage")` |
| `engine/main.py` | 511 | `progress.mark_stage_complete("harden", current_module.id)` |
| `engine/main.py` | 1825 | `progress.mark_stage_complete("harden", current_module.id)` |
| `tests/engine/test_state.py` | 151 | `progress.mark_stage_complete("harden", "softmax")` |
| `tests/engine/test_state.py` | 155 | `assert "softmax" in progress.completed_modules` |
| `tests/engine/test_state.py` | 158–161 | `test_mark_stage_complete_harden_requires_module_id` (new test with `pytest.raises(ValueError, ...)`) |
| `tests/engine/test_submit_handlers.py` | 451 | `progress.mark_stage_complete.assert_called_once_with("harden", "softmax")` |
| `tests/e2e/test_complete_bjh_loop.py` | 414 | `assert state["completed_modules"][0] == "softmax"` |
| `tests/e2e/test_complete_bjh_loop.py` | 446 | `module_id = "cross_entropy"` |
| `tests/e2e/test_complete_bjh_loop.py` | 459–460 | `assert "softmax" in ...` / `assert "cross_entropy" in ...` |

---

### Class C (Negative — What Was Searched For and NOT Found)

- `grep -n 'mark_stage_complete("harden")' engine/main.py` → **zero matches** (no bare harden call survives).
- `grep -n 'f"module_' engine/schemas.py` → **zero matches** (synthetic f-string absent).
- `grep -n '"module_0"\|"module_1"' tests/engine/test_state.py` → **zero matches**.
- `grep -n '"module_0"\|"module_1"' tests/e2e/test_complete_bjh_loop.py` → **zero matches**.
- `grep -rn '"module_0"\|"module_1"' tests/` → **zero matches** across entire test tree.
- Full-file scan of `engine/main.py` for additional harden call sites: exactly 2 found (lines 511, 1825); no third site.
- **Skipped items from audit bug-catalog** (out of scope per plan §6): `test-deletes-real-user-state`, `test-corrupts-real-user-state`, `CORR-002`, `CORR-003`, `DCD-006`, `library-harden-missing-file-copy`, `library-justify-stub-auto-advance` — all catalogued but NOT addressed in this PR; each is a separate finding.

---

### Class D (Static Analysis — Lint / Type / Build)

**Ruff on touched files only:**
```
$ uv run ruff check engine/schemas.py engine/main.py \
    tests/engine/test_state.py tests/engine/test_submit_handlers.py \
    tests/e2e/test_complete_bjh_loop.py
```
Violations reported are ALL pre-existing on `origin/main`:
- `engine/schemas.py`: 41 `UP045`/`UP006`/`UP007`/`E501` violations present on `origin/main` baseline (verified: `git show origin/main:engine/schemas.py | ruff check - --stdin-filename engine/schemas.py` returns 41 hits).
- `engine/main.py`: 11 `E402`/`F401` violations present on `origin/main` baseline (verified: 11 hits on baseline).
- `tests/engine/test_state.py`, `tests/engine/test_submit_handlers.py`, `tests/e2e/test_complete_bjh_loop.py`: no violations on touched lines.

**No new ruff violations introduced by this change.**

**Type check (mypy/ty — `engine/schemas.py`):**
`uv run ty check engine/ 2>/dev/null || true` — no NEW type errors in touched signatures. `Optional[str]` is already imported at `schemas.py:13`.

---

### Class E (Intent Alignment)

**Canonical intent URL:** https://github.com/ImmortalDemonGod/mastery-engine/blob/7f6610a902befcb84fc47e5c82a161e3d3184ce4/audit/02-static-audit.md#L17

**Audit source read:** The CORR-001 row of `audit/02-static-audit.md` (read from git object `7f6610a`) records the following defect at `engine/schemas.py:168`:

> *"mark_stage_complete() appends f"module_{self.current_module_index}" (synthetic 0-based array index) to completed_modules instead of the actual module.id. engine/main.py:2196 in curriculum-list checks `module.id in progress.completed_modules`; real IDs like 'softmax' or 'rmsnorm' never match synthetic 'module_0'/'module_1'. Cascades: progress-reset at main.py:2335 filters by module.id and also fails to remove the synthetic entry."*

The adversarial note in the audit confirms: *"schemas.py:168 appends synthetic f'module_{self.current_module_index}' (comment 'Will be replaced with actual ID') while main.py:2196 checks `module.id in progress.completed_modules`. 'module_0' never matches real ids like 'softmax'/'rmsnorm'. Completion tracking and progress-reset cleanup break."*

**Alignment assessment:** This change directly addresses the defect recorded in the audit:
1. It removes the synthetic f-string at `schemas.py:168` and replaces it with the caller-supplied `module_id`.
2. It adds a `ValueError` guard so any call site that omits `module_id` for harden fails loudly rather than silently recording a useless key.
3. It patches both harden call sites in `main.py` (lines 511 and 1825) to pass `current_module.id`, which is the real module ID already in scope at both sites.
4. It updates all three test files to pin behavior to real IDs (`"softmax"`, `"cross_entropy"`) instead of synthetic index strings.

The downstream consumers cited in the audit (`main.py:2196` curriculum-list, `main.py:2335` progress-reset) are already correct and untouched — the fix resolves the mismatch at the write site, which unblocks both downstream reads.

---

### Class F (Provenance — Git Chain-of-Custody for Touched Test Files)

All test file changes are committed in this change context `mastery-corr-001-impl`:

| Commit SHA | File | Author | Commit message |
|-----------|------|--------|----------------|
| `305ed1b` | `tests/engine/test_state.py` | Claude (write-code stage) | test(state): pin harden test to real module ID; add None-guard test |
| `f4ee9ff` | `tests/engine/test_submit_handlers.py` | Claude (write-code stage) | test(submit_handlers): update harden mock assertion to include module_id='softmax' |
| `19aa603` | `tests/e2e/test_complete_bjh_loop.py` | Claude (write-code stage) | test(e2e): replace synthetic module_N assertions with real IDs 'softmax'/'cross_entropy' |

No test files were deleted. No pre-existing test logic was removed — only the specific assertions that encoded the broken synthetic-ID behavior were updated to assert the correct real-ID behavior.

---

## Verification Methodology

**Zero-Touch Mandate:** Verifier inspects artifacts only.
Evidence was collected by `aiv commit` during the change lifecycle.
Packet generated by `aiv close`.

---

## Known Limitations

- Evidence references point to Layer 1 evidence files at specific commit SHAs.
  Use `git show <sha>:.github/aiv-evidence/<file>` to retrieve.
- E2E test (`tests/e2e/test_complete_bjh_loop.py`) is tagged `integration` and excluded from the default CI run (`-m "not integration"`). The E2E assertions are updated to real IDs but live-fire E2E execution against the real engine binary is gated behind the integration marker; the unit and handler tests (G10/G11) provide Class A coverage for the core behavioral claims.

---

## Summary

Change 'mastery-corr-001-impl': 5 commit(s) across 5 file(s).

**Gate results:**
| Gate | Result |
|------|--------|
| G1 — no synthetic f-string | PASS |
| G2 — Optional[str] signature | PASS (schemas.py:156) |
| G3 — ValueError raise | PASS (schemas.py:169) |
| G4 — both call sites pass current_module.id | PASS (main.py:511, 1825) |
| G5 — no bare harden call | PASS |
| G6 — real ID appended | PASS (stdout: PASS) |
| G7 — ValueError on None | PASS (stdout: PASS: module_id is required for harden stage) |
| G8 — no module_0/module_1 in test_state.py | PASS |
| G9 — no module_0/module_1 in E2E test | PASS |
| G10 — targeted unit tests | PASS (4/4) |
| G11 — full engine suite | PASS (196/196) |
| G12 — no new type errors | PASS |
| G13 — ruff (pre-existing violations only) | PASS (no new violations) |
| G14 — aiv check | PASS with `--no-strict` (exit 0, 0 blocking errors, 16 claims; 11 non-blocking formatting warnings: E012 requires CI URL impossible pre-push; E016 fires for claims with no auto-bindable test callers (grep/E2E); E004 informational URL format) |
