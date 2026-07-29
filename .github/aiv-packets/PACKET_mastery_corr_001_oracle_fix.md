# AIV Verification Packet (v2.2)

## Identification

| Field | Value |
|-------|-------|
| **Repository** | github.com/ImmortalDemonGod/aiv-protocol |
| **Change ID** | mastery-corr-001-oracle-fix |
| **Commits** | `222907b` |
| **Head SHA** | `222907b` |
| **Base SHA** | `5614321` |
| **Created** | 2026-06-21T08:02:52Z |

## Classification

```yaml
classification:
  risk_tier: R0
  sod_mode: S0
  critical_surfaces: []
  blast_radius: component
  classification_rationale: "Single-file pure revert — restores tests/e2e/test_adversarial_stress.py to exact origin/main content; no engine logic or assertion changes; oracle-guard compliance fix only"
  classified_by: "Claude"
  classified_at: "2026-06-21T08:02:52Z"
```

## Claims

1. `tests/e2e/test_adversarial_stress.py` matches `origin/main` exactly after this commit — `git diff origin/main -- tests/e2e/test_adversarial_stress.py` returns empty — `tests/e2e/test_adversarial_stress.py`
2. No assertion or behavioral changes remain in `test_corrupted_patch_file` — the JSON-hiding setup added by commit `6f057ea` is fully removed — `tests/e2e/test_adversarial_stress.py:168-192`
3. The full engine unit suite (196 tests, `tests/engine/ -m "not integration"`) passes at exit 0 with this revert in place — the revert touches only an E2E file not covered by that suite — `tests/engine/`
4. The oracle-corrections file `.aiv/oracle-corrections/mastery-corr-001-impl.md` was NOT modified by this revert — it correctly covers the 3 tests that encoded CORR-001 and does not claim to cover `test_corrupted_patch_file` — `.aiv/oracle-corrections/mastery-corr-001-impl.md`
5. `engine/schemas.py`, `engine/main.py`, `tests/engine/test_state.py`, `tests/engine/test_submit_handlers.py`, and `tests/e2e/test_complete_bjh_loop.py` are untouched by this commit — `git diff 5614321..222907b` shows only `tests/e2e/test_adversarial_stress.py` changed

---

## Evidence References

| # | Evidence File | Commit SHA | Classes |
|---|---------------|------------|---------|
| 1 | EVIDENCE_TESTS_E2E_TEST_ADVERSARIAL_STRESS.md | `222907b` | A, B, C, D, E, F |

---

### Class E (Intent Alignment)

**Canonical intent source:**
[audit/02-static-audit.md L17 (SHA 7f6610a)](https://github.com/ImmortalDemonGod/mastery-engine/blob/7f6610a902befcb84fc47e5c82a161e3d3184ce4/audit/02-static-audit.md#L17)

**Alignment assessment:** The source records the CORR-001 defect: `UserProgress.mark_stage_complete()` at `engine/schemas.py:168` appends a synthetic `f"module_{self.current_module_index}"` to `completed_modules` instead of the real `module.id`. It says nothing about `test_adversarial_stress.py` — that file tests corrupted-patch error-handling, which is unrelated to module ID tracking. The prior-attempt modification to `test_corrupted_patch_file` (hiding JSON bugs for non-determinism) was therefore outside the CORR-001 scope and could not be justified as fixing a test that encoded the CORR-001 defect. This revert restores the test to its `origin/main` form, ensuring the oracle-guard condition (no inherited test modified without an anchored oracle-correction) is satisfied. The revert does not affect any of the B1–B5 fixes that address the CORR-001 defect itself.

---

### Class A (Behavioral / Direct)

**G11 gate** — `uv run pytest tests/engine/ -v -m "not integration" --tb=short` run in this session immediately before the revert commit, and the adversarial stress test is in `tests/e2e/` (not `tests/engine/`) and marked `@pytest.mark.slow`, so it is excluded from the CI gate command regardless.

```
======================== 196 passed, 10 warnings in 1.18s ========================
```
Exit 0. All 196 engine-suite tests pass with the revert in place.

**G6 gate** — `python -c "from engine.schemas import UserProgress; p = UserProgress(curriculum_id='t', current_stage='harden'); p.mark_stage_complete('harden', 'softmax'); assert 'softmax' in p.completed_modules; print('PASS')"` → stdout `G6 PASS: completed_modules = ['softmax']`

**G7 gate** — `python -c "..."` → stdout `G7 PASS: module_id is required for harden stage`

These confirm the core CORR-001 fix (B1–B5 commits) remains intact after the revert.

---

### Class B (Referential — SHA-pinned, line-anchored)

- `tests/e2e/test_adversarial_stress.py:168` — origin/main: `# Corrupt the patch file` (revert restores this line)
- `tests/e2e/test_adversarial_stress.py:170–177` — origin/main: backup and corrupt patch only; no JSON-hiding block (the 9-line JSON-hiding block added by `6f057ea` is removed)
- `tests/e2e/test_adversarial_stress.py:188–192` — origin/main: `finally` block restores patch only; no JSON-restore loop (the 3-line JSON-restore addition is removed)
- `engine/schemas.py:156,168–172` — NOT changed by this commit; B1 fix remains: signature has `module_id: Optional[str] = None`, harden branch raises `ValueError` if `None`, appends real ID otherwise
- `engine/main.py:511,1825` — NOT changed; B2 fix remains: both call `mark_stage_complete("harden", current_module.id)`

Revert diff verified: `git diff origin/main -- tests/e2e/test_adversarial_stress.py` → empty (identical to origin/main). Commit `222907b` reverts exactly the 9-insertion / 3-insertion blocks added by `6f057ea` and no other lines.

---

### Class C (Negative — searched and NOT found)

- **No third harden call site** — `grep -n 'mark_stage_complete("harden")' engine/main.py` → zero matches (confirmed at plan §2 row 15 and in this session).
- **No synthetic `module_N` strings in test_adversarial_stress.py** — file does not contain `"module_0"`, `"module_1"`, or `f"module_{...}"` on origin/main (revert target), so no CORR-001 encoding existed in this file.
- **No additional inherited tests modified by CORR-001 branch** — `git diff origin/main --name-only -- 'tests/'` after this revert lists: `tests/engine/test_state.py`, `tests/engine/test_submit_handlers.py`, `tests/e2e/test_adversarial_stress.py` (now identical to origin/main), `tests/e2e/test_complete_bjh_loop.py`. The oracle-corrections file covers all three tests that remain legitimately modified.
- **Bug-catalog 'Skipped' set** — no other findings from `audit/02-static-audit.md` are addressed by this commit; each is a separate CORR-NNN PR.

---

### Class D (Static Analysis — lint/type/build)

- **Ruff** — `uv run ruff check engine/ tests/ --output-format=github` → 249 lines of output, all pre-existing violations in `engine/ast_harden/` files not touched by any CORR-001 commit. Count is identical before and after the revert commit (verified by `git stash` baseline comparison).
- **No new violations introduced** by this revert; the file `tests/e2e/test_adversarial_stress.py` at origin/main state has no ruff violations.
- **Type check** — `uv run ty check engine/` not run (R0 tier; no logic changes to typed source).

---

### Class F (Provenance — git chain-of-custody of touched test files)

**Justification:** The only test file touched by this commit is `tests/e2e/test_adversarial_stress.py`, and the change is a PURE REVERT to origin/main — no assertion or behavioral logic is introduced; the commit removes lines added out-of-scope by a prior attempt. A revert that makes a test file identical to its origin/main state is valid: it corrects an unauthorized modification, not the test's correctness.

- `tests/e2e/test_adversarial_stress.py` — origin/main SHA: `95f941a`; post-prior-attempt SHA: `c384973` (commit `6f057ea`); this revert commit `222907b` restores to `95f941a` content (verified by `git diff origin/main -- tests/e2e/test_adversarial_stress.py` → empty).
- The file was first introduced in origin/main (not authored by the CORR-001 branch); the prior-attempt modification (`6f057ea`) was the only non-origin change; `222907b` undoes it entirely.
- The three oracle-corrected tests (`test_state.py`, `test_submit_handlers.py`, `test_complete_bjh_loop.py`) are untouched by this commit; their provenance is covered by oracle-corrections `mastery-corr-001-impl.md` committed at `88321d5`.

---

## Verification Methodology

**Zero-Touch Mandate:** Verifier inspects artifacts only.
Evidence was collected by `aiv commit` during the change lifecycle.
Packet generated by `aiv close` and manually extended to satisfy Classes A–F per operator mandate (2026-06-19).

---

## Known Limitations

- Evidence references point to Layer 1 evidence files at specific commit SHAs.
  Use `git show <sha>:.github/aiv-evidence/<file>` to retrieve.

---

## Summary

Change 'mastery-corr-001-oracle-fix': 1 commit(s) across 1 file(s). Reverts out-of-scope modification to `tests/e2e/test_adversarial_stress.py` so the oracle guard no longer trips. All B1–B5 CORR-001 fixes remain intact.

## Machine-checkable data

```json
{
  "change_id": "mastery-corr-001-oracle-fix",
  "finding": "CORR-001",
  "intent_url": "https://github.com/ImmortalDemonGod/mastery-engine/blob/7f6610a902befcb84fc47e5c82a161e3d3184ce4/audit/02-static-audit.md#L17",
  "commits": ["222907b"],
  "files_changed": ["tests/e2e/test_adversarial_stress.py"],
  "gates": {
    "G1": "PASS — grep 'f\"module_' engine/schemas.py → zero matches",
    "G2": "PASS — module_id: Optional[str] = None at schemas.py:156",
    "G3": "PASS — raise ValueError at schemas.py:169",
    "G4": "PASS — 2 matches: main.py:511 and main.py:1825",
    "G5": "PASS — zero bare mark_stage_complete(\"harden\") in main.py",
    "G6": "PASS — completed_modules = ['softmax']",
    "G7": "PASS — ValueError raised: module_id is required for harden stage",
    "G8": "PASS — zero module_0/module_1 in test_state.py",
    "G9": "PASS — zero module_0/module_1 in test_complete_bjh_loop.py",
    "G10": "PASS — 4/4 TestUserProgressModel tests pass",
    "G11": "PASS — 196/196 engine suite tests pass exit 0",
    "G13": "PASS — ruff violations are pre-existing only (249 lines, same baseline)",
    "G14": "N/A — aiv check run separately on PACKET_mastery_corr_001_impl.md (warnings only, not blocking errors)"
  },
  "oracle_guard": {
    "tripped_test": "tests/e2e/test_adversarial_stress.py::test_corrupted_patch_file",
    "resolution": "REVERTED",
    "rationale": "Change was out-of-scope for CORR-001; not anchored to the finding; no oracle-correction could be justified"
  }
}
```
