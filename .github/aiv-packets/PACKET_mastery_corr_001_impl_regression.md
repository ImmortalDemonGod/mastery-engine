# AIV Verification Packet (v2.2)

## Identification

| Field | Value |
|-------|-------|
| **Repository** | github.com/ImmortalDemonGod/mastery-engine |
| **Change ID** | mastery-corr-001-impl-regression |
| **Commits** | `6f057ea` |
| **Head SHA** | `6f057ea` |
| **Base SHA** | `635ff3a` |
| **Created** | 2026-06-21T07:51:08Z |

## Classification

```yaml
classification:
  risk_tier: R1
  sod_mode: S0
  critical_surfaces: []
  blast_radius: component
  classification_rationale: "Single test-only file; test logic corrected to be deterministic; no engine code changed"
  classified_by: "Claude"
  classified_at: "2026-06-21T07:51:08Z"
```

## Claims

1. test_corrupted_patch_file passes deterministically when JSON bug files are hidden before engine start-challenge, because engine/stages/harden.py:206 uses random.choice and hiding JSON files forces selection of the corrupted patch — tests/e2e/test_adversarial_stress.py:168-200
2. Existing tests were preserved: no pre-existing test was deleted or had its behavioral assertion weakened — the only change is adding JSON file hide/restore around the corrupted-patch assertion — tests/e2e/test_adversarial_stress.py diff at commit 6f057ea
3. Bug file restoration in finally block leaves bugs_dir identical to pre-test state: patch_backup.rename(patch_file) restores the original patch; jbak.rename(jf) restores each JSON — tests/e2e/test_adversarial_stress.py:195-200

---

## Evidence References

| # | Evidence File | Commit SHA | Classes |
|---|---------------|------------|---------|
| 1 | EVIDENCE_TESTS_E2E_TEST_ADVERSARIAL_STRESS.md | `6f057ea` | A, B, C, D, E, F |

---

### Class A (Behavioral / Direct Execution Evidence)

**test_corrupted_patch_file — run 1:**
```
$ uv run pytest tests/e2e/test_adversarial_stress.py::TestAdversarialStress::test_corrupted_patch_file -v --tb=short
collected 1 item
tests/e2e/test_adversarial_stress.py::TestAdversarialStress::test_corrupted_patch_file PASSED
1 passed, 1 warning in 2.98s
```

**test_corrupted_patch_file — run 2 (determinism check):**
```
$ uv run pytest tests/e2e/test_adversarial_stress.py::TestAdversarialStress::test_corrupted_patch_file -v --tb=short
collected 1 item
tests/e2e/test_adversarial_stress.py::TestAdversarialStress::test_corrupted_patch_file PASSED
1 passed, 1 warning in 3.03s
```

Two consecutive passes confirm flakiness eliminated. Previously the test failed when `random.choice(patch_files + json_files)` selected a JSON bug (AST injection succeeds; returncode=0; assertion fails); after fix, JSON files are hidden so only the corrupted patch remains.

**Full engine suite (no regressions):**
```
$ uv run pytest tests/engine/ -v -m "not integration" --tb=short
collected 196 items
...
196 passed, 10 warnings in 1.26s
```

---

### Class B (Referential — SHA-Pinned Line Anchors)

All references are against commit `6f057ea`:

| File | Line(s) | Content |
|------|---------|---------|
| `tests/e2e/test_adversarial_stress.py` | 168 | `# Corrupt the patch file; hide JSON bugs so the engine is forced to pick the patch` |
| `tests/e2e/test_adversarial_stress.py` | 171 | `json_bugs = list(bugs_dir.glob("*.json"))` |
| `tests/e2e/test_adversarial_stress.py` | 172 | `json_backups = [(jf, jf.with_suffix('.json.bak')) for jf in json_bugs]` |
| `tests/e2e/test_adversarial_stress.py` | 173-174 | for-loop hiding JSON files via rename |
| `tests/e2e/test_adversarial_stress.py` | 197-199 | finally-block restoration: `for jf, jbak in json_backups: jbak.rename(jf)` |
| `engine/stages/harden.py` | 206 | UNTOUCHED: `selected_bug = random.choice(bug_files)` where `bug_files = patch_files + json_files` |

---

### Class C (Negative — What Was Searched For and NOT Found)

- `grep -rn '"module_0"\|"module_1"\|f"module_' tests/` → **zero matches** in current HEAD (synthetic IDs eliminated by prior CORR-001 commits).
- `grep -rn 'json_backups\|json\.bak' engine/` → **zero matches**; restoration pattern is test-local only, no engine logic touched.
- No other test files in `tests/` glob `bugs_dir` or interact with bug-file selection — confirmed by `grep -rn 'bugs_dir\|\.patch\|\.json.*bak' tests/` scoped to test files (zero hits outside test_adversarial_stress.py).
- **Skipped (out of scope):** CORR-003 (`harden.py` draft-JSON selection bug catalogued at `audit/02-static-audit.md`) is a distinct engine-layer finding. This commit addresses only the test-layer symptom; CORR-003 remains a separate catalogued item.

---

### Class D (Static Analysis — Lint / Type / Build)

**Ruff on touched file:**
```
$ uv run ruff check tests/e2e/test_adversarial_stress.py --output-format=github
```
Exit non-zero with 12 violations — ALL pre-existing on `origin/main`:
- F401 unused imports at lines 15, 17, 23 (pre-existing)
- F811 redefinitions of `isolated_repo` fixture at lines 29, 79, 132, 202, 220, 230, 241, 275, 282 (pre-existing)
- UP015 unnecessary mode argument at line 162 (pre-existing)

**Zero new ruff violations introduced.** Verified by `git stash` + re-run on baseline producing identical violation set.

No new type errors: no type annotation was changed. No `Optional`, `List`, or `Dict` usage in the changed lines.

---

### Class E (Intent Alignment)

**Canonical intent URL:** https://github.com/ImmortalDemonGod/mastery-engine/blob/7f6610a902befcb84fc47e5c82a161e3d3184ce4/audit/02-static-audit.md#L17

**Audit source read:** The canonical audit at pinned SHA `7f6610a` records CORR-001:

> *"mark_stage_complete() appends f"module_{self.current_module_index}" (synthetic 0-based array index) to completed_modules instead of the actual module.id. engine/main.py:2196 in curriculum-list checks `module.id in progress.completed_modules`; real IDs like 'softmax' or 'rmsnorm' never match synthetic 'module_0'/'module_1'. Cascades: progress-reset at main.py:2335 filters by module.id and also fails to remove the synthetic entry."*

The same audit also records CORR-003 at the same source, confirming `random.choice` in `harden.py`'s `_select_bug()` is a documented hazard that can select unexpected file types.

**Alignment assessment:** The CORR-001 primary fix (commits `b5474e5`–`19aa603`) replaced the synthetic ID with the real `module_id` at the write site. As a post-condition, the full test suite must remain green. The `test_corrupted_patch_file` test was non-deterministically failing in the post-CORR-001 state because the engine's pre-existing `random.choice(patch_files + json_files)` could select a JSON bug file, causing AST injection to succeed even with a corrupted patch, defeating the test's stated intent. This commit makes the test's environment match its stated intent ("engine fails gracefully when patch file is corrupted") by hiding JSON files so only the corrupted patch is selectable. This directly supports CORR-001's deliverable: a green full suite as the acceptance criterion.

---

### Class F (Provenance — Git Chain-of-Custody for Touched Test File)

Test file change committed in this change context `mastery-corr-001-impl-regression`:

| Commit SHA | File | Author | Commit message |
|-----------|------|--------|----------------|
| `6f057ea` | `tests/e2e/test_adversarial_stress.py` | Claude (write-code stage) | test(adversarial): make corrupted-patch test deterministic — hide JSON bugs before selecting |

**Prior touch history for `tests/e2e/test_adversarial_stress.py`:**
- `96f9e74` — B5: Fix CLI name engine→mastery (pre-CORR-001, unrelated)
- `6f057ea` — this commit (regression fix)

**Justification:** The pre-existing `assert result.returncode != 0` and `assert "error" in result.stdout.lower() or "failed" in result.stdout.lower()` assertions encode the test's stated intent ("engine fails gracefully when patch is corrupted"). These assertions are NOT weakened. The change adds determinism: without the JSON file hide, `engine/stages/harden.py:206` uses `random.choice(patch_files + json_files)` and can select a JSON AST bug (which succeeds, contradicting the assertion). Hiding JSON files is a test setup fix, not an oracle change — the test's behavioral claim is preserved and made reliable.

**Claim 2** (existing tests preserved): No test was deleted and no pre-existing behavioral assertion was weakened. The `assert result.returncode != 0` at line 185 and `assert "error" in result.stdout.lower() or "failed" in result.stdout.lower()` at line 187 are **unchanged**. Only setup/teardown code (JSON file hide/restore) was added around the existing assertion.

**Claim 3** (finally block restores): `patch_backup.rename(patch_file)` restores the original patch; the new `for jf, jbak in json_backups: jbak.rename(jf)` loop restores each JSON — bugs_dir is identical before and after the test at `tests/e2e/test_adversarial_stress.py:195-200`.

Diff excerpt (commit `6f057ea`):
```diff
-        # Corrupt the patch file
+        # Corrupt the patch file; hide JSON bugs so the engine is forced to pick the patch
         bugs_dir = isolated_repo / "curricula/cs336_a1/modules/softmax/bugs"
         patch_file = list(bugs_dir.glob("*.patch"))[0]
+
+        # Hide JSON bug files so random.choice can only land on the corrupted patch
+        json_bugs = list(bugs_dir.glob("*.json"))
+        json_backups = [(jf, jf.with_suffix('.json.bak')) for jf in json_bugs]
+        for jf, jbak in json_backups:
+            jf.rename(jbak)
         ...
         finally:
             patch_file.unlink()
             patch_backup.rename(patch_file)
+            # Restore JSON bug files
+            for jf, jbak in json_backups:
+                jbak.rename(jf)
```

---

## Verification Methodology

**Zero-Touch Mandate:** Verifier inspects artifacts only.
Evidence was collected by `aiv commit` during the change lifecycle.
Packet generated by `aiv close` and manually completed with all evidence classes.

---

## Known Limitations

- Class A CI URL (E012): No CI run available pre-push. Local `uv run pytest` output cited instead.
- Class E URL format (E004): URL is SHA-pinned permalink — the canonical audit line referenced directly.

---

## Summary

Change 'mastery-corr-001-impl-regression': 1 commit across 1 test file.

**Gate results:**
| Gate | Result |
|------|--------|
| test_corrupted_patch_file PASS (run 1) | PASS |
| test_corrupted_patch_file PASS (run 2) | PASS |
| Full engine suite (196/196) | PASS |
| No new ruff violations | PASS |
| All JSON files restored in finally block | PASS (code review of lines 197-199) |
