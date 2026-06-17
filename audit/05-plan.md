# 05 — Execution-Ready Plan

> 20 ordered change items. Converged after 1 round(s): every item maps to a concrete diff target with a runnable verification signal.

| # | ID | Links to | Location | Change | Verification signal | Depends on |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | C1-test-state-isolation-fixture | Stage-2 finding: test-deletes-real-user-state (HIGH; root-cause foundation shared with test-corrupts-real-user-state, e2e-tests-read-write-real-user-state, adversarial-test-mutates-real-state) | tests/conftest.py (new autouse fixture); reads against engine/state.py:32 (STATE_FILE = Path.home()/'.mastery_progress.json') | Add an autouse session/function fixture `isolate_user_state` in tests/conftest.py that (a) sets monkeypatch.setenv('HOME', str(tmp_path)) and (b) monkeypatches engine.state.STATE_FILE to tmp_path/'.mastery_progress.json' so NO test can read or write the real ~/.mastery_progress.json. Export the patched path via the fixture return value so e2e tests can reference it instead of Path.home(). This is prerequisite for safely re-enabling the destructive-skipped e2e suite. | Run `HOME=$(mktemp -d) pytest tests/e2e tests/engine -q` then `test ! -e ~/.mastery_progress.json \|\| echo TOUCHED-REAL-STATE`; the real home state file is never created/modified (no 'TOUCHED-REAL-STATE'). `pytest --fixtures \| grep isolate_user_state` lists the fixture as autouse. | — |
| 2 | C2-fix-error-handling-test-state-writes | Stage-2 finding: test-corrupts-real-user-state (HIGH) and test-deletes-real-user-state (HIGH) | tests/e2e/test_error_handling.py:124-128 (unlink) and :284-285 (malformed JSON write) | Replace `state_file = Path.home() / '.mastery_progress.json'` at both sites with the tmp path supplied by the C1 `isolate_user_state` fixture. The unconditional `state_file.unlink()` in test_next_without_init and the malformed-JSON write in test_state_file_corruption_handling must target the fixture tmp path, never Path.home(). Remove the destructive-skip marker so the file is collected. | `grep -n "Path.home()" tests/e2e/test_error_handling.py` returns nothing; `pytest tests/e2e/test_error_handling.py -q` passes and `~/.mastery_progress.json` is untouched (re-check via C1 verification command). | C1-test-state-isolation-fixture |
| 3 | C3-fix-bjh-and-adversarial-test-state-writes | Stage-2 finding: e2e-tests-read-write-real-user-state (MEDIUM) and adversarial-test-mutates-real-state (MEDIUM) | tests/e2e/test_complete_bjh_loop.py:224,350,436,451; tests/e2e/test_adversarial_stress.py:161-166; tests/e2e/test_full_softmax_loop.py | Repoint every `Path.home() / '.mastery_progress.json'` read/write in these e2e files to the C1 fixture tmp path. In test_adversarial_stress.test_corrupted_patch_file wrap the read-mutate-write of state in the fixture path so the real state file is not mutated even on early failure. Re-enable collection (remove destructive-skip). | `grep -rn "Path.home()" tests/e2e/` returns no state-file references; `pytest tests/e2e -q` passes with `~/.mastery_progress.json` unchanged. | C1-test-state-isolation-fixture |
| 4 | B1-correct-completed-module-id | Stage-2 finding: CORR-001 (HIGH) | engine/schemas.py:156-172 (mark_stage_complete) and call site engine/main.py:511 (LINEAR harden completion) | Change mark_stage_complete signature to `mark_stage_complete(self, stage: str, module_id: str \| None = None)`. In the harden branch (schemas.py:168) append the passed `module_id` to completed_modules instead of the synthetic `f"module_{self.current_module_index}"`; raise/log if module_id is None for harden. Update the LINEAR harden caller at main.py:511 to pass `current_module.id` (current_module = manifest.modules[progress.current_module_index]). Leave build/justify calls unchanged. | After completing the softmax module end-to-end, `python -c "import json,os;print(json.load(open(os.path.expanduser('~/.mastery_progress.json')))['completed_modules'])"` prints `['softmax']` (not `['module_0']`); `mastery curriculum-list` footer shows 'Progress: 1/22 modules completed' and the softmax row renders ✅. Update unit test tests/engine/test_state.py::test_mark_stage_complete_harden_advances_module to assert completed_modules == [the real module id] and it passes. | — |
| 5 | B2-exclude-draft-bug-specs-from-selection | Stage-2 finding: CORR-003 (HIGH) | engine/stages/harden.py:194-211 (_select_bug) | Filter draft files out of the glob in _select_bug: change `json_files = list(bugs_dir.glob('*.json'))` to exclude files whose stem ends in '_draft' (e.g. `json_files = [p for p in bugs_dir.glob('*.json') if not p.stem.endswith('_draft')]`). This prevents selecting the 16 *_draft.json specs under cs336_a1 bug dirs that have no matching *_draft_symptom.txt, which currently raises HardenChallengeError at harden.py:213. | Add a unit test placing a `*_draft.json` (no symptom file) plus a valid `*.json` (with symptom) in a tmp bugs_dir; assert _select_bug never returns the draft across many iterations and never raises 'Symptom file missing'. `pytest tests/engine/test_stages.py -k select_bug` passes. `find curricula/cs336_a1 -name '*_draft.json' \| wc -l` confirms drafts exist (>0) yet selection succeeds. | — |
| 6 | B3-use-absolute-shadow-worktree-dir | Stage-2 findings: CORR-005 (MEDIUM), CORR-004 (MEDIUM), harden-shadow-path-relative (MEDIUM) | engine/main.py:464 (_submit_harden_stage) and engine/stages/harden.py:78 (present_challenge); reference SHADOW_WORKTREE_DIR defined at engine/main.py:73-77 via engine/utils.find_project_root | Replace the relative `shadow_worktree = Path('.mastery_engine_worktree')` at main.py:464 with the module-level absolute `SHADOW_WORKTREE_DIR` (already computed from find_project_root at main.py:73-77). In engine/stages/harden.py:78 (and :247 present_library_challenge) import and use the same find_project_root-derived absolute path instead of the relative Path('.mastery_engine_worktree'). Ensures file copies and subprocess cwd= target the real worktree regardless of os.getcwd(). | `grep -rn "Path('.mastery_engine_worktree')" engine/` returns no hits in main.py:464 or harden.py. Add a test that invokes _submit_harden_stage from a cwd != project root (monkeypatch os.chdir to tmp) and asserts harden_workspace resolves under the project root, not cwd; test passes. | — |
| 7 | B5-library-harden-copy-fixed-file | Stage-2 finding: library-harden-missing-file-copy (HIGH) | engine/main.py:741-757 (_submit_library_workflow harden branch) | Before `result = validator_subsys.execute(validator_path, shadow_workspace)` at main.py:757, copy the learner's fixed file from the harden workspace into the shadow worktree, mirroring the LINEAR path at main.py:493-496 (shadow_dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(harden_file, shadow_dest)). Resolve harden_file/shadow_dest for the active LIBRARY problem (problem id based, analogous to main.py:486-491). Without this the validator runs against unmodified shadow-worktree code and the fix is never actually tested. | Add an integration test for a LIBRARY-mode problem: submit a correct fix, assert the validator executes against the copied file (exit_code 0 and 'Harden Validation Passed' panel) and that submitting a wrong fix fails. `pytest -k library_harden` passes; manual: a deliberately wrong LIBRARY fix no longer spuriously passes. | B3-use-absolute-shadow-worktree-dir |
| 8 | B4-library-justify-real-llm-grading | Stage-2 findings: library-justify-stub-auto-advance (HIGH), CORR-002 (HIGH) | engine/main.py:708-739 (_submit_library_workflow justify branch, the '# TODO: Implement proper editor integration' block at :718-727) | Replace the stub that prints 'Editor integration pending...' and unconditionally sets progress.current_stage='harden' with real evaluation: collect the learner's answers for the pattern's JustifyQuestions and call the same LLMService.evaluate_justification path used by LINEAR mode (engine/main.py ~line 390-401). Only mark pattern theory complete and advance to harden when evaluation returns is_correct; otherwise render Socratic feedback and remain in justify. Mirror the LINEAR justify handler rather than auto-passing. | Add a test with a mocked LLMService returning is_correct=False: assert progress.current_stage stays 'justify', pattern NOT in completed_patterns, and feedback rendered. With is_correct=True it advances to 'harden'. `pytest -k library_justify` passes. `grep -n "marking theory as complete" engine/main.py` returns nothing. | — |
| 9 | B6-handle-justify-only-module-type | Stage-2 finding: DCD-006 (HIGH) | engine/main.py:555-586 (_submit_linear_workflow stage routing); curricula/cs336_a1/manifest.json ('unicode' module module_type='justify_only') | In _submit_linear_workflow, branch on current_module.module_type before stage routing: for module_type=='justify_only' modules, skip the build and harden stages (do not attempt validator/harden bug selection, which do not exist for these modules) and complete the module after a passing justify. Currently only progress.current_stage is checked, so a justify_only module like 'unicode' would attempt a nonexistent build/harden and stall. | Add a test driving the 'unicode' (justify_only) module: after a passing justify, mark_stage_complete advances module_index past it WITHOUT entering build/harden, and no HardenChallengeError/validator lookup occurs. `pytest -k justify_only` passes; `mastery status` on a justify_only module never shows a harden start-challenge prompt. | B1-correct-completed-module-id |
| 10 | B7-fix-cosine-validator-wrong-file | Stage-2 finding: cosine-validator-wrong-file (CRITICAL) | curricula/cs336_a1/modules/cosine_schedule/validator.sh:18 | The validator copies cs336_basics/optimizer.py but the test suite imports get_lr_cosine_schedule from cs336_basics.utils (tests/adapters.py:11-15) where the reference impl lives (modes/developer/cs336_basics/utils.py:75). Change line 18 to copy the file that actually contains get_lr_cosine_schedule (cs336_basics/utils.py) into $SHADOW_WORKTREE/cs336_basics/utils.py — or copy both optimizer.py and utils.py. The learner's cosine-schedule edits are otherwise never seen by the validator. | Run the cosine_schedule validator against the developer reference solution (developer mode): `bash curricula/cs336_a1/modules/cosine_schedule/validator.sh` exits 0; then introduce a wrong get_lr_cosine_schedule and confirm the validator exits non-zero (it now reads the learner's file). | — |
| 11 | B8-fix-cosine-build-prompt-drift | Stage-2 finding: cosine-schedule-wrong-function-name-and-file (HIGH) | curricula/cs336_a1/modules/cosine_schedule/build_prompt.txt:135,148,326 | Update build_prompt.txt to match the actual contract corrected in B7: 'FILE TO MODIFY' should name cs336_basics/utils.py (not optimizer.py), the function signature should be `def get_lr_cosine_schedule(...)` (not lr_cosine_schedule), and the referenced pytest target at line 326 should match the real test path/nodeid. Eliminates the doc/code drift that misdirects the learner. | `grep -n 'lr_cosine_schedule\\|optimizer.py' curricula/cs336_a1/modules/cosine_schedule/build_prompt.txt` shows only get_lr_cosine_schedule and utils.py references; the function name and file named in the prompt match tests/adapters.py:11-15 and the B7 validator copy target. | B7-fix-cosine-validator-wrong-file |
| 12 | B9-fix-nucleus-sampling-mask | Stage-2 finding: text-generation-nucleus-sampling-mask-inverted (HIGH) | curricula/cs336_a1/modules/text_generation/build_prompt.txt:310-323 (mask = cumsum_probs >= p) | Correct the inverted nucleus (top-p) mask in the build-prompt reference code so it matches the algorithm described at lines 138-144 (keep the lowest-index tokens whose cumulative probability first reaches p). Replace the keep-condition so the token that crosses the threshold is retained and only strictly-later tokens are masked out (standard nucleus sampling: shift the cumsum>=p mask by one so the boundary token is kept). The current `mask = cumsum_probs >= p` masks the wrong side and contradicts the stated spec. | A unit/golden check: for probs and cumsum example [A,B,C] (cumsum=0.95) at p=0.95, the corrected mask retains {A,B,C} as the spec states; the documented worked example and the code agree. Reviewer diff shows the boundary-token off-by-one is fixed. | — |
| 13 | B10-add-missing-re-import | Stage-2 finding: enrich-problems-missing-re-import (HIGH) | scripts/enrich_problems.py (top-of-file imports; usages at :159,162,168,173,178,193,199-201) | Add `import re` to the import block of scripts/enrich_problems.py. _extract_examples() calls re.findall/re.search and references re.DOTALL but `re` is never imported, so any call raises NameError: name 're' is not defined. | `python -c "import ast,sys; ast.parse(open('scripts/enrich_problems.py').read())"` ok and `grep -n '^import re' scripts/enrich_problems.py` matches; invoking _extract_examples() on a sample problem string returns parsed examples without NameError. | — |
| 14 | S1-llm-fail-closed-and-resolve-test-contradiction | Stage-2 findings: mock-llm-auth-bypass (security) and missing-api-key-behavior-contradiction (MEDIUM) | engine/services/llm_service.py:55-71 (mock-mode init) and the two contradicting tests tests/engine/test_llm_service.py:55-61 vs tests/integration/test_llm_service.py:70-83 | Make Justify grading fail-closed by default: when OPENAI_API_KEY is absent, do NOT silently enable auto-pass mock mode; instead raise ConfigurationError (as tests/integration/test_llm_service.py:70 expects) UNLESS an explicit opt-in flag (e.g. env MASTERY_LLM_MOCK=1) is set for portfolio/demo viewing. Update mock evaluate_justification (llm_service.py:108-122) to only run under that explicit flag. Resolve the contradiction by aligning the unit test (tests/engine/test_llm_service.py) to assert ConfigurationError when the flag is unset and mock-mode only when MASTERY_LLM_MOCK=1. | `OPENAI_API_KEY= pytest tests/engine/test_llm_service.py tests/integration/test_llm_service.py -q` passes with both files now consistent; running `mastery submit` on a justify stage with no key and no MASTERY_LLM_MOCK raises a clear ConfigurationError instead of auto-passing (no '🎭 MOCK MODE' auto-pass on the gating path). | C1-test-state-isolation-fixture |
| 15 | S2-sandbox-student-code-execution | Stage-2 finding: unsandboxed-student-code-execution (MEDIUM); Stage-4 goal-gap: 'process isolation hardening (nsjail/rlimit runtime sandbox) for arbitrary learner code' | engine/validator.py:108-116 (subprocess.run of validator_path) | Wrap validator execution in resource/runtime limits so arbitrary student code cannot run with full user privileges. Minimum: apply POSIX rlimits via subprocess preexec_fn (resource.setrlimit RLIMIT_CPU, RLIMIT_AS, RLIMIT_FSIZE, RLIMIT_NOFILE) in addition to the existing timeout=300. Preferred (per Stage-4 research): execute under nsjail (snekbox pattern) with seccomp-bpf + cgroups when available, falling back to rlimits. Keep cwd/env handling intact. | Add a test whose validator spawns a fork bomb / large allocation and assert it is killed by the rlimit (non-zero/terminated) rather than exhausting the host; `pytest -k sandbox` passes. `grep -n 'setrlimit\\|nsjail' engine/validator.py` shows the limiter is wired into the subprocess path. | — |
| 16 | D1-resolve-dead-ast-injectors | Stage-4 goal-gap: 'portfolio claim integrity — README 71-79 Runtime AST Mutation showcase relies on engine/ast_harden/softmax_v2_1.py and softmax_poc.py which Stage-3 measured at 0% coverage (dead)' | engine/ast_harden/softmax_poc.py (0%, lines 10-252), engine/ast_harden/softmax_v2_1.py (0%, lines 14-357); README.md:71-79; engine/services/ast_service.py | Either (a) wire the live harden injection path (services/ast_service.py + generic_injector.py) to use softmax_v2_1.inject_softmax_bug_v2_1 so the README-claimed AST mutation is actually exercised, OR (b) delete softmax_poc.py and the unused softmax_v2_1.py and update README's 'Runtime AST Mutation' claim to reference the injector that is actually invoked. Do not leave 0%-covered code presented as the central portfolio achievement. Decide based on which injector the harden runtime calls (confirm via ast_service import graph). | After the change, the injector referenced by README.md is reachable from the harden submit path: a coverage run `pytest --cov=engine/ast_harden tests/` shows the showcased injector >0% (if wired) OR the dead files no longer exist (`test ! -e engine/ast_harden/softmax_poc.py`) and README no longer cites them. No file presented as a key feature sits at 0% coverage. | B2-exclude-draft-bug-specs-from-selection |
| 17 | D2-fix-pretokenization-example | Stage-2 finding: pretokenization-example-broken (LOW, doc_code_drift) | modes/developer/cs336_basics/pretokenization_example.py:53 | Line 53 passes Python's Ellipsis (...) as the first arg to open(), raising TypeError at import/compile if pytest discovery reaches modes/. Either replace `...` with a real/parameterized path argument (e.g. a sys.argv or a named example file), or guard the example under `if __name__ == '__main__':`, or move it out of an importable .py into a docstring/markdown so test collection cannot import-fail it. | `python -c "import ast; ast.parse(open('modes/developer/cs336_basics/pretokenization_example.py').read())"` ok and importing the module (or running it with no args) no longer raises 'TypeError: expected str, bytes or os.PathLike object, not ellipsis'; `pytest --collect-only` does not error on this file. | — |
| 18 | D3-restore-or-document-bpe-merge-assertion | Stage-2 finding: bpe-merge-correctness-assertion-commented-out (MEDIUM, intent_mismatch) | tests/test_train_bpe.py:51-54 (commented `# assert merges == reference_merges`) | The exact-merge-order correctness assertion is commented out with a note that tie-breaking differs from the reference. Replace the silent comment-out with a deterministic, defensible check: either (a) assert equality against a reference produced with the SAME documented tie-break ((count, earliest_index, lexicographic)), or (b) assert a weaker but real invariant (set/multiset of merges, or merge-count) and document why exact order is not asserted. Do not leave the correctness intent silently disabled. | tests/test_train_bpe.py:54 no longer contains a commented-out assert; the file contains an active assertion (exact or documented-weaker) and `pytest tests/test_train_bpe.py -q` passes (or is explicitly hardware-gated with a skip reason, not a silent no-op). | — |
| 19 | G1-harden-bug-quality-pipeline | Stage-4 goal-gap: 'Harden-stage semantic-bug pedagogical quality — misconception-targeted AST mutations + equivalent-mutant pre-screen (Goal 1 BJH depth, Goal 2 architecture)' | engine/ast_harden/generic_injector.py and engine/stages/harden.py (bug-generation/selection pipeline) | Add an equivalent-mutant pre-screen to the harden bug pipeline: after injecting a mutation, run the module validator against the mutated code and only keep the bug if the test suite FAILS (proving the bug is detectable), discarding semantically-equivalent mutants. Bias the operator catalog toward documented student misconceptions (comparison-operator inversion, off-by-one constant shift, wrong reduction axis, sign flip, wrong-variable-in-scope) per Stage-4 research, rather than arbitrary node swaps. This makes injected bugs reliably detectable and diagnostic. | Add a test asserting every bug produced by the pipeline causes at least one validator failure (no equivalent mutants survive) and that produced mutations belong to the misconception operator set; `pytest -k mutant_prescreen` passes. Manual: running start-challenge repeatedly never yields a bug whose 'fix' is identical to the original. | B2-exclude-draft-bug-specs-from-selection |
| 20 | G2-justify-llm-cost-optimization | Stage-4 goal-gap: 'LLM-as-evaluator cost optimization for Justify stage (Goal 2 portfolio architecture claim)' | engine/services/llm_service.py (evaluate_justification prompt construction + model routing) | Implement the cost-optimization the portfolio claims: (1) move the long, reused system prompt + rubric into a cacheable prefix (provider prompt caching) so per-evaluation token cost drops on repeated rubric reuse, and (2) add a model cascade — route clear pass/fail answers through a cheap small model and escalate only borderline scores to the capable model. Keep G-Eval-style chain-of-thought per-criterion scoring. Make the model ids configurable. | Add a test with a mocked client asserting (a) the rubric/system prefix is sent in a cache-eligible position and (b) borderline answers trigger the escalation call while clear ones do not. Measured: a repeated-evaluation benchmark shows reduced token/$ per Justify call versus the single-model baseline; `pytest -k llm_cost` passes. | S1-llm-fail-closed-and-resolve-test-contradiction |

---
### machine-readable artifact
```json
{
  "items": [
    {
      "id": "C1-test-state-isolation-fixture",
      "linkTo": "Stage-2 finding: test-deletes-real-user-state (HIGH; root-cause foundation shared with test-corrupts-real-user-state, e2e-tests-read-write-real-user-state, adversarial-test-mutates-real-state)",
      "location": "tests/conftest.py (new autouse fixture); reads against engine/state.py:32 (STATE_FILE = Path.home()/'.mastery_progress.json')",
      "change": "Add an autouse session/function fixture `isolate_user_state` in tests/conftest.py that (a) sets monkeypatch.setenv('HOME', str(tmp_path)) and (b) monkeypatches engine.state.STATE_FILE to tmp_path/'.mastery_progress.json' so NO test can read or write the real ~/.mastery_progress.json. Export the patched path via the fixture return value so e2e tests can reference it instead of Path.home(). This is prerequisite for safely re-enabling the destructive-skipped e2e suite.",
      "verificationSignal": "Run `HOME=$(mktemp -d) pytest tests/e2e tests/engine -q` then `test ! -e ~/.mastery_progress.json || echo TOUCHED-REAL-STATE`; the real home state file is never created/modified (no 'TOUCHED-REAL-STATE'). `pytest --fixtures | grep isolate_user_state` lists the fixture as autouse.",
      "dependsOn": [],
      "order": 1
    },
    {
      "id": "C2-fix-error-handling-test-state-writes",
      "linkTo": "Stage-2 finding: test-corrupts-real-user-state (HIGH) and test-deletes-real-user-state (HIGH)",
      "location": "tests/e2e/test_error_handling.py:124-128 (unlink) and :284-285 (malformed JSON write)",
      "change": "Replace `state_file = Path.home() / '.mastery_progress.json'` at both sites with the tmp path supplied by the C1 `isolate_user_state` fixture. The unconditional `state_file.unlink()` in test_next_without_init and the malformed-JSON write in test_state_file_corruption_handling must target the fixture tmp path, never Path.home(). Remove the destructive-skip marker so the file is collected.",
      "verificationSignal": "`grep -n \"Path.home()\" tests/e2e/test_error_handling.py` returns nothing; `pytest tests/e2e/test_error_handling.py -q` passes and `~/.mastery_progress.json` is untouched (re-check via C1 verification command).",
      "dependsOn": [
        "C1-test-state-isolation-fixture"
      ],
      "order": 2
    },
    {
      "id": "C3-fix-bjh-and-adversarial-test-state-writes",
      "linkTo": "Stage-2 finding: e2e-tests-read-write-real-user-state (MEDIUM) and adversarial-test-mutates-real-state (MEDIUM)",
      "location": "tests/e2e/test_complete_bjh_loop.py:224,350,436,451; tests/e2e/test_adversarial_stress.py:161-166; tests/e2e/test_full_softmax_loop.py",
      "change": "Repoint every `Path.home() / '.mastery_progress.json'` read/write in these e2e files to the C1 fixture tmp path. In test_adversarial_stress.test_corrupted_patch_file wrap the read-mutate-write of state in the fixture path so the real state file is not mutated even on early failure. Re-enable collection (remove destructive-skip).",
      "verificationSignal": "`grep -rn \"Path.home()\" tests/e2e/` returns no state-file references; `pytest tests/e2e -q` passes with `~/.mastery_progress.json` unchanged.",
      "dependsOn": [
        "C1-test-state-isolation-fixture"
      ],
      "order": 3
    },
    {
      "id": "B1-correct-completed-module-id",
      "linkTo": "Stage-2 finding: CORR-001 (HIGH)",
      "location": "engine/schemas.py:156-172 (mark_stage_complete) and call site engine/main.py:511 (LINEAR harden completion)",
      "change": "Change mark_stage_complete signature to `mark_stage_complete(self, stage: str, module_id: str | None = None)`. In the harden branch (schemas.py:168) append the passed `module_id` to completed_modules instead of the synthetic `f\"module_{self.current_module_index}\"`; raise/log if module_id is None for harden. Update the LINEAR harden caller at main.py:511 to pass `current_module.id` (current_module = manifest.modules[progress.current_module_index]). Leave build/justify calls unchanged.",
      "verificationSignal": "After completing the softmax module end-to-end, `python -c \"import json,os;print(json.load(open(os.path.expanduser('~/.mastery_progress.json')))['completed_modules'])\"` prints `['softmax']` (not `['module_0']`); `mastery curriculum-list` footer shows 'Progress: 1/22 modules completed' and the softmax row renders ✅. Update unit test tests/engine/test_state.py::test_mark_stage_complete_harden_advances_module to assert completed_modules == [the real module id] and it passes.",
      "dependsOn": [],
      "order": 4
    },
    {
      "id": "B2-exclude-draft-bug-specs-from-selection",
      "linkTo": "Stage-2 finding: CORR-003 (HIGH)",
      "location": "engine/stages/harden.py:194-211 (_select_bug)",
      "change": "Filter draft files out of the glob in _select_bug: change `json_files = list(bugs_dir.glob('*.json'))` to exclude files whose stem ends in '_draft' (e.g. `json_files = [p for p in bugs_dir.glob('*.json') if not p.stem.endswith('_draft')]`). This prevents selecting the 16 *_draft.json specs under cs336_a1 bug dirs that have no matching *_draft_symptom.txt, which currently raises HardenChallengeError at harden.py:213.",
      "verificationSignal": "Add a unit test placing a `*_draft.json` (no symptom file) plus a valid `*.json` (with symptom) in a tmp bugs_dir; assert _select_bug never returns the draft across many iterations and never raises 'Symptom file missing'. `pytest tests/engine/test_stages.py -k select_bug` passes. `find curricula/cs336_a1 -name '*_draft.json' | wc -l` confirms drafts exist (>0) yet selection succeeds.",
      "dependsOn": [],
      "order": 5
    },
    {
      "id": "B3-use-absolute-shadow-worktree-dir",
      "linkTo": "Stage-2 findings: CORR-005 (MEDIUM), CORR-004 (MEDIUM), harden-shadow-path-relative (MEDIUM)",
      "location": "engine/main.py:464 (_submit_harden_stage) and engine/stages/harden.py:78 (present_challenge); reference SHADOW_WORKTREE_DIR defined at engine/main.py:73-77 via engine/utils.find_project_root",
      "change": "Replace the relative `shadow_worktree = Path('.mastery_engine_worktree')` at main.py:464 with the module-level absolute `SHADOW_WORKTREE_DIR` (already computed from find_project_root at main.py:73-77). In engine/stages/harden.py:78 (and :247 present_library_challenge) import and use the same find_project_root-derived absolute path instead of the relative Path('.mastery_engine_worktree'). Ensures file copies and subprocess cwd= target the real worktree regardless of os.getcwd().",
      "verificationSignal": "`grep -rn \"Path('.mastery_engine_worktree')\" engine/` returns no hits in main.py:464 or harden.py. Add a test that invokes _submit_harden_stage from a cwd != project root (monkeypatch os.chdir to tmp) and asserts harden_workspace resolves under the project root, not cwd; test passes.",
      "dependsOn": [],
      "order": 6
    },
    {
      "id": "B5-library-harden-copy-fixed-file",
      "linkTo": "Stage-2 finding: library-harden-missing-file-copy (HIGH)",
      "location": "engine/main.py:741-757 (_submit_library_workflow harden branch)",
      "change": "Before `result = validator_subsys.execute(validator_path, shadow_workspace)` at main.py:757, copy the learner's fixed file from the harden workspace into the shadow worktree, mirroring the LINEAR path at main.py:493-496 (shadow_dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(harden_file, shadow_dest)). Resolve harden_file/shadow_dest for the active LIBRARY problem (problem id based, analogous to main.py:486-491). Without this the validator runs against unmodified shadow-worktree code and the fix is never actually tested.",
      "verificationSignal": "Add an integration test for a LIBRARY-mode problem: submit a correct fix, assert the validator executes against the copied file (exit_code 0 and 'Harden Validation Passed' panel) and that submitting a wrong fix fails. `pytest -k library_harden` passes; manual: a deliberately wrong LIBRARY fix no longer spuriously passes.",
      "dependsOn": [
        "B3-use-absolute-shadow-worktree-dir"
      ],
      "order": 7
    },
    {
      "id": "B4-library-justify-real-llm-grading",
      "linkTo": "Stage-2 findings: library-justify-stub-auto-advance (HIGH), CORR-002 (HIGH)",
      "location": "engine/main.py:708-739 (_submit_library_workflow justify branch, the '# TODO: Implement proper editor integration' block at :718-727)",
      "change": "Replace the stub that prints 'Editor integration pending...' and unconditionally sets progress.current_stage='harden' with real evaluation: collect the learner's answers for the pattern's JustifyQuestions and call the same LLMService.evaluate_justification path used by LINEAR mode (engine/main.py ~line 390-401). Only mark pattern theory complete and advance to harden when evaluation returns is_correct; otherwise render Socratic feedback and remain in justify. Mirror the LINEAR justify handler rather than auto-passing.",
      "verificationSignal": "Add a test with a mocked LLMService returning is_correct=False: assert progress.current_stage stays 'justify', pattern NOT in completed_patterns, and feedback rendered. With is_correct=True it advances to 'harden'. `pytest -k library_justify` passes. `grep -n \"marking theory as complete\" engine/main.py` returns nothing.",
      "dependsOn": [],
      "order": 8
    },
    {
      "id": "B6-handle-justify-only-module-type",
      "linkTo": "Stage-2 finding: DCD-006 (HIGH)",
      "location": "engine/main.py:555-586 (_submit_linear_workflow stage routing); curricula/cs336_a1/manifest.json ('unicode' module module_type='justify_only')",
      "change": "In _submit_linear_workflow, branch on current_module.module_type before stage routing: for module_type=='justify_only' modules, skip the build and harden stages (do not attempt validator/harden bug selection, which do not exist for these modules) and complete the module after a passing justify. Currently only progress.current_stage is checked, so a justify_only module like 'unicode' would attempt a nonexistent build/harden and stall.",
      "verificationSignal": "Add a test driving the 'unicode' (justify_only) module: after a passing justify, mark_stage_complete advances module_index past it WITHOUT entering build/harden, and no HardenChallengeError/validator lookup occurs. `pytest -k justify_only` passes; `mastery status` on a justify_only module never shows a harden start-challenge prompt.",
      "dependsOn": [
        "B1-correct-completed-module-id"
      ],
      "order": 9
    },
    {
      "id": "B7-fix-cosine-validator-wrong-file",
      "linkTo": "Stage-2 finding: cosine-validator-wrong-file (CRITICAL)",
      "location": "curricula/cs336_a1/modules/cosine_schedule/validator.sh:18",
      "change": "The validator copies cs336_basics/optimizer.py but the test suite imports get_lr_cosine_schedule from cs336_basics.utils (tests/adapters.py:11-15) where the reference impl lives (modes/developer/cs336_basics/utils.py:75). Change line 18 to copy the file that actually contains get_lr_cosine_schedule (cs336_basics/utils.py) into $SHADOW_WORKTREE/cs336_basics/utils.py — or copy both optimizer.py and utils.py. The learner's cosine-schedule edits are otherwise never seen by the validator.",
      "verificationSignal": "Run the cosine_schedule validator against the developer reference solution (developer mode): `bash curricula/cs336_a1/modules/cosine_schedule/validator.sh` exits 0; then introduce a wrong get_lr_cosine_schedule and confirm the validator exits non-zero (it now reads the learner's file).",
      "dependsOn": [],
      "order": 10
    },
    {
      "id": "B8-fix-cosine-build-prompt-drift",
      "linkTo": "Stage-2 finding: cosine-schedule-wrong-function-name-and-file (HIGH)",
      "location": "curricula/cs336_a1/modules/cosine_schedule/build_prompt.txt:135,148,326",
      "change": "Update build_prompt.txt to match the actual contract corrected in B7: 'FILE TO MODIFY' should name cs336_basics/utils.py (not optimizer.py), the function signature should be `def get_lr_cosine_schedule(...)` (not lr_cosine_schedule), and the referenced pytest target at line 326 should match the real test path/nodeid. Eliminates the doc/code drift that misdirects the learner.",
      "verificationSignal": "`grep -n 'lr_cosine_schedule\\|optimizer.py' curricula/cs336_a1/modules/cosine_schedule/build_prompt.txt` shows only get_lr_cosine_schedule and utils.py references; the function name and file named in the prompt match tests/adapters.py:11-15 and the B7 validator copy target.",
      "dependsOn": [
        "B7-fix-cosine-validator-wrong-file"
      ],
      "order": 11
    },
    {
      "id": "B9-fix-nucleus-sampling-mask",
      "linkTo": "Stage-2 finding: text-generation-nucleus-sampling-mask-inverted (HIGH)",
      "location": "curricula/cs336_a1/modules/text_generation/build_prompt.txt:310-323 (mask = cumsum_probs >= p)",
      "change": "Correct the inverted nucleus (top-p) mask in the build-prompt reference code so it matches the algorithm described at lines 138-144 (keep the lowest-index tokens whose cumulative probability first reaches p). Replace the keep-condition so the token that crosses the threshold is retained and only strictly-later tokens are masked out (standard nucleus sampling: shift the cumsum>=p mask by one so the boundary token is kept). The current `mask = cumsum_probs >= p` masks the wrong side and contradicts the stated spec.",
      "verificationSignal": "A unit/golden check: for probs and cumsum example [A,B,C] (cumsum=0.95) at p=0.95, the corrected mask retains {A,B,C} as the spec states; the documented worked example and the code agree. Reviewer diff shows the boundary-token off-by-one is fixed.",
      "dependsOn": [],
      "order": 12
    },
    {
      "id": "B10-add-missing-re-import",
      "linkTo": "Stage-2 finding: enrich-problems-missing-re-import (HIGH)",
      "location": "scripts/enrich_problems.py (top-of-file imports; usages at :159,162,168,173,178,193,199-201)",
      "change": "Add `import re` to the import block of scripts/enrich_problems.py. _extract_examples() calls re.findall/re.search and references re.DOTALL but `re` is never imported, so any call raises NameError: name 're' is not defined.",
      "verificationSignal": "`python -c \"import ast,sys; ast.parse(open('scripts/enrich_problems.py').read())\"` ok and `grep -n '^import re' scripts/enrich_problems.py` matches; invoking _extract_examples() on a sample problem string returns parsed examples without NameError.",
      "dependsOn": [],
      "order": 13
    },
    {
      "id": "S1-llm-fail-closed-and-resolve-test-contradiction",
      "linkTo": "Stage-2 findings: mock-llm-auth-bypass (security) and missing-api-key-behavior-contradiction (MEDIUM)",
      "location": "engine/services/llm_service.py:55-71 (mock-mode init) and the two contradicting tests tests/engine/test_llm_service.py:55-61 vs tests/integration/test_llm_service.py:70-83",
      "change": "Make Justify grading fail-closed by default: when OPENAI_API_KEY is absent, do NOT silently enable auto-pass mock mode; instead raise ConfigurationError (as tests/integration/test_llm_service.py:70 expects) UNLESS an explicit opt-in flag (e.g. env MASTERY_LLM_MOCK=1) is set for portfolio/demo viewing. Update mock evaluate_justification (llm_service.py:108-122) to only run under that explicit flag. Resolve the contradiction by aligning the unit test (tests/engine/test_llm_service.py) to assert ConfigurationError when the flag is unset and mock-mode only when MASTERY_LLM_MOCK=1.",
      "verificationSignal": "`OPENAI_API_KEY= pytest tests/engine/test_llm_service.py tests/integration/test_llm_service.py -q` passes with both files now consistent; running `mastery submit` on a justify stage with no key and no MASTERY_LLM_MOCK raises a clear ConfigurationError instead of auto-passing (no '🎭 MOCK MODE' auto-pass on the gating path).",
      "dependsOn": [
        "C1-test-state-isolation-fixture"
      ],
      "order": 14
    },
    {
      "id": "S2-sandbox-student-code-execution",
      "linkTo": "Stage-2 finding: unsandboxed-student-code-execution (MEDIUM); Stage-4 goal-gap: 'process isolation hardening (nsjail/rlimit runtime sandbox) for arbitrary learner code'",
      "location": "engine/validator.py:108-116 (subprocess.run of validator_path)",
      "change": "Wrap validator execution in resource/runtime limits so arbitrary student code cannot run with full user privileges. Minimum: apply POSIX rlimits via subprocess preexec_fn (resource.setrlimit RLIMIT_CPU, RLIMIT_AS, RLIMIT_FSIZE, RLIMIT_NOFILE) in addition to the existing timeout=300. Preferred (per Stage-4 research): execute under nsjail (snekbox pattern) with seccomp-bpf + cgroups when available, falling back to rlimits. Keep cwd/env handling intact.",
      "verificationSignal": "Add a test whose validator spawns a fork bomb / large allocation and assert it is killed by the rlimit (non-zero/terminated) rather than exhausting the host; `pytest -k sandbox` passes. `grep -n 'setrlimit\\|nsjail' engine/validator.py` shows the limiter is wired into the subprocess path.",
      "dependsOn": [],
      "order": 15
    },
    {
      "id": "D1-resolve-dead-ast-injectors",
      "linkTo": "Stage-4 goal-gap: 'portfolio claim integrity — README 71-79 Runtime AST Mutation showcase relies on engine/ast_harden/softmax_v2_1.py and softmax_poc.py which Stage-3 measured at 0% coverage (dead)'",
      "location": "engine/ast_harden/softmax_poc.py (0%, lines 10-252), engine/ast_harden/softmax_v2_1.py (0%, lines 14-357); README.md:71-79; engine/services/ast_service.py",
      "change": "Either (a) wire the live harden injection path (services/ast_service.py + generic_injector.py) to use softmax_v2_1.inject_softmax_bug_v2_1 so the README-claimed AST mutation is actually exercised, OR (b) delete softmax_poc.py and the unused softmax_v2_1.py and update README's 'Runtime AST Mutation' claim to reference the injector that is actually invoked. Do not leave 0%-covered code presented as the central portfolio achievement. Decide based on which injector the harden runtime calls (confirm via ast_service import graph).",
      "verificationSignal": "After the change, the injector referenced by README.md is reachable from the harden submit path: a coverage run `pytest --cov=engine/ast_harden tests/` shows the showcased injector >0% (if wired) OR the dead files no longer exist (`test ! -e engine/ast_harden/softmax_poc.py`) and README no longer cites them. No file presented as a key feature sits at 0% coverage.",
      "dependsOn": [
        "B2-exclude-draft-bug-specs-from-selection"
      ],
      "order": 16
    },
    {
      "id": "D2-fix-pretokenization-example",
      "linkTo": "Stage-2 finding: pretokenization-example-broken (LOW, doc_code_drift)",
      "location": "modes/developer/cs336_basics/pretokenization_example.py:53",
      "change": "Line 53 passes Python's Ellipsis (...) as the first arg to open(), raising TypeError at import/compile if pytest discovery reaches modes/. Either replace `...` with a real/parameterized path argument (e.g. a sys.argv or a named example file), or guard the example under `if __name__ == '__main__':`, or move it out of an importable .py into a docstring/markdown so test collection cannot import-fail it.",
      "verificationSignal": "`python -c \"import ast; ast.parse(open('modes/developer/cs336_basics/pretokenization_example.py').read())\"` ok and importing the module (or running it with no args) no longer raises 'TypeError: expected str, bytes or os.PathLike object, not ellipsis'; `pytest --collect-only` does not error on this file.",
      "dependsOn": [],
      "order": 17
    },
    {
      "id": "D3-restore-or-document-bpe-merge-assertion",
      "linkTo": "Stage-2 finding: bpe-merge-correctness-assertion-commented-out (MEDIUM, intent_mismatch)",
      "location": "tests/test_train_bpe.py:51-54 (commented `# assert merges == reference_merges`)",
      "change": "The exact-merge-order correctness assertion is commented out with a note that tie-breaking differs from the reference. Replace the silent comment-out with a deterministic, defensible check: either (a) assert equality against a reference produced with the SAME documented tie-break ((count, earliest_index, lexicographic)), or (b) assert a weaker but real invariant (set/multiset of merges, or merge-count) and document why exact order is not asserted. Do not leave the correctness intent silently disabled.",
      "verificationSignal": "tests/test_train_bpe.py:54 no longer contains a commented-out assert; the file contains an active assertion (exact or documented-weaker) and `pytest tests/test_train_bpe.py -q` passes (or is explicitly hardware-gated with a skip reason, not a silent no-op).",
      "dependsOn": [],
      "order": 18
    },
    {
      "id": "G1-harden-bug-quality-pipeline",
      "linkTo": "Stage-4 goal-gap: 'Harden-stage semantic-bug pedagogical quality — misconception-targeted AST mutations + equivalent-mutant pre-screen (Goal 1 BJH depth, Goal 2 architecture)'",
      "location": "engine/ast_harden/generic_injector.py and engine/stages/harden.py (bug-generation/selection pipeline)",
      "change": "Add an equivalent-mutant pre-screen to the harden bug pipeline: after injecting a mutation, run the module validator against the mutated code and only keep the bug if the test suite FAILS (proving the bug is detectable), discarding semantically-equivalent mutants. Bias the operator catalog toward documented student misconceptions (comparison-operator inversion, off-by-one constant shift, wrong reduction axis, sign flip, wrong-variable-in-scope) per Stage-4 research, rather than arbitrary node swaps. This makes injected bugs reliably detectable and diagnostic.",
      "verificationSignal": "Add a test asserting every bug produced by the pipeline causes at least one validator failure (no equivalent mutants survive) and that produced mutations belong to the misconception operator set; `pytest -k mutant_prescreen` passes. Manual: running start-challenge repeatedly never yields a bug whose 'fix' is identical to the original.",
      "dependsOn": [
        "B2-exclude-draft-bug-specs-from-selection"
      ],
      "order": 19
    },
    {
      "id": "G2-justify-llm-cost-optimization",
      "linkTo": "Stage-4 goal-gap: 'LLM-as-evaluator cost optimization for Justify stage (Goal 2 portfolio architecture claim)'",
      "location": "engine/services/llm_service.py (evaluate_justification prompt construction + model routing)",
      "change": "Implement the cost-optimization the portfolio claims: (1) move the long, reused system prompt + rubric into a cacheable prefix (provider prompt caching) so per-evaluation token cost drops on repeated rubric reuse, and (2) add a model cascade — route clear pass/fail answers through a cheap small model and escalate only borderline scores to the capable model. Keep G-Eval-style chain-of-thought per-criterion scoring. Make the model ids configurable.",
      "verificationSignal": "Add a test with a mocked client asserting (a) the rubric/system prefix is sent in a cache-eligible position and (b) borderline answers trigger the escalation call while clear ones do not. Measured: a repeated-evaluation benchmark shows reduced token/$ per Justify call versus the single-model baseline; `pytest -k llm_cost` passes.",
      "dependsOn": [
        "S1-llm-fail-closed-and-resolve-test-contradiction"
      ],
      "order": 20
    }
  ],
  "convergenceRounds": 1
}
```
