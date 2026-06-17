# 03 — Execution / Dynamic Surface

## Measured coverage
185 passed, 10 warnings in 4.95s. Engine package total coverage: 42% (3043 stmts, 1759 missed). Per-module highlights: schemas.py 100%, state.py 100%, utils.py 100%, workspace.py 100%, validator.py 92%, stages/justify.py 95%, stages/harden.py 74%, services/llm_service.py 65%, main.py 54%, curriculum.py 60%, services/ast_service.py 23%, ast_harden/pattern_matcher.py 8%, ast_harden/generic_injector.py 10%, ast_harden/softmax_poc.py 0%, ast_harden/softmax_v2_1.py 0%, dev_tools/bug_author.py 0%.

**Line coverage:** 42%

## Observed behaviors (entry points driven)
| Entry point | Observed behavior |
| --- | --- |
| mastery --help (rc=0) | Full command listing rendered including active commands (submit, show, start-challenge, init, curriculum-list, progress-reset, reset, cleanup, select, status, create-bug) and deprecated commands (next, submit-build, submit-justification, submit-fix). No crash or missing command. |
| mastery curriculum-list (rc=0) | Loaded existing state (curriculum=cs336_a1, module_index=0, stage=harden) and loaded LINEAR curriculum with 22 modules. Displayed tabular view showing module 'softmax' as 🔵 (in-progress at HARDEN stage) and all 21 remaining modules as ⚪ (Not Started). Footer shows 'Progress: 0/22 modules completed'. Tip line references 'engine show <module_id>' instead of 'mastery show', suggesting a minor branding inconsistency in output strings. |
| mastery status (rc=0) | Loaded same state. Displayed progress table: Curriculum=cs336_a1, Type=Linear, Current Module='Numerically Stable Softmax (1/22)', Current Stage=HARDEN, Completed Modules=0. Next Action panel shows 'Run mastery start-challenge to begin the harden challenge'. State persisted from a prior session; the harden stage is pending start-challenge invocation. |
| pytest tests/engine (185 tests) | All 185 tests passed. Notable live-log observations: (1) LLMService mock mode triggered when OPENAI_API_KEY absent — WARNING emitted at llm_service.py:65 confirming fail-open path; (2) HardenChallengeError raised and caught when shadow worktree absent (harden.py:80 and :268); (3) 'Not Initialized' Rich panel rendered correctly when worktree missing (main.py); (4) 'All Modules Complete' panel rendered correctly; (5) Full submit→validate cycle rendered 'Validation Passed', 'Bug Fixed', 'Module Complete' Rich panels with performance metric display; (6) Justify fast-filter rejection and LLM correct/incorrect paths all exercised with mock LLM; (7) Patch application failure logged at workspace.py:164; (8) Validator timeout and subprocess errors logged at validator.py:131/136. |
| tests/engine/test_llm_service.py::test_init_missing_api_key_enables_mock_mode | WARNING logged: '⚠️  No OpenAI API key found. LLMService operating in MOCK mode. Justify stage will auto-pass with simulated feedback.' Confirms mock-llm-auth-bypass is live behavior in the test environment (no OPENAI_API_KEY set). |
| tests/engine/test_state.py::TestUserProgressModel::test_mark_stage_complete_harden_advances_module | Test PASSED. schemas.py 100% covered, confirming mark_stage_complete(stage='harden') code path at line 168 was executed. Direct code read confirms line 168 appends f"module_{self.current_module_index}" (with comment 'Will be replaced with actual ID') rather than the actual module ID — CORR-001 is live code, not a dead branch. |
| tests/engine/test_stages.py::TestHardenRunner::test_select_bug_success | Test PASSED using mocked bugs_dir. Code at harden.py:195-196 confirmed to glob both *.patch and *.json files, then line 210 appends '_symptom.txt' to the full stem. The test passes because it uses a controlled mock — draft JSON files are not present in the mock, so the draft-selection defect (CORR-003) is not exercised. |

## Deltas applied to Stage-2 findings
| Finding | Status | Runtime evidence |
| --- | --- | --- |
| CORR-001 | confirmed | Direct read of schemas.py:168 confirms `module_id = f"module_{self.current_module_index}"  # Will be replaced with actual ID` is the live code. schemas.py is 100% covered, meaning this line executed during the test run. The curriculum-list CLI output shows 0/22 modules completed with state at module_index=0 stage=harden — consistent with the bug (no module has yet been fully completed to expose the wrong-ID write, but the code defect is structurally present). test_mark_stage_complete_harden_advances_module PASSED without asserting completed_modules content against real module IDs. |
| CORR-002 | confirmed | Duplicate of library-justify-stub-auto-advance. main.py:613-811 is entirely unexecuted in the test run (coverage report lists 613-811 in missing lines for main.py). The LIBRARY workflow code block containing lines 718-727 was never reached. Static analysis of the code is the sole basis; runtime cannot add further evidence. |
| CORR-003 | confirmed | Direct read of harden.py:195-197 confirms `patch_files = list(bugs_dir.glob('*.patch'))` + `json_files = list(bugs_dir.glob('*.json'))` with random.choice at line 206, and `symptom_name = selected_bug.stem + '_symptom.txt'` at line 210. harden.py has 74% coverage and line 196 is not in the missing-lines list, confirming this path was executed. The test_select_bug tests use mocked bugs_dir with controlled files only, so draft JSON files never appeared in the mocked glob — the draft-selection defect was not caught. Static find confirms 16 *_draft.json files under cs336_a1 bugs dirs and zero *_draft_symptom.txt files. |
| CORR-004 | confirmed | Direct read of harden.py:78 (present_challenge) and :247 (present_library_challenge) confirmed relative Path('.mastery_engine_worktree') used instead of the absolute SHADOW_WORKTREE_DIR. harden.py line 78 is not in the missing lines (74% coverage), so it was executed. test_present_challenge_no_shadow_worktree and test_present_challenge_success both PASSED — but tests mock the path lookup, so the CWD-dependency is not exercised. No os.chdir call exists anywhere in engine/ (confirmed by static analysis). |
| CORR-005 | confirmed | Coverage shows main.py lines 425-443 and 468-477 are missing; line 464 falls in the covered range (444-467). test_submit_routes_to_harden_handler PASSED with mocked dependencies. Static read of main.py:464 confirms `shadow_worktree = Path('.mastery_engine_worktree')` (relative), diverging from the module-level absolute SHADOW_WORKTREE_DIR defined at lines 72-77. Tests use mocks that bypass actual path resolution. |
| library-harden-missing-file-copy | confirmed | main.py lines 613-811 are entirely unexecuted in the test run (listed in coverage missing lines). The LIBRARY workflow _submit_library_workflow is unreachable from the tests/engine test suite. The CLI run used cs336_a1 (LINEAR mode), never triggering the LIBRARY path. Static analysis of lines 754-757 vs 494-496 remains the sole evidence; runtime provides no new confirmation or refutation. |
| library-justify-stub-auto-advance | confirmed | Same evidence as CORR-002: main.py:613-811 is entirely unexecuted per coverage report. Lines 718-727 (the TODO auto-advance) are in this unexecuted block. Static analysis is the only basis. |
| test-deletes-real-user-state | confirmed | tests/e2e/test_error_handling.py was not collected in the pytest run (scope was tests/engine only). The finding is confirmed purely by static analysis: line 126 unconditionally calls state_file.unlink() where state_file = Path.home() / '.mastery_progress.json'. Runtime cannot speak to this — the test was explicitly excluded (destructive-skip). |
| test-corrupts-real-user-state | confirmed | tests/e2e/test_error_handling.py not collected. Static analysis confirmed line 284-285 writes malformed JSON to Path.home() / '.mastery_progress.json' with no restore. Runtime cannot speak to this (destructive-skip). |
| mock-llm-auth-bypass | confirmed | Directly observed in pytest live-log: 'WARNING engine.services.llm_service:llm_service.py:65 ⚠️  No OpenAI API key found. LLMService operating in MOCK mode. Justify stage will auto-pass with simulated feedback.' emitted by test_init_missing_api_key_enables_mock_mode. llm_service.py is 65% covered; lines 60-71 (mock-mode init) were exercised. Confirms the fail-open path is live in the sandbox (no OPENAI_API_KEY set). |
| missing-api-key-behavior-contradiction | refined | Runtime confirms mock mode IS the actual engine behavior: test_init_missing_api_key_enables_mock_mode (tests/engine/test_llm_service.py) PASSED, WARNING logged, llm_service.py:60-71 covered at 65%. This means tests/integration/test_llm_service.py:78-79 (which asserts ConfigurationError on missing key) contradicts both the live code and the passing unit test. The integration test was not collected in this run; if it were, it would fail. The intent_mismatch is refined: the unit test accurately describes real behavior; the integration test is the incorrect assertion. |
| harden-shadow-path-relative | confirmed | Coverage confirms main.py lines around 464 were executed (not in missing ranges 425-443 or 468-477). test_submit_routes_to_harden_handler PASSED with mocked harden stage. Static read of line 464 confirms `shadow_worktree = Path('.mastery_engine_worktree')` as a relative path. No os.chdir in engine/ to make CWD equal to project root. Consistent with CORR-005 (same defect, different location). |
| unsandboxed-student-code-execution | confirmed | validator.py is 92% covered. test_execute_success_with_performance, test_execute_timeout, test_execute_subprocess_error all PASSED — confirming the subprocess.run path at validator.py:108 is exercised with only timeout/env/cwd, no namespace or seccomp. The design defect is structural and cannot be refuted by test passage. |
| e2e-tests-read-write-real-user-state | confirmed | tests/e2e/test_complete_bjh_loop.py was not collected. Static analysis confirmed line 224 reads Path.home() / '.mastery_progress.json' with no HOME isolation. Runtime observation of mastery status/curriculum-list confirms state.py:32 STATE_FILE = Path.home() / '.mastery_progress.json' is the live path — the real user state file exists at this path (curriculum=cs336_a1, module_index=0, stage=harden observed in CLI output). |
| adversarial-test-mutates-real-state | confirmed | tests/e2e/test_adversarial_stress.py not collected. Static analysis confirmed lines 161-166 mutate Path.home() / '.mastery_progress.json'. Confirmed real state file exists at that path (visible in CLI output showing curriculum=cs336_a1 state). Runtime cannot speak further. |
| pretokenization-example-broken | confirmed | tests/test_tokenizer.py was not collected in this run (pytest scoped to tests/engine). The finding is confirmed purely by static analysis: modes/developer/cs336_basics/pretokenization_example.py:53 uses Python Ellipsis (...) as the first argument to open(), which raises TypeError at import time. This is a module-level statement. Not refuted by runtime. |
| bpe-merge-correctness-assertion-commented-out | confirmed | tests/test_train_bpe.py was not collected (hardware-gated, requires cs336_basics PyTorch code). Cannot reach from engine tests. Static analysis confirmed line 54 is commented out. Runtime provides no new evidence. |

## Un-executed regions (100% accounting)
| Location | Reason |
| --- | --- |
| tests/e2e/test_error_handling.py, tests/e2e/test_complete_bjh_loop.py, tests/e2e/test_adversarial_stress.py, tests/e2e/test_full_softmax_loop.py | destructive-skip |
| tests/integration/test_llm_service.py | requires-credentials |
| tests/test_tokenizer.py, tests/test_train_bpe.py, tests/test_model.py, tests/test_optimizer.py, tests/test_serialization.py | hardware-gated |
| tests/conftest.py Snapshot.assert_match (pickle.load) and ts_state_dict fixture (torch.load model.pt) | hardware-gated |
| engine/ast_harden/softmax_poc.py (0% coverage, lines 10-252), engine/ast_harden/softmax_v2_1.py (0% coverage, lines 14-357) | dead |
| engine/dev_tools/bug_author.py (0% coverage, lines 7-975) | requires-credentials |
| engine/main.py:613-811 (LIBRARY workflow: _submit_library_workflow, _submit_library_harden_stage, _submit_library_justify_stage) | external-service |
| engine/main.py:1771-1914 (deprecated submit_fix handler), engine/main.py:978-1096 (deprecated submit_justification handler), engine/main.py:1193-1243 (init shadow-worktree setup branches) | other |
| engine/services/ast_service.py lines 28,57-79,84-100,113,128-167,171-174,186-188,193-261,265-269,280-281,285-290 (AST injection orchestration) | hardware-gated |
| engine/ast_harden/pattern_matcher.py lines 34-466 majority (8% covered), engine/ast_harden/generic_injector.py lines 34-205 majority (10% covered) | hardware-gated |
| engine/stages/harden.py:273-328 (present_library_challenge body), harden.py:249 (library challenge branch) | external-service |
| engine/services/llm_service.py:218-270 (mock-mode evaluate_justification path returning auto-pass response) | requires-credentials |
| curricula/cs336_a1/modules/*/validator.sh (all 22 validators: adamw, attention, bpe_tokenizer, etc.) — pytest tests run against cs336_basics/ PyTorch implementations requiring GPU shadow worktree | hardware-gated |
| scripts/generate_module.py, scripts/enrich_problems.py, scripts/systematic_llm_evaluation.py, scripts/add_successful_to_golden.py, scripts/generate_ground_truth.py, scripts/auto_fix_drafts.py, scripts/fix_draft_pattern.py, scripts/verify_ground_truth.py | requires-credentials |
| mastery start-challenge command (engine/stages/harden.py present_challenge path, engine/main.py start_challenge handler) — requires initialized shadow worktree git worktree | other |

---
### machine-readable artifact
```json
{
  "coverage": {
    "summary": "185 passed, 10 warnings in 4.95s. Engine package total coverage: 42% (3043 stmts, 1759 missed). Per-module highlights: schemas.py 100%, state.py 100%, utils.py 100%, workspace.py 100%, validator.py 92%, stages/justify.py 95%, stages/harden.py 74%, services/llm_service.py 65%, main.py 54%, curriculum.py 60%, services/ast_service.py 23%, ast_harden/pattern_matcher.py 8%, ast_harden/generic_injector.py 10%, ast_harden/softmax_poc.py 0%, ast_harden/softmax_v2_1.py 0%, dev_tools/bug_author.py 0%.",
    "linePct": 42
  },
  "observedBehaviors": [
    {
      "entryPoint": "mastery --help (rc=0)",
      "observed": "Full command listing rendered including active commands (submit, show, start-challenge, init, curriculum-list, progress-reset, reset, cleanup, select, status, create-bug) and deprecated commands (next, submit-build, submit-justification, submit-fix). No crash or missing command."
    },
    {
      "entryPoint": "mastery curriculum-list (rc=0)",
      "observed": "Loaded existing state (curriculum=cs336_a1, module_index=0, stage=harden) and loaded LINEAR curriculum with 22 modules. Displayed tabular view showing module 'softmax' as 🔵 (in-progress at HARDEN stage) and all 21 remaining modules as ⚪ (Not Started). Footer shows 'Progress: 0/22 modules completed'. Tip line references 'engine show <module_id>' instead of 'mastery show', suggesting a minor branding inconsistency in output strings."
    },
    {
      "entryPoint": "mastery status (rc=0)",
      "observed": "Loaded same state. Displayed progress table: Curriculum=cs336_a1, Type=Linear, Current Module='Numerically Stable Softmax (1/22)', Current Stage=HARDEN, Completed Modules=0. Next Action panel shows 'Run mastery start-challenge to begin the harden challenge'. State persisted from a prior session; the harden stage is pending start-challenge invocation."
    },
    {
      "entryPoint": "pytest tests/engine (185 tests)",
      "observed": "All 185 tests passed. Notable live-log observations: (1) LLMService mock mode triggered when OPENAI_API_KEY absent — WARNING emitted at llm_service.py:65 confirming fail-open path; (2) HardenChallengeError raised and caught when shadow worktree absent (harden.py:80 and :268); (3) 'Not Initialized' Rich panel rendered correctly when worktree missing (main.py); (4) 'All Modules Complete' panel rendered correctly; (5) Full submit→validate cycle rendered 'Validation Passed', 'Bug Fixed', 'Module Complete' Rich panels with performance metric display; (6) Justify fast-filter rejection and LLM correct/incorrect paths all exercised with mock LLM; (7) Patch application failure logged at workspace.py:164; (8) Validator timeout and subprocess errors logged at validator.py:131/136."
    },
    {
      "entryPoint": "tests/engine/test_llm_service.py::test_init_missing_api_key_enables_mock_mode",
      "observed": "WARNING logged: '⚠️  No OpenAI API key found. LLMService operating in MOCK mode. Justify stage will auto-pass with simulated feedback.' Confirms mock-llm-auth-bypass is live behavior in the test environment (no OPENAI_API_KEY set)."
    },
    {
      "entryPoint": "tests/engine/test_state.py::TestUserProgressModel::test_mark_stage_complete_harden_advances_module",
      "observed": "Test PASSED. schemas.py 100% covered, confirming mark_stage_complete(stage='harden') code path at line 168 was executed. Direct code read confirms line 168 appends f\"module_{self.current_module_index}\" (with comment 'Will be replaced with actual ID') rather than the actual module ID — CORR-001 is live code, not a dead branch."
    },
    {
      "entryPoint": "tests/engine/test_stages.py::TestHardenRunner::test_select_bug_success",
      "observed": "Test PASSED using mocked bugs_dir. Code at harden.py:195-196 confirmed to glob both *.patch and *.json files, then line 210 appends '_symptom.txt' to the full stem. The test passes because it uses a controlled mock — draft JSON files are not present in the mock, so the draft-selection defect (CORR-003) is not exercised."
    }
  ],
  "findingDeltas": [
    {
      "findingId": "CORR-001",
      "status": "confirmed",
      "evidence": "Direct read of schemas.py:168 confirms `module_id = f\"module_{self.current_module_index}\"  # Will be replaced with actual ID` is the live code. schemas.py is 100% covered, meaning this line executed during the test run. The curriculum-list CLI output shows 0/22 modules completed with state at module_index=0 stage=harden — consistent with the bug (no module has yet been fully completed to expose the wrong-ID write, but the code defect is structurally present). test_mark_stage_complete_harden_advances_module PASSED without asserting completed_modules content against real module IDs."
    },
    {
      "findingId": "CORR-002",
      "status": "confirmed",
      "evidence": "Duplicate of library-justify-stub-auto-advance. main.py:613-811 is entirely unexecuted in the test run (coverage report lists 613-811 in missing lines for main.py). The LIBRARY workflow code block containing lines 718-727 was never reached. Static analysis of the code is the sole basis; runtime cannot add further evidence."
    },
    {
      "findingId": "CORR-003",
      "status": "confirmed",
      "evidence": "Direct read of harden.py:195-197 confirms `patch_files = list(bugs_dir.glob('*.patch'))` + `json_files = list(bugs_dir.glob('*.json'))` with random.choice at line 206, and `symptom_name = selected_bug.stem + '_symptom.txt'` at line 210. harden.py has 74% coverage and line 196 is not in the missing-lines list, confirming this path was executed. The test_select_bug tests use mocked bugs_dir with controlled files only, so draft JSON files never appeared in the mocked glob — the draft-selection defect was not caught. Static find confirms 16 *_draft.json files under cs336_a1 bugs dirs and zero *_draft_symptom.txt files."
    },
    {
      "findingId": "CORR-004",
      "status": "confirmed",
      "evidence": "Direct read of harden.py:78 (present_challenge) and :247 (present_library_challenge) confirmed relative Path('.mastery_engine_worktree') used instead of the absolute SHADOW_WORKTREE_DIR. harden.py line 78 is not in the missing lines (74% coverage), so it was executed. test_present_challenge_no_shadow_worktree and test_present_challenge_success both PASSED — but tests mock the path lookup, so the CWD-dependency is not exercised. No os.chdir call exists anywhere in engine/ (confirmed by static analysis)."
    },
    {
      "findingId": "CORR-005",
      "status": "confirmed",
      "evidence": "Coverage shows main.py lines 425-443 and 468-477 are missing; line 464 falls in the covered range (444-467). test_submit_routes_to_harden_handler PASSED with mocked dependencies. Static read of main.py:464 confirms `shadow_worktree = Path('.mastery_engine_worktree')` (relative), diverging from the module-level absolute SHADOW_WORKTREE_DIR defined at lines 72-77. Tests use mocks that bypass actual path resolution."
    },
    {
      "findingId": "library-harden-missing-file-copy",
      "status": "confirmed",
      "evidence": "main.py lines 613-811 are entirely unexecuted in the test run (listed in coverage missing lines). The LIBRARY workflow _submit_library_workflow is unreachable from the tests/engine test suite. The CLI run used cs336_a1 (LINEAR mode), never triggering the LIBRARY path. Static analysis of lines 754-757 vs 494-496 remains the sole evidence; runtime provides no new confirmation or refutation."
    },
    {
      "findingId": "library-justify-stub-auto-advance",
      "status": "confirmed",
      "evidence": "Same evidence as CORR-002: main.py:613-811 is entirely unexecuted per coverage report. Lines 718-727 (the TODO auto-advance) are in this unexecuted block. Static analysis is the only basis."
    },
    {
      "findingId": "test-deletes-real-user-state",
      "status": "confirmed",
      "evidence": "tests/e2e/test_error_handling.py was not collected in the pytest run (scope was tests/engine only). The finding is confirmed purely by static analysis: line 126 unconditionally calls state_file.unlink() where state_file = Path.home() / '.mastery_progress.json'. Runtime cannot speak to this — the test was explicitly excluded (destructive-skip)."
    },
    {
      "findingId": "test-corrupts-real-user-state",
      "status": "confirmed",
      "evidence": "tests/e2e/test_error_handling.py not collected. Static analysis confirmed line 284-285 writes malformed JSON to Path.home() / '.mastery_progress.json' with no restore. Runtime cannot speak to this (destructive-skip)."
    },
    {
      "findingId": "mock-llm-auth-bypass",
      "status": "confirmed",
      "evidence": "Directly observed in pytest live-log: 'WARNING engine.services.llm_service:llm_service.py:65 ⚠️  No OpenAI API key found. LLMService operating in MOCK mode. Justify stage will auto-pass with simulated feedback.' emitted by test_init_missing_api_key_enables_mock_mode. llm_service.py is 65% covered; lines 60-71 (mock-mode init) were exercised. Confirms the fail-open path is live in the sandbox (no OPENAI_API_KEY set)."
    },
    {
      "findingId": "missing-api-key-behavior-contradiction",
      "status": "refined",
      "evidence": "Runtime confirms mock mode IS the actual engine behavior: test_init_missing_api_key_enables_mock_mode (tests/engine/test_llm_service.py) PASSED, WARNING logged, llm_service.py:60-71 covered at 65%. This means tests/integration/test_llm_service.py:78-79 (which asserts ConfigurationError on missing key) contradicts both the live code and the passing unit test. The integration test was not collected in this run; if it were, it would fail. The intent_mismatch is refined: the unit test accurately describes real behavior; the integration test is the incorrect assertion."
    },
    {
      "findingId": "harden-shadow-path-relative",
      "status": "confirmed",
      "evidence": "Coverage confirms main.py lines around 464 were executed (not in missing ranges 425-443 or 468-477). test_submit_routes_to_harden_handler PASSED with mocked harden stage. Static read of line 464 confirms `shadow_worktree = Path('.mastery_engine_worktree')` as a relative path. No os.chdir in engine/ to make CWD equal to project root. Consistent with CORR-005 (same defect, different location)."
    },
    {
      "findingId": "unsandboxed-student-code-execution",
      "status": "confirmed",
      "evidence": "validator.py is 92% covered. test_execute_success_with_performance, test_execute_timeout, test_execute_subprocess_error all PASSED — confirming the subprocess.run path at validator.py:108 is exercised with only timeout/env/cwd, no namespace or seccomp. The design defect is structural and cannot be refuted by test passage."
    },
    {
      "findingId": "e2e-tests-read-write-real-user-state",
      "status": "confirmed",
      "evidence": "tests/e2e/test_complete_bjh_loop.py was not collected. Static analysis confirmed line 224 reads Path.home() / '.mastery_progress.json' with no HOME isolation. Runtime observation of mastery status/curriculum-list confirms state.py:32 STATE_FILE = Path.home() / '.mastery_progress.json' is the live path — the real user state file exists at this path (curriculum=cs336_a1, module_index=0, stage=harden observed in CLI output)."
    },
    {
      "findingId": "adversarial-test-mutates-real-state",
      "status": "confirmed",
      "evidence": "tests/e2e/test_adversarial_stress.py not collected. Static analysis confirmed lines 161-166 mutate Path.home() / '.mastery_progress.json'. Confirmed real state file exists at that path (visible in CLI output showing curriculum=cs336_a1 state). Runtime cannot speak further."
    },
    {
      "findingId": "pretokenization-example-broken",
      "status": "confirmed",
      "evidence": "tests/test_tokenizer.py was not collected in this run (pytest scoped to tests/engine). The finding is confirmed purely by static analysis: modes/developer/cs336_basics/pretokenization_example.py:53 uses Python Ellipsis (...) as the first argument to open(), which raises TypeError at import time. This is a module-level statement. Not refuted by runtime."
    },
    {
      "findingId": "bpe-merge-correctness-assertion-commented-out",
      "status": "confirmed",
      "evidence": "tests/test_train_bpe.py was not collected (hardware-gated, requires cs336_basics PyTorch code). Cannot reach from engine tests. Static analysis confirmed line 54 is commented out. Runtime provides no new evidence."
    }
  ],
  "unexecutedRegions": [
    {
      "location": "tests/e2e/test_error_handling.py, tests/e2e/test_complete_bjh_loop.py, tests/e2e/test_adversarial_stress.py, tests/e2e/test_full_softmax_loop.py",
      "reason": "destructive-skip"
    },
    {
      "location": "tests/integration/test_llm_service.py",
      "reason": "requires-credentials"
    },
    {
      "location": "tests/test_tokenizer.py, tests/test_train_bpe.py, tests/test_model.py, tests/test_optimizer.py, tests/test_serialization.py",
      "reason": "hardware-gated"
    },
    {
      "location": "tests/conftest.py Snapshot.assert_match (pickle.load) and ts_state_dict fixture (torch.load model.pt)",
      "reason": "hardware-gated"
    },
    {
      "location": "engine/ast_harden/softmax_poc.py (0% coverage, lines 10-252), engine/ast_harden/softmax_v2_1.py (0% coverage, lines 14-357)",
      "reason": "dead"
    },
    {
      "location": "engine/dev_tools/bug_author.py (0% coverage, lines 7-975)",
      "reason": "requires-credentials"
    },
    {
      "location": "engine/main.py:613-811 (LIBRARY workflow: _submit_library_workflow, _submit_library_harden_stage, _submit_library_justify_stage)",
      "reason": "external-service"
    },
    {
      "location": "engine/main.py:1771-1914 (deprecated submit_fix handler), engine/main.py:978-1096 (deprecated submit_justification handler), engine/main.py:1193-1243 (init shadow-worktree setup branches)",
      "reason": "other"
    },
    {
      "location": "engine/services/ast_service.py lines 28,57-79,84-100,113,128-167,171-174,186-188,193-261,265-269,280-281,285-290 (AST injection orchestration)",
      "reason": "hardware-gated"
    },
    {
      "location": "engine/ast_harden/pattern_matcher.py lines 34-466 majority (8% covered), engine/ast_harden/generic_injector.py lines 34-205 majority (10% covered)",
      "reason": "hardware-gated"
    },
    {
      "location": "engine/stages/harden.py:273-328 (present_library_challenge body), harden.py:249 (library challenge branch)",
      "reason": "external-service"
    },
    {
      "location": "engine/services/llm_service.py:218-270 (mock-mode evaluate_justification path returning auto-pass response)",
      "reason": "requires-credentials"
    },
    {
      "location": "curricula/cs336_a1/modules/*/validator.sh (all 22 validators: adamw, attention, bpe_tokenizer, etc.) — pytest tests run against cs336_basics/ PyTorch implementations requiring GPU shadow worktree",
      "reason": "hardware-gated"
    },
    {
      "location": "scripts/generate_module.py, scripts/enrich_problems.py, scripts/systematic_llm_evaluation.py, scripts/add_successful_to_golden.py, scripts/generate_ground_truth.py, scripts/auto_fix_drafts.py, scripts/fix_draft_pattern.py, scripts/verify_ground_truth.py",
      "reason": "requires-credentials"
    },
    {
      "location": "mastery start-challenge command (engine/stages/harden.py present_challenge path, engine/main.py start_challenge handler) — requires initialized shadow worktree git worktree",
      "reason": "other"
    }
  ]
}
```
