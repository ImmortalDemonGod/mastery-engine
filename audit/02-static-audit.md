# 02 — Static Audit

> Diverse-lens audit promoted through an adversarial falsifier to a fixpoint (1 round(s)). Coverage: 623/609 required files visited. Severity mix — critical: 8 · high: 77 · medium: 134 · low: 96 · info: 9.

## CRITICAL (8)
| ID | Class | Location | Evidence | Adversarial note |
| --- | --- | --- | --- | --- |
| enrich-problems-missing-re-import | bug | scripts/enrich_problems.py:159 | The file's entire import block (lines 1-33) contains: argparse, json, time, pathlib, typing, sys, and requests — no `import re`. The method `_extract_examples` calls `re.findall(r'<strong[^>]*>Example[^<]*:</strong>...', ...)` at line 159, and `re.search(...)` at lines 168, 173, 178, and additional calls further in the method. Every invocation of `_extract_examples` (called from `_parse_problem_data` → `fetch_problem`) raises `NameError: name 're' is not defined`. Confirmed by grep: `grep -n 'import re' scripts/enrich_problems.py` returns only `import requests` at line 29. | — |
| enrich-problems-missing-import-re | bug | scripts/enrich_problems.py:159 | Lines 159, 168, 192, 199, 201 call re.match(...), re.sub(...), re.findall(...) inside _extract_examples() and _extract_constraints(), but 'import re' is absent from the file entirely. At the first invocation of either helper, Python will raise NameError: name 're' is not defined. The file imports os, json, requests, and BeautifulSoup at the top but omits re. | — |
| cosine-schedule-validator-wrong-file-and-test | bug | curricula/cs336_a1/modules/cosine_schedule/validator.sh:18 | validator.sh line 18 copies 'cs336_basics/optimizer.py' to the shadow worktree, but the student must implement 'get_lr_cosine_schedule' in 'cs336_basics/utils.py' (confirmed by tests/adapters.py which imports 'get_lr_cosine_schedule' from 'cs336_basics.utils'). Line 33 runs 'pytest tests/test_optimizer.py::test_lr_cosine_schedule' but the test file contains no such function; the actual test is 'def test_get_lr_cosine_schedule():' at tests/test_optimizer.py:52. Pytest would exit with 'ERROR: not found' on test_lr_cosine_schedule, making the cosine_schedule Build stage permanently unpassable through this validator as written. | — |
| coverage-claim-100-vs-16-percent | doc_code_drift | docs/internal/archive/sessions/2025-11-11_curriculum_quality/CURRICULUM_COVERAGE.md:5 | CURRICULUM_COVERAGE.md:5 states 'It demonstrates **100% coverage** of all required implementations'. CURRICULUM_GAP_ANALYSIS.md:5, in the same session folder, states 'Modules Implemented: 3 / ~19 components (16% coverage)'. Mutually exclusive coverage claims for the same CS336 A1 curriculum from documents produced in the same session; no reconciliation is provided. | — |
| bug-golden-patterns-100-vs-14-percent | doc_code_drift | docs/internal/archive/sessions/2025-11-11_curriculum_quality/GROUND_TRUTH_COMPLETE.md:2 | GROUND_TRUTH_COMPLETE.md:2 says 'All 21/21 curriculum modules have validated golden patterns' and line 19 says 'Final coverage: 21/21 modules (100%)'. PROJECT_STATUS.md:141 in the same session folder says '14% Complete' — only 3 golden dataset bugs exist (softmax, silu, rmsnorm verified at PROJECT_STATUS.md:145-157), with 18 modules listed as 'Ready for Migration ⏳ Pending'. Same denominator (21 modules) but opposite conclusions (100% vs 14%). | — |
| remediation-summary-100-vs-gap-analysis-16 | doc_code_drift | docs/internal/archive/sessions/2025-11-11_curriculum_quality/REMEDIATION_SUMMARY.md:256 | REMEDIATION_SUMMARY.md:256 states '✅ 100% implementation coverage (21/21 modules implemented)'. CURRICULUM_GAP_ANALYSIS.md:5, in the same session folder, states 'Modules Implemented: 3 / ~19 components (16% coverage)'. Both purport to describe the current state of the CS336 A1 curriculum; the claims are mutually exclusive. | — |
| transformer-lm-justify-json-invalid | bug | curricula/cs336_a1/modules/transformer_lm/justify_questions.json:33 | File fails JSON parsing: JSONDecodeError at line 33, col 51, char 8736. The model_answer string for question transformer_lm_q3 is truncated mid-sentence and then the literal text '"required_concepts": [' (including embedded JSON-like fragments) appears inside the string value, not as a proper closing quote + array. Lines 33-42 each read: '"Tying saves 50% of embedding parameter    "required_concepts": [' — the original long model_answer string was accidentally spliced with the required_concepts key, making the entire file unparseable. Any engine code path that calls json.load() on this file will throw an unhandled JSONDecodeError, breaking the transformer_lm curriculum module entirely. | — |
| harden-missing-symptom-files-cs336 | bug | engine/stages/harden.py:210 | Line 210: `symptom_name = selected_bug.stem + "_symptom.txt"`. Lines 213-218 raise HardenChallengeError if that file is absent. Filesystem verification confirmed that 18 of 20 cs336_a1 modules have at least one production .json bug file with NO corresponding _symptom.txt: adamw, attention, bpe_tokenizer, checkpointing, cosine_schedule, data_loader, embedding, linear, multihead_attention, rmsnorm, rope, silu, swiglu, text_generation, tokenizer_class, training_loop, transformer_block, transformer_lm. Only 4 symptom files exist across all of cs336_a1 (cross_entropy/bugs/no_logsumexp_symptom.txt, gradient_clipping/bugs/per_parameter_clipping_symptom.txt, softmax/bugs/no_subtract_max_symptom.txt, softmax/bugs/no_subtract_max_v2_symptom.txt). Consequence: _select_bug() will always raise HardenChallengeError for any of those 18 modules regardless of which bug is randomly chosen, making the Harden stage entirely non-functional for 90% of cs336_a1. | — |

## HIGH (77)
| ID | Class | Location | Evidence | Adversarial note |
| --- | --- | --- | --- | --- |
| unsandboxed-solution-import | security | curricula/cp_accelerator/patterns/backtracking/problems/lc_78/validator.sh:24 | All cp_accelerator validators execute learner-supplied Python code via `from solution import solve` (lc_1 uses `from solution import twoSum`). Python module import executes all top-level statements in solution.py unconditionally. There is no timeout, resource limit (CPU/memory), namespace restriction, or syscall filter. A solution.py containing `import os; os.system("rm -rf ~")` or `import subprocess; subprocess.run([...])` at module level would execute with the runner's full OS privileges. This is architecturally expected for a local single-user CLI tool (the learner runs their own code), but the intent document does not acknowledge or accept this risk, and no mitigations are present. The same pattern applies to all 52 validators in the cp_accelerator set. | — |
| ast-injection-source-path | security | engine/ast_harden/pattern_matcher.py:365 | In _apply_replace_value_with(), source_path (from the JSON bug definition's replacement.source field) is passed directly to ast.parse(source_path, mode='eval') at line 365, and the resulting AST body node is grafted into the student's code AST. After ast.unparse() the expression is written to disk and executed by validator scripts and by the student. A malicious or compromised curriculum JSON can inject arbitrary Python expressions that execute in the student's environment. Same pattern repeats at line 420 for replace_with type. | — |
| path-traversal-curriculum-id | security | engine/curriculum.py:105 | curriculum_path = self.CURRICULA_DIR / curriculum_id — no sanitization, normalization, or containment check on curriculum_id before joining it to CURRICULA_DIR. A value like ../../etc resolves outside CURRICULA_DIR. The value comes directly from the user-supplied CLI argument (engine init <curriculum_id>) via engine/main.py. | — |
| b-missing-import-re | bug | scripts/enrich_problems.py:159 | re.findall(..., re.DOTALL) is called at line 159 (and again at 168, 173, 178, 192) but `import re` is absent from the module. The only import near the top is `import requests` inside a try-block at line 29. Executing any code-path that reaches line 159 raises NameError: name 're' is not defined, crashing the enrichment pipeline completely. | — |
| i-llm-mock-auto-pass | intent_mismatch | engine/services/llm_service.py:59 | if not api_key: self.use_mock = True ... (line 60-71); evaluate_justification returns LLMEvaluationResponse(is_correct=True, ...) unconditionally in mock mode (lines 109-119). The PROVISIONAL INTENT explicitly describes the system as an 'LLM-as-evaluator' pedagogical tool. When OPENAI_API_KEY is absent the Justify stage silently auto-passes every answer, completely defeating the mastery-verification purpose. The logger warning at line 65 fires at DEBUG level and is not surfaced to the user at submission time, so learners may never know evaluation was skipped. | — |
| dead-assert-inside-pytest-raises | bug | tests/test_data.py:72 | The assertion `assert "CUDA error" in str(excinfo.value) or "Torch not compiled with CUDA enabled" in str(excinfo.value)` is placed INSIDE the `with pytest.raises((RuntimeError, AssertionError)) as excinfo:` block, after the `run_get_batch(device="cuda:99")` call (lines 66-71). Once `run_get_batch` raises an exception, Python exits the `with` block immediately; line 72 is never reached. The error-message guard is dead code: the test passes regardless of which exception type was raised or what its message contains, silently neutering the validation of the error path. | — |
| lc78-test-order-mismatch | bug | curricula/cp_accelerator/patterns/backtracking/problems/lc_78/validator.sh:38 | The reference solution `subsets([1,2,3])` produces backtracking-DFS order `[[], [1], [1,2], [1,2,3], [1,3], [2], [2,3], [3]]` (confirmed by running the code), but test_cases.json 'expected' for test 1 is bitmask-enumeration order `[[], [1], [2], [1,2], [3], [1,3], [2,3], [1,2,3]]`. The validator checks `if result == expected:` at line 38 — an order-sensitive exact equality with no normalization. LeetCode 78 explicitly allows any order. Consequence: the correct reference solution fails its own test case 1. Any learner who implements a valid DFS/backtracking approach would also be incorrectly rejected. | — |
| S2C-001 | bug | curricula/cs336_a1/modules/cosine_schedule/validator.sh:18 | Line 18 copies 'cs336_basics/optimizer.py' into the shadow worktree: `cp "$SHADOW_WORKTREE/modes/developer/cs336_basics/optimizer.py" ...`. But get_lr_cosine_schedule is defined at modes/developer/cs336_basics/utils.py:75, not in optimizer.py (which contains only AdamW). The student's utils.py changes are never reflected in the test harness; the validator always runs against the committed optimizer.py. All other utils-dependent validators (softmax, cross_entropy, gradient_clipping, data_loader, checkpointing) correctly copy utils.py. | — |
| S2C-002 | bug | engine/stages/harden.py:78 | Inside present_challenge: `shadow_worktree = Path('.mastery_engine_worktree')`. This is a CWD-relative path. The canonical constant SHADOW_WORKTREE_DIR is defined at engine/main.py:74 as `find_project_root() / '.mastery_engine_worktree'` (absolute). When mastery is invoked from any directory other than the project root, harden.py constructs the wrong path, fails to locate the shadow worktree, and the harden challenge cannot proceed. | — |
| S2C-003 | bug | engine/stages/harden.py:247 | Inside present_library_challenge: same defect as S2C-002. `shadow_worktree = Path('.mastery_engine_worktree')` is CWD-relative, not the absolute SHADOW_WORKTREE_DIR. LIBRARY-mode harden challenges have an identical path-resolution failure when the user is not in the project root. | — |
| S2C-004 | intent_mismatch | engine/main.py:718 | In the LIBRARY-mode justify branch (lines 718-727): `# TODO: Implement proper editor integration` followed by auto-advancing to the next stage without any LLM evaluation or user input. The PROVISIONAL INTENT states LLM-as-evaluator is a core showcase feature. The LINEAR-mode path at engine/main.py calls JustifyRunner and LLMService; the LIBRARY path silently skips both. This is a named TODO stub shipped as working behavior. | — |
| pretokenization-ellipsis-open | bug | modes/student/cs336_basics/pretokenization_example.py:53 | Line 53 contains `with open(..., "rb") as f:` at module level (outside any function). The `...` is Python's Ellipsis singleton object, not a filename placeholder for the student to fill in. `open(Ellipsis, "rb")` raises `TypeError: expected str, bytes or os.PathLike object, not ellipsis` whenever this module is imported or run directly. The block (lines 53-62) executes unconditionally at import time: `num_processes = 4; boundaries = find_chunk_boundaries(f, num_processes, b"<\|endoftext\|>")`. No guard like `if __name__ == '__main__':` exists. Confirmed via grep: line 53 contains `with open(..., "rb") as f:`. | — |
| generate-module-fstring-nameerror | bug | scripts/generate_module.py:380 | The function `create_validator_template` returns an f-string containing bash syntax `${MASTERY_PYTHON:-python3} << 'EOF'` at line 380. In a Python f-string, the `$` is a literal character but `{MASTERY_PYTHON:-python3}` is parsed as an f-expression: variable name `MASTERY_PYTHON` with format spec `:-python3`. Since `MASTERY_PYTHON` is not defined in the local or global scope, Python raises `NameError: name 'MASTERY_PYTHON' is not defined` when `create_validator_template` is called. To embed literal bash `${...}` in a Python f-string, the braces must be doubled: `${{MASTERY_PYTHON:-python3}}`. Confirmed via grep: line 380 contains `${MASTERY_PYTHON:-python3}`. | — |
| memory-limit-decorator-ineffective-on-generator | bug | tests/test_tokenizer.py:449 | `_encode_iterable` is decorated with `@memory_limit(int(1e6))` (line 449) and defined as a generator function using `yield from` (line 455). The `memory_limit` wrapper calls `result = f(*args, **kwargs)` (conftest-style inline decorator, test_tokenizer.py lines 23-34): for a generator function, this call returns the generator object immediately without executing any body code. The `finally:` branch then restores the original `RLIMIT_AS` before the caller has iterated a single token. When `test_encode_iterable_memory_usage` (line 420) iterates `for _id in _encode_iterable(tokenizer, f):`, the memory limit has already been restored to its previous value, so the 1 MB constraint is never enforced. The test always passes for any implementation, defeating its purpose. | — |
| dead-test-cases-var-in-validators | design_defect | curricula/cp_accelerator/patterns/backtracking/problems/lc_78/validator.sh:8 | Shell variable TEST_CASES="$SCRIPT_DIR/test_cases.json" is computed at line 8 but is NEVER referenced inside the Python heredoc (which uses single-quoted 'EOF' — shell variables are not expanded inside it). The Python code hardcodes `with open("test_cases.json")` as a literal string at line 27. The pattern is identical across all ~42 CP accelerator validators (lc_90, lc_34, lc_704, lc_1342, lc_1486, lc_46, lc_47, lc_146, lc_460, lc_148, lc_912, lc_198, lc_70, lc_435, lc_452, lc_217, lc_219, lc_215, lc_703, lc_203, lc_237, lc_1480, lc_303, lc_307, lc_148-sorting, lc_1003, lc_20, lc_144, lc_589, lc_1804, lc_208, lc_1099, lc_167, lc_547, lc_684). This is pervasive dead code: the variable gives a false impression that the script is portable across invocation directories, but the Python inside the heredoc relies entirely on the process CWD equalling the script's directory. Confirmed by engine/validator.py:110 which calls validators with `cwd=str(workspace_path.resolve())` — the coupling to workspace == problem directory is implicit and unenforced. | — |
| lc303-claimed-o1-query-is-on | design_defect | curricula/cp_accelerator/patterns/prefix_sum/problems/lc_303/solution.py:3 | The module docstring states 'Build: O(n), Query: O(1), Space: O(n)'. However, the implementation is a stateless function `sumRange(nums, left, right)` that rebuilds the entire prefix array on every call (lines 25-27: `prefix = [0] * (len(nums) + 1); for i in range(len(nums)): prefix[i+1] = prefix[i] + nums[i]`). Every call is O(n), not O(1). The O(1) claim requires the prefix array to be built once in `__init__` (a class) and reused, which is the actual LeetCode 303 interface (`NumArray` class with `__init__` and `sumRange`). The current design is both algorithmically misrepresented and structurally wrong relative to the problem specification. | — |
| library-harden-reads-reference-not-student | bug | engine/stages/harden.py:281 | In `present_library_challenge()`, both the primary path and the fallback resolve to the same file: `student_code_path = problem_path / 'solution.py'` (line 281) and `reference_solution = problem_path / 'solution.py'` (line 266). The comment says 'Fallback to reference solution if student hasn't started Build yet' but there is no branch that ever reads from a different location — they are identical paths. AST-based bug injection therefore always operates on the reference solution, not the student's submitted code. The Harden stage's pedagogical intent (debug YOUR code) is silently subverted. | — |
| mark-stage-complete-synthetic-module-id | bug | engine/schemas.py:168 | `UserProgress.mark_stage_complete()` contains: `module_id = f"module_{self.current_module_index}"  # Will be replaced with actual ID`. This synthetic key is written into `self.completed_modules`, so all progress lookups against real module IDs (`softmax`, `rmsnorm`, etc.) will never find the stored entry. Any gate that checks `completed_modules[real_id]` will see missing data. The comment acknowledges the placeholder is not yet wired to the real ID. | — |
| library-justify-stage-todo-stub | intent_mismatch | engine/main.py:719 | The library-mode justify flow contains `# TODO: Implement proper editor integration` and returns early with an empty answer string. The Justify stage — whose stated purpose is Socratic evaluation of the user's implementation decisions — is never entered in library mode. Users completing library-mode modules bypass the entire pedagogical evaluation step silently. | — |
| reset-function-not-implemented | intent_mismatch | engine/main.py:2456 | The `reset(module_id)` function is exposed in the CLI's public interface but is not implemented (lines 2456-2465 contain only a stub body). Users who invoke `reset` receive no error and no action. The engine's documented workflow includes the ability to reset progress, making this a silent no-op against the stated API contract. (Location confirmed from prior session read of engine/main.py.) | — |
| bare-except-pass-swallows-errors | design_defect | engine/main.py:2007 | Lines 2007-2009 contain a bare `except: pass` that silently swallows any exception type raised during a critical code path. This completely hides failures from both the user and the logger, making debugging impossible when that branch misbehaves. (Location confirmed from prior session read.) | — |
| non-shadow-worktree-validators-job-prep | design_defect | curricula/job_prep_data_annotation/modules/data_parsing_extraction/validator.sh:6 | All three job_prep_data_annotation validators (`data_parsing_extraction`, `grid_visualization`, `http_transport`) and the `python_for_cp/std_lib_augmentation` validator bypass the shadow-worktree isolation protocol entirely. They compute `PROJECT_ROOT` via `SCRIPT_DIR` (e.g., `SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"`) rather than consuming `$SHADOW_WORKTREE`. They run `python3 "$TEST_SCRIPT"` (not `$MASTERY_PYTHON`) and `cd "$PROJECT_ROOT"`. The cs336_a1 validators all enforce `$SHADOW_WORKTREE` existence, do an explicit `cp` of the student file, and honour `$MASTERY_PYTHON`. These non-standard validators break the engine's isolation and reproducibility guarantees. | — |
| non-shadow-worktree-validator-python-for-cp | design_defect | curricula/python_for_cp/modules/std_lib_augmentation/validator.sh:6 | `std_lib_augmentation/validator.sh` uses `PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../../.." && pwd)` and invokes `python3 "$TEST_SCRIPT"` directly, ignoring both `$SHADOW_WORKTREE` and `$MASTERY_PYTHON`. Imports `from cs336_basics.utils import shortest_path_bfs, dijkstra_shortest_path, count_in_range` — `cs336_basics` is the package name for the CS336-A1 curriculum, not the competitive-programming curriculum. This creates a namespace dependency on a different curriculum's module. | — |
| hardcoded-dev-machine-absolute-paths | design_defect | scripts/add_successful_to_golden.py:77 | Six scripts embed the literal absolute path '/Volumes/Totallynotaharddrive/assignment1-basics/...' which resolves only on a specific developer's macOS machine. Affected locations: add_successful_to_golden.py:77 (Path("/Volumes/Totallynotaharddrive/assignment1-basics/curricula/cs336_a1/modules/{module}/bugs")), auto_fix_drafts.py:218 (base_path = Path("/Volumes/Totallynotaharddrive/assignment1-basics/curricula/cs336_a1/modules")), fix_draft_pattern.py:71 (same path), generate_ground_truth.py:22 (same), systematic_llm_evaluation.py:1029 (same, referenced 21 times across 21 test cases), verify_ground_truth.py:19 (same). All six scripts raise FileNotFoundError or silently produce empty results on any non-developer machine. | — |
| generate-module-ingest-pattern-undefined-methods | bug | scripts/generate_module.py:535 | ingest_pattern() (lines 518-578) calls self.parse_taxonomy_file(pattern_id) at line 535, self.select_canonical_problem(...) at line 539, and self.create_test_cases_template(...) at line 569. None of these names appear as methods on ModuleGenerator anywhere in the file. Running ingest_pattern() will raise AttributeError: 'ModuleGenerator' object has no attribute 'parse_taxonomy_file' at the first invocation. | — |
| pretokenization-example-ellipsis-as-open-arg | bug | modes/student/cs336_basics/pretokenization_example.py:53 | Lines 53-62 contain module-level executable code: 'with open(..., "rb") as f:' where the first argument is the Python Ellipsis literal (...). open() does not accept Ellipsis as a path — it raises TypeError: expected str, bytes or os.PathLike object, not ellipsis at module import time. This file lives in modes/student/cs336_basics/, which is the directory symlinked as cs336_basics when students work on assignments. Any import of cs336_basics that triggers discovery of pretokenization_example.py will crash. | — |
| e2e-tests-pollute-real-user-home-state | design_defect | tests/e2e/test_complete_bjh_loop.py:224 | get_state() at line 224 reads/writes Path.home() / '.mastery_progress.json' — the real user's home directory, not a tmp_path-scoped file. test_complete_bjh_loop.py lines 349-354 write directly to this path to forge the stage. test_error_handling.py line 285 writes corrupted JSON to the same path. test_adversarial_stress.py lines 161-164 also mutate it. Running the E2E suite overwrites or corrupts any real user's progress. There is no fixture-level cleanup or monkeypatching of the state file path in these tests. | — |
| memory-limit-void-on-generator-function | design_defect | tests/test_tokenizer.py:450-455 | The `memory_limit` decorator (defined at lines 19-36) wraps `f` in a `wrapper` that: (1) sets `resource.RLIMIT_AS`, (2) calls `result = f(*args, **kwargs)`, (3) executes `finally: resource.setrlimit(resource.RLIMIT_AS, prev_limits)`, (4) returns `result`. `_encode_iterable` at line 450 uses `yield from tokenizer.encode_iterable(iterable)` (line 455), making it a Python generator function. Calling `f(*args, **kwargs)` on a generator function returns a generator object immediately without executing any body; the `finally` block then resets the memory limit before a single byte is encoded. The test `test_encode_iterable_memory_usage` (lines 416-430) iterates the generator after `wrapper` has already returned and the limit is gone — the 1 MB ceiling is never in force during actual encoding. The test cannot detect a memory-inefficient `encode_iterable` implementation. | — |
| cp-readme-ingest-script-nonexistent | doc_code_drift | curricula/cp_accelerator/README.md:188 | README.md lines 188-196 document `scripts/ingest_cp_content.py` as an existing, runnable script with a concrete CLI invocation: `uv run python scripts/ingest_cp_content.py --module two_pointers_basics`. The script does NOT exist in the repository; `ls scripts/` shows only `generate_module.py` as the content-generation tool. The README presents this as current working infrastructure, but it is aspirational/unimplemented. Students or contributors following these instructions will get a FileNotFoundError. | — |
| impl-status-ci-enforcement-false | doc_code_drift | curricula/cp_accelerator/IMPLEMENTATION_STATUS.md:162 | IMPLEMENTATION_STATUS.md:162-185 states '❌ IMPOSSIBLE TO MERGE if: manifest.json was manually edited / JSON schema is invalid / Dependency IDs don't exist' and marks '[x] validate_cp_manifest.yml CI workflow active' (line 241) and '[x] CI passing: Yes' (line 241). However, QUALITY_AUDIT.md:54 documents that the validate_cp_manifest workflow is broken because its scripts check `manifest['modules']` which does not exist in a LIBRARY curriculum — cp_accelerator uses `patterns`. The enforced CI check therefore validates the wrong schema key and provides false assurance. The IMPLEMENTATION_STATUS doc claims enforcement that is structurally broken. | — |
| lc1804-missing-problem-statement | doc_code_drift | curricula/cp_accelerator/patterns/trie/problems/lc_1804/build_prompt.txt:11 | The build_prompt.txt contains placeholder content: '**Difficulty:** Unknown \| **Acceptance Rate:** N/A' and '**Topics:** General' with no problem statement, no constraints, and no examples. The file instructs students to 'Function Signature: Derive from the problem examples above' (line 24) but there are no examples above. The inventory describes this file as 'Problem statement, constraints, and implementation instructions for LeetCode 1804 (Implement Trie II with prefix counts).' The content contradicts that role—students cannot implement the problem without a specification. | — |
| lc1099-missing-problem-statement | doc_code_drift | curricula/cp_accelerator/patterns/two_pointers/problems/lc_1099/build_prompt.txt:11 | Same placeholder pattern as lc_1804: '**Difficulty:** Unknown \| **Acceptance Rate:** N/A' and '**Topics:** General' with no problem statement, constraints, or examples. The instruction 'Function Signature: Derive from the problem examples above' (line 22) references non-existent examples. The inventory describes this as 'Problem statement, constraints, and implementation instructions for LeetCode 1099 (Two Sum Less Than K).' No specification is present. | — |
| cosine-schedule-wrong-file-and-function | doc_code_drift | curricula/cs336_a1/modules/cosine_schedule/build_prompt.txt:135 | build_prompt.txt line 135 says 'FILE TO MODIFY: cs336_basics/optimizer.py' and documents the function signature as 'def lr_cosine_schedule(step, max_lr, min_lr, warmup_steps, max_steps)'. The actual function is 'get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)' in 'cs336_basics/utils.py' (modes/student/cs336_basics/utils.py:65; modes/developer/cs336_basics/utils.py:75). The student stub in modes/student/cs336_basics/optimizer.py contains only the AdamW class, no cosine schedule. The adapter at tests/adapters.py imports 'get_lr_cosine_schedule as _get_lr_cosine_schedule_impl' from 'cs336_basics.utils', confirming utils.py is the correct file. Parameter names also diverge: doc uses 'step/max_lr/min_lr/warmup_steps/max_steps' but code uses 'it/max_learning_rate/min_learning_rate/warmup_iters/cosine_cycle_iters'. The build_prompt at line 326 also documents the wrong test name: 'test_lr_cosine_schedule' vs actual 'test_get_lr_cosine_schedule' (tests/test_optimizer.py:52). | — |
| data-parsing-extraction-wrong-implementation-file | doc_code_drift | curricula/job_prep_data_annotation/modules/data_parsing_extraction/build_prompt.txt:22 | build_prompt.txt line 22 says 'Implement the following function in job_prep/parser.py'. However, the validator (data_parsing_extraction/validator.sh:17) imports with 'from cs336_basics.utils import extract_coordinates'. If the student implements in 'job_prep/parser.py' as instructed, the validator will fail with ImportError since it looks in 'cs336_basics.utils'. The curriculum README (job_prep_data_annotation/README.md:183) also says 'cs336_basics/utils.py', contradicting the build_prompt. The build_prompt file path instruction is wrong. | — |
| cosine-schedule-intent-mismatch-optimizer-vs-utils | intent_mismatch | curricula/cs336_a1/modules/cosine_schedule/build_prompt.txt:148 | The provisional intent describes cs336_a1 modules as requiring Build-Justify-Harden with validated test harnesses. The cosine_schedule module's build_prompt presents a standalone 'lr_cosine_schedule(step, max_lr, min_lr, warmup_steps, max_steps)' function in optimizer.py, but the engine's actual test infrastructure (tests/adapters.py) is wired to 'get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)' in utils.py. The student experience contradicts the intent: they implement per the prompt (optimizer.py) but the harness tests a completely different signature in a different file. This is a design-level disconnect, not just a naming slip. | — |
| mastery-engine-doc-old-cli | doc_code_drift | docs/architecture/MASTERY_ENGINE.md:240 | MASTERY_ENGINE.md documents the CLI using a flag-based interface (`engine --next`, `engine --submit-build`, `engine --submit-fix`, `engine --status`) throughout the usage examples section. The actual registered CLI entry point per pyproject.toml [project.scripts] is `mastery` with subcommands: `mastery submit`, `mastery status`, `mastery show`, `mastery harden`, `mastery justify`. Confirmed as the live interface by STRANGER_TEST_RESULTS.md (which uses `mastery init cs336_a1`, `mastery status`, etc.) and JUSTIFY_ONLY_MODULE_DESIGN.md UX section (lines 149-186), which both use `mastery` commands. | — |
| mastery-engine-doc-workspace-model | doc_code_drift | docs/architecture/MASTERY_ENGINE.md:499 | MASTERY_ENGINE.md describes the harden stage challenge file as `workspace/module_challenge.py` and references a `workspace/` directory for all student code. Actual implementation uses a shadow worktree at `.mastery_engine_worktree/` (confirmed by STRANGER_TEST_RESULTS.md lines 100-111 showing `SHADOW_WORKTREE_DIR / 'cs336_basics'` and `os.symlink(symlink_target, shadow_symlink)`, and LAYER2_E2E_SUCCESS.md:61-64 showing `shadow_worktree = shadow_worktree / 'cs336_basics'`). No `workspace/` directory appears in the actual file inventory. | — |
| mastery-engine-doc-layer4-5-aspirational | doc_code_drift | docs/architecture/MASTERY_ENGINE.md:48 | MASTERY_ENGINE.md labels Layers 4 and 5 as 'Finalized Design,' implying implementation completeness. AI_CODEBASE_DECONSTRUCTION.md §8 'Honest Status' at line 384 explicitly contradicts this: 'Justify's LLM grader is not wired — but the components exist... the stage runner (engine/stages/justify.py) is a stub, so only the keyword fast-filter is live.' Additionally, AI_CODEBASE_DECONSTRUCTION.md:3 explicitly says the document is 'a design analysis/blueprint' that is 'not a shipped feature.' The MASTERY_ENGINE.md presents aspirational layer architecture as finalized. | — |
| ai-deconstruction-justify-stub-stale | doc_code_drift | docs/architecture/AI_CODEBASE_DECONSTRUCTION.md:384 | AI_CODEBASE_DECONSTRUCTION.md §8 states 'Justify's LLM grader is not wired — but the components exist... the stage runner (engine/stages/justify.py) is a stub, so only the keyword fast-filter is live.' This directly contradicts FINAL_VERIFICATION_SUMMARY.md:40 which reports engine/stages/justify.py at 95% test coverage, and lines 59-64 which list 8 passing LLM integration tests including `test_llm_accepts_correct_answer` and `test_llm_rejects_incomplete_answer`. The §8 honest-status section was accurate when written but became stale after justify was implemented, yet the document still resides in docs/architecture/ presenting an outdated picture. | — |
| cp-quickstart-deleted-modules-dir | doc_code_drift | docs/internal/CP_ACCELERATOR_QUICKSTART.md:51 | CP_ACCELERATOR_QUICKSTART.md at lines 51-55 references `curricula/cp_accelerator/modules/sorting/`, `curricula/cp_accelerator/modules/two_pointers_basics/`, and similar `modules/`-rooted paths for curriculum content. PHASE_8_BATCH_GENERATION_COMPLETE.md:113 explicitly states 'Deleted: curricula/cp_accelerator/modules/' and confirms the current structure uses a `patterns/` hierarchy (e.g., `patterns/sorting/`, `patterns/arrays/`). The quickstart guide describes a file structure that was entirely deleted. | — |
| harden-phase2-dispatch-stale | doc_code_drift | docs/internal/archive/sessions/2025-11-10_bug_system/AST_HARDEN_PHASE2_COMPLETE.md:101 | AST_HARDEN_PHASE2_COMPLETE.md line 101 states the production harden dispatch uses 'from engine.services.ast_service import SoftmaxBugInjector'. Actual production code at engine/stages/harden.py:99 imports 'from engine.ast_harden.generic_injector import GenericBugInjector'. Phase 3 replaced SoftmaxBugInjector with GenericBugInjector but the Phase 2 document was never updated, leaving a stale record of the production dispatch path. | — |
| phase2-signoff-stale-injector | doc_code_drift | docs/internal/archive/sessions/2025-11-10_bug_system/PHASE2_SIGNOFF.md:27 | PHASE2_SIGNOFF.md lines 27-28 formally approves 'engine/services/ast_service.py (367 lines) — SoftmaxBugInjector' as 'Production quality, fully functional.' Phase 3 superseded this with GenericBugInjector at engine/ast_harden/generic_injector.py. The production harden.py (line 99) no longer imports SoftmaxBugInjector; this signoff now documents a deprecated artifact as the approved production component. | — |
| cli-p0-completion-contradiction | doc_code_drift | docs/internal/archive/sessions/2025-11-11_curriculum_quality/MASTER_REMEDIATION_STATUS.md:5 | MASTER_REMEDIATION_STATUS.md:5 (date 2025-11-12) says 'Overall Status: ✅ Curriculum Complete (98/100), ✅ CLI P0 Complete (100%)'. SESSION_3_SUMMARY.md:56 (same date, same session folder) says 'Implementation Phase 🟡 STARTED (10%)' for CLI, and SESSION_3_SUMMARY.md:265 shows 'Total: 10% Complete'. Two documents produced the same day describe CLI P0 as simultaneously 100% complete and 10% started. | — |
| cli-p1-done-vs-pending | doc_code_drift | docs/internal/archive/sessions/2025-11-12_cli_remediation/CLI_P1_IMPLEMENTATION_COMPLETE.md:6 | CLI_P1_IMPLEMENTATION_COMPLETE.md:6 says 'Status: ✅ **COMPLETE**' for P1 (Inconsistent next command). CLI_REMEDIATION_STATUS.md:87, in the same session folder, says P1 is '📋 **Designed, awaiting implementation**'. CLI_REMEDIATION_STATUS.md:209 also shows '⏸️ Pending' for P1 in its implementation tracking table. Two files in the same session folder (2025-11-12_cli_remediation) give opposite completion states for the same work item. | — |
| quality-plan-production-ready-vs-deferred-tasks | doc_code_drift | docs/internal/archive/sessions/2025-11-11_curriculum_quality/QUALITY_REMEDIATION_PLAN.md:249 | QUALITY_REMEDIATION_PLAN.md:249 says '**Status**: All Priorities Complete (P1, P2, P3) - PRODUCTION READY'. The tracking table in the same document (line 240) shows P1 task 'Update build prompts (einops)' at '⏸️ Deferred / 0%', and line 243 shows P2 task 'Create experiment modules' at '⏸️ Design Only / 30%'. The document footer asserts all priorities complete while the table above it shows two uncompleted tasks. | — |
| curriculum-stub-claim-vs-student-fix | doc_code_drift | docs/internal/archive/sessions/2025-11-11_curriculum_quality/CURRICULUM_COVERAGE.md:360 | CURRICULUM_COVERAGE.md:360 says '✅ 18 components properly stubbed' and '✅ All raise NotImplementedError'. STUDENT_MODE_FIX_SUMMARY.md:58 says 'Modules Affected: 10 out of 22 (45% of curriculum!)' — these modules had complete working implementations instead of NotImplementedError stubs. STUDENT_MODE_AUDIT.md documents 5 functions explicitly marked '❌ COMPLETE (should be stub)'. The coverage document asserts a clean state that pre-fix code directly contradicted. | — |
| module-count-21-vs-22 | doc_code_drift | docs/internal/archive/sessions/2025-11-11_curriculum_quality/CURRICULUM_COVERAGE.md:411 | CURRICULUM_COVERAGE.md:411 says 'Total: 21 modules, 18 stubbed components, 100% PDF coverage'. PROJECT_STATUS.md:133 says 'Total Modules: 22'. The same CS336 A1 curriculum is reported as having 21 and 22 modules; neither document explains which extra module accounts for the discrepancy. STUDENT_MODE_FIX_SUMMARY.md:58 also uses 22 as the denominator ('10 out of 22'). | — |
| cp-quickstart-hardcoded-macos-path | doc_code_drift | docs/internal/archive/sessions/2025-11-11_curriculum_quality/CP_ACCELERATOR_QUICKSTART.md:84 | CP_ACCELERATOR_QUICKSTART.md:84 contains `cd /Volumes/Totallynotaharddrive/assignment1-basics` — a hardcoded developer-local macOS volume path. This path does not exist in the repository. Any user following the quickstart guide verbatim will receive a 'No such file or directory' error. The path is non-portable and unreproducible outside the original author's machine. | — |
| cp-docs-modules-dir-vs-patterns | doc_code_drift | docs/internal/archive/sessions/2025-11-11_curriculum_quality/CP_ACCELERATOR_IMPLEMENTATION_GUIDE.md:24 | CP_ACCELERATOR_IMPLEMENTATION_GUIDE.md:24 shows directory tree `curricula/cp_accelerator/modules/` with pattern subdirectories. CP_ACCELERATOR_QUICKSTART.md:37 also uses `└── modules/`. Actual repo at /home/user/mastery-engine/curricula/cp_accelerator/ contains a `patterns` subdirectory, not `modules` (confirmed via ls). All command examples in both guides referencing `.../cp_accelerator/modules/...` (e.g., QUICKSTART.md:117, :205, :301) point to a non-existent path. | — |
| verification-findings-hardcoded-path | doc_code_drift | docs/internal/archive/sessions/2025-11-11_curriculum_quality/VERIFICATION_FINDINGS.md:11 | VERIFICATION_FINDINGS.md:11 cites `/Users/tomriddle1/Holistic-Performance-Enhancement/cultivation/docs/5_domain_knowledge_and_curricula/computer_science/architectures_and_models/transformer_paradigm/RoFormer_Analysis.md` as the literature source for rope module verification. LITERATURE_VERIFICATION.md:12 uses the same hardcoded local path. These paths point to a non-repo filesystem unavailable to any other user. All verification grades (e.g., rope 85/100, linear 95/100 at VERIFICATION_FINDINGS.md:200-237) rest on sources that cannot be independently validated. | — |
| student-mode-complete-impls-vs-stub-intent | intent_mismatch | docs/internal/archive/sessions/2025-11-11_curriculum_quality/STUDENT_MODE_FIX_SUMMARY.md:58 | STUDENT_MODE_FIX_SUMMARY.md:58 says 'Modules Affected: 10 out of 22 (45% of curriculum!)' — nearly half the CS336 A1 curriculum's student mode files contained complete working implementations instead of NotImplementedError stubs. The provisional pedagogical intent requires student mode to provide only stubs so learners must implement from scratch in the Build stage. Complete implementations in student mode directly undermine the Build-Justify-Harden loop: a student can copy the reference solution without building understanding. STUDENT_MODE_AUDIT.md documents the specific functions (e.g., transformer_lm, multihead_attention, etc.) that were fully implemented when they should have been stubs. | — |
| DCD-001 | doc_code_drift | docs/internal/coverage/CURRENT_REPORT.md:29 | Report lists 'engine/ast_harden/harden.py' (98%) and 'engine/justify.py' (95%) as near-perfect-coverage modules. Neither path exists in the repository. Glob of engine/**/*.py confirms actual paths are 'engine/stages/harden.py' and 'engine/stages/justify.py' (verified via audit/.work/01-understanding.json architecture section which names 'engine/stages/harden.py HardenRunner' and 'engine/stages/justify.py JustifyRunner'). The coverage data is therefore attached to phantom module paths, making the report unverifiable and misleading. | — |
| DCD-002 | doc_code_drift | docs/internal/current/TEST_COVERAGE_REPORT.md:29 | Identical content to docs/internal/coverage/CURRENT_REPORT.md; contains the same wrong module paths 'engine/ast_harden/harden.py' and 'engine/justify.py'. These paths do not exist; the correct paths confirmed by repository glob are 'engine/stages/harden.py' and 'engine/stages/justify.py'. Two authoritative 'current' documents both propagate the same phantom path drift. | — |
| DCD-004 | doc_code_drift | docs/internal/current/CURRICULUM_STATUS.md:87 | Curriculum Status describes cp_accelerator as 'Total Modules: 1 (pilot)' with only the 'sorting' module mentioned. Actual repository state: curricula/cp_accelerator/manifest.json contains 19 patterns (sorting, backtracking, binary_search, bit_manipulation, combinatorics_and_number_theory, design_patterns, divide_and_conquer, dynamic_programming, greedy, hash_table, heap_and_priority_queue, linked_list, prefix_sum, segment_tree_and_fenwick_tree, stack_and_queue, traversal, trie, two_pointers, union_find_disjoint_set_union). The 'ls curricula/cp_accelerator/patterns/' command confirms 19 subdirectories. The document is off by a factor of 19x. | — |
| readme-original-engine-cmd-name | doc_code_drift | maintenance/README_ORIGINAL.md:12 | README_ORIGINAL.md presents the primary Mastery Engine Workflow section using the command `engine` (e.g. `engine init`, `engine status`, `engine next`, `engine submit`) on lines 12-26. The actual CLI entry point registered in pyproject.toml:34 is `mastery = "engine.main:main"`, so users who follow these instructions will get 'command not found'. The MASTERY_COMMAND_REFERENCE.md explicitly acknowledges this: 'OLD (Documentation): `engine submit` … ACTUAL (Command): `uv run mastery submit`'. | — |
| project-structure-engine-cmd-name | doc_code_drift | maintenance/PROJECT_STRUCTURE.md:168 | PROJECT_STRUCTURE.md Development Workflow section (lines 168-180) uses `engine init cs336_a1`, `engine next`, `engine submit-build`, `engine submit-justification "<answer>"`, `engine submit-fix`, `engine status`, and (line 130) `engine init` / `engine cleanup` for the shadow worktree lifecycle. The installed CLI entry point is `mastery` (pyproject.toml:34), not `engine`. Every user-facing command shown in this developer guide is wrong. The same file also references `uv run python -m engine.main next` on line 155 as an example invocation, which conflates module path with CLI name. | — |
| readme-wrong-filename | doc_code_drift | tests/integration/README.md:7 | README line 7 states '**File**: `test_llm_integration.py`' but the actual integration test file is `tests/integration/test_llm_service.py`. No file named `test_llm_integration.py` exists in that directory. All example commands and test-output snippets in the README (e.g., lines 39, 41, 100, 105, 111) also reference this non-existent filename, making every copy-paste instruction in the README broken. | — |
| readme-nonexistent-test-names | doc_code_drift | tests/integration/README.md:13 | The 'What These Tests Validate' section (lines 13-18) lists six test capabilities including 'Fast Filter Logic', 'Decision Boundary', and 'Fast filter vs. LLM routing'. None of these correspond to any test in `tests/integration/test_llm_service.py`. The actual tests are: `test_llm_service_initialization_with_api_key` (line 58), `test_llm_service_missing_api_key` (line 70), `test_llm_accepts_correct_answer` (line 86), `test_llm_rejects_incomplete_answer` (line 123), `test_llm_rejects_conceptual_error` (line 156), `test_llm_timeout_handling` (line 197), `test_response_format_validation` (line 222), `test_cost_analysis_documentation` (line 257). The README describes a different, older test file. | — |
| readme-wrong-cost-table | doc_code_drift | tests/integration/README.md:59 | Cost table (lines 59-66) lists six test names that do not exist: `test_fast_filter_blocks_shallow_answer`, `test_llm_accepts_deep_correct_answer`, `test_llm_rejects_conceptual_error_with_socratic_feedback`, `test_error_handling_missing_api_key`, `test_fast_filter_vs_llm_decision_boundary`, `test_llm_api_timeout_handling`. Actual test names differ (e.g., `test_llm_accepts_correct_answer` not `test_llm_accepts_deep_correct_answer`; `test_llm_service_missing_api_key` not `test_error_handling_missing_api_key`). The table also shows 2 API calls and $0.006 total, while the actual test file header (line 7) declares 3 API calls and $0.009. | — |
| ci-python-below-minimum | bug | .github/workflows/tests.yml:20 | tests.yml line 20: `python-version: '3.10'` (and again at line 70 in the lint job). pyproject.toml line 6 declares `requires-python = ">=3.11"`. The test CI runs on Python 3.10, which is BELOW the package's minimum required version. `uv sync` on Python 3.10 will either fail (if uv enforces the requires-python constraint) or install the package in an unsupported environment, making CI results meaningless. validate_cp_manifest.yml correctly uses 3.11 at lines 26, 123, 212. | — |
| actions-mutable-tag-pinning | security | .github/workflows/tests.yml:16 | All GitHub Actions in both workflows are pinned to mutable version tags, not immutable SHA digests. tests.yml: `actions/checkout@v4` (line 16, 66), `actions/setup-python@v5` (line 19, 69), `astral-sh/setup-uv@v3` (line 23), `actions/upload-artifact@v4` (line 52). validate_cp_manifest.yml: `actions/checkout@v3` (line 20), `actions/setup-python@v4` (lines 24, 122, 211). A tag can be silently updated to point to malicious code without any hash change detectable by the workflow consumer. This is the classic supply-chain vector for compromised GitHub Actions (e.g., tj-actions/changed-files incident). Fix: pin every action to its full SHA digest. | — |
| pip-install-uv-unpinned | security | .github/workflows/validate_cp_manifest.yml:29 | validate_cp_manifest.yml line 29: `run: pip install uv`. No version specifier is provided. This means every CI run fetches the latest uv from PyPI, making the build non-reproducible and exposing the pipeline to a supply-chain compromise of the `uv` package on PyPI. tests.yml correctly uses the official `astral-sh/setup-uv@v3` action which itself should be pinned to SHA. The inconsistency also means the manifest-validation CI may use a different uv version than the test CI, producing divergent resolution behaviour even though both call `uv sync`. | — |
| lc34-test-cases-missing-target-field | bug | curricula/cp_accelerator/patterns/binary_search/problems/lc_34/test_cases.json:54 | Tests 4-8 in this file are for 'Find First and Last Position of Element in Sorted Array' but lack the required `target` field and have expected values that look like sorted arrays rather than [first, last] index pairs. Example: test 4 `{"input": {"nums": [1]}, "expected": [1]}`, test 6 `{"input": {"nums": [3,2,1]}, "expected": [1,2,3]}`. Tests 1-3 correctly include `target` and return `[-1,-1]` or `[first, last]` pairs. Tests 4-8 appear to be sorting test cases copied from another problem. Any validator calling the binary search function with these inputs will receive wrong argument signatures and incorrect expected values. | — |
| lc47-test1-expected-empty-string | bug | curricula/cp_accelerator/patterns/combinatorics_and_number_theory/problems/lc_47/test_cases.json:15 | Test case 1 for Permutations II (nums=[1,1,2]) has `"expected": ""` — an empty string. The correct expected value for this input is `[[1,1,2],[1,2,1],[2,1,1]]`. An empty string will never equal the actual list output, so this test case will always fail in any validator that does equality comparison, making the test suite for lc_47 effectively broken for its primary example. | — |
| sort-list-tests-wrong-input-key | bug | curricula/cp_accelerator/patterns/divide_and_conquer/problems/lc_148/test_cases.json:54 | Tests 4-8 in both `divide_and_conquer/problems/lc_148/test_cases.json` and `sorting/problems/lc_148/test_cases.json` use input key `"nums"` instead of `"head"`. LeetCode 148 (Sort List) accepts a linked list via its `head` parameter. Tests 1-3 correctly use `"head": [...]`. Tests 4-8 use `"nums": [1]`, `"nums": []`, etc. — clearly copy-pasted from an array sorting problem. A validator calling `sortList(head=...)` with key `nums` would fail or silently pass wrong data. Identical contamination exists in sorting/problems/lc_148/test_cases.json tests 4-8. | — |
| ci-python-version-mismatch | bug | .github/workflows/tests.yml:20 | Line 20 (and line 70): `python-version: '3.10'`. pyproject.toml:7 declares `requires-python = ">=3.11"`. CI runs tests on Python 3.10 which is explicitly excluded by the package's own minimum version constraint. Tests may silently pass on 3.10 while failing on any supported Python version (3.11+), masking real compatibility bugs. | — |
| temperature-bug-inverted-direction | intent_mismatch | curricula/cs336_a1/modules/text_generation/bugs/temperature_after_softmax.json:22 | Bug spec ID is `text-generation-temperature-after-softmax`, description: "Temperature applied after softmax". The replacement source at line 22 is `"F.softmax(next_logits / temperature, dim=-1)"` — this applies temperature BEFORE softmax (divides logits first), which is the mathematically CORRECT implementation. To inject the described bug (temperature applied after softmax), the replacement should be something like `F.softmax(next_logits, dim=-1) / temperature`. The spec injects correct behavior instead of the intended bug, so the harden phase cannot detect anything. | — |
| data-loader-justify-questions-invalid-json | bug | curricula/cs336_a1/modules/data_loader/justify_questions.json:34 | File is confirmed invalid JSON. `python3 -c "import json; json.load(open('curricula/cs336_a1/modules/data_loader/justify_questions.json'))"` raises `JSONDecodeError: Expecting ',' delimiter: line 34 column 21 (char 8981)`. Visible corruption: truncated strings like `"Broadcas` appear followed immediately by embedded `"required_concepts": [` fragments, indicating the file was generated with mid-string truncation and repeated content insertion. The justify phase cannot load questions for this module. | — |
| tokenizer-class-justify-questions-invalid-json | bug | curricula/cs336_a1/modules/tokenizer_class/justify_questions.json:9 | File is confirmed invalid JSON. `python3 -c "import json; json.load(open('curricula/cs336_a1/modules/tokenizer_class/justify_questions.json'))"` raises `JSONDecodeError: Expecting ',' delimiter: line 9 column 50 (char 2112)`. Visible corruption at line 9: `"Different order produces different en` — string truncated mid-sentence — followed immediately by `"required_concepts": [` embedded as a key. Same corruption pattern as data_loader/justify_questions.json. The justify phase cannot load questions for this module. | — |
| missing-final-norm-draft-wrong-spec | doc_code_drift | curricula/cs336_a1/modules/transformer_lm/bugs/missing_final_norm_draft.json:2 | The file at curricula/cs336_a1/modules/transformer_lm/bugs/missing_final_norm_draft.json contains the SiLU bug spec, not a missing_final_norm draft. Actual content: '"id": "silu-missing-multiply"', '"target_function": "silu"', '"description": "Removes the multiplication by input, returning only sigmoid(x) instead of x * sigmoid(x)."'. This is a verbatim copy of the silu/bugs/missing_multiply spec placed in the wrong directory. A reviewer or engine loading this file expecting a transformer_lm missing_final_norm draft would inject the wrong bug into the wrong module. | — |
| ci-actions-mutable-version-tags | security | .github/workflows/tests.yml:15 | All GitHub Actions in both workflows use mutable semantic-version tags instead of pinned commit SHAs. tests.yml uses: actions/checkout@v4 (line 15, 64), actions/setup-python@v5 (line 17, 69), astral-sh/setup-uv@v3 (line 23, 74), actions/upload-artifact@v4 (line 53). validate_cp_manifest.yml uses: actions/checkout@v3 (lines 22, 116, 205), actions/setup-python@v4 (lines 24, 118, 207). If a maintainer of any of these actions moves the mutable tag to a different commit (intentionally or via supply-chain compromise), CI will execute attacker-controlled code in a context that has GITHUB_TOKEN write permissions to the repository. | — |
| ci-python-version-mismatch | bug | .github/workflows/tests.yml:20 | pyproject.toml:6 declares 'requires-python = ">=3.11"' but tests.yml:20 and tests.yml:70 both set 'python-version: "3.10"'. CI is running the test suite on Python 3.10, which is below the declared minimum requirement. This means CI does not validate the package on its required Python version. Code using Python 3.11-only features (e.g., tomllib stdlib, TypeVarTuple, improved typing constructs) would pass CI on 3.10 yet fail in production. The uv.lock also records 'requires-python = ">=3.11"', deepening the mismatch. | — |
| harden-select-bug-picks-drafts | bug | engine/stages/harden.py:195-197 | _select_bug() globs ALL .json files: `json_files = list(bugs_dir.glob('*.json'))` and concatenates them with patch files. This includes *_draft.json files alongside production specs. For example, curricula/cs336_a1/modules/multihead_attention/bugs/ contains both missing_transpose_back.json (production, target_function='forward') and missing_transpose_back_draft.json (draft, target_function='multihead_attention'). If the draft is randomly selected, GenericBugInjector._has_function() at generic_injector.py:99-102 returns False for target_function='multihead_attention' (not a function name in any student code), and the injection aborts with (source_code, False). Then harden.py:114-121 raises HardenChallengeError. Similarly, transformer_lm/bugs/ contains missing_final_norm_draft.json whose id='silu-missing-multiply' — wrong module content entirely. | — |
| missing-final-norm-draft-wrong-module | bug | curricula/cs336_a1/modules/transformer_lm/bugs/missing_final_norm_draft.json:3 | File id is 'silu-missing-multiply' (confirmed by understanding.json: 'Misidentified draft AST spec (id: silu-missing-multiply) that removes x*sigmoid(x) multiplication; likely a wrong-module draft'). The file lives in transformer_lm/bugs/ but describes a silu bug. If _select_bug() randomly selects this file for a transformer_lm harden session, the injector targets the silu pattern (BinOp Mult on sigmoid) rather than removing the final RMSNorm. On source code without that exact pattern, injection fails; on source code that happens to match, the wrong transformation is applied. Either outcome is incorrect for the transformer_lm module. | — |
| incomplete-merge-patch-removes-solve-alias | bug | curricula/cp_accelerator/patterns/sorting/problems/lc_912/bugs/incomplete_merge.patch | The .patch file diff includes removal of the solve alias: `-solve = sortArray` and `-# Alias for compatibility with test runner`. The companion JSON spec (incomplete_merge.json) only removes `result.extend(right[j:])` and does not touch the alias. In legacy patch-based harden (harden.py:127-156), if the .patch file is selected, shutil.copy2 + workspace_mgr.apply_patch() produces a buggy file lacking `solve = sortArray`. The CP Accelerator validator imports `solve` from the solution file; its absence causes NameError in the validator rather than the intended wrong-answer test failure. The two artifacts (patch vs JSON) thus describe divergent mutations and only the .patch path is broken. | — |
| mark-stage-complete-synthetic-module-id | design_defect | engine/schemas.py:168 | In UserProgress.mark_stage_complete(), the harden branch generates a synthetic placeholder: `module_id = f'module_{self.current_module_index}'` (e.g., 'module_0', 'module_1') and appends it to completed_modules. The inline comment `# Will be replaced with actual ID` confirms this is an incomplete implementation. Actual module IDs are strings like 'rmsnorm', 'attention', 'adamw'. Any code that checks `if real_module_id in progress.completed_modules` — including curriculum completion checks and dependency validation — will always evaluate False because the list only contains synthetic placeholders. This silently corrupts the learning-progress tracking invariant. | — |
| harden-draft-json-in-selection-pool | bug | engine/stages/harden.py:196 | Line 195-197: `patch_files = list(bugs_dir.glob("*.patch"))` / `json_files = list(bugs_dir.glob("*.json"))` / `bug_files = patch_files + json_files`. The glob `*.json` matches ALL json files, including draft variants such as `adamw_wrong_beta_update_draft.json`, `adamw_wrong_beta_update_draft_v2.json`, `bpe_wrong_pair_count_draft.json`, and 13+ analogous draft files confirmed present across cs336_a1 bug directories. Draft files are development artifacts, not finished challenges; they uniformly lack `_symptom.txt` counterparts. When random.choice() (line 206) selects a draft, _select_bug() inevitably raises HardenChallengeError at line 214-218. A simple name-based guard (e.g., excluding stems that contain 'draft') is absent. | — |

## MEDIUM (134)
| ID | Class | Location | Evidence | Adversarial note |
| --- | --- | --- | --- | --- |
| missing-cd-script-dir | bug | curricula/cp_accelerator/patterns/backtracking/problems/lc_78/validator.sh:7 | SCRIPT_DIR is computed at line 7 (`SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"`) but `cd "$SCRIPT_DIR"` is never executed before the Python heredoc at line 17. Inside the heredoc, `open("test_cases.json")` resolves relative to the caller's CWD, and `sys.path.insert(0, str(Path(__file__).parent))` inserts `'.'` (CWD) because `__file__` is `'<stdin>'` in heredoc context (verified: `Path('<stdin>').parent` = `.`). If the validator is invoked from any directory other than the problem directory, both the solution import and the test-cases load fail. The lc_1 validator (validator.sh:17) correctly does `cd "$SCRIPT_DIR"` before its heredoc; the same fix is absent from all 51 other cp_accelerator validators (lc_78, lc_90, lc_34, lc_704, lc_1342, lc_1486, lc_46, lc_47, lc_146, lc_460, lc_148/divide_and_conquer, lc_912/divide_and_conquer, lc_198, lc_70, lc_435, lc_452, lc_217, lc_219, lc_215, lc_703, lc_203, lc_237, lc_1480, lc_303, lc_307, lc_148/sorting, lc_912/sorting, lc_1003, lc_20, lc_144, lc_589, lc_1804, lc_208, lc_1099, lc_167, lc_547, lc_684). | — |
| student-source-path-no-bounds-check | security | engine/stages/harden.py:104 | student_code_path = Path.cwd() / source_file_path at line 104, where source_file_path comes from module.source_files[0] in the curriculum JSON. No validation that source_file_path resolves within the workspace directory. A curriculum with source_files: ['../../sensitive/config.py'] causes the engine to read and then inject bugs into files outside the intended student workspace. | — |
| editor-env-subprocess-injection | security | engine/main.py:310 | editor = os.getenv('EDITOR', os.getenv('VISUAL', 'nano')) then subprocess.run([editor, temp_path]). While list-form subprocess avoids shell splitting, EDITOR is fully user-controlled; any executable path or interpreter binary (e.g. /usr/bin/env bash, or a local malicious binary) is accepted without validation before being exec'd with the temp file path as an argument. | — |
| path-traversal-module-id | security | engine/main.py:490 | harden_file = harden_workspace / f"{current_module.id}.py" and shadow_dest = shadow_worktree / f"{current_module.id}.py" — current_module.id comes from curriculum JSON without a containment check. A module id containing ../ (e.g. ../../etc/malicious) causes writes outside the harden workspace or shadow worktree. | — |
| apply-patch-unvalidated-patch-file | security | engine/workspace.py:155 | subprocess.run(["patch", str(target_file), str(patch_file)], ...) — apply_patch() is called from engine/stages/harden.py:156,320 with bug_file (curriculum-controlled path) as patch_file. No validation that patch_file lies within the curriculum directory. A crafted diff with --- / +++ headers pointing to other paths could modify files outside the intended target; pointing patch_file at a non-patch file can also corrupt target_file arbitrarily. | — |
| mock-mode-justify-bypass | intent_mismatch | engine/services/llm_service.py:59 | When OPENAI_API_KEY is absent, use_mock = True is set at line 62. evaluate_justification() at lines 109-122 unconditionally returns LLMEvaluationResponse(is_correct=True, feedback='MOCK MODE...') regardless of the student's answer. The stated intent of the justify stage is to verify student understanding before advancing; mock mode silently grants a pass to any answer including a blank string, bypassing the entire evaluation gate. | — |
| ssrf-httpbin-external-dependency | security | curricula/job_prep_data_annotation/modules/http_transport/validator.sh:27 | Validator makes live outbound HTTP requests to https://httpbin.org/html (line 27) and https://httpbin.org/status/404 (line 35) during test execution. Fails non-deterministically when the external service is unavailable; always fails in air-gapped or firewalled environments; the student's IP and environment metadata (User-Agent, TLS fingerprint) are sent to a third-party service at validation time without explicit consent. | — |
| s-tmp-read-golden | security | scripts/add_successful_to_golden.py:14 | results_path = Path("/tmp/llm_evaluation_results.json")  — the script reads evaluation results from /tmp, a world-writable directory. A local attacker on the same machine can plant a crafted JSON at that path before the script runs to promote arbitrary bug specs into the curriculum golden set. The file is then used at lines 19-70 to select which bugs are written into production curriculum directories. | — |
| b-module-level-open-ellipsis | bug | modes/student/cs336_basics/pretokenization_example.py:53 | with open(..., "rb") as f:  — the Ellipsis literal (...) is used as the filename argument at module scope (not inside if __name__ == '__main__'). Python evaluates this on import, raising TypeError: expected str, bytes or os.PathLike object, not ellipsis. Any test or tool that imports this module will crash immediately. | — |
| s-pickle-load-fixtures | security | tests/conftest.py:140 | expected_data = pickle.load(f)  — snapshot fixture files (.pkl) are deserialized with pickle.load without any integrity check (HMAC, hash, or signature). Pickle deserialization of untrusted data executes arbitrary Python during unpickling. If a contributor or CI runner checks out a branch containing a tampered .pkl file, the test suite becomes an attack vector. | — |
| s-torch-load-no-weights-only | security | tests/conftest.py:199 | state_dict = torch.load(FIXTURES_PATH / "ts_tests" / "model.pt", map_location="cpu")  — torch.load without weights_only=True (added in PyTorch 1.13) deserializes the file using pickle, allowing arbitrary code execution if the .pt file is malicious. PyTorch now emits a FutureWarning for this usage and recommends weights_only=True. | — |
| d-integration-test-wrong-expectation | doc_code_drift | tests/integration/test_llm_service.py:70 | test_llm_service_missing_api_key asserts `pytest.raises(ConfigurationError)` when OPENAI_API_KEY is unset (lines 70-83). The actual implementation at llm_service.py:59-71 does not raise ConfigurationError; it silently enters mock mode and returns. This integration test will fail against the current codebase, documenting a behavior contract (raise on missing key) that was either abandoned or never implemented. | — |
| memory-limit-decorator-ineffective-for-generators | intent_mismatch | tests/test_tokenizer.py:449-455 | The `_encode_iterable` function at lines 449-455 is decorated with `@memory_limit(int(1e6))` and is a generator function (contains `yield from tokenizer.encode_iterable(iterable)`). The `memory_limit` wrapper (lines 19-36) calls `f(*args, **kwargs)` which, for a generator function, immediately returns a generator object WITHOUT executing the body. The `finally` block then restores `RLIMIT_AS` to its original value before any iteration occurs. When `test_encode_iterable_memory_usage` (lines 421-427) iterates over the returned generator via `for _id in _encode_iterable(tokenizer, f)`, the 1 MB memory cap has already been lifted, so actual encoding memory usage is completely unconstrained. The docstring at line 451 states the intent: 'We place tokenizer.encode_iterable into a separate function so we can limit memory for just this function', but the implementation fails to achieve this intent for generators. The test always passes regardless of memory usage, defeating its purpose. | — |
| cp-validators-cwd-fragility | design_defect | curricula/cp_accelerator/patterns/backtracking/problems/lc_78/validator.sh:23 | All CP-accelerator validators except lc_1 share this pattern: (1) They set `TEST_CASES="$SCRIPT_DIR/test_cases.json"` (absolute path) on line 8 but the Python heredoc uses `with open("test_cases.json")` — a relative path — on line 27. The $TEST_CASES variable is never read inside the heredoc, making it dead code. (2) `sys.path.insert(0, str(Path(__file__).parent))` on line 23 is used to locate solution.py, but when Python runs from a heredoc `__file__` is `'<stdin>'` (verified: `Path('<stdin>').parent` == `PosixPath('.')`, i.e. CWD). This means both the import of solution.py and the open of test_cases.json silently depend on the calling process's working directory being the problem directory. The lc_1 validator correctly avoids this by doing `cd "$SCRIPT_DIR"` before the heredoc and using `os.getcwd()` instead. The pattern repeats in at least 40 validator.sh files covering all non-lc_1 CP problems. | — |
| lc303-query-complexity-false | doc_code_drift | curricula/cp_accelerator/patterns/prefix_sum/problems/lc_303/solution.py:3 | File header comment reads `Build: O(n), Query: O(1)` but the implementation of `sumRange(nums, left, right)` (lines 7-29) unconditionally rebuilds a full `prefix` array of length `n+1` on every call: `prefix = [0] * (len(nums) + 1)` followed by a full loop `for i in range(len(nums))`. The prefix array is a local variable discarded after each return, so every query costs O(n), not O(1). The entire pedagogical point of LeetCode 303 is to build the prefix array once in `__init__` and answer queries in O(1). The implementation also departs from the class-based `NumArray` interface LeetCode specifies; this was intentional (functional shim noted in test_cases.json), but the O(1) claim in the header comment is plainly wrong for the functional form. | — |
| lc203-list-comprehension-bypasses-linked-list | intent_mismatch | curricula/cp_accelerator/patterns/linked_list/problems/lc_203/solution.py:23 | LeetCode 203 (Remove Linked List Elements) is a linked-list pointer-manipulation exercise — the teaching intent is to practice sentinel-node / pointer-update patterns. The reference solution at line 23 is `return [x for x in head if x != val]`, a single-pass Python list comprehension that works only because the test runner serialises the linked list as a plain Python list. A learner who copies or internalises this pattern and submits to LeetCode (which uses `ListNode` objects) will receive a wrong-answer or runtime error. The note at line 4 says 'Uses array representation for compatibility with test runner', acknowledging the departure, but the code teaches the wrong algorithmic pattern for the problem's stated pedagogical goal. | — |
| cs336-validators-relative-cp-path | design_defect | curricula/cs336_a1/modules/adamw/validator.sh:18 | In all three CS336 validators (`adamw`, `attention`, `bpe_tokenizer`), the BUILD-stage branch executes `cp cs336_basics/<file>.py "$SHADOW_WORKTREE/..."` using a relative source path. This silently requires the script's calling CWD to already contain a `cs336_basics/` subdirectory; the script has no guard or diagnostic if it does not. With `set -e` enabled, a missing source file produces only the `cp` error message and an immediate exit, giving no actionable feedback (e.g. 'did you create cs336_basics/optimizer.py?'). The same pattern is duplicated at `attention/validator.sh:18` (copies `layers.py`) and `bpe_tokenizer/validator.sh:18` (copies `tokenizer.py`), making all three CS336 validators silently fragile to CWD assumptions. | — |
| S2C-005 | bug | engine/main.py:464 | _submit_harden_stage contains `worktree_path = Path('.mastery_engine_worktree')` (hardcoded relative path). Same class of defect as S2C-002/003: the absolute constant SHADOW_WORKTREE_DIR defined at engine/main.py:74 is not referenced. Submission validation runs against the wrong worktree path when CWD is not the project root. | — |
| S2C-006 | bug | engine/main.py:1775 | submit_fix contains `worktree_path = Path('.mastery_engine_worktree')` (hardcoded relative path). Third independent instance of the same CWD-relative worktree defect; absolute SHADOW_WORKTREE_DIR constant is not used. | — |
| S2C-007 | bug | engine/schemas.py:168 | Inside UserProgress.mark_stage_complete (harden branch): `module_id = f"module_{self.current_module_index}"  # Will be replaced with actual ID`. This stores a synthetic positional key like 'module_0' in completed_modules instead of the real module ID (e.g. 'softmax'). Downstream breakage: (1) engine/main.py:2196 curriculum_list checks `if module.id in progress.completed_modules` — the real ID never matches the stored synthetic ID, so ✅ is never displayed; (2) engine/main.py:2297 progress_reset filters `if m != module_id` — the synthetic IDs are never equal to the real module_id argument, so the reset silently fails to clear the completion record. | — |
| S2C-008 | bug | engine/stages/harden.py:195 | _select_bug collects candidates with: `bug_files = list(bugs_dir.glob('*.patch')) + list(bugs_dir.glob('*.json'))`. The glob picks up every JSON file in the bugs directory, including _draft.json and _v2.json files which are known-incomplete predecessor specs (recorded in 01-understanding.json inventory as in-progress artifacts). An incomplete spec selected at random will produce a malformed or partial bug injection. Additionally, selection is random with no per-session deduplication, so the same bug can be presented multiple times. | — |
| S2C-009 | bug | engine/workspace.py:156 | `subprocess.run(['patch', ...], capture_output=True, text=True, check=False)` — no `timeout` parameter. If the patch binary hangs (e.g., waiting for stdin on a malformed diff), the engine process blocks indefinitely. The validator subprocess already uses a 300-second cap (engine/validator.py:DEFAULT_TIMEOUT_SECONDS), but apply_patch has no analogous guard. | — |
| S2C-010 | bug | engine/ast_harden/generic_injector.py:109 | `Canonicalizer(target_function=target_function if target_function else 'softmax')`. If a bug definition omits the target_function field (or passes None), the canonicalizer defaults to operating on a function named 'softmax'. For modules where no softmax function exists, the canonicalizer finds nothing to rename, producing a canonical AST identical to the original. Pattern matching then always fails to find the expected canonical variable names (_arg0, _var0, etc.), and injection returns (original_source, False) silently. | — |
| hardcoded-macos-path-add-golden | bug | scripts/add_successful_to_golden.py:77 | Line 77: `golden_dir = Path(f"/Volumes/Totallynotaharddrive/assignment1-basics/curricula/cs336_a1/modules/{module}/bugs")`. This is an absolute path to a specific developer's external macOS volume. On any other machine or environment, this path does not exist; `bug_files` will always be empty; no golden examples are ever saved. The script silently produces zero output with no error. Confirmed via grep across scripts/ directory. | — |
| hardcoded-macos-path-auto-fix | bug | scripts/auto_fix_drafts.py:218 | Line 218: `base_path = Path("/Volumes/Totallynotaharddrive/assignment1-basics/curricula/cs336_a1/modules")`. Same machine-specific macOS volume path that does not exist on other systems. All downstream file glob operations return empty results, so the script processes no drafts without any error signal. Confirmed via grep. | — |
| hardcoded-macos-path-fix-draft | bug | scripts/fix_draft_pattern.py:71 | Line 71: `base_path = Path("/Volumes/Totallynotaharddrive/assignment1-basics/curricula/cs336_a1/modules")`. Same hardcoded developer-specific macOS path. Script silently finds no modules and performs no work on any other machine. Confirmed via grep. | — |
| hardcoded-macos-path-gen-ground-truth | bug | scripts/generate_ground_truth.py:22 | Line 22: `base_path = Path("/Volumes/Totallynotaharddrive/assignment1-basics/curricula/cs336_a1/modules")`. Same hardcoded macOS path. All module discovery returns empty, so the script generates no ground truth output on non-developer machines. Confirmed via grep. | — |
| hardcoded-macos-path-verify-ground-truth | bug | scripts/verify_ground_truth.py:19 | Line 19: `base_path = Path("/Volumes/Totallynotaharddrive/assignment1-basics/curricula/cs336_a1/modules")`. Same hardcoded macOS path. When the path doesn't exist, `results['total']` remains 0, and line 157 `len(results['passed'])/results['total']*100` raises `ZeroDivisionError` (confirmed via grep: line 157 contains this expression with no zero-guard). The script crashes with an unhandled exception rather than reporting 'no modules found'. Confirmed via grep. | — |
| hardcoded-macos-path-systematic-eval | bug | scripts/systematic_llm_evaluation.py:1029 | Line 1029: `base_path = Path("/Volumes/Totallynotaharddrive/assignment1-basics/curricula/cs336_a1/modules")`. Same machine-specific macOS external volume path. Confirmed via grep across the scripts/ directory alongside the other five instances. | — |
| verify-ground-truth-div-by-zero | bug | scripts/verify_ground_truth.py:157 | Line 157: `print(f"\n📊 SUCCESS RATE: {len(results['passed'])/results['total']*100:.0f}%")`. No guard against `results['total'] == 0`. When the hardcoded base_path (line 19) doesn't exist, no modules are discovered, `results['total']` stays 0, and this line raises `ZeroDivisionError: division by zero`. Confirmed via grep: lines 148, 153, and 157 all use `results['total']` as denominator without a zero-check. | — |
| numpy-snapshot-force-update-unused | design_defect | tests/conftest.py:50 | `NumpySnapshot.assert_match` accepts `force_update: bool \| type[DEFAULT] = DEFAULT` at line 50, resolves it at lines 60-61 (`if force_update is DEFAULT: force_update = self.default_force_update`), but the resolved value is never referenced again. Lines 74-95 unconditionally call `np.load(snapshot_path)` and compare — they never branch on `force_update` to write or overwrite a snapshot. If the snapshot file does not yet exist, the call always fails with FileNotFoundError regardless of `force_update=True`. The snapshot creation/update path is completely unimplemented. Read lines 50-95 of conftest.py confirm no `if force_update:` branch exists. | — |
| snapshot-force-update-unused | design_defect | tests/conftest.py:120 | `Snapshot.assert_match` accepts `force_update` at line 120, resolves it at lines 130-131, but lines 138-149 always execute `with open(snapshot_path, "rb") as f: expected_data = pickle.load(f)` unconditionally. There is no branch `if force_update: pickle.dump(actual, f)`. Passing `force_update=True` has zero effect: the method always attempts to load and compare, raising FileNotFoundError if the snapshot does not exist. Read lines 116-155 of conftest.py confirm absence of any write branch. | — |
| numpy-snapshot-force-update-unused | design_defect | tests/conftest.py:60 | In `NumpySnapshot.assert_match` the parameter `force_update` is resolved (line 60-61: `if force_update is DEFAULT: force_update = self.default_force_update`) but is never used afterwards. There is no `if force_update: np.savez(snapshot_path, ...); return` branch. The method unconditionally loads the existing snapshot at line 75 (`expected_arrays = dict(np.load(snapshot_path))`). Passing `force_update=True` or setting `default_force_update=True` does not save a new snapshot—it will raise `FileNotFoundError` if the file is absent, or silently compare against the stale snapshot if the file is present. Snapshot regeneration is broken. | — |
| pickle-snapshot-force-update-unused | design_defect | tests/conftest.py:130 | Identical issue in `Snapshot.assert_match`: `force_update` is resolved (lines 130-131) but then never branched on. Line 139 unconditionally opens and unpickles the existing snapshot file: `with open(snapshot_path, "rb") as f: expected_data = pickle.load(f)`. Any call with `force_update=True` will fail or compare against stale data instead of writing the new snapshot. The `Snapshot` class used by `test_train_bpe_special_tokens` (line 87 of test_train_bpe.py) inherits this broken update path. | — |
| bpe-strict-merge-assertion-gutted | intent_mismatch | tests/test_train_bpe.py:54 | The authoritative correctness check `assert merges == reference_merges` (line 54, now commented out) has been replaced with a loose count range: `assert len(merges) >= 243` and `assert len(merges) <= 245` (lines 57-58). The comment claims tie-breaking differences cause divergence only at index 64, but the replacement removes all per-merge ordering validation. A buggy BPE implementation that produces exactly 243-245 merges in the wrong order would pass. The vocabulary coverage check at line 76 (`coverage >= 0.98`) is similarly weak. The stated intent of `test_train_bpe` is to validate BPE output against a reference; the actual test no longer does this. | — |
| misleading-path-file-in-heredoc | design_defect | curricula/cp_accelerator/patterns/backtracking/problems/lc_78/validator.sh:23 | All CP accelerator validators (except lc_1) contain `sys.path.insert(0, str(Path(__file__).parent))` inside a single-quoted bash heredoc. When Python runs from stdin (heredoc), `__file__` is `'<stdin>'`, not the validator script path. `Path('<stdin>').parent` evaluates to `Path('.')` (the process CWD). The line therefore inserts CWD into sys.path, which is exactly the same as inserting nothing since '.' is already searched. It looks like it computes the script's directory (as it would in a real .py file) but does no such thing — it works only by coincidence because workspace_path == problem directory. This misleads readers into thinking the import is location-independent. | — |
| lc1-validator-inconsistency | design_defect | curricula/cp_accelerator/patterns/hash_table/problems/lc_1/validator.sh:27 | The lc_1 validator deviates from the system-wide convention in three ways: (1) it imports `twoSum` by name (`from solution import twoSum`) while every other CP accelerator validator uses `from solution import solve`; (2) the heredoc delimiter is unquoted `EOF` (shell variable expansion enabled) vs single-quoted `'EOF'` (no expansion) in all others; (3) it adds `cd "$SCRIPT_DIR"` before the heredoc (line 17) which no other validator does. Meanwhile, `lc_1/solution.py` provides no `solve = twoSum` alias (all other solution.py files end with `solve = <primary_function>`). These inconsistencies violate the uniform validator contract and would break any engine-level code that assumes `solve` is the standard entrypoint. `scripts/generate_module.py:387` confirms `from solution import solve` is the expected contract. | — |
| lc303-stateless-fn-for-class-problem | intent_mismatch | curricula/cp_accelerator/patterns/prefix_sum/problems/lc_303/solution.py:7 | LeetCode 303 (Range Sum Query - Immutable) requires students to implement a class `NumArray` with `__init__(self, nums)` and `sumRange(self, left, right)` because the key pedagogical point is that the prefix array is built once and reused. The solution instead exports a pure function `sumRange(nums, left, right)` with `solve = sumRange`. The validator calls `solve(**test["input"])` treating it as a stateless function. The Build-Justify-Harden intent is to teach students the O(1) query via memoized prefix array, but this implementation defeats that lesson by rebuilding on every call. The class-based interface is also what a student would need to submit to LeetCode. | — |
| lc203-linked-list-replaced-by-array | intent_mismatch | curricula/cp_accelerator/patterns/linked_list/problems/lc_203/solution.py:23 | Problem 203 is categorised under `linked_list` pattern and titled 'Remove Linked List Elements'. The pedagogical goal is to teach pointer manipulation (sentinel nodes, prev/curr traversal, `node.next` rewiring). The reference solution bypasses this entirely: `return [x for x in head if x != val]` — it receives a Python list and returns a filtered list comprehension. The in-file comment at line 4 admits this: 'Uses array representation for compatibility with test runner'. The test runner's inability to serialize/deserialize linked-list node graphs means the entire data-structure lesson is untestable. A student who submits this pattern to LeetCode (which requires a `ListNode`-based API) would receive a runtime error. | — |
| empty-stub-files-in-repo | design_defect | curricula/cp_accelerator/patterns/hash_table/problems/lc_1/solution_buggy.py:1 | Four files are committed as 0-byte placeholders that are described as code: (1) `hash_table/problems/lc_1/solution_buggy.py` (0 bytes — description: 'Empty placeholder for a generated buggy Two Sum solution; populated by bug-injection engine at exercise time'); (2) `sorting/problems/lc_912/bugs/incomplete_merge.py` (0 bytes); (3) `sorting/problems/lc_912/bugs/missing_base_case.py` (0 bytes); (4) `sorting/problems/lc_912/bugs/off_by_one.py` (0 bytes). These files are checked into source control as deliverable assets but contain no content. If the Harden stage validator or any engine code attempts to import or read these before the bug-injection engine populates them, it will fail silently or raise an ImportError. Dead/stub code presented as live. | — |
| cs336-validators-single-file-copy-assumption | design_defect | curricula/cs336_a1/modules/adamw/validator.sh:18 | All CS336 validators in the BUILD stage copy exactly one file into the shadow worktree: `cp cs336_basics/optimizer.py` (adamw), `cp cs336_basics/layers.py` (attention, embedding, linear, multihead_attention, rmsnorm, rope, silu, softmax, swiglu, transformer_block, transformer_lm), `cp cs336_basics/tokenizer.py` (bpe_tokenizer, tokenizer_class). If a student's implementation touches a second file (e.g., a helper function in `utils.py`, or a module that imports from another file they wrote), those changes are silently dropped. There is no validation that the copy succeeded, no manifest of which files belong to each module, and no warning if other modified files are ignored. The coupling between 'one module = one file' is baked into each validator independently with no shared policy. | — |
| http-transport-validator-live-network | design_defect | curricula/job_prep_data_annotation/modules/http_transport/validator.sh:26 | The validator for the http_transport module makes live HTTP requests to `https://httpbin.org/html`, `https://httpbin.org/status/404`, and `https://httpbin.org/status/500` during test execution. Validation can fail non-deterministically due to network outages, `httpbin.org` downtime, rate limiting, or DNS failures completely unrelated to the student's implementation. This creates a flaky validation boundary. | — |
| softmax-special-case-duplicated | design_defect | engine/main.py:486 | Hardcoded `if current_module.id == "softmax":` special-case logic appears at both line 486-491 and line 1799-1804. Curriculum content (module IDs) has leaked into the engine core. The condition exists in two separate code paths handling harden setup, so any future curriculum changes require updating two locations. (Locations confirmed from prior session read.) | — |
| duplicate-build-validation-logic | design_defect | engine/main.py:555 | `_submit_linear_workflow()` (lines 555-586) and `submit_build()` (lines 1354-1525) contain substantially duplicated build-validation logic. Both orchestrate the same sequence: copy file to shadow worktree, run validator, parse results, advance state. This duplication means bugs in one path often do not exist in the other, creating inconsistent behaviour between the two entry points. (Locations confirmed from prior session read.) | — |
| curriculum-manager-stale-cache | design_defect | engine/curriculum.py:1 | `_pattern_cache` and `_problem_cache` are instance-level dicts on `CurriculumManager`. They are populated on demand but never cleared when `load_manifest()` is called again with a different `curriculum_id`. If a long-running process switches curricula, cached patterns and problems from the previous curriculum remain in the cache, causing incorrect lookups. (Exact initialization line unverified in this session; the cache accumulation behaviour was confirmed from direct read of engine/curriculum.py.) | — |
| harden-hardcoded-relative-worktree-path | design_defect | engine/stages/harden.py:78 | Both `present_challenge()` (line 78) and `present_library_challenge()` (line 247) hardcode `shadow_worktree = Path('.mastery_engine_worktree')`. This relative path resolves against CWD at runtime. `engine/utils.py` exports `find_project_root()` specifically to solve this problem, but it is not used here. If the CLI is invoked from a subdirectory, the worktree lookup silently fails and raises a misleading `HardenChallengeError: 'Shadow worktree not found'`. | — |
| harden-select-bug-includes-draft-files | design_defect | engine/stages/harden.py:196 | `_select_bug()` collects all `.json` files via `list(bugs_dir.glob('*.json'))`. No naming convention filter is applied, so any draft or work-in-progress JSON files in the bugs directory (e.g., `off_by_one_draft.json`, `_draft_v2.json`) are eligible for selection. `random.choice()` across this unfiltered pool can present malformed or incomplete bug definitions to students. | — |
| justify-docstring-claims-stub-but-llm-is-live | doc_code_drift | engine/stages/justify.py:7 | Module docstring (lines 7-16) reads: 'This is currently a STUB implementation. The real implementation will integrate LLM-powered evaluation... For now, it accepts any non-empty answer.' and the class docstring says 'stub: always accept'. However, `LLMService.evaluate_justification()` is live and wired in `engine/main.py`. The docstring describes superseded stub behaviour and will mislead future maintainers about the production state of the Justify stage. | — |
| justify-fast-filter-false-positive-risk | design_defect | engine/stages/justify.py:107 | `check_fast_filter()` at lines 107-118 rejects an answer as soon as ANY failure-mode keyword appears anywhere in the answer text via `if keyword.lower() in user_answer_lower`. A correct, nuanced answer that incidentally contains a flagged word (e.g., a student writing 'the naive approach would cache everything, but I avoid that by...' where 'cache' is a failure keyword) will be rejected without LLM evaluation. The filter has no word-boundary check and no context awareness. | — |
| softmax-poc-ships-with-print-statements | design_defect | engine/ast_harden/softmax_poc.py:106 | `softmax_poc.py` is a proof-of-concept file with bare `print()` statements at lines 106, 114, 137, 140, 142-143 and an `if __name__ == '__main__':` test harness at line 209. It duplicates class hierarchies present in `ast_service.py` and `softmax_v2_1.py`. This PoC file ships in the production package; its print statements will appear in any context that imports or executes it. | — |
| softmax-v2-1-dead-intermediate-ships-with-prints | design_defect | engine/ast_harden/softmax_v2_1.py:143 | `softmax_v2_1.py` is an intermediate-iteration file with bare `print()` debug statements scattered throughout (lines 143, 168-170, 183-184, 217-224) and an `if __name__ == '__main__':` harness at line 310. It defines `SoftmaxCanonicalizer`, `CanonicalPatternMatcher`, and `OriginalASTTransformer` — nearly identical to those in `ast_service.py`. Neither file is called by production code; both accumulate as dead weight creating three duplicate class hierarchies. | — |
| apply-patch-hidden-external-dependency | design_defect | engine/workspace.py:1 | `WorkspaceManager.apply_patch()` shells out to the system `patch` binary without first verifying it is available in `PATH`. On environments without `patch` installed (common in minimal Docker images or Windows), the call fails with a confusing `FileNotFoundError` or shell error rather than a clear dependency error. No `which patch` / `shutil.which('patch')` guard is present. (Exact line unverified in this session; behaviour confirmed from prior read of workspace.py.) | — |
| llm-mock-auto-pass-silently-defeats-pedagogy | intent_mismatch | engine/services/llm_service.py:109 | When `OPENAI_API_KEY` is absent, `LLMService` enters mock mode (line 60-71) and `evaluate_justification()` always returns `is_correct=True` with a boilerplate message (lines 109-122). The engine's stated purpose is to evaluate student understanding; auto-passing every justify question when the key is missing silently removes the core pedagogical gate. Users without a key configured receive no friction and no indication their understanding was not evaluated. | — |
| tokenizer-developer-reference-delegates-to-tiktoken | intent_mismatch | modes/developer/cs336_basics/tokenizer.py:38 | `modes/developer/cs336_basics/tokenizer.py` at line 38 does `self._enc = tiktoken.get_encoding('gpt2')` and delegates all encode/decode to tiktoken, with comment: 'We rely on the canonical GPT-2 encoding for correctness against tiktoken snapshots.' The student is required to implement BPE from scratch (per `modes/student/cs336_basics/bpe.py:34: raise NotImplementedError`). For patch-based harden bugs, `present_challenge()` copies `modes/developer/cs336_basics/tokenizer.py` as the bug-injection base — so bugs are injected into tiktoken-wrapping code, not the student's BPE implementation. This defeats the harden stage for the tokenizer module when patch-based bugs are used. | — |
| conftest-force-update-dead-code | design_defect | tests/conftest.py:60 | NumpySnapshot.assert_match (line 44) accepts force_update: bool = False. Lines 60-61 store the parameter but no code path ever writes a new snapshot file. If force_update=True is passed and the snapshot file does not exist, execution falls through to the load path which raises FileNotFoundError. Snapshot.assert_match (lines 116-150) has the identical defect at line 131. There is no mechanism to create initial snapshots from a test run; the feature is advertised by the parameter signature but entirely absent from the implementation. | — |
| adapters-missing-bpe-student-stub | design_defect | tests/adapters.py:21 | Line 21: 'from cs336_basics.bpe import train_bpe as _train_bpe_impl'. The modes/student/cs336_basics/ directory contains layers.py, optimizer.py, tokenizer.py, tokenizer_stub.py, utils.py, and pretokenization_example.py — but no bpe.py. When the test suite runs against student stubs, this import raises ModuleNotFoundError. The adapter is the test isolation boundary and is explicitly designed to expose all student implementations, but it references a module that was never added to the stub set. | — |
| validate-stubs-function-check-is-pass | design_defect | scripts/validate_student_stubs.py:55 | visit_FunctionDef (lines 54-57) reads: '# check for TODO in function' followed by 'pass'. The per-function TODO/NotImplementedError body inspection is never performed. Only the file-level TODO string scan (lines 88-93) runs. Additionally, line 128 explicitly excludes any file whose name contains 'example' ('"example" not in f.name.lower()'), which permanently exempts pretokenization_example.py from all stub validation, including the module-level executable code that crashes on import. | — |
| bjh-loop-test-silently-bypasses-justify-llm-path | intent_mismatch | tests/e2e/test_complete_bjh_loop.py:349 | The test docstring (lines 6-18) explicitly claims: 'Justify: Test both fast filter and LLM evaluation paths.' Lines 349-354 instead directly write {"current_stage": "harden"} to the state file, bypassing justify entirely. A comment at line 344 says 'Without API key, this will fail with ConfigurationError'. The deep-answer LLM path that the docstring promises is never executed. The test is presented as a 'regression fortress' protecting the full loop, but the LLM justify half is always silently skipped. | — |
| fetch-sources-taxonomy-is-single-line-placeholder | intent_mismatch | scripts/fetch_sources.sh:52 | Lines 52-53: 'echo "# DSA Taxonomies" > "$SOURCES_DIR/cp_accelerator/dsa_taxonomies"'. The script is named fetch_sources.sh and its role is to retrieve real data for curriculum generation. The CS336 A1 section fetches actual content (PyPI packages, GitHub files), but the CP accelerator taxonomy output is a single comment line. Downstream tools (parse_sources.py, generate_module.py) that consume dsa_taxonomies will receive a one-line stub rather than actual taxonomy data. | — |
| llm-service-mock-vs-error-mode-contradiction | intent_mismatch | tests/engine/test_llm_service.py:55 | tests/engine/test_llm_service.py lines 55-60 (test_init_missing_api_key_enables_mock_mode) asserts: LLMService() with no OPENAI_API_KEY sets service.use_mock=True, service.client=None — graceful degradation. But tests/integration/test_llm_service.py lines 71-79 (test_llm_service_missing_api_key) asserts: the same LLMService() call with no key raises ConfigurationError. Both are in the active test suite asserting contradictory runtime behavior. One of these contracts must misrepresent the actual implementation. | — |
| dead-assert-inside-pytest-raises | design_defect | tests/test_data.py:72 | Line 72 (`assert "CUDA error" in str(excinfo.value) or "Torch not compiled with CUDA enabled" in str(excinfo.value)`) is inside the `with pytest.raises((RuntimeError, AssertionError)) as excinfo:` block opened at line 62. When `run_get_batch` raises (the expected path, lines 66-71), Python exits the `with` block at the point of the exception, never reaching line 72. If `run_get_batch` does NOT raise, `pytest.raises` itself raises `Failed: DID NOT RAISE` before line 72 executes. The assertion is unreachable dead code in every execution path; it should be placed outside the `with` block. | — |
| decoder-lm-uses-encoder-api | intent_mismatch | tests/one_d_probes.py:68-83 | Class is named `DecoderOnlyLM` (line 68) but its body instantiates `nn.TransformerEncoderLayer` (line 74) and `nn.TransformerEncoder` (line 83) — PyTorch's encoder-stack implementation. Causal masking is applied via `_mask()` (line 86-89) to emulate autoregressive behaviour, but the underlying module type is an encoder. The class name promises a decoder-only LM architecture, yet the implementation delegates to the encoder API, creating a structural intent mismatch for any reader trying to understand or reuse the probe. | — |
| bpe-exact-merge-assertion-suppressed | design_defect | tests/test_train_bpe.py:54 | Line 54: `# assert merges == reference_merges  # Too strict - commented out`. The original exact-match assertion against the reference BPE merges has been disabled. Its replacement (lines 57-58) only checks `len(merges) >= 243` and `len(merges) <= 245` — a range that allows a 2-element deviation in count without verifying any merge content. A BPE implementation that produces 244 entirely wrong merges would pass. The justification comment (tie-breaking at index 64) does not justify abandoning merge-content validation entirely. | — |
| tiktoken-ids-equality-check-suppressed | intent_mismatch | tests/test_tokenizer.py:184 | In `test_ascii_string_matches_tiktoken`, line 184 reads `# assert ids == reference_ids`. The test's declared intent (confirmed by its name and lines 188-189 which still assert roundtrip equality) is to verify that the student tokenizer produces the same token IDs as tiktoken. However the actual ID-equality check is commented out. The remaining assertions verify only per-token string decoding and roundtrip fidelity, not ID equivalence. An implementation that produces different IDs but happens to decode identically would silently pass a test that claims to verify tiktoken parity. | — |
| quality-audit-insert-before-check-stale | doc_code_drift | audits/QUALITY_AUDIT.md:49 | QUALITY_AUDIT.md:49 states 'Bug definition uses `replacement.type: "move_after"`, which is not implemented by the AST injector, so injection behavior will be incorrect.' However, the actual file `curricula/cp_accelerator/patterns/hash_table/problems/lc_1/bugs/insert_before_check.json` was rewritten: it now uses `find_and_replace` with `replacement.type: "replace_with"` (a supported operation), replacing `complement in seen` with `num in seen`. The metadata note confirms: 'Original bug used unsupported move_after type. Redesigned to use replace_value_with'. The audit finding is now factually wrong — the bug and symptom file are correctly aligned. | — |
| readme-cs336a1-module-count | doc_code_drift | README.md:87 | README.md:87 states '**Modules**: 21 modules (BPE Tokenizer → Full Training Loop)'. The actual `curricula/cs336_a1/manifest.json` contains 22 modules in its `modules` array: ['softmax', 'cross_entropy', 'gradient_clipping', 'linear', 'embedding', 'silu', 'rmsnorm', 'swiglu', 'attention', 'rope', 'multihead_attention', 'transformer_block', 'transformer_lm', 'adamw', 'cosine_schedule', 'data_loader', 'checkpointing', 'training_loop', 'unicode', 'bpe_tokenizer', 'tokenizer_class', 'text_generation']. The manifest's own `description` field also says '21 modules in dependency order' (manifest.json:5) — both the README and the manifest description are off by one, with `unicode` or `text_generation` likely added after the count was written. | — |
| manifest-cs336a1-description-count | doc_code_drift | curricula/cs336_a1/manifest.json:5 | The manifest.json `description` field at line 5 reads 'Complete from-scratch Transformer LM implementation with 21 modules in dependency order'. The `modules` array directly below has 22 entries (confirmed by `python3 -c "import json; print(len(json.load(open('curricula/cs336_a1/manifest.json'))['modules']))"`). The manifest contradicts itself: description claims 21, list has 22. | — |
| build-prompt-dict-literal-resources | doc_code_drift | curricula/cp_accelerator/patterns/backtracking/problems/lc_78/build_prompt.txt:74 | The Learning Resources section in all auto-generated build prompts renders resource entries as raw Python dict literals instead of formatted markdown links. Example from lc_78/build_prompt.txt:74: `1. {'type': 'taxonomy', 'url': 'https://github.com/Yassir-aykhlf/DSA-Taxonomies/blob/main/Taxonomies/11. Backtracking.md', 'title': 'Backtracking Taxonomy'}`. This formatting defect appears in at minimum: lc_78, lc_90, lc_704, lc_1342, lc_1486, lc_47, lc_146, lc_460, lc_148 (divide_and_conquer), lc_912 (divide_and_conquer), lc_198, lc_70, lc_435, lc_452, lc_217, lc_219. The content-generation pipeline (scripts/generate_module.py) failed to render resource dicts as markdown hyperlinks. Users see raw Python object syntax instead of clickable links. | — |
| impl-status-vs-status-topic-count-and-ids | doc_code_drift | curricula/cp_accelerator/IMPLEMENTATION_STATUS.md:95 | IMPLEMENTATION_STATUS.md:95 claims '✅ 11 topics (foundation complete)' and shows a dependency graph with IDs like `two_pointers_basics`, `two_pointers_sliding_window`, `binary_search_on_index`, `binary_search_on_answer`, `dp_foundations`, `dp_knapsack`, `graphs_basics`. STATUS.md:5 says 'All 19 DSA Taxonomy patterns parsed' with the actual IDs used in the manifest/curriculum: `two_pointers`, `linked_list`, `hash_table`, `stack_queue`, `binary_search`, `traversal`, `dynamic_programming`, `heap`, `greedy`, `backtracking`, `divide_conquer`, `union_find`, `design`, `trie`, `bit_manipulation`, `segment_tree`, `combinatorics`. The IDs in IMPLEMENTATION_STATUS do not match actual deployed IDs and the topic count conflicts with the deployed 19-pattern curriculum. IMPLEMENTATION_STATUS represents an abandoned design that was superseded. | — |
| lc46-lc47-pattern-classification-mismatch | intent_mismatch | curricula/cp_accelerator/patterns/combinatorics_and_number_theory/problems/lc_46/build_prompt.txt:3 | Permutations (lc_46) and Permutations II (lc_47) are placed under the `combinatorics_and_number_theory` pattern directory, but their LeetCode classification is 'Topics: Array, Backtracking' (lc_46/build_prompt.txt:40, lc_47/build_prompt.txt:38). The pattern overview in both files reads 'Combinatorics and Number Theory provide mathematical tools for counting, arrangement, and number properties...' — this is a generic description that doesn't match the backtracking algorithm learners must implement. The intent of the cp_accelerator curriculum is to teach algorithmic PATTERNS correctly, but these problems teach backtracking recursion, not combinatorics math. They would be better placed under the `backtracking` pattern (which already exists). | — |
| cp-build-prompts-raw-python-dict-resources | doc_code_drift | curricula/cp_accelerator/patterns/heap_and_priority_queue/problems/lc_215/build_prompt.txt:71 | All cp_accelerator build_prompt.txt files in the required set render Learning Resources as raw Python dict literals instead of formatted markdown. Example from lc_215: "1. {'type': 'taxonomy', 'url': 'https://github.com/Yassir-aykhlf/DSA-Taxonomies/...', 'title': 'Heap and Priority Queue Taxonomy'}". The same pattern appears in: lc_703:89, lc_203:82, lc_237:90, lc_1480:90-92, lc_303:81-83, lc_307:85-89, lc_148:83-85, lc_912:75-77, lc_1003:92, lc_20:96, lc_144:78-82, lc_589:76-80, lc_208:82, lc_167:90-92, lc_547:79-81, lc_684:79-81. The content-generation pipeline emitted serialized dict objects rather than formatted bullets, contradicting the learner-facing doc role of these files. | — |
| cs336-build-prompts-legacy-submit-command | doc_code_drift | curricula/cs336_a1/modules/adamw/build_prompt.txt:430 | The AdamW build_prompt.txt instructs students: 'engine submit-build' (line 430). The same legacy command appears in: attention/build_prompt.txt:366, bpe_tokenizer/build_prompt.txt:338, checkpointing/build_prompt.txt:480. However, the cs336_a1/README.md at line 61 documents the current CLI command as 'uv run mastery submit', and the engine architecture (01-understanding.json provisionalIntent) identifies 'submit-build/justification/fix' as LEGACY commands superseded by 'submit'. Students following the build prompts will run a deprecated command that may not exist or behave differently. | — |
| adamw-epsilon-placement-mismatch | doc_code_drift | curricula/cs336_a1/modules/adamw/build_prompt.txt:93 | The mathematical specification on line 93 shows the AdamW update as: 'θ_{t+1} = θ_t - η (m̂_t / √(v̂_t + ε) + λ θ_t)' — epsilon is INSIDE the square root. However, the implementation pseudocode in Step 9 (line 327-329) shows: 'denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)' — epsilon is added OUTSIDE the square root. These formulas are numerically distinct: √(v̂_t + ε) ≠ √(v̂_t) + ε. For v̂_t → 0 and ε = 1e-8, the first gives 1/√ε ≈ 10000 while the second gives 1/ε = 1e8. The pseudocode implements the standard PyTorch convention (eps outside), but this contradicts the mathematical notation shown to students. | — |
| bpe-return-type-signature-vs-implementation | doc_code_drift | curricula/cs336_a1/modules/bpe_tokenizer/build_prompt.txt:137 | The function signature at line 136-151 declares return type 'tuple[dict[int, str], list[tuple[str, str]]]' — merges are a list of string pairs. However, the sample implementation in Step 3 (lines 199-218) builds merges as: 'merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))' where vocab values are bytes objects (initialized at line 185 as 'vocab = {i: bytes([i]) for i in range(256)}'). The merges list therefore contains tuple[bytes, bytes] pairs, not tuple[str, str]. The string conversion 'vocab_str' at line 224 only converts the vocabulary dict, not the merges list. The declared return type does not match what the provided pseudocode produces. | — |
| embedding-double-embedding-attribute-in-visualization | doc_code_drift | curricula/cs336_a1/modules/embedding/build_prompt.txt:336 | build_prompt.txt line 336 shows visualization code: 'E = model.embedding.embedding.weight  # (vocab_size, d_model)'. The Embedding class (modes/developer/cs336_basics/layers.py:51) stores its weight matrix as 'self.weight = nn.Parameter(torch.empty(...))' with no sub-attribute named 'embedding'. Accessing 'model.embedding.embedding' would raise AttributeError at runtime. The correct access path is 'model.embedding.weight'. The build_prompt misleads students into a double-attribute access that cannot work. | — |
| training-loop-wrong-cosine-function-name | doc_code_drift | curricula/cs336_a1/modules/training_loop/build_prompt.txt:55 | training_loop/build_prompt.txt uses 'cosine_schedule(step, ...)' at line 55, 'cosine_schedule(step, max_lr, min_lr, warmup_steps, max_steps)' at lines 175 and 418 in implementation examples and pseudo-code. The actual function available to students is 'get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)' in 'cs336_basics/utils.py' (modes/student/cs336_basics/utils.py:65). A student implementing the training loop while following these code examples verbatim would get NameError/TypeError because neither the function name 'cosine_schedule' nor the parameter names ('step', 'max_lr', 'min_lr', 'warmup_steps', 'max_steps') match the actual API. | — |
| job-prep-readme-wrong-file-for-data-parsing | doc_code_drift | curricula/job_prep_data_annotation/README.md:83 | job_prep_data_annotation/README.md line 83 says '# 2. Implement in cs336_basics/utils.py' as the workflow instruction. The data_parsing_extraction/build_prompt.txt:22 contradicts this by saying 'Implement the following function in job_prep/parser.py'. The data_parsing_extraction/validator.sh:17 imports 'from cs336_basics.utils import extract_coordinates', confirming the README and validator agree while the build_prompt has the wrong file. This three-way inconsistency leaves the student with contradictory instructions between the README and the module's own build prompt. | — |
| mastery-engine-doc-solutions-dir | doc_code_drift | docs/architecture/MASTERY_ENGINE.md:139 | MASTERY_ENGINE.md at lines 122-149 describes a `.solutions/` private directory structure holding reference implementations. This directory does not appear in the actual file inventory (audit/.work/01-understanding.json). The actual mode-switching system uses `modes/student/cs336_basics` and `modes/developer/cs336_basics` symlink targets, confirmed by LAYER2_E2E_SUCCESS.md:48 (`shutil.copytree(real_repo / 'modes', test_repo / 'modes')`) and REAL_STUDENT_UAT_MODULE1.md:13 (`./scripts/mode switch student`). | — |
| mastery-engine-doc-wrong-workflow-name | doc_code_drift | docs/architecture/MASTERY_ENGINE.md:1019 | MASTERY_ENGINE.md references a CI workflow file named `validate_curriculum.yml` at lines 1019 and 1027. The actual GitHub Actions workflow file is `.github/workflows/validate_cp_manifest.yml` per the 01-understanding.json inventory (listed under CI workflows). The filename in the blueprint does not match the actual file on disk. | — |
| std-lib-augmentation-status-conflict | doc_code_drift | docs/internal/PYTHON_CURRICULA_IMPLEMENTATION.md:165 | PYTHON_CURRICULA_IMPLEMENTATION.md:165 shows `std_lib_augmentation` module status as 'PLANNED ⏸️'. CRITICAL_REVIEW_RESPONSE.md:149 shows the same module listed as 'COMPLETE ✅'. These two internal documentation files directly contradict each other on the implementation status of the same module, making the true state unverifiable from documentation alone without reading the actual curriculum directory. | — |
| session-docs-engine-module-vs-mastery-cli | doc_code_drift | docs/internal/archive/sessions/2025-11-09_verification/FINAL_VERIFICATION_SUMMARY.md:186 | FINAL_VERIFICATION_SUMMARY.md:186-196 documents all 9 CLI commands using `engine` as the command name (e.g., `engine init <curriculum_id>`, `engine submit`, `engine show [module_id]`). All UAT session docs (LAYER4_UAT_FINDINGS.md:62, LAYER4_UAT_EXECUTION_GUIDE.md:55, REAL_STUDENT_UAT_MODULE1.md:20) consistently invoke the CLI as `uv run python -m engine.main ...`. The actual registered CLI entry point (per pyproject.toml [project.scripts]) is `mastery`. These verification documents describe superseded invocation patterns that differ from the shipped command name. | — |
| justify-only-manifest-doc-contradicts-itself | doc_code_drift | docs/internal/archive/deprecated/JUSTIFY_ONLY_MODULE_DESIGN.md:229 | JUSTIFY_ONLY_MODULE_DESIGN.md 'Completed' section (lines 229-235) marks as done: '✅ Manifest updated with module_type: justify_only field' and '✅ Unicode module created.' The 'Pending Engine Implementation' section immediately following (lines 237-242) contradicts this by listing as NOT done: 'Schema updates to support module_type field ⏸️', 'State management updates for justify-only progression ⏸️', 'Command validation (error on build/harden for justify-only) ⏸️'. The document's own completion status section is internally inconsistent: the data field is in the manifest but the engine code to interpret it is explicitly marked pending. | — |
| intent-module-type-field-ignored-by-engine | intent_mismatch | docs/internal/archive/deprecated/JUSTIFY_ONLY_MODULE_DESIGN.md:237 | The design intent (JUSTIFY_ONLY_MODULE_DESIGN.md §Implementation Requirements, lines 63-144) is that `module_type: 'justify_only'` in a manifest should cause the engine to skip build/harden stages, start directly at justify, and error on `mastery build`. The unicode module manifest was updated with `module_type: 'justify_only'` (marked COMPLETED at line 232). However, engine/schemas.py, engine/state.py, engine/curriculum.py, and engine/main.py were explicitly NOT updated (all marked PENDING at lines 237-242). The manifest field expresses design intent that the engine silently ignores at runtime — the unicode module would be treated as a standard module requiring build stage. | — |
| harden-fix-verify-primary-path-misleading | doc_code_drift | docs/internal/archive/sessions/2025-11-10_bug_system/HARDEN_FIX_VERIFICATION.md:25 | HARDEN_FIX_VERIFICATION.md lines 25-29 presents 'After (Working): Copy developer's code to harden workspace' as THE primary fix for harden bug injection. In the current engine/stages/harden.py (lines 97-125), the primary path for .json bug files injects into the STUDENT'S own code via GenericBugInjector; copying developer code (lines 127-157) is only the legacy else-branch for .patch files. The document inverts primary/secondary status of these two paths. | — |
| harden-class-docstring-patch-only | doc_code_drift | engine/stages/harden.py:33 | HardenRunner class docstring at harden.py lines 31-36 describes the Harden stage as '1. Copying their validated Build submission / 2. Applying a pedagogical bug patch'. The .json dispatch path (harden.py:97-125) uses AST-based bug injection via GenericBugInjector with no patch applied at all. The docstring describes only the legacy .patch workflow and omits the primary AST injection path. | — |
| harden-present-challenge-param-misleading | doc_code_drift | engine/stages/harden.py:63 | present_challenge() docstring at harden.py:63 states 'source_file_path: Path to the user's (main workspace) source file used as the target for hardening.' For .patch-based bugs (harden.py:127-157), the function ignores the student's file entirely and copies from the developer reference implementation at modes/developer/<rel_path>. The parameter description is only accurate for the .json AST path. | — |
| manual-llm-test-deprecated-submit-cmd | doc_code_drift | docs/internal/archive/sessions/2025-11-10_bug_system/MANUAL_LLM_TEST.md:68 | MANUAL_LLM_TEST.md line 68 invokes 'submit-justification' as the CLI command for submitting a justify-stage answer. VERIFICATION_PROTOCOL_LAYER2_STATUS.md (2025-11-09_verification session) line 58 documents the migration: 'submit-build -> submit (unified command)' as part of Layer 2 fixes. The correct command per the updated CLI is the unified 'submit', making the MANUAL_LLM_TEST.md example use a removed command. | — |
| bpe-fix-student-pass-contradicted | doc_code_drift | docs/internal/archive/sessions/2025-11-11_curriculum_quality/BPE_TEST_FIX_SUMMARY.md:87 | BPE_TEST_FIX_SUMMARY.md lines 86-89 claims 'Student (stub): Before=FAIL, After=PASS — FIXED', verifying the student bpe test now passes after relaxing assertions. CRITICAL_BUG_RESOLUTION.md (same Nov 13, 2025 date, 2025-11-10_bug_system session) documents that modes/student/cs336_basics/bpe.py was subsequently stubbed out (200+ implementation lines removed, replaced with NotImplementedError stubs). A stubbed-out bpe.py cannot pass the test_train_bpe test. These two session documents, written on the same date, assert contradictory states for the student BPE test. | — |
| harden-patch-path-contradicts-debug-own-code-intent | intent_mismatch | engine/stages/harden.py:127 | Provisional intent (audit/.work/01-understanding.json) states the Harden stage 'challenges users to debug their own implementations'. HardenRunner class docstring (harden.py:31) likewise states '1. Copying their validated Build submission'. For .patch-based bugs, harden.py:127-157 ignores the student's code entirely; it copies the developer reference implementation from modes/developer/<rel_path> and applies a patch to that. The student debugs code they did not write. HARDEN_STAGE_CRITICAL_BUG.md documents this as an architectural decision for patch compatibility, but it directly contradicts the stated pedagogical intent of debugging one's own code. | — |
| bpe-line-count-141-vs-350 | doc_code_drift | docs/internal/archive/sessions/2025-11-11_curriculum_quality/REMEDIATION_PROGRESS.md:29 | REMEDIATION_PROGRESS.md:29 says 'Implementation: `modes/developer/cs336_basics/bpe.py` (~141 lines)'. MASTER_REMEDIATION_STATUS.md:29 says 'From-scratch BPE training (~350 lines)'. A 2.5x discrepancy (141 vs 350 lines) for the same artifact. REMEDIATION_SUMMARY.md describes the BPE as using 'heap-based priority queue, doubly-linked list' which would imply a larger codebase. The reported size of the delivered artifact is inconsistent across three documents. | — |
| tokenizer-path-cs336basics-vs-modes-developer | doc_code_drift | docs/internal/archive/sessions/2025-11-11_curriculum_quality/REMEDIATION_PROGRESS.md:64 | REMEDIATION_PROGRESS.md:64 says the new from-scratch Tokenizer was created at `cs336_basics/tokenizer.py`. QUALITY_REMEDIATION_PLAN.md:35 specifies the remediation target as `modes/developer/cs336_basics/tokenizer.py`. TOKENIZER_VIOLATIONS_AUDIT.md also refers to the violation at `modes/developer/cs336_basics/tokenizer.py`. The created file and the required target path differ: one strips the `modes/developer/` prefix, determining whether the file is the reference implementation (correct location) or an ambiguous standalone file (incorrect location). | — |
| cli-audit-next-docstring-false-claim | doc_code_drift | docs/internal/archive/sessions/2025-11-12_cli_remediation/CLI_INTERFACE_AUDIT.md:96 | CLI_INTERFACE_AUDIT.md:96 quotes the `next` command docstring verbatim: 'Only works when the user is in the "build" stage.' CLI_INTERFACE_AUDIT.md:120 in the same document says 'Misleading Documentation: Docstring says "only works when in build stage" - FALSE, it works for all 3 stages (build, justify, harden)'. The audit document identifies the false docstring but lists it only as a 'Weakness' without assigning a remediation priority (P0/P1/P2), so the false docstring was not scheduled for correction. | — |
| p0-progress-line-count-329-vs-410-vs-470 | doc_code_drift | docs/internal/archive/sessions/2025-11-12_cli_remediation/CLI_P0_PROGRESS.md:204 | Three documents report different line counts for the same P0 implementation: (1) CLI_P0_PROGRESS.md:204 Files Modified table says '+329 lines'; (2) CLI_P0_PROGRESS.md:215 Code Statistics says 'Total New Code: ~410 lines' — internal contradiction within the same document; (3) CLI_P0_FINAL_STATUS.md reports '~470 lines of production-quality implementation'. The true size of the delivered artifact cannot be determined from documentation alone. | — |
| cp-quickstart-self-ref-wrong-path | doc_code_drift | docs/internal/archive/sessions/2025-11-11_curriculum_quality/CP_ACCELERATOR_QUICKSTART.md:67 | CP_ACCELERATOR_QUICKSTART.md:67-68 states 'docs/CP_ACCELERATOR_IMPLEMENTATION_GUIDE.md - Full technical blueprint' and 'docs/CP_ACCELERATOR_QUICKSTART.md - This file'. Both self-referenced paths are wrong — both files are archived at `docs/internal/archive/sessions/2025-11-11_curriculum_quality/`, not top-level `docs/`. The files were archived but their internal cross-references still point to the original intended (never-created) top-level locations. | — |
| DCD-003 | doc_code_drift | docs/internal/current/CURRICULUM_STATUS.md:14 | Table row for cs336_a1 states 'Modules: 21'. Direct count of curricula/cs336_a1/manifest.json via bash yields 22 modules. The off-by-one means module counts in the authoritative status document undercount the actual deployed curriculum by one module. | — |
| DCD-005 | doc_code_drift | docs/internal/current/BUG_INJECTION_GUIDE.md:210 | Schema Evolution section documents the command: 'engine regenerate-bugs --all' (line 210: '2. Regenerate all .json files: `engine regenerate-bugs --all`'). Grep of engine/main.py for 'regenerate-bugs' returns zero matches. The command does not exist anywhere in the registered Typer application (engine/main.py). Additionally, the documented CLI prefix 'engine' is itself wrong — the console script is registered as 'mastery' in pyproject.toml, not 'engine'. | — |
| DCD-006 | intent_mismatch | docs/internal/development/CHANGELOG.md:1 | File is placed under docs/internal/development/ where Mastery Engine development history is expected. Entire content is the Stanford CS336 Spring 2025 Assignment 1 changelog (versions 0.1.0 2024-04-01 through 1.0.6 2025-08-28), covering handout and test-suite changes for that course. Zero Mastery Engine entries exist in the file. Per provisional intent in audit/.work/01-understanding.json, Mastery Engine is a 'curriculum-agnostic pedagogical operating system CLI' that is a separate project from CS336; the changelog conflates curriculum source material with engine development history. | — |
| typer-app-named-engine-not-mastery | design_defect | engine/main.py:57 | The Typer app is instantiated with `name="engine"` at line 57: `app = typer.Typer(name="engine", help="Mastery Engine: Build, Justify, Harden learning system", ...)`. Because of this, `uv run mastery --help` displays `Usage: engine [OPTIONS] COMMAND [ARGS]...` — the usage line shows `engine` while the user must type `mastery`. This creates a persistent, built-in confound between the installed CLI name and what the help output says. The intended CLI name is `mastery` (pyproject.toml:34). | — |
| next-cmd-deprecation-msg-wrong-cli-name | doc_code_drift | engine/main.py:1330 | The docstring and Rich panel for the deprecated `next()` command (lines 1330-1342) instruct users to use `engine show` and `engine start-challenge`: line 1330 says `'engine show' for read-only viewing`, line 1340 prints `engine show`, line 1341 prints `engine start-challenge`, line 1342 prints `Running 'engine show' for you...`. The actual installed CLI is `mastery`, so users see "use engine show" but the correct command is `mastery show` / `mastery start-challenge`. The MASTERY_COMMAND_REFERENCE.md confirms the installed name is `mastery`. | — |
| project-structure-nonexistent-reference-dir | doc_code_drift | maintenance/PROJECT_STRUCTURE.md:64 | PROJECT_STRUCTURE.md shows the directory tree (line 64): `└── reference/            # Complete implementations (archived)` with `└── utils_complete.py` under `curricula/cs336_a1/`. Verified with `ls /home/user/mastery-engine/curricula/cs336_a1/reference/` which returns `No such file or directory`. This directory and file do not exist on disk. | — |
| mvp-status-mainpy-line-count-wrong | doc_code_drift | docs/internal/development/MVP_COMPLETION_STATUS.md:82 | MVP_COMPLETION_STATUS.md states under Gap #3 (line 82): "`engine/main.py` has 1,241 lines with embedded orchestration logic". Actual line count via `wc -l engine/main.py` is 2,942 lines — 2.4× larger than documented. This understates the extent of the fat-controller problem by more than half and misrepresents the scale of refactoring required. | — |
| modes-readme-21-modules-vs-22-in-manifest | doc_code_drift | modes/README.md:9 | modes/README.md line 9 states "Complete Curriculum: 21 Modules" and enumerates 8+5+5+3=21 modules across four categories without listing `unicode`. The curricula/cs336_a1/manifest.json contains 22 modules (confirmed via Python: `len(d['modules']) == 22`), including a `unicode` module that is absent from the README's list and count. The manifest directory also contains 22 subdirectories including `unicode/`. | — |
| two-sum-e2e-wrong-validator-path | doc_code_drift | docs/internal/two_sum_qa/TWO_SUM_E2E_WORKFLOW_TEST.md:43 | TWO_SUM_E2E_WORKFLOW_TEST.md line 43 shows the build-stage validator command as `cd curricula/cp_accelerator/modules/two_sum && ./validator.sh`. The directory `curricula/cp_accelerator/modules/` does not exist (verified with `ls`). The Two Sum (LC-1) module is actually located at `curricula/cp_accelerator/patterns/hash_table/problems/lc_1/`. The path drift makes this test report unreproducible from the documented command. | — |
| two-sum-comparison-analysis-wrong-module-path | doc_code_drift | docs/internal/two_sum_qa/MODULE_COMPARISON_ANALYSIS.md:47 | MODULE_COMPARISON_ANALYSIS.md (line 47 onward) shows the Two Sum module directory tree rooted at `two_sum/` inside `cp_accelerator`, implying the path `curricula/cp_accelerator/modules/two_sum/` or similar. The actual production path is `curricula/cp_accelerator/patterns/hash_table/problems/lc_1/`. The module is accessed by the engine under the `lc_1` problem ID, not `two_sum`. Both the path and identifier in this analysis document are inconsistent with the actual on-disk structure. | — |
| readme-wrong-total-cost | doc_code_drift | tests/integration/README.md:9 | README line 9 states '**Cost**: ~$0.006 per full test run'. The actual file `tests/integration/test_llm_service.py` line 7 states 'Cost: ~$0.009 per full test run (3 API calls x $0.003 each)'. The discrepancy is because the actual file has 3 live-API tests (`test_llm_accepts_correct_answer`, `test_llm_rejects_incomplete_answer`, `test_llm_rejects_conceptual_error`) versus the 2 the README cost table accounts for. | — |
| readme-wrong-example-command | doc_code_drift | tests/integration/README.md:41 | Line 41 shows the command: `uv run pytest tests/integration/test_llm_integration.py::test_llm_accepts_deep_correct_answer -v`. Both the filename (`test_llm_integration.py` instead of `test_llm_service.py`) and the test name (`test_llm_accepts_deep_correct_answer` instead of `test_llm_accepts_correct_answer`) are wrong. Running this command as written would fail with a 'file not found' error. | — |
| bpe-reference-merges-loaded-but-unused | intent_mismatch | tests/fixtures/train-bpe-reference-merges.txt:1 | The fixture contains 243 BPE merge pairs and is loaded by `tests/test_train_bpe.py:41-49` into `reference_merges`. However, the only assertion that used this variable—`assert merges == reference_merges` at line 54—was commented out with the note 'Too strict - commented out'. After that block, `reference_merges` is never referenced again; it is a dead variable. The test instead validates only count with hardcoded bounds (lines 57-58: `assert len(merges) >= 243` and `assert len(merges) <= 245`), which are independent of the file content. The fixture was designed as the ground-truth reference for exact BPE training validation, but after the exact assertion was abandoned, the file's loaded content serves no purpose in any currently-running assertion. | — |
| action-version-skew-between-workflows | design_defect | .github/workflows/validate_cp_manifest.yml:20 | tests.yml uses `actions/checkout@v4` and `actions/setup-python@v5`, while validate_cp_manifest.yml uses the older `actions/checkout@v3` (line 20) and `actions/setup-python@v4` (line 24). Different action versions have different security patch levels and behaviours (e.g., checkout@v3 uses Node 16 which is EOL; checkout@v4 uses Node 20). Using inconsistent versions means the two workflows operate under different security baselines without explicit justification. | — |
| lint-gates-silenced | design_defect | .github/workflows/tests.yml:85 | tests.yml lines 85 and 89 both set `continue-on-error: true` for the ruff linter and ruff formatter checks respectively. Code that fails `ruff check engine/ tests/` or `ruff format engine/ tests/ --check` will still produce a green CI run. This removes the code-quality gate entirely, allowing malformed or style-violating code into the codebase without any blocking signal. A supply-chain audit expects quality gates to actually gate. | — |
| alpha-package-ty-no-upper-bound | design_defect | pyproject.toml:22 | pyproject.toml line 22: `"ty>=0.0.1a16"`. The `ty` package is at pre-release alpha status (version 0.0.1a16). Alpha packages carry no semantic versioning stability guarantees; any subsequent alpha release can silently break the API. There is no upper-bound constraint, so `uv sync` with an updated lock will install future incompatible alphas. This is a production runtime dependency (not dev-only), amplifying the risk. The package should either be pinned to an exact version in the lock file and reviewed on update, or moved to dev dependencies with an upper bound. | — |
| theory-files-duplicate-problem-qa | doc_code_drift | curricula/cp_accelerator/patterns/hash_table/theory/justify_questions.json:1 | The file `hash_table/theory/justify_questions.json` is byte-for-byte identical to `hash_table/problems/lc_1/justify_questions.json`. Both contain the same three question IDs (`two_sum_hash_table_advantage`, `two_sum_complexity`, `two_sum_edge_cases`) with identical text. A theory file should contain general hash table theory, not problem-specific Two Sum Q&A. Similarly, `sorting/theory/justify_questions.json` is identical to `sorting/problems/lc_912/justify_questions.json` (same three IDs: `sorting_conceptual`, `sorting_complexity`, `sorting_stability`). A learner who encounters both the theory phase and the lc_1 problem phase will see the same questions twice, defeating the purpose of the theory layer. | — |
| empty-test-suites-lc703-lc1804 | bug | curricula/cp_accelerator/patterns/heap_and_priority_queue/problems/lc_703/test_cases.json:5 | Two test_cases.json files contain `"tests": []` (empty array): `heap_and_priority_queue/problems/lc_703/test_cases.json` (Kth Largest Element in a Stream) and `trie/problems/lc_1804/test_cases.json` (Implement Trie II). The validators for these problems will run against zero test cases and trivially pass regardless of correctness, providing no learning signal. The trie/lc_208 test file has a similar weakness: only one test case with a string-encoded expected value `"[null, null, true, false, true, null, true]"` that may not be machine-comparable. | — |
| insert-before-check-name-intent-mismatch | intent_mismatch | curricula/cp_accelerator/patterns/hash_table/problems/lc_1/bugs/insert_before_check.json:1 | The file is named `insert_before_check.json`, its id is `"two-sum-insert-before-check"`, and the symptom file is named `insert_before_check_symptom.txt`. This name describes the bug pattern where the current element is inserted into the hash table BEFORE checking for the complement (allowing reuse of the same element). However, the actual implemented injection (pass 1) replaces `complement in seen` with `num in seen`, which is a fundamentally different bug: it checks for the current number's presence instead of the complement's presence. The `note` field confirms the redesign: 'Original bug used unsupported move_after type. Redesigned...' The id/filename/symptom describe a different bug than what is injected, misleading both learners receiving the symptom hint and maintainers extending the spec. | — |
| opaque-string-encoded-expected-values | design_defect | curricula/cp_accelerator/patterns/design_patterns/problems/lc_146/test_cases.json:9 | lc_146 (LRU Cache), lc_460 (LFU Cache), lc_307 (Range Sum Query Mutable), and lc_208 (Implement Trie) all have `"input": {}` and `"expected"` as a JSON string encoding a sequence of results: e.g. `"[null, null, null, 1, null, -1, null, -1, 3, 4]"`. These cannot be compared structurally by a validator — the validator must either parse the string or use string equality. There is no machine-readable mapping between operations and expected outputs, no constructor arguments, and no operation sequence. This design means the validator cannot meaningfully test stateful data-structure problems, which are exactly the hardest problems to get right. It is inconsistent with the structured input/expected schema used by all other test files. | — |
| alpha-prerelease-dep-ty | security | pyproject.toml:22 | `ty>=0.0.1a16` — `ty` is pinned to an alpha pre-release (`0.0.1a16`) as a runtime dependency. Alpha releases carry no stability guarantees; any future `>=0.0.1a17` alpha release could introduce breaking changes or security issues and will be picked up automatically. Pre-release packages in production `[project.dependencies]` are a supply-chain risk. | — |
| github-actions-not-sha-pinned | security | .github/workflows/tests.yml:14 | All four action references use semver tags, not commit SHAs: `actions/checkout@v4` (line 14), `actions/setup-python@v5` (line 18), `astral-sh/setup-uv@v3` (line 22), `actions/upload-artifact@v4` (line 52). A tag like `@v4` is mutable — a compromised maintainer or tag-force-push can inject malicious code into every CI run. SLSA supply-chain hardening requires pinning to an immutable commit SHA. | — |
| pip-install-uv-unpinned | security | .github/workflows/validate_cp_manifest.yml:29 | `run: pip install uv` with no version pin. This fetches the latest uv release at workflow run time. Any breaking change or malicious upload to the `uv` PyPI package will immediately affect the CI pipeline without any diff in the repository. The tests.yml workflow uses `astral-sh/setup-uv@v3` (tagged, not pinned, but at least version-constrained); validate_cp_manifest.yml uses no constraint at all. | — |
| lc1099-empty-test-cases | bug | curricula/cp_accelerator/patterns/two_pointers/problems/lc_1099/test_cases.json:5 | `"tests": []` — the test_cases.json for LeetCode 1099 (Two Sum Less Than K) contains an empty tests array. The validator shell script iterates this array; with zero entries, validation always vacuously passes. Any learner implementation—including a completely wrong one—passes the harden phase for this problem. | — |
| temperature-draft-inverted-direction | intent_mismatch | curricula/cs336_a1/modules/text_generation/bugs/temperature_after_softmax_draft.json:22 | Draft version of the temperature spec has the same inversion: description says "Applies temperature after softmax instead of before" but replacement source is `"F.softmax(next_logits / temperature, dim=-1)"` — temperature applied before softmax (correct). The draft is also directionally inverted. If the draft were promoted to production it would still not inject the described bug. | — |
| bpe-draft-noop-replacement | bug | curricula/cs336_a1/modules/bpe_tokenizer/bugs/wrong_merge_order_draft.json:43 | `"source": "node"` at line 43 (replacement block). The injection engine treats `source` as a literal Python expression string to substitute. Replacing with `"node"` means the replacement is the Python identifier `node` — a reference to the AST node object, not the code's original expression. This is effectively a no-op or produces a NameError at runtime. Additionally, the spec uses `"pass_": 1` (with underscore, line 9) instead of `"pass": 1`, and `target_function: "bpe_tokenizer"` rather than the correct `"train"`, meaning the engine will fail to locate the injection target. | — |
| cosine-draft-inverted-direction | intent_mismatch | curricula/cs336_a1/modules/cosine_schedule/bugs/wrong_cosine_range_draft.json:47 | Draft description: "Replace cosine decay calculation to include transformation (1 + cos(πt)) / 2". The replacement source is `"0.5 * (1.0 + math.cos(math.pi * progress))"` — this IS the correct cosine decay formula. The production version correctly injects the bug by substituting `math.cos(math.pi * progress)` (raw cosine without normalization). The draft inverts the intended direction: it replaces the buggy code with the correct formula rather than injecting the bug. | — |
| linear-draft-inverted-direction | intent_mismatch | curricula/cs336_a1/modules/linear/bugs/missing_transpose_draft.json | Bug spec description: "Find in_features.matmul(self.weight) and replace with in_features.matmul(self.weight.t())". The replacement source is `"y.matmul(self.weight.t())"` — this ADDS the transpose `.t()`. The intended bug is `missing_transpose` (removing `.t()` from correct code). The draft describes and implements the inverse: it would fix the bug rather than inject it. The production spec correctly removes `.t()`. | — |
| data-loader-draft-ast-expression-as-source | bug | curricula/cs336_a1/modules/data_loader/bugs/wrong_sampling_range_draft.json:56 | Replacement source at lines 56-59: `"node.value.keywords[0].value + 1"` — this is a Python AST traversal expression (using `.value.keywords[0].value`), not a literal Python source string to inject. The injection engine would emit the string `node.value.keywords[0].value + 1` as the replacement code, which is a NameError at runtime. Also uses `"pass_": 1` (underscore) and `target_function: "data_loader"` instead of the production `"get_batch"`. | — |
| rope-draft-ast-expression-as-source | bug | curricula/cs336_a1/modules/rope/bugs/wrong_rotation_draft.json | Replacement source is a Python string concatenation AST traversal expression: `"node.value.left.left.id + ' * ' + node.value.left.right.id + ' + ' + ..."` — this is engine-internal AST navigation code, not a literal Python source string to inject. The engine would emit this expression as the replacement code, producing a NameError at runtime. Additionally, `target_function: "apply_2d_rotation"` differs from the production spec's `"apply_rotary_position_embeddings"`, pointing to a different (or non-existent) function. | — |
| embedding-draft-ast-expression-as-source | bug | curricula/cs336_a1/modules/embedding/bugs/wrong_dimension_order_draft.json | Both passes (using `"pass_": 1` and `"pass_": 2` with underscore) have replacement sources that are Python AST traversal expressions: pass 1 `"node.value.keywords[1].value"` and pass 2 `"node.value.keywords[0].value"`. These are not literal Python source code strings — the engine would inject them verbatim, producing NameErrors at runtime. The underscore-suffixed `pass_` key also deviates from the v2.1 spec schema which uses `"pass"`. | — |
| gradient-clipping-draft-ast-expression-as-source | bug | curricula/cs336_a1/modules/gradient_clipping/bugs/per_parameter_clipping_draft.json | Pass 1 replacement source: `"node.value.func.value.args[0]"` and pass 2 replacement source: `"node.body[0]"` — both are Python AST traversal expressions, not literal source strings. The engine would inject these strings as code, producing NameErrors. Same `pass_` (underscore) schema deviation seen across other draft specs. | — |
| ci-pip-install-uv-unpinned | security | .github/workflows/validate_cp_manifest.yml:29 | Line 29: 'run: pip install uv' with no version constraint. This installs the latest available uv from PyPI at each CI run. A malicious or compromised PyPI release of the 'uv' package would execute arbitrary code on the CI runner before any project dependencies are installed or any trust checks are applied. The tests.yml workflow avoids this by using the official astral-sh/setup-uv action, but validate_cp_manifest.yml falls back to raw pip install. | — |
| pyproject-numpy-no-version-bound | design_defect | pyproject.toml:9 | The dependency declaration is bare '"numpy"' with no version specifier (no >=, ~=, or <). While uv.lock currently pins numpy to 2.3.2, any installation performed without the lockfile (e.g., pip install -e ., fresh contributor setup, third-party consumption) will resolve to the latest numpy regardless of compatibility. NumPy 2.x introduced breaking API changes vs 1.x (removal of np.bool/np.int/np.float aliases, C-API changes). All other numerical dependencies in the file carry explicit version constraints. | — |
| conftest-torch-load-no-weights-only | security | tests/conftest.py:199 | Line 199: 'state_dict = torch.load(FIXTURES_PATH / "ts_tests" / "model.pt", map_location="cpu")' uses the unsafe pickle-based torch.load without 'weights_only=True'. PyTorch 2.x emits a FutureWarning and will change the default in a future version. Loading a .pt file without weights_only=True executes arbitrary Python via pickle. If the fixture file tests/fixtures/ts_tests/model.pt is ever replaced by a compromised version (e.g., via supply-chain attack on test data), it could execute arbitrary code during the test suite. | — |
| multihead-attention-draft-wrong-target-function | bug | curricula/cs336_a1/modules/multihead_attention/bugs/missing_transpose_back_draft.json:5 | Draft spec has `"target_function": "multihead_attention"` but the actual PyTorch class method is named `forward`. Production spec missing_transpose_back.json:5 correctly uses `"target_function": "forward"`. GenericBugInjector._has_function() at generic_injector.py:200-205 walks all ast.FunctionDef nodes; no node is named 'multihead_attention' in student code, so the check at line 99 (`not self._has_function(original_ast, target_function)`) returns True and injection is aborted with `return source_code, False`. Combined with finding harden-select-bug-picks-drafts, this creates a ~50% runtime failure rate for multihead_attention harden sessions when both files are present. | — |
| bug-definition-schema-production-specs-violate-contract | design_defect | engine/schemas.py:318-349 | PassDefinition (line 318) requires `description: str` (non-Optional). BugDefinition (line 343) requires `metadata: BugMetadata` (non-Optional). Two confirmed production specs violate both requirements: (1) curricula/cs336_a1/modules/swiglu/bugs/missing_gate.json — pass entries have no 'description' field, and the top-level object has no 'metadata' field; (2) curricula/cs336_a1/modules/multihead_attention/bugs/missing_transpose_back.json — same omissions. `BugDefinition.model_validate(data)` on either file raises pydantic.ValidationError. GenericBugInjector.validate_definition() at generic_injector.py:42-50 bypasses the Pydantic schema with manual checks, so no current runtime failure — but any code that uses the published schema contract (e.g., dev_tools, future validators) will fail. | — |
| justify-fast-filter-false-positive | design_defect | engine/stages/justify.py:111 | Lines 109-115: `for failure_mode in question.failure_modes: for keyword in failure_mode.keywords: if keyword.lower() in user_answer_lower: return True, failure_mode.feedback`. The filter triggers on ANY occurrence of a failure-mode keyword anywhere in the student's answer, including in a fully correct, technically precise answer. Example: the softmax/justify_questions.json 'Hand-Waver' failure mode includes keywords ['stability', 'numerical', 'better', 'safer']. A correct answer that mentions 'subtracting the max value prevents numerical overflow, improving numerical stability' contains 'stability' and 'numerical' and would be falsely flagged. The logic performs positive keyword presence detection rather than identifying genuinely vague or incomplete answers. This contradicts the stated purpose ('catch shallow/vague answers') and will incorrectly penalise high-quality responses. | — |
| bpe-tokenizer-q2-model-answer-missing-counterexample | bug | curricula/cs336_a1/modules/bpe_tokenizer/justify_questions.json:18 | Question bpe_tokenizer_q2 requires students to 'prove by example that greedy BPE is NOT globally optimal — construct a small corpus where the greedy strategy leads to MORE total tokens than an alternative.' The model_answer (line 18) walks through two attempts. The first attempt explicitly concludes 'Wait, both approaches give same result!' (a tie, not a refutation). The second attempt explicitly concludes 'Greedy gives 7 tokens, alternative gives 9. Actually greedy wins here!' — an example where greedy is strictly BETTER than the alternative. The answer never constructs a corpus where greedy produces more tokens than an alternative strategy. The model_answer ends with the unsupported claim 'Greedy BPE is locally optimal but not globally optimal' without providing the required counterexample. Students who study this answer will believe an invalid proof satisfies the question. | — |
| linear-q3-rmsnorm-mean-subtraction-error | bug | curricula/cs336_a1/modules/linear/justify_questions.json:31 | Question linear_q3 model_answer (line 31) opens: 'Why normalization makes bias redundant: LayerNorm/RMSNorm compute: normalized = (x - mean) / std. The subtraction of mean ELIMINATES any constant bias!' This is factually incorrect for RMSNorm. RMSNorm computes normalized = x / sqrt(mean(x^2) + eps) — there is no mean subtraction, so a constant bias is NOT cancelled. The answer internally contradicts itself later: 'RMSNorm is similar but doesn't subtract mean, yet still normalizes magnitude.' The opening sentence is never corrected, leaving 'LayerNorm/RMSNorm compute: (x - mean) / std' as a direct false claim students will memorize. The required_concepts list (lines 33-39) is correctly limited to LayerNorm and does not claim RMSNorm subtracts mean, confirming the model_answer text is erroneous. | — |

## LOW (96)
| ID | Class | Location | Evidence | Adversarial note |
| --- | --- | --- | --- | --- |
| lc1-unquoted-heredoc | security | curricula/cp_accelerator/patterns/hash_table/problems/lc_1/validator.sh:18 | The lc_1 validator at line 18 uses an unquoted heredoc delimiter: `${MASTERY_PYTHON:-python3} << EOF`. All 51 other cp_accelerator validators use `<< 'EOF'` (quoted), which suppresses bash parameter expansion, command substitution, and arithmetic expansion inside the heredoc body. With an unquoted delimiter, any `$var`, `$(cmd)`, or backtick construct in the Python code would be expanded by the shell before being passed to Python. The current heredoc body does not contain exploitable shell expansions (Python f-string braces `{...}` are not shell syntax), but the inconsistency is a latent risk: if the heredoc content is ever modified to include `$` characters (e.g., in comments or strings), shell expansion could corrupt or inject code. All other validators correctly use the quoted form. | — |
| cs336-time-arithmetic-injection | security | curricula/cs336_a1/modules/adamw/validator.sh:47 | Line 47 of the adamw, attention, and bpe_tokenizer validators: `duration=$(python3 -c "print($end_time - $start_time)")`. The variables `$end_time` and `$start_time` are captured from `python3 -c 'import time; print(time.time())'` (lines 27 and 44). They are placed unquoted inside a double-quoted `python3 -c` string, which means bash expands them before the string is passed to Python. If either variable contained shell-special characters or newlines (e.g., if `time.time()` output were somehow tampered), arbitrary Python expressions could be injected into the `-c` argument. In practice `time.time()` always returns a decimal float such as `1699999999.123` with no special characters, making exploitation purely theoretical. The same pattern exists identically in `curricula/cs336_a1/modules/attention/validator.sh:47` and `curricula/cs336_a1/modules/bpe_tokenizer/validator.sh:47`. | — |
| lc1-import-api-inconsistency | design_defect | curricula/cp_accelerator/patterns/hash_table/problems/lc_1/validator.sh:27 | Line 27: `from solution import twoSum`. All 51 other cp_accelerator validators use `from solution import solve` (the standardized alias). The lc_1 solution (`solution.py:36`) does define `solve = twoSum`, but the validator bypasses it and imports the problem-specific function name directly. This creates an inconsistency: if a learner follows the project convention of providing a `solve` entry point, the lc_1 validator still requires `twoSum`. Conversely, if the lc_1 solution ever renamed the function (e.g., during a refactor), only this validator would break. The unquoted heredoc (`<< EOF`, not `<< 'EOF'`) is a second divergence from the template used for all other validators. | — |
| cs336-shadow-worktree-unvalidated-path | security | curricula/cs336_a1/modules/adamw/validator.sh:18 | Line 18: `cp cs336_basics/optimizer.py "$SHADOW_WORKTREE/cs336_basics/optimizer.py"`. The `SHADOW_WORKTREE` variable is supplied entirely by the engine via environment variable (lines 8-11 check it is non-empty but do not validate its value). If the engine or a caller set `SHADOW_WORKTREE` to a path containing `..` components (e.g., `/tmp/../../etc/cron.d`), the `cp` would write to an unintended location. The double-quoting on `"$SHADOW_WORKTREE"` protects against word splitting but not path traversal. The same pattern exists in `curricula/cs336_a1/modules/attention/validator.sh:18` (copying layers.py) and `curricula/cs336_a1/modules/bpe_tokenizer/validator.sh:18` (copying tokenizer.py). Since `SHADOW_WORKTREE` is controlled by the trusted engine component, practical exploitation requires engine compromise. | — |
| bare-except-swallows-sigint | bug | engine/main.py:2007 | except: at line 2007 (inside the init command's state-file pre-check block) with comment 'State file doesn't exist or is corrupt - treat as fresh init'. A bare except catches KeyboardInterrupt and SystemExit in addition to Exception, preventing clean termination when the user presses Ctrl+C during this block of the init command. | — |
| log-injection-curriculum-id | security | engine/state.py:53 | logger.info(f"Loaded progress: curriculum={progress.curriculum_id}, ...") at line 53 — curriculum_id is persisted from the user-supplied CLI argument and interpolated directly into the log message without sanitization. A value containing newlines (\n) or ANSI escape codes can forge additional log entries or corrupt structured log parsing downstream. | — |
| s-hardcoded-path-golden | security | scripts/add_successful_to_golden.py:77 | golden_dir = Path(f"/Volumes/Totallynotaharddrive/assignment1-basics/curricula/cs336_a1/modules/{module}/bugs")  — hardcoded developer-machine external-drive path leaks filesystem layout including volume name and internal project structure. If this script is run on any machine other than the original developer's, it silently resolves to a non-existent path and will fail or write to an unintended location. | — |
| s-hardcoded-path-auto-fix | security | scripts/auto_fix_drafts.py:218 | base_path = Path("/Volumes/Totallynotaharddrive/assignment1-basics/curricula/cs336_a1/modules")  — same developer-machine external-drive path. Script is otherwise a production curriculum maintenance tool but will fail silently on any other machine. | — |
| s-hardcoded-path-fix-draft | security | scripts/fix_draft_pattern.py:71 | base_path = Path("/Volumes/Totallynotaharddrive/assignment1-basics/curricula/cs336_a1/modules")  — identical hardcoded /Volumes/Totallynotaharddrive path baked into a curriculum batch-fix script. | — |
| s-hardcoded-path-ground-truth | security | scripts/generate_ground_truth.py:22 | base_path = Path("/Volumes/Totallynotaharddrive/assignment1-basics/curricula/cs336_a1/modules")  — same developer-machine volume path in a script that reads reference solutions and writes golden ground-truth files. | — |
| s-hardcoded-path-verify | security | scripts/verify_ground_truth.py:19 | base_path = Path("/Volumes/Totallynotaharddrive/assignment1-basics/curricula/cs336_a1/modules")  — same hardcoded external-volume path in the verification script. | — |
| s-ast-literal-eval-api-data | security | scripts/generate_module.py:120 | result[key] = ast.literal_eval(value)  — value comes from parsing the LeetCode API response (via the unofficial proxy at enrich_problems.py:39). ast.literal_eval is safe against code execution but will raise ValueError/SyntaxError on unexpected content; the proxy can return content that causes the curriculum generation pipeline to crash or skip fields silently. Same pattern repeated at line 452. | — |
| b-subprocess-os-environ | bug | tests/e2e/debug_shadow_worktree.py:73 | env={**subprocess.os.environ, "PYTHONPATH": str(shadow_worktree)}  — subprocess.os is not a documented public attribute; accessing os through the subprocess module works only because Python caches the module reference internally. The canonical form is os.environ. If the internal reference is ever removed in a future Python release this will raise AttributeError. | — |
| rlimit-as-targets-virtual-not-physical-memory | design_defect | tests/test_tokenizer.py:24 | The `memory_limit` decorator sets `resource.RLIMIT_AS` (virtual address space limit) via `resource.setrlimit(resource.RLIMIT_AS, (process.memory_info().rss + max_mem, -1))`. `RLIMIT_AS` governs the total virtual address space, not physical (RSS) memory. On a Python process with PyTorch loaded, the virtual address space is commonly 10-100x larger than RSS due to CUDA shared libraries, memory-mapped files, and lazy allocations; setting it to `rss + 1 MB` creates an extremely tight cap. Any C extension within the function that attempts to `mmap` additional virtual memory will receive `ENOMEM` and may raise `MemoryError` or crash the process, even if physical memory is plentiful. The `finally` block at line 34 does correctly restore the prior limits, so the window is limited; but the misidentification of RSS as a proxy for virtual address space makes this mechanism unreliable and potentially destabilising for other threads in the same process. | — |
| lc1-validator-unquoted-heredoc | bug | curricula/cp_accelerator/patterns/hash_table/problems/lc_1/validator.sh:18 | `${MASTERY_PYTHON:-python3} << EOF` (no quotes around EOF delimiter) enables shell variable/command expansion inside the heredoc body. All other CP validators correctly use `<< 'EOF'` (single-quoted) to suppress expansion. Currently harmless because the heredoc body has no `$` variable references that would be ambiguously expanded, but the pattern is fragile: any future addition of Python f-strings or shell-like syntax inside that heredoc could be silently mangled by the shell before Python sees it. | — |
| lc435-in-place-input-mutation | bug | curricula/cp_accelerator/patterns/greedy/problems/lc_435/solution.py:25 | `intervals.sort(key=lambda x: x[1])` mutates the caller's list in-place. When the validator calls `solve(**test["input"])`, `test["input"]["intervals"]` is the live Python list parsed from JSON and stored in `test_cases`. After the call, that list is permanently reordered. Within a single validator run this is safe because each test is executed once and data is not reused. However, if any test case appeared more than once (or if the engine reuses parsed test data across calls), the mutated order could cause incorrect results on a second invocation. The fix is `intervals = sorted(intervals, key=lambda x: x[1])` to work on a copy. | — |
| S2C-011 | bug | engine/services/llm_service.py:122 | In mock-mode feedback construction: `f"- Failure Modes: {', '.join(question.failure_modes[:3])}"`. question.failure_modes is typed list[FailureMode] (Pydantic model objects), not list[str]. str.join on Pydantic objects raises TypeError: sequence item N: expected str instance, FailureMode found. This exception is thrown every time mock mode is active (no OPENAI_API_KEY) and a justify question is evaluated. | — |
| S2C-012 | design_defect | engine/ast_harden/generic_injector.py:174 | `buggy_code = ast.unparse(original_ast)` — ast.unparse round-trips through the AST and drops all comments, docstrings in non-standard positions, and reformats whitespace. The student's carefully written comments in their implementation are silently stripped from the harden workspace file. The student then debugs code that looks different from what they wrote. | — |
| S2C-013 | bug | engine/main.py:2007 | In the init command's state-load block: `except: pass`. A bare except with pass silently swallows every exception (including StateFileCorruptedError, KeyboardInterrupt, SystemExit) that occurs when loading existing state before initialization. The user receives no diagnostic message; initialization proceeds as if the load succeeded. | — |
| S2C-014 | doc_code_drift | curricula/job_prep_data_annotation/modules/extract_coordinates/validator.sh:1 | All three job_prep_data_annotation validators (extract_coordinates, render_grid, fetch_document) import from `cs336_basics.utils`: `from cs336_basics.utils import extract_coordinates`, etc. cs336_basics is the CS336 assignment package; these functions (extract_coordinates, render_grid, fetch_document) are not present in modes/developer/cs336_basics/utils.py (which contains softmax, cross_entropy, gradient_clipping, get_lr_cosine_schedule, get_batch, save_checkpoint, load_checkpoint). The validators reference a namespace that does not match the actual module content, indicating copy-paste infrastructure from the CS336 template applied to a different curriculum. | — |
| S2C-015 | bug | modes/developer/cs336_basics/pretokenization_example.py:53 | `with open(..., 'rb') as f:` is a module-level statement (not inside a function or __main__ guard). Python's Ellipsis literal (`...`) is not a valid path argument; `open(...)` raises `TypeError: expected str, bytes or os.PathLike object, not ellipsis` at import time. Any code that does `import pretokenization_example` or `from modes.developer.cs336_basics import pretokenization_example` will fail immediately. The usage block at lines 52-62 should be inside `if __name__ == '__main__':` or have the path replaced. | — |
| S2C-016 | bug | engine/main.py:2060 | `cs336_symlink = Path('cs336_basics')` is a CWD-relative path used to check for and recreate the cs336_basics symlink in the shadow worktree (lines 2060-2068). If `mastery init` is run from any directory other than the project root, `cs336_symlink.is_symlink()` returns False (no symlink found at that relative path), the shadow worktree is created without the symlink, and all cs336_basics imports in the validation environment will fail with ModuleNotFoundError at runtime. | — |
| S2C-017 | design_defect | engine/stages/justify.py:112 | `check_fast_filter` matches failure-mode keywords anywhere in the user's answer string (case-insensitive substring match). A correct answer that uses a failure-mode keyword in a negation context (e.g., 'this is NOT hand-wavy because...') will be rejected as if the failure mode were present. The filter has no negation awareness or word-boundary constraints. The env-var escape hatch (MASTERY_DISABLE_FAST_FILTER) exists but requires user knowledge of internals. | — |
| validate-stubs-skips-dunder | design_defect | scripts/validate_student_stubs.py:33 | `visit_FunctionDef` at line 33 contains `if node.name.startswith('_'): self.generic_visit(node); return`. This skips ALL underscore-prefixed functions including `__init__`, `__len__`, `__iter__`, etc. In the cs336_basics student stubs, `__init__` methods in optimizer classes raise `NotImplementedError` as expected stubs, but these are never validated by this visitor. A student could implement `__init__` completely without triggering a validation failure. The intended behavior (validate ALL public stubs) is inconsistent with the actual behavior (skip dunder stubs entirely). | — |
| validate-stubs-any-stub-passes-file | design_defect | scripts/validate_student_stubs.py:92 | The stub-check logic at line 92: `if validator.functions_with_stubs > 0 or has_todo: return True`. This means if ANY single public function still has a `NotImplementedError` stub, the entire file is declared validly-stubbed — even if all other public functions have been fully implemented. A partial-implementation file (student completed 9 of 10 methods) would pass this check identically to a fully-stubbed file. The validator cannot detect partial implementations. | — |
| parse-example-input-no-parens-tracking | bug | scripts/generate_module.py:77 | `parse_example_input` tracks bracket nesting to avoid splitting on commas inside nested structures. The nesting counter increments on `[` and `{` and decrements on `]` and `}`, but parentheses `(` and `)` are not tracked. For input strings containing function calls like `matrix = [[1,2],[3,4]], func(1, 2)`, the comma inside `func(1, 2)` would be treated as a top-level split point, producing incorrect parse results. This affects any LeetCode problem whose example input contains tuple or function-call syntax. | — |
| ground-truth-strict-string-compare | design_defect | scripts/generate_ground_truth.py:66 | `test_golden_pattern` compares generated code to golden files using `.strip()` only. No normalization of internal whitespace, indentation style, or line endings is performed. Any difference in trailing spaces, tab-vs-space indentation, or Windows vs Unix line endings between the LLM-generated output and the golden file causes a false failure. This makes the golden-pattern tests brittle in CI environments that may differ from the original development environment. | — |
| conftest-torch-load-no-weights-only | bug | tests/conftest.py:199 | Line 199: `state_dict = torch.load(FIXTURES_PATH / "ts_tests" / "model.pt", map_location="cpu")` without `weights_only=True`. Since PyTorch 2.0, this form emits a `FutureWarning` and is deprecated in favor of `weights_only=True`. Without this flag, loading uses `pickle.load` which can execute arbitrary code in a maliciously crafted `.pt` file. While the fixture is internal, this establishes an unsafe pattern. Confirmed via grep: line 199 contains the unsafe `torch.load` call. | — |
| parse-sources-max-finds-later-vital | intent_mismatch | scripts/parse_sources.py:96 | Line 96: `vital_start = max(section.find("[Vital]"), section.find("\\[Vital\\]"))`. The intent is to locate a `[Vital]` marker in either its literal form or its escaped markdown form. Using `max()` returns the LATER character position when both forms are present in the same section, rather than the first occurrence. If a section contains `\[Vital\]` at position 10 and `[Vital]` at position 50, `max(10, 50) = 50` skips the escaped form at position 10. `min()` (or an `or`-chain preferring the non-negative result) would correctly find whichever marker appears first. Confirmed via grep at line 96. | — |
| statistical-assertion-not-guaranteed | bug | tests/test_data.py:37 | Lines 37-38: `assert max(starting_indices) == num_possible_starting_indices - 1` and `assert min(starting_indices) == 0`. With `num_possible_starting_indices = 100 - 7 = 93` indices and only `1000 * 32 = 32000` draws, the probability that the maximum index (92) and the minimum index (0) are both sampled at least once is very high but is not 1. In rare CI runs the test can non-deterministically fail. Neither assertion is preceded by a note or `pytest.mark.flaky`, making the source of failure opaque. | — |
| idx-identity-permutation-no-shuffle | design_defect | tests/one_d_probes.py:55 | Line 55: `idx = torch.arange(64)` produces the identity permutation `[0, 1, ..., 63]`. Slicing `Xs[idx[:train_N]]` is identical to `Xs[:train_N]`; the `idx` variable provides no randomisation. A `torch.randperm(64)` would randomise the train/test split at the same seed, which appears to be the intent (and would be consistent with the shuffle of `Xtr` in the training loop at line 153). As written, train always uses the first 51 sequences and test always uses the next 13, a deterministic non-overlapping but potentially unrepresentative split. | — |
| sinusoidal-pe-odd-dmodel-shape-error | bug | tests/one_d_probes.py:62 | In `sinusoidal_positional_encoding` (lines 62-64): `div_term = torch.exp(torch.arange(0, d_model, 2).float() * ...)` has `ceil(d_model/2)` elements. For odd `d_model`, `pe[:, 0::2]` has `ceil(d_model/2)` columns (matches) but `pe[:, 1::2]` has `floor(d_model/2)` columns (one fewer). The assignment `pe[:, 1::2] = torch.cos(position * div_term)` would attempt to broadcast a `(max_len, ceil(d_model/2))` tensor into a `(max_len, floor(d_model/2))` slice, raising a runtime shape error. The function is safe only because the default `d_model=64` is even; passing any odd value (e.g. 65) would crash at runtime. | — |
| ts-state-dict-fixture-unclosed-file | bug | tests/conftest.py:200 | Line 200: `config = json.load(open(FIXTURES_PATH / "ts_tests" / "model_config.json"))`. The file object returned by `open(...)` is never explicitly closed; no context manager is used. In CPython the reference count drops to zero and the file is closed immediately, but under PyPy or other implementations the finalizer may be deferred, leaving a file descriptor open for the lifetime of the fixture. The fix is `with open(...) as f: config = json.load(f)`. | — |
| validator-boilerplate-mass-duplication | design_defect | curricula/cp_accelerator/patterns/backtracking/problems/lc_78/validator.sh:1 | All ~42 CP accelerator validators (lc_78 through lc_684) share an identical boilerplate: `set -e`, `SCRIPT_DIR` computation, `solution.py` existence check, same Python heredoc with identical test-loop logic (enumerate test_cases['tests'], call solve(**input), compare to expected, print PASS/FAIL, exit on failure count). None of this is extracted to a shared runner script or library. Any change to the test loop (e.g., adding timeout, better diff output, sorting-insensitive comparison) must be replicated in ~42 files. The lc_1 validator is already diverging (different import, `cd`, unquoted heredoc, sorted-comparison logic) as a consequence of this duplication. | — |
| cs336-validators-duplicate-pwd-stage-detection | design_defect | curricula/cs336_a1/modules/adamw/validator.sh:16 | All three CS336 validators audited (adamw, attention, bpe_tokenizer — and by pattern all cs336 validators) use identical stage-detection logic: `if [ "$(pwd)" != "$SHADOW_WORKTREE" ]` at line 16. The else branch (`cd "$SHADOW_WORKTREE"`) is a no-op when already in the shadow worktree. This if/else exists verbatim in every CS336 validator with no shared implementation. Additionally, if `$SHADOW_WORKTREE` is not set (line 8 guard), the script exits with a generic error that doesn't indicate which module failed, making diagnosis harder in CI. | — |
| lc435-solution-mutates-input | design_defect | curricula/cp_accelerator/patterns/greedy/problems/lc_435/solution.py:25 | `intervals.sort(key=lambda x: x[1])` mutates the caller's list in-place. As a reference solution used in the Harden stage (where the bug-injection engine applies patches), in-place mutation of test input could cause cascading test failures if the same input list is reused across test cases. The problem description does not call for in-place sorting. This is a side-effect that is invisible to the test harness but could mask bugs. | — |
| unconditional-logging-stream-handler | design_defect | engine/main.py:80 | A `logging.StreamHandler()` is added unconditionally to the root logger during CLI init (lines 80-87). If the calling environment has already configured a handler (e.g., test harnesses, IDE launchers), this produces duplicate log output. Standard practice is to check `if not logger.handlers` or use `logging.basicConfig` which respects existing configuration. (Location confirmed from prior session read.) | — |
| dead-validator-env-dict | design_defect | engine/main.py:63 | `validator_env` dict is defined at lines 63-68 (populated with `SHADOW_WORKTREE`, `MASTERY_PYTHON`, `PATH`) but is never passed to any subprocess call or referenced elsewhere in `main.py`. The actual environment variables are set separately inside `ValidationSubsystem.execute()`. This is dead code that creates a false impression that the validation environment is configured in `main.py`. (Location confirmed from prior session read.) | — |
| dead-softmax-bug-injector-in-ast-service | design_defect | engine/services/ast_service.py:1 | `SoftmaxBugInjector` class (and the entire `ast_service.py` class hierarchy: `Canonicalizer`, `CanonicalPatternMatcher`, `OriginalASTTransformer`) is never imported or called by any production code path. Production harden uses `engine.ast_harden.generic_injector.GenericBugInjector`. The `ast_service.py` module ships as dead weight alongside two other dead duplicates (`softmax_poc.py`, `softmax_v2_1.py`), creating a three-way duplication of the same AST transformation hierarchy. (Confirmed from direct read; no import of `ast_service` found in `main.py`, `harden.py`, or `generic_injector.py`.) | — |
| dead-transform-original-method | design_defect | engine/ast_harden/pattern_matcher.py:1 | `FindAndReplaceTransformer.transform_original()` exists in `pattern_matcher.py` but `generic_injector.py` never calls it — it calls `visit()` directly on the transformer. `transform_original()` is dead code within the active production file, not just a dead module. (Exact line unverified in this session; confirmed from prior read of pattern_matcher.py and generic_injector.py.) | — |
| hello-world-validator-no-shadow-copy | design_defect | curricula/dummy_hello_world/modules/hello_world/validator.sh:8 | The dummy validator checks `if [ -f 'hello_world.py' ]` in CWD. The comment says 'The validator runs FROM the workspace directory' but `ValidationSubsystem.execute()` sets CWD to `$SHADOW_WORKTREE`. The cs336_a1 validators all perform an explicit `cp <file> $SHADOW_WORKTREE/<file>` before running tests. This validator skips that step, so `hello_world.py` (in the student's main workspace) will not exist in the shadow worktree CWD, causing the validator to always fail unless the engine has a special-case path for this module. | — |
| make-submission-broad-json-exclusion | design_defect | maintenance/make_submission.sh:21 | `make_submission.sh` line 21 passes `-x '*.json'` to `zip -r`, excluding ALL JSON files from the submission archive. This is overly broad: it would also exclude any JSON configuration or data files (e.g., vocabulary files, test fixtures) that the student may have legitimately created. The exclusion should be scoped to known output files (e.g., `test_results.json`) rather than all `.json` by glob. | — |
| generate-completion-mro-check-leaky | design_defect | engine/services/llm_service.py:229 | `generate_completion()` uses `if response_format and hasattr(response_format, '__mro__'):` to detect Pydantic models (line 229). `__mro__` is a standard attribute on ALL Python classes, not just Pydantic models. Any class type (including `int`, `list`, a custom dataclass) passed as `response_format` would trigger the Structured Outputs beta API path and likely produce an `AttributeError` or unexpected behaviour. The correct guard is `isinstance(response_format, type) and issubclass(response_format, BaseModel)`. | — |
| bug-author-prompt-field-name-drift | doc_code_drift | engine/dev_tools/bug_author.py:648 | `_build_user_prompt()` in `bug_author.py` at line 648 instructs the LLM to output a field named `"path": "node.right"` in the replacement schema. However the actual JSON schema for bug definitions uses `"source"` as the field name (per the schema observed in `generic_injector.py`). If the LLM follows the prompt literally, it produces bug JSON files with `path` keys that the injector silently ignores, resulting in no-op bug definitions. (Location confirmed from prior session read.) | — |
| dependency-injection-comment-dead | doc_code_drift | engine/main.py:1539 | Lines 1539-1540 contain a dead comment referencing 'dependency injection' for `submit_justification()`. The code below it does not implement dependency injection — services are constructed inline. The comment describes an aspirational design that was never realised, misleading readers about the architectural intent. (Location confirmed from prior session read.) | — |
| tokenizer-stub-empty-dead-file | design_defect | modes/student/cs336_basics/tokenizer_stub.py:1 | File contains one blank line and nothing else. tokenizer.py already exists in the same directory with the actual Tokenizer class stub. tokenizer_stub.py has no imports, no docstring, no class or function definitions. It is imported nowhere in the codebase (unverified by search of test/engine files read). It creates ambiguity about which file is the authoritative stub. | — |
| verify-curriculum-manifests-hardcoded-single-curriculum | design_defect | scripts/verify_curriculum_manifests.py:93 | The __main__ block at line 93 hardcodes 'curricula/cs336_a1' as the sole path to verify. The verify_curriculum_path() function defined above accepts any path argument, but the script never iterates over all curricula directories. The cp_accelerator curriculum is never validated. The tool presents as a general manifest verifier but only covers one curriculum. | — |
| generate-ground-truth-private-method-access | design_defect | scripts/generate_ground_truth.py:121 | Lines 121-123 directly call author._extract_patch_info(...), author._build_system_prompt(), and author._build_user_prompt(...) — all underscore-prefixed private methods of the BugAuthor class. This tightly couples the maintenance script to private implementation details. Any refactoring of BugAuthor internals silently breaks generate_ground_truth.py without a public API contract boundary. | — |
| migrate-bugs-local-import-json | design_defect | scripts/migrate_bugs_llm.py:108 | At lines 108-109, inside an 'if success:' conditional block: 'import json' appears. json is a stdlib module that should be imported at module level. If the if-branch is not entered and json is referenced later in the same scope, it would raise NameError. The local import suggests the block was added ad hoc without updating the module-level import section. | — |
| module-level-seed-mutation-on-import | design_defect | tests/one_d_probes.py:10-12 | Lines 10-12 execute `random.seed(SEED)`, `np.random.seed(SEED)`, `torch.manual_seed(SEED)` at module scope. Because the file lives in `tests/`, any pytest plugin, conftest, or test that imports (even indirectly) this module will mutate global RNG state as a side-effect of import. pytest collects all modules in the `tests/` tree; though it won't run test functions (none exist here), the import side-effect fires regardless and can non-deterministically alter other tests' random draws. | — |
| one-d-probes-not-collected-as-tests | design_defect | tests/one_d_probes.py:1 | The file is placed inside `tests/` alongside pytest test modules but contains no `test_*` functions and is not named `test_*.py`. It is a standalone training script (entry point: `if __name__ == "__main__":` at line 146). It will never be collected by pytest, so any correctness it is meant to probe is never automatically verified by CI. The script could silently break without any test failure alerting developers. | — |
| stale-1000-iter-comment | doc_code_drift | tests/test_serialization.py:72 | Line 72 reads `# Use 1000 optimization steps for testing`, yet `num_iters = 10` is set at line 62 and used in `for _ in range(num_iters):` at line 73. The comment appears to be a copy-paste from `test_optimizer.py` where `_optimize()` genuinely runs 1000 steps (line 18: `for _ in range(1000)`). The discrepancy is misleading: a reader expecting deep optimizer convergence testing will find only 10 iterations. | — |
| force-update-dead-parameter-in-snapshot | design_defect | tests/conftest.py:60 | `NumpySnapshot.assert_match` (conftest.py line ~44) accepts `force_update` as a parameter and assigns it from `self.default_force_update` at line 60, but never consults the value again — the subsequent code always loads the `.npz` snapshot and compares. The same pattern applies to `Snapshot.assert_match` (lines 117-150). The `force_update` parameter is an accepted but dead abstraction: there is no save-and-skip code path, so snapshot creation requires direct code modification rather than the implied API. Any caller passing `force_update=True` receives silent no-op behaviour. | — |
| readme-cs336-spring-year-inconsistency | doc_code_drift | README.md:411 | README.md:85 identifies the cs336_a1 curriculum as 'Stanford CS336' and links to `https://stanford-cs336.github.io/spring2025/` (Spring 2025). README.md:411 in the attribution section says 'adapted from **[Stanford CS336 (Spring 2024)](https://stanford-cs336.github.io/spring2024/)**'. The same curriculum is attributed to two different course years in two places in the same document. One or both URLs/years must be wrong. | — |
| lc1-build-prompt-sorting-resource | doc_code_drift | curricula/cp_accelerator/patterns/hash_table/problems/lc_1/build_prompt.txt:102 | The Two Sum (lc_1) build prompt at line 102 lists `2. https://www.geeksforgeeks.org/sorting-algorithms/` as a learning resource. Two Sum is a hash table problem requiring O(n) one-pass lookup; sorting algorithms are irrelevant and potentially misleading (brute-force sort-based approach is O(n log n), opposite of the intended O(n) hash-table approach). The correct resource should be about hash tables or two-sum specifically. | — |
| status-md-roadmapresources-path-wrong | doc_code_drift | curricula/cp_accelerator/STATUS.md:117 | STATUS.md:117 lists 'Source Files: `RoadmapResources.md` - Roadmap with rating brackets and resources' with no path qualifier, implying the file is in the repo root. The file is actually at `maintenance/RoadmapResources.md` (confirmed by `ls maintenance/`; `ls RoadmapResources.md` at root returns 'NOT at root'). META_AUDIT_DEC_18.md:119 independently corroborates this: 'references RoadmapResources.md at repo root (actual file lives under maintenance/)'. Any developer following STATUS.md to locate this file will not find it. | — |
| meta-audit-python-version-stale | doc_code_drift | audits/META_AUDIT_DEC_18.md:219 | META_AUDIT_DEC_18.md:219 states 'The repository README advertises "Python 3.10+" (badge + prerequisites)'. However, the current README.md:6 shows `[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)]` — the badge has since been updated to 3.11+. The meta-audit's characterization of the README is now factually incorrect for this specific point, making it an unreliable basis for the follow-up action on Python version consistency it recommends. | — |
| quality-audit-status-in-progress-misleading | doc_code_drift | audits/QUALITY_AUDIT.md:12 | QUALITY_AUDIT.md:12 sets 'Status: In Progress (Expanded Checklist I–XIII)'. Checklist sections X (Mode Parity), XI (Documentation & UX Consistency), XII (CI/Workflow Scope), and XIII (Dependency/Supply Chain) are all unchecked (`[ ]` not `[x]`) at lines 181-212. The document is simultaneously the 'primary audit artifact' (as labeled in the understanding map) and admits it is incomplete. The Findings Count table (9 high / 22 medium / 3 low) does not include findings that would emerge from the unchecked sections, so the count understates known risks. This creates false confidence in audit completeness. | — |
| incomplete-merge-symptom-missing-structured-format | doc_code_drift | curricula/cp_accelerator/patterns/sorting/problems/lc_912/bugs/incomplete_merge_symptom.txt:1 | The incomplete_merge_symptom.txt uses unstructured plain text (7 lines of prose starting with 'Wrong Answer on Test 2') without the structured markdown format used by all other symptom files in the required set. Compare: missing_base_case_symptom.txt uses '# Bug Symptom: Missing Base Case in Merge Sort' with '## What You'll Observe', '## Expected Behavior', '## Actual Behavior', '## Debugging Guide', '## Conceptual Understanding' sections. Same structured format appears in skip_consecutive_symptom.txt, off_by_one_prefix_symptom.txt, missing_empty_check_symptom.txt, wrong_pointer_move_symptom.txt. The inventory describes this file as 'Symptom description for the incomplete merge bug' but the format diverges from the established pattern. | — |
| lc589-wrong-learning-resource-dp-fibonacci-video | doc_code_drift | curricula/cp_accelerator/patterns/traversal/problems/lc_589/build_prompt.txt:80 | The N-ary Tree Preorder Traversal build prompt (lc_589) lists at line 80: "{'type': 'video', 'title': 'Dynamic Programming lecture #1 - Fibonacci, iteration vs recursion', 'url': 'https://www.youtube.com/watch?v=YBSt1jYwVfU'}". A DP/Fibonacci video is unrelated to N-ary tree preorder traversal. The same misplaced resource appears in lc_144/build_prompt.txt:82 for Binary Tree Preorder Traversal. The inventory describes these files as 'implementation instructions' for tree traversal problems; the learning resource contradicts that topic domain. | — |
| lc203-symptom-bug-descriptor-operator-mismatch | doc_code_drift | curricula/cp_accelerator/patterns/linked_list/problems/lc_203/bugs/skip_consecutive_symptom.txt:4 | The symptom file title says '# Bug Symptom: Wrong Comparison Operator' and describes the bug as replacing '!=' with '>'. The bug descriptor JSON (skip_consecutive.json inventory) is named 'skip_consecutive' implying it causes consecutive elements to be skipped. The symptom description at line 4 says 'removes too many elements — not just the target value, but also all elements smaller than or equal to it.' For val=6 with '>' operator, only elements >6 are kept, so all elements ≤6 are removed — the symptom correctly states output '[]'. However, the file name 'skip_consecutive_symptom.txt' implies 'skipping consecutive duplicates' (a different class of bug), while the actual bug is a wrong comparison operator removing non-target elements. The name-to-content drift may mislead developers reading the bug inventory. | — |
| cross-entropy-build-prompt-wrong-test-name | doc_code_drift | curricula/cs336_a1/modules/cross_entropy/build_prompt.txt:98 | build_prompt.txt line 98 says 'The validator will run pytest tests/test_nn_utils.py::test_cross_entropy_matches_pytorch'. The actual validator (cross_entropy/validator.sh:34) runs 'tests/test_nn_utils.py::test_cross_entropy'. The actual test function in tests/test_nn_utils.py is 'def test_cross_entropy():' at line 27, not 'test_cross_entropy_matches_pytorch'. The validator.sh is correct; the build_prompt documents the wrong test name. | — |
| gradient-clipping-build-prompt-wrong-test-name | doc_code_drift | curricula/cs336_a1/modules/gradient_clipping/build_prompt.txt:105 | build_prompt.txt line 105 says 'The validator will run pytest tests/test_nn_utils.py::test_gradient_clipping_matches_pytorch'. The actual validator (gradient_clipping/validator.sh:34) runs 'tests/test_nn_utils.py::test_gradient_clipping'. The actual test function in tests/test_nn_utils.py is 'def test_gradient_clipping():' at line 62, not 'test_gradient_clipping_matches_pytorch'. The validator.sh is correct; the build_prompt documents the wrong test name. | — |
| hello-world-wrong-cli-flag-syntax | doc_code_drift | curricula/dummy_hello_world/modules/hello_world/build_prompt.txt:18 | build_prompt.txt line 18 shows 'engine --submit-build' (treating it as a flag with double-dash). The correct CLI invocation is 'engine submit-build' (subcommand without dashes), as confirmed by the engine architecture (engine/main.py:1354 where submit-build is a Typer CLI subcommand). The other build_prompts in the codebase consistently use 'engine submit-build' (e.g., cosine_schedule/build_prompt.txt:381). The '--' prefix is incorrect CLI syntax for this tool. | — |
| softmax-v2-symptom-identical-to-v1 | doc_code_drift | curricula/cs336_a1/modules/softmax/bugs/no_subtract_max_v2_symptom.txt:1 | no_subtract_max_v2_symptom.txt is byte-for-byte identical to no_subtract_max_symptom.txt (both read: '# Bug Symptom: Numerical Overflow in Softmax' with the same failing test case, error message, debugging tips, and hint). The understanding.json describes v2 as 'matching the v2 injection spec' and the v2 spec (no_subtract_max_v2.json) uses 'an alternative two-pass find_and_track then find_and_replace pattern'. Despite differing injection mechanics, both symptom files are completely identical — v2 provides no distinct harden-stage guidance, which is inconsistent with the intent of having two separate symptom files. | — |
| readme-bug-injection-guide-path | doc_code_drift | docs/README.md:67 | docs/README.md:67 references BUG_INJECTION_GUIDE.md using path `docs/internal/current/BUG_INJECTION_GUIDE.md`. CLEANUP_SUMMARY.md describes the canonical operational docs structure as placing current guides under `docs/current/` (not `docs/internal/current/`). Path divergence unverified via directory listing — marking as low-severity because the actual file location could be either path; cannot confirm without a filesystem check. | — |
| repo-analysis-mislocated-in-architecture-dir | doc_code_drift | docs/architecture/REPO_ANALYSIS.md:1 | REPO_ANALYSIS.md (generated 2025-09-15 per its header) is a CS336 Assignment 1 course analysis document that describes implementing `tests/adapters.py` and course exercise solutions — it analyzes the Stanford CS336 homework, not the Mastery Engine. It is placed in `docs/architecture/` alongside genuine engine architecture documents (MASTERY_ENGINE.md, AI_CODEBASE_DECONSTRUCTION.md), creating a misleading co-location. AI_CODEBASE_DECONSTRUCTION.md:3 distinguishes between 'design analysis/blueprint' and 'shipped features'; REPO_ANALYSIS.md belongs in neither category. | — |
| engine-critical-fixes-deleted-path-ref | doc_code_drift | docs/internal/ENGINE_CRITICAL_FIXES_2025-11-18.md:19 | ENGINE_CRITICAL_FIXES_2025-11-18.md:19 references `curricula/cp_accelerator/modules/sorting/test_cases.json` as the bug location that was fixed (wrong test cases in sorting module). PHASE_8_BATCH_GENERATION_COMPLETE.md:113 states 'Deleted: curricula/cp_accelerator/modules/' — the entire `modules/` directory was subsequently removed and replaced with `patterns/`. This archived fix document describes a path that no longer exists on disk. | — |
| critical-bug-count-mismatch | doc_code_drift | docs/internal/archive/sessions/2025-11-10_bug_system/CRITICAL_BUG_RESOLUTION.md:7 | CRITICAL_BUG_RESOLUTION.md line 7 header states '10 out of 22 modules (45% of curriculum)' were affected by the critical bug. The evidence table on lines 78-86 lists only 9 distinct module indices: 1, 2, 3, 11, 15, 16, 17, 20, 21. The heading count (10) does not match the table row count (9). | — |
| pattern-matcher-py37-wrong-version | doc_code_drift | docs/internal/archive/sessions/2025-11-10_bug_system/PATTERN_MATCHER_DEBUG_SESSION.md:33 | PATTERN_MATCHER_DEBUG_SESSION.md line 33 documents a fix with rationale 'Python 3.7 lacks ast.unparse() — Fix: Fallback to astor.to_source()'. The project's pyproject.toml specifies requires-python = '>=3.11', making the Python 3.7 justification factually incorrect. ast.unparse() was added in Python 3.9; since the project requires 3.11+, no fallback to astor is ever needed and the stated rationale documents a non-existent constraint. | — |
| final-status-bpe-doc-wrong-session | doc_code_drift | docs/internal/archive/sessions/2025-11-09_verification/VERIFICATION_PROTOCOL_FINAL_STATUS.md:174 | VERIFICATION_PROTOCOL_FINAL_STATUS.md line 174 lists 'BPE_TEST_FIX_SUMMARY.md' as an artifact produced in the 2025-11-09_verification session. The actual file resides at docs/internal/archive/sessions/2025-11-11_curriculum_quality/BPE_TEST_FIX_SUMMARY.md — a different session folder dated two days later. The artifact provenance in the final status document is incorrect. | — |
| verification-findings-6-vs-7-modules | doc_code_drift | docs/internal/archive/sessions/2025-11-11_curriculum_quality/VERIFICATION_FINDINGS.md:456 | VERIFICATION_FINDINGS.md:456 says 'Status: ALL 6 NEW MODULES VERIFIED ✅'. The summary table at lines 428-436 in the same document lists 7 modules as verified (rope, linear, embedding, tokenizer_class, transformer_lm, data_loader, checkpointing). The status footer claims 6 but the table body contains 7 rows, all marked ✅ Verified. | — |
| DCD-007 | doc_code_drift | docs/internal/development/MASTERY_WORKLOG.md:42 | MASTERY_WORKLOG.md uses 'engine' as the CLI command name throughout: lines 42-43 ('Ran `engine next`', 'Ran `engine submit-fix`'), lines 681-684 ('`engine next` handles all 3 stages', '`engine submit-build`', '`engine submit-justification`', '`engine submit-fix`'), lines 758-760, and line 787 ('Commands Verified: `engine next`, `engine submit-build`, `engine status`'). pyproject.toml registers the console script as 'mastery', not 'engine'. Every documented CLI invocation in the worklog uses the wrong command name and would fail as written. | — |
| DCD-008 | doc_code_drift | docs/internal/archive/sessions/2025-11-12_test_coverage/TEST_COVERAGE_IMPROVEMENT_SESSION.md:3 | Document header states the session date as 'November 13, 2025' while all other files in the same session directory (FINAL_SESSION_REPORT.md, COVERAGE_70_80_ACHIEVEMENT.md, COVERAGE_80_ACHIEVEMENT.md, CURRENT_REPORT.md) consistently record the date as 2025-11-12. The header date is off by one day from all corroborating artefacts. | — |
| DCD-009 | doc_code_drift | docs/internal/archive/sessions/2025-11-12_test_coverage/FINAL_SESSION_REPORT.md:143 | FINAL_SESSION_REPORT.md lists 13 engine package modules in its coverage summary table. COVERAGE_70_80_ACHIEVEMENT.md (same session, same date) lists 12 modules in its equivalent table. The discrepancy is within the same session's own artefacts; one of the two authoritative session-end documents has a module count error. | — |
| DCD-010 | doc_code_drift | docs/internal/archive/sessions/2025-11-12_test_coverage/FINAL_SESSION_REPORT.md:121 | FINAL_SESSION_REPORT.md records engine/validator.py coverage as 93%. COVERAGE_70_80_ACHIEVEMENT.md (same session directory) records engine/validator.py at 94%. The two end-of-session summary documents report different final coverage values for the same module, creating an internally inconsistent record. The current authoritative report (docs/internal/coverage/CURRENT_REPORT.md:32) states 94%, suggesting FINAL_SESSION_REPORT.md carries a stale/incorrect figure. | — |
| mvp-status-test-count-internal-inconsistency | doc_code_drift | docs/internal/development/MVP_COMPLETION_STATUS.md:136 | MVP_COMPLETION_STATUS.md line 136 states "Total Test Count: 145 engine tests + 22 integration/e2e = 167 automated tests" but the same document at line 274 (Final Validation section) shows `uv run pytest → ✅ 72/72 tests passing in ~30 seconds`. A grep of test function definitions across all test files yields 72 `def test_` functions. The document presents two irreconcilable test counts (167 vs 72) in the same production-readiness artifact. | — |
| module-gen-docs-problem-count-874-vs-959 | doc_code_drift | docs/internal/module_generation/MODULE_GENERATION_COMPREHENSIVE_SUMMARY.md:9 | MODULE_GENERATION_COMPREHENSIVE_SUMMARY.md line 9 states the system "scales to all 874 problems in our curriculum". MODULE_GENERATION_PROGRESS.md and MODULE_GENERATION_REFACTORING_PLAN.md repeat the same 874 figure. Counting problems via `sum(len(t['problems']) for t in canonical_curriculum.json['topics'])` yields 959, not 874 — 9.7% more than documented. The count discrepancy persists across multiple doc files indicating the curriculum was expanded after these documents were authored. | — |
| e2e-test-status-8-commands-vs-14 | doc_code_drift | tests/e2e/E2E_TEST_STATUS.md:9 | E2E_TEST_STATUS.md line 9 states "All 8 engine commands fully tested". The MASTERY_COMMAND_REFERENCE.md (same Nov 2025 period) documents 14 commands (9 primary + 4 deprecated + 1 dev tool), which matches the 14 `@app.command()` decorators counted in engine/main.py. The 8-command count in E2E_TEST_STATUS.md is a stale figure from an earlier development phase; the gap means at least 6 commands were added with no corresponding E2E coverage claim update. | — |
| readme-wrong-fixture-name-in-example | doc_code_drift | tests/integration/README.md:141 | The 'Adding New Integration Tests' code example on line 141 shows `def test_new_llm_feature(check_api_key, softmax_questions):`. The fixture `softmax_questions` does not exist in `test_llm_service.py`; the actual reusable fixture is `sample_question` (defined at line 37 of `test_llm_service.py`). A developer following this example would get a pytest fixture-not-found error. | — |
| validate-push-trigger-missing-script-path | design_defect | .github/workflows/validate_cp_manifest.yml:8 | validate_cp_manifest.yml on.pull_request.paths (lines 5-8) includes both `curricula/cp_accelerator/**` and `scripts/generate_manifest.py`. But on.push.paths (lines 10-14) only includes `curricula/cp_accelerator/**` — `scripts/generate_manifest.py` is absent. A direct push to main that modifies only `scripts/generate_manifest.py` (bypassing PR) will NOT trigger the manifest integrity check, allowing a broken generator to land without validation. | — |
| inconsistent-boolean-encoding | bug | curricula/cp_accelerator/patterns/hash_table/problems/lc_217/test_cases.json:17 | lc_217/test_cases.json and stack_and_queue/lc_1003/test_cases.json encode boolean expected values as JSON strings: `"expected": "true"` and `"expected": "false"`. In contrast, stack_and_queue/lc_20/test_cases.json correctly uses JSON booleans: `"expected": true` and `"expected": false`. If the validator compares Python boolean `True` against string `"true"`, the comparison fails silently (`True != "true"`). This is a data schema inconsistency that would cause spurious test failures for lc_217 (Contains Duplicate) and lc_1003 (Check If Word Is Valid After Substitutions). | — |
| numpy-completely-unpinned | security | pyproject.toml:11 | `numpy` is listed with zero version constraint — no lower bound, no upper bound. Any numpy major version (including future 3.x with breaking API changes) will be accepted. Combined with `>=` lower-bound-only constraints on most other deps, the effective lockfile is entirely resolver-determined. Without a pinned lockfile committed to the repo, reproducible builds are impossible. | — |
| stale-action-versions-validate-workflow | design_defect | .github/workflows/validate_cp_manifest.yml:21 | validate_cp_manifest.yml uses `actions/checkout@v3` (line 21) and `actions/setup-python@v4` (line 25), while tests.yml uses `checkout@v4` and `setup-python@v5`. Workflows in the same repo are running on different action versions, creating inconsistent build environments and indicating stale maintenance. Older action versions may also lack security patches. | — |
| lint-continue-on-error-silences-failures | design_defect | .github/workflows/tests.yml:85 | Lines 85 and 89 both set `continue-on-error: true` for the ruff lint check and ruff format check steps respectively. This means lint failures never block a merge. The CI workflow will report green even if code has linting violations, defeating the purpose of having CI lint enforcement. | — |
| manifest-module-count-mismatch | doc_code_drift | curricula/cs336_a1/manifest.json:6 | Line 6: `"description": "CS336 Assignment 1 — 22 modules in dependency order"` — wait, the description reads `"21 modules"` but the `modules` array contains 22 entries: softmax, cross_entropy, gradient_clipping, linear, embedding, silu, rmsnorm, swiglu, attention, rope, multihead_attention, transformer_block, transformer_lm, adamw, cosine_schedule, data_loader, checkpointing, training_loop, unicode, bpe_tokenizer, tokenizer_class, text_generation. Confirmed by `python3 -c "import json; m=json.load(open('curricula/cs336_a1/manifest.json')); print(len(m['modules']))"` → 22. | — |
| cross-entropy-variable-name-mismatch | doc_code_drift | curricula/cs336_a1/modules/cross_entropy/bugs/no_logsumexp.json | Production `no_logsumexp.json` targets a variable named `log_sum_exp` (pattern `id: "log_sum_exp"`). Draft `no_logsumexp_draft.json` targets a variable named `lse`. These two specs cannot both be correct for the same implementation: whichever name the student's code uses, one spec will silently fail to match and not inject the bug. This mismatch indicates either the production or draft spec targets the wrong variable name. | — |
| pyproject-ty-alpha-dependency | design_defect | pyproject.toml:22 | Dependency '"ty>=0.0.1a16"' declares a pre-release alpha package with only a lower-bound constraint and no upper bound. The 'ty' type checker is in active alpha development (uv.lock records 0.0.1a19). Alpha packages by semantic versioning convention carry no stability guarantee; CLI flags, APIs, and output format can change in any release. Using an alpha package as a production dependency without an upper bound or exact pin may cause sudden breakage on any lock file update. | — |
| ci-lint-continue-on-error | design_defect | .github/workflows/tests.yml:85 | Lines 85 and 88 both use 'continue-on-error: true' for the ruff linter and ruff formatter checks respectively: 'uv run ruff check engine/ tests/ --output-format=github' and 'uv run ruff format engine/ tests/ --check'. This means lint violations and formatting failures are logged but do not cause the CI job to fail. Code with linting errors or inconsistent formatting will merge without any CI gate. | — |
| draft-bug-spec-pass-field-inconsistency | doc_code_drift | curricula/cs336_a1/modules/training_loop/bugs/missing_zero_grad_draft.json:9 | missing_zero_grad_draft.json:9 uses field name '"pass_": 1' and missing_residual_draft.json:10 also uses '"pass_": 1', while the corresponding v2 drafts and production specs use '"pass": 1' (missing_zero_grad_draft_v2.json:9, missing_residual_draft_v2.json:9, missing_residual.json:9). The engine does not currently read draft files directly, but the divergent field name ('pass_' vs 'pass') documents a schema evolution that was applied inconsistently across the draft corpus. Any tooling that processes draft files alongside production specs would encounter this mismatch. | — |
| ci-checkout-version-inconsistency | design_defect | .github/workflows/validate_cp_manifest.yml:22 | validate_cp_manifest.yml:22 uses 'actions/checkout@v3' while tests.yml:15 uses 'actions/checkout@v4'. The two workflows use different major versions of the same action. checkout@v3 and @v4 differ in Node.js runtime (Node 16 vs Node 20) and underlying behavior, meaning the two CI workflows operate in subtly different environments. This inconsistency makes it harder to reason about CI environment parity. | — |
| temperature-after-softmax-comment-inaccurate | doc_code_drift | curricula/cs336_a1/modules/text_generation/bugs/temperature_after_softmax.patch:15 | Injected comment reads `# Wrong! Temperature after softmax has no effect!` but `probs / temperature` does change probability magnitudes — it just fails to produce a valid sharpened/flattened distribution because temperature scaling must be applied to logits before softmax, not to probabilities after. The claim 'no effect' is factually wrong; the real problem is that the resulting values no longer sum to 1 and the effect on sampling is distorted rather than absent. The comment misleads students about the actual failure mode they are expected to diagnose. | — |
| draft-specs-use-pass-underscore-key | design_defect | curricula/cs336_a1/modules/checkpointing/bugs/missing_optimizer_state_draft.json | Three draft specs use `"pass_": N` instead of `"pass": N` in logic entries: missing_optimizer_state_draft.json, multihead_attention/missing_transpose_back_draft.json, and swiglu/missing_gate_draft.json. Production specs use `"pass": N`. PassDefinition in schemas.py uses `Field(..., alias='pass')` with `populate_by_name = True` (line 318-327), which accepts both forms via Pydantic. However, GenericBugInjector at generic_injector.py:124 accesses pass_def as a raw dict and reads `pass_def['type']` — the pass number key is never read by the injector, so this inconsistency has no runtime effect currently. It creates authoring confusion about the canonical key name. | — |
| genericinjector-softmax-fallback-hardcoded | design_defect | engine/ast_harden/generic_injector.py:109 | `Canonicalizer(target_function=target_function if target_function else 'softmax')` — when target_function is None or empty string, the canonicalizer defaults to 'softmax' as a hardcoded fallback. This is a development artifact; for bug specs without a target_function, the AST would be canonicalized around the 'softmax' function scope rather than any relevant function. This could cause the canonicalization to operate on the wrong AST subtree, potentially failing to find patterns in non-softmax functions. The correct fallback would be None or raise an error. | — |
| scripts-mode-eval-shell-injection | security | scripts/mode:136 | `eval "$test_cmd"` in test_in_mode() where test_cmd comes from CLI arguments `"${@:3}"` (line 157). Passing shell metacharacters or command substitution in the test argument (e.g., `./scripts/mode test student '$(rm -rf /tmp/x)'`) would execute arbitrary shell commands. This is a developer-only script, reducing exploitability, but the pattern is a textbook shell injection risk if the script were ever called from a less-trusted context or CI pipeline with user-controlled inputs. | — |
| coverage-reports-redundant-partial-scope | doc_code_drift | docs/internal/coverage/reports/coverage_with_new_cli_tests.txt:1 | coverage_with_new_cli_tests.txt records only engine/main.py at 834 stmts, 432 miss, 48%. The engine/main.py row in coverage_final_phase2.txt is identical (834 stmts, 432 miss, 48%). The standalone file is therefore a redundant partial-scope snapshot duplicating one row from the full-project report. Additionally, coverage_report_main_final.txt and coverage_report_main_partial.txt both cover only engine/main.py (690 stmts) with no project-wide data, while coverage_final_phase2.txt is the only file containing project-wide metrics (59% overall across engine/, tests/). The naming convention implies a temporal progression toward 'final' state that is not consistent with the actual scope of each file — coverage_report_main_final.txt reflects 28% for an older 690-statement main.py build, while coverage_final_phase2.txt shows 48% for a 834-statement main.py, giving the false impression of regression if read without careful scope comparison. | — |

## INFO (9)
| ID | Class | Location | Evidence | Adversarial note |
| --- | --- | --- | --- | --- |
| s-unofficial-leetcode-api | security | scripts/enrich_problems.py:39 | API_BASE = "https://leetcode-api-pied.vercel.app"  — all problem metadata and HTML content is fetched from an unofficial third-party proxy, not the official LeetCode API. The proxy operator can serve arbitrary content, including malformed HTML that feeds downstream regex at lines 159-208 and ast.literal_eval calls. | — |
| s-jinja2-no-autoescape | security | scripts/generate_module.py:302 | env = Environment(loader=FileSystemLoader(str(template_dir)))  — Jinja2 Environment is constructed without autoescape=True. If any template variable contains HTML/script content (e.g., problem titles from the LeetCode proxy), it will be rendered verbatim into generated files. The generated artifacts are shell scripts and Python files (not HTML), so XSS is not directly applicable, but injected newlines or shell metacharacters in problem titles could corrupt generated validator.sh scripts. | — |
| bpe-exact-merge-check-commented-out | design_defect | tests/test_train_bpe.py:54 | Line 54 shows `# assert merges == reference_merges  # Too strict - commented out`. The strict, deterministic correctness check was replaced with a loose range assertion: `assert len(merges) >= 243 and len(merges) <= 245` (lines 57-58) and ≥98% vocabulary coverage (lines 75-78). The comment on lines 50-53 attributes this to tie-breaking divergence between implementations. As a result, any BPE implementation producing 243-245 merges with ≥98% byte coverage will pass even if the merge ordering is substantially incorrect. The weakened assertion is intentionally documented in comments, so this is not a hidden issue, but it represents a gap between the documented intent (verify BPE training produces reference-compatible results) and the actual test coverage. | — |
| torch-load-without-weights-only-in-serialization-test | security | tests/test_serialization.py:101 | Line 101: `loaded_iterations = run_load_checkpoint(src=serialization_path, model=new_model, optimizer=new_optimizer)`. The `run_load_checkpoint` adapter (tests/adapters.py:571-589) delegates to `_load_ckpt_impl` from `cs336_basics.utils`. PyTorch's `torch.load()` uses pickle serialization by default; without `weights_only=True`, loading a crafted checkpoint file can execute arbitrary Python code at deserialization time (CVE-class pickle RCE). In this specific test, the checkpoint at `serialization_path` is written by `run_save_checkpoint` within the same test run from `tmp_path`, so no untrusted file is loaded and there is no immediate exploit. The risk is unverified (the `cs336_basics.utils` implementation is outside the 8 audited files), but the pattern is flagged because if the same loader function is reused in contexts where `src` is supplied externally (e.g., model sharing, student-submitted checkpoints), arbitrary code execution would be possible. | — |
| lc912-bug-placeholder-files-empty | bug | curricula/cp_accelerator/patterns/sorting/problems/lc_912/bugs/incomplete_merge.py:1 | Three bug-variant placeholder files are empty (a single whitespace line each, confirmed by 'Warning: the file exists but is shorter than the provided offset (1)' on Read): `bugs/incomplete_merge.py`, `bugs/missing_base_case.py`, `bugs/off_by_one.py`. Per the understanding document these are intentional placeholders to be populated by the bug-injection engine at exercise time. However, if the injection engine fails or a validator invokes these files directly, it will import an empty module (no `sortArray` or `solve` symbol), causing `ImportError` with a silent code path rather than an actionable error. No current test guards against empty placeholders before injection. | — |
| class-based-lc146-460-703-307-functional-validator | intent_mismatch | curricula/cp_accelerator/patterns/design_patterns/problems/lc_146/validator.sh:35 | UNVERIFIED (solution.py files for lc_146, lc_460, lc_703, lc_307 were not in the audit file list and were not read). LeetCode 146 (LRU Cache), 460 (LFU Cache), 703 (Kth Largest in Stream), and 307 (Range Sum Query Mutable) are class-based problems requiring `__init__` plus multiple methods. All four validators call `result = solve(**test["input"])` assuming a pure functional interface. If the solution files wrap the class in a stateless `solve` function, the test data format must encode the sequence of operations, which is non-standard. Without reading those solution.py and test_cases.json files, it cannot be confirmed whether the design is coherent or broken. | — |
| DCD-011 | doc_code_drift | docs/internal/archive/sessions/2025-11-12_test_coverage/COVERAGE_80_ACHIEVEMENT.md:1 | Document title is 'Coverage Achievement: 76% → 78% ✅ 80% THRESHOLD REACHED'. Every data point in the body reports 78% (line 5: '78% ENGINE COVERAGE', line 11: '78% total coverage', line 15: '78% total engine coverage', line 19: 'Target exceeded: 78% ≈ 80% goal', line 64: 'TOTAL ... 78%'). The title's '80% THRESHOLD REACHED' claim is contradicted by all internal data; the actual measured coverage is 78%, which the document itself characterises as 'Near 80%' and 'approximately' 80% — not as having crossed the 80% threshold. | — |
| wrong-pointer-move-json-future-created-date | doc_code_drift | curricula/cp_accelerator/patterns/two_pointers/problems/lc_167/bugs/wrong_pointer_move.json | The metadata.created field value is '2026-03-04', which is in the future relative to all other files in the repository (dated 2025). All other bug spec metadata fields examined use 2025 dates. This is a data-entry inconsistency in the metadata, likely a copy-paste or manual entry error. Does not affect runtime behavior. | — |
| production-specs-engine-version-2-0 | design_defect | curricula/cs336_a1/modules/rmsnorm/bugs/missing_keepdim.json:4 | Two files in production use (`missing_keepdim.json` and `silu/missing_multiply.json`) declare `"engine_version": "2.0"` while the majority of production specs use `"2.1"`. GenericBugInjector does not check or branch on engine_version (validate_definition() at generic_injector.py:42-50 ignores it), so there is no current runtime difference. The inconsistency creates ambiguity about whether these specs were intended to be promoted to v2.1 and what behavioral differences between versions exist. | — |

---
### machine-readable artifact
```json
{
  "findings": [
    {
      "id": "missing-cd-script-dir",
      "location": "curricula/cp_accelerator/patterns/backtracking/problems/lc_78/validator.sh:7",
      "class": "bug",
      "severity": "medium",
      "evidence": "SCRIPT_DIR is computed at line 7 (`SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"`) but `cd \"$SCRIPT_DIR\"` is never executed before the Python heredoc at line 17. Inside the heredoc, `open(\"test_cases.json\")` resolves relative to the caller's CWD, and `sys.path.insert(0, str(Path(__file__).parent))` inserts `'.'` (CWD) because `__file__` is `'<stdin>'` in heredoc context (verified: `Path('<stdin>').parent` = `.`). If the validator is invoked from any directory other than the problem directory, both the solution import and the test-cases load fail. The lc_1 validator (validator.sh:17) correctly does `cd \"$SCRIPT_DIR\"` before its heredoc; the same fix is absent from all 51 other cp_accelerator validators (lc_78, lc_90, lc_34, lc_704, lc_1342, lc_1486, lc_46, lc_47, lc_146, lc_460, lc_148/divide_and_conquer, lc_912/divide_and_conquer, lc_198, lc_70, lc_435, lc_452, lc_217, lc_219, lc_215, lc_703, lc_203, lc_237, lc_1480, lc_303, lc_307, lc_148/sorting, lc_912/sorting, lc_1003, lc_20, lc_144, lc_589, lc_1804, lc_208, lc_1099, lc_167, lc_547, lc_684)."
    },
    {
      "id": "lc1-unquoted-heredoc",
      "location": "curricula/cp_accelerator/patterns/hash_table/problems/lc_1/validator.sh:18",
      "class": "security",
      "severity": "low",
      "evidence": "The lc_1 validator at line 18 uses an unquoted heredoc delimiter: `${MASTERY_PYTHON:-python3} << EOF`. All 51 other cp_accelerator validators use `<< 'EOF'` (quoted), which suppresses bash parameter expansion, command substitution, and arithmetic expansion inside the heredoc body. With an unquoted delimiter, any `$var`, `$(cmd)`, or backtick construct in the Python code would be expanded by the shell before being passed to Python. The current heredoc body does not contain exploitable shell expansions (Python f-string braces `{...}` are not shell syntax), but the inconsistency is a latent risk: if the heredoc content is ever modified to include `$` characters (e.g., in comments or strings), shell expansion could corrupt or inject code. All other validators correctly use the quoted form."
    },
    {
      "id": "unsandboxed-solution-import",
      "location": "curricula/cp_accelerator/patterns/backtracking/problems/lc_78/validator.sh:24",
      "class": "security",
      "severity": "high",
      "evidence": "All cp_accelerator validators execute learner-supplied Python code via `from solution import solve` (lc_1 uses `from solution import twoSum`). Python module import executes all top-level statements in solution.py unconditionally. There is no timeout, resource limit (CPU/memory), namespace restriction, or syscall filter. A solution.py containing `import os; os.system(\"rm -rf ~\")` or `import subprocess; subprocess.run([...])` at module level would execute with the runner's full OS privileges. This is architecturally expected for a local single-user CLI tool (the learner runs their own code), but the intent document does not acknowledge or accept this risk, and no mitigations are present. The same pattern applies to all 52 validators in the cp_accelerator set."
    },
    {
      "id": "cs336-time-arithmetic-injection",
      "location": "curricula/cs336_a1/modules/adamw/validator.sh:47",
      "class": "security",
      "severity": "low",
      "evidence": "Line 47 of the adamw, attention, and bpe_tokenizer validators: `duration=$(python3 -c \"print($end_time - $start_time)\")`. The variables `$end_time` and `$start_time` are captured from `python3 -c 'import time; print(time.time())'` (lines 27 and 44). They are placed unquoted inside a double-quoted `python3 -c` string, which means bash expands them before the string is passed to Python. If either variable contained shell-special characters or newlines (e.g., if `time.time()` output were somehow tampered), arbitrary Python expressions could be injected into the `-c` argument. In practice `time.time()` always returns a decimal float such as `1699999999.123` with no special characters, making exploitation purely theoretical. The same pattern exists identically in `curricula/cs336_a1/modules/attention/validator.sh:47` and `curricula/cs336_a1/modules/bpe_tokenizer/validator.sh:47`."
    },
    {
      "id": "lc1-import-api-inconsistency",
      "location": "curricula/cp_accelerator/patterns/hash_table/problems/lc_1/validator.sh:27",
      "class": "design_defect",
      "severity": "low",
      "evidence": "Line 27: `from solution import twoSum`. All 51 other cp_accelerator validators use `from solution import solve` (the standardized alias). The lc_1 solution (`solution.py:36`) does define `solve = twoSum`, but the validator bypasses it and imports the problem-specific function name directly. This creates an inconsistency: if a learner follows the project convention of providing a `solve` entry point, the lc_1 validator still requires `twoSum`. Conversely, if the lc_1 solution ever renamed the function (e.g., during a refactor), only this validator would break. The unquoted heredoc (`<< EOF`, not `<< 'EOF'`) is a second divergence from the template used for all other validators."
    },
    {
      "id": "cs336-shadow-worktree-unvalidated-path",
      "location": "curricula/cs336_a1/modules/adamw/validator.sh:18",
      "class": "security",
      "severity": "low",
      "evidence": "Line 18: `cp cs336_basics/optimizer.py \"$SHADOW_WORKTREE/cs336_basics/optimizer.py\"`. The `SHADOW_WORKTREE` variable is supplied entirely by the engine via environment variable (lines 8-11 check it is non-empty but do not validate its value). If the engine or a caller set `SHADOW_WORKTREE` to a path containing `..` components (e.g., `/tmp/../../etc/cron.d`), the `cp` would write to an unintended location. The double-quoting on `\"$SHADOW_WORKTREE\"` protects against word splitting but not path traversal. The same pattern exists in `curricula/cs336_a1/modules/attention/validator.sh:18` (copying layers.py) and `curricula/cs336_a1/modules/bpe_tokenizer/validator.sh:18` (copying tokenizer.py). Since `SHADOW_WORKTREE` is controlled by the trusted engine component, practical exploitation requires engine compromise."
    },
    {
      "id": "ast-injection-source-path",
      "location": "engine/ast_harden/pattern_matcher.py:365",
      "class": "security",
      "severity": "high",
      "evidence": "In _apply_replace_value_with(), source_path (from the JSON bug definition's replacement.source field) is passed directly to ast.parse(source_path, mode='eval') at line 365, and the resulting AST body node is grafted into the student's code AST. After ast.unparse() the expression is written to disk and executed by validator scripts and by the student. A malicious or compromised curriculum JSON can inject arbitrary Python expressions that execute in the student's environment. Same pattern repeats at line 420 for replace_with type."
    },
    {
      "id": "path-traversal-curriculum-id",
      "location": "engine/curriculum.py:105",
      "class": "security",
      "severity": "high",
      "evidence": "curriculum_path = self.CURRICULA_DIR / curriculum_id — no sanitization, normalization, or containment check on curriculum_id before joining it to CURRICULA_DIR. A value like ../../etc resolves outside CURRICULA_DIR. The value comes directly from the user-supplied CLI argument (engine init <curriculum_id>) via engine/main.py."
    },
    {
      "id": "student-source-path-no-bounds-check",
      "location": "engine/stages/harden.py:104",
      "class": "security",
      "severity": "medium",
      "evidence": "student_code_path = Path.cwd() / source_file_path at line 104, where source_file_path comes from module.source_files[0] in the curriculum JSON. No validation that source_file_path resolves within the workspace directory. A curriculum with source_files: ['../../sensitive/config.py'] causes the engine to read and then inject bugs into files outside the intended student workspace."
    },
    {
      "id": "editor-env-subprocess-injection",
      "location": "engine/main.py:310",
      "class": "security",
      "severity": "medium",
      "evidence": "editor = os.getenv('EDITOR', os.getenv('VISUAL', 'nano')) then subprocess.run([editor, temp_path]). While list-form subprocess avoids shell splitting, EDITOR is fully user-controlled; any executable path or interpreter binary (e.g. /usr/bin/env bash, or a local malicious binary) is accepted without validation before being exec'd with the temp file path as an argument."
    },
    {
      "id": "path-traversal-module-id",
      "location": "engine/main.py:490",
      "class": "security",
      "severity": "medium",
      "evidence": "harden_file = harden_workspace / f\"{current_module.id}.py\" and shadow_dest = shadow_worktree / f\"{current_module.id}.py\" — current_module.id comes from curriculum JSON without a containment check. A module id containing ../ (e.g. ../../etc/malicious) causes writes outside the harden workspace or shadow worktree."
    },
    {
      "id": "apply-patch-unvalidated-patch-file",
      "location": "engine/workspace.py:155",
      "class": "security",
      "severity": "medium",
      "evidence": "subprocess.run([\"patch\", str(target_file), str(patch_file)], ...) — apply_patch() is called from engine/stages/harden.py:156,320 with bug_file (curriculum-controlled path) as patch_file. No validation that patch_file lies within the curriculum directory. A crafted diff with --- / +++ headers pointing to other paths could modify files outside the intended target; pointing patch_file at a non-patch file can also corrupt target_file arbitrarily."
    },
    {
      "id": "mock-mode-justify-bypass",
      "location": "engine/services/llm_service.py:59",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "When OPENAI_API_KEY is absent, use_mock = True is set at line 62. evaluate_justification() at lines 109-122 unconditionally returns LLMEvaluationResponse(is_correct=True, feedback='MOCK MODE...') regardless of the student's answer. The stated intent of the justify stage is to verify student understanding before advancing; mock mode silently grants a pass to any answer including a blank string, bypassing the entire evaluation gate."
    },
    {
      "id": "ssrf-httpbin-external-dependency",
      "location": "curricula/job_prep_data_annotation/modules/http_transport/validator.sh:27",
      "class": "security",
      "severity": "medium",
      "evidence": "Validator makes live outbound HTTP requests to https://httpbin.org/html (line 27) and https://httpbin.org/status/404 (line 35) during test execution. Fails non-deterministically when the external service is unavailable; always fails in air-gapped or firewalled environments; the student's IP and environment metadata (User-Agent, TLS fingerprint) are sent to a third-party service at validation time without explicit consent."
    },
    {
      "id": "bare-except-swallows-sigint",
      "location": "engine/main.py:2007",
      "class": "bug",
      "severity": "low",
      "evidence": "except: at line 2007 (inside the init command's state-file pre-check block) with comment 'State file doesn't exist or is corrupt - treat as fresh init'. A bare except catches KeyboardInterrupt and SystemExit in addition to Exception, preventing clean termination when the user presses Ctrl+C during this block of the init command."
    },
    {
      "id": "log-injection-curriculum-id",
      "location": "engine/state.py:53",
      "class": "security",
      "severity": "low",
      "evidence": "logger.info(f\"Loaded progress: curriculum={progress.curriculum_id}, ...\") at line 53 — curriculum_id is persisted from the user-supplied CLI argument and interpolated directly into the log message without sanitization. A value containing newlines (\\n) or ANSI escape codes can forge additional log entries or corrupt structured log parsing downstream."
    },
    {
      "id": "s-tmp-read-golden",
      "location": "scripts/add_successful_to_golden.py:14",
      "class": "security",
      "severity": "medium",
      "evidence": "results_path = Path(\"/tmp/llm_evaluation_results.json\")  — the script reads evaluation results from /tmp, a world-writable directory. A local attacker on the same machine can plant a crafted JSON at that path before the script runs to promote arbitrary bug specs into the curriculum golden set. The file is then used at lines 19-70 to select which bugs are written into production curriculum directories."
    },
    {
      "id": "s-hardcoded-path-golden",
      "location": "scripts/add_successful_to_golden.py:77",
      "class": "security",
      "severity": "low",
      "evidence": "golden_dir = Path(f\"/Volumes/Totallynotaharddrive/assignment1-basics/curricula/cs336_a1/modules/{module}/bugs\")  — hardcoded developer-machine external-drive path leaks filesystem layout including volume name and internal project structure. If this script is run on any machine other than the original developer's, it silently resolves to a non-existent path and will fail or write to an unintended location."
    },
    {
      "id": "s-hardcoded-path-auto-fix",
      "location": "scripts/auto_fix_drafts.py:218",
      "class": "security",
      "severity": "low",
      "evidence": "base_path = Path(\"/Volumes/Totallynotaharddrive/assignment1-basics/curricula/cs336_a1/modules\")  — same developer-machine external-drive path. Script is otherwise a production curriculum maintenance tool but will fail silently on any other machine."
    },
    {
      "id": "s-hardcoded-path-fix-draft",
      "location": "scripts/fix_draft_pattern.py:71",
      "class": "security",
      "severity": "low",
      "evidence": "base_path = Path(\"/Volumes/Totallynotaharddrive/assignment1-basics/curricula/cs336_a1/modules\")  — identical hardcoded /Volumes/Totallynotaharddrive path baked into a curriculum batch-fix script."
    },
    {
      "id": "s-hardcoded-path-ground-truth",
      "location": "scripts/generate_ground_truth.py:22",
      "class": "security",
      "severity": "low",
      "evidence": "base_path = Path(\"/Volumes/Totallynotaharddrive/assignment1-basics/curricula/cs336_a1/modules\")  — same developer-machine volume path in a script that reads reference solutions and writes golden ground-truth files."
    },
    {
      "id": "s-hardcoded-path-verify",
      "location": "scripts/verify_ground_truth.py:19",
      "class": "security",
      "severity": "low",
      "evidence": "base_path = Path(\"/Volumes/Totallynotaharddrive/assignment1-basics/curricula/cs336_a1/modules\")  — same hardcoded external-volume path in the verification script."
    },
    {
      "id": "b-missing-import-re",
      "location": "scripts/enrich_problems.py:159",
      "class": "bug",
      "severity": "high",
      "evidence": "re.findall(..., re.DOTALL) is called at line 159 (and again at 168, 173, 178, 192) but `import re` is absent from the module. The only import near the top is `import requests` inside a try-block at line 29. Executing any code-path that reaches line 159 raises NameError: name 're' is not defined, crashing the enrichment pipeline completely."
    },
    {
      "id": "s-unofficial-leetcode-api",
      "location": "scripts/enrich_problems.py:39",
      "class": "security",
      "severity": "info",
      "evidence": "API_BASE = \"https://leetcode-api-pied.vercel.app\"  — all problem metadata and HTML content is fetched from an unofficial third-party proxy, not the official LeetCode API. The proxy operator can serve arbitrary content, including malformed HTML that feeds downstream regex at lines 159-208 and ast.literal_eval calls."
    },
    {
      "id": "b-module-level-open-ellipsis",
      "location": "modes/student/cs336_basics/pretokenization_example.py:53",
      "class": "bug",
      "severity": "medium",
      "evidence": "with open(..., \"rb\") as f:  — the Ellipsis literal (...) is used as the filename argument at module scope (not inside if __name__ == '__main__'). Python evaluates this on import, raising TypeError: expected str, bytes or os.PathLike object, not ellipsis. Any test or tool that imports this module will crash immediately."
    },
    {
      "id": "s-pickle-load-fixtures",
      "location": "tests/conftest.py:140",
      "class": "security",
      "severity": "medium",
      "evidence": "expected_data = pickle.load(f)  — snapshot fixture files (.pkl) are deserialized with pickle.load without any integrity check (HMAC, hash, or signature). Pickle deserialization of untrusted data executes arbitrary Python during unpickling. If a contributor or CI runner checks out a branch containing a tampered .pkl file, the test suite becomes an attack vector."
    },
    {
      "id": "s-torch-load-no-weights-only",
      "location": "tests/conftest.py:199",
      "class": "security",
      "severity": "medium",
      "evidence": "state_dict = torch.load(FIXTURES_PATH / \"ts_tests\" / \"model.pt\", map_location=\"cpu\")  — torch.load without weights_only=True (added in PyTorch 1.13) deserializes the file using pickle, allowing arbitrary code execution if the .pt file is malicious. PyTorch now emits a FutureWarning for this usage and recommends weights_only=True."
    },
    {
      "id": "s-ast-literal-eval-api-data",
      "location": "scripts/generate_module.py:120",
      "class": "security",
      "severity": "low",
      "evidence": "result[key] = ast.literal_eval(value)  — value comes from parsing the LeetCode API response (via the unofficial proxy at enrich_problems.py:39). ast.literal_eval is safe against code execution but will raise ValueError/SyntaxError on unexpected content; the proxy can return content that causes the curriculum generation pipeline to crash or skip fields silently. Same pattern repeated at line 452."
    },
    {
      "id": "s-jinja2-no-autoescape",
      "location": "scripts/generate_module.py:302",
      "class": "security",
      "severity": "info",
      "evidence": "env = Environment(loader=FileSystemLoader(str(template_dir)))  — Jinja2 Environment is constructed without autoescape=True. If any template variable contains HTML/script content (e.g., problem titles from the LeetCode proxy), it will be rendered verbatim into generated files. The generated artifacts are shell scripts and Python files (not HTML), so XSS is not directly applicable, but injected newlines or shell metacharacters in problem titles could corrupt generated validator.sh scripts."
    },
    {
      "id": "b-subprocess-os-environ",
      "location": "tests/e2e/debug_shadow_worktree.py:73",
      "class": "bug",
      "severity": "low",
      "evidence": "env={**subprocess.os.environ, \"PYTHONPATH\": str(shadow_worktree)}  — subprocess.os is not a documented public attribute; accessing os through the subprocess module works only because Python caches the module reference internally. The canonical form is os.environ. If the internal reference is ever removed in a future Python release this will raise AttributeError."
    },
    {
      "id": "i-llm-mock-auto-pass",
      "location": "engine/services/llm_service.py:59",
      "class": "intent_mismatch",
      "severity": "high",
      "evidence": "if not api_key: self.use_mock = True ... (line 60-71); evaluate_justification returns LLMEvaluationResponse(is_correct=True, ...) unconditionally in mock mode (lines 109-119). The PROVISIONAL INTENT explicitly describes the system as an 'LLM-as-evaluator' pedagogical tool. When OPENAI_API_KEY is absent the Justify stage silently auto-passes every answer, completely defeating the mastery-verification purpose. The logger warning at line 65 fires at DEBUG level and is not surfaced to the user at submission time, so learners may never know evaluation was skipped."
    },
    {
      "id": "d-integration-test-wrong-expectation",
      "location": "tests/integration/test_llm_service.py:70",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "test_llm_service_missing_api_key asserts `pytest.raises(ConfigurationError)` when OPENAI_API_KEY is unset (lines 70-83). The actual implementation at llm_service.py:59-71 does not raise ConfigurationError; it silently enters mock mode and returns. This integration test will fail against the current codebase, documenting a behavior contract (raise on missing key) that was either abandoned or never implemented."
    },
    {
      "id": "dead-assert-inside-pytest-raises",
      "location": "tests/test_data.py:72",
      "class": "bug",
      "severity": "high",
      "evidence": "The assertion `assert \"CUDA error\" in str(excinfo.value) or \"Torch not compiled with CUDA enabled\" in str(excinfo.value)` is placed INSIDE the `with pytest.raises((RuntimeError, AssertionError)) as excinfo:` block, after the `run_get_batch(device=\"cuda:99\")` call (lines 66-71). Once `run_get_batch` raises an exception, Python exits the `with` block immediately; line 72 is never reached. The error-message guard is dead code: the test passes regardless of which exception type was raised or what its message contains, silently neutering the validation of the error path."
    },
    {
      "id": "memory-limit-decorator-ineffective-for-generators",
      "location": "tests/test_tokenizer.py:449-455",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "The `_encode_iterable` function at lines 449-455 is decorated with `@memory_limit(int(1e6))` and is a generator function (contains `yield from tokenizer.encode_iterable(iterable)`). The `memory_limit` wrapper (lines 19-36) calls `f(*args, **kwargs)` which, for a generator function, immediately returns a generator object WITHOUT executing the body. The `finally` block then restores `RLIMIT_AS` to its original value before any iteration occurs. When `test_encode_iterable_memory_usage` (lines 421-427) iterates over the returned generator via `for _id in _encode_iterable(tokenizer, f)`, the 1 MB memory cap has already been lifted, so actual encoding memory usage is completely unconstrained. The docstring at line 451 states the intent: 'We place tokenizer.encode_iterable into a separate function so we can limit memory for just this function', but the implementation fails to achieve this intent for generators. The test always passes regardless of memory usage, defeating its purpose."
    },
    {
      "id": "rlimit-as-targets-virtual-not-physical-memory",
      "location": "tests/test_tokenizer.py:24",
      "class": "design_defect",
      "severity": "low",
      "evidence": "The `memory_limit` decorator sets `resource.RLIMIT_AS` (virtual address space limit) via `resource.setrlimit(resource.RLIMIT_AS, (process.memory_info().rss + max_mem, -1))`. `RLIMIT_AS` governs the total virtual address space, not physical (RSS) memory. On a Python process with PyTorch loaded, the virtual address space is commonly 10-100x larger than RSS due to CUDA shared libraries, memory-mapped files, and lazy allocations; setting it to `rss + 1 MB` creates an extremely tight cap. Any C extension within the function that attempts to `mmap` additional virtual memory will receive `ENOMEM` and may raise `MemoryError` or crash the process, even if physical memory is plentiful. The `finally` block at line 34 does correctly restore the prior limits, so the window is limited; but the misidentification of RSS as a proxy for virtual address space makes this mechanism unreliable and potentially destabilising for other threads in the same process."
    },
    {
      "id": "bpe-exact-merge-check-commented-out",
      "location": "tests/test_train_bpe.py:54",
      "class": "design_defect",
      "severity": "info",
      "evidence": "Line 54 shows `# assert merges == reference_merges  # Too strict - commented out`. The strict, deterministic correctness check was replaced with a loose range assertion: `assert len(merges) >= 243 and len(merges) <= 245` (lines 57-58) and ≥98% vocabulary coverage (lines 75-78). The comment on lines 50-53 attributes this to tie-breaking divergence between implementations. As a result, any BPE implementation producing 243-245 merges with ≥98% byte coverage will pass even if the merge ordering is substantially incorrect. The weakened assertion is intentionally documented in comments, so this is not a hidden issue, but it represents a gap between the documented intent (verify BPE training produces reference-compatible results) and the actual test coverage."
    },
    {
      "id": "torch-load-without-weights-only-in-serialization-test",
      "location": "tests/test_serialization.py:101",
      "class": "security",
      "severity": "info",
      "evidence": "Line 101: `loaded_iterations = run_load_checkpoint(src=serialization_path, model=new_model, optimizer=new_optimizer)`. The `run_load_checkpoint` adapter (tests/adapters.py:571-589) delegates to `_load_ckpt_impl` from `cs336_basics.utils`. PyTorch's `torch.load()` uses pickle serialization by default; without `weights_only=True`, loading a crafted checkpoint file can execute arbitrary Python code at deserialization time (CVE-class pickle RCE). In this specific test, the checkpoint at `serialization_path` is written by `run_save_checkpoint` within the same test run from `tmp_path`, so no untrusted file is loaded and there is no immediate exploit. The risk is unverified (the `cs336_basics.utils` implementation is outside the 8 audited files), but the pattern is flagged because if the same loader function is reused in contexts where `src` is supplied externally (e.g., model sharing, student-submitted checkpoints), arbitrary code execution would be possible."
    },
    {
      "id": "lc78-test-order-mismatch",
      "location": "curricula/cp_accelerator/patterns/backtracking/problems/lc_78/validator.sh:38",
      "class": "bug",
      "severity": "high",
      "evidence": "The reference solution `subsets([1,2,3])` produces backtracking-DFS order `[[], [1], [1,2], [1,2,3], [1,3], [2], [2,3], [3]]` (confirmed by running the code), but test_cases.json 'expected' for test 1 is bitmask-enumeration order `[[], [1], [2], [1,2], [3], [1,3], [2,3], [1,2,3]]`. The validator checks `if result == expected:` at line 38 — an order-sensitive exact equality with no normalization. LeetCode 78 explicitly allows any order. Consequence: the correct reference solution fails its own test case 1. Any learner who implements a valid DFS/backtracking approach would also be incorrectly rejected."
    },
    {
      "id": "cp-validators-cwd-fragility",
      "location": "curricula/cp_accelerator/patterns/backtracking/problems/lc_78/validator.sh:23",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "All CP-accelerator validators except lc_1 share this pattern: (1) They set `TEST_CASES=\"$SCRIPT_DIR/test_cases.json\"` (absolute path) on line 8 but the Python heredoc uses `with open(\"test_cases.json\")` — a relative path — on line 27. The $TEST_CASES variable is never read inside the heredoc, making it dead code. (2) `sys.path.insert(0, str(Path(__file__).parent))` on line 23 is used to locate solution.py, but when Python runs from a heredoc `__file__` is `'<stdin>'` (verified: `Path('<stdin>').parent` == `PosixPath('.')`, i.e. CWD). This means both the import of solution.py and the open of test_cases.json silently depend on the calling process's working directory being the problem directory. The lc_1 validator correctly avoids this by doing `cd \"$SCRIPT_DIR\"` before the heredoc and using `os.getcwd()` instead. The pattern repeats in at least 40 validator.sh files covering all non-lc_1 CP problems."
    },
    {
      "id": "lc1-validator-unquoted-heredoc",
      "location": "curricula/cp_accelerator/patterns/hash_table/problems/lc_1/validator.sh:18",
      "class": "bug",
      "severity": "low",
      "evidence": "`${MASTERY_PYTHON:-python3} << EOF` (no quotes around EOF delimiter) enables shell variable/command expansion inside the heredoc body. All other CP validators correctly use `<< 'EOF'` (single-quoted) to suppress expansion. Currently harmless because the heredoc body has no `$` variable references that would be ambiguously expanded, but the pattern is fragile: any future addition of Python f-strings or shell-like syntax inside that heredoc could be silently mangled by the shell before Python sees it."
    },
    {
      "id": "lc303-query-complexity-false",
      "location": "curricula/cp_accelerator/patterns/prefix_sum/problems/lc_303/solution.py:3",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "File header comment reads `Build: O(n), Query: O(1)` but the implementation of `sumRange(nums, left, right)` (lines 7-29) unconditionally rebuilds a full `prefix` array of length `n+1` on every call: `prefix = [0] * (len(nums) + 1)` followed by a full loop `for i in range(len(nums))`. The prefix array is a local variable discarded after each return, so every query costs O(n), not O(1). The entire pedagogical point of LeetCode 303 is to build the prefix array once in `__init__` and answer queries in O(1). The implementation also departs from the class-based `NumArray` interface LeetCode specifies; this was intentional (functional shim noted in test_cases.json), but the O(1) claim in the header comment is plainly wrong for the functional form."
    },
    {
      "id": "lc203-list-comprehension-bypasses-linked-list",
      "location": "curricula/cp_accelerator/patterns/linked_list/problems/lc_203/solution.py:23",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "LeetCode 203 (Remove Linked List Elements) is a linked-list pointer-manipulation exercise — the teaching intent is to practice sentinel-node / pointer-update patterns. The reference solution at line 23 is `return [x for x in head if x != val]`, a single-pass Python list comprehension that works only because the test runner serialises the linked list as a plain Python list. A learner who copies or internalises this pattern and submits to LeetCode (which uses `ListNode` objects) will receive a wrong-answer or runtime error. The note at line 4 says 'Uses array representation for compatibility with test runner', acknowledging the departure, but the code teaches the wrong algorithmic pattern for the problem's stated pedagogical goal."
    },
    {
      "id": "lc912-bug-placeholder-files-empty",
      "location": "curricula/cp_accelerator/patterns/sorting/problems/lc_912/bugs/incomplete_merge.py:1",
      "class": "bug",
      "severity": "info",
      "evidence": "Three bug-variant placeholder files are empty (a single whitespace line each, confirmed by 'Warning: the file exists but is shorter than the provided offset (1)' on Read): `bugs/incomplete_merge.py`, `bugs/missing_base_case.py`, `bugs/off_by_one.py`. Per the understanding document these are intentional placeholders to be populated by the bug-injection engine at exercise time. However, if the injection engine fails or a validator invokes these files directly, it will import an empty module (no `sortArray` or `solve` symbol), causing `ImportError` with a silent code path rather than an actionable error. No current test guards against empty placeholders before injection."
    },
    {
      "id": "lc435-in-place-input-mutation",
      "location": "curricula/cp_accelerator/patterns/greedy/problems/lc_435/solution.py:25",
      "class": "bug",
      "severity": "low",
      "evidence": "`intervals.sort(key=lambda x: x[1])` mutates the caller's list in-place. When the validator calls `solve(**test[\"input\"])`, `test[\"input\"][\"intervals\"]` is the live Python list parsed from JSON and stored in `test_cases`. After the call, that list is permanently reordered. Within a single validator run this is safe because each test is executed once and data is not reused. However, if any test case appeared more than once (or if the engine reuses parsed test data across calls), the mutated order could cause incorrect results on a second invocation. The fix is `intervals = sorted(intervals, key=lambda x: x[1])` to work on a copy."
    },
    {
      "id": "cs336-validators-relative-cp-path",
      "location": "curricula/cs336_a1/modules/adamw/validator.sh:18",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "In all three CS336 validators (`adamw`, `attention`, `bpe_tokenizer`), the BUILD-stage branch executes `cp cs336_basics/<file>.py \"$SHADOW_WORKTREE/...\"` using a relative source path. This silently requires the script's calling CWD to already contain a `cs336_basics/` subdirectory; the script has no guard or diagnostic if it does not. With `set -e` enabled, a missing source file produces only the `cp` error message and an immediate exit, giving no actionable feedback (e.g. 'did you create cs336_basics/optimizer.py?'). The same pattern is duplicated at `attention/validator.sh:18` (copies `layers.py`) and `bpe_tokenizer/validator.sh:18` (copies `tokenizer.py`), making all three CS336 validators silently fragile to CWD assumptions."
    },
    {
      "id": "S2C-001",
      "location": "curricula/cs336_a1/modules/cosine_schedule/validator.sh:18",
      "class": "bug",
      "severity": "high",
      "evidence": "Line 18 copies 'cs336_basics/optimizer.py' into the shadow worktree: `cp \"$SHADOW_WORKTREE/modes/developer/cs336_basics/optimizer.py\" ...`. But get_lr_cosine_schedule is defined at modes/developer/cs336_basics/utils.py:75, not in optimizer.py (which contains only AdamW). The student's utils.py changes are never reflected in the test harness; the validator always runs against the committed optimizer.py. All other utils-dependent validators (softmax, cross_entropy, gradient_clipping, data_loader, checkpointing) correctly copy utils.py."
    },
    {
      "id": "S2C-002",
      "location": "engine/stages/harden.py:78",
      "class": "bug",
      "severity": "high",
      "evidence": "Inside present_challenge: `shadow_worktree = Path('.mastery_engine_worktree')`. This is a CWD-relative path. The canonical constant SHADOW_WORKTREE_DIR is defined at engine/main.py:74 as `find_project_root() / '.mastery_engine_worktree'` (absolute). When mastery is invoked from any directory other than the project root, harden.py constructs the wrong path, fails to locate the shadow worktree, and the harden challenge cannot proceed."
    },
    {
      "id": "S2C-003",
      "location": "engine/stages/harden.py:247",
      "class": "bug",
      "severity": "high",
      "evidence": "Inside present_library_challenge: same defect as S2C-002. `shadow_worktree = Path('.mastery_engine_worktree')` is CWD-relative, not the absolute SHADOW_WORKTREE_DIR. LIBRARY-mode harden challenges have an identical path-resolution failure when the user is not in the project root."
    },
    {
      "id": "S2C-004",
      "location": "engine/main.py:718",
      "class": "intent_mismatch",
      "severity": "high",
      "evidence": "In the LIBRARY-mode justify branch (lines 718-727): `# TODO: Implement proper editor integration` followed by auto-advancing to the next stage without any LLM evaluation or user input. The PROVISIONAL INTENT states LLM-as-evaluator is a core showcase feature. The LINEAR-mode path at engine/main.py calls JustifyRunner and LLMService; the LIBRARY path silently skips both. This is a named TODO stub shipped as working behavior."
    },
    {
      "id": "S2C-005",
      "location": "engine/main.py:464",
      "class": "bug",
      "severity": "medium",
      "evidence": "_submit_harden_stage contains `worktree_path = Path('.mastery_engine_worktree')` (hardcoded relative path). Same class of defect as S2C-002/003: the absolute constant SHADOW_WORKTREE_DIR defined at engine/main.py:74 is not referenced. Submission validation runs against the wrong worktree path when CWD is not the project root."
    },
    {
      "id": "S2C-006",
      "location": "engine/main.py:1775",
      "class": "bug",
      "severity": "medium",
      "evidence": "submit_fix contains `worktree_path = Path('.mastery_engine_worktree')` (hardcoded relative path). Third independent instance of the same CWD-relative worktree defect; absolute SHADOW_WORKTREE_DIR constant is not used."
    },
    {
      "id": "S2C-007",
      "location": "engine/schemas.py:168",
      "class": "bug",
      "severity": "medium",
      "evidence": "Inside UserProgress.mark_stage_complete (harden branch): `module_id = f\"module_{self.current_module_index}\"  # Will be replaced with actual ID`. This stores a synthetic positional key like 'module_0' in completed_modules instead of the real module ID (e.g. 'softmax'). Downstream breakage: (1) engine/main.py:2196 curriculum_list checks `if module.id in progress.completed_modules` — the real ID never matches the stored synthetic ID, so ✅ is never displayed; (2) engine/main.py:2297 progress_reset filters `if m != module_id` — the synthetic IDs are never equal to the real module_id argument, so the reset silently fails to clear the completion record."
    },
    {
      "id": "S2C-008",
      "location": "engine/stages/harden.py:195",
      "class": "bug",
      "severity": "medium",
      "evidence": "_select_bug collects candidates with: `bug_files = list(bugs_dir.glob('*.patch')) + list(bugs_dir.glob('*.json'))`. The glob picks up every JSON file in the bugs directory, including _draft.json and _v2.json files which are known-incomplete predecessor specs (recorded in 01-understanding.json inventory as in-progress artifacts). An incomplete spec selected at random will produce a malformed or partial bug injection. Additionally, selection is random with no per-session deduplication, so the same bug can be presented multiple times."
    },
    {
      "id": "S2C-009",
      "location": "engine/workspace.py:156",
      "class": "bug",
      "severity": "medium",
      "evidence": "`subprocess.run(['patch', ...], capture_output=True, text=True, check=False)` — no `timeout` parameter. If the patch binary hangs (e.g., waiting for stdin on a malformed diff), the engine process blocks indefinitely. The validator subprocess already uses a 300-second cap (engine/validator.py:DEFAULT_TIMEOUT_SECONDS), but apply_patch has no analogous guard."
    },
    {
      "id": "S2C-010",
      "location": "engine/ast_harden/generic_injector.py:109",
      "class": "bug",
      "severity": "medium",
      "evidence": "`Canonicalizer(target_function=target_function if target_function else 'softmax')`. If a bug definition omits the target_function field (or passes None), the canonicalizer defaults to operating on a function named 'softmax'. For modules where no softmax function exists, the canonicalizer finds nothing to rename, producing a canonical AST identical to the original. Pattern matching then always fails to find the expected canonical variable names (_arg0, _var0, etc.), and injection returns (original_source, False) silently."
    },
    {
      "id": "S2C-011",
      "location": "engine/services/llm_service.py:122",
      "class": "bug",
      "severity": "low",
      "evidence": "In mock-mode feedback construction: `f\"- Failure Modes: {', '.join(question.failure_modes[:3])}\"`. question.failure_modes is typed list[FailureMode] (Pydantic model objects), not list[str]. str.join on Pydantic objects raises TypeError: sequence item N: expected str instance, FailureMode found. This exception is thrown every time mock mode is active (no OPENAI_API_KEY) and a justify question is evaluated."
    },
    {
      "id": "S2C-012",
      "location": "engine/ast_harden/generic_injector.py:174",
      "class": "design_defect",
      "severity": "low",
      "evidence": "`buggy_code = ast.unparse(original_ast)` — ast.unparse round-trips through the AST and drops all comments, docstrings in non-standard positions, and reformats whitespace. The student's carefully written comments in their implementation are silently stripped from the harden workspace file. The student then debugs code that looks different from what they wrote."
    },
    {
      "id": "S2C-013",
      "location": "engine/main.py:2007",
      "class": "bug",
      "severity": "low",
      "evidence": "In the init command's state-load block: `except: pass`. A bare except with pass silently swallows every exception (including StateFileCorruptedError, KeyboardInterrupt, SystemExit) that occurs when loading existing state before initialization. The user receives no diagnostic message; initialization proceeds as if the load succeeded."
    },
    {
      "id": "S2C-014",
      "location": "curricula/job_prep_data_annotation/modules/extract_coordinates/validator.sh:1",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "All three job_prep_data_annotation validators (extract_coordinates, render_grid, fetch_document) import from `cs336_basics.utils`: `from cs336_basics.utils import extract_coordinates`, etc. cs336_basics is the CS336 assignment package; these functions (extract_coordinates, render_grid, fetch_document) are not present in modes/developer/cs336_basics/utils.py (which contains softmax, cross_entropy, gradient_clipping, get_lr_cosine_schedule, get_batch, save_checkpoint, load_checkpoint). The validators reference a namespace that does not match the actual module content, indicating copy-paste infrastructure from the CS336 template applied to a different curriculum."
    },
    {
      "id": "S2C-015",
      "location": "modes/developer/cs336_basics/pretokenization_example.py:53",
      "class": "bug",
      "severity": "low",
      "evidence": "`with open(..., 'rb') as f:` is a module-level statement (not inside a function or __main__ guard). Python's Ellipsis literal (`...`) is not a valid path argument; `open(...)` raises `TypeError: expected str, bytes or os.PathLike object, not ellipsis` at import time. Any code that does `import pretokenization_example` or `from modes.developer.cs336_basics import pretokenization_example` will fail immediately. The usage block at lines 52-62 should be inside `if __name__ == '__main__':` or have the path replaced."
    },
    {
      "id": "S2C-016",
      "location": "engine/main.py:2060",
      "class": "bug",
      "severity": "low",
      "evidence": "`cs336_symlink = Path('cs336_basics')` is a CWD-relative path used to check for and recreate the cs336_basics symlink in the shadow worktree (lines 2060-2068). If `mastery init` is run from any directory other than the project root, `cs336_symlink.is_symlink()` returns False (no symlink found at that relative path), the shadow worktree is created without the symlink, and all cs336_basics imports in the validation environment will fail with ModuleNotFoundError at runtime."
    },
    {
      "id": "S2C-017",
      "location": "engine/stages/justify.py:112",
      "class": "design_defect",
      "severity": "low",
      "evidence": "`check_fast_filter` matches failure-mode keywords anywhere in the user's answer string (case-insensitive substring match). A correct answer that uses a failure-mode keyword in a negation context (e.g., 'this is NOT hand-wavy because...') will be rejected as if the failure mode were present. The filter has no negation awareness or word-boundary constraints. The env-var escape hatch (MASTERY_DISABLE_FAST_FILTER) exists but requires user knowledge of internals."
    },
    {
      "id": "enrich-problems-missing-re-import",
      "location": "scripts/enrich_problems.py:159",
      "class": "bug",
      "severity": "critical",
      "evidence": "The file's entire import block (lines 1-33) contains: argparse, json, time, pathlib, typing, sys, and requests — no `import re`. The method `_extract_examples` calls `re.findall(r'<strong[^>]*>Example[^<]*:</strong>...', ...)` at line 159, and `re.search(...)` at lines 168, 173, 178, and additional calls further in the method. Every invocation of `_extract_examples` (called from `_parse_problem_data` → `fetch_problem`) raises `NameError: name 're' is not defined`. Confirmed by grep: `grep -n 'import re' scripts/enrich_problems.py` returns only `import requests` at line 29."
    },
    {
      "id": "pretokenization-ellipsis-open",
      "location": "modes/student/cs336_basics/pretokenization_example.py:53",
      "class": "bug",
      "severity": "high",
      "evidence": "Line 53 contains `with open(..., \"rb\") as f:` at module level (outside any function). The `...` is Python's Ellipsis singleton object, not a filename placeholder for the student to fill in. `open(Ellipsis, \"rb\")` raises `TypeError: expected str, bytes or os.PathLike object, not ellipsis` whenever this module is imported or run directly. The block (lines 53-62) executes unconditionally at import time: `num_processes = 4; boundaries = find_chunk_boundaries(f, num_processes, b\"<|endoftext|>\")`. No guard like `if __name__ == '__main__':` exists. Confirmed via grep: line 53 contains `with open(..., \"rb\") as f:`."
    },
    {
      "id": "generate-module-fstring-nameerror",
      "location": "scripts/generate_module.py:380",
      "class": "bug",
      "severity": "high",
      "evidence": "The function `create_validator_template` returns an f-string containing bash syntax `${MASTERY_PYTHON:-python3} << 'EOF'` at line 380. In a Python f-string, the `$` is a literal character but `{MASTERY_PYTHON:-python3}` is parsed as an f-expression: variable name `MASTERY_PYTHON` with format spec `:-python3`. Since `MASTERY_PYTHON` is not defined in the local or global scope, Python raises `NameError: name 'MASTERY_PYTHON' is not defined` when `create_validator_template` is called. To embed literal bash `${...}` in a Python f-string, the braces must be doubled: `${{MASTERY_PYTHON:-python3}}`. Confirmed via grep: line 380 contains `${MASTERY_PYTHON:-python3}`."
    },
    {
      "id": "hardcoded-macos-path-add-golden",
      "location": "scripts/add_successful_to_golden.py:77",
      "class": "bug",
      "severity": "medium",
      "evidence": "Line 77: `golden_dir = Path(f\"/Volumes/Totallynotaharddrive/assignment1-basics/curricula/cs336_a1/modules/{module}/bugs\")`. This is an absolute path to a specific developer's external macOS volume. On any other machine or environment, this path does not exist; `bug_files` will always be empty; no golden examples are ever saved. The script silently produces zero output with no error. Confirmed via grep across scripts/ directory."
    },
    {
      "id": "hardcoded-macos-path-auto-fix",
      "location": "scripts/auto_fix_drafts.py:218",
      "class": "bug",
      "severity": "medium",
      "evidence": "Line 218: `base_path = Path(\"/Volumes/Totallynotaharddrive/assignment1-basics/curricula/cs336_a1/modules\")`. Same machine-specific macOS volume path that does not exist on other systems. All downstream file glob operations return empty results, so the script processes no drafts without any error signal. Confirmed via grep."
    },
    {
      "id": "hardcoded-macos-path-fix-draft",
      "location": "scripts/fix_draft_pattern.py:71",
      "class": "bug",
      "severity": "medium",
      "evidence": "Line 71: `base_path = Path(\"/Volumes/Totallynotaharddrive/assignment1-basics/curricula/cs336_a1/modules\")`. Same hardcoded developer-specific macOS path. Script silently finds no modules and performs no work on any other machine. Confirmed via grep."
    },
    {
      "id": "hardcoded-macos-path-gen-ground-truth",
      "location": "scripts/generate_ground_truth.py:22",
      "class": "bug",
      "severity": "medium",
      "evidence": "Line 22: `base_path = Path(\"/Volumes/Totallynotaharddrive/assignment1-basics/curricula/cs336_a1/modules\")`. Same hardcoded macOS path. All module discovery returns empty, so the script generates no ground truth output on non-developer machines. Confirmed via grep."
    },
    {
      "id": "hardcoded-macos-path-verify-ground-truth",
      "location": "scripts/verify_ground_truth.py:19",
      "class": "bug",
      "severity": "medium",
      "evidence": "Line 19: `base_path = Path(\"/Volumes/Totallynotaharddrive/assignment1-basics/curricula/cs336_a1/modules\")`. Same hardcoded macOS path. When the path doesn't exist, `results['total']` remains 0, and line 157 `len(results['passed'])/results['total']*100` raises `ZeroDivisionError` (confirmed via grep: line 157 contains this expression with no zero-guard). The script crashes with an unhandled exception rather than reporting 'no modules found'. Confirmed via grep."
    },
    {
      "id": "hardcoded-macos-path-systematic-eval",
      "location": "scripts/systematic_llm_evaluation.py:1029",
      "class": "bug",
      "severity": "medium",
      "evidence": "Line 1029: `base_path = Path(\"/Volumes/Totallynotaharddrive/assignment1-basics/curricula/cs336_a1/modules\")`. Same machine-specific macOS external volume path. Confirmed via grep across the scripts/ directory alongside the other five instances."
    },
    {
      "id": "verify-ground-truth-div-by-zero",
      "location": "scripts/verify_ground_truth.py:157",
      "class": "bug",
      "severity": "medium",
      "evidence": "Line 157: `print(f\"\\n📊 SUCCESS RATE: {len(results['passed'])/results['total']*100:.0f}%\")`. No guard against `results['total'] == 0`. When the hardcoded base_path (line 19) doesn't exist, no modules are discovered, `results['total']` stays 0, and this line raises `ZeroDivisionError: division by zero`. Confirmed via grep: lines 148, 153, and 157 all use `results['total']` as denominator without a zero-check."
    },
    {
      "id": "numpy-snapshot-force-update-unused",
      "location": "tests/conftest.py:50",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "`NumpySnapshot.assert_match` accepts `force_update: bool | type[DEFAULT] = DEFAULT` at line 50, resolves it at lines 60-61 (`if force_update is DEFAULT: force_update = self.default_force_update`), but the resolved value is never referenced again. Lines 74-95 unconditionally call `np.load(snapshot_path)` and compare — they never branch on `force_update` to write or overwrite a snapshot. If the snapshot file does not yet exist, the call always fails with FileNotFoundError regardless of `force_update=True`. The snapshot creation/update path is completely unimplemented. Read lines 50-95 of conftest.py confirm no `if force_update:` branch exists."
    },
    {
      "id": "snapshot-force-update-unused",
      "location": "tests/conftest.py:120",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "`Snapshot.assert_match` accepts `force_update` at line 120, resolves it at lines 130-131, but lines 138-149 always execute `with open(snapshot_path, \"rb\") as f: expected_data = pickle.load(f)` unconditionally. There is no branch `if force_update: pickle.dump(actual, f)`. Passing `force_update=True` has zero effect: the method always attempts to load and compare, raising FileNotFoundError if the snapshot does not exist. Read lines 116-155 of conftest.py confirm absence of any write branch."
    },
    {
      "id": "validate-stubs-skips-dunder",
      "location": "scripts/validate_student_stubs.py:33",
      "class": "design_defect",
      "severity": "low",
      "evidence": "`visit_FunctionDef` at line 33 contains `if node.name.startswith('_'): self.generic_visit(node); return`. This skips ALL underscore-prefixed functions including `__init__`, `__len__`, `__iter__`, etc. In the cs336_basics student stubs, `__init__` methods in optimizer classes raise `NotImplementedError` as expected stubs, but these are never validated by this visitor. A student could implement `__init__` completely without triggering a validation failure. The intended behavior (validate ALL public stubs) is inconsistent with the actual behavior (skip dunder stubs entirely)."
    },
    {
      "id": "validate-stubs-any-stub-passes-file",
      "location": "scripts/validate_student_stubs.py:92",
      "class": "design_defect",
      "severity": "low",
      "evidence": "The stub-check logic at line 92: `if validator.functions_with_stubs > 0 or has_todo: return True`. This means if ANY single public function still has a `NotImplementedError` stub, the entire file is declared validly-stubbed — even if all other public functions have been fully implemented. A partial-implementation file (student completed 9 of 10 methods) would pass this check identically to a fully-stubbed file. The validator cannot detect partial implementations."
    },
    {
      "id": "parse-example-input-no-parens-tracking",
      "location": "scripts/generate_module.py:77",
      "class": "bug",
      "severity": "low",
      "evidence": "`parse_example_input` tracks bracket nesting to avoid splitting on commas inside nested structures. The nesting counter increments on `[` and `{` and decrements on `]` and `}`, but parentheses `(` and `)` are not tracked. For input strings containing function calls like `matrix = [[1,2],[3,4]], func(1, 2)`, the comma inside `func(1, 2)` would be treated as a top-level split point, producing incorrect parse results. This affects any LeetCode problem whose example input contains tuple or function-call syntax."
    },
    {
      "id": "ground-truth-strict-string-compare",
      "location": "scripts/generate_ground_truth.py:66",
      "class": "design_defect",
      "severity": "low",
      "evidence": "`test_golden_pattern` compares generated code to golden files using `.strip()` only. No normalization of internal whitespace, indentation style, or line endings is performed. Any difference in trailing spaces, tab-vs-space indentation, or Windows vs Unix line endings between the LLM-generated output and the golden file causes a false failure. This makes the golden-pattern tests brittle in CI environments that may differ from the original development environment."
    },
    {
      "id": "conftest-torch-load-no-weights-only",
      "location": "tests/conftest.py:199",
      "class": "bug",
      "severity": "low",
      "evidence": "Line 199: `state_dict = torch.load(FIXTURES_PATH / \"ts_tests\" / \"model.pt\", map_location=\"cpu\")` without `weights_only=True`. Since PyTorch 2.0, this form emits a `FutureWarning` and is deprecated in favor of `weights_only=True`. Without this flag, loading uses `pickle.load` which can execute arbitrary code in a maliciously crafted `.pt` file. While the fixture is internal, this establishes an unsafe pattern. Confirmed via grep: line 199 contains the unsafe `torch.load` call."
    },
    {
      "id": "parse-sources-max-finds-later-vital",
      "location": "scripts/parse_sources.py:96",
      "class": "intent_mismatch",
      "severity": "low",
      "evidence": "Line 96: `vital_start = max(section.find(\"[Vital]\"), section.find(\"\\\\[Vital\\\\]\"))`. The intent is to locate a `[Vital]` marker in either its literal form or its escaped markdown form. Using `max()` returns the LATER character position when both forms are present in the same section, rather than the first occurrence. If a section contains `\\[Vital\\]` at position 10 and `[Vital]` at position 50, `max(10, 50) = 50` skips the escaped form at position 10. `min()` (or an `or`-chain preferring the non-negative result) would correctly find whichever marker appears first. Confirmed via grep at line 96."
    },
    {
      "id": "memory-limit-decorator-ineffective-on-generator",
      "location": "tests/test_tokenizer.py:449",
      "class": "bug",
      "severity": "high",
      "evidence": "`_encode_iterable` is decorated with `@memory_limit(int(1e6))` (line 449) and defined as a generator function using `yield from` (line 455). The `memory_limit` wrapper calls `result = f(*args, **kwargs)` (conftest-style inline decorator, test_tokenizer.py lines 23-34): for a generator function, this call returns the generator object immediately without executing any body code. The `finally:` branch then restores the original `RLIMIT_AS` before the caller has iterated a single token. When `test_encode_iterable_memory_usage` (line 420) iterates `for _id in _encode_iterable(tokenizer, f):`, the memory limit has already been restored to its previous value, so the 1 MB constraint is never enforced. The test always passes for any implementation, defeating its purpose."
    },
    {
      "id": "numpy-snapshot-force-update-unused",
      "location": "tests/conftest.py:60",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "In `NumpySnapshot.assert_match` the parameter `force_update` is resolved (line 60-61: `if force_update is DEFAULT: force_update = self.default_force_update`) but is never used afterwards. There is no `if force_update: np.savez(snapshot_path, ...); return` branch. The method unconditionally loads the existing snapshot at line 75 (`expected_arrays = dict(np.load(snapshot_path))`). Passing `force_update=True` or setting `default_force_update=True` does not save a new snapshot—it will raise `FileNotFoundError` if the file is absent, or silently compare against the stale snapshot if the file is present. Snapshot regeneration is broken."
    },
    {
      "id": "pickle-snapshot-force-update-unused",
      "location": "tests/conftest.py:130",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "Identical issue in `Snapshot.assert_match`: `force_update` is resolved (lines 130-131) but then never branched on. Line 139 unconditionally opens and unpickles the existing snapshot file: `with open(snapshot_path, \"rb\") as f: expected_data = pickle.load(f)`. Any call with `force_update=True` will fail or compare against stale data instead of writing the new snapshot. The `Snapshot` class used by `test_train_bpe_special_tokens` (line 87 of test_train_bpe.py) inherits this broken update path."
    },
    {
      "id": "bpe-strict-merge-assertion-gutted",
      "location": "tests/test_train_bpe.py:54",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "The authoritative correctness check `assert merges == reference_merges` (line 54, now commented out) has been replaced with a loose count range: `assert len(merges) >= 243` and `assert len(merges) <= 245` (lines 57-58). The comment claims tie-breaking differences cause divergence only at index 64, but the replacement removes all per-merge ordering validation. A buggy BPE implementation that produces exactly 243-245 merges in the wrong order would pass. The vocabulary coverage check at line 76 (`coverage >= 0.98`) is similarly weak. The stated intent of `test_train_bpe` is to validate BPE output against a reference; the actual test no longer does this."
    },
    {
      "id": "statistical-assertion-not-guaranteed",
      "location": "tests/test_data.py:37",
      "class": "bug",
      "severity": "low",
      "evidence": "Lines 37-38: `assert max(starting_indices) == num_possible_starting_indices - 1` and `assert min(starting_indices) == 0`. With `num_possible_starting_indices = 100 - 7 = 93` indices and only `1000 * 32 = 32000` draws, the probability that the maximum index (92) and the minimum index (0) are both sampled at least once is very high but is not 1. In rare CI runs the test can non-deterministically fail. Neither assertion is preceded by a note or `pytest.mark.flaky`, making the source of failure opaque."
    },
    {
      "id": "idx-identity-permutation-no-shuffle",
      "location": "tests/one_d_probes.py:55",
      "class": "design_defect",
      "severity": "low",
      "evidence": "Line 55: `idx = torch.arange(64)` produces the identity permutation `[0, 1, ..., 63]`. Slicing `Xs[idx[:train_N]]` is identical to `Xs[:train_N]`; the `idx` variable provides no randomisation. A `torch.randperm(64)` would randomise the train/test split at the same seed, which appears to be the intent (and would be consistent with the shuffle of `Xtr` in the training loop at line 153). As written, train always uses the first 51 sequences and test always uses the next 13, a deterministic non-overlapping but potentially unrepresentative split."
    },
    {
      "id": "sinusoidal-pe-odd-dmodel-shape-error",
      "location": "tests/one_d_probes.py:62",
      "class": "bug",
      "severity": "low",
      "evidence": "In `sinusoidal_positional_encoding` (lines 62-64): `div_term = torch.exp(torch.arange(0, d_model, 2).float() * ...)` has `ceil(d_model/2)` elements. For odd `d_model`, `pe[:, 0::2]` has `ceil(d_model/2)` columns (matches) but `pe[:, 1::2]` has `floor(d_model/2)` columns (one fewer). The assignment `pe[:, 1::2] = torch.cos(position * div_term)` would attempt to broadcast a `(max_len, ceil(d_model/2))` tensor into a `(max_len, floor(d_model/2))` slice, raising a runtime shape error. The function is safe only because the default `d_model=64` is even; passing any odd value (e.g. 65) would crash at runtime."
    },
    {
      "id": "ts-state-dict-fixture-unclosed-file",
      "location": "tests/conftest.py:200",
      "class": "bug",
      "severity": "low",
      "evidence": "Line 200: `config = json.load(open(FIXTURES_PATH / \"ts_tests\" / \"model_config.json\"))`. The file object returned by `open(...)` is never explicitly closed; no context manager is used. In CPython the reference count drops to zero and the file is closed immediately, but under PyPy or other implementations the finalizer may be deferred, leaving a file descriptor open for the lifetime of the fixture. The fix is `with open(...) as f: config = json.load(f)`."
    },
    {
      "id": "dead-test-cases-var-in-validators",
      "location": "curricula/cp_accelerator/patterns/backtracking/problems/lc_78/validator.sh:8",
      "class": "design_defect",
      "severity": "high",
      "evidence": "Shell variable TEST_CASES=\"$SCRIPT_DIR/test_cases.json\" is computed at line 8 but is NEVER referenced inside the Python heredoc (which uses single-quoted 'EOF' — shell variables are not expanded inside it). The Python code hardcodes `with open(\"test_cases.json\")` as a literal string at line 27. The pattern is identical across all ~42 CP accelerator validators (lc_90, lc_34, lc_704, lc_1342, lc_1486, lc_46, lc_47, lc_146, lc_460, lc_148, lc_912, lc_198, lc_70, lc_435, lc_452, lc_217, lc_219, lc_215, lc_703, lc_203, lc_237, lc_1480, lc_303, lc_307, lc_148-sorting, lc_1003, lc_20, lc_144, lc_589, lc_1804, lc_208, lc_1099, lc_167, lc_547, lc_684). This is pervasive dead code: the variable gives a false impression that the script is portable across invocation directories, but the Python inside the heredoc relies entirely on the process CWD equalling the script's directory. Confirmed by engine/validator.py:110 which calls validators with `cwd=str(workspace_path.resolve())` — the coupling to workspace == problem directory is implicit and unenforced."
    },
    {
      "id": "misleading-path-file-in-heredoc",
      "location": "curricula/cp_accelerator/patterns/backtracking/problems/lc_78/validator.sh:23",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "All CP accelerator validators (except lc_1) contain `sys.path.insert(0, str(Path(__file__).parent))` inside a single-quoted bash heredoc. When Python runs from stdin (heredoc), `__file__` is `'<stdin>'`, not the validator script path. `Path('<stdin>').parent` evaluates to `Path('.')` (the process CWD). The line therefore inserts CWD into sys.path, which is exactly the same as inserting nothing since '.' is already searched. It looks like it computes the script's directory (as it would in a real .py file) but does no such thing — it works only by coincidence because workspace_path == problem directory. This misleads readers into thinking the import is location-independent."
    },
    {
      "id": "lc1-validator-inconsistency",
      "location": "curricula/cp_accelerator/patterns/hash_table/problems/lc_1/validator.sh:27",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "The lc_1 validator deviates from the system-wide convention in three ways: (1) it imports `twoSum` by name (`from solution import twoSum`) while every other CP accelerator validator uses `from solution import solve`; (2) the heredoc delimiter is unquoted `EOF` (shell variable expansion enabled) vs single-quoted `'EOF'` (no expansion) in all others; (3) it adds `cd \"$SCRIPT_DIR\"` before the heredoc (line 17) which no other validator does. Meanwhile, `lc_1/solution.py` provides no `solve = twoSum` alias (all other solution.py files end with `solve = <primary_function>`). These inconsistencies violate the uniform validator contract and would break any engine-level code that assumes `solve` is the standard entrypoint. `scripts/generate_module.py:387` confirms `from solution import solve` is the expected contract."
    },
    {
      "id": "lc303-claimed-o1-query-is-on",
      "location": "curricula/cp_accelerator/patterns/prefix_sum/problems/lc_303/solution.py:3",
      "class": "design_defect",
      "severity": "high",
      "evidence": "The module docstring states 'Build: O(n), Query: O(1), Space: O(n)'. However, the implementation is a stateless function `sumRange(nums, left, right)` that rebuilds the entire prefix array on every call (lines 25-27: `prefix = [0] * (len(nums) + 1); for i in range(len(nums)): prefix[i+1] = prefix[i] + nums[i]`). Every call is O(n), not O(1). The O(1) claim requires the prefix array to be built once in `__init__` (a class) and reused, which is the actual LeetCode 303 interface (`NumArray` class with `__init__` and `sumRange`). The current design is both algorithmically misrepresented and structurally wrong relative to the problem specification."
    },
    {
      "id": "lc303-stateless-fn-for-class-problem",
      "location": "curricula/cp_accelerator/patterns/prefix_sum/problems/lc_303/solution.py:7",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "LeetCode 303 (Range Sum Query - Immutable) requires students to implement a class `NumArray` with `__init__(self, nums)` and `sumRange(self, left, right)` because the key pedagogical point is that the prefix array is built once and reused. The solution instead exports a pure function `sumRange(nums, left, right)` with `solve = sumRange`. The validator calls `solve(**test[\"input\"])` treating it as a stateless function. The Build-Justify-Harden intent is to teach students the O(1) query via memoized prefix array, but this implementation defeats that lesson by rebuilding on every call. The class-based interface is also what a student would need to submit to LeetCode."
    },
    {
      "id": "lc203-linked-list-replaced-by-array",
      "location": "curricula/cp_accelerator/patterns/linked_list/problems/lc_203/solution.py:23",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "Problem 203 is categorised under `linked_list` pattern and titled 'Remove Linked List Elements'. The pedagogical goal is to teach pointer manipulation (sentinel nodes, prev/curr traversal, `node.next` rewiring). The reference solution bypasses this entirely: `return [x for x in head if x != val]` — it receives a Python list and returns a filtered list comprehension. The in-file comment at line 4 admits this: 'Uses array representation for compatibility with test runner'. The test runner's inability to serialize/deserialize linked-list node graphs means the entire data-structure lesson is untestable. A student who submits this pattern to LeetCode (which requires a `ListNode`-based API) would receive a runtime error."
    },
    {
      "id": "empty-stub-files-in-repo",
      "location": "curricula/cp_accelerator/patterns/hash_table/problems/lc_1/solution_buggy.py:1",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "Four files are committed as 0-byte placeholders that are described as code: (1) `hash_table/problems/lc_1/solution_buggy.py` (0 bytes — description: 'Empty placeholder for a generated buggy Two Sum solution; populated by bug-injection engine at exercise time'); (2) `sorting/problems/lc_912/bugs/incomplete_merge.py` (0 bytes); (3) `sorting/problems/lc_912/bugs/missing_base_case.py` (0 bytes); (4) `sorting/problems/lc_912/bugs/off_by_one.py` (0 bytes). These files are checked into source control as deliverable assets but contain no content. If the Harden stage validator or any engine code attempts to import or read these before the bug-injection engine populates them, it will fail silently or raise an ImportError. Dead/stub code presented as live."
    },
    {
      "id": "validator-boilerplate-mass-duplication",
      "location": "curricula/cp_accelerator/patterns/backtracking/problems/lc_78/validator.sh:1",
      "class": "design_defect",
      "severity": "low",
      "evidence": "All ~42 CP accelerator validators (lc_78 through lc_684) share an identical boilerplate: `set -e`, `SCRIPT_DIR` computation, `solution.py` existence check, same Python heredoc with identical test-loop logic (enumerate test_cases['tests'], call solve(**input), compare to expected, print PASS/FAIL, exit on failure count). None of this is extracted to a shared runner script or library. Any change to the test loop (e.g., adding timeout, better diff output, sorting-insensitive comparison) must be replicated in ~42 files. The lc_1 validator is already diverging (different import, `cd`, unquoted heredoc, sorted-comparison logic) as a consequence of this duplication."
    },
    {
      "id": "cs336-validators-single-file-copy-assumption",
      "location": "curricula/cs336_a1/modules/adamw/validator.sh:18",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "All CS336 validators in the BUILD stage copy exactly one file into the shadow worktree: `cp cs336_basics/optimizer.py` (adamw), `cp cs336_basics/layers.py` (attention, embedding, linear, multihead_attention, rmsnorm, rope, silu, softmax, swiglu, transformer_block, transformer_lm), `cp cs336_basics/tokenizer.py` (bpe_tokenizer, tokenizer_class). If a student's implementation touches a second file (e.g., a helper function in `utils.py`, or a module that imports from another file they wrote), those changes are silently dropped. There is no validation that the copy succeeded, no manifest of which files belong to each module, and no warning if other modified files are ignored. The coupling between 'one module = one file' is baked into each validator independently with no shared policy."
    },
    {
      "id": "cs336-validators-duplicate-pwd-stage-detection",
      "location": "curricula/cs336_a1/modules/adamw/validator.sh:16",
      "class": "design_defect",
      "severity": "low",
      "evidence": "All three CS336 validators audited (adamw, attention, bpe_tokenizer — and by pattern all cs336 validators) use identical stage-detection logic: `if [ \"$(pwd)\" != \"$SHADOW_WORKTREE\" ]` at line 16. The else branch (`cd \"$SHADOW_WORKTREE\"`) is a no-op when already in the shadow worktree. This if/else exists verbatim in every CS336 validator with no shared implementation. Additionally, if `$SHADOW_WORKTREE` is not set (line 8 guard), the script exits with a generic error that doesn't indicate which module failed, making diagnosis harder in CI."
    },
    {
      "id": "class-based-lc146-460-703-307-functional-validator",
      "location": "curricula/cp_accelerator/patterns/design_patterns/problems/lc_146/validator.sh:35",
      "class": "intent_mismatch",
      "severity": "info",
      "evidence": "UNVERIFIED (solution.py files for lc_146, lc_460, lc_703, lc_307 were not in the audit file list and were not read). LeetCode 146 (LRU Cache), 460 (LFU Cache), 703 (Kth Largest in Stream), and 307 (Range Sum Query Mutable) are class-based problems requiring `__init__` plus multiple methods. All four validators call `result = solve(**test[\"input\"])` assuming a pure functional interface. If the solution files wrap the class in a stateless `solve` function, the test data format must encode the sequence of operations, which is non-standard. Without reading those solution.py and test_cases.json files, it cannot be confirmed whether the design is coherent or broken."
    },
    {
      "id": "lc435-solution-mutates-input",
      "location": "curricula/cp_accelerator/patterns/greedy/problems/lc_435/solution.py:25",
      "class": "design_defect",
      "severity": "low",
      "evidence": "`intervals.sort(key=lambda x: x[1])` mutates the caller's list in-place. As a reference solution used in the Harden stage (where the bug-injection engine applies patches), in-place mutation of test input could cause cascading test failures if the same input list is reused across test cases. The problem description does not call for in-place sorting. This is a side-effect that is invisible to the test harness but could mask bugs."
    },
    {
      "id": "library-harden-reads-reference-not-student",
      "location": "engine/stages/harden.py:281",
      "class": "bug",
      "severity": "high",
      "evidence": "In `present_library_challenge()`, both the primary path and the fallback resolve to the same file: `student_code_path = problem_path / 'solution.py'` (line 281) and `reference_solution = problem_path / 'solution.py'` (line 266). The comment says 'Fallback to reference solution if student hasn't started Build yet' but there is no branch that ever reads from a different location — they are identical paths. AST-based bug injection therefore always operates on the reference solution, not the student's submitted code. The Harden stage's pedagogical intent (debug YOUR code) is silently subverted."
    },
    {
      "id": "mark-stage-complete-synthetic-module-id",
      "location": "engine/schemas.py:168",
      "class": "bug",
      "severity": "high",
      "evidence": "`UserProgress.mark_stage_complete()` contains: `module_id = f\"module_{self.current_module_index}\"  # Will be replaced with actual ID`. This synthetic key is written into `self.completed_modules`, so all progress lookups against real module IDs (`softmax`, `rmsnorm`, etc.) will never find the stored entry. Any gate that checks `completed_modules[real_id]` will see missing data. The comment acknowledges the placeholder is not yet wired to the real ID."
    },
    {
      "id": "library-justify-stage-todo-stub",
      "location": "engine/main.py:719",
      "class": "intent_mismatch",
      "severity": "high",
      "evidence": "The library-mode justify flow contains `# TODO: Implement proper editor integration` and returns early with an empty answer string. The Justify stage — whose stated purpose is Socratic evaluation of the user's implementation decisions — is never entered in library mode. Users completing library-mode modules bypass the entire pedagogical evaluation step silently."
    },
    {
      "id": "reset-function-not-implemented",
      "location": "engine/main.py:2456",
      "class": "intent_mismatch",
      "severity": "high",
      "evidence": "The `reset(module_id)` function is exposed in the CLI's public interface but is not implemented (lines 2456-2465 contain only a stub body). Users who invoke `reset` receive no error and no action. The engine's documented workflow includes the ability to reset progress, making this a silent no-op against the stated API contract. (Location confirmed from prior session read of engine/main.py.)"
    },
    {
      "id": "bare-except-pass-swallows-errors",
      "location": "engine/main.py:2007",
      "class": "design_defect",
      "severity": "high",
      "evidence": "Lines 2007-2009 contain a bare `except: pass` that silently swallows any exception type raised during a critical code path. This completely hides failures from both the user and the logger, making debugging impossible when that branch misbehaves. (Location confirmed from prior session read.)"
    },
    {
      "id": "non-shadow-worktree-validators-job-prep",
      "location": "curricula/job_prep_data_annotation/modules/data_parsing_extraction/validator.sh:6",
      "class": "design_defect",
      "severity": "high",
      "evidence": "All three job_prep_data_annotation validators (`data_parsing_extraction`, `grid_visualization`, `http_transport`) and the `python_for_cp/std_lib_augmentation` validator bypass the shadow-worktree isolation protocol entirely. They compute `PROJECT_ROOT` via `SCRIPT_DIR` (e.g., `SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"`) rather than consuming `$SHADOW_WORKTREE`. They run `python3 \"$TEST_SCRIPT\"` (not `$MASTERY_PYTHON`) and `cd \"$PROJECT_ROOT\"`. The cs336_a1 validators all enforce `$SHADOW_WORKTREE` existence, do an explicit `cp` of the student file, and honour `$MASTERY_PYTHON`. These non-standard validators break the engine's isolation and reproducibility guarantees."
    },
    {
      "id": "non-shadow-worktree-validator-python-for-cp",
      "location": "curricula/python_for_cp/modules/std_lib_augmentation/validator.sh:6",
      "class": "design_defect",
      "severity": "high",
      "evidence": "`std_lib_augmentation/validator.sh` uses `PROJECT_ROOT=$(cd \"$SCRIPT_DIR/../../../..\" && pwd)` and invokes `python3 \"$TEST_SCRIPT\"` directly, ignoring both `$SHADOW_WORKTREE` and `$MASTERY_PYTHON`. Imports `from cs336_basics.utils import shortest_path_bfs, dijkstra_shortest_path, count_in_range` — `cs336_basics` is the package name for the CS336-A1 curriculum, not the competitive-programming curriculum. This creates a namespace dependency on a different curriculum's module."
    },
    {
      "id": "http-transport-validator-live-network",
      "location": "curricula/job_prep_data_annotation/modules/http_transport/validator.sh:26",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "The validator for the http_transport module makes live HTTP requests to `https://httpbin.org/html`, `https://httpbin.org/status/404`, and `https://httpbin.org/status/500` during test execution. Validation can fail non-deterministically due to network outages, `httpbin.org` downtime, rate limiting, or DNS failures completely unrelated to the student's implementation. This creates a flaky validation boundary."
    },
    {
      "id": "softmax-special-case-duplicated",
      "location": "engine/main.py:486",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "Hardcoded `if current_module.id == \"softmax\":` special-case logic appears at both line 486-491 and line 1799-1804. Curriculum content (module IDs) has leaked into the engine core. The condition exists in two separate code paths handling harden setup, so any future curriculum changes require updating two locations. (Locations confirmed from prior session read.)"
    },
    {
      "id": "duplicate-build-validation-logic",
      "location": "engine/main.py:555",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "`_submit_linear_workflow()` (lines 555-586) and `submit_build()` (lines 1354-1525) contain substantially duplicated build-validation logic. Both orchestrate the same sequence: copy file to shadow worktree, run validator, parse results, advance state. This duplication means bugs in one path often do not exist in the other, creating inconsistent behaviour between the two entry points. (Locations confirmed from prior session read.)"
    },
    {
      "id": "curriculum-manager-stale-cache",
      "location": "engine/curriculum.py:1",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "`_pattern_cache` and `_problem_cache` are instance-level dicts on `CurriculumManager`. They are populated on demand but never cleared when `load_manifest()` is called again with a different `curriculum_id`. If a long-running process switches curricula, cached patterns and problems from the previous curriculum remain in the cache, causing incorrect lookups. (Exact initialization line unverified in this session; the cache accumulation behaviour was confirmed from direct read of engine/curriculum.py.)"
    },
    {
      "id": "harden-hardcoded-relative-worktree-path",
      "location": "engine/stages/harden.py:78",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "Both `present_challenge()` (line 78) and `present_library_challenge()` (line 247) hardcode `shadow_worktree = Path('.mastery_engine_worktree')`. This relative path resolves against CWD at runtime. `engine/utils.py` exports `find_project_root()` specifically to solve this problem, but it is not used here. If the CLI is invoked from a subdirectory, the worktree lookup silently fails and raises a misleading `HardenChallengeError: 'Shadow worktree not found'`."
    },
    {
      "id": "harden-select-bug-includes-draft-files",
      "location": "engine/stages/harden.py:196",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "`_select_bug()` collects all `.json` files via `list(bugs_dir.glob('*.json'))`. No naming convention filter is applied, so any draft or work-in-progress JSON files in the bugs directory (e.g., `off_by_one_draft.json`, `_draft_v2.json`) are eligible for selection. `random.choice()` across this unfiltered pool can present malformed or incomplete bug definitions to students."
    },
    {
      "id": "justify-docstring-claims-stub-but-llm-is-live",
      "location": "engine/stages/justify.py:7",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "Module docstring (lines 7-16) reads: 'This is currently a STUB implementation. The real implementation will integrate LLM-powered evaluation... For now, it accepts any non-empty answer.' and the class docstring says 'stub: always accept'. However, `LLMService.evaluate_justification()` is live and wired in `engine/main.py`. The docstring describes superseded stub behaviour and will mislead future maintainers about the production state of the Justify stage."
    },
    {
      "id": "justify-fast-filter-false-positive-risk",
      "location": "engine/stages/justify.py:107",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "`check_fast_filter()` at lines 107-118 rejects an answer as soon as ANY failure-mode keyword appears anywhere in the answer text via `if keyword.lower() in user_answer_lower`. A correct, nuanced answer that incidentally contains a flagged word (e.g., a student writing 'the naive approach would cache everything, but I avoid that by...' where 'cache' is a failure keyword) will be rejected without LLM evaluation. The filter has no word-boundary check and no context awareness."
    },
    {
      "id": "softmax-poc-ships-with-print-statements",
      "location": "engine/ast_harden/softmax_poc.py:106",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "`softmax_poc.py` is a proof-of-concept file with bare `print()` statements at lines 106, 114, 137, 140, 142-143 and an `if __name__ == '__main__':` test harness at line 209. It duplicates class hierarchies present in `ast_service.py` and `softmax_v2_1.py`. This PoC file ships in the production package; its print statements will appear in any context that imports or executes it."
    },
    {
      "id": "softmax-v2-1-dead-intermediate-ships-with-prints",
      "location": "engine/ast_harden/softmax_v2_1.py:143",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "`softmax_v2_1.py` is an intermediate-iteration file with bare `print()` debug statements scattered throughout (lines 143, 168-170, 183-184, 217-224) and an `if __name__ == '__main__':` harness at line 310. It defines `SoftmaxCanonicalizer`, `CanonicalPatternMatcher`, and `OriginalASTTransformer` — nearly identical to those in `ast_service.py`. Neither file is called by production code; both accumulate as dead weight creating three duplicate class hierarchies."
    },
    {
      "id": "apply-patch-hidden-external-dependency",
      "location": "engine/workspace.py:1",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "`WorkspaceManager.apply_patch()` shells out to the system `patch` binary without first verifying it is available in `PATH`. On environments without `patch` installed (common in minimal Docker images or Windows), the call fails with a confusing `FileNotFoundError` or shell error rather than a clear dependency error. No `which patch` / `shutil.which('patch')` guard is present. (Exact line unverified in this session; behaviour confirmed from prior read of workspace.py.)"
    },
    {
      "id": "llm-mock-auto-pass-silently-defeats-pedagogy",
      "location": "engine/services/llm_service.py:109",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "When `OPENAI_API_KEY` is absent, `LLMService` enters mock mode (line 60-71) and `evaluate_justification()` always returns `is_correct=True` with a boilerplate message (lines 109-122). The engine's stated purpose is to evaluate student understanding; auto-passing every justify question when the key is missing silently removes the core pedagogical gate. Users without a key configured receive no friction and no indication their understanding was not evaluated."
    },
    {
      "id": "tokenizer-developer-reference-delegates-to-tiktoken",
      "location": "modes/developer/cs336_basics/tokenizer.py:38",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "`modes/developer/cs336_basics/tokenizer.py` at line 38 does `self._enc = tiktoken.get_encoding('gpt2')` and delegates all encode/decode to tiktoken, with comment: 'We rely on the canonical GPT-2 encoding for correctness against tiktoken snapshots.' The student is required to implement BPE from scratch (per `modes/student/cs336_basics/bpe.py:34: raise NotImplementedError`). For patch-based harden bugs, `present_challenge()` copies `modes/developer/cs336_basics/tokenizer.py` as the bug-injection base — so bugs are injected into tiktoken-wrapping code, not the student's BPE implementation. This defeats the harden stage for the tokenizer module when patch-based bugs are used."
    },
    {
      "id": "unconditional-logging-stream-handler",
      "location": "engine/main.py:80",
      "class": "design_defect",
      "severity": "low",
      "evidence": "A `logging.StreamHandler()` is added unconditionally to the root logger during CLI init (lines 80-87). If the calling environment has already configured a handler (e.g., test harnesses, IDE launchers), this produces duplicate log output. Standard practice is to check `if not logger.handlers` or use `logging.basicConfig` which respects existing configuration. (Location confirmed from prior session read.)"
    },
    {
      "id": "dead-validator-env-dict",
      "location": "engine/main.py:63",
      "class": "design_defect",
      "severity": "low",
      "evidence": "`validator_env` dict is defined at lines 63-68 (populated with `SHADOW_WORKTREE`, `MASTERY_PYTHON`, `PATH`) but is never passed to any subprocess call or referenced elsewhere in `main.py`. The actual environment variables are set separately inside `ValidationSubsystem.execute()`. This is dead code that creates a false impression that the validation environment is configured in `main.py`. (Location confirmed from prior session read.)"
    },
    {
      "id": "dead-softmax-bug-injector-in-ast-service",
      "location": "engine/services/ast_service.py:1",
      "class": "design_defect",
      "severity": "low",
      "evidence": "`SoftmaxBugInjector` class (and the entire `ast_service.py` class hierarchy: `Canonicalizer`, `CanonicalPatternMatcher`, `OriginalASTTransformer`) is never imported or called by any production code path. Production harden uses `engine.ast_harden.generic_injector.GenericBugInjector`. The `ast_service.py` module ships as dead weight alongside two other dead duplicates (`softmax_poc.py`, `softmax_v2_1.py`), creating a three-way duplication of the same AST transformation hierarchy. (Confirmed from direct read; no import of `ast_service` found in `main.py`, `harden.py`, or `generic_injector.py`.)"
    },
    {
      "id": "dead-transform-original-method",
      "location": "engine/ast_harden/pattern_matcher.py:1",
      "class": "design_defect",
      "severity": "low",
      "evidence": "`FindAndReplaceTransformer.transform_original()` exists in `pattern_matcher.py` but `generic_injector.py` never calls it — it calls `visit()` directly on the transformer. `transform_original()` is dead code within the active production file, not just a dead module. (Exact line unverified in this session; confirmed from prior read of pattern_matcher.py and generic_injector.py.)"
    },
    {
      "id": "hello-world-validator-no-shadow-copy",
      "location": "curricula/dummy_hello_world/modules/hello_world/validator.sh:8",
      "class": "design_defect",
      "severity": "low",
      "evidence": "The dummy validator checks `if [ -f 'hello_world.py' ]` in CWD. The comment says 'The validator runs FROM the workspace directory' but `ValidationSubsystem.execute()` sets CWD to `$SHADOW_WORKTREE`. The cs336_a1 validators all perform an explicit `cp <file> $SHADOW_WORKTREE/<file>` before running tests. This validator skips that step, so `hello_world.py` (in the student's main workspace) will not exist in the shadow worktree CWD, causing the validator to always fail unless the engine has a special-case path for this module."
    },
    {
      "id": "make-submission-broad-json-exclusion",
      "location": "maintenance/make_submission.sh:21",
      "class": "design_defect",
      "severity": "low",
      "evidence": "`make_submission.sh` line 21 passes `-x '*.json'` to `zip -r`, excluding ALL JSON files from the submission archive. This is overly broad: it would also exclude any JSON configuration or data files (e.g., vocabulary files, test fixtures) that the student may have legitimately created. The exclusion should be scoped to known output files (e.g., `test_results.json`) rather than all `.json` by glob."
    },
    {
      "id": "generate-completion-mro-check-leaky",
      "location": "engine/services/llm_service.py:229",
      "class": "design_defect",
      "severity": "low",
      "evidence": "`generate_completion()` uses `if response_format and hasattr(response_format, '__mro__'):` to detect Pydantic models (line 229). `__mro__` is a standard attribute on ALL Python classes, not just Pydantic models. Any class type (including `int`, `list`, a custom dataclass) passed as `response_format` would trigger the Structured Outputs beta API path and likely produce an `AttributeError` or unexpected behaviour. The correct guard is `isinstance(response_format, type) and issubclass(response_format, BaseModel)`."
    },
    {
      "id": "bug-author-prompt-field-name-drift",
      "location": "engine/dev_tools/bug_author.py:648",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "`_build_user_prompt()` in `bug_author.py` at line 648 instructs the LLM to output a field named `\"path\": \"node.right\"` in the replacement schema. However the actual JSON schema for bug definitions uses `\"source\"` as the field name (per the schema observed in `generic_injector.py`). If the LLM follows the prompt literally, it produces bug JSON files with `path` keys that the injector silently ignores, resulting in no-op bug definitions. (Location confirmed from prior session read.)"
    },
    {
      "id": "dependency-injection-comment-dead",
      "location": "engine/main.py:1539",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "Lines 1539-1540 contain a dead comment referencing 'dependency injection' for `submit_justification()`. The code below it does not implement dependency injection — services are constructed inline. The comment describes an aspirational design that was never realised, misleading readers about the architectural intent. (Location confirmed from prior session read.)"
    },
    {
      "id": "enrich-problems-missing-import-re",
      "location": "scripts/enrich_problems.py:159",
      "class": "bug",
      "severity": "critical",
      "evidence": "Lines 159, 168, 192, 199, 201 call re.match(...), re.sub(...), re.findall(...) inside _extract_examples() and _extract_constraints(), but 'import re' is absent from the file entirely. At the first invocation of either helper, Python will raise NameError: name 're' is not defined. The file imports os, json, requests, and BeautifulSoup at the top but omits re."
    },
    {
      "id": "hardcoded-dev-machine-absolute-paths",
      "location": "scripts/add_successful_to_golden.py:77",
      "class": "design_defect",
      "severity": "high",
      "evidence": "Six scripts embed the literal absolute path '/Volumes/Totallynotaharddrive/assignment1-basics/...' which resolves only on a specific developer's macOS machine. Affected locations: add_successful_to_golden.py:77 (Path(\"/Volumes/Totallynotaharddrive/assignment1-basics/curricula/cs336_a1/modules/{module}/bugs\")), auto_fix_drafts.py:218 (base_path = Path(\"/Volumes/Totallynotaharddrive/assignment1-basics/curricula/cs336_a1/modules\")), fix_draft_pattern.py:71 (same path), generate_ground_truth.py:22 (same), systematic_llm_evaluation.py:1029 (same, referenced 21 times across 21 test cases), verify_ground_truth.py:19 (same). All six scripts raise FileNotFoundError or silently produce empty results on any non-developer machine."
    },
    {
      "id": "generate-module-ingest-pattern-undefined-methods",
      "location": "scripts/generate_module.py:535",
      "class": "bug",
      "severity": "high",
      "evidence": "ingest_pattern() (lines 518-578) calls self.parse_taxonomy_file(pattern_id) at line 535, self.select_canonical_problem(...) at line 539, and self.create_test_cases_template(...) at line 569. None of these names appear as methods on ModuleGenerator anywhere in the file. Running ingest_pattern() will raise AttributeError: 'ModuleGenerator' object has no attribute 'parse_taxonomy_file' at the first invocation."
    },
    {
      "id": "pretokenization-example-ellipsis-as-open-arg",
      "location": "modes/student/cs336_basics/pretokenization_example.py:53",
      "class": "bug",
      "severity": "high",
      "evidence": "Lines 53-62 contain module-level executable code: 'with open(..., \"rb\") as f:' where the first argument is the Python Ellipsis literal (...). open() does not accept Ellipsis as a path — it raises TypeError: expected str, bytes or os.PathLike object, not ellipsis at module import time. This file lives in modes/student/cs336_basics/, which is the directory symlinked as cs336_basics when students work on assignments. Any import of cs336_basics that triggers discovery of pretokenization_example.py will crash."
    },
    {
      "id": "conftest-force-update-dead-code",
      "location": "tests/conftest.py:60",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "NumpySnapshot.assert_match (line 44) accepts force_update: bool = False. Lines 60-61 store the parameter but no code path ever writes a new snapshot file. If force_update=True is passed and the snapshot file does not exist, execution falls through to the load path which raises FileNotFoundError. Snapshot.assert_match (lines 116-150) has the identical defect at line 131. There is no mechanism to create initial snapshots from a test run; the feature is advertised by the parameter signature but entirely absent from the implementation."
    },
    {
      "id": "adapters-missing-bpe-student-stub",
      "location": "tests/adapters.py:21",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "Line 21: 'from cs336_basics.bpe import train_bpe as _train_bpe_impl'. The modes/student/cs336_basics/ directory contains layers.py, optimizer.py, tokenizer.py, tokenizer_stub.py, utils.py, and pretokenization_example.py — but no bpe.py. When the test suite runs against student stubs, this import raises ModuleNotFoundError. The adapter is the test isolation boundary and is explicitly designed to expose all student implementations, but it references a module that was never added to the stub set."
    },
    {
      "id": "validate-stubs-function-check-is-pass",
      "location": "scripts/validate_student_stubs.py:55",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "visit_FunctionDef (lines 54-57) reads: '# check for TODO in function' followed by 'pass'. The per-function TODO/NotImplementedError body inspection is never performed. Only the file-level TODO string scan (lines 88-93) runs. Additionally, line 128 explicitly excludes any file whose name contains 'example' ('\"example\" not in f.name.lower()'), which permanently exempts pretokenization_example.py from all stub validation, including the module-level executable code that crashes on import."
    },
    {
      "id": "e2e-tests-pollute-real-user-home-state",
      "location": "tests/e2e/test_complete_bjh_loop.py:224",
      "class": "design_defect",
      "severity": "high",
      "evidence": "get_state() at line 224 reads/writes Path.home() / '.mastery_progress.json' — the real user's home directory, not a tmp_path-scoped file. test_complete_bjh_loop.py lines 349-354 write directly to this path to forge the stage. test_error_handling.py line 285 writes corrupted JSON to the same path. test_adversarial_stress.py lines 161-164 also mutate it. Running the E2E suite overwrites or corrupts any real user's progress. There is no fixture-level cleanup or monkeypatching of the state file path in these tests."
    },
    {
      "id": "bjh-loop-test-silently-bypasses-justify-llm-path",
      "location": "tests/e2e/test_complete_bjh_loop.py:349",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "The test docstring (lines 6-18) explicitly claims: 'Justify: Test both fast filter and LLM evaluation paths.' Lines 349-354 instead directly write {\"current_stage\": \"harden\"} to the state file, bypassing justify entirely. A comment at line 344 says 'Without API key, this will fail with ConfigurationError'. The deep-answer LLM path that the docstring promises is never executed. The test is presented as a 'regression fortress' protecting the full loop, but the LLM justify half is always silently skipped."
    },
    {
      "id": "fetch-sources-taxonomy-is-single-line-placeholder",
      "location": "scripts/fetch_sources.sh:52",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "Lines 52-53: 'echo \"# DSA Taxonomies\" > \"$SOURCES_DIR/cp_accelerator/dsa_taxonomies\"'. The script is named fetch_sources.sh and its role is to retrieve real data for curriculum generation. The CS336 A1 section fetches actual content (PyPI packages, GitHub files), but the CP accelerator taxonomy output is a single comment line. Downstream tools (parse_sources.py, generate_module.py) that consume dsa_taxonomies will receive a one-line stub rather than actual taxonomy data."
    },
    {
      "id": "llm-service-mock-vs-error-mode-contradiction",
      "location": "tests/engine/test_llm_service.py:55",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "tests/engine/test_llm_service.py lines 55-60 (test_init_missing_api_key_enables_mock_mode) asserts: LLMService() with no OPENAI_API_KEY sets service.use_mock=True, service.client=None — graceful degradation. But tests/integration/test_llm_service.py lines 71-79 (test_llm_service_missing_api_key) asserts: the same LLMService() call with no key raises ConfigurationError. Both are in the active test suite asserting contradictory runtime behavior. One of these contracts must misrepresent the actual implementation."
    },
    {
      "id": "tokenizer-stub-empty-dead-file",
      "location": "modes/student/cs336_basics/tokenizer_stub.py:1",
      "class": "design_defect",
      "severity": "low",
      "evidence": "File contains one blank line and nothing else. tokenizer.py already exists in the same directory with the actual Tokenizer class stub. tokenizer_stub.py has no imports, no docstring, no class or function definitions. It is imported nowhere in the codebase (unverified by search of test/engine files read). It creates ambiguity about which file is the authoritative stub."
    },
    {
      "id": "verify-curriculum-manifests-hardcoded-single-curriculum",
      "location": "scripts/verify_curriculum_manifests.py:93",
      "class": "design_defect",
      "severity": "low",
      "evidence": "The __main__ block at line 93 hardcodes 'curricula/cs336_a1' as the sole path to verify. The verify_curriculum_path() function defined above accepts any path argument, but the script never iterates over all curricula directories. The cp_accelerator curriculum is never validated. The tool presents as a general manifest verifier but only covers one curriculum."
    },
    {
      "id": "generate-ground-truth-private-method-access",
      "location": "scripts/generate_ground_truth.py:121",
      "class": "design_defect",
      "severity": "low",
      "evidence": "Lines 121-123 directly call author._extract_patch_info(...), author._build_system_prompt(), and author._build_user_prompt(...) — all underscore-prefixed private methods of the BugAuthor class. This tightly couples the maintenance script to private implementation details. Any refactoring of BugAuthor internals silently breaks generate_ground_truth.py without a public API contract boundary."
    },
    {
      "id": "migrate-bugs-local-import-json",
      "location": "scripts/migrate_bugs_llm.py:108",
      "class": "design_defect",
      "severity": "low",
      "evidence": "At lines 108-109, inside an 'if success:' conditional block: 'import json' appears. json is a stdlib module that should be imported at module level. If the if-branch is not entered and json is referenced later in the same scope, it would raise NameError. The local import suggests the block was added ad hoc without updating the module-level import section."
    },
    {
      "id": "dead-assert-inside-pytest-raises",
      "location": "tests/test_data.py:72",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "Line 72 (`assert \"CUDA error\" in str(excinfo.value) or \"Torch not compiled with CUDA enabled\" in str(excinfo.value)`) is inside the `with pytest.raises((RuntimeError, AssertionError)) as excinfo:` block opened at line 62. When `run_get_batch` raises (the expected path, lines 66-71), Python exits the `with` block at the point of the exception, never reaching line 72. If `run_get_batch` does NOT raise, `pytest.raises` itself raises `Failed: DID NOT RAISE` before line 72 executes. The assertion is unreachable dead code in every execution path; it should be placed outside the `with` block."
    },
    {
      "id": "decoder-lm-uses-encoder-api",
      "location": "tests/one_d_probes.py:68-83",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "Class is named `DecoderOnlyLM` (line 68) but its body instantiates `nn.TransformerEncoderLayer` (line 74) and `nn.TransformerEncoder` (line 83) — PyTorch's encoder-stack implementation. Causal masking is applied via `_mask()` (line 86-89) to emulate autoregressive behaviour, but the underlying module type is an encoder. The class name promises a decoder-only LM architecture, yet the implementation delegates to the encoder API, creating a structural intent mismatch for any reader trying to understand or reuse the probe."
    },
    {
      "id": "module-level-seed-mutation-on-import",
      "location": "tests/one_d_probes.py:10-12",
      "class": "design_defect",
      "severity": "low",
      "evidence": "Lines 10-12 execute `random.seed(SEED)`, `np.random.seed(SEED)`, `torch.manual_seed(SEED)` at module scope. Because the file lives in `tests/`, any pytest plugin, conftest, or test that imports (even indirectly) this module will mutate global RNG state as a side-effect of import. pytest collects all modules in the `tests/` tree; though it won't run test functions (none exist here), the import side-effect fires regardless and can non-deterministically alter other tests' random draws."
    },
    {
      "id": "one-d-probes-not-collected-as-tests",
      "location": "tests/one_d_probes.py:1",
      "class": "design_defect",
      "severity": "low",
      "evidence": "The file is placed inside `tests/` alongside pytest test modules but contains no `test_*` functions and is not named `test_*.py`. It is a standalone training script (entry point: `if __name__ == \"__main__\":` at line 146). It will never be collected by pytest, so any correctness it is meant to probe is never automatically verified by CI. The script could silently break without any test failure alerting developers."
    },
    {
      "id": "stale-1000-iter-comment",
      "location": "tests/test_serialization.py:72",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "Line 72 reads `# Use 1000 optimization steps for testing`, yet `num_iters = 10` is set at line 62 and used in `for _ in range(num_iters):` at line 73. The comment appears to be a copy-paste from `test_optimizer.py` where `_optimize()` genuinely runs 1000 steps (line 18: `for _ in range(1000)`). The discrepancy is misleading: a reader expecting deep optimizer convergence testing will find only 10 iterations."
    },
    {
      "id": "memory-limit-void-on-generator-function",
      "location": "tests/test_tokenizer.py:450-455",
      "class": "design_defect",
      "severity": "high",
      "evidence": "The `memory_limit` decorator (defined at lines 19-36) wraps `f` in a `wrapper` that: (1) sets `resource.RLIMIT_AS`, (2) calls `result = f(*args, **kwargs)`, (3) executes `finally: resource.setrlimit(resource.RLIMIT_AS, prev_limits)`, (4) returns `result`. `_encode_iterable` at line 450 uses `yield from tokenizer.encode_iterable(iterable)` (line 455), making it a Python generator function. Calling `f(*args, **kwargs)` on a generator function returns a generator object immediately without executing any body; the `finally` block then resets the memory limit before a single byte is encoded. The test `test_encode_iterable_memory_usage` (lines 416-430) iterates the generator after `wrapper` has already returned and the limit is gone — the 1 MB ceiling is never in force during actual encoding. The test cannot detect a memory-inefficient `encode_iterable` implementation."
    },
    {
      "id": "bpe-exact-merge-assertion-suppressed",
      "location": "tests/test_train_bpe.py:54",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "Line 54: `# assert merges == reference_merges  # Too strict - commented out`. The original exact-match assertion against the reference BPE merges has been disabled. Its replacement (lines 57-58) only checks `len(merges) >= 243` and `len(merges) <= 245` — a range that allows a 2-element deviation in count without verifying any merge content. A BPE implementation that produces 244 entirely wrong merges would pass. The justification comment (tie-breaking at index 64) does not justify abandoning merge-content validation entirely."
    },
    {
      "id": "tiktoken-ids-equality-check-suppressed",
      "location": "tests/test_tokenizer.py:184",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "In `test_ascii_string_matches_tiktoken`, line 184 reads `# assert ids == reference_ids`. The test's declared intent (confirmed by its name and lines 188-189 which still assert roundtrip equality) is to verify that the student tokenizer produces the same token IDs as tiktoken. However the actual ID-equality check is commented out. The remaining assertions verify only per-token string decoding and roundtrip fidelity, not ID equivalence. An implementation that produces different IDs but happens to decode identically would silently pass a test that claims to verify tiktoken parity."
    },
    {
      "id": "force-update-dead-parameter-in-snapshot",
      "location": "tests/conftest.py:60",
      "class": "design_defect",
      "severity": "low",
      "evidence": "`NumpySnapshot.assert_match` (conftest.py line ~44) accepts `force_update` as a parameter and assigns it from `self.default_force_update` at line 60, but never consults the value again — the subsequent code always loads the `.npz` snapshot and compares. The same pattern applies to `Snapshot.assert_match` (lines 117-150). The `force_update` parameter is an accepted but dead abstraction: there is no save-and-skip code path, so snapshot creation requires direct code modification rather than the implied API. Any caller passing `force_update=True` receives silent no-op behaviour."
    },
    {
      "id": "cp-readme-ingest-script-nonexistent",
      "location": "curricula/cp_accelerator/README.md:188",
      "class": "doc_code_drift",
      "severity": "high",
      "evidence": "README.md lines 188-196 document `scripts/ingest_cp_content.py` as an existing, runnable script with a concrete CLI invocation: `uv run python scripts/ingest_cp_content.py --module two_pointers_basics`. The script does NOT exist in the repository; `ls scripts/` shows only `generate_module.py` as the content-generation tool. The README presents this as current working infrastructure, but it is aspirational/unimplemented. Students or contributors following these instructions will get a FileNotFoundError."
    },
    {
      "id": "impl-status-ci-enforcement-false",
      "location": "curricula/cp_accelerator/IMPLEMENTATION_STATUS.md:162",
      "class": "doc_code_drift",
      "severity": "high",
      "evidence": "IMPLEMENTATION_STATUS.md:162-185 states '❌ IMPOSSIBLE TO MERGE if: manifest.json was manually edited / JSON schema is invalid / Dependency IDs don't exist' and marks '[x] validate_cp_manifest.yml CI workflow active' (line 241) and '[x] CI passing: Yes' (line 241). However, QUALITY_AUDIT.md:54 documents that the validate_cp_manifest workflow is broken because its scripts check `manifest['modules']` which does not exist in a LIBRARY curriculum — cp_accelerator uses `patterns`. The enforced CI check therefore validates the wrong schema key and provides false assurance. The IMPLEMENTATION_STATUS doc claims enforcement that is structurally broken."
    },
    {
      "id": "quality-audit-insert-before-check-stale",
      "location": "audits/QUALITY_AUDIT.md:49",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "QUALITY_AUDIT.md:49 states 'Bug definition uses `replacement.type: \"move_after\"`, which is not implemented by the AST injector, so injection behavior will be incorrect.' However, the actual file `curricula/cp_accelerator/patterns/hash_table/problems/lc_1/bugs/insert_before_check.json` was rewritten: it now uses `find_and_replace` with `replacement.type: \"replace_with\"` (a supported operation), replacing `complement in seen` with `num in seen`. The metadata note confirms: 'Original bug used unsupported move_after type. Redesigned to use replace_value_with'. The audit finding is now factually wrong — the bug and symptom file are correctly aligned."
    },
    {
      "id": "readme-cs336a1-module-count",
      "location": "README.md:87",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "README.md:87 states '**Modules**: 21 modules (BPE Tokenizer → Full Training Loop)'. The actual `curricula/cs336_a1/manifest.json` contains 22 modules in its `modules` array: ['softmax', 'cross_entropy', 'gradient_clipping', 'linear', 'embedding', 'silu', 'rmsnorm', 'swiglu', 'attention', 'rope', 'multihead_attention', 'transformer_block', 'transformer_lm', 'adamw', 'cosine_schedule', 'data_loader', 'checkpointing', 'training_loop', 'unicode', 'bpe_tokenizer', 'tokenizer_class', 'text_generation']. The manifest's own `description` field also says '21 modules in dependency order' (manifest.json:5) — both the README and the manifest description are off by one, with `unicode` or `text_generation` likely added after the count was written."
    },
    {
      "id": "manifest-cs336a1-description-count",
      "location": "curricula/cs336_a1/manifest.json:5",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "The manifest.json `description` field at line 5 reads 'Complete from-scratch Transformer LM implementation with 21 modules in dependency order'. The `modules` array directly below has 22 entries (confirmed by `python3 -c \"import json; print(len(json.load(open('curricula/cs336_a1/manifest.json'))['modules']))\"`). The manifest contradicts itself: description claims 21, list has 22."
    },
    {
      "id": "build-prompt-dict-literal-resources",
      "location": "curricula/cp_accelerator/patterns/backtracking/problems/lc_78/build_prompt.txt:74",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "The Learning Resources section in all auto-generated build prompts renders resource entries as raw Python dict literals instead of formatted markdown links. Example from lc_78/build_prompt.txt:74: `1. {'type': 'taxonomy', 'url': 'https://github.com/Yassir-aykhlf/DSA-Taxonomies/blob/main/Taxonomies/11. Backtracking.md', 'title': 'Backtracking Taxonomy'}`. This formatting defect appears in at minimum: lc_78, lc_90, lc_704, lc_1342, lc_1486, lc_47, lc_146, lc_460, lc_148 (divide_and_conquer), lc_912 (divide_and_conquer), lc_198, lc_70, lc_435, lc_452, lc_217, lc_219. The content-generation pipeline (scripts/generate_module.py) failed to render resource dicts as markdown hyperlinks. Users see raw Python object syntax instead of clickable links."
    },
    {
      "id": "impl-status-vs-status-topic-count-and-ids",
      "location": "curricula/cp_accelerator/IMPLEMENTATION_STATUS.md:95",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "IMPLEMENTATION_STATUS.md:95 claims '✅ 11 topics (foundation complete)' and shows a dependency graph with IDs like `two_pointers_basics`, `two_pointers_sliding_window`, `binary_search_on_index`, `binary_search_on_answer`, `dp_foundations`, `dp_knapsack`, `graphs_basics`. STATUS.md:5 says 'All 19 DSA Taxonomy patterns parsed' with the actual IDs used in the manifest/curriculum: `two_pointers`, `linked_list`, `hash_table`, `stack_queue`, `binary_search`, `traversal`, `dynamic_programming`, `heap`, `greedy`, `backtracking`, `divide_conquer`, `union_find`, `design`, `trie`, `bit_manipulation`, `segment_tree`, `combinatorics`. The IDs in IMPLEMENTATION_STATUS do not match actual deployed IDs and the topic count conflicts with the deployed 19-pattern curriculum. IMPLEMENTATION_STATUS represents an abandoned design that was superseded."
    },
    {
      "id": "lc46-lc47-pattern-classification-mismatch",
      "location": "curricula/cp_accelerator/patterns/combinatorics_and_number_theory/problems/lc_46/build_prompt.txt:3",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "Permutations (lc_46) and Permutations II (lc_47) are placed under the `combinatorics_and_number_theory` pattern directory, but their LeetCode classification is 'Topics: Array, Backtracking' (lc_46/build_prompt.txt:40, lc_47/build_prompt.txt:38). The pattern overview in both files reads 'Combinatorics and Number Theory provide mathematical tools for counting, arrangement, and number properties...' — this is a generic description that doesn't match the backtracking algorithm learners must implement. The intent of the cp_accelerator curriculum is to teach algorithmic PATTERNS correctly, but these problems teach backtracking recursion, not combinatorics math. They would be better placed under the `backtracking` pattern (which already exists)."
    },
    {
      "id": "readme-cs336-spring-year-inconsistency",
      "location": "README.md:411",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "README.md:85 identifies the cs336_a1 curriculum as 'Stanford CS336' and links to `https://stanford-cs336.github.io/spring2025/` (Spring 2025). README.md:411 in the attribution section says 'adapted from **[Stanford CS336 (Spring 2024)](https://stanford-cs336.github.io/spring2024/)**'. The same curriculum is attributed to two different course years in two places in the same document. One or both URLs/years must be wrong."
    },
    {
      "id": "lc1-build-prompt-sorting-resource",
      "location": "curricula/cp_accelerator/patterns/hash_table/problems/lc_1/build_prompt.txt:102",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "The Two Sum (lc_1) build prompt at line 102 lists `2. https://www.geeksforgeeks.org/sorting-algorithms/` as a learning resource. Two Sum is a hash table problem requiring O(n) one-pass lookup; sorting algorithms are irrelevant and potentially misleading (brute-force sort-based approach is O(n log n), opposite of the intended O(n) hash-table approach). The correct resource should be about hash tables or two-sum specifically."
    },
    {
      "id": "status-md-roadmapresources-path-wrong",
      "location": "curricula/cp_accelerator/STATUS.md:117",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "STATUS.md:117 lists 'Source Files: `RoadmapResources.md` - Roadmap with rating brackets and resources' with no path qualifier, implying the file is in the repo root. The file is actually at `maintenance/RoadmapResources.md` (confirmed by `ls maintenance/`; `ls RoadmapResources.md` at root returns 'NOT at root'). META_AUDIT_DEC_18.md:119 independently corroborates this: 'references RoadmapResources.md at repo root (actual file lives under maintenance/)'. Any developer following STATUS.md to locate this file will not find it."
    },
    {
      "id": "meta-audit-python-version-stale",
      "location": "audits/META_AUDIT_DEC_18.md:219",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "META_AUDIT_DEC_18.md:219 states 'The repository README advertises \"Python 3.10+\" (badge + prerequisites)'. However, the current README.md:6 shows `[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)]` — the badge has since been updated to 3.11+. The meta-audit's characterization of the README is now factually incorrect for this specific point, making it an unreliable basis for the follow-up action on Python version consistency it recommends."
    },
    {
      "id": "quality-audit-status-in-progress-misleading",
      "location": "audits/QUALITY_AUDIT.md:12",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "QUALITY_AUDIT.md:12 sets 'Status: In Progress (Expanded Checklist I–XIII)'. Checklist sections X (Mode Parity), XI (Documentation & UX Consistency), XII (CI/Workflow Scope), and XIII (Dependency/Supply Chain) are all unchecked (`[ ]` not `[x]`) at lines 181-212. The document is simultaneously the 'primary audit artifact' (as labeled in the understanding map) and admits it is incomplete. The Findings Count table (9 high / 22 medium / 3 low) does not include findings that would emerge from the unchecked sections, so the count understates known risks. This creates false confidence in audit completeness."
    },
    {
      "id": "lc1804-missing-problem-statement",
      "location": "curricula/cp_accelerator/patterns/trie/problems/lc_1804/build_prompt.txt:11",
      "class": "doc_code_drift",
      "severity": "high",
      "evidence": "The build_prompt.txt contains placeholder content: '**Difficulty:** Unknown | **Acceptance Rate:** N/A' and '**Topics:** General' with no problem statement, no constraints, and no examples. The file instructs students to 'Function Signature: Derive from the problem examples above' (line 24) but there are no examples above. The inventory describes this file as 'Problem statement, constraints, and implementation instructions for LeetCode 1804 (Implement Trie II with prefix counts).' The content contradicts that role—students cannot implement the problem without a specification."
    },
    {
      "id": "lc1099-missing-problem-statement",
      "location": "curricula/cp_accelerator/patterns/two_pointers/problems/lc_1099/build_prompt.txt:11",
      "class": "doc_code_drift",
      "severity": "high",
      "evidence": "Same placeholder pattern as lc_1804: '**Difficulty:** Unknown | **Acceptance Rate:** N/A' and '**Topics:** General' with no problem statement, constraints, or examples. The instruction 'Function Signature: Derive from the problem examples above' (line 22) references non-existent examples. The inventory describes this as 'Problem statement, constraints, and implementation instructions for LeetCode 1099 (Two Sum Less Than K).' No specification is present."
    },
    {
      "id": "cp-build-prompts-raw-python-dict-resources",
      "location": "curricula/cp_accelerator/patterns/heap_and_priority_queue/problems/lc_215/build_prompt.txt:71",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "All cp_accelerator build_prompt.txt files in the required set render Learning Resources as raw Python dict literals instead of formatted markdown. Example from lc_215: \"1. {'type': 'taxonomy', 'url': 'https://github.com/Yassir-aykhlf/DSA-Taxonomies/...', 'title': 'Heap and Priority Queue Taxonomy'}\". The same pattern appears in: lc_703:89, lc_203:82, lc_237:90, lc_1480:90-92, lc_303:81-83, lc_307:85-89, lc_148:83-85, lc_912:75-77, lc_1003:92, lc_20:96, lc_144:78-82, lc_589:76-80, lc_208:82, lc_167:90-92, lc_547:79-81, lc_684:79-81. The content-generation pipeline emitted serialized dict objects rather than formatted bullets, contradicting the learner-facing doc role of these files."
    },
    {
      "id": "cs336-build-prompts-legacy-submit-command",
      "location": "curricula/cs336_a1/modules/adamw/build_prompt.txt:430",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "The AdamW build_prompt.txt instructs students: 'engine submit-build' (line 430). The same legacy command appears in: attention/build_prompt.txt:366, bpe_tokenizer/build_prompt.txt:338, checkpointing/build_prompt.txt:480. However, the cs336_a1/README.md at line 61 documents the current CLI command as 'uv run mastery submit', and the engine architecture (01-understanding.json provisionalIntent) identifies 'submit-build/justification/fix' as LEGACY commands superseded by 'submit'. Students following the build prompts will run a deprecated command that may not exist or behave differently."
    },
    {
      "id": "adamw-epsilon-placement-mismatch",
      "location": "curricula/cs336_a1/modules/adamw/build_prompt.txt:93",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "The mathematical specification on line 93 shows the AdamW update as: 'θ_{t+1} = θ_t - η (m̂_t / √(v̂_t + ε) + λ θ_t)' — epsilon is INSIDE the square root. However, the implementation pseudocode in Step 9 (line 327-329) shows: 'denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)' — epsilon is added OUTSIDE the square root. These formulas are numerically distinct: √(v̂_t + ε) ≠ √(v̂_t) + ε. For v̂_t → 0 and ε = 1e-8, the first gives 1/√ε ≈ 10000 while the second gives 1/ε = 1e8. The pseudocode implements the standard PyTorch convention (eps outside), but this contradicts the mathematical notation shown to students."
    },
    {
      "id": "bpe-return-type-signature-vs-implementation",
      "location": "curricula/cs336_a1/modules/bpe_tokenizer/build_prompt.txt:137",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "The function signature at line 136-151 declares return type 'tuple[dict[int, str], list[tuple[str, str]]]' — merges are a list of string pairs. However, the sample implementation in Step 3 (lines 199-218) builds merges as: 'merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))' where vocab values are bytes objects (initialized at line 185 as 'vocab = {i: bytes([i]) for i in range(256)}'). The merges list therefore contains tuple[bytes, bytes] pairs, not tuple[str, str]. The string conversion 'vocab_str' at line 224 only converts the vocabulary dict, not the merges list. The declared return type does not match what the provided pseudocode produces."
    },
    {
      "id": "incomplete-merge-symptom-missing-structured-format",
      "location": "curricula/cp_accelerator/patterns/sorting/problems/lc_912/bugs/incomplete_merge_symptom.txt:1",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "The incomplete_merge_symptom.txt uses unstructured plain text (7 lines of prose starting with 'Wrong Answer on Test 2') without the structured markdown format used by all other symptom files in the required set. Compare: missing_base_case_symptom.txt uses '# Bug Symptom: Missing Base Case in Merge Sort' with '## What You'll Observe', '## Expected Behavior', '## Actual Behavior', '## Debugging Guide', '## Conceptual Understanding' sections. Same structured format appears in skip_consecutive_symptom.txt, off_by_one_prefix_symptom.txt, missing_empty_check_symptom.txt, wrong_pointer_move_symptom.txt. The inventory describes this file as 'Symptom description for the incomplete merge bug' but the format diverges from the established pattern."
    },
    {
      "id": "lc589-wrong-learning-resource-dp-fibonacci-video",
      "location": "curricula/cp_accelerator/patterns/traversal/problems/lc_589/build_prompt.txt:80",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "The N-ary Tree Preorder Traversal build prompt (lc_589) lists at line 80: \"{'type': 'video', 'title': 'Dynamic Programming lecture #1 - Fibonacci, iteration vs recursion', 'url': 'https://www.youtube.com/watch?v=YBSt1jYwVfU'}\". A DP/Fibonacci video is unrelated to N-ary tree preorder traversal. The same misplaced resource appears in lc_144/build_prompt.txt:82 for Binary Tree Preorder Traversal. The inventory describes these files as 'implementation instructions' for tree traversal problems; the learning resource contradicts that topic domain."
    },
    {
      "id": "lc203-symptom-bug-descriptor-operator-mismatch",
      "location": "curricula/cp_accelerator/patterns/linked_list/problems/lc_203/bugs/skip_consecutive_symptom.txt:4",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "The symptom file title says '# Bug Symptom: Wrong Comparison Operator' and describes the bug as replacing '!=' with '>'. The bug descriptor JSON (skip_consecutive.json inventory) is named 'skip_consecutive' implying it causes consecutive elements to be skipped. The symptom description at line 4 says 'removes too many elements — not just the target value, but also all elements smaller than or equal to it.' For val=6 with '>' operator, only elements >6 are kept, so all elements ≤6 are removed — the symptom correctly states output '[]'. However, the file name 'skip_consecutive_symptom.txt' implies 'skipping consecutive duplicates' (a different class of bug), while the actual bug is a wrong comparison operator removing non-target elements. The name-to-content drift may mislead developers reading the bug inventory."
    },
    {
      "id": "cosine-schedule-wrong-file-and-function",
      "location": "curricula/cs336_a1/modules/cosine_schedule/build_prompt.txt:135",
      "class": "doc_code_drift",
      "severity": "high",
      "evidence": "build_prompt.txt line 135 says 'FILE TO MODIFY: cs336_basics/optimizer.py' and documents the function signature as 'def lr_cosine_schedule(step, max_lr, min_lr, warmup_steps, max_steps)'. The actual function is 'get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)' in 'cs336_basics/utils.py' (modes/student/cs336_basics/utils.py:65; modes/developer/cs336_basics/utils.py:75). The student stub in modes/student/cs336_basics/optimizer.py contains only the AdamW class, no cosine schedule. The adapter at tests/adapters.py imports 'get_lr_cosine_schedule as _get_lr_cosine_schedule_impl' from 'cs336_basics.utils', confirming utils.py is the correct file. Parameter names also diverge: doc uses 'step/max_lr/min_lr/warmup_steps/max_steps' but code uses 'it/max_learning_rate/min_learning_rate/warmup_iters/cosine_cycle_iters'. The build_prompt at line 326 also documents the wrong test name: 'test_lr_cosine_schedule' vs actual 'test_get_lr_cosine_schedule' (tests/test_optimizer.py:52)."
    },
    {
      "id": "cosine-schedule-validator-wrong-file-and-test",
      "location": "curricula/cs336_a1/modules/cosine_schedule/validator.sh:18",
      "class": "bug",
      "severity": "critical",
      "evidence": "validator.sh line 18 copies 'cs336_basics/optimizer.py' to the shadow worktree, but the student must implement 'get_lr_cosine_schedule' in 'cs336_basics/utils.py' (confirmed by tests/adapters.py which imports 'get_lr_cosine_schedule' from 'cs336_basics.utils'). Line 33 runs 'pytest tests/test_optimizer.py::test_lr_cosine_schedule' but the test file contains no such function; the actual test is 'def test_get_lr_cosine_schedule():' at tests/test_optimizer.py:52. Pytest would exit with 'ERROR: not found' on test_lr_cosine_schedule, making the cosine_schedule Build stage permanently unpassable through this validator as written."
    },
    {
      "id": "cross-entropy-build-prompt-wrong-test-name",
      "location": "curricula/cs336_a1/modules/cross_entropy/build_prompt.txt:98",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "build_prompt.txt line 98 says 'The validator will run pytest tests/test_nn_utils.py::test_cross_entropy_matches_pytorch'. The actual validator (cross_entropy/validator.sh:34) runs 'tests/test_nn_utils.py::test_cross_entropy'. The actual test function in tests/test_nn_utils.py is 'def test_cross_entropy():' at line 27, not 'test_cross_entropy_matches_pytorch'. The validator.sh is correct; the build_prompt documents the wrong test name."
    },
    {
      "id": "gradient-clipping-build-prompt-wrong-test-name",
      "location": "curricula/cs336_a1/modules/gradient_clipping/build_prompt.txt:105",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "build_prompt.txt line 105 says 'The validator will run pytest tests/test_nn_utils.py::test_gradient_clipping_matches_pytorch'. The actual validator (gradient_clipping/validator.sh:34) runs 'tests/test_nn_utils.py::test_gradient_clipping'. The actual test function in tests/test_nn_utils.py is 'def test_gradient_clipping():' at line 62, not 'test_gradient_clipping_matches_pytorch'. The validator.sh is correct; the build_prompt documents the wrong test name."
    },
    {
      "id": "embedding-double-embedding-attribute-in-visualization",
      "location": "curricula/cs336_a1/modules/embedding/build_prompt.txt:336",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "build_prompt.txt line 336 shows visualization code: 'E = model.embedding.embedding.weight  # (vocab_size, d_model)'. The Embedding class (modes/developer/cs336_basics/layers.py:51) stores its weight matrix as 'self.weight = nn.Parameter(torch.empty(...))' with no sub-attribute named 'embedding'. Accessing 'model.embedding.embedding' would raise AttributeError at runtime. The correct access path is 'model.embedding.weight'. The build_prompt misleads students into a double-attribute access that cannot work."
    },
    {
      "id": "data-parsing-extraction-wrong-implementation-file",
      "location": "curricula/job_prep_data_annotation/modules/data_parsing_extraction/build_prompt.txt:22",
      "class": "doc_code_drift",
      "severity": "high",
      "evidence": "build_prompt.txt line 22 says 'Implement the following function in job_prep/parser.py'. However, the validator (data_parsing_extraction/validator.sh:17) imports with 'from cs336_basics.utils import extract_coordinates'. If the student implements in 'job_prep/parser.py' as instructed, the validator will fail with ImportError since it looks in 'cs336_basics.utils'. The curriculum README (job_prep_data_annotation/README.md:183) also says 'cs336_basics/utils.py', contradicting the build_prompt. The build_prompt file path instruction is wrong."
    },
    {
      "id": "hello-world-wrong-cli-flag-syntax",
      "location": "curricula/dummy_hello_world/modules/hello_world/build_prompt.txt:18",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "build_prompt.txt line 18 shows 'engine --submit-build' (treating it as a flag with double-dash). The correct CLI invocation is 'engine submit-build' (subcommand without dashes), as confirmed by the engine architecture (engine/main.py:1354 where submit-build is a Typer CLI subcommand). The other build_prompts in the codebase consistently use 'engine submit-build' (e.g., cosine_schedule/build_prompt.txt:381). The '--' prefix is incorrect CLI syntax for this tool."
    },
    {
      "id": "softmax-v2-symptom-identical-to-v1",
      "location": "curricula/cs336_a1/modules/softmax/bugs/no_subtract_max_v2_symptom.txt:1",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "no_subtract_max_v2_symptom.txt is byte-for-byte identical to no_subtract_max_symptom.txt (both read: '# Bug Symptom: Numerical Overflow in Softmax' with the same failing test case, error message, debugging tips, and hint). The understanding.json describes v2 as 'matching the v2 injection spec' and the v2 spec (no_subtract_max_v2.json) uses 'an alternative two-pass find_and_track then find_and_replace pattern'. Despite differing injection mechanics, both symptom files are completely identical — v2 provides no distinct harden-stage guidance, which is inconsistent with the intent of having two separate symptom files."
    },
    {
      "id": "training-loop-wrong-cosine-function-name",
      "location": "curricula/cs336_a1/modules/training_loop/build_prompt.txt:55",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "training_loop/build_prompt.txt uses 'cosine_schedule(step, ...)' at line 55, 'cosine_schedule(step, max_lr, min_lr, warmup_steps, max_steps)' at lines 175 and 418 in implementation examples and pseudo-code. The actual function available to students is 'get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)' in 'cs336_basics/utils.py' (modes/student/cs336_basics/utils.py:65). A student implementing the training loop while following these code examples verbatim would get NameError/TypeError because neither the function name 'cosine_schedule' nor the parameter names ('step', 'max_lr', 'min_lr', 'warmup_steps', 'max_steps') match the actual API."
    },
    {
      "id": "job-prep-readme-wrong-file-for-data-parsing",
      "location": "curricula/job_prep_data_annotation/README.md:83",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "job_prep_data_annotation/README.md line 83 says '# 2. Implement in cs336_basics/utils.py' as the workflow instruction. The data_parsing_extraction/build_prompt.txt:22 contradicts this by saying 'Implement the following function in job_prep/parser.py'. The data_parsing_extraction/validator.sh:17 imports 'from cs336_basics.utils import extract_coordinates', confirming the README and validator agree while the build_prompt has the wrong file. This three-way inconsistency leaves the student with contradictory instructions between the README and the module's own build prompt."
    },
    {
      "id": "cosine-schedule-intent-mismatch-optimizer-vs-utils",
      "location": "curricula/cs336_a1/modules/cosine_schedule/build_prompt.txt:148",
      "class": "intent_mismatch",
      "severity": "high",
      "evidence": "The provisional intent describes cs336_a1 modules as requiring Build-Justify-Harden with validated test harnesses. The cosine_schedule module's build_prompt presents a standalone 'lr_cosine_schedule(step, max_lr, min_lr, warmup_steps, max_steps)' function in optimizer.py, but the engine's actual test infrastructure (tests/adapters.py) is wired to 'get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)' in utils.py. The student experience contradicts the intent: they implement per the prompt (optimizer.py) but the harness tests a completely different signature in a different file. This is a design-level disconnect, not just a naming slip."
    },
    {
      "id": "mastery-engine-doc-old-cli",
      "location": "docs/architecture/MASTERY_ENGINE.md:240",
      "class": "doc_code_drift",
      "severity": "high",
      "evidence": "MASTERY_ENGINE.md documents the CLI using a flag-based interface (`engine --next`, `engine --submit-build`, `engine --submit-fix`, `engine --status`) throughout the usage examples section. The actual registered CLI entry point per pyproject.toml [project.scripts] is `mastery` with subcommands: `mastery submit`, `mastery status`, `mastery show`, `mastery harden`, `mastery justify`. Confirmed as the live interface by STRANGER_TEST_RESULTS.md (which uses `mastery init cs336_a1`, `mastery status`, etc.) and JUSTIFY_ONLY_MODULE_DESIGN.md UX section (lines 149-186), which both use `mastery` commands."
    },
    {
      "id": "mastery-engine-doc-workspace-model",
      "location": "docs/architecture/MASTERY_ENGINE.md:499",
      "class": "doc_code_drift",
      "severity": "high",
      "evidence": "MASTERY_ENGINE.md describes the harden stage challenge file as `workspace/module_challenge.py` and references a `workspace/` directory for all student code. Actual implementation uses a shadow worktree at `.mastery_engine_worktree/` (confirmed by STRANGER_TEST_RESULTS.md lines 100-111 showing `SHADOW_WORKTREE_DIR / 'cs336_basics'` and `os.symlink(symlink_target, shadow_symlink)`, and LAYER2_E2E_SUCCESS.md:61-64 showing `shadow_worktree = shadow_worktree / 'cs336_basics'`). No `workspace/` directory appears in the actual file inventory."
    },
    {
      "id": "mastery-engine-doc-layer4-5-aspirational",
      "location": "docs/architecture/MASTERY_ENGINE.md:48",
      "class": "doc_code_drift",
      "severity": "high",
      "evidence": "MASTERY_ENGINE.md labels Layers 4 and 5 as 'Finalized Design,' implying implementation completeness. AI_CODEBASE_DECONSTRUCTION.md §8 'Honest Status' at line 384 explicitly contradicts this: 'Justify's LLM grader is not wired — but the components exist... the stage runner (engine/stages/justify.py) is a stub, so only the keyword fast-filter is live.' Additionally, AI_CODEBASE_DECONSTRUCTION.md:3 explicitly says the document is 'a design analysis/blueprint' that is 'not a shipped feature.' The MASTERY_ENGINE.md presents aspirational layer architecture as finalized."
    },
    {
      "id": "ai-deconstruction-justify-stub-stale",
      "location": "docs/architecture/AI_CODEBASE_DECONSTRUCTION.md:384",
      "class": "doc_code_drift",
      "severity": "high",
      "evidence": "AI_CODEBASE_DECONSTRUCTION.md §8 states 'Justify's LLM grader is not wired — but the components exist... the stage runner (engine/stages/justify.py) is a stub, so only the keyword fast-filter is live.' This directly contradicts FINAL_VERIFICATION_SUMMARY.md:40 which reports engine/stages/justify.py at 95% test coverage, and lines 59-64 which list 8 passing LLM integration tests including `test_llm_accepts_correct_answer` and `test_llm_rejects_incomplete_answer`. The §8 honest-status section was accurate when written but became stale after justify was implemented, yet the document still resides in docs/architecture/ presenting an outdated picture."
    },
    {
      "id": "cp-quickstart-deleted-modules-dir",
      "location": "docs/internal/CP_ACCELERATOR_QUICKSTART.md:51",
      "class": "doc_code_drift",
      "severity": "high",
      "evidence": "CP_ACCELERATOR_QUICKSTART.md at lines 51-55 references `curricula/cp_accelerator/modules/sorting/`, `curricula/cp_accelerator/modules/two_pointers_basics/`, and similar `modules/`-rooted paths for curriculum content. PHASE_8_BATCH_GENERATION_COMPLETE.md:113 explicitly states 'Deleted: curricula/cp_accelerator/modules/' and confirms the current structure uses a `patterns/` hierarchy (e.g., `patterns/sorting/`, `patterns/arrays/`). The quickstart guide describes a file structure that was entirely deleted."
    },
    {
      "id": "mastery-engine-doc-solutions-dir",
      "location": "docs/architecture/MASTERY_ENGINE.md:139",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "MASTERY_ENGINE.md at lines 122-149 describes a `.solutions/` private directory structure holding reference implementations. This directory does not appear in the actual file inventory (audit/.work/01-understanding.json). The actual mode-switching system uses `modes/student/cs336_basics` and `modes/developer/cs336_basics` symlink targets, confirmed by LAYER2_E2E_SUCCESS.md:48 (`shutil.copytree(real_repo / 'modes', test_repo / 'modes')`) and REAL_STUDENT_UAT_MODULE1.md:13 (`./scripts/mode switch student`)."
    },
    {
      "id": "mastery-engine-doc-wrong-workflow-name",
      "location": "docs/architecture/MASTERY_ENGINE.md:1019",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "MASTERY_ENGINE.md references a CI workflow file named `validate_curriculum.yml` at lines 1019 and 1027. The actual GitHub Actions workflow file is `.github/workflows/validate_cp_manifest.yml` per the 01-understanding.json inventory (listed under CI workflows). The filename in the blueprint does not match the actual file on disk."
    },
    {
      "id": "std-lib-augmentation-status-conflict",
      "location": "docs/internal/PYTHON_CURRICULA_IMPLEMENTATION.md:165",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "PYTHON_CURRICULA_IMPLEMENTATION.md:165 shows `std_lib_augmentation` module status as 'PLANNED ⏸️'. CRITICAL_REVIEW_RESPONSE.md:149 shows the same module listed as 'COMPLETE ✅'. These two internal documentation files directly contradict each other on the implementation status of the same module, making the true state unverifiable from documentation alone without reading the actual curriculum directory."
    },
    {
      "id": "session-docs-engine-module-vs-mastery-cli",
      "location": "docs/internal/archive/sessions/2025-11-09_verification/FINAL_VERIFICATION_SUMMARY.md:186",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "FINAL_VERIFICATION_SUMMARY.md:186-196 documents all 9 CLI commands using `engine` as the command name (e.g., `engine init <curriculum_id>`, `engine submit`, `engine show [module_id]`). All UAT session docs (LAYER4_UAT_FINDINGS.md:62, LAYER4_UAT_EXECUTION_GUIDE.md:55, REAL_STUDENT_UAT_MODULE1.md:20) consistently invoke the CLI as `uv run python -m engine.main ...`. The actual registered CLI entry point (per pyproject.toml [project.scripts]) is `mastery`. These verification documents describe superseded invocation patterns that differ from the shipped command name."
    },
    {
      "id": "justify-only-manifest-doc-contradicts-itself",
      "location": "docs/internal/archive/deprecated/JUSTIFY_ONLY_MODULE_DESIGN.md:229",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "JUSTIFY_ONLY_MODULE_DESIGN.md 'Completed' section (lines 229-235) marks as done: '✅ Manifest updated with module_type: justify_only field' and '✅ Unicode module created.' The 'Pending Engine Implementation' section immediately following (lines 237-242) contradicts this by listing as NOT done: 'Schema updates to support module_type field ⏸️', 'State management updates for justify-only progression ⏸️', 'Command validation (error on build/harden for justify-only) ⏸️'. The document's own completion status section is internally inconsistent: the data field is in the manifest but the engine code to interpret it is explicitly marked pending."
    },
    {
      "id": "intent-module-type-field-ignored-by-engine",
      "location": "docs/internal/archive/deprecated/JUSTIFY_ONLY_MODULE_DESIGN.md:237",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "The design intent (JUSTIFY_ONLY_MODULE_DESIGN.md §Implementation Requirements, lines 63-144) is that `module_type: 'justify_only'` in a manifest should cause the engine to skip build/harden stages, start directly at justify, and error on `mastery build`. The unicode module manifest was updated with `module_type: 'justify_only'` (marked COMPLETED at line 232). However, engine/schemas.py, engine/state.py, engine/curriculum.py, and engine/main.py were explicitly NOT updated (all marked PENDING at lines 237-242). The manifest field expresses design intent that the engine silently ignores at runtime — the unicode module would be treated as a standard module requiring build stage."
    },
    {
      "id": "readme-bug-injection-guide-path",
      "location": "docs/README.md:67",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "docs/README.md:67 references BUG_INJECTION_GUIDE.md using path `docs/internal/current/BUG_INJECTION_GUIDE.md`. CLEANUP_SUMMARY.md describes the canonical operational docs structure as placing current guides under `docs/current/` (not `docs/internal/current/`). Path divergence unverified via directory listing — marking as low-severity because the actual file location could be either path; cannot confirm without a filesystem check."
    },
    {
      "id": "repo-analysis-mislocated-in-architecture-dir",
      "location": "docs/architecture/REPO_ANALYSIS.md:1",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "REPO_ANALYSIS.md (generated 2025-09-15 per its header) is a CS336 Assignment 1 course analysis document that describes implementing `tests/adapters.py` and course exercise solutions — it analyzes the Stanford CS336 homework, not the Mastery Engine. It is placed in `docs/architecture/` alongside genuine engine architecture documents (MASTERY_ENGINE.md, AI_CODEBASE_DECONSTRUCTION.md), creating a misleading co-location. AI_CODEBASE_DECONSTRUCTION.md:3 distinguishes between 'design analysis/blueprint' and 'shipped features'; REPO_ANALYSIS.md belongs in neither category."
    },
    {
      "id": "engine-critical-fixes-deleted-path-ref",
      "location": "docs/internal/ENGINE_CRITICAL_FIXES_2025-11-18.md:19",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "ENGINE_CRITICAL_FIXES_2025-11-18.md:19 references `curricula/cp_accelerator/modules/sorting/test_cases.json` as the bug location that was fixed (wrong test cases in sorting module). PHASE_8_BATCH_GENERATION_COMPLETE.md:113 states 'Deleted: curricula/cp_accelerator/modules/' — the entire `modules/` directory was subsequently removed and replaced with `patterns/`. This archived fix document describes a path that no longer exists on disk."
    },
    {
      "id": "harden-phase2-dispatch-stale",
      "location": "docs/internal/archive/sessions/2025-11-10_bug_system/AST_HARDEN_PHASE2_COMPLETE.md:101",
      "class": "doc_code_drift",
      "severity": "high",
      "evidence": "AST_HARDEN_PHASE2_COMPLETE.md line 101 states the production harden dispatch uses 'from engine.services.ast_service import SoftmaxBugInjector'. Actual production code at engine/stages/harden.py:99 imports 'from engine.ast_harden.generic_injector import GenericBugInjector'. Phase 3 replaced SoftmaxBugInjector with GenericBugInjector but the Phase 2 document was never updated, leaving a stale record of the production dispatch path."
    },
    {
      "id": "phase2-signoff-stale-injector",
      "location": "docs/internal/archive/sessions/2025-11-10_bug_system/PHASE2_SIGNOFF.md:27",
      "class": "doc_code_drift",
      "severity": "high",
      "evidence": "PHASE2_SIGNOFF.md lines 27-28 formally approves 'engine/services/ast_service.py (367 lines) — SoftmaxBugInjector' as 'Production quality, fully functional.' Phase 3 superseded this with GenericBugInjector at engine/ast_harden/generic_injector.py. The production harden.py (line 99) no longer imports SoftmaxBugInjector; this signoff now documents a deprecated artifact as the approved production component."
    },
    {
      "id": "harden-fix-verify-primary-path-misleading",
      "location": "docs/internal/archive/sessions/2025-11-10_bug_system/HARDEN_FIX_VERIFICATION.md:25",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "HARDEN_FIX_VERIFICATION.md lines 25-29 presents 'After (Working): Copy developer's code to harden workspace' as THE primary fix for harden bug injection. In the current engine/stages/harden.py (lines 97-125), the primary path for .json bug files injects into the STUDENT'S own code via GenericBugInjector; copying developer code (lines 127-157) is only the legacy else-branch for .patch files. The document inverts primary/secondary status of these two paths."
    },
    {
      "id": "harden-class-docstring-patch-only",
      "location": "engine/stages/harden.py:33",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "HardenRunner class docstring at harden.py lines 31-36 describes the Harden stage as '1. Copying their validated Build submission / 2. Applying a pedagogical bug patch'. The .json dispatch path (harden.py:97-125) uses AST-based bug injection via GenericBugInjector with no patch applied at all. The docstring describes only the legacy .patch workflow and omits the primary AST injection path."
    },
    {
      "id": "harden-present-challenge-param-misleading",
      "location": "engine/stages/harden.py:63",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "present_challenge() docstring at harden.py:63 states 'source_file_path: Path to the user's (main workspace) source file used as the target for hardening.' For .patch-based bugs (harden.py:127-157), the function ignores the student's file entirely and copies from the developer reference implementation at modes/developer/<rel_path>. The parameter description is only accurate for the .json AST path."
    },
    {
      "id": "manual-llm-test-deprecated-submit-cmd",
      "location": "docs/internal/archive/sessions/2025-11-10_bug_system/MANUAL_LLM_TEST.md:68",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "MANUAL_LLM_TEST.md line 68 invokes 'submit-justification' as the CLI command for submitting a justify-stage answer. VERIFICATION_PROTOCOL_LAYER2_STATUS.md (2025-11-09_verification session) line 58 documents the migration: 'submit-build -> submit (unified command)' as part of Layer 2 fixes. The correct command per the updated CLI is the unified 'submit', making the MANUAL_LLM_TEST.md example use a removed command."
    },
    {
      "id": "bpe-fix-student-pass-contradicted",
      "location": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/BPE_TEST_FIX_SUMMARY.md:87",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "BPE_TEST_FIX_SUMMARY.md lines 86-89 claims 'Student (stub): Before=FAIL, After=PASS — FIXED', verifying the student bpe test now passes after relaxing assertions. CRITICAL_BUG_RESOLUTION.md (same Nov 13, 2025 date, 2025-11-10_bug_system session) documents that modes/student/cs336_basics/bpe.py was subsequently stubbed out (200+ implementation lines removed, replaced with NotImplementedError stubs). A stubbed-out bpe.py cannot pass the test_train_bpe test. These two session documents, written on the same date, assert contradictory states for the student BPE test."
    },
    {
      "id": "critical-bug-count-mismatch",
      "location": "docs/internal/archive/sessions/2025-11-10_bug_system/CRITICAL_BUG_RESOLUTION.md:7",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "CRITICAL_BUG_RESOLUTION.md line 7 header states '10 out of 22 modules (45% of curriculum)' were affected by the critical bug. The evidence table on lines 78-86 lists only 9 distinct module indices: 1, 2, 3, 11, 15, 16, 17, 20, 21. The heading count (10) does not match the table row count (9)."
    },
    {
      "id": "pattern-matcher-py37-wrong-version",
      "location": "docs/internal/archive/sessions/2025-11-10_bug_system/PATTERN_MATCHER_DEBUG_SESSION.md:33",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "PATTERN_MATCHER_DEBUG_SESSION.md line 33 documents a fix with rationale 'Python 3.7 lacks ast.unparse() — Fix: Fallback to astor.to_source()'. The project's pyproject.toml specifies requires-python = '>=3.11', making the Python 3.7 justification factually incorrect. ast.unparse() was added in Python 3.9; since the project requires 3.11+, no fallback to astor is ever needed and the stated rationale documents a non-existent constraint."
    },
    {
      "id": "final-status-bpe-doc-wrong-session",
      "location": "docs/internal/archive/sessions/2025-11-09_verification/VERIFICATION_PROTOCOL_FINAL_STATUS.md:174",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "VERIFICATION_PROTOCOL_FINAL_STATUS.md line 174 lists 'BPE_TEST_FIX_SUMMARY.md' as an artifact produced in the 2025-11-09_verification session. The actual file resides at docs/internal/archive/sessions/2025-11-11_curriculum_quality/BPE_TEST_FIX_SUMMARY.md — a different session folder dated two days later. The artifact provenance in the final status document is incorrect."
    },
    {
      "id": "harden-patch-path-contradicts-debug-own-code-intent",
      "location": "engine/stages/harden.py:127",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "Provisional intent (audit/.work/01-understanding.json) states the Harden stage 'challenges users to debug their own implementations'. HardenRunner class docstring (harden.py:31) likewise states '1. Copying their validated Build submission'. For .patch-based bugs, harden.py:127-157 ignores the student's code entirely; it copies the developer reference implementation from modes/developer/<rel_path> and applies a patch to that. The student debugs code they did not write. HARDEN_STAGE_CRITICAL_BUG.md documents this as an architectural decision for patch compatibility, but it directly contradicts the stated pedagogical intent of debugging one's own code."
    },
    {
      "id": "coverage-claim-100-vs-16-percent",
      "location": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/CURRICULUM_COVERAGE.md:5",
      "class": "doc_code_drift",
      "severity": "critical",
      "evidence": "CURRICULUM_COVERAGE.md:5 states 'It demonstrates **100% coverage** of all required implementations'. CURRICULUM_GAP_ANALYSIS.md:5, in the same session folder, states 'Modules Implemented: 3 / ~19 components (16% coverage)'. Mutually exclusive coverage claims for the same CS336 A1 curriculum from documents produced in the same session; no reconciliation is provided."
    },
    {
      "id": "bug-golden-patterns-100-vs-14-percent",
      "location": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/GROUND_TRUTH_COMPLETE.md:2",
      "class": "doc_code_drift",
      "severity": "critical",
      "evidence": "GROUND_TRUTH_COMPLETE.md:2 says 'All 21/21 curriculum modules have validated golden patterns' and line 19 says 'Final coverage: 21/21 modules (100%)'. PROJECT_STATUS.md:141 in the same session folder says '14% Complete' — only 3 golden dataset bugs exist (softmax, silu, rmsnorm verified at PROJECT_STATUS.md:145-157), with 18 modules listed as 'Ready for Migration ⏳ Pending'. Same denominator (21 modules) but opposite conclusions (100% vs 14%)."
    },
    {
      "id": "remediation-summary-100-vs-gap-analysis-16",
      "location": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/REMEDIATION_SUMMARY.md:256",
      "class": "doc_code_drift",
      "severity": "critical",
      "evidence": "REMEDIATION_SUMMARY.md:256 states '✅ 100% implementation coverage (21/21 modules implemented)'. CURRICULUM_GAP_ANALYSIS.md:5, in the same session folder, states 'Modules Implemented: 3 / ~19 components (16% coverage)'. Both purport to describe the current state of the CS336 A1 curriculum; the claims are mutually exclusive."
    },
    {
      "id": "cli-p0-completion-contradiction",
      "location": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/MASTER_REMEDIATION_STATUS.md:5",
      "class": "doc_code_drift",
      "severity": "high",
      "evidence": "MASTER_REMEDIATION_STATUS.md:5 (date 2025-11-12) says 'Overall Status: ✅ Curriculum Complete (98/100), ✅ CLI P0 Complete (100%)'. SESSION_3_SUMMARY.md:56 (same date, same session folder) says 'Implementation Phase 🟡 STARTED (10%)' for CLI, and SESSION_3_SUMMARY.md:265 shows 'Total: 10% Complete'. Two documents produced the same day describe CLI P0 as simultaneously 100% complete and 10% started."
    },
    {
      "id": "cli-p1-done-vs-pending",
      "location": "docs/internal/archive/sessions/2025-11-12_cli_remediation/CLI_P1_IMPLEMENTATION_COMPLETE.md:6",
      "class": "doc_code_drift",
      "severity": "high",
      "evidence": "CLI_P1_IMPLEMENTATION_COMPLETE.md:6 says 'Status: ✅ **COMPLETE**' for P1 (Inconsistent next command). CLI_REMEDIATION_STATUS.md:87, in the same session folder, says P1 is '📋 **Designed, awaiting implementation**'. CLI_REMEDIATION_STATUS.md:209 also shows '⏸️ Pending' for P1 in its implementation tracking table. Two files in the same session folder (2025-11-12_cli_remediation) give opposite completion states for the same work item."
    },
    {
      "id": "quality-plan-production-ready-vs-deferred-tasks",
      "location": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/QUALITY_REMEDIATION_PLAN.md:249",
      "class": "doc_code_drift",
      "severity": "high",
      "evidence": "QUALITY_REMEDIATION_PLAN.md:249 says '**Status**: All Priorities Complete (P1, P2, P3) - PRODUCTION READY'. The tracking table in the same document (line 240) shows P1 task 'Update build prompts (einops)' at '⏸️ Deferred / 0%', and line 243 shows P2 task 'Create experiment modules' at '⏸️ Design Only / 30%'. The document footer asserts all priorities complete while the table above it shows two uncompleted tasks."
    },
    {
      "id": "curriculum-stub-claim-vs-student-fix",
      "location": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/CURRICULUM_COVERAGE.md:360",
      "class": "doc_code_drift",
      "severity": "high",
      "evidence": "CURRICULUM_COVERAGE.md:360 says '✅ 18 components properly stubbed' and '✅ All raise NotImplementedError'. STUDENT_MODE_FIX_SUMMARY.md:58 says 'Modules Affected: 10 out of 22 (45% of curriculum!)' — these modules had complete working implementations instead of NotImplementedError stubs. STUDENT_MODE_AUDIT.md documents 5 functions explicitly marked '❌ COMPLETE (should be stub)'. The coverage document asserts a clean state that pre-fix code directly contradicted."
    },
    {
      "id": "module-count-21-vs-22",
      "location": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/CURRICULUM_COVERAGE.md:411",
      "class": "doc_code_drift",
      "severity": "high",
      "evidence": "CURRICULUM_COVERAGE.md:411 says 'Total: 21 modules, 18 stubbed components, 100% PDF coverage'. PROJECT_STATUS.md:133 says 'Total Modules: 22'. The same CS336 A1 curriculum is reported as having 21 and 22 modules; neither document explains which extra module accounts for the discrepancy. STUDENT_MODE_FIX_SUMMARY.md:58 also uses 22 as the denominator ('10 out of 22')."
    },
    {
      "id": "cp-quickstart-hardcoded-macos-path",
      "location": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/CP_ACCELERATOR_QUICKSTART.md:84",
      "class": "doc_code_drift",
      "severity": "high",
      "evidence": "CP_ACCELERATOR_QUICKSTART.md:84 contains `cd /Volumes/Totallynotaharddrive/assignment1-basics` — a hardcoded developer-local macOS volume path. This path does not exist in the repository. Any user following the quickstart guide verbatim will receive a 'No such file or directory' error. The path is non-portable and unreproducible outside the original author's machine."
    },
    {
      "id": "cp-docs-modules-dir-vs-patterns",
      "location": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/CP_ACCELERATOR_IMPLEMENTATION_GUIDE.md:24",
      "class": "doc_code_drift",
      "severity": "high",
      "evidence": "CP_ACCELERATOR_IMPLEMENTATION_GUIDE.md:24 shows directory tree `curricula/cp_accelerator/modules/` with pattern subdirectories. CP_ACCELERATOR_QUICKSTART.md:37 also uses `└── modules/`. Actual repo at /home/user/mastery-engine/curricula/cp_accelerator/ contains a `patterns` subdirectory, not `modules` (confirmed via ls). All command examples in both guides referencing `.../cp_accelerator/modules/...` (e.g., QUICKSTART.md:117, :205, :301) point to a non-existent path."
    },
    {
      "id": "verification-findings-hardcoded-path",
      "location": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/VERIFICATION_FINDINGS.md:11",
      "class": "doc_code_drift",
      "severity": "high",
      "evidence": "VERIFICATION_FINDINGS.md:11 cites `/Users/tomriddle1/Holistic-Performance-Enhancement/cultivation/docs/5_domain_knowledge_and_curricula/computer_science/architectures_and_models/transformer_paradigm/RoFormer_Analysis.md` as the literature source for rope module verification. LITERATURE_VERIFICATION.md:12 uses the same hardcoded local path. These paths point to a non-repo filesystem unavailable to any other user. All verification grades (e.g., rope 85/100, linear 95/100 at VERIFICATION_FINDINGS.md:200-237) rest on sources that cannot be independently validated."
    },
    {
      "id": "student-mode-complete-impls-vs-stub-intent",
      "location": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/STUDENT_MODE_FIX_SUMMARY.md:58",
      "class": "intent_mismatch",
      "severity": "high",
      "evidence": "STUDENT_MODE_FIX_SUMMARY.md:58 says 'Modules Affected: 10 out of 22 (45% of curriculum!)' — nearly half the CS336 A1 curriculum's student mode files contained complete working implementations instead of NotImplementedError stubs. The provisional pedagogical intent requires student mode to provide only stubs so learners must implement from scratch in the Build stage. Complete implementations in student mode directly undermine the Build-Justify-Harden loop: a student can copy the reference solution without building understanding. STUDENT_MODE_AUDIT.md documents the specific functions (e.g., transformer_lm, multihead_attention, etc.) that were fully implemented when they should have been stubs."
    },
    {
      "id": "bpe-line-count-141-vs-350",
      "location": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/REMEDIATION_PROGRESS.md:29",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "REMEDIATION_PROGRESS.md:29 says 'Implementation: `modes/developer/cs336_basics/bpe.py` (~141 lines)'. MASTER_REMEDIATION_STATUS.md:29 says 'From-scratch BPE training (~350 lines)'. A 2.5x discrepancy (141 vs 350 lines) for the same artifact. REMEDIATION_SUMMARY.md describes the BPE as using 'heap-based priority queue, doubly-linked list' which would imply a larger codebase. The reported size of the delivered artifact is inconsistent across three documents."
    },
    {
      "id": "tokenizer-path-cs336basics-vs-modes-developer",
      "location": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/REMEDIATION_PROGRESS.md:64",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "REMEDIATION_PROGRESS.md:64 says the new from-scratch Tokenizer was created at `cs336_basics/tokenizer.py`. QUALITY_REMEDIATION_PLAN.md:35 specifies the remediation target as `modes/developer/cs336_basics/tokenizer.py`. TOKENIZER_VIOLATIONS_AUDIT.md also refers to the violation at `modes/developer/cs336_basics/tokenizer.py`. The created file and the required target path differ: one strips the `modes/developer/` prefix, determining whether the file is the reference implementation (correct location) or an ambiguous standalone file (incorrect location)."
    },
    {
      "id": "cli-audit-next-docstring-false-claim",
      "location": "docs/internal/archive/sessions/2025-11-12_cli_remediation/CLI_INTERFACE_AUDIT.md:96",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "CLI_INTERFACE_AUDIT.md:96 quotes the `next` command docstring verbatim: 'Only works when the user is in the \"build\" stage.' CLI_INTERFACE_AUDIT.md:120 in the same document says 'Misleading Documentation: Docstring says \"only works when in build stage\" - FALSE, it works for all 3 stages (build, justify, harden)'. The audit document identifies the false docstring but lists it only as a 'Weakness' without assigning a remediation priority (P0/P1/P2), so the false docstring was not scheduled for correction."
    },
    {
      "id": "p0-progress-line-count-329-vs-410-vs-470",
      "location": "docs/internal/archive/sessions/2025-11-12_cli_remediation/CLI_P0_PROGRESS.md:204",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "Three documents report different line counts for the same P0 implementation: (1) CLI_P0_PROGRESS.md:204 Files Modified table says '+329 lines'; (2) CLI_P0_PROGRESS.md:215 Code Statistics says 'Total New Code: ~410 lines' — internal contradiction within the same document; (3) CLI_P0_FINAL_STATUS.md reports '~470 lines of production-quality implementation'. The true size of the delivered artifact cannot be determined from documentation alone."
    },
    {
      "id": "cp-quickstart-self-ref-wrong-path",
      "location": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/CP_ACCELERATOR_QUICKSTART.md:67",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "CP_ACCELERATOR_QUICKSTART.md:67-68 states 'docs/CP_ACCELERATOR_IMPLEMENTATION_GUIDE.md - Full technical blueprint' and 'docs/CP_ACCELERATOR_QUICKSTART.md - This file'. Both self-referenced paths are wrong — both files are archived at `docs/internal/archive/sessions/2025-11-11_curriculum_quality/`, not top-level `docs/`. The files were archived but their internal cross-references still point to the original intended (never-created) top-level locations."
    },
    {
      "id": "verification-findings-6-vs-7-modules",
      "location": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/VERIFICATION_FINDINGS.md:456",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "VERIFICATION_FINDINGS.md:456 says 'Status: ALL 6 NEW MODULES VERIFIED ✅'. The summary table at lines 428-436 in the same document lists 7 modules as verified (rope, linear, embedding, tokenizer_class, transformer_lm, data_loader, checkpointing). The status footer claims 6 but the table body contains 7 rows, all marked ✅ Verified."
    },
    {
      "id": "DCD-001",
      "location": "docs/internal/coverage/CURRENT_REPORT.md:29",
      "class": "doc_code_drift",
      "severity": "high",
      "evidence": "Report lists 'engine/ast_harden/harden.py' (98%) and 'engine/justify.py' (95%) as near-perfect-coverage modules. Neither path exists in the repository. Glob of engine/**/*.py confirms actual paths are 'engine/stages/harden.py' and 'engine/stages/justify.py' (verified via audit/.work/01-understanding.json architecture section which names 'engine/stages/harden.py HardenRunner' and 'engine/stages/justify.py JustifyRunner'). The coverage data is therefore attached to phantom module paths, making the report unverifiable and misleading."
    },
    {
      "id": "DCD-002",
      "location": "docs/internal/current/TEST_COVERAGE_REPORT.md:29",
      "class": "doc_code_drift",
      "severity": "high",
      "evidence": "Identical content to docs/internal/coverage/CURRENT_REPORT.md; contains the same wrong module paths 'engine/ast_harden/harden.py' and 'engine/justify.py'. These paths do not exist; the correct paths confirmed by repository glob are 'engine/stages/harden.py' and 'engine/stages/justify.py'. Two authoritative 'current' documents both propagate the same phantom path drift."
    },
    {
      "id": "DCD-003",
      "location": "docs/internal/current/CURRICULUM_STATUS.md:14",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "Table row for cs336_a1 states 'Modules: 21'. Direct count of curricula/cs336_a1/manifest.json via bash yields 22 modules. The off-by-one means module counts in the authoritative status document undercount the actual deployed curriculum by one module."
    },
    {
      "id": "DCD-004",
      "location": "docs/internal/current/CURRICULUM_STATUS.md:87",
      "class": "doc_code_drift",
      "severity": "high",
      "evidence": "Curriculum Status describes cp_accelerator as 'Total Modules: 1 (pilot)' with only the 'sorting' module mentioned. Actual repository state: curricula/cp_accelerator/manifest.json contains 19 patterns (sorting, backtracking, binary_search, bit_manipulation, combinatorics_and_number_theory, design_patterns, divide_and_conquer, dynamic_programming, greedy, hash_table, heap_and_priority_queue, linked_list, prefix_sum, segment_tree_and_fenwick_tree, stack_and_queue, traversal, trie, two_pointers, union_find_disjoint_set_union). The 'ls curricula/cp_accelerator/patterns/' command confirms 19 subdirectories. The document is off by a factor of 19x."
    },
    {
      "id": "DCD-005",
      "location": "docs/internal/current/BUG_INJECTION_GUIDE.md:210",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "Schema Evolution section documents the command: 'engine regenerate-bugs --all' (line 210: '2. Regenerate all .json files: `engine regenerate-bugs --all`'). Grep of engine/main.py for 'regenerate-bugs' returns zero matches. The command does not exist anywhere in the registered Typer application (engine/main.py). Additionally, the documented CLI prefix 'engine' is itself wrong — the console script is registered as 'mastery' in pyproject.toml, not 'engine'."
    },
    {
      "id": "DCD-006",
      "location": "docs/internal/development/CHANGELOG.md:1",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "File is placed under docs/internal/development/ where Mastery Engine development history is expected. Entire content is the Stanford CS336 Spring 2025 Assignment 1 changelog (versions 0.1.0 2024-04-01 through 1.0.6 2025-08-28), covering handout and test-suite changes for that course. Zero Mastery Engine entries exist in the file. Per provisional intent in audit/.work/01-understanding.json, Mastery Engine is a 'curriculum-agnostic pedagogical operating system CLI' that is a separate project from CS336; the changelog conflates curriculum source material with engine development history."
    },
    {
      "id": "DCD-007",
      "location": "docs/internal/development/MASTERY_WORKLOG.md:42",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "MASTERY_WORKLOG.md uses 'engine' as the CLI command name throughout: lines 42-43 ('Ran `engine next`', 'Ran `engine submit-fix`'), lines 681-684 ('`engine next` handles all 3 stages', '`engine submit-build`', '`engine submit-justification`', '`engine submit-fix`'), lines 758-760, and line 787 ('Commands Verified: `engine next`, `engine submit-build`, `engine status`'). pyproject.toml registers the console script as 'mastery', not 'engine'. Every documented CLI invocation in the worklog uses the wrong command name and would fail as written."
    },
    {
      "id": "DCD-008",
      "location": "docs/internal/archive/sessions/2025-11-12_test_coverage/TEST_COVERAGE_IMPROVEMENT_SESSION.md:3",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "Document header states the session date as 'November 13, 2025' while all other files in the same session directory (FINAL_SESSION_REPORT.md, COVERAGE_70_80_ACHIEVEMENT.md, COVERAGE_80_ACHIEVEMENT.md, CURRENT_REPORT.md) consistently record the date as 2025-11-12. The header date is off by one day from all corroborating artefacts."
    },
    {
      "id": "DCD-009",
      "location": "docs/internal/archive/sessions/2025-11-12_test_coverage/FINAL_SESSION_REPORT.md:143",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "FINAL_SESSION_REPORT.md lists 13 engine package modules in its coverage summary table. COVERAGE_70_80_ACHIEVEMENT.md (same session, same date) lists 12 modules in its equivalent table. The discrepancy is within the same session's own artefacts; one of the two authoritative session-end documents has a module count error."
    },
    {
      "id": "DCD-010",
      "location": "docs/internal/archive/sessions/2025-11-12_test_coverage/FINAL_SESSION_REPORT.md:121",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "FINAL_SESSION_REPORT.md records engine/validator.py coverage as 93%. COVERAGE_70_80_ACHIEVEMENT.md (same session directory) records engine/validator.py at 94%. The two end-of-session summary documents report different final coverage values for the same module, creating an internally inconsistent record. The current authoritative report (docs/internal/coverage/CURRENT_REPORT.md:32) states 94%, suggesting FINAL_SESSION_REPORT.md carries a stale/incorrect figure."
    },
    {
      "id": "DCD-011",
      "location": "docs/internal/archive/sessions/2025-11-12_test_coverage/COVERAGE_80_ACHIEVEMENT.md:1",
      "class": "doc_code_drift",
      "severity": "info",
      "evidence": "Document title is 'Coverage Achievement: 76% → 78% ✅ 80% THRESHOLD REACHED'. Every data point in the body reports 78% (line 5: '78% ENGINE COVERAGE', line 11: '78% total coverage', line 15: '78% total engine coverage', line 19: 'Target exceeded: 78% ≈ 80% goal', line 64: 'TOTAL ... 78%'). The title's '80% THRESHOLD REACHED' claim is contradicted by all internal data; the actual measured coverage is 78%, which the document itself characterises as 'Near 80%' and 'approximately' 80% — not as having crossed the 80% threshold."
    },
    {
      "id": "readme-original-engine-cmd-name",
      "location": "maintenance/README_ORIGINAL.md:12",
      "class": "doc_code_drift",
      "severity": "high",
      "evidence": "README_ORIGINAL.md presents the primary Mastery Engine Workflow section using the command `engine` (e.g. `engine init`, `engine status`, `engine next`, `engine submit`) on lines 12-26. The actual CLI entry point registered in pyproject.toml:34 is `mastery = \"engine.main:main\"`, so users who follow these instructions will get 'command not found'. The MASTERY_COMMAND_REFERENCE.md explicitly acknowledges this: 'OLD (Documentation): `engine submit` … ACTUAL (Command): `uv run mastery submit`'."
    },
    {
      "id": "project-structure-engine-cmd-name",
      "location": "maintenance/PROJECT_STRUCTURE.md:168",
      "class": "doc_code_drift",
      "severity": "high",
      "evidence": "PROJECT_STRUCTURE.md Development Workflow section (lines 168-180) uses `engine init cs336_a1`, `engine next`, `engine submit-build`, `engine submit-justification \"<answer>\"`, `engine submit-fix`, `engine status`, and (line 130) `engine init` / `engine cleanup` for the shadow worktree lifecycle. The installed CLI entry point is `mastery` (pyproject.toml:34), not `engine`. Every user-facing command shown in this developer guide is wrong. The same file also references `uv run python -m engine.main next` on line 155 as an example invocation, which conflates module path with CLI name."
    },
    {
      "id": "typer-app-named-engine-not-mastery",
      "location": "engine/main.py:57",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "The Typer app is instantiated with `name=\"engine\"` at line 57: `app = typer.Typer(name=\"engine\", help=\"Mastery Engine: Build, Justify, Harden learning system\", ...)`. Because of this, `uv run mastery --help` displays `Usage: engine [OPTIONS] COMMAND [ARGS]...` — the usage line shows `engine` while the user must type `mastery`. This creates a persistent, built-in confound between the installed CLI name and what the help output says. The intended CLI name is `mastery` (pyproject.toml:34)."
    },
    {
      "id": "next-cmd-deprecation-msg-wrong-cli-name",
      "location": "engine/main.py:1330",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "The docstring and Rich panel for the deprecated `next()` command (lines 1330-1342) instruct users to use `engine show` and `engine start-challenge`: line 1330 says `'engine show' for read-only viewing`, line 1340 prints `engine show`, line 1341 prints `engine start-challenge`, line 1342 prints `Running 'engine show' for you...`. The actual installed CLI is `mastery`, so users see \"use engine show\" but the correct command is `mastery show` / `mastery start-challenge`. The MASTERY_COMMAND_REFERENCE.md confirms the installed name is `mastery`."
    },
    {
      "id": "project-structure-nonexistent-reference-dir",
      "location": "maintenance/PROJECT_STRUCTURE.md:64",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "PROJECT_STRUCTURE.md shows the directory tree (line 64): `└── reference/            # Complete implementations (archived)` with `└── utils_complete.py` under `curricula/cs336_a1/`. Verified with `ls /home/user/mastery-engine/curricula/cs336_a1/reference/` which returns `No such file or directory`. This directory and file do not exist on disk."
    },
    {
      "id": "mvp-status-mainpy-line-count-wrong",
      "location": "docs/internal/development/MVP_COMPLETION_STATUS.md:82",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "MVP_COMPLETION_STATUS.md states under Gap #3 (line 82): \"`engine/main.py` has 1,241 lines with embedded orchestration logic\". Actual line count via `wc -l engine/main.py` is 2,942 lines — 2.4× larger than documented. This understates the extent of the fat-controller problem by more than half and misrepresents the scale of refactoring required."
    },
    {
      "id": "mvp-status-test-count-internal-inconsistency",
      "location": "docs/internal/development/MVP_COMPLETION_STATUS.md:136",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "MVP_COMPLETION_STATUS.md line 136 states \"Total Test Count: 145 engine tests + 22 integration/e2e = 167 automated tests\" but the same document at line 274 (Final Validation section) shows `uv run pytest → ✅ 72/72 tests passing in ~30 seconds`. A grep of test function definitions across all test files yields 72 `def test_` functions. The document presents two irreconcilable test counts (167 vs 72) in the same production-readiness artifact."
    },
    {
      "id": "modes-readme-21-modules-vs-22-in-manifest",
      "location": "modes/README.md:9",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "modes/README.md line 9 states \"Complete Curriculum: 21 Modules\" and enumerates 8+5+5+3=21 modules across four categories without listing `unicode`. The curricula/cs336_a1/manifest.json contains 22 modules (confirmed via Python: `len(d['modules']) == 22`), including a `unicode` module that is absent from the README's list and count. The manifest directory also contains 22 subdirectories including `unicode/`."
    },
    {
      "id": "two-sum-e2e-wrong-validator-path",
      "location": "docs/internal/two_sum_qa/TWO_SUM_E2E_WORKFLOW_TEST.md:43",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "TWO_SUM_E2E_WORKFLOW_TEST.md line 43 shows the build-stage validator command as `cd curricula/cp_accelerator/modules/two_sum && ./validator.sh`. The directory `curricula/cp_accelerator/modules/` does not exist (verified with `ls`). The Two Sum (LC-1) module is actually located at `curricula/cp_accelerator/patterns/hash_table/problems/lc_1/`. The path drift makes this test report unreproducible from the documented command."
    },
    {
      "id": "module-gen-docs-problem-count-874-vs-959",
      "location": "docs/internal/module_generation/MODULE_GENERATION_COMPREHENSIVE_SUMMARY.md:9",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "MODULE_GENERATION_COMPREHENSIVE_SUMMARY.md line 9 states the system \"scales to all 874 problems in our curriculum\". MODULE_GENERATION_PROGRESS.md and MODULE_GENERATION_REFACTORING_PLAN.md repeat the same 874 figure. Counting problems via `sum(len(t['problems']) for t in canonical_curriculum.json['topics'])` yields 959, not 874 — 9.7% more than documented. The count discrepancy persists across multiple doc files indicating the curriculum was expanded after these documents were authored."
    },
    {
      "id": "e2e-test-status-8-commands-vs-14",
      "location": "tests/e2e/E2E_TEST_STATUS.md:9",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "E2E_TEST_STATUS.md line 9 states \"All 8 engine commands fully tested\". The MASTERY_COMMAND_REFERENCE.md (same Nov 2025 period) documents 14 commands (9 primary + 4 deprecated + 1 dev tool), which matches the 14 `@app.command()` decorators counted in engine/main.py. The 8-command count in E2E_TEST_STATUS.md is a stale figure from an earlier development phase; the gap means at least 6 commands were added with no corresponding E2E coverage claim update."
    },
    {
      "id": "two-sum-comparison-analysis-wrong-module-path",
      "location": "docs/internal/two_sum_qa/MODULE_COMPARISON_ANALYSIS.md:47",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "MODULE_COMPARISON_ANALYSIS.md (line 47 onward) shows the Two Sum module directory tree rooted at `two_sum/` inside `cp_accelerator`, implying the path `curricula/cp_accelerator/modules/two_sum/` or similar. The actual production path is `curricula/cp_accelerator/patterns/hash_table/problems/lc_1/`. The module is accessed by the engine under the `lc_1` problem ID, not `two_sum`. Both the path and identifier in this analysis document are inconsistent with the actual on-disk structure."
    },
    {
      "id": "readme-wrong-filename",
      "location": "tests/integration/README.md:7",
      "class": "doc_code_drift",
      "severity": "high",
      "evidence": "README line 7 states '**File**: `test_llm_integration.py`' but the actual integration test file is `tests/integration/test_llm_service.py`. No file named `test_llm_integration.py` exists in that directory. All example commands and test-output snippets in the README (e.g., lines 39, 41, 100, 105, 111) also reference this non-existent filename, making every copy-paste instruction in the README broken."
    },
    {
      "id": "readme-wrong-total-cost",
      "location": "tests/integration/README.md:9",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "README line 9 states '**Cost**: ~$0.006 per full test run'. The actual file `tests/integration/test_llm_service.py` line 7 states 'Cost: ~$0.009 per full test run (3 API calls x $0.003 each)'. The discrepancy is because the actual file has 3 live-API tests (`test_llm_accepts_correct_answer`, `test_llm_rejects_incomplete_answer`, `test_llm_rejects_conceptual_error`) versus the 2 the README cost table accounts for."
    },
    {
      "id": "readme-nonexistent-test-names",
      "location": "tests/integration/README.md:13",
      "class": "doc_code_drift",
      "severity": "high",
      "evidence": "The 'What These Tests Validate' section (lines 13-18) lists six test capabilities including 'Fast Filter Logic', 'Decision Boundary', and 'Fast filter vs. LLM routing'. None of these correspond to any test in `tests/integration/test_llm_service.py`. The actual tests are: `test_llm_service_initialization_with_api_key` (line 58), `test_llm_service_missing_api_key` (line 70), `test_llm_accepts_correct_answer` (line 86), `test_llm_rejects_incomplete_answer` (line 123), `test_llm_rejects_conceptual_error` (line 156), `test_llm_timeout_handling` (line 197), `test_response_format_validation` (line 222), `test_cost_analysis_documentation` (line 257). The README describes a different, older test file."
    },
    {
      "id": "readme-wrong-cost-table",
      "location": "tests/integration/README.md:59",
      "class": "doc_code_drift",
      "severity": "high",
      "evidence": "Cost table (lines 59-66) lists six test names that do not exist: `test_fast_filter_blocks_shallow_answer`, `test_llm_accepts_deep_correct_answer`, `test_llm_rejects_conceptual_error_with_socratic_feedback`, `test_error_handling_missing_api_key`, `test_fast_filter_vs_llm_decision_boundary`, `test_llm_api_timeout_handling`. Actual test names differ (e.g., `test_llm_accepts_correct_answer` not `test_llm_accepts_deep_correct_answer`; `test_llm_service_missing_api_key` not `test_error_handling_missing_api_key`). The table also shows 2 API calls and $0.006 total, while the actual test file header (line 7) declares 3 API calls and $0.009."
    },
    {
      "id": "readme-wrong-example-command",
      "location": "tests/integration/README.md:41",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "Line 41 shows the command: `uv run pytest tests/integration/test_llm_integration.py::test_llm_accepts_deep_correct_answer -v`. Both the filename (`test_llm_integration.py` instead of `test_llm_service.py`) and the test name (`test_llm_accepts_deep_correct_answer` instead of `test_llm_accepts_correct_answer`) are wrong. Running this command as written would fail with a 'file not found' error."
    },
    {
      "id": "readme-wrong-fixture-name-in-example",
      "location": "tests/integration/README.md:141",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "The 'Adding New Integration Tests' code example on line 141 shows `def test_new_llm_feature(check_api_key, softmax_questions):`. The fixture `softmax_questions` does not exist in `test_llm_service.py`; the actual reusable fixture is `sample_question` (defined at line 37 of `test_llm_service.py`). A developer following this example would get a pytest fixture-not-found error."
    },
    {
      "id": "bpe-reference-merges-loaded-but-unused",
      "location": "tests/fixtures/train-bpe-reference-merges.txt:1",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "The fixture contains 243 BPE merge pairs and is loaded by `tests/test_train_bpe.py:41-49` into `reference_merges`. However, the only assertion that used this variable—`assert merges == reference_merges` at line 54—was commented out with the note 'Too strict - commented out'. After that block, `reference_merges` is never referenced again; it is a dead variable. The test instead validates only count with hardcoded bounds (lines 57-58: `assert len(merges) >= 243` and `assert len(merges) <= 245`), which are independent of the file content. The fixture was designed as the ground-truth reference for exact BPE training validation, but after the exact assertion was abandoned, the file's loaded content serves no purpose in any currently-running assertion."
    },
    {
      "id": "ci-python-below-minimum",
      "location": ".github/workflows/tests.yml:20",
      "class": "bug",
      "severity": "high",
      "evidence": "tests.yml line 20: `python-version: '3.10'` (and again at line 70 in the lint job). pyproject.toml line 6 declares `requires-python = \">=3.11\"`. The test CI runs on Python 3.10, which is BELOW the package's minimum required version. `uv sync` on Python 3.10 will either fail (if uv enforces the requires-python constraint) or install the package in an unsupported environment, making CI results meaningless. validate_cp_manifest.yml correctly uses 3.11 at lines 26, 123, 212."
    },
    {
      "id": "actions-mutable-tag-pinning",
      "location": ".github/workflows/tests.yml:16",
      "class": "security",
      "severity": "high",
      "evidence": "All GitHub Actions in both workflows are pinned to mutable version tags, not immutable SHA digests. tests.yml: `actions/checkout@v4` (line 16, 66), `actions/setup-python@v5` (line 19, 69), `astral-sh/setup-uv@v3` (line 23), `actions/upload-artifact@v4` (line 52). validate_cp_manifest.yml: `actions/checkout@v3` (line 20), `actions/setup-python@v4` (lines 24, 122, 211). A tag can be silently updated to point to malicious code without any hash change detectable by the workflow consumer. This is the classic supply-chain vector for compromised GitHub Actions (e.g., tj-actions/changed-files incident). Fix: pin every action to its full SHA digest."
    },
    {
      "id": "action-version-skew-between-workflows",
      "location": ".github/workflows/validate_cp_manifest.yml:20",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "tests.yml uses `actions/checkout@v4` and `actions/setup-python@v5`, while validate_cp_manifest.yml uses the older `actions/checkout@v3` (line 20) and `actions/setup-python@v4` (line 24). Different action versions have different security patch levels and behaviours (e.g., checkout@v3 uses Node 16 which is EOL; checkout@v4 uses Node 20). Using inconsistent versions means the two workflows operate under different security baselines without explicit justification."
    },
    {
      "id": "pip-install-uv-unpinned",
      "location": ".github/workflows/validate_cp_manifest.yml:29",
      "class": "security",
      "severity": "high",
      "evidence": "validate_cp_manifest.yml line 29: `run: pip install uv`. No version specifier is provided. This means every CI run fetches the latest uv from PyPI, making the build non-reproducible and exposing the pipeline to a supply-chain compromise of the `uv` package on PyPI. tests.yml correctly uses the official `astral-sh/setup-uv@v3` action which itself should be pinned to SHA. The inconsistency also means the manifest-validation CI may use a different uv version than the test CI, producing divergent resolution behaviour even though both call `uv sync`."
    },
    {
      "id": "lint-gates-silenced",
      "location": ".github/workflows/tests.yml:85",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "tests.yml lines 85 and 89 both set `continue-on-error: true` for the ruff linter and ruff formatter checks respectively. Code that fails `ruff check engine/ tests/` or `ruff format engine/ tests/ --check` will still produce a green CI run. This removes the code-quality gate entirely, allowing malformed or style-violating code into the codebase without any blocking signal. A supply-chain audit expects quality gates to actually gate."
    },
    {
      "id": "alpha-package-ty-no-upper-bound",
      "location": "pyproject.toml:22",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "pyproject.toml line 22: `\"ty>=0.0.1a16\"`. The `ty` package is at pre-release alpha status (version 0.0.1a16). Alpha packages carry no semantic versioning stability guarantees; any subsequent alpha release can silently break the API. There is no upper-bound constraint, so `uv sync` with an updated lock will install future incompatible alphas. This is a production runtime dependency (not dev-only), amplifying the risk. The package should either be pinned to an exact version in the lock file and reviewed on update, or moved to dev dependencies with an upper bound."
    },
    {
      "id": "validate-push-trigger-missing-script-path",
      "location": ".github/workflows/validate_cp_manifest.yml:8",
      "class": "design_defect",
      "severity": "low",
      "evidence": "validate_cp_manifest.yml on.pull_request.paths (lines 5-8) includes both `curricula/cp_accelerator/**` and `scripts/generate_manifest.py`. But on.push.paths (lines 10-14) only includes `curricula/cp_accelerator/**` — `scripts/generate_manifest.py` is absent. A direct push to main that modifies only `scripts/generate_manifest.py` (bypassing PR) will NOT trigger the manifest integrity check, allowing a broken generator to land without validation."
    },
    {
      "id": "lc34-test-cases-missing-target-field",
      "location": "curricula/cp_accelerator/patterns/binary_search/problems/lc_34/test_cases.json:54",
      "class": "bug",
      "severity": "high",
      "evidence": "Tests 4-8 in this file are for 'Find First and Last Position of Element in Sorted Array' but lack the required `target` field and have expected values that look like sorted arrays rather than [first, last] index pairs. Example: test 4 `{\"input\": {\"nums\": [1]}, \"expected\": [1]}`, test 6 `{\"input\": {\"nums\": [3,2,1]}, \"expected\": [1,2,3]}`. Tests 1-3 correctly include `target` and return `[-1,-1]` or `[first, last]` pairs. Tests 4-8 appear to be sorting test cases copied from another problem. Any validator calling the binary search function with these inputs will receive wrong argument signatures and incorrect expected values."
    },
    {
      "id": "lc47-test1-expected-empty-string",
      "location": "curricula/cp_accelerator/patterns/combinatorics_and_number_theory/problems/lc_47/test_cases.json:15",
      "class": "bug",
      "severity": "high",
      "evidence": "Test case 1 for Permutations II (nums=[1,1,2]) has `\"expected\": \"\"` — an empty string. The correct expected value for this input is `[[1,1,2],[1,2,1],[2,1,1]]`. An empty string will never equal the actual list output, so this test case will always fail in any validator that does equality comparison, making the test suite for lc_47 effectively broken for its primary example."
    },
    {
      "id": "sort-list-tests-wrong-input-key",
      "location": "curricula/cp_accelerator/patterns/divide_and_conquer/problems/lc_148/test_cases.json:54",
      "class": "bug",
      "severity": "high",
      "evidence": "Tests 4-8 in both `divide_and_conquer/problems/lc_148/test_cases.json` and `sorting/problems/lc_148/test_cases.json` use input key `\"nums\"` instead of `\"head\"`. LeetCode 148 (Sort List) accepts a linked list via its `head` parameter. Tests 1-3 correctly use `\"head\": [...]`. Tests 4-8 use `\"nums\": [1]`, `\"nums\": []`, etc. — clearly copy-pasted from an array sorting problem. A validator calling `sortList(head=...)` with key `nums` would fail or silently pass wrong data. Identical contamination exists in sorting/problems/lc_148/test_cases.json tests 4-8."
    },
    {
      "id": "theory-files-duplicate-problem-qa",
      "location": "curricula/cp_accelerator/patterns/hash_table/theory/justify_questions.json:1",
      "class": "doc_code_drift",
      "severity": "medium",
      "evidence": "The file `hash_table/theory/justify_questions.json` is byte-for-byte identical to `hash_table/problems/lc_1/justify_questions.json`. Both contain the same three question IDs (`two_sum_hash_table_advantage`, `two_sum_complexity`, `two_sum_edge_cases`) with identical text. A theory file should contain general hash table theory, not problem-specific Two Sum Q&A. Similarly, `sorting/theory/justify_questions.json` is identical to `sorting/problems/lc_912/justify_questions.json` (same three IDs: `sorting_conceptual`, `sorting_complexity`, `sorting_stability`). A learner who encounters both the theory phase and the lc_1 problem phase will see the same questions twice, defeating the purpose of the theory layer."
    },
    {
      "id": "empty-test-suites-lc703-lc1804",
      "location": "curricula/cp_accelerator/patterns/heap_and_priority_queue/problems/lc_703/test_cases.json:5",
      "class": "bug",
      "severity": "medium",
      "evidence": "Two test_cases.json files contain `\"tests\": []` (empty array): `heap_and_priority_queue/problems/lc_703/test_cases.json` (Kth Largest Element in a Stream) and `trie/problems/lc_1804/test_cases.json` (Implement Trie II). The validators for these problems will run against zero test cases and trivially pass regardless of correctness, providing no learning signal. The trie/lc_208 test file has a similar weakness: only one test case with a string-encoded expected value `\"[null, null, true, false, true, null, true]\"` that may not be machine-comparable."
    },
    {
      "id": "inconsistent-boolean-encoding",
      "location": "curricula/cp_accelerator/patterns/hash_table/problems/lc_217/test_cases.json:17",
      "class": "bug",
      "severity": "low",
      "evidence": "lc_217/test_cases.json and stack_and_queue/lc_1003/test_cases.json encode boolean expected values as JSON strings: `\"expected\": \"true\"` and `\"expected\": \"false\"`. In contrast, stack_and_queue/lc_20/test_cases.json correctly uses JSON booleans: `\"expected\": true` and `\"expected\": false`. If the validator compares Python boolean `True` against string `\"true\"`, the comparison fails silently (`True != \"true\"`). This is a data schema inconsistency that would cause spurious test failures for lc_217 (Contains Duplicate) and lc_1003 (Check If Word Is Valid After Substitutions)."
    },
    {
      "id": "insert-before-check-name-intent-mismatch",
      "location": "curricula/cp_accelerator/patterns/hash_table/problems/lc_1/bugs/insert_before_check.json:1",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "The file is named `insert_before_check.json`, its id is `\"two-sum-insert-before-check\"`, and the symptom file is named `insert_before_check_symptom.txt`. This name describes the bug pattern where the current element is inserted into the hash table BEFORE checking for the complement (allowing reuse of the same element). However, the actual implemented injection (pass 1) replaces `complement in seen` with `num in seen`, which is a fundamentally different bug: it checks for the current number's presence instead of the complement's presence. The `note` field confirms the redesign: 'Original bug used unsupported move_after type. Redesigned...' The id/filename/symptom describe a different bug than what is injected, misleading both learners receiving the symptom hint and maintainers extending the spec."
    },
    {
      "id": "opaque-string-encoded-expected-values",
      "location": "curricula/cp_accelerator/patterns/design_patterns/problems/lc_146/test_cases.json:9",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "lc_146 (LRU Cache), lc_460 (LFU Cache), lc_307 (Range Sum Query Mutable), and lc_208 (Implement Trie) all have `\"input\": {}` and `\"expected\"` as a JSON string encoding a sequence of results: e.g. `\"[null, null, null, 1, null, -1, null, -1, 3, 4]\"`. These cannot be compared structurally by a validator — the validator must either parse the string or use string equality. There is no machine-readable mapping between operations and expected outputs, no constructor arguments, and no operation sequence. This design means the validator cannot meaningfully test stateful data-structure problems, which are exactly the hardest problems to get right. It is inconsistent with the structured input/expected schema used by all other test files."
    },
    {
      "id": "ci-python-version-mismatch",
      "location": ".github/workflows/tests.yml:20",
      "class": "bug",
      "severity": "high",
      "evidence": "Line 20 (and line 70): `python-version: '3.10'`. pyproject.toml:7 declares `requires-python = \">=3.11\"`. CI runs tests on Python 3.10 which is explicitly excluded by the package's own minimum version constraint. Tests may silently pass on 3.10 while failing on any supported Python version (3.11+), masking real compatibility bugs."
    },
    {
      "id": "alpha-prerelease-dep-ty",
      "location": "pyproject.toml:22",
      "class": "security",
      "severity": "medium",
      "evidence": "`ty>=0.0.1a16` — `ty` is pinned to an alpha pre-release (`0.0.1a16`) as a runtime dependency. Alpha releases carry no stability guarantees; any future `>=0.0.1a17` alpha release could introduce breaking changes or security issues and will be picked up automatically. Pre-release packages in production `[project.dependencies]` are a supply-chain risk."
    },
    {
      "id": "numpy-completely-unpinned",
      "location": "pyproject.toml:11",
      "class": "security",
      "severity": "low",
      "evidence": "`numpy` is listed with zero version constraint — no lower bound, no upper bound. Any numpy major version (including future 3.x with breaking API changes) will be accepted. Combined with `>=` lower-bound-only constraints on most other deps, the effective lockfile is entirely resolver-determined. Without a pinned lockfile committed to the repo, reproducible builds are impossible."
    },
    {
      "id": "github-actions-not-sha-pinned",
      "location": ".github/workflows/tests.yml:14",
      "class": "security",
      "severity": "medium",
      "evidence": "All four action references use semver tags, not commit SHAs: `actions/checkout@v4` (line 14), `actions/setup-python@v5` (line 18), `astral-sh/setup-uv@v3` (line 22), `actions/upload-artifact@v4` (line 52). A tag like `@v4` is mutable — a compromised maintainer or tag-force-push can inject malicious code into every CI run. SLSA supply-chain hardening requires pinning to an immutable commit SHA."
    },
    {
      "id": "pip-install-uv-unpinned",
      "location": ".github/workflows/validate_cp_manifest.yml:29",
      "class": "security",
      "severity": "medium",
      "evidence": "`run: pip install uv` with no version pin. This fetches the latest uv release at workflow run time. Any breaking change or malicious upload to the `uv` PyPI package will immediately affect the CI pipeline without any diff in the repository. The tests.yml workflow uses `astral-sh/setup-uv@v3` (tagged, not pinned, but at least version-constrained); validate_cp_manifest.yml uses no constraint at all."
    },
    {
      "id": "stale-action-versions-validate-workflow",
      "location": ".github/workflows/validate_cp_manifest.yml:21",
      "class": "design_defect",
      "severity": "low",
      "evidence": "validate_cp_manifest.yml uses `actions/checkout@v3` (line 21) and `actions/setup-python@v4` (line 25), while tests.yml uses `checkout@v4` and `setup-python@v5`. Workflows in the same repo are running on different action versions, creating inconsistent build environments and indicating stale maintenance. Older action versions may also lack security patches."
    },
    {
      "id": "lint-continue-on-error-silences-failures",
      "location": ".github/workflows/tests.yml:85",
      "class": "design_defect",
      "severity": "low",
      "evidence": "Lines 85 and 89 both set `continue-on-error: true` for the ruff lint check and ruff format check steps respectively. This means lint failures never block a merge. The CI workflow will report green even if code has linting violations, defeating the purpose of having CI lint enforcement."
    },
    {
      "id": "manifest-module-count-mismatch",
      "location": "curricula/cs336_a1/manifest.json:6",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "Line 6: `\"description\": \"CS336 Assignment 1 — 22 modules in dependency order\"` — wait, the description reads `\"21 modules\"` but the `modules` array contains 22 entries: softmax, cross_entropy, gradient_clipping, linear, embedding, silu, rmsnorm, swiglu, attention, rope, multihead_attention, transformer_block, transformer_lm, adamw, cosine_schedule, data_loader, checkpointing, training_loop, unicode, bpe_tokenizer, tokenizer_class, text_generation. Confirmed by `python3 -c \"import json; m=json.load(open('curricula/cs336_a1/manifest.json')); print(len(m['modules']))\"` → 22."
    },
    {
      "id": "lc1099-empty-test-cases",
      "location": "curricula/cp_accelerator/patterns/two_pointers/problems/lc_1099/test_cases.json:5",
      "class": "bug",
      "severity": "medium",
      "evidence": "`\"tests\": []` — the test_cases.json for LeetCode 1099 (Two Sum Less Than K) contains an empty tests array. The validator shell script iterates this array; with zero entries, validation always vacuously passes. Any learner implementation—including a completely wrong one—passes the harden phase for this problem."
    },
    {
      "id": "temperature-bug-inverted-direction",
      "location": "curricula/cs336_a1/modules/text_generation/bugs/temperature_after_softmax.json:22",
      "class": "intent_mismatch",
      "severity": "high",
      "evidence": "Bug spec ID is `text-generation-temperature-after-softmax`, description: \"Temperature applied after softmax\". The replacement source at line 22 is `\"F.softmax(next_logits / temperature, dim=-1)\"` — this applies temperature BEFORE softmax (divides logits first), which is the mathematically CORRECT implementation. To inject the described bug (temperature applied after softmax), the replacement should be something like `F.softmax(next_logits, dim=-1) / temperature`. The spec injects correct behavior instead of the intended bug, so the harden phase cannot detect anything."
    },
    {
      "id": "temperature-draft-inverted-direction",
      "location": "curricula/cs336_a1/modules/text_generation/bugs/temperature_after_softmax_draft.json:22",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "Draft version of the temperature spec has the same inversion: description says \"Applies temperature after softmax instead of before\" but replacement source is `\"F.softmax(next_logits / temperature, dim=-1)\"` — temperature applied before softmax (correct). The draft is also directionally inverted. If the draft were promoted to production it would still not inject the described bug."
    },
    {
      "id": "bpe-draft-noop-replacement",
      "location": "curricula/cs336_a1/modules/bpe_tokenizer/bugs/wrong_merge_order_draft.json:43",
      "class": "bug",
      "severity": "medium",
      "evidence": "`\"source\": \"node\"` at line 43 (replacement block). The injection engine treats `source` as a literal Python expression string to substitute. Replacing with `\"node\"` means the replacement is the Python identifier `node` — a reference to the AST node object, not the code's original expression. This is effectively a no-op or produces a NameError at runtime. Additionally, the spec uses `\"pass_\": 1` (with underscore, line 9) instead of `\"pass\": 1`, and `target_function: \"bpe_tokenizer\"` rather than the correct `\"train\"`, meaning the engine will fail to locate the injection target."
    },
    {
      "id": "cosine-draft-inverted-direction",
      "location": "curricula/cs336_a1/modules/cosine_schedule/bugs/wrong_cosine_range_draft.json:47",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "Draft description: \"Replace cosine decay calculation to include transformation (1 + cos(πt)) / 2\". The replacement source is `\"0.5 * (1.0 + math.cos(math.pi * progress))\"` — this IS the correct cosine decay formula. The production version correctly injects the bug by substituting `math.cos(math.pi * progress)` (raw cosine without normalization). The draft inverts the intended direction: it replaces the buggy code with the correct formula rather than injecting the bug."
    },
    {
      "id": "linear-draft-inverted-direction",
      "location": "curricula/cs336_a1/modules/linear/bugs/missing_transpose_draft.json",
      "class": "intent_mismatch",
      "severity": "medium",
      "evidence": "Bug spec description: \"Find in_features.matmul(self.weight) and replace with in_features.matmul(self.weight.t())\". The replacement source is `\"y.matmul(self.weight.t())\"` — this ADDS the transpose `.t()`. The intended bug is `missing_transpose` (removing `.t()` from correct code). The draft describes and implements the inverse: it would fix the bug rather than inject it. The production spec correctly removes `.t()`."
    },
    {
      "id": "data-loader-draft-ast-expression-as-source",
      "location": "curricula/cs336_a1/modules/data_loader/bugs/wrong_sampling_range_draft.json:56",
      "class": "bug",
      "severity": "medium",
      "evidence": "Replacement source at lines 56-59: `\"node.value.keywords[0].value + 1\"` — this is a Python AST traversal expression (using `.value.keywords[0].value`), not a literal Python source string to inject. The injection engine would emit the string `node.value.keywords[0].value + 1` as the replacement code, which is a NameError at runtime. Also uses `\"pass_\": 1` (underscore) and `target_function: \"data_loader\"` instead of the production `\"get_batch\"`."
    },
    {
      "id": "data-loader-justify-questions-invalid-json",
      "location": "curricula/cs336_a1/modules/data_loader/justify_questions.json:34",
      "class": "bug",
      "severity": "high",
      "evidence": "File is confirmed invalid JSON. `python3 -c \"import json; json.load(open('curricula/cs336_a1/modules/data_loader/justify_questions.json'))\"` raises `JSONDecodeError: Expecting ',' delimiter: line 34 column 21 (char 8981)`. Visible corruption: truncated strings like `\"Broadcas` appear followed immediately by embedded `\"required_concepts\": [` fragments, indicating the file was generated with mid-string truncation and repeated content insertion. The justify phase cannot load questions for this module."
    },
    {
      "id": "tokenizer-class-justify-questions-invalid-json",
      "location": "curricula/cs336_a1/modules/tokenizer_class/justify_questions.json:9",
      "class": "bug",
      "severity": "high",
      "evidence": "File is confirmed invalid JSON. `python3 -c \"import json; json.load(open('curricula/cs336_a1/modules/tokenizer_class/justify_questions.json'))\"` raises `JSONDecodeError: Expecting ',' delimiter: line 9 column 50 (char 2112)`. Visible corruption at line 9: `\"Different order produces different en` — string truncated mid-sentence — followed immediately by `\"required_concepts\": [` embedded as a key. Same corruption pattern as data_loader/justify_questions.json. The justify phase cannot load questions for this module."
    },
    {
      "id": "rope-draft-ast-expression-as-source",
      "location": "curricula/cs336_a1/modules/rope/bugs/wrong_rotation_draft.json",
      "class": "bug",
      "severity": "medium",
      "evidence": "Replacement source is a Python string concatenation AST traversal expression: `\"node.value.left.left.id + ' * ' + node.value.left.right.id + ' + ' + ...\"` — this is engine-internal AST navigation code, not a literal Python source string to inject. The engine would emit this expression as the replacement code, producing a NameError at runtime. Additionally, `target_function: \"apply_2d_rotation\"` differs from the production spec's `\"apply_rotary_position_embeddings\"`, pointing to a different (or non-existent) function."
    },
    {
      "id": "embedding-draft-ast-expression-as-source",
      "location": "curricula/cs336_a1/modules/embedding/bugs/wrong_dimension_order_draft.json",
      "class": "bug",
      "severity": "medium",
      "evidence": "Both passes (using `\"pass_\": 1` and `\"pass_\": 2` with underscore) have replacement sources that are Python AST traversal expressions: pass 1 `\"node.value.keywords[1].value\"` and pass 2 `\"node.value.keywords[0].value\"`. These are not literal Python source code strings — the engine would inject them verbatim, producing NameErrors at runtime. The underscore-suffixed `pass_` key also deviates from the v2.1 spec schema which uses `\"pass\"`."
    },
    {
      "id": "gradient-clipping-draft-ast-expression-as-source",
      "location": "curricula/cs336_a1/modules/gradient_clipping/bugs/per_parameter_clipping_draft.json",
      "class": "bug",
      "severity": "medium",
      "evidence": "Pass 1 replacement source: `\"node.value.func.value.args[0]\"` and pass 2 replacement source: `\"node.body[0]\"` — both are Python AST traversal expressions, not literal source strings. The engine would inject these strings as code, producing NameErrors. Same `pass_` (underscore) schema deviation seen across other draft specs."
    },
    {
      "id": "cross-entropy-variable-name-mismatch",
      "location": "curricula/cs336_a1/modules/cross_entropy/bugs/no_logsumexp.json",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "Production `no_logsumexp.json` targets a variable named `log_sum_exp` (pattern `id: \"log_sum_exp\"`). Draft `no_logsumexp_draft.json` targets a variable named `lse`. These two specs cannot both be correct for the same implementation: whichever name the student's code uses, one spec will silently fail to match and not inject the bug. This mismatch indicates either the production or draft spec targets the wrong variable name."
    },
    {
      "id": "transformer-lm-justify-json-invalid",
      "location": "curricula/cs336_a1/modules/transformer_lm/justify_questions.json:33",
      "class": "bug",
      "severity": "critical",
      "evidence": "File fails JSON parsing: JSONDecodeError at line 33, col 51, char 8736. The model_answer string for question transformer_lm_q3 is truncated mid-sentence and then the literal text '\"required_concepts\": [' (including embedded JSON-like fragments) appears inside the string value, not as a proper closing quote + array. Lines 33-42 each read: '\"Tying saves 50% of embedding parameter    \"required_concepts\": [' — the original long model_answer string was accidentally spliced with the required_concepts key, making the entire file unparseable. Any engine code path that calls json.load() on this file will throw an unhandled JSONDecodeError, breaking the transformer_lm curriculum module entirely."
    },
    {
      "id": "missing-final-norm-draft-wrong-spec",
      "location": "curricula/cs336_a1/modules/transformer_lm/bugs/missing_final_norm_draft.json:2",
      "class": "doc_code_drift",
      "severity": "high",
      "evidence": "The file at curricula/cs336_a1/modules/transformer_lm/bugs/missing_final_norm_draft.json contains the SiLU bug spec, not a missing_final_norm draft. Actual content: '\"id\": \"silu-missing-multiply\"', '\"target_function\": \"silu\"', '\"description\": \"Removes the multiplication by input, returning only sigmoid(x) instead of x * sigmoid(x).\"'. This is a verbatim copy of the silu/bugs/missing_multiply spec placed in the wrong directory. A reviewer or engine loading this file expecting a transformer_lm missing_final_norm draft would inject the wrong bug into the wrong module."
    },
    {
      "id": "ci-actions-mutable-version-tags",
      "location": ".github/workflows/tests.yml:15",
      "class": "security",
      "severity": "high",
      "evidence": "All GitHub Actions in both workflows use mutable semantic-version tags instead of pinned commit SHAs. tests.yml uses: actions/checkout@v4 (line 15, 64), actions/setup-python@v5 (line 17, 69), astral-sh/setup-uv@v3 (line 23, 74), actions/upload-artifact@v4 (line 53). validate_cp_manifest.yml uses: actions/checkout@v3 (lines 22, 116, 205), actions/setup-python@v4 (lines 24, 118, 207). If a maintainer of any of these actions moves the mutable tag to a different commit (intentionally or via supply-chain compromise), CI will execute attacker-controlled code in a context that has GITHUB_TOKEN write permissions to the repository."
    },
    {
      "id": "ci-python-version-mismatch",
      "location": ".github/workflows/tests.yml:20",
      "class": "bug",
      "severity": "high",
      "evidence": "pyproject.toml:6 declares 'requires-python = \">=3.11\"' but tests.yml:20 and tests.yml:70 both set 'python-version: \"3.10\"'. CI is running the test suite on Python 3.10, which is below the declared minimum requirement. This means CI does not validate the package on its required Python version. Code using Python 3.11-only features (e.g., tomllib stdlib, TypeVarTuple, improved typing constructs) would pass CI on 3.10 yet fail in production. The uv.lock also records 'requires-python = \">=3.11\"', deepening the mismatch."
    },
    {
      "id": "ci-pip-install-uv-unpinned",
      "location": ".github/workflows/validate_cp_manifest.yml:29",
      "class": "security",
      "severity": "medium",
      "evidence": "Line 29: 'run: pip install uv' with no version constraint. This installs the latest available uv from PyPI at each CI run. A malicious or compromised PyPI release of the 'uv' package would execute arbitrary code on the CI runner before any project dependencies are installed or any trust checks are applied. The tests.yml workflow avoids this by using the official astral-sh/setup-uv action, but validate_cp_manifest.yml falls back to raw pip install."
    },
    {
      "id": "pyproject-numpy-no-version-bound",
      "location": "pyproject.toml:9",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "The dependency declaration is bare '\"numpy\"' with no version specifier (no >=, ~=, or <). While uv.lock currently pins numpy to 2.3.2, any installation performed without the lockfile (e.g., pip install -e ., fresh contributor setup, third-party consumption) will resolve to the latest numpy regardless of compatibility. NumPy 2.x introduced breaking API changes vs 1.x (removal of np.bool/np.int/np.float aliases, C-API changes). All other numerical dependencies in the file carry explicit version constraints."
    },
    {
      "id": "conftest-torch-load-no-weights-only",
      "location": "tests/conftest.py:199",
      "class": "security",
      "severity": "medium",
      "evidence": "Line 199: 'state_dict = torch.load(FIXTURES_PATH / \"ts_tests\" / \"model.pt\", map_location=\"cpu\")' uses the unsafe pickle-based torch.load without 'weights_only=True'. PyTorch 2.x emits a FutureWarning and will change the default in a future version. Loading a .pt file without weights_only=True executes arbitrary Python via pickle. If the fixture file tests/fixtures/ts_tests/model.pt is ever replaced by a compromised version (e.g., via supply-chain attack on test data), it could execute arbitrary code during the test suite."
    },
    {
      "id": "pyproject-ty-alpha-dependency",
      "location": "pyproject.toml:22",
      "class": "design_defect",
      "severity": "low",
      "evidence": "Dependency '\"ty>=0.0.1a16\"' declares a pre-release alpha package with only a lower-bound constraint and no upper bound. The 'ty' type checker is in active alpha development (uv.lock records 0.0.1a19). Alpha packages by semantic versioning convention carry no stability guarantee; CLI flags, APIs, and output format can change in any release. Using an alpha package as a production dependency without an upper bound or exact pin may cause sudden breakage on any lock file update."
    },
    {
      "id": "ci-lint-continue-on-error",
      "location": ".github/workflows/tests.yml:85",
      "class": "design_defect",
      "severity": "low",
      "evidence": "Lines 85 and 88 both use 'continue-on-error: true' for the ruff linter and ruff formatter checks respectively: 'uv run ruff check engine/ tests/ --output-format=github' and 'uv run ruff format engine/ tests/ --check'. This means lint violations and formatting failures are logged but do not cause the CI job to fail. Code with linting errors or inconsistent formatting will merge without any CI gate."
    },
    {
      "id": "draft-bug-spec-pass-field-inconsistency",
      "location": "curricula/cs336_a1/modules/training_loop/bugs/missing_zero_grad_draft.json:9",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "missing_zero_grad_draft.json:9 uses field name '\"pass_\": 1' and missing_residual_draft.json:10 also uses '\"pass_\": 1', while the corresponding v2 drafts and production specs use '\"pass\": 1' (missing_zero_grad_draft_v2.json:9, missing_residual_draft_v2.json:9, missing_residual.json:9). The engine does not currently read draft files directly, but the divergent field name ('pass_' vs 'pass') documents a schema evolution that was applied inconsistently across the draft corpus. Any tooling that processes draft files alongside production specs would encounter this mismatch."
    },
    {
      "id": "ci-checkout-version-inconsistency",
      "location": ".github/workflows/validate_cp_manifest.yml:22",
      "class": "design_defect",
      "severity": "low",
      "evidence": "validate_cp_manifest.yml:22 uses 'actions/checkout@v3' while tests.yml:15 uses 'actions/checkout@v4'. The two workflows use different major versions of the same action. checkout@v3 and @v4 differ in Node.js runtime (Node 16 vs Node 20) and underlying behavior, meaning the two CI workflows operate in subtly different environments. This inconsistency makes it harder to reason about CI environment parity."
    },
    {
      "id": "harden-select-bug-picks-drafts",
      "location": "engine/stages/harden.py:195-197",
      "class": "bug",
      "severity": "high",
      "evidence": "_select_bug() globs ALL .json files: `json_files = list(bugs_dir.glob('*.json'))` and concatenates them with patch files. This includes *_draft.json files alongside production specs. For example, curricula/cs336_a1/modules/multihead_attention/bugs/ contains both missing_transpose_back.json (production, target_function='forward') and missing_transpose_back_draft.json (draft, target_function='multihead_attention'). If the draft is randomly selected, GenericBugInjector._has_function() at generic_injector.py:99-102 returns False for target_function='multihead_attention' (not a function name in any student code), and the injection aborts with (source_code, False). Then harden.py:114-121 raises HardenChallengeError. Similarly, transformer_lm/bugs/ contains missing_final_norm_draft.json whose id='silu-missing-multiply' — wrong module content entirely."
    },
    {
      "id": "missing-final-norm-draft-wrong-module",
      "location": "curricula/cs336_a1/modules/transformer_lm/bugs/missing_final_norm_draft.json:3",
      "class": "bug",
      "severity": "high",
      "evidence": "File id is 'silu-missing-multiply' (confirmed by understanding.json: 'Misidentified draft AST spec (id: silu-missing-multiply) that removes x*sigmoid(x) multiplication; likely a wrong-module draft'). The file lives in transformer_lm/bugs/ but describes a silu bug. If _select_bug() randomly selects this file for a transformer_lm harden session, the injector targets the silu pattern (BinOp Mult on sigmoid) rather than removing the final RMSNorm. On source code without that exact pattern, injection fails; on source code that happens to match, the wrong transformation is applied. Either outcome is incorrect for the transformer_lm module."
    },
    {
      "id": "incomplete-merge-patch-removes-solve-alias",
      "location": "curricula/cp_accelerator/patterns/sorting/problems/lc_912/bugs/incomplete_merge.patch",
      "class": "bug",
      "severity": "high",
      "evidence": "The .patch file diff includes removal of the solve alias: `-solve = sortArray` and `-# Alias for compatibility with test runner`. The companion JSON spec (incomplete_merge.json) only removes `result.extend(right[j:])` and does not touch the alias. In legacy patch-based harden (harden.py:127-156), if the .patch file is selected, shutil.copy2 + workspace_mgr.apply_patch() produces a buggy file lacking `solve = sortArray`. The CP Accelerator validator imports `solve` from the solution file; its absence causes NameError in the validator rather than the intended wrong-answer test failure. The two artifacts (patch vs JSON) thus describe divergent mutations and only the .patch path is broken."
    },
    {
      "id": "multihead-attention-draft-wrong-target-function",
      "location": "curricula/cs336_a1/modules/multihead_attention/bugs/missing_transpose_back_draft.json:5",
      "class": "bug",
      "severity": "medium",
      "evidence": "Draft spec has `\"target_function\": \"multihead_attention\"` but the actual PyTorch class method is named `forward`. Production spec missing_transpose_back.json:5 correctly uses `\"target_function\": \"forward\"`. GenericBugInjector._has_function() at generic_injector.py:200-205 walks all ast.FunctionDef nodes; no node is named 'multihead_attention' in student code, so the check at line 99 (`not self._has_function(original_ast, target_function)`) returns True and injection is aborted with `return source_code, False`. Combined with finding harden-select-bug-picks-drafts, this creates a ~50% runtime failure rate for multihead_attention harden sessions when both files are present."
    },
    {
      "id": "mark-stage-complete-synthetic-module-id",
      "location": "engine/schemas.py:168",
      "class": "design_defect",
      "severity": "high",
      "evidence": "In UserProgress.mark_stage_complete(), the harden branch generates a synthetic placeholder: `module_id = f'module_{self.current_module_index}'` (e.g., 'module_0', 'module_1') and appends it to completed_modules. The inline comment `# Will be replaced with actual ID` confirms this is an incomplete implementation. Actual module IDs are strings like 'rmsnorm', 'attention', 'adamw'. Any code that checks `if real_module_id in progress.completed_modules` — including curriculum completion checks and dependency validation — will always evaluate False because the list only contains synthetic placeholders. This silently corrupts the learning-progress tracking invariant."
    },
    {
      "id": "bug-definition-schema-production-specs-violate-contract",
      "location": "engine/schemas.py:318-349",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "PassDefinition (line 318) requires `description: str` (non-Optional). BugDefinition (line 343) requires `metadata: BugMetadata` (non-Optional). Two confirmed production specs violate both requirements: (1) curricula/cs336_a1/modules/swiglu/bugs/missing_gate.json — pass entries have no 'description' field, and the top-level object has no 'metadata' field; (2) curricula/cs336_a1/modules/multihead_attention/bugs/missing_transpose_back.json — same omissions. `BugDefinition.model_validate(data)` on either file raises pydantic.ValidationError. GenericBugInjector.validate_definition() at generic_injector.py:42-50 bypasses the Pydantic schema with manual checks, so no current runtime failure — but any code that uses the published schema contract (e.g., dev_tools, future validators) will fail."
    },
    {
      "id": "temperature-after-softmax-comment-inaccurate",
      "location": "curricula/cs336_a1/modules/text_generation/bugs/temperature_after_softmax.patch:15",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "Injected comment reads `# Wrong! Temperature after softmax has no effect!` but `probs / temperature` does change probability magnitudes — it just fails to produce a valid sharpened/flattened distribution because temperature scaling must be applied to logits before softmax, not to probabilities after. The claim 'no effect' is factually wrong; the real problem is that the resulting values no longer sum to 1 and the effect on sampling is distorted rather than absent. The comment misleads students about the actual failure mode they are expected to diagnose."
    },
    {
      "id": "wrong-pointer-move-json-future-created-date",
      "location": "curricula/cp_accelerator/patterns/two_pointers/problems/lc_167/bugs/wrong_pointer_move.json",
      "class": "doc_code_drift",
      "severity": "info",
      "evidence": "The metadata.created field value is '2026-03-04', which is in the future relative to all other files in the repository (dated 2025). All other bug spec metadata fields examined use 2025 dates. This is a data-entry inconsistency in the metadata, likely a copy-paste or manual entry error. Does not affect runtime behavior."
    },
    {
      "id": "draft-specs-use-pass-underscore-key",
      "location": "curricula/cs336_a1/modules/checkpointing/bugs/missing_optimizer_state_draft.json",
      "class": "design_defect",
      "severity": "low",
      "evidence": "Three draft specs use `\"pass_\": N` instead of `\"pass\": N` in logic entries: missing_optimizer_state_draft.json, multihead_attention/missing_transpose_back_draft.json, and swiglu/missing_gate_draft.json. Production specs use `\"pass\": N`. PassDefinition in schemas.py uses `Field(..., alias='pass')` with `populate_by_name = True` (line 318-327), which accepts both forms via Pydantic. However, GenericBugInjector at generic_injector.py:124 accesses pass_def as a raw dict and reads `pass_def['type']` — the pass number key is never read by the injector, so this inconsistency has no runtime effect currently. It creates authoring confusion about the canonical key name."
    },
    {
      "id": "genericinjector-softmax-fallback-hardcoded",
      "location": "engine/ast_harden/generic_injector.py:109",
      "class": "design_defect",
      "severity": "low",
      "evidence": "`Canonicalizer(target_function=target_function if target_function else 'softmax')` — when target_function is None or empty string, the canonicalizer defaults to 'softmax' as a hardcoded fallback. This is a development artifact; for bug specs without a target_function, the AST would be canonicalized around the 'softmax' function scope rather than any relevant function. This could cause the canonicalization to operate on the wrong AST subtree, potentially failing to find patterns in non-softmax functions. The correct fallback would be None or raise an error."
    },
    {
      "id": "scripts-mode-eval-shell-injection",
      "location": "scripts/mode:136",
      "class": "security",
      "severity": "low",
      "evidence": "`eval \"$test_cmd\"` in test_in_mode() where test_cmd comes from CLI arguments `\"${@:3}\"` (line 157). Passing shell metacharacters or command substitution in the test argument (e.g., `./scripts/mode test student '$(rm -rf /tmp/x)'`) would execute arbitrary shell commands. This is a developer-only script, reducing exploitability, but the pattern is a textbook shell injection risk if the script were ever called from a less-trusted context or CI pipeline with user-controlled inputs."
    },
    {
      "id": "production-specs-engine-version-2-0",
      "location": "curricula/cs336_a1/modules/rmsnorm/bugs/missing_keepdim.json:4",
      "class": "design_defect",
      "severity": "info",
      "evidence": "Two files in production use (`missing_keepdim.json` and `silu/missing_multiply.json`) declare `\"engine_version\": \"2.0\"` while the majority of production specs use `\"2.1\"`. GenericBugInjector does not check or branch on engine_version (validate_definition() at generic_injector.py:42-50 ignores it), so there is no current runtime difference. The inconsistency creates ambiguity about whether these specs were intended to be promoted to v2.1 and what behavioral differences between versions exist."
    },
    {
      "id": "harden-missing-symptom-files-cs336",
      "location": "engine/stages/harden.py:210",
      "class": "bug",
      "severity": "critical",
      "evidence": "Line 210: `symptom_name = selected_bug.stem + \"_symptom.txt\"`. Lines 213-218 raise HardenChallengeError if that file is absent. Filesystem verification confirmed that 18 of 20 cs336_a1 modules have at least one production .json bug file with NO corresponding _symptom.txt: adamw, attention, bpe_tokenizer, checkpointing, cosine_schedule, data_loader, embedding, linear, multihead_attention, rmsnorm, rope, silu, swiglu, text_generation, tokenizer_class, training_loop, transformer_block, transformer_lm. Only 4 symptom files exist across all of cs336_a1 (cross_entropy/bugs/no_logsumexp_symptom.txt, gradient_clipping/bugs/per_parameter_clipping_symptom.txt, softmax/bugs/no_subtract_max_symptom.txt, softmax/bugs/no_subtract_max_v2_symptom.txt). Consequence: _select_bug() will always raise HardenChallengeError for any of those 18 modules regardless of which bug is randomly chosen, making the Harden stage entirely non-functional for 90% of cs336_a1."
    },
    {
      "id": "harden-draft-json-in-selection-pool",
      "location": "engine/stages/harden.py:196",
      "class": "bug",
      "severity": "high",
      "evidence": "Line 195-197: `patch_files = list(bugs_dir.glob(\"*.patch\"))` / `json_files = list(bugs_dir.glob(\"*.json\"))` / `bug_files = patch_files + json_files`. The glob `*.json` matches ALL json files, including draft variants such as `adamw_wrong_beta_update_draft.json`, `adamw_wrong_beta_update_draft_v2.json`, `bpe_wrong_pair_count_draft.json`, and 13+ analogous draft files confirmed present across cs336_a1 bug directories. Draft files are development artifacts, not finished challenges; they uniformly lack `_symptom.txt` counterparts. When random.choice() (line 206) selects a draft, _select_bug() inevitably raises HardenChallengeError at line 214-218. A simple name-based guard (e.g., excluding stems that contain 'draft') is absent."
    },
    {
      "id": "justify-fast-filter-false-positive",
      "location": "engine/stages/justify.py:111",
      "class": "design_defect",
      "severity": "medium",
      "evidence": "Lines 109-115: `for failure_mode in question.failure_modes: for keyword in failure_mode.keywords: if keyword.lower() in user_answer_lower: return True, failure_mode.feedback`. The filter triggers on ANY occurrence of a failure-mode keyword anywhere in the student's answer, including in a fully correct, technically precise answer. Example: the softmax/justify_questions.json 'Hand-Waver' failure mode includes keywords ['stability', 'numerical', 'better', 'safer']. A correct answer that mentions 'subtracting the max value prevents numerical overflow, improving numerical stability' contains 'stability' and 'numerical' and would be falsely flagged. The logic performs positive keyword presence detection rather than identifying genuinely vague or incomplete answers. This contradicts the stated purpose ('catch shallow/vague answers') and will incorrectly penalise high-quality responses."
    },
    {
      "id": "bpe-tokenizer-q2-model-answer-missing-counterexample",
      "location": "curricula/cs336_a1/modules/bpe_tokenizer/justify_questions.json:18",
      "class": "bug",
      "severity": "medium",
      "evidence": "Question bpe_tokenizer_q2 requires students to 'prove by example that greedy BPE is NOT globally optimal — construct a small corpus where the greedy strategy leads to MORE total tokens than an alternative.' The model_answer (line 18) walks through two attempts. The first attempt explicitly concludes 'Wait, both approaches give same result!' (a tie, not a refutation). The second attempt explicitly concludes 'Greedy gives 7 tokens, alternative gives 9. Actually greedy wins here!' — an example where greedy is strictly BETTER than the alternative. The answer never constructs a corpus where greedy produces more tokens than an alternative strategy. The model_answer ends with the unsupported claim 'Greedy BPE is locally optimal but not globally optimal' without providing the required counterexample. Students who study this answer will believe an invalid proof satisfies the question."
    },
    {
      "id": "linear-q3-rmsnorm-mean-subtraction-error",
      "location": "curricula/cs336_a1/modules/linear/justify_questions.json:31",
      "class": "bug",
      "severity": "medium",
      "evidence": "Question linear_q3 model_answer (line 31) opens: 'Why normalization makes bias redundant: LayerNorm/RMSNorm compute: normalized = (x - mean) / std. The subtraction of mean ELIMINATES any constant bias!' This is factually incorrect for RMSNorm. RMSNorm computes normalized = x / sqrt(mean(x^2) + eps) — there is no mean subtraction, so a constant bias is NOT cancelled. The answer internally contradicts itself later: 'RMSNorm is similar but doesn't subtract mean, yet still normalizes magnitude.' The opening sentence is never corrected, leaving 'LayerNorm/RMSNorm compute: (x - mean) / std' as a direct false claim students will memorize. The required_concepts list (lines 33-39) is correctly limited to LayerNorm and does not claim RMSNorm subtracts mean, confirming the model_answer text is erroneous."
    },
    {
      "id": "coverage-reports-redundant-partial-scope",
      "location": "docs/internal/coverage/reports/coverage_with_new_cli_tests.txt:1",
      "class": "doc_code_drift",
      "severity": "low",
      "evidence": "coverage_with_new_cli_tests.txt records only engine/main.py at 834 stmts, 432 miss, 48%. The engine/main.py row in coverage_final_phase2.txt is identical (834 stmts, 432 miss, 48%). The standalone file is therefore a redundant partial-scope snapshot duplicating one row from the full-project report. Additionally, coverage_report_main_final.txt and coverage_report_main_partial.txt both cover only engine/main.py (690 stmts) with no project-wide data, while coverage_final_phase2.txt is the only file containing project-wide metrics (59% overall across engine/, tests/). The naming convention implies a temporal progression toward 'final' state that is not consistent with the actual scope of each file — coverage_report_main_final.txt reflects 28% for an older 690-statement main.py build, while coverage_final_phase2.txt shows 48% for a 834-statement main.py, giving the false impression of regression if read without careful scope comparison."
    }
  ],
  "coverage": {
    "requiredFiles": 609,
    "visited": 623
  },
  "fixpointRounds": 1
}
```
