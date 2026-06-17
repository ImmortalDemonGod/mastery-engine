# 01 — Comprehensive Understanding

> Stage 1 of the forensic audit pipeline. Coverage denominator: **609** files (post-ignore). This document is the denominator for every later stage.

## Architecture
The Mastery Engine is a Python 3.11+ CLI learning platform implementing a three-stage "Build-Justify-Harden" pedagogical loop, packaged as the `mastery` console script (pyproject.toml:34 -> engine.main:main at engine/main.py:2937). The CLI layer is a Typer app (engine/main.py:56) with Rich-rendered output that dispatches subcommands (init, show/next, submit, start-challenge, cleanup, flag-issue, plus legacy submit-build/justification/fix). It wires together six core subsystems imported at engine/main.py:38-51: (1) StateManager (engine/state.py) persists per-user UserProgress (engine/schemas.py CurriculumType/UserProgress) to a JSON state file with corruption handling; (2) CurriculumManager (engine/curriculum.py:31) loads manifest.json-described curricula in LINEAR (sequential modules) or LIBRARY (freeform) modes, raising CurriculumNotFound/Invalid errors; (3) WorkspaceManager (engine/workspace.py:24) implements process isolation via an ephemeral Git "shadow worktree" (.mastery_engine_worktree, configured engine/main.py:70-77) so user code runs against an isolated filesystem copy, applying bug patches (PatchApplicationError at engine/workspace.py:182); (4) ValidationSubsystem (engine/validator.py:26) shells out to per-problem validator.sh / pytest harnesses with timeout and execution-error handling; (5) the Harden stage (engine/stages/harden.py HardenRunner) drives runtime AST mutation through engine/ast_harden/ — GenericBugInjector (engine/ast_harden/generic_injector.py:19), a pattern_matcher, and the softmax v2.1 injector — which parse solutions into an AST, match semantic patterns, and surgically inject logic bugs described by per-problem bugs/*.json specs (with .patch and _symptom.txt companions); and (6) the Justify stage (engine/stages/justify.py JustifyRunner) plus LLMService (engine/services/llm_service.py:27) which evaluates natural-language justifications against rubrics using the OpenAI client (GPT-4o, Chain-of-Thought), degrading to a mock auto-pass mode when no API key is configured (ConfigurationError at engine/services/llm_service.py:329). Supporting code: engine/services/ast_service.py and engine/dev_tools/bug_author.py author/apply mutations, engine/utils.py provides find_project_root and helpers. Content lives under curricula/ — cs336_a1 (Stanford CS336 transformer-from-scratch, 21-22 modules listed in curricula/cs336_a1/manifest.json) and cp_accelerator (competitive-programming LeetCode patterns with a canonical_curriculum.json source-of-truth regenerated into manifest.json by scripts/generate_manifest.py) — each problem/module bundling build_prompt.txt, justify_questions.json, solution.py, test_cases.json, validator.sh and bug specs. A scripts/ directory provides the content pipeline (generate_module.py, parse_sources.py, enrich_problems.py, generate_ground_truth.py, manifest validation). Two GitHub Actions workflows (.github/workflows/tests.yml, validate_cp_manifest.yml) run pytest+ruff and enforce cp_accelerator manifest integrity. Tests under tests/ split into engine unit tests (tests/engine/), e2e BJH-loop tests (tests/e2e/), integration LLM tests, and the inherited CS336 model/tokenizer/optimizer tests. Extensive docs/ (much under docs/internal/archive session logs) and audits/ capture development history.

## Provisional intent (PROVISIONAL — refined/replaced in Stage 4)
PROVISIONAL INTENT (Stage-1 hypothesis, to be refined in Stage 4): This project exists to be a curriculum-agnostic "pedagogical operating system" CLI that teaches deep technical mastery of software/ML concepts by forcing learners through a Build-Justify-Harden loop: implement a component from a spec (Build, validated by automated test harnesses), defend their conceptual understanding in natural language (Justify, graded by an LLM), and debug a deliberately AST-injected semantic bug in their own working code under Git shadow-worktree isolation (Harden). It appears built primarily as an engineering portfolio / self-study platform showcasing runtime AST mutation, process isolation, LLM-as-evaluator, and an automated content-generation pipeline, shipping with two concrete curricula (Stanford CS336 language-modeling and a competitive-programming/LeetCode accelerator). This intent is PROVISIONAL and serves as the coverage-of-intent denominator against which Stage 2 judges defects until Stage 4 confirms or revises it.

## Entry points
| Name | Kind | Location | Description |
| --- | --- | --- | --- |
| mastery | cli_command | pyproject.toml:34 | Top-level CLI command installed by pip/uv that maps to engine.main:main, exposing all mastery subcommands (init, show, submit, start-challenge, cleanup, etc.). |
| Engine Tests / test | github-actions-job | .github/workflows/tests.yml:10 | CI job that installs dependencies and runs pytest over tests/engine/ (excluding integration tests) with coverage reporting. |
| Engine Tests / lint | github-actions-job | .github/workflows/tests.yml:60 | CI job that runs ruff linter and formatter check over engine/ and tests/ directories. |
| CP Accelerator - Manifest Integrity Check / validate-manifest | github-actions-job | .github/workflows/validate_cp_manifest.yml:15 | CI job that validates canonical_curriculum.json, regenerates manifest.json via scripts/generate_manifest.py, and asserts no manual edits were made to manifest.json. |
| CP Accelerator - Manifest Integrity Check / schema-validation | github-actions-job | .github/workflows/validate_cp_manifest.yml:112 | CI job that validates the structure of canonical_curriculum.json and manifest.json against required field schemas using inline Python. |
| CP Accelerator - Manifest Integrity Check / dependency-graph-analysis | github-actions-job | .github/workflows/validate_cp_manifest.yml:201 | CI job that generates a dependency graph statistics report for curriculum patterns (roots, leaves, totals) from manifest.json. |
| subsets | function | curricula/cp_accelerator/patterns/backtracking/problems/lc_78/solution.py:7 | Backtracking function generating all subsets (power set) of a unique-element array. |
| solve | exported_api | curricula/cp_accelerator/patterns/backtracking/problems/lc_78/solution.py:36 | Public alias for `subsets` used by the mastery engine test runner to invoke the lc_78 solution. |
| validator.sh (lc_78) | script | curricula/cp_accelerator/patterns/backtracking/problems/lc_78/validator.sh:1 | Shell entry point that loads solution.py and runs it against lc_78 test cases; exits non-zero on failure. |
| validator.sh (lc_90) | script | curricula/cp_accelerator/patterns/backtracking/problems/lc_90/validator.sh:1 | Shell entry point that loads solution.py and runs it against lc_90 Subsets II test cases. |
| search | function | curricula/cp_accelerator/patterns/binary_search/problems/lc_704/solution.py:7 | Iterative binary search returning the index of target in a sorted array, or -1 if not found. |
| solve | exported_api | curricula/cp_accelerator/patterns/binary_search/problems/lc_704/solution.py:38 | Public alias for `search` used by the mastery engine test runner to invoke the lc_704 solution. |
| validator.sh (lc_34) | script | curricula/cp_accelerator/patterns/binary_search/problems/lc_34/validator.sh:1 | Shell entry point that loads solution.py and runs it against lc_34 Find First and Last Position test cases. |
| validator.sh (lc_704) | script | curricula/cp_accelerator/patterns/binary_search/problems/lc_704/validator.sh:1 | Shell entry point that loads solution.py and runs it against lc_704 Binary Search test cases. |
| validator.sh (lc_1342) | script | curricula/cp_accelerator/patterns/bit_manipulation/problems/lc_1342/validator.sh:1 | Shell entry point that loads solution.py and validates it against lc_1342 Number of Steps test cases. |
| validator.sh (lc_1486) | script | curricula/cp_accelerator/patterns/bit_manipulation/problems/lc_1486/validator.sh:1 | Shell entry point that loads solution.py and validates it against lc_1486 XOR Operation test cases. |
| validator.sh (lc_46) | script | curricula/cp_accelerator/patterns/combinatorics_and_number_theory/problems/lc_46/validator.sh:1 | Shell entry point that loads solution.py and validates it against lc_46 Permutations test cases. |
| validator.sh (lc_47) | script | curricula/cp_accelerator/patterns/combinatorics_and_number_theory/problems/lc_47/validator.sh:1 | Shell entry point that loads solution.py and validates it against lc_47 Permutations II test cases. |
| validator.sh (lc_146) | script | curricula/cp_accelerator/patterns/design_patterns/problems/lc_146/validator.sh:1 | Shell entry point that loads solution.py and validates it against lc_146 LRU Cache test cases. |
| validator.sh (lc_460) | script | curricula/cp_accelerator/patterns/design_patterns/problems/lc_460/validator.sh:1 | Shell entry point that loads solution.py and validates it against lc_460 LFU Cache test cases. |
| validator.sh (lc_148) | script | curricula/cp_accelerator/patterns/divide_and_conquer/problems/lc_148/validator.sh:1 | Shell entry point that loads solution.py and validates it against lc_148 Sort List test cases. |
| validator.sh (lc_912) | script | curricula/cp_accelerator/patterns/divide_and_conquer/problems/lc_912/validator.sh:1 | Shell entry point that loads solution.py and validates it against lc_912 Sort an Array test cases. |
| validator.sh (lc_198) | script | curricula/cp_accelerator/patterns/dynamic_programming/problems/lc_198/validator.sh:1 | Shell entry point that loads solution.py and validates it against lc_198 House Robber test cases. |
| climbStairs | function | curricula/cp_accelerator/patterns/dynamic_programming/problems/lc_70/solution.py:7 | O(1)-space DP function counting distinct ways to climb n stairs taking 1 or 2 steps at a time. |
| solve | exported_api | curricula/cp_accelerator/patterns/dynamic_programming/problems/lc_70/solution.py:36 | Public alias for `climbStairs` used by the mastery engine test runner to invoke the lc_70 solution. |
| validator.sh (lc_70) | script | curricula/cp_accelerator/patterns/dynamic_programming/problems/lc_70/validator.sh:1 | Shell entry point that loads solution.py and validates it against lc_70 Climbing Stairs test cases. |
| eraseOverlapIntervals | function | curricula/cp_accelerator/patterns/greedy/problems/lc_435/solution.py:7 | Greedy function returning the minimum number of intervals to remove to make the rest non-overlapping. |
| solve | exported_api | curricula/cp_accelerator/patterns/greedy/problems/lc_435/solution.py:42 | Public alias for `eraseOverlapIntervals` used by the mastery engine test runner to invoke the lc_435 solution. |
| validator.sh (lc_435) | script | curricula/cp_accelerator/patterns/greedy/problems/lc_435/validator.sh:1 | Shell entry point that loads solution.py and validates it against lc_435 Non-overlapping Intervals test cases. |
| validator.sh (lc_452) | script | curricula/cp_accelerator/patterns/greedy/problems/lc_452/validator.sh:1 | Shell entry point that loads solution.py and validates it against lc_452 Minimum Arrows test cases. |
| twoSum | exported-function | curricula/cp_accelerator/patterns/hash_table/problems/lc_1/solution.py:6 | Returns indices of two numbers in nums that sum to target; O(n) time using a hash map. |
| validator (lc_1) | cli-script | curricula/cp_accelerator/patterns/hash_table/problems/lc_1/validator.sh:1 | Bash entry point that imports twoSum and validates it against test_cases.json, exiting non-zero on failure. |
| validator (lc_217) | cli-script | curricula/cp_accelerator/patterns/hash_table/problems/lc_217/validator.sh:1 | Bash entry point that imports containsDuplicate and validates it against test_cases.json. |
| validator (lc_219) | cli-script | curricula/cp_accelerator/patterns/hash_table/problems/lc_219/validator.sh:1 | Bash entry point that imports containsNearbyDuplicate and validates it against test_cases.json. |
| validator (lc_215) | cli-script | curricula/cp_accelerator/patterns/heap_and_priority_queue/problems/lc_215/validator.sh:1 | Bash entry point that imports findKthLargest and validates it against test_cases.json. |
| validator (lc_703) | cli-script | curricula/cp_accelerator/patterns/heap_and_priority_queue/problems/lc_703/validator.sh:1 | Bash entry point that imports KthLargest class and validates it against test_cases.json. |
| removeElements | exported-function | curricula/cp_accelerator/patterns/linked_list/problems/lc_203/solution.py:8 | Removes all linked list elements equal to val; O(n) using array representation for test-runner compatibility. |
| validator (lc_203) | cli-script | curricula/cp_accelerator/patterns/linked_list/problems/lc_203/validator.sh:1 | Bash entry point that imports removeElements and validates it against test_cases.json. |
| validator (lc_237) | cli-script | curricula/cp_accelerator/patterns/linked_list/problems/lc_237/validator.sh:1 | Bash entry point that imports deleteNode and validates it against test_cases.json. |
| validator (lc_1480) | cli-script | curricula/cp_accelerator/patterns/prefix_sum/problems/lc_1480/validator.sh:1 | Bash entry point that imports runningSum and validates it against test_cases.json. |
| sumRange | exported-function | curricula/cp_accelerator/patterns/prefix_sum/problems/lc_303/solution.py:7 | Returns prefix-sum range query result for indices [left, right] inclusive; O(1) after O(n) build. |
| validator (lc_303) | cli-script | curricula/cp_accelerator/patterns/prefix_sum/problems/lc_303/validator.sh:1 | Bash entry point that imports sumRange and validates it against test_cases.json. |
| validator (lc_307) | cli-script | curricula/cp_accelerator/patterns/segment_tree_and_fenwick_tree/problems/lc_307/validator.sh:1 | Bash entry point that imports NumArray class and validates it against test_cases.json. |
| validator (lc_148) | cli-script | curricula/cp_accelerator/patterns/sorting/problems/lc_148/validator.sh:1 | Bash entry point that imports sortList and validates it against test_cases.json. |
| sortArray | exported-function | curricula/cp_accelerator/patterns/sorting/problems/lc_912/solution.py:6 | Sorts an integer array using merge sort; O(n log n) time, O(n) space. |
| validator (lc_912) | cli-script | curricula/cp_accelerator/patterns/sorting/problems/lc_912/validator.sh:1 | Bash entry point that imports sortArray and validates it against test_cases.json. |
| validator (lc_1003) | cli-script | curricula/cp_accelerator/patterns/stack_and_queue/problems/lc_1003/validator.sh:1 | Bash entry point that imports isValid and validates it against test_cases.json for lc_1003. |
| isValid | exported-function | curricula/cp_accelerator/patterns/stack_and_queue/problems/lc_20/solution.py:7 | Validates bracket nesting in string s using a stack; O(n) time and space. |
| validator (lc_20) | cli-script | curricula/cp_accelerator/patterns/stack_and_queue/problems/lc_20/validator.sh:1 | Bash entry point that imports isValid and validates it against test_cases.json for lc_20. |
| validator.sh (lc_144) | shell-script | curricula/cp_accelerator/patterns/traversal/problems/lc_144/validator.sh:1 | Runs student solution.py against LC-144 example test cases and reports pass/fail counts. |
| validator.sh (lc_589) | shell-script | curricula/cp_accelerator/patterns/traversal/problems/lc_589/validator.sh:1 | Runs student solution.py against LC-589 example test cases and reports pass/fail counts. |
| validator.sh (lc_1804) | shell-script | curricula/cp_accelerator/patterns/trie/problems/lc_1804/validator.sh:1 | Runs student solution.py against LC-1804 example test cases and reports pass/fail counts. |
| validator.sh (lc_208) | shell-script | curricula/cp_accelerator/patterns/trie/problems/lc_208/validator.sh:1 | Runs student solution.py against LC-208 example test cases and reports pass/fail counts. |
| validator.sh (lc_1099) | shell-script | curricula/cp_accelerator/patterns/two_pointers/problems/lc_1099/validator.sh:1 | Runs student solution.py against LC-1099 example test cases and reports pass/fail counts. |
| validator.sh (lc_167) | shell-script | curricula/cp_accelerator/patterns/two_pointers/problems/lc_167/validator.sh:1 | Runs student solution.py against LC-167 example and edge-case test cases, reports pass/fail counts. |
| validator.sh (lc_547) | shell-script | curricula/cp_accelerator/patterns/union_find_disjoint_set_union/problems/lc_547/validator.sh:1 | Runs student solution.py against LC-547 example test cases and reports pass/fail counts. |
| validator.sh (lc_684) | shell-script | curricula/cp_accelerator/patterns/union_find_disjoint_set_union/problems/lc_684/validator.sh:1 | Runs student solution.py against LC-684 example test cases and reports pass/fail counts. |
| twoSum | function | curricula/cp_accelerator/patterns/two_pointers/problems/lc_167/solution.py:7 | Reference O(n)/O(1) two-pointer implementation returning 1-indexed pair indices for LC-167. |
| solve | function | curricula/cp_accelerator/patterns/two_pointers/problems/lc_167/solution.py:38 | Alias for twoSum used by the generic test runner (validator.sh calls solve(**test['input'])). |
| validator.sh (adamw) | shell-script | curricula/cs336_a1/modules/adamw/validator.sh:1 | Runs pytest for the AdamW optimizer implementation inside a shadow worktree. |
| validator.sh (attention) | shell-script | curricula/cs336_a1/modules/attention/validator.sh:1 | Runs pytest for the scaled dot-product attention implementation inside a shadow worktree. |
| validator.sh (bpe_tokenizer) | shell-script | curricula/cs336_a1/modules/bpe_tokenizer/validator.sh:1 | Runs pytest for the BPE tokenizer training implementation inside a shadow worktree. |
| validator.sh (checkpointing) | shell-script | curricula/cs336_a1/modules/checkpointing/validator.sh:1 | Runs pytest for the model checkpointing save/load implementation inside a shadow worktree. |
| validator.sh (cosine_schedule) | shell-script | curricula/cs336_a1/modules/cosine_schedule/validator.sh:1 | Runs pytest for the cosine LR schedule implementation inside a shadow worktree. |
| validator.sh (cross_entropy) | shell-script | curricula/cs336_a1/modules/cross_entropy/validator.sh:1 | Runs pytest for the numerically stable cross-entropy implementation inside a shadow worktree. |
| validator.sh (data_loader) | shell-script | curricula/cs336_a1/modules/data_loader/validator.sh:1 | Runs pytest for the LM data loader (get_batch) implementation inside a shadow worktree. |
| validate_data_loader | shell-script | curricula/cs336_a1/modules/data_loader/validator.sh:1 | Mastery engine validation entry point: copies utils.py to shadow worktree and runs pytest tests/test_training.py::test_get_batch. |
| validate_embedding | shell-script | curricula/cs336_a1/modules/embedding/validator.sh:1 | Mastery engine validation entry point: copies layers.py to shadow worktree and runs pytest tests/test_model.py::test_embedding. |
| validate_gradient_clipping | shell-script | curricula/cs336_a1/modules/gradient_clipping/validator.sh:1 | Mastery engine validation entry point: runs pytest tests/test_nn_utils.py::test_gradient_clipping in shadow worktree. |
| validate_linear | shell-script | curricula/cs336_a1/modules/linear/validator.sh:1 | Mastery engine validation entry point: runs pytest tests/test_model.py::test_linear in shadow worktree. |
| validate_multihead_attention | shell-script | curricula/cs336_a1/modules/multihead_attention/validator.sh:1 | Mastery engine validation entry point: runs pytest tests/test_model.py::test_multihead_self_attention_with_rope in shadow worktree. |
| validate_rmsnorm | shell-script | curricula/cs336_a1/modules/rmsnorm/validator.sh:1 | Mastery engine validation entry point: runs pytest tests/test_model.py::test_rmsnorm in shadow worktree. |
| validate_rope | shell-script | curricula/cs336_a1/modules/rope/validator.sh:1 | Mastery engine validation entry point: runs pytest tests/test_model.py::test_rope in shadow worktree. |
| validate_silu | shell-script | curricula/cs336_a1/modules/silu/validator.sh:1 | Mastery engine validation entry point: runs pytest tests/test_model.py::test_silu_matches_pytorch in shadow worktree. |
| validate_softmax | shell-script | curricula/cs336_a1/modules/softmax/validator.sh:1 | Mastery engine validation entry point: runs pytest tests/test_nn_utils.py::test_softmax_matches_pytorch in shadow worktree. |
| validate_swiglu | shell-script | curricula/cs336_a1/modules/swiglu/validator.sh:1 | Mastery engine validation entry point: runs pytest tests/test_model.py::test_swiglu in shadow worktree. |
| validate_text_generation | shell-script | curricula/cs336_a1/modules/text_generation/validator.sh:1 | Mastery engine validation entry point: runs pytest tests/test_generation.py::test_generate in shadow worktree. |
| validate_tokenizer_class | shell-script | curricula/cs336_a1/modules/tokenizer_class/validator.sh:1 | Mastery engine validation entry point: runs pytest tests/test_tokenizer.py::test_tokenizer_class in shadow worktree. |
| training_loop_validator | script | curricula/cs336_a1/modules/training_loop/validator.sh:1 | Invoked by the ValidationSubsystem to run pytest test_training.py::test_train_loop against a shadow worktree copy of training.py. |
| transformer_block_validator | script | curricula/cs336_a1/modules/transformer_block/validator.sh:1 | Invoked by the ValidationSubsystem to run pytest test_model.py::test_transformer_block against a shadow worktree copy of layers.py. |
| transformer_lm_validator | script | curricula/cs336_a1/modules/transformer_lm/validator.sh:1 | Invoked by the ValidationSubsystem to run pytest test_model.py::test_transformer_lm against a shadow worktree copy of layers.py. |
| hello_world_validator | script | curricula/dummy_hello_world/modules/hello_world/validator.sh:1 | Invoked by the ValidationSubsystem to check that hello_world.py exists in the student workspace, returning PERFORMANCE_SECONDS. |
| data_parsing_extraction_validator | script | curricula/job_prep_data_annotation/modules/data_parsing_extraction/validator.sh:1 | Invoked by the ValidationSubsystem; runs embedded Python tests for extract_coordinates() including format variation and performance checks. |
| grid_visualization_validator | script | curricula/job_prep_data_annotation/modules/grid_visualization/validator.sh:1 | Invoked by the ValidationSubsystem; runs embedded Python tests for render_grid() including aliasing detection and edge coordinates. |
| http_transport_validator | script | curricula/job_prep_data_annotation/modules/http_transport/validator.sh:1 | Invoked by the ValidationSubsystem; runs embedded Python tests for fetch_document() covering HTTP GET success and error-handling paths. |
| std_lib_augmentation_validator | script | curricula/python_for_cp/modules/std_lib_augmentation/validator.sh:1 | Invoked by the ValidationSubsystem; runs embedded Python tests for shortest_path_bfs, dijkstra_shortest_path, and count_in_range. |
| main | script_entry_point | engine/main.py:2937 | Top-level entry point that calls the Typer app; registered as the 'engine' CLI command in pyproject.toml. |
| submit | cli_command | engine/main.py:815 | Auto-detects the current BJH stage (build/justify/harden) and runs the appropriate validation workflow. |
| show | cli_command | engine/main.py:1100 | Read-only display of the current or specified module/problem challenge content (build prompt, justify question, or harden instructions). |
| start-challenge | cli_command | engine/main.py:1156 | Initializes the Harden stage by injecting a bug into the shadow worktree and displaying the symptom description. |
| next | cli_command | engine/main.py:1324 | Deprecated command that forwards to 'show'; retained for backward compatibility. |
| submit-build | cli_command | engine/main.py:1354 | Deprecated legacy command to validate a Build stage implementation; replaced by 'submit'. |
| submit-justification | cli_command | engine/main.py:1529 | Deprecated legacy command to submit an inline Justify stage answer string; replaced by 'submit'. |
| submit-fix | cli_command | engine/main.py:1724 | Deprecated legacy command to validate a Harden stage bug fix; replaced by 'submit'. |
| init | cli_command | engine/main.py:1917 | Initializes the Mastery Engine: verifies git repo, validates curriculum, creates shadow git worktree, syncs uncommitted files, and writes initial state. |
| curriculum-list | cli_command | engine/main.py:2160 | Displays all modules in the current curriculum with their completion status in a Rich table. |
| progress-reset | cli_command | engine/main.py:2256 | Resets a specific module's progress back to the build stage after user confirmation. |
| reset | cli_command | engine/main.py:2384 | Resets module or entire curriculum progress; --hard restores all files from the shadow worktree pristine state. |
| cleanup | cli_command | engine/main.py:2487 | Removes the shadow git worktree to free disk space when finished with the curriculum. |
| select | cli_command | engine/main.py:2684 | LIBRARY-mode command to set the active pattern and problem, resetting the user to the Build stage. |
| status | cli_command | engine/main.py:2804 | Displays current learning progress (curriculum, module/problem, stage, completions) for both LINEAR and LIBRARY curricula. |
| create-bug | cli_command | engine/main.py:2862 | Developer tool that uses LLM few-shot learning to generate a v2.1 JSON bug definition from a .patch file. |
| inject_softmax_bug | exported_function | engine/ast_harden/softmax_poc.py:147 | Complete Phase-1 PoC pipeline (parse → canonicalize → inject → unparse) for the softmax no-subtract-max bug; returns (buggy_code, success). |
| inject_softmax_bug_v2_1 | exported_function | engine/ast_harden/softmax_v2_1.py:229 | v2.1 two-phase pipeline that canonicalizes for matching but transforms the original AST, preserving student variable names; returns (buggy_code, success). |
| __main__ | __main__ | engine/ast_harden/softmax_poc.py:209 | Runnable test harness for inject_softmax_bug with two standard/alternative-naming softmax implementations. |
| __main__ | __main__ | engine/ast_harden/softmax_v2_1.py:310 | Runnable test harness for inject_softmax_bug_v2_1 demonstrating the two-phase mapping approach on two softmax variants. |
| make_submission.sh | script | maintenance/make_submission.sh:1 | Runs the test suite via pytest then zips the project directory into cs336-spring2025-assignment-1-submission.zip for submission. |
| train_bpe | exported_function | modes/developer/cs336_basics/bpe.py:10 | Trains a BPE tokenizer from a text corpus and returns (vocab dict, merges list). |
| Tokenizer | exported_class | modes/developer/cs336_basics/tokenizer.py:8 | Byte-level BPE tokenizer with encode, decode, and encode_iterable backed by tiktoken GPT-2 encoding. |
| AdamW | exported_class | modes/developer/cs336_basics/optimizer.py:9 | AdamW optimizer with decoupled weight decay extending torch.optim.Optimizer. |
| transformer_lm | exported_function | modes/developer/cs336_basics/layers.py:339 | Full transformer language model forward pass returning logits given input token indices and a weights dict. |
| transformer_block | exported_function | modes/developer/cs336_basics/layers.py:223 | Single pre-norm transformer block applying RoPE multi-head attention and SwiGLU FFN with residual connections. |
| rope | exported_function | modes/developer/cs336_basics/layers.py:285 | Applies Rotary Positional Embeddings in-place to a query or key tensor. |
| scaled_dot_product_attention | exported_function | modes/developer/cs336_basics/layers.py:129 | Numerically stable scaled dot-product attention with optional boolean causal mask. |
| multihead_self_attention_with_rope | exported_function | modes/developer/cs336_basics/layers.py:398 | Batched causal multi-head self-attention with per-head RoPE applied to Q and K. |
| find_chunk_boundaries | exported_function | modes/developer/cs336_basics/pretokenization_example.py:5 | Splits a binary file into N byte-aligned chunks at special-token boundaries for parallel pre-tokenization. |
| softmax | exported_function | modes/developer/cs336_basics/utils.py:5 | Numerically stable softmax using subtract-max trick with float32 upcasting. |
| cross_entropy | exported_function | modes/developer/cs336_basics/utils.py:25 | Numerically stable cross-entropy loss using log-sum-exp trick, averaged over batch. |
| gradient_clipping | exported_function | modes/developer/cs336_basics/utils.py:50 | Clips gradients in-place so global L2 norm does not exceed max_l2_norm. |
| get_lr_cosine_schedule | exported_function | modes/developer/cs336_basics/utils.py:75 | Returns scalar LR for a given iteration following linear warmup then cosine decay schedule. |
| get_batch | exported_function | modes/developer/cs336_basics/utils.py:114 | Randomly samples (x, y) LongTensor batches from a 1D token-ID array for language model training. |
| save_checkpoint | exported_function | modes/developer/cs336_basics/utils.py:142 | Serializes model and optimizer state dicts plus iteration count to a file path or file-like. |
| load_checkpoint | exported_function | modes/developer/cs336_basics/utils.py:154 | Restores model and optimizer state from a checkpoint and returns the saved iteration number. |
| train_bpe | exported_function | modes/student/cs336_basics/bpe.py:8 | Student stub for BPE training; raises NotImplementedError with step-by-step implementation guide. |
| generate | exported_function | modes/student/cs336_basics/generation.py:5 | Student stub for autoregressive text generation with temperature/top-k/top-p; raises NotImplementedError. |
| AdamW | exported_class | modes/student/cs336_basics/optimizer.py:9 | Student stub AdamW class; both __init__ and step raise NotImplementedError. |
| Tokenizer | exported_class | modes/student/cs336_basics/tokenizer.py:5 | Complete Tokenizer provided to students as a working helper using tiktoken GPT-2 encoding. |
| Tokenizer | exported_class | modes/student/cs336_basics/tokenizer_stub.py:5 | Student stub Tokenizer class for from-scratch BPE tokenizer implementation exercise. |
| find_chunk_boundaries | exported_function | modes/student/cs336_basics/pretokenization_example.py:5 | Provided-complete file chunking helper for parallel corpus pre-tokenization (student copy). |
| main | __main__ | scripts/add_successful_to_golden.py:101 | CLI entry: reads /tmp/llm_evaluation_results.json and interactively promotes successful bug definitions to the golden dataset. |
| main | __main__ | scripts/auto_fix_drafts.py:295 | CLI entry: applies four hardcoded fixes to draft AST injection patterns and saves corrected JSON files. |
| main | __main__ | scripts/enrich_problems.py:373 | CLI entry (--rate-limit, --input, --output): fetches LeetCode problem details and writes enriched canonical_curriculum.json. |
| fetch_sources.sh | shell_script | scripts/fetch_sources.sh:1 | Shell entry: clones 30-Days-Of-Python and creates CP accelerator taxonomy placeholder into .sources/. |
| main | __main__ | scripts/fix_draft_pattern.py:175 | CLI entry: interactively walks draft AST patterns, tests them against patches, and helps the developer fix and promote each. |
| main | __main__ | scripts/generate_ground_truth.py:274 | CLI entry: uses gpt-4o to generate golden AST injection pattern JSON for all CS336-A1 modules that lack one. |
| main | __main__ | scripts/generate_manifest.py:305 | CLI entry (--validate-only, --canonical, --output): validates and regenerates manifest.json from canonical_curriculum.json. |
| main | __main__ | scripts/generate_module.py:581 | CLI entry (--problem-id, --all, --force, --limit-per-pattern): generates module assets for one or all problems in the cp_accelerator curriculum. |
| batch_generate_all | function | scripts/generate_module.py:691 | Batch sub-entry invoked by main() when --all flag is set; generates problem directories for every pattern in the library curriculum. |
| main | __main__ | scripts/migrate_bugs_llm.py:141 | CLI entry: scans curricula/cs336_a1 for .patch files without JSON counterparts and generates them via BugAuthor LLM. |
| mode | shell_cli | scripts/mode:149 | CLI entry: dispatches to status/switch/test sub-commands managing student↔developer workspace symlink via .active-mode. |
| main | __main__ | scripts/parse_sources.py:604 | CLI entry (--validate-urls): parses DSA taxonomy files and RoadmapResources.md to produce curricula/cp_accelerator/canonical_curriculum.json. |
| main | __main__ | scripts/systematic_llm_evaluation.py:1267 | CLI entry: runs all 21 CS336-A1 bugs through the LLM evaluator (3 attempts each) and prints statistics plus regression check. |
| check_regression | function | scripts/systematic_llm_evaluation.py:1222 | Post-evaluation regression gate: compares current success count against a baseline of 3/4 bugs and warns on regressions. |
| test_ci.sh | shell_script | scripts/test_ci.sh:1 | Shell entry: runs uv run pytest tests/engine/ -m 'not integration' to replicate GitHub Actions CI locally. |
| test_library_loading | __main__ | scripts/test_library_loading.py:132 | Test entry: exercises CurriculumManager LIBRARY loading, path resolution, and on-disk file existence assertions. |
| main | __main__ | scripts/validate_student_stubs.py:177 | CLI entry: validates all modes/student/**/*.py files contain NotImplementedError stubs; exits 1 if complete implementations found. |
| verify_curriculum_manifest | __main__ | scripts/verify_curriculum_manifests.py:92 | CLI entry: verifies curricula/cs336_a1 manifest by checking all declared module directories have required files. |
| main | __main__ | scripts/verify_ground_truth.py:162 | CLI entry: tests every golden AST injection pattern against its patch transformation and exits 1 if any pattern fails. |
| __main__ (debug_shadow_worktree) | script | tests/e2e/debug_shadow_worktree.py:81 | Runs pytest on the debug_shadow_worktree script itself when executed directly to exercise shadow-worktree inspection logic. |
| test_train_bpe_speed | test_function | tests/test_train_bpe.py:8 | Asserts BPE training on corpus.en with vocab_size=500 completes in under 1.5 seconds. |
| test_train_bpe | test_function | tests/test_train_bpe.py:27 | Validates learned merges count (243-245) and vocabulary coverage (>=98%) against GPT-2 reference files. |
| test_train_bpe_special_tokens | test_function | tests/test_train_bpe.py:87 | Verifies special tokens appear in vocab and are never merged into other tokens, with snapshot assertion. |

## File-role summary
- **config**: 96
- **doc**: 207
- **asset**: 106
- **source**: 105
- **test**: 81
- **generated**: 14

## Full inventory (609 files)
| Path | Role | Purpose |
| --- | --- | --- |
| .env.example | config | Template for required environment variables (OPENAI_API_KEY and optional debug flags); copy to .env before running. |
| .gitignore | config | Git ignore rules for Python build artifacts, virtual envs, test caches, IDE files, and project-specific derived files like the shadow worktree and mode symlink. |
| LICENSE | doc | MIT License for original engine code with third-party attribution notices for Stanford CS336, LeetCode, and 30 Days of Python curriculum content. |
| NOTICE | doc | NOTICE file listing all third-party content attributions and clarifying which components are original engineering work under MIT License. |
| README.md | doc | Primary project documentation covering quick-start demo, CLI usage, architecture overview, curriculum descriptions, and the Build-Justify-Harden pedagogical loop. |
| pyproject.toml | config | Python project manifest defining package metadata, all runtime dependencies, the `mastery` CLI entry point, pytest/ruff tool configuration, and uv build system settings. |
| .github/workflows/tests.yml | config | GitHub Actions CI workflow that runs pytest unit tests and ruff lint/format checks for the engine package on push/PR to main. |
| .github/workflows/validate_cp_manifest.yml | config | GitHub Actions CI workflow that validates and regenerates the cp_accelerator manifest.json, checks schema integrity, and analyzes the dependency graph on changes to curricula/cp_accelerator. |
| audits/META_AUDIT_DEC_18.md | doc | Meta-audit identifying coverage gaps and blind spots in QUALITY_AUDIT.md, with a prioritized follow-up audit backlog for unreviewed repository surfaces. |
| audits/QUALITY_AUDIT.md | doc | Primary quality and resilience audit artifact for the Mastery Engine, cataloguing 34 findings (9 high/22 medium/3 low) across architecture, security, CI, curricula, and scripts. |
| curricula/cp_accelerator/IMPLEMENTATION_STATUS.md | doc | Documents architectural decisions, solved design flaws, and implementation milestones for the CP Accelerator curriculum pipeline. |
| curricula/cp_accelerator/README.md | doc | Overview of the Competitive Programming Accelerator curriculum with attribution, content ownership, and original engineering contributions. |
| curricula/cp_accelerator/STATUS.md | doc | Current completion status showing all 19 DSA taxonomy patterns parsed and resources partially extracted as of 2025-11-17. |
| curricula/cp_accelerator/canonical_curriculum.json | config | Canonical source-of-truth JSON defining the full curriculum structure, patterns, resources, and rating brackets for the CP Accelerator. |
| curricula/cp_accelerator/manifest.json | config | Curriculum manifest JSON with metadata, version, sources, and pattern list for the cp_accelerator curriculum type. |
| curricula/cp_accelerator/patterns/backtracking/problems/lc_78/bugs/missing_copy.json | config | AST-based bug injection spec that replaces the list shallow copy `current[:]` with a reference `current` in the Subsets backtracking solution. |
| curricula/cp_accelerator/patterns/backtracking/problems/lc_78/bugs/missing_copy_symptom.txt | doc | Human-readable symptom description and debugging guide for the missing list copy bug in the Subsets backtracking problem. |
| curricula/cp_accelerator/patterns/backtracking/problems/lc_78/build_prompt.txt | asset | Problem statement, constraints, and implementation instructions presented to the learner for LeetCode 78 Subsets. |
| curricula/cp_accelerator/patterns/backtracking/problems/lc_78/justify_questions.json | asset | Structured Socratic Q&A with model answers and failure-mode feedback for the Subsets backtracking justification phase. |
| curricula/cp_accelerator/patterns/backtracking/problems/lc_78/solution.py | source | Reference solution for LeetCode 78 Subsets using backtracking with shallow-copy result accumulation; exports `solve` alias for test runner. |
| curricula/cp_accelerator/patterns/backtracking/problems/lc_78/test_cases.json | test | Example test cases from the LeetCode 78 Subsets problem statement used by the local validator. |
| curricula/cp_accelerator/patterns/backtracking/problems/lc_78/validator.sh | test | Shell script that imports solution.py and runs it against test_cases.json to validate the learner's Subsets implementation. |
| curricula/cp_accelerator/patterns/backtracking/problems/lc_90/build_prompt.txt | asset | Problem statement and implementation instructions presented to the learner for LeetCode 90 Subsets II (with duplicates). |
| curricula/cp_accelerator/patterns/backtracking/problems/lc_90/test_cases.json | test | Example test cases from the LeetCode 90 Subsets II problem statement used by the local validator. |
| curricula/cp_accelerator/patterns/backtracking/problems/lc_90/validator.sh | test | Shell script that imports solution.py and runs it against test_cases.json to validate the learner's Subsets II implementation. |
| curricula/cp_accelerator/patterns/binary_search/problems/lc_34/build_prompt.txt | asset | Problem statement and implementation instructions presented to the learner for LeetCode 34 Find First and Last Position of Element. |
| curricula/cp_accelerator/patterns/binary_search/problems/lc_34/test_cases.json | test | Example test cases from the LeetCode 34 problem statement used by the local validator. |
| curricula/cp_accelerator/patterns/binary_search/problems/lc_34/validator.sh | test | Shell script that imports solution.py and validates the learner's Find First and Last Position implementation. |
| curricula/cp_accelerator/patterns/binary_search/problems/lc_704/bugs/wrong_loop_condition.json | config | AST-based bug injection spec that replaces `left <= right` with `left < right` in the Binary Search while loop condition. |
| curricula/cp_accelerator/patterns/binary_search/problems/lc_704/bugs/wrong_loop_condition_symptom.txt | doc | Human-readable symptom description showing wrong loop condition causes Binary Search to miss targets when left equals right. |
| curricula/cp_accelerator/patterns/binary_search/problems/lc_704/build_prompt.txt | asset | Problem statement and implementation instructions presented to the learner for LeetCode 704 Binary Search. |
| curricula/cp_accelerator/patterns/binary_search/problems/lc_704/justify_questions.json | asset | Structured Socratic Q&A with model answers and failure-mode feedback for the Binary Search justification phase. |
| curricula/cp_accelerator/patterns/binary_search/problems/lc_704/solution.py | source | Reference solution for LeetCode 704 Binary Search using iterative halving; exports `solve` alias for the test runner. |
| curricula/cp_accelerator/patterns/binary_search/problems/lc_704/test_cases.json | test | Example test cases from the LeetCode 704 Binary Search problem statement used by the local validator. |
| curricula/cp_accelerator/patterns/binary_search/problems/lc_704/validator.sh | test | Shell script that imports solution.py and validates the learner's Binary Search implementation against example test cases. |
| curricula/cp_accelerator/patterns/bit_manipulation/problems/lc_1342/build_prompt.txt | asset | Problem statement and implementation instructions presented to the learner for LeetCode 1342 Number of Steps to Reduce a Number to Zero. |
| curricula/cp_accelerator/patterns/bit_manipulation/problems/lc_1342/test_cases.json | test | Example test cases from the LeetCode 1342 problem statement used by the local validator. |
| curricula/cp_accelerator/patterns/bit_manipulation/problems/lc_1342/validator.sh | test | Shell script that imports solution.py and validates the learner's Number of Steps to Zero implementation. |
| curricula/cp_accelerator/patterns/bit_manipulation/problems/lc_1486/build_prompt.txt | asset | Problem statement and implementation instructions presented to the learner for LeetCode 1486 XOR Operation in an Array. |
| curricula/cp_accelerator/patterns/bit_manipulation/problems/lc_1486/test_cases.json | test | Example test cases from the LeetCode 1486 XOR Operation problem statement used by the local validator. |
| curricula/cp_accelerator/patterns/bit_manipulation/problems/lc_1486/validator.sh | test | Shell script that imports solution.py and validates the learner's XOR Operation in an Array implementation. |
| curricula/cp_accelerator/patterns/combinatorics_and_number_theory/problems/lc_46/build_prompt.txt | asset | Problem statement and implementation instructions presented to the learner for LeetCode 46 Permutations. |
| curricula/cp_accelerator/patterns/combinatorics_and_number_theory/problems/lc_46/test_cases.json | test | Example test cases from the LeetCode 46 Permutations problem statement used by the local validator. |
| curricula/cp_accelerator/patterns/combinatorics_and_number_theory/problems/lc_46/validator.sh | test | Shell script that imports solution.py and validates the learner's Permutations implementation. |
| curricula/cp_accelerator/patterns/combinatorics_and_number_theory/problems/lc_47/build_prompt.txt | asset | Problem statement and implementation instructions presented to the learner for LeetCode 47 Permutations II (with duplicates). |
| curricula/cp_accelerator/patterns/combinatorics_and_number_theory/problems/lc_47/test_cases.json | test | Example test cases from the LeetCode 47 Permutations II problem statement used by the local validator. |
| curricula/cp_accelerator/patterns/combinatorics_and_number_theory/problems/lc_47/validator.sh | test | Shell script that imports solution.py and validates the learner's Permutations II implementation. |
| curricula/cp_accelerator/patterns/design_patterns/problems/lc_146/build_prompt.txt | asset | Problem statement and implementation instructions presented to the learner for LeetCode 146 LRU Cache. |
| curricula/cp_accelerator/patterns/design_patterns/problems/lc_146/test_cases.json | test | Example test cases from the LeetCode 146 LRU Cache problem statement used by the local validator. |
| curricula/cp_accelerator/patterns/design_patterns/problems/lc_146/validator.sh | test | Shell script that imports solution.py and validates the learner's LRU Cache implementation. |
| curricula/cp_accelerator/patterns/design_patterns/problems/lc_460/build_prompt.txt | asset | Problem statement and implementation instructions presented to the learner for LeetCode 460 LFU Cache. |
| curricula/cp_accelerator/patterns/design_patterns/problems/lc_460/test_cases.json | test | Example test cases from the LeetCode 460 LFU Cache problem statement used by the local validator. |
| curricula/cp_accelerator/patterns/design_patterns/problems/lc_460/validator.sh | test | Shell script that imports solution.py and validates the learner's LFU Cache implementation. |
| curricula/cp_accelerator/patterns/divide_and_conquer/problems/lc_148/build_prompt.txt | asset | Problem statement and implementation instructions presented to the learner for LeetCode 148 Sort List. |
| curricula/cp_accelerator/patterns/divide_and_conquer/problems/lc_148/test_cases.json | test | Example test cases from the LeetCode 148 Sort List problem statement used by the local validator. |
| curricula/cp_accelerator/patterns/divide_and_conquer/problems/lc_148/validator.sh | test | Shell script that imports solution.py and validates the learner's Sort List implementation. |
| curricula/cp_accelerator/patterns/divide_and_conquer/problems/lc_912/build_prompt.txt | asset | Problem statement and implementation instructions presented to the learner for LeetCode 912 Sort an Array. |
| curricula/cp_accelerator/patterns/divide_and_conquer/problems/lc_912/test_cases.json | test | Example test cases from the LeetCode 912 Sort an Array problem statement used by the local validator. |
| curricula/cp_accelerator/patterns/divide_and_conquer/problems/lc_912/validator.sh | test | Shell script that imports solution.py and validates the learner's Sort an Array implementation. |
| curricula/cp_accelerator/patterns/dynamic_programming/problems/lc_198/build_prompt.txt | asset | Problem statement and implementation instructions presented to the learner for LeetCode 198 House Robber. |
| curricula/cp_accelerator/patterns/dynamic_programming/problems/lc_198/test_cases.json | test | Example test cases from the LeetCode 198 House Robber problem statement used by the local validator. |
| curricula/cp_accelerator/patterns/dynamic_programming/problems/lc_198/validator.sh | test | Shell script that imports solution.py and validates the learner's House Robber implementation. |
| curricula/cp_accelerator/patterns/dynamic_programming/problems/lc_70/bugs/wrong_base_case.json | config | AST-based bug injection spec that changes the DP base case from `n <= 2` to `n <= 1` in the Climbing Stairs solution. |
| curricula/cp_accelerator/patterns/dynamic_programming/problems/lc_70/bugs/wrong_base_case_symptom.txt | doc | Human-readable symptom description showing wrong base case causes Climbing Stairs to return incorrect values for n=2 and all larger n. |
| curricula/cp_accelerator/patterns/dynamic_programming/problems/lc_70/build_prompt.txt | asset | Problem statement and implementation instructions presented to the learner for LeetCode 70 Climbing Stairs. |
| curricula/cp_accelerator/patterns/dynamic_programming/problems/lc_70/justify_questions.json | asset | Structured Socratic Q&A with model answers and failure-mode feedback for the Climbing Stairs DP justification phase. |
| curricula/cp_accelerator/patterns/dynamic_programming/problems/lc_70/solution.py | source | Reference solution for LeetCode 70 Climbing Stairs using O(1)-space DP with Fibonacci recurrence; exports `solve` alias for the test runner. |
| curricula/cp_accelerator/patterns/dynamic_programming/problems/lc_70/test_cases.json | test | Example test cases from the LeetCode 70 Climbing Stairs problem statement used by the local validator. |
| curricula/cp_accelerator/patterns/dynamic_programming/problems/lc_70/validator.sh | test | Shell script that imports solution.py and validates the learner's Climbing Stairs implementation. |
| curricula/cp_accelerator/patterns/greedy/problems/lc_435/bugs/sort_by_start.json | config | AST-based bug injection spec that replaces sort-by-end-time with sort-by-start-time in the Non-overlapping Intervals greedy solution. |
| curricula/cp_accelerator/patterns/greedy/problems/lc_435/bugs/sort_by_start_symptom.txt | doc | Human-readable symptom description showing sort-by-start causes the greedy interval algorithm to keep long blocking intervals. |
| curricula/cp_accelerator/patterns/greedy/problems/lc_435/build_prompt.txt | asset | Problem statement and implementation instructions presented to the learner for LeetCode 435 Non-overlapping Intervals. |
| curricula/cp_accelerator/patterns/greedy/problems/lc_435/justify_questions.json | asset | Structured Socratic Q&A with model answers and failure-mode feedback for the greedy interval scheduling justification phase. |
| curricula/cp_accelerator/patterns/greedy/problems/lc_435/solution.py | source | Reference solution for LeetCode 435 Non-overlapping Intervals using greedy sort-by-end-time; exports `solve` alias for the test runner. |
| curricula/cp_accelerator/patterns/greedy/problems/lc_435/test_cases.json | test | Example test cases from the LeetCode 435 Non-overlapping Intervals problem statement used by the local validator. |
| curricula/cp_accelerator/patterns/greedy/problems/lc_435/validator.sh | test | Shell script that imports solution.py and validates the learner's Non-overlapping Intervals implementation. |
| curricula/cp_accelerator/patterns/greedy/problems/lc_452/build_prompt.txt | asset | Problem statement and implementation instructions presented to the learner for LeetCode 452 Minimum Number of Arrows to Burst Balloons. |
| curricula/cp_accelerator/patterns/greedy/problems/lc_452/test_cases.json | test | Example test cases from the LeetCode 452 Minimum Arrows problem statement used by the local validator. |
| curricula/cp_accelerator/patterns/greedy/problems/lc_452/validator.sh | test | Shell script that imports solution.py and validates the learner's Minimum Arrows to Burst Balloons implementation. |
| curricula/cp_accelerator/patterns/hash_table/problems/lc_1/bugs/insert_before_check.json | config | AST-based bug injection spec that replaces complement lookup with current-number lookup in the Two Sum hash table solution. |
| curricula/cp_accelerator/patterns/hash_table/problems/lc_1/bugs/insert_before_check_symptom.txt | doc | Human-readable symptom description for the insert-before-check bug in Two Sum, with step-by-step walkthrough and debugging hint. |
| curricula/cp_accelerator/patterns/hash_table/problems/lc_1/build_prompt.txt | doc | Problem statement and build challenge for LeetCode 1 (Two Sum) with hints, learning resources, and mastery submit instructions. |
| curricula/cp_accelerator/patterns/hash_table/problems/lc_1/justify_questions.json | asset | Justification Q&A set with model answers and failure-mode feedback for evaluating learner understanding of Two Sum and hash table trade-offs. |
| curricula/cp_accelerator/patterns/hash_table/problems/lc_1/solution.py | source | O(n) reference solution for Two Sum using a hash map (seen dict) to find complement pairs in a single pass. |
| curricula/cp_accelerator/patterns/hash_table/problems/lc_1/solution_buggy.py | asset | Empty placeholder (0 bytes) for a generated buggy Two Sum solution; populated by the bug-injection engine at exercise time. |
| curricula/cp_accelerator/patterns/hash_table/problems/lc_1/test_cases.json | asset | Eight JSON test cases (including negatives, zeros, large array) for Two Sum consumed by validator.sh. |
| curricula/cp_accelerator/patterns/hash_table/problems/lc_1/validator.sh | source | Bash CLI script that imports twoSum from solution.py via inline Python and reports pass/fail against test_cases.json. |
| curricula/cp_accelerator/patterns/hash_table/problems/lc_217/build_prompt.txt | doc | Problem statement and build challenge for LeetCode 217 (Contains Duplicate) with hash-table pattern overview and submit instructions. |
| curricula/cp_accelerator/patterns/hash_table/problems/lc_217/test_cases.json | asset | JSON test cases for Contains Duplicate consumed by validator.sh. |
| curricula/cp_accelerator/patterns/hash_table/problems/lc_217/validator.sh | source | Bash CLI script that imports containsDuplicate from solution.py and runs it against test_cases.json. |
| curricula/cp_accelerator/patterns/hash_table/problems/lc_219/build_prompt.txt | doc | Problem statement and build challenge for LeetCode 219 (Contains Duplicate II) with hash-table pattern overview and submit instructions. |
| curricula/cp_accelerator/patterns/hash_table/problems/lc_219/test_cases.json | asset | JSON test cases for Contains Duplicate II consumed by validator.sh. |
| curricula/cp_accelerator/patterns/hash_table/problems/lc_219/validator.sh | source | Bash CLI script that imports containsNearbyDuplicate from solution.py and runs it against test_cases.json. |
| curricula/cp_accelerator/patterns/hash_table/theory/justify_questions.json | asset | Theory-level justification Q&A for hash table patterns, covering advantage over brute force, complexity analysis, and edge cases. |
| curricula/cp_accelerator/patterns/heap_and_priority_queue/problems/lc_215/build_prompt.txt | doc | Problem statement and build challenge for LeetCode 215 (Kth Largest Element in an Array) with submit instructions. |
| curricula/cp_accelerator/patterns/heap_and_priority_queue/problems/lc_215/test_cases.json | asset | JSON test cases for Kth Largest Element in an Array consumed by validator.sh. |
| curricula/cp_accelerator/patterns/heap_and_priority_queue/problems/lc_215/validator.sh | source | Bash CLI script that imports findKthLargest from solution.py and runs it against test_cases.json. |
| curricula/cp_accelerator/patterns/heap_and_priority_queue/problems/lc_703/build_prompt.txt | doc | Problem statement and build challenge for LeetCode 703 (Kth Largest Element in a Stream) with submit instructions. |
| curricula/cp_accelerator/patterns/heap_and_priority_queue/problems/lc_703/test_cases.json | asset | JSON test cases for Kth Largest Element in a Stream consumed by validator.sh. |
| curricula/cp_accelerator/patterns/heap_and_priority_queue/problems/lc_703/validator.sh | source | Bash CLI script that imports the KthLargest class from solution.py and runs it against test_cases.json. |
| curricula/cp_accelerator/patterns/linked_list/problems/lc_203/bugs/skip_consecutive.json | config | AST bug-injection spec (engine v2.1) that replaces != with > in the removeElements filter to cause wrong comparison operator bug. |
| curricula/cp_accelerator/patterns/linked_list/problems/lc_203/bugs/skip_consecutive_symptom.txt | doc | Symptom description for the wrong comparison operator bug in Remove Linked List Elements, with expected vs actual output and debugging guide. |
| curricula/cp_accelerator/patterns/linked_list/problems/lc_203/build_prompt.txt | doc | Problem statement and build challenge for LeetCode 203 (Remove Linked List Elements) with submit instructions. |
| curricula/cp_accelerator/patterns/linked_list/problems/lc_203/justify_questions.json | asset | Justification Q&A set with model answers and failure-mode feedback for evaluating learner understanding of Remove Linked List Elements. |
| curricula/cp_accelerator/patterns/linked_list/problems/lc_203/solution.py | source | O(n) reference solution for Remove Linked List Elements using array representation compatible with the test runner. |
| curricula/cp_accelerator/patterns/linked_list/problems/lc_203/test_cases.json | asset | JSON test cases for Remove Linked List Elements consumed by validator.sh. |
| curricula/cp_accelerator/patterns/linked_list/problems/lc_203/validator.sh | source | Bash CLI script that imports removeElements from solution.py and runs it against test_cases.json. |
| curricula/cp_accelerator/patterns/linked_list/problems/lc_237/build_prompt.txt | doc | Problem statement and build challenge for LeetCode 237 (Delete Node in a Linked List) with submit instructions. |
| curricula/cp_accelerator/patterns/linked_list/problems/lc_237/test_cases.json | asset | JSON test cases for Delete Node in a Linked List consumed by validator.sh. |
| curricula/cp_accelerator/patterns/linked_list/problems/lc_237/validator.sh | source | Bash CLI script that imports deleteNode from solution.py and runs it against test_cases.json. |
| curricula/cp_accelerator/patterns/prefix_sum/problems/lc_1480/build_prompt.txt | doc | Problem statement and build challenge for LeetCode 1480 (Running Sum of 1d Array) with submit instructions. |
| curricula/cp_accelerator/patterns/prefix_sum/problems/lc_1480/test_cases.json | asset | JSON test cases for Running Sum of 1d Array consumed by validator.sh. |
| curricula/cp_accelerator/patterns/prefix_sum/problems/lc_1480/validator.sh | source | Bash CLI script that imports runningSum from solution.py and runs it against test_cases.json. |
| curricula/cp_accelerator/patterns/prefix_sum/problems/lc_303/bugs/off_by_one_prefix.json | config | AST bug-injection spec (engine v2.1) that introduces an off-by-one error in the prefix sum range query boundary. |
| curricula/cp_accelerator/patterns/prefix_sum/problems/lc_303/bugs/off_by_one_prefix_symptom.txt | doc | Symptom description for the off-by-one prefix sum bug in Range Sum Query, with expected vs actual output and debugging guide. |
| curricula/cp_accelerator/patterns/prefix_sum/problems/lc_303/build_prompt.txt | doc | Problem statement and build challenge for LeetCode 303 (Range Sum Query - Immutable) with submit instructions. |
| curricula/cp_accelerator/patterns/prefix_sum/problems/lc_303/justify_questions.json | asset | Justification Q&A set with model answers for evaluating learner understanding of Range Sum Query and prefix sum technique. |
| curricula/cp_accelerator/patterns/prefix_sum/problems/lc_303/solution.py | source | O(n) build / O(1) query reference solution for Range Sum Query using a prefix sum array. |
| curricula/cp_accelerator/patterns/prefix_sum/problems/lc_303/test_cases.json | asset | JSON test cases for Range Sum Query - Immutable consumed by validator.sh. |
| curricula/cp_accelerator/patterns/prefix_sum/problems/lc_303/validator.sh | source | Bash CLI script that imports sumRange from solution.py and runs it against test_cases.json. |
| curricula/cp_accelerator/patterns/segment_tree_and_fenwick_tree/problems/lc_307/build_prompt.txt | doc | Problem statement and build challenge for LeetCode 307 (Range Sum Query - Mutable) with submit instructions. |
| curricula/cp_accelerator/patterns/segment_tree_and_fenwick_tree/problems/lc_307/test_cases.json | asset | JSON test cases for Range Sum Query - Mutable consumed by validator.sh. |
| curricula/cp_accelerator/patterns/segment_tree_and_fenwick_tree/problems/lc_307/validator.sh | source | Bash CLI script that imports the NumArray class from solution.py and runs it against test_cases.json. |
| curricula/cp_accelerator/patterns/sorting/problems/lc_148/build_prompt.txt | doc | Problem statement and build challenge for LeetCode 148 (Sort List) with submit instructions. |
| curricula/cp_accelerator/patterns/sorting/problems/lc_148/test_cases.json | asset | JSON test cases for Sort List consumed by validator.sh. |
| curricula/cp_accelerator/patterns/sorting/problems/lc_148/validator.sh | source | Bash CLI script that imports sortList from solution.py and runs it against test_cases.json. |
| curricula/cp_accelerator/patterns/sorting/problems/lc_912/bugs/incomplete_merge.json | config | AST bug-injection spec (engine v2.1) that deletes the result.extend(right[j:]) statement from the merge function, omitting trailing right-array elements. |
| curricula/cp_accelerator/patterns/sorting/problems/lc_912/bugs/incomplete_merge.patch | doc | Unified diff showing the incomplete_merge mutation against the lc_912 reference solution for audit and review. |
| curricula/cp_accelerator/patterns/sorting/problems/lc_912/bugs/incomplete_merge.py | asset | Empty placeholder (0 bytes) for the generated incomplete-merge buggy solution; populated by the bug-injection engine at exercise time. |
| curricula/cp_accelerator/patterns/sorting/problems/lc_912/bugs/incomplete_merge_symptom.txt | doc | Symptom description for the incomplete merge bug in Sort an Array: output shorter than input with debugging hint about leftover elements. |
| curricula/cp_accelerator/patterns/sorting/problems/lc_912/bugs/missing_base_case.json | config | AST bug-injection spec (engine v2.1) that deletes the base case from sortArray, causing infinite recursion on single-element arrays. |
| curricula/cp_accelerator/patterns/sorting/problems/lc_912/bugs/missing_base_case.py | asset | Empty placeholder (0 bytes) for the generated missing-base-case buggy solution; populated by the bug-injection engine at exercise time. |
| curricula/cp_accelerator/patterns/sorting/problems/lc_912/bugs/missing_base_case_symptom.txt | doc | Symptom description for the missing base case bug in merge sort: RecursionError with debugging guide explaining why base cases are essential. |
| curricula/cp_accelerator/patterns/sorting/problems/lc_912/bugs/off_by_one.py | asset | Empty placeholder (0 bytes) for the generated off-by-one buggy merge sort; populated by the bug-injection engine at exercise time. |
| curricula/cp_accelerator/patterns/sorting/problems/lc_912/build_prompt.txt | doc | Problem statement and build challenge for LeetCode 912 (Sort an Array) with merge-sort pattern overview and submit instructions. |
| curricula/cp_accelerator/patterns/sorting/problems/lc_912/justify_questions.json | asset | Justification Q&A set with model answers for evaluating learner understanding of merge sort invariants and complexity. |
| curricula/cp_accelerator/patterns/sorting/problems/lc_912/solution.py | source | O(n log n) reference solution for Sort an Array using merge sort with a separate merge helper function. |
| curricula/cp_accelerator/patterns/sorting/problems/lc_912/test_cases.json | asset | JSON test cases for Sort an Array consumed by validator.sh. |
| curricula/cp_accelerator/patterns/sorting/problems/lc_912/validator.sh | source | Bash CLI script that imports sortArray from solution.py and runs it against test_cases.json. |
| curricula/cp_accelerator/patterns/sorting/theory/justify_questions.json | asset | Theory-level justification Q&A for sorting patterns, covering merge sort invariant, divide-and-conquer correctness, and complexity. |
| curricula/cp_accelerator/patterns/stack_and_queue/problems/lc_1003/build_prompt.txt | doc | Problem statement and build challenge for LeetCode 1003 (Check If Word Is Valid After Substitutions) with submit instructions. |
| curricula/cp_accelerator/patterns/stack_and_queue/problems/lc_1003/test_cases.json | asset | JSON test cases for Check If Word Is Valid After Substitutions consumed by validator.sh. |
| curricula/cp_accelerator/patterns/stack_and_queue/problems/lc_1003/validator.sh | source | Bash CLI script that imports isValid from solution.py and runs it against test_cases.json for lc_1003. |
| curricula/cp_accelerator/patterns/stack_and_queue/problems/lc_20/bugs/missing_empty_check.json | config | AST bug-injection spec (engine v2.1) that removes the empty-stack guard before stack[-1], causing IndexError on unmatched closing brackets. |
| curricula/cp_accelerator/patterns/stack_and_queue/problems/lc_20/bugs/missing_empty_check_symptom.txt | doc | Symptom description for the missing empty-stack check bug in Valid Parentheses: IndexError on inputs like ')(' with debugging guide. |
| curricula/cp_accelerator/patterns/stack_and_queue/problems/lc_20/build_prompt.txt | doc | Problem statement and build challenge for LeetCode 20 (Valid Parentheses) with stack pattern overview and submit instructions. |
| curricula/cp_accelerator/patterns/stack_and_queue/problems/lc_20/justify_questions.json | asset | Justification Q&A set with model answers for evaluating learner understanding of Valid Parentheses and stack-based matching. |
| curricula/cp_accelerator/patterns/stack_and_queue/problems/lc_20/solution.py | source | O(n) reference solution for Valid Parentheses using a stack to match bracket pairs with closing-bracket lookup. |
| curricula/cp_accelerator/patterns/stack_and_queue/problems/lc_20/test_cases.json | asset | JSON test cases for Valid Parentheses consumed by validator.sh. |
| curricula/cp_accelerator/patterns/stack_and_queue/problems/lc_20/validator.sh | source | Bash CLI script that imports isValid from solution.py and runs it against test_cases.json for lc_20. |
| curricula/cp_accelerator/patterns/traversal/problems/lc_144/build_prompt.txt | doc | Problem statement, constraints, and implementation instructions for LeetCode 144 (Binary Tree Preorder Traversal). |
| curricula/cp_accelerator/patterns/traversal/problems/lc_144/test_cases.json | config | Example test cases (input/expected pairs) used by validator.sh to check LC-144 solutions. |
| curricula/cp_accelerator/patterns/traversal/problems/lc_144/validator.sh | source | Shell script that imports student solution.py and runs it against test_cases.json, reporting pass/fail for LC-144. |
| curricula/cp_accelerator/patterns/traversal/problems/lc_589/build_prompt.txt | doc | Problem statement, constraints, and implementation instructions for LeetCode 589 (N-ary Tree Preorder Traversal). |
| curricula/cp_accelerator/patterns/traversal/problems/lc_589/test_cases.json | config | Example test cases used by validator.sh to check LC-589 N-ary preorder traversal solutions. |
| curricula/cp_accelerator/patterns/traversal/problems/lc_589/validator.sh | source | Shell script that imports student solution.py and runs it against test_cases.json for LC-589. |
| curricula/cp_accelerator/patterns/trie/problems/lc_1804/build_prompt.txt | doc | Problem statement, constraints, and implementation instructions for LeetCode 1804 (Implement Trie II with prefix counts). |
| curricula/cp_accelerator/patterns/trie/problems/lc_1804/test_cases.json | config | Example test cases used by validator.sh to check LC-1804 Trie II solutions. |
| curricula/cp_accelerator/patterns/trie/problems/lc_1804/validator.sh | source | Shell script that imports student solution.py and runs it against test_cases.json for LC-1804. |
| curricula/cp_accelerator/patterns/trie/problems/lc_208/build_prompt.txt | doc | Problem statement, constraints, and implementation instructions for LeetCode 208 (Implement Trie with insert/search/startsWith). |
| curricula/cp_accelerator/patterns/trie/problems/lc_208/test_cases.json | config | Example test cases used by validator.sh to check LC-208 Trie solutions. |
| curricula/cp_accelerator/patterns/trie/problems/lc_208/validator.sh | source | Shell script that imports student solution.py and runs it against test_cases.json for LC-208. |
| curricula/cp_accelerator/patterns/two_pointers/problems/lc_1099/build_prompt.txt | doc | Problem statement, constraints, and implementation instructions for LeetCode 1099 (Two Sum Less Than K). |
| curricula/cp_accelerator/patterns/two_pointers/problems/lc_1099/test_cases.json | config | Example test cases used by validator.sh to check LC-1099 Two Sum Less Than K solutions. |
| curricula/cp_accelerator/patterns/two_pointers/problems/lc_1099/validator.sh | source | Shell script that imports student solution.py and runs it against test_cases.json for LC-1099. |
| curricula/cp_accelerator/patterns/two_pointers/problems/lc_167/bugs/wrong_pointer_move.json | config | AST-based bug injection descriptor that swaps the two-pointer convergence condition (< vs >) in twoSum, making pointers diverge. |
| curricula/cp_accelerator/patterns/two_pointers/problems/lc_167/bugs/wrong_pointer_move_symptom.txt | doc | Describes the observable symptom (empty-list return) and debugging guide for the wrong_pointer_move injected bug. |
| curricula/cp_accelerator/patterns/two_pointers/problems/lc_167/build_prompt.txt | doc | Problem statement, constraints, and implementation instructions for LeetCode 167 (Two Sum II – sorted array). |
| curricula/cp_accelerator/patterns/two_pointers/problems/lc_167/justify_questions.json | config | Conceptual Justify-stage questions (with model answers and failure modes) about the two-pointer technique for LC-167. |
| curricula/cp_accelerator/patterns/two_pointers/problems/lc_167/solution.py | source | Reference implementation of Two Sum II using O(n)/O(1) two-pointer approach; exposes twoSum and solve entry points. |
| curricula/cp_accelerator/patterns/two_pointers/problems/lc_167/test_cases.json | config | Example and edge-case test cases (including negatives, boundaries) for validating LC-167 solutions. |
| curricula/cp_accelerator/patterns/two_pointers/problems/lc_167/validator.sh | source | Shell script that imports student solution.py and runs it against test_cases.json for LC-167. |
| curricula/cp_accelerator/patterns/union_find_disjoint_set_union/problems/lc_547/build_prompt.txt | doc | Problem statement, constraints, and implementation instructions for LeetCode 547 (Number of Provinces via Union-Find). |
| curricula/cp_accelerator/patterns/union_find_disjoint_set_union/problems/lc_547/test_cases.json | config | Example test cases used by validator.sh to check LC-547 Number of Provinces solutions. |
| curricula/cp_accelerator/patterns/union_find_disjoint_set_union/problems/lc_547/validator.sh | source | Shell script that imports student solution.py and runs it against test_cases.json for LC-547. |
| curricula/cp_accelerator/patterns/union_find_disjoint_set_union/problems/lc_684/build_prompt.txt | doc | Problem statement, constraints, and implementation instructions for LeetCode 684 (Redundant Connection via Union-Find). |
| curricula/cp_accelerator/patterns/union_find_disjoint_set_union/problems/lc_684/test_cases.json | config | Example test cases used by validator.sh to check LC-684 Redundant Connection solutions. |
| curricula/cp_accelerator/patterns/union_find_disjoint_set_union/problems/lc_684/validator.sh | source | Shell script that imports student solution.py and runs it against test_cases.json for LC-684. |
| curricula/cs336_a1/README.md | doc | Attribution, curriculum overview, module structure, usage commands, and educational philosophy for the CS336 A1 curriculum. |
| curricula/cs336_a1/manifest.json | config | Module registry listing all 22 CS336 A1 modules in dependency order with paths, types, and performance baselines. |
| curricula/cs336_a1/modules/adamw/bugs/missing_bias_correction.json | config | AST-based 4-pass bug injection descriptor removing bias-correction terms from the AdamW optimizer step. |
| curricula/cs336_a1/modules/adamw/bugs/missing_bias_correction.patch | config | Git-unified-diff patch showing the missing_bias_correction bug as applied to cs336_basics/optimizer.py. |
| curricula/cs336_a1/modules/adamw/build_prompt.txt | doc | Build challenge instructions and mathematical specification for implementing the AdamW optimizer from scratch. |
| curricula/cs336_a1/modules/adamw/justify_questions.json | config | Conceptual Justify-stage questions (with model answers and failure modes) about AdamW optimizer internals. |
| curricula/cs336_a1/modules/adamw/validator.sh | source | Shell script running pytest for the AdamW optimizer implementation inside a shadow worktree. |
| curricula/cs336_a1/modules/attention/bugs/missing_scale.json | config | AST-based bug injection descriptor that removes the 1/sqrt(d_k) scaling factor from scaled dot-product attention. |
| curricula/cs336_a1/modules/attention/bugs/missing_scale.patch | config | Git-unified-diff patch showing the missing_scale bug as applied to the attention implementation. |
| curricula/cs336_a1/modules/attention/build_prompt.txt | doc | Build challenge instructions and mathematical specification for implementing scaled dot-product attention. |
| curricula/cs336_a1/modules/attention/justify_questions.json | config | Conceptual Justify-stage questions (with model answers and failure modes) about the attention mechanism. |
| curricula/cs336_a1/modules/attention/validator.sh | source | Shell script running pytest for the scaled dot-product attention implementation in a shadow worktree. |
| curricula/cs336_a1/modules/bpe_tokenizer/bugs/wrong_merge_order.json | config | AST-based bug injection descriptor reversing BPE merge insertion order (prepend instead of append). |
| curricula/cs336_a1/modules/bpe_tokenizer/bugs/wrong_merge_order.patch | config | Git-unified-diff patch showing the wrong_merge_order bug as applied to the BPE tokenizer implementation. |
| curricula/cs336_a1/modules/bpe_tokenizer/bugs/wrong_merge_order_draft.json | config | Draft version of the wrong_merge_order bug descriptor with more verbose description field. |
| curricula/cs336_a1/modules/bpe_tokenizer/build_prompt.txt | doc | Build challenge instructions and algorithmic specification for implementing BPE tokenizer training. |
| curricula/cs336_a1/modules/bpe_tokenizer/justify_questions.json | config | Conceptual Justify-stage questions (with model answers and failure modes) about BPE tokenization. |
| curricula/cs336_a1/modules/bpe_tokenizer/validator.sh | source | Shell script running pytest for the BPE tokenizer training implementation in a shadow worktree. |
| curricula/cs336_a1/modules/checkpointing/bugs/missing_optimizer_state.json | config | AST-based bug injection descriptor that omits optimizer state from the checkpoint save call. |
| curricula/cs336_a1/modules/checkpointing/bugs/missing_optimizer_state.patch | config | Git-unified-diff patch showing the missing_optimizer_state bug as applied to the checkpointing code. |
| curricula/cs336_a1/modules/checkpointing/bugs/missing_optimizer_state_draft.json | config | Draft version of the missing_optimizer_state bug descriptor. |
| curricula/cs336_a1/modules/checkpointing/build_prompt.txt | doc | Build challenge instructions and specification for implementing model checkpoint save and load. |
| curricula/cs336_a1/modules/checkpointing/justify_questions.json | config | Conceptual Justify-stage questions (with model answers and failure modes) about model checkpointing. |
| curricula/cs336_a1/modules/checkpointing/validator.sh | source | Shell script running pytest for the checkpointing save/load implementation in a shadow worktree. |
| curricula/cs336_a1/modules/cosine_schedule/bugs/wrong_cosine_range.json | config | AST-based bug injection descriptor introducing the wrong cosine oscillation range in the LR schedule. |
| curricula/cs336_a1/modules/cosine_schedule/bugs/wrong_cosine_range.patch | config | Git-unified-diff patch showing the wrong_cosine_range bug as applied to the scheduler code. |
| curricula/cs336_a1/modules/cosine_schedule/bugs/wrong_cosine_range_draft.json | config | Draft version of the wrong_cosine_range bug descriptor. |
| curricula/cs336_a1/modules/cosine_schedule/build_prompt.txt | doc | Build challenge instructions and mathematical specification for implementing cosine LR schedule with linear warmup. |
| curricula/cs336_a1/modules/cosine_schedule/justify_questions.json | config | Conceptual Justify-stage questions (with model answers and failure modes) about cosine LR scheduling. |
| curricula/cs336_a1/modules/cosine_schedule/validator.sh | source | Shell script running pytest for the cosine LR schedule implementation in a shadow worktree. |
| curricula/cs336_a1/modules/cross_entropy/bugs/no_logsumexp.json | config | AST-based bug injection descriptor removing the logsumexp numerical stability trick from cross-entropy loss. |
| curricula/cs336_a1/modules/cross_entropy/bugs/no_logsumexp.patch | config | Git-unified-diff patch showing the no_logsumexp bug as applied to the loss function code. |
| curricula/cs336_a1/modules/cross_entropy/bugs/no_logsumexp_draft.json | config | Draft version of the no_logsumexp bug descriptor. |
| curricula/cs336_a1/modules/cross_entropy/bugs/no_logsumexp_symptom.txt | doc | Describes the NaN/inf symptom and debugging guide for the missing logsumexp numerical stability bug. |
| curricula/cs336_a1/modules/cross_entropy/build_prompt.txt | doc | Build challenge instructions and mathematical specification for implementing numerically stable cross-entropy loss. |
| curricula/cs336_a1/modules/cross_entropy/justify_questions.json | config | Conceptual Justify-stage questions (with model answers and failure modes) about numerically stable cross-entropy. |
| curricula/cs336_a1/modules/cross_entropy/validator.sh | source | Shell script running pytest for the numerically stable cross-entropy implementation in a shadow worktree. |
| curricula/cs336_a1/modules/data_loader/bugs/wrong_sampling_range.json | config | AST-based bug injection descriptor using an off-by-one high bound in randint, causing out-of-bounds token sampling. |
| curricula/cs336_a1/modules/data_loader/bugs/wrong_sampling_range.patch | config | Git-unified-diff patch showing the wrong_sampling_range bug as applied to the data loader code. |
| curricula/cs336_a1/modules/data_loader/bugs/wrong_sampling_range_draft.json | config | Draft version of the wrong_sampling_range bug descriptor. |
| curricula/cs336_a1/modules/data_loader/build_prompt.txt | doc | Build challenge instructions and specification for implementing the language model get_batch data loader. |
| curricula/cs336_a1/modules/data_loader/justify_questions.json | config | Conceptual Justify-stage questions (with model answers and failure modes) about LM data loading and sampling. |
| curricula/cs336_a1/modules/data_loader/validator.sh | source | Shell script that copies layers and runs pytest tests/test_training.py::test_get_batch in a shadow worktree to validate the data loader implementation. |
| curricula/cs336_a1/modules/embedding/bugs/wrong_dimension_order.json | config | Production AST bug-injection spec (engine_version 2.1) that swaps num_embeddings and embedding_dim in the nn.Embedding constructor call. |
| curricula/cs336_a1/modules/embedding/bugs/wrong_dimension_order.patch | asset | Unified diff showing the dimension-swap bug as applied to cs336_basics/layers.py for the Embedding module. |
| curricula/cs336_a1/modules/embedding/bugs/wrong_dimension_order_draft.json | config | Draft (v2.0, multi-pass) AST bug-injection spec for swapping embedding dimensions; predecessor to the production wrong_dimension_order.json. |
| curricula/cs336_a1/modules/embedding/build_prompt.txt | doc | Student-facing instructional guide covering embedding theory and specifying how to implement the Embedding class from scratch using nn.Parameter. |
| curricula/cs336_a1/modules/embedding/justify_questions.json | asset | Five conceptual Q&A pairs (with model answers and required concepts) for post-build assessment of the token embedding module. |
| curricula/cs336_a1/modules/embedding/validator.sh | source | Shell script that copies layers.py and runs pytest tests/test_model.py::test_embedding in a shadow worktree to validate the Embedding implementation. |
| curricula/cs336_a1/modules/gradient_clipping/bugs/per_parameter_clipping.json | config | Production AST bug-injection spec (engine_version 2.1) that replaces global-norm gradient clipping with per-parameter clipping. |
| curricula/cs336_a1/modules/gradient_clipping/bugs/per_parameter_clipping.patch | asset | Unified diff showing the per-parameter clipping bug applied to the clip_gradients_by_global_norm function. |
| curricula/cs336_a1/modules/gradient_clipping/bugs/per_parameter_clipping_draft.json | config | Draft AST bug-injection spec (LLM-generated, tier: complex) for the per-parameter clipping bug; predecessor to the production spec. |
| curricula/cs336_a1/modules/gradient_clipping/bugs/per_parameter_clipping_symptom.txt | doc | Student-facing description of gradient direction distortion symptoms caused by the per-parameter clipping bug, with debugging hints and correct algorithm. |
| curricula/cs336_a1/modules/gradient_clipping/build_prompt.txt | doc | Student-facing instructional guide for implementing global L2 gradient norm clipping, including mathematical derivation and implementation steps. |
| curricula/cs336_a1/modules/gradient_clipping/justify_questions.json | asset | Conceptual Q&A pairs with model answers for post-build assessment of the gradient clipping module. |
| curricula/cs336_a1/modules/gradient_clipping/validator.sh | source | Shell script that runs pytest tests/test_nn_utils.py::test_gradient_clipping in a shadow worktree to validate the gradient clipping implementation. |
| curricula/cs336_a1/modules/linear/bugs/missing_transpose.json | config | Production AST bug-injection spec (engine_version 2.1) that removes the .t() weight transpose in the Linear layer forward pass. |
| curricula/cs336_a1/modules/linear/bugs/missing_transpose.patch | asset | Unified diff showing the missing weight transpose bug as applied to the Linear layer in cs336_basics/layers.py. |
| curricula/cs336_a1/modules/linear/bugs/missing_transpose_draft.json | config | First draft AST bug-injection spec for the missing weight transpose bug in the Linear layer forward pass. |
| curricula/cs336_a1/modules/linear/bugs/missing_transpose_draft_v2.json | config | Second draft (v2.1, author: auto_fixed) AST spec for the missing transpose bug; uses replace_value_with instead of replace_with. |
| curricula/cs336_a1/modules/linear/build_prompt.txt | doc | Student-facing instructional guide for implementing the Linear (fully connected) layer with optional bias from scratch. |
| curricula/cs336_a1/modules/linear/justify_questions.json | asset | Conceptual Q&A pairs with model answers for post-build assessment of the linear layer module. |
| curricula/cs336_a1/modules/linear/validator.sh | source | Shell script that runs pytest tests/test_model.py::test_linear in a shadow worktree to validate the Linear layer implementation. |
| curricula/cs336_a1/modules/multihead_attention/bugs/missing_transpose_back.json | config | Production AST bug-injection spec for the missing transpose-back operation in multihead self-attention output reshaping. |
| curricula/cs336_a1/modules/multihead_attention/bugs/missing_transpose_back.patch | asset | Unified diff showing the missing transpose_back bug applied to the multihead attention implementation in cs336_basics/layers.py. |
| curricula/cs336_a1/modules/multihead_attention/bugs/missing_transpose_back_draft.json | config | Draft AST bug-injection spec for the missing transpose_back bug in multihead attention; predecessor to the production spec. |
| curricula/cs336_a1/modules/multihead_attention/build_prompt.txt | doc | Student-facing instructional guide for implementing multi-head self-attention with RoPE positional encoding. |
| curricula/cs336_a1/modules/multihead_attention/justify_questions.json | asset | Conceptual Q&A pairs with model answers for post-build assessment of the multihead attention module. |
| curricula/cs336_a1/modules/multihead_attention/validator.sh | source | Shell script that runs pytest tests/test_model.py::test_multihead_self_attention_with_rope in a shadow worktree to validate multihead attention. |
| curricula/cs336_a1/modules/rmsnorm/bugs/missing_keepdim.json | config | Production AST bug-injection spec that removes keepdim=True from the mean computation in RMSNorm, causing a broadcasting shape error. |
| curricula/cs336_a1/modules/rmsnorm/bugs/missing_keepdim.patch | asset | Unified diff showing the missing keepdim=True bug applied to the RMSNorm layer in cs336_basics/layers.py. |
| curricula/cs336_a1/modules/rmsnorm/build_prompt.txt | doc | Student-facing instructional guide for implementing the RMSNorm normalization layer used in modern LLMs. |
| curricula/cs336_a1/modules/rmsnorm/justify_questions.json | asset | Conceptual Q&A pairs with model answers for post-build assessment of the RMSNorm module. |
| curricula/cs336_a1/modules/rmsnorm/validator.sh | source | Shell script that runs pytest tests/test_model.py::test_rmsnorm in a shadow worktree to validate the RMSNorm implementation. |
| curricula/cs336_a1/modules/rope/bugs/wrong_rotation.json | config | Production AST bug-injection spec for injecting a wrong rotation bug into the RoPE positional encoding implementation. |
| curricula/cs336_a1/modules/rope/bugs/wrong_rotation.patch | asset | Unified diff showing the wrong rotation bug applied to the RoPE positional encoding in cs336_basics/layers.py. |
| curricula/cs336_a1/modules/rope/bugs/wrong_rotation_draft.json | config | Draft AST bug-injection spec for the wrong rotation bug in RoPE; predecessor to the production spec. |
| curricula/cs336_a1/modules/rope/build_prompt.txt | doc | Student-facing instructional guide for implementing Rotary Positional Encoding (RoPE) for Transformer attention. |
| curricula/cs336_a1/modules/rope/justify_questions.json | asset | Conceptual Q&A pairs with model answers for post-build assessment of the RoPE module. |
| curricula/cs336_a1/modules/rope/validator.sh | source | Shell script that runs pytest tests/test_model.py::test_rope in a shadow worktree to validate the RoPE implementation. |
| curricula/cs336_a1/modules/silu/bugs/missing_multiply.json | config | Production AST bug-injection spec that removes the element-wise multiply in the SiLU activation, reducing it to identity or sigmoid only. |
| curricula/cs336_a1/modules/silu/bugs/missing_multiply.patch | asset | Unified diff showing the missing multiply bug applied to the SiLU activation function in cs336_basics/layers.py. |
| curricula/cs336_a1/modules/silu/build_prompt.txt | doc | Student-facing instructional guide for implementing the SiLU (Swish) activation function x * sigmoid(x). |
| curricula/cs336_a1/modules/silu/justify_questions.json | asset | Conceptual Q&A pairs with model answers for post-build assessment of the SiLU module. |
| curricula/cs336_a1/modules/silu/validator.sh | source | Shell script that runs pytest tests/test_model.py::test_silu_matches_pytorch in a shadow worktree to validate the SiLU implementation. |
| curricula/cs336_a1/modules/softmax/bugs/no_subtract_max.json | config | Production AST bug-injection spec (two-pass) that removes the subtract-max numerical stability trick from the softmax implementation. |
| curricula/cs336_a1/modules/softmax/bugs/no_subtract_max.patch | asset | Unified diff showing the no-subtract-max bug applied to the softmax function, exposing it to numerical overflow. |
| curricula/cs336_a1/modules/softmax/bugs/no_subtract_max_symptom.txt | doc | Student-facing description of NaN overflow symptoms from the missing subtract-max trick, with failing test case and fix guidance. |
| curricula/cs336_a1/modules/softmax/bugs/no_subtract_max_v2.json | config | Second version of the AST spec for removing subtract-max from softmax, using an alternative two-pass find_and_track then find_and_replace pattern. |
| curricula/cs336_a1/modules/softmax/bugs/no_subtract_max_v2_symptom.txt | doc | Student-facing symptom description for the v2 no-subtract-max softmax bug variant, matching the v2 injection spec. |
| curricula/cs336_a1/modules/softmax/build_prompt.txt | doc | Student-facing instructional guide for implementing numerically stable softmax using the subtract-max trick. |
| curricula/cs336_a1/modules/softmax/justify_questions.json | asset | Conceptual Q&A pairs with model answers for post-build assessment of the numerically stable softmax module. |
| curricula/cs336_a1/modules/softmax/validator.sh | source | Shell script that runs pytest tests/test_nn_utils.py::test_softmax_matches_pytorch in a shadow worktree to validate the softmax implementation. |
| curricula/cs336_a1/modules/swiglu/bugs/missing_gate.json | config | Production AST bug-injection spec that removes the gate computation from the SwiGLU gated activation forward pass. |
| curricula/cs336_a1/modules/swiglu/bugs/missing_gate.patch | asset | Unified diff showing the missing gate bug applied to the SwiGLU activation in cs336_basics/layers.py. |
| curricula/cs336_a1/modules/swiglu/bugs/missing_gate_draft.json | config | First draft AST bug-injection spec for the missing gate bug in SwiGLU; predecessor to the production spec. |
| curricula/cs336_a1/modules/swiglu/bugs/missing_gate_draft_v2.json | config | Second draft (v2.1) AST bug-injection spec for the missing gate bug in SwiGLU; intermediate version between draft and production. |
| curricula/cs336_a1/modules/swiglu/build_prompt.txt | doc | Student-facing instructional guide for implementing the SwiGLU gated activation function used in modern LLM feed-forward blocks. |
| curricula/cs336_a1/modules/swiglu/justify_questions.json | asset | Conceptual Q&A pairs with model answers for post-build assessment of the SwiGLU module. |
| curricula/cs336_a1/modules/swiglu/validator.sh | source | Shell script that runs pytest tests/test_model.py::test_swiglu in a shadow worktree to validate the SwiGLU implementation. |
| curricula/cs336_a1/modules/text_generation/bugs/temperature_after_softmax.json | config | Production AST bug-injection spec that misplaces temperature scaling to after softmax instead of before it in text generation. |
| curricula/cs336_a1/modules/text_generation/bugs/temperature_after_softmax.patch | asset | Unified diff showing the temperature_after_softmax bug applied to the text generation function. |
| curricula/cs336_a1/modules/text_generation/bugs/temperature_after_softmax_draft.json | config | Draft AST bug-injection spec for the temperature_after_softmax bug; predecessor to the production spec. |
| curricula/cs336_a1/modules/text_generation/build_prompt.txt | doc | Student-facing instructional guide for implementing temperature-based autoregressive text generation. |
| curricula/cs336_a1/modules/text_generation/justify_questions.json | asset | Conceptual Q&A pairs with model answers for post-build assessment of the text generation module. |
| curricula/cs336_a1/modules/text_generation/validator.sh | source | Shell script that runs pytest tests/test_generation.py::test_generate in a shadow worktree to validate the text generation implementation. |
| curricula/cs336_a1/modules/tokenizer_class/bugs/wrong_merge_order.json | config | Production AST bug-injection spec for the wrong merge order bug in the BPE tokenizer class encode/merge logic. |
| curricula/cs336_a1/modules/tokenizer_class/bugs/wrong_merge_order.patch | asset | Unified diff showing the wrong merge order bug applied to the BPE Tokenizer class in cs336_basics/. |
| curricula/cs336_a1/modules/tokenizer_class/bugs/wrong_merge_order_draft.json | config | Draft AST bug-injection spec for the wrong merge order bug in the tokenizer class; predecessor to the production spec. |
| curricula/cs336_a1/modules/tokenizer_class/build_prompt.txt | doc | Student-facing instructional guide for implementing the BPE Tokenizer class with encode, decode, and train_from_iterator methods. |
| curricula/cs336_a1/modules/tokenizer_class/justify_questions.json | asset | Conceptual Q&A pairs with model answers for post-build assessment of the BPE tokenizer class module. |
| curricula/cs336_a1/modules/tokenizer_class/validator.sh | source | Shell script that runs pytest tests/test_tokenizer.py::test_tokenizer_class in a shadow worktree to validate the Tokenizer class implementation. |
| curricula/cs336_a1/modules/training_loop/bugs/missing_zero_grad.json | config | Finalized AST bug-injection spec that deletes the optimizer.zero_grad() call in the training loop (engine_version 2.1). |
| curricula/cs336_a1/modules/training_loop/bugs/missing_zero_grad.patch | asset | Unified diff that removes optimizer.zero_grad() and adds an explanatory bug comment in cs336_basics/training.py. |
| curricula/cs336_a1/modules/training_loop/bugs/missing_zero_grad_draft.json | config | Draft v1 AST bug-injection spec (pass_ key, extra null fields) for deleting the zero_grad call in training_loop. |
| curricula/cs336_a1/modules/training_loop/bugs/missing_zero_grad_draft_v2.json | config | Draft v2 AST bug-injection spec (auto_fixed metadata, engine_version 2.1) for deleting zero_grad in train_step. |
| curricula/cs336_a1/modules/training_loop/build_prompt.txt | doc | Student-facing instructional prompt explaining the full training loop (forward, backward, clip, step) for cs336_a1. |
| curricula/cs336_a1/modules/training_loop/justify_questions.json | config | Five Q&A assessment items with model answers testing conceptual understanding of the PyTorch training loop. |
| curricula/cs336_a1/modules/training_loop/validator.sh | test | Shell validator that copies training.py to a shadow worktree and runs pytest tests/test_training.py::test_train_loop. |
| curricula/cs336_a1/modules/transformer_block/bugs/missing_residual.json | config | Finalized AST bug-injection spec that replaces x = x + attn_out with x = attn_out to remove the residual connection. |
| curricula/cs336_a1/modules/transformer_block/bugs/missing_residual.patch | asset | Unified diff that drops the residual addition in TransformerBlock.forward() and adds an explanatory bug comment. |
| curricula/cs336_a1/modules/transformer_block/bugs/missing_residual_draft.json | config | Draft v1 AST bug-injection spec (pass_ key, extra null fields) for removing the residual connection in transformer_block. |
| curricula/cs336_a1/modules/transformer_block/bugs/missing_residual_draft_v2.json | config | Draft v2 AST bug-injection spec (auto_fixed metadata) for replacing x = x + attn_out with x = attn_out. |
| curricula/cs336_a1/modules/transformer_block/build_prompt.txt | doc | Student-facing instructional prompt for implementing a complete pre-norm Transformer block with RMSNorm, attention, and SwiGLU. |
| curricula/cs336_a1/modules/transformer_block/justify_questions.json | config | Q&A assessment items with model answers testing understanding of residual connections, pre-norm vs post-norm, and attention. |
| curricula/cs336_a1/modules/transformer_block/validator.sh | test | Shell validator that copies layers.py to a shadow worktree and runs pytest tests/test_model.py::test_transformer_block. |
| curricula/cs336_a1/modules/transformer_lm/bugs/missing_final_norm.json | config | Finalized AST bug-injection spec that deletes the ln_final RMSNorm application before the LM head. |
| curricula/cs336_a1/modules/transformer_lm/bugs/missing_final_norm.patch | asset | Unified diff that removes the final RMSNorm block in transformer_lm and replaces it with an explanatory bug comment. |
| curricula/cs336_a1/modules/transformer_lm/bugs/missing_final_norm_draft.json | config | Misidentified draft AST spec (id: silu-missing-multiply) that removes x*sigmoid(x) multiplication; likely a wrong-module draft. |
| curricula/cs336_a1/modules/transformer_lm/build_prompt.txt | doc | Student-facing instructional prompt for assembling the full TransformerLM with embedding, stacked blocks, and final norm. |
| curricula/cs336_a1/modules/transformer_lm/justify_questions.json | config | Q&A assessment items with model answers on autoregressive target shifting, weight tying, and final norm necessity. |
| curricula/cs336_a1/modules/transformer_lm/validator.sh | test | Shell validator that copies layers.py to a shadow worktree and runs pytest tests/test_model.py::test_transformer_lm. |
| curricula/cs336_a1/modules/unicode/README.md | doc | Explains that the unicode module is theory-only (justify stage only), covering UTF-8 encoding, normalization, and grapheme clusters. |
| curricula/cs336_a1/modules/unicode/justify_questions.json | config | Five comprehensive Q&A items with model answers covering UTF-8 variable-length encoding, normalization, and grapheme clusters. |
| curricula/dummy_hello_world/manifest.json | config | Curriculum manifest for dummy_hello_world defining the single hello_world module with baseline_perf_seconds=0.001. |
| curricula/dummy_hello_world/modules/hello_world/bugs/typo.patch | asset | Unified diff that introduces a typo ('Enginne') in greet()'s return string to demonstrate the Harden stage. |
| curricula/dummy_hello_world/modules/hello_world/bugs/typo_symptom.txt | doc | Student-facing description of the typo bug symptom and instructions to fix the spelling in hello_world.py. |
| curricula/dummy_hello_world/modules/hello_world/build_prompt.txt | doc | Student-facing build challenge to create a hello_world.py function returning 'Hello, Mastery Engine!'. |
| curricula/dummy_hello_world/modules/hello_world/justify_questions.json | config | Single justify Q&A item asking why the implementation uses a function rather than a direct print statement. |
| curricula/dummy_hello_world/modules/hello_world/validator.sh | test | Shell validator that checks for the existence of hello_world.py in the workspace and exits 0 if found. |
| curricula/job_prep_data_annotation/README.md | doc | Overview of the job_prep_data_annotation curriculum: three modules teaching HTTP, HTML parsing, and 2D grids for DataAnnotation assessments. |
| curricula/job_prep_data_annotation/manifest.json | config | Curriculum manifest defining http_transport, data_parsing_extraction, and grid_visualization modules with dependencies and metadata. |
| curricula/job_prep_data_annotation/modules/data_parsing_extraction/bugs/fragile_split.patch | asset | Unified diff replacing BeautifulSoup+regex coordinate parser with a brittle split()-based parser that breaks on whitespace. |
| curricula/job_prep_data_annotation/modules/data_parsing_extraction/build_prompt.txt | doc | Student-facing build challenge to implement extract_coordinates() using BeautifulSoup and regex for robust HTML parsing. |
| curricula/job_prep_data_annotation/modules/data_parsing_extraction/justify_questions.json | config | Q&A assessment items on why split() is brittle for HTML parsing and how regex handles whitespace variations. |
| curricula/job_prep_data_annotation/modules/data_parsing_extraction/validator.sh | test | Shell validator with embedded Python test suite for extract_coordinates() covering consistent/inconsistent formats and performance. |
| curricula/job_prep_data_annotation/modules/grid_visualization/bugs/reference_copying.patch | asset | Unified diff replacing the list-comprehension 2D grid init with aliased [[' ']*width]*height to introduce the reference-copying bug. |
| curricula/job_prep_data_annotation/modules/grid_visualization/build_prompt.txt | doc | Student-facing build challenge to implement render_grid() converting sparse coordinates to a dense 2D list without aliasing. |
| curricula/job_prep_data_annotation/modules/grid_visualization/justify_questions.json | config | Q&A assessment items on Python's shallow vs deep copy semantics and why [[' ']*w]*h creates aliased rows. |
| curricula/job_prep_data_annotation/modules/grid_visualization/validator.sh | test | Shell validator with embedded Python test suite for render_grid() covering sparse grids, reference-copying detection, and performance. |
| curricula/job_prep_data_annotation/modules/http_transport/bugs/open_trap.patch | asset | Unified diff replacing requests.get() with open() in fetch_document() to simulate the file-vs-network I/O confusion bug. |
| curricula/job_prep_data_annotation/modules/http_transport/build_prompt.txt | doc | Student-facing build challenge to implement fetch_document() using requests.get() with proper status-code error handling. |
| curricula/job_prep_data_annotation/modules/http_transport/justify_questions.json | config | Q&A assessment items on why open() fails for URLs and the fundamental distinction between file I/O and network I/O. |
| curricula/job_prep_data_annotation/modules/http_transport/validator.sh | test | Shell validator with embedded Python test suite for fetch_document() testing HTTP GET, error handling, and status codes. |
| curricula/python_for_cp/manifest.json | config | Curriculum manifest for python_for_cp defining pythonic_structures, concise_logic, and std_lib_augmentation modules. |
| curricula/python_for_cp/modules/std_lib_augmentation/bugs/list_pop_performance.patch | asset | Unified diff replacing deque.popleft() with list.pop(0) in BFS to introduce O(n) per-operation performance regression. |
| curricula/python_for_cp/modules/std_lib_augmentation/bugs/missing_visited_set.patch | asset | Unified diff removing the visited set from Dijkstra, causing nodes to be processed multiple times in the heap. |
| curricula/python_for_cp/modules/std_lib_augmentation/build_prompt.txt | doc | Student-facing build challenge to implement BFS with deque, Dijkstra with heapq, and range-count with bisect. |
| curricula/python_for_cp/modules/std_lib_augmentation/justify_questions.json | config | Q&A assessment items on deque vs list complexity, Dijkstra visited-set necessity, and bisect binary search use cases. |
| curricula/python_for_cp/modules/std_lib_augmentation/validator.sh | test | Shell validator with embedded Python test suite for shortest_path_bfs, dijkstra_shortest_path, and count_in_range functions. |
| docs/INDEX.md | doc | Navigation index listing all current, architecture, and internal Mastery Engine documentation with audience labels. |
| docs/README.md | doc | Top-level documentation overview describing the three engineering pillars and directing users/contributors to the right sub-docs. |
| docs/STRANGER_TEST_RESULTS.md | doc | End-to-end verification report from a clean-slate README Quick Start test on Nov 19 2025, finding and fixing 3 critical bugs. |
| docs/architecture/AI_CODEBASE_DECONSTRUCTION.md | doc | Design analysis (blueprint, not shipped) for applying the Mastery Engine to force comprehension of AI-generated codebases. |
| docs/architecture/MASTERY_ENGINE.md | doc | Comprehensive v5.0 technical blueprint of the Mastery Engine's Build-Justify-Harden pedagogy and layered architecture. |
| docs/architecture/REPO_ANALYSIS.md | doc | Auto-generated repository analysis for CS336 Assignment 1, covering structure, tests, and implementation contracts. |
| docs/internal/CLEANUP_SUMMARY.md | doc | Session log summarising the Nov 17 2025 docs cleanup that reduced root .md files from 71 to 2 and added clear hierarchy. |
| docs/internal/CP_ACCELERATOR_QUICKSTART.md | doc | Quick-start guide and rationale for the CP Accelerator canonical-source-of-truth architecture using expert-curated JSON. |
| docs/internal/CP_SOURCE_VERIFICATION.md | doc | Verification report documenting the fix of placeholder URLs and titles in canonical_curriculum.json with real parsed data. |
| docs/internal/CRITICAL_REVIEW_RESPONSE.md | doc | Response to pre-deployment critical review documenting mitigations for BeautifulSoup fluency gap and one other risk. |
| docs/internal/DOCS_CLEANUP_2025-11-18.md | doc | Change log for Nov 18 2025 reorganisation of 12 scattered docs into two_sum_qa/ and module_generation/ subdirectories. |
| docs/internal/ENGINE_CRITICAL_FIXES_2025-11-18.md | doc | Post-mortem and fix log for three critical engine bugs (wrong test cases, path fragility, init idempotency) found in session logs. |
| docs/internal/PHASE_8_BATCH_GENERATION_COMPLETE.md | doc | Completion report for Phase 8 breadth-first content population populating 2 problems per pattern across the 959-problem CP taxonomy. |
| docs/internal/PUBLIC_RELEASE_COMPLETE.md | doc | Release engineering completion report (commit d694ff3) transforming the repo from course assignment to public portfolio piece. |
| docs/internal/PYTHON_CURRICULA_IMPLEMENTATION.md | doc | Implementation report for two new linear Python curricula (job_prep_data_annotation and one other) targeting DataAnnotation skill gaps. |
| docs/internal/README.md | doc | Index and purpose statement for the internal/ directory, noting these are journey artifacts not required for product use. |
| docs/internal/REAL_CLI_TRANSFORMATION.md | doc | Fix log addressing three issues that made the Mastery Engine feel like a script rather than a real CLI tool like git or npm. |
| docs/internal/archive/README.md | doc | Overview and navigation guide for the historical archive directory, directing readers to current docs for up-to-date information. |
| docs/internal/archive/deprecated/BATCH_MIGRATION_GUIDE.md | doc | Deprecated execution guide for batch migration of legacy .patch bug files to v2.1 JSON format with pre-flight checklist. |
| docs/internal/archive/deprecated/EXPERIMENT_MODULE_DESIGN.md | doc | Deprecated design document for an experimental-investigation module type extending BJH with scientific hypothesis and ablation structure. |
| docs/internal/archive/deprecated/JUSTIFY_ONLY_MODULE_DESIGN.md | doc | Deprecated design document for a theory-only justify module type assessing conceptual understanding without requiring implementation. |
| docs/internal/archive/sessions/2025-11-08_systematic_improvements/SYSTEMATIC_FIXING_SESSION.md | doc | Session report on systematic analysis of multi-attempt patterns to diagnose and eliminate first-attempt failure root causes. |
| docs/internal/archive/sessions/2025-11-08_systematic_improvements/SYSTEMATIC_IMPROVEMENT_FINAL.md | doc | Final ~4.5-hour session summary confirming all 4 training examples work correctly and next bottleneck (patch extraction) identified. |
| docs/internal/archive/sessions/2025-11-08_systematic_improvements/SYSTEMATIC_IMPROVEMENT_SESSION.md | doc | Progress report for Session 1 adding regression checks, manual LLM analysis, and permanent evaluation script improvements. |
| docs/internal/archive/sessions/2025-11-08_systematic_improvements/SYSTEMATIC_IMPROVEMENT_SESSION_2.md | doc | Completion report for Session 2 using manual analysis to surface issues that statistics missed, diagnosing P0/P1 bottlenecks. |
| docs/internal/archive/sessions/2025-11-08_systematic_improvements/SYSTEMATIC_SESSION_FINAL.md | doc | Complete ~7-hour systematic session analysis confirming 4/4 training data correctness after evidence-based fixes. |
| docs/internal/archive/sessions/2025-11-09_verification/FINAL_VERIFICATION_SUMMARY.md | doc | Final pre-launch verification summary declaring Mastery Engine v1.0 production-ready with 197 passing automated tests. |
| docs/internal/archive/sessions/2025-11-09_verification/LAYER2_E2E_SUCCESS.md | doc | E2E test fix completion report confirming the full BJH loop test passes in 16.51s after shadow worktree symlink fix. |
| docs/internal/archive/sessions/2025-11-09_verification/LAYER4_UAT_EXECUTION_GUIDE.md | doc | Layer 4 UAT execution guide for the Student Zero Gauntlet clean-slate test (later self-invalidated due to methodology flaws). |
| docs/internal/archive/sessions/2025-11-09_verification/LAYER4_UAT_FINDINGS.md | doc | Invalidated UAT findings report: tester copied from developer mode instead of implementing stubs, rendering all results invalid. |
| docs/internal/archive/sessions/2025-11-09_verification/REAL_STUDENT_UAT_MODULE1.md | doc | Real student UAT report for Module 1 (softmax) build stage using a genuine clean-slate setup with rm of all prior state. |
| docs/internal/archive/sessions/2025-11-09_verification/VERIFICATION_PROTOCOL_FINAL_STATUS.md | doc | Verification Protocol v3.0 pre-launch status report showing 93% foundation complete and cleared for Layer 4 UAT. |
| docs/internal/archive/sessions/2025-11-09_verification/VERIFICATION_PROTOCOL_FINAL_STATUS_V3.md | doc | 5-hour verification session completion report with Layers 1-3 fully done, Layer 4 partial, rated exceptional quality. |
| docs/internal/archive/sessions/2025-11-09_verification/VERIFICATION_PROTOCOL_LAYER1_STATUS.md | doc | Layer 1 static verification status report showing 135/145 engine tests passing with 10 minor failures listed. |
| docs/internal/archive/sessions/2025-11-09_verification/VERIFICATION_PROTOCOL_LAYER2_COMPLETE.md | doc | Layer 2 completion report documenting the shadow worktree symlink fix that enabled successful E2E Build stage validation. |
| docs/internal/archive/sessions/2025-11-09_verification/VERIFICATION_PROTOCOL_LAYER2_STATUS.md | doc | Layer 2 partial-status report showing test infrastructure updated but validation blocked by pytest configuration issues. |
| docs/internal/archive/sessions/2025-11-09_verification/VERIFICATION_PROTOCOL_LAYER3_COMPLETE.md | doc | Layer 3 completion report adding multi-module progression validation and 7 adversarial stress tests to the regression suite. |
| docs/internal/archive/sessions/2025-11-09_verification/VERIFICATION_PROTOCOL_LAYERS_1_2_COMPLETE.md | doc | Combined Layers 1-2 completion report confirming production-ready foundation with full E2E BJH loop automated coverage. |
| docs/internal/archive/sessions/2025-11-10_bug_system/AST_HARDEN_PHASE2_COMPLETE.md | doc | Phase 2 completion report integrating AST-based bug injection into production HardenRunner for the softmax module. |
| docs/internal/archive/sessions/2025-11-10_bug_system/AST_HARDEN_PHASE2_FINAL.md | doc | Phase 2 final verification confirming end-to-end AST bug injection with student variable name preservation battle-tested. |
| docs/internal/archive/sessions/2025-11-10_bug_system/BOTTLENECK_DIAGNOSIS.md | doc | Manual analysis report identifying wrong replacement strategy as the bottleneck causing 0% LLM success despite 95.8% node accuracy. |
| docs/internal/archive/sessions/2025-11-10_bug_system/COMPLETE_SUCCESS_SUMMARY.md | doc | Summary confirming all systematic improvements implemented, with AdamW bias-correction bug injection producing correct buggy code. |
| docs/internal/archive/sessions/2025-11-10_bug_system/CRITICAL_BUG_RESOLUTION.md | doc | Critical bug resolution report for student mode containing complete implementations in 10 of 22 modules, enabling bypassing of validation. |
| docs/internal/archive/sessions/2025-11-10_bug_system/EVALUATION_FIXED_ANALYSIS.md | doc | Analysis showing evaluation success improved from 0% to 50% by switching from text comparison to AST-based functional comparison. |
| docs/internal/archive/sessions/2025-11-10_bug_system/GPT4O_TEST_RESULTS.md | doc | Test results comparing gpt-4o vs gpt-4o-mini for bug authoring, showing 100% improvement in first-try success (25% to 50%). |
| docs/internal/archive/sessions/2025-11-10_bug_system/HARDEN_FIX_VERIFICATION.md | doc | Verification report confirming the fatal harden.py flaw (copying student code instead of reference for patching) was fixed. |
| docs/internal/archive/sessions/2025-11-10_bug_system/HARDEN_STAGE_CRITICAL_BUG.md | doc | Critical bug report identifying the fatal harden.py architectural flaw where student code was patched instead of the reference. |
| docs/internal/archive/sessions/2025-11-10_bug_system/LLM_PROMPT_REVIEW.md | doc | Review of the LLM prompt structure (system + user template) used for evaluating student justify-stage answers in JSON format. |
| docs/internal/archive/sessions/2025-11-10_bug_system/LLM_TOOL_DIAGNOSTIC_ANALYSIS.md | doc | Comprehensive diagnostic analysis of LLM bug generation failures, establishing a 4-bug golden dataset and validated transformation types. |
| docs/internal/archive/sessions/2025-11-10_bug_system/MANUAL_LLM_TEST.md | doc | Manual test procedure guide for one-time live OpenAI API integration validation of the Justify stage (~$0.01 cost). |
| docs/internal/archive/sessions/2025-11-10_bug_system/NEXT_BOTTLENECK_IDENTIFIED.md | doc | Manual analysis report pinpointing the silu wrong-replacement-strategy bottleneck after statistics showed 0% unknown failures. |
| docs/internal/archive/sessions/2025-11-10_bug_system/PATTERN_MATCHER_DEBUG_SESSION.md | doc | 3-hour debug session log resolving multiple AST pattern matching bugs including canonical variable renaming and indentation issues. |
| docs/internal/archive/sessions/2025-11-10_bug_system/PHASE2_SIGNOFF.md | doc | Formal sign-off approving Phase 2 AST-based bug injection for production use, rated exceptional quality. |
| docs/internal/archive/sessions/2025-11-10_bug_system/PHASE3_COMPLETION_REPORT.md | doc | Phase 3 completion report confirming the generic data-driven JSON bug engine replaced hardcoded AST injection, validated on 3 bug types. |
| docs/internal/archive/sessions/2025-11-10_bug_system/PHASE3_IMPLEMENTATION_PLAN.md | doc | Incremental test-driven implementation plan for Phase 3 generalization of SoftmaxBugInjector into a JSON-driven engine. |
| docs/internal/archive/sessions/2025-11-10_bug_system/PHASE4_FINAL_SIGNOFF.md | doc | Phase 4 final sign-off approving LLM-powered bug authoring tool with 83% bug-creation time reduction (13h to 2.3h for 17 bugs). |
| docs/internal/archive/sessions/2025-11-10_bug_system/PHASE4_LLM_TOOL.md | doc | Design and architecture document for Phase 4 LLM-powered bug authoring tool converting legacy .patch files to v2.1 JSON format. |
| docs/internal/archive/sessions/2025-11-10_bug_system/SESSION_COMPLETE_SUMMARY.md | doc | Complete session summary confirming all systematic improvement requirements met, with silu bottleneck diagnosed as the next P0 issue. |
| docs/internal/archive/sessions/2025-11-10_bug_system/TRAINING_DATA_VALIDATION.md | doc | Validation report confirming all 4/4 golden training examples (silu, attention, rmsnorm, adamw) produce correct buggy code. |
| docs/internal/archive/sessions/2025-11-11_curriculum_quality/BPE_TEST_FIX_SUMMARY.md | doc | Fix summary for BPE test that required exact merge order matching, causing false failures due to tie-breaking ambiguity in BPE algorithms. |
| docs/internal/archive/sessions/2025-11-11_curriculum_quality/COMPREHENSIVE_21_MODULE_EVALUATION.md | doc | Evaluation report for all 21 curriculum modules using gpt-4o, finding 10/21 (48%) actual success including 5 false-negative corrections. |
| docs/internal/archive/sessions/2025-11-11_curriculum_quality/COMPREHENSIVE_FIX_SUMMARY.md | doc | Session summary confirming all blockers fixed across 21 modules, with 3 critical bugs resolved via manual and statistical analysis. |
| docs/internal/archive/sessions/2025-11-11_curriculum_quality/COMPREHENSIVE_REMEDIATION_SUMMARY.md | doc | Multi-session (~20h) remediation summary spanning curriculum quality (98/100) and CLI interface improvements across 3 sessions. |
| docs/internal/archive/sessions/2025-11-11_curriculum_quality/CP_ACCELERATOR_IMPLEMENTATION_GUIDE.md | doc | Implementation guide for the CP Accelerator curriculum pack synthesising DSA Pattern Taxonomy and CP Roadmap into a BJH curriculum. |
| docs/internal/archive/sessions/2025-11-11_curriculum_quality/CP_ACCELERATOR_QUICKSTART.md | doc | Quick-start guide for CP Accelerator describing rating-driven progression from 0 to 1900+ through 19 algorithmic patterns. |
| docs/internal/archive/sessions/2025-11-11_curriculum_quality/CURRICULUM_COVERAGE.md | doc | Coverage map proving 100% alignment of all 21 curriculum modules to CS336 Assignment 1 PDF spec with from-scratch ethos. |
| docs/internal/archive/sessions/2025-11-11_curriculum_quality/CURRICULUM_GAP_ANALYSIS.md | doc | Gap analysis identifying only 3 of ~19 CS336 components initially covered (16%) and listing all unimplemented modules. |
| docs/internal/archive/sessions/2025-11-11_curriculum_quality/EINOPS_VIOLATIONS_AUDIT.md | doc | Audit finding 5 einops violations in multihead_self_attention reference implementation contrary to CS336 PDF §3.3 requirement. |
| docs/internal/archive/sessions/2025-11-11_curriculum_quality/GROUND_TRUTH_COMPLETE.md | doc | Completion report for creating and validating 21 AST-based bug.json golden patterns covering 100% of curriculum modules. |
| docs/internal/archive/sessions/2025-11-11_curriculum_quality/LITERATURE_VERIFICATION.md | doc | Literature verification guide mapping each curriculum module to ground-truth CS paper sources for pedagogical claim validation. |
| docs/internal/archive/sessions/2025-11-11_curriculum_quality/MASTER_REMEDIATION_STATUS.md | doc | Archived status summary tracking curriculum (98/100) and CLI P0 (100%) remediation progress across three work sessions. |
| docs/internal/archive/sessions/2025-11-11_curriculum_quality/PROJECT_STATUS.md | doc | Archived project status report declaring Phases 1-4 complete and production-ready for the AST bug injection engine. |
| docs/internal/archive/sessions/2025-11-11_curriculum_quality/QUALITY_REMEDIATION_PLAN.md | doc | Archived remediation plan for fixing curriculum internal consistency flaws identified against the CS336 Assignment 1 PDF ground truth. |
| docs/internal/archive/sessions/2025-11-11_curriculum_quality/REMEDIATION_PROGRESS.md | doc | Archived progress tracker for the 2025-11-12 curriculum quality remediation session recording completed audit and planning phases. |
| docs/internal/archive/sessions/2025-11-11_curriculum_quality/REMEDIATION_SUMMARY.md | doc | Archived summary confirming completion of Priority 1 and 2 curriculum remediation with engine support pending. |
| docs/internal/archive/sessions/2025-11-11_curriculum_quality/SESSION_3_SUMMARY.md | doc | Archived summary of Session 3 covering CLI interface systematic analysis and Phase 1 implementation planning. |
| docs/internal/archive/sessions/2025-11-11_curriculum_quality/STUDENT_MODE_AUDIT.md | doc | Archived audit mapping all student-mode module files that incorrectly contained full implementations instead of required stubs. |
| docs/internal/archive/sessions/2025-11-11_curriculum_quality/STUDENT_MODE_FIX_SUMMARY.md | doc | Archived summary documenting the critical fix that replaced full implementations with proper stubs in student mode files. |
| docs/internal/archive/sessions/2025-11-11_curriculum_quality/TOKENIZER_VIOLATIONS_AUDIT.md | doc | Archived audit confirming developer-mode BPE and Tokenizer reference implementations critically violate the from-scratch pedagogical constraint. |
| docs/internal/archive/sessions/2025-11-11_curriculum_quality/VERIFICATION_FINDINGS.md | doc | Archived findings from comparing curriculum build prompts against ground-truth literature to verify accuracy of the RoPE module and others. |
| docs/internal/archive/sessions/2025-11-12_cli_remediation/CLI_INTERFACE_AUDIT.md | doc | Archived audit report of engine/main.py CLI command interface identifying gaps, inconsistencies, and remediation priorities. |
| docs/internal/archive/sessions/2025-11-12_cli_remediation/CLI_P0_FINAL_STATUS.md | doc | Archived final status report confirming 100% feature parity achieved after approximately five hours of P0 CLI implementation work. |
| docs/internal/archive/sessions/2025-11-12_cli_remediation/CLI_P0_IMPLEMENTATION_COMPLETE.md | doc | Archived completion notice for the P0 CLI-001 command proliferation fix delivering the unified submit command in approximately 1.5 hours. |
| docs/internal/archive/sessions/2025-11-12_cli_remediation/CLI_P0_IMPLEMENTATION_PLAN.md | doc | Archived implementation plan describing the strategy for adding a unified submit command to resolve CLI-001 command proliferation. |
| docs/internal/archive/sessions/2025-11-12_cli_remediation/CLI_P0_PROGRESS.md | doc | Archived progress tracker for the P0 unified submit command implementation recording completion of Phases 1-3. |
| docs/internal/archive/sessions/2025-11-12_cli_remediation/CLI_P1_IMPLEMENTATION_COMPLETE.md | doc | Archived completion notice for the P1 CLI-002 inconsistent command behavior fix declared complete on 2025-11-12. |
| docs/internal/archive/sessions/2025-11-12_cli_remediation/CLI_REMEDIATION_COMPLETE.md | doc | Archived final report confirming all P0, P1, and P2 CLI remediation priorities completed in approximately six total hours. |
| docs/internal/archive/sessions/2025-11-12_cli_remediation/CLI_REMEDIATION_PLAN.md | doc | Archived remediation plan for the engine/main.py CLI interface specifying issues to fix and the execution-ready approach. |
| docs/internal/archive/sessions/2025-11-12_cli_remediation/CLI_REMEDIATION_STATUS.md | doc | Archived status report confirming planning complete and P0 core implementation complete for the CLI remediation session. |
| docs/internal/archive/sessions/2025-11-12_test_coverage/COMPLETE_COMPREHENSIVE_REPORT.md | doc | Archived comprehensive report covering CLI remediation and test coverage improvements across five systematic phases over ~10 hours. |
| docs/internal/archive/sessions/2025-11-12_test_coverage/COMPLETE_SESSION_SUMMARY.md | doc | Archived session summary for the combined CLI and test coverage work completed in approximately eight hours with all objectives exceeded. |
| docs/internal/archive/sessions/2025-11-12_test_coverage/COVERAGE_70_80_ACHIEVEMENT.md | doc | Archived report documenting the engine test coverage increase from 64% to 76%, exceeding the 70-80% target. |
| docs/internal/archive/sessions/2025-11-12_test_coverage/COVERAGE_80_ACHIEVEMENT.md | doc | Archived report documenting the engine test coverage increase from 76% to 78%, approaching the 80% threshold. |
| docs/internal/archive/sessions/2025-11-12_test_coverage/EXCEPTIONAL_RIGOR_FINAL_REPORT.md | doc | Archived report detailing five engine bug fixes, seven permanent diagnostics, and twelve manual LLM analyses completed with exceptional rigor. |
| docs/internal/archive/sessions/2025-11-12_test_coverage/FINAL_SESSION_REPORT.md | doc | Archived final session report for the combined CLI plus test coverage work across three major phases, declared production-ready. |
| docs/internal/archive/sessions/2025-11-12_test_coverage/FINAL_SESSION_SUMMARY.md | doc | Archived final summary confirming all systematic improvement objectives satisfied including permanent improvements and regression guards. |
| docs/internal/archive/sessions/2025-11-12_test_coverage/TEST_COVERAGE_FINAL_REPORT.md | doc | Archived final test coverage report after the three-hour systematic measurement and improvement session on 2025-11-12. |
| docs/internal/archive/sessions/2025-11-12_test_coverage/TEST_COVERAGE_IMPROVEMENT_SESSION.md | doc | Archived session record documenting the BPE test fix and coverage improvement rated five-star quality, completed November 13, 2025. |
| docs/internal/archive/sessions/2025-11-12_test_coverage/TEST_COVERAGE_SESSION_SUMMARY.md | doc | Archived session summary covering OpenAI SDK dependency fix, coverage baseline measurement, and incremental coverage improvements on 2025-11-12. |
| docs/internal/archive/sessions/2025-11-12_test_coverage/TEST_FIX_SUMMARY.md | doc | Archived fix summary confirming all engine tests passing after resolving one failing test in the engine test suite. |
| docs/internal/assignment/cs336_spring2025_assignment1_basics.pdf | asset | CS336 Spring 2025 Assignment 1 PDF handout serving as the ground-truth pedagogical specification for curriculum content. |
| docs/internal/coverage/CURRENT_REPORT.md | doc | Current test coverage report showing 78% overall coverage across 145 tests with 100% pass rate and production-ready status. |
| docs/internal/coverage/FINAL_COVERAGE_REPORT.txt | generated | Captured pytest-cov output showing final per-module statement/miss/cover numbers including engine/main.py at 36%. |
| docs/internal/coverage/baselines/baseline_20251112_180245.txt | generated | Captured pytest output baseline recording e2e test failures from a macOS environment on 2025-11-12T18:02. |
| docs/internal/coverage/baselines/baseline_no_e2e_20251112_180450.txt | generated | Captured pytest output baseline of passing non-e2e engine test results captured at 2025-11-12T18:04. |
| docs/internal/coverage/baselines/baseline_no_e2e_20251112_180628.txt | generated | Captured pytest output baseline for non-e2e tests from a second run at 2025-11-12T18:06. |
| docs/internal/coverage/baselines/baseline_no_e2e_full_20251112_180753.txt | generated | Captured full pytest session output baseline for 132 selected non-e2e tests on Python 3.13.1 at 2025-11-12T18:07. |
| docs/internal/coverage/baselines/stages_baseline.txt | generated | Captured pytest-cov baseline showing 31% combined coverage for engine/stages/harden.py and engine/stages/justify.py. |
| docs/internal/coverage/reports/coverage_after_cli_additions.txt | generated | Captured pytest-cov report for engine/main.py showing 48% coverage after CLI test additions. |
| docs/internal/coverage/reports/coverage_final_phase2.txt | generated | Captured pytest-cov full-engine coverage report after Phase 2 improvements showing engine/main.py at 48%. |
| docs/internal/coverage/reports/coverage_report_engine.txt | generated | Captured pytest-cov coverage report for the engine package showing engine/main.py at only 3% before test improvements. |
| docs/internal/coverage/reports/coverage_report_engine_final.txt | generated | Captured pytest-cov final full-engine coverage report showing engine/main.py at 34% after remediation. |
| docs/internal/coverage/reports/coverage_report_main_final.txt | generated | Captured pytest-cov coverage report for engine/main.py showing 28% coverage near end of remediation work. |
| docs/internal/coverage/reports/coverage_report_main_partial.txt | generated | Captured pytest-cov coverage report for engine/main.py at 15% coverage during partial mid-session remediation. |
| docs/internal/coverage/reports/coverage_report_no_e2e.txt | generated | Captured full pytest-cov report excluding e2e tests, including psutil and external macOS package paths from the original dev machine. |
| docs/internal/coverage/reports/coverage_with_new_cli_tests.txt | generated | Captured pytest-cov report for engine/main.py showing 48% coverage after adding new CLI-focused tests. |
| docs/internal/current/BUG_INJECTION_GUIDE.md | doc | Curriculum-author guide explaining the two-tier runtime AST bug injection architecture and how to define bug descriptors. |
| docs/internal/current/CURRICULUM_STATUS.md | doc | Current production status for two curricula: cs336_a1 (21 modules, 98/100) and cp_accelerator with module counts and quality ratings. |
| docs/internal/current/TEST_COVERAGE_REPORT.md | doc | Current test coverage report showing 78% overall coverage with 145 passing tests and production-ready status. |
| docs/internal/development/CHANGELOG.md | doc | Changelog tracking versioned code and handout changes to the CS336 assignment starting from v1.0.6 on 2025-08-28. |
| docs/internal/development/IMPLEMENTATION_PLAN.md | doc | Strategic implementation guide for completing CS336 Assignment 1, synthesizing best practices and providing an ordered implementation plan. |
| docs/internal/development/MASTERY_WORKLOG.md | doc | Reverse-chronological worklog documenting the systematic transformation of the CS336 repository into Mastery Engine v1.0. |
| docs/internal/development/MVP_COMPLETION_STATUS.md | doc | Production readiness declaration for Mastery Engine v1.0 MVP completed on November 12, 2025. |
| docs/internal/development/WORKLOG.md | doc | Reverse-chronological research worklog with scientific hypothesis/evidence entries for ML experiments starting 2025-09-16. |
| docs/internal/module_generation/MODULE_GENERATION_COMPREHENSIVE_SUMMARY.md | doc | Summary of the automated module generation system covering Phase 1 and Phase 2 results for 874 LeetCode problems. |
| docs/internal/module_generation/MODULE_GENERATION_PHASE2_RESULTS.md | doc | Results report for Phase 2 automated build prompt generation validated against LC-912 Sort an Array with quality improvements over manual creation. |
| docs/internal/module_generation/MODULE_GENERATION_PHASE3_DIAGNOSTIC.md | doc | Diagnostic report for Phase 3.1 module generation robustness testing on LC-200 Number of Islands. |
| docs/internal/module_generation/MODULE_GENERATION_PHASE3_RESULTS.md | doc | Results report for Phase 3.1 robustness testing on LC-200 confirming graceful degradation and automatic fallback mechanisms. |
| docs/internal/module_generation/MODULE_GENERATION_POC_RESULTS.md | doc | Proof-of-concept results for automated test case generation validated against LC-912, producing results equivalent to manual creation. |
| docs/internal/module_generation/MODULE_GENERATION_PROGRESS.md | doc | Progress summary for automating module asset generation for 874 LeetCode problems via a Curriculum-as-Code pipeline. |
| docs/internal/module_generation/MODULE_GENERATION_REFACTORING_PLAN.md | doc | Refactoring plan to evolve ingest_cp_content.py into generate_module.py for structured canonical_curriculum.json data consumption. |
| docs/internal/module_generation/README.md | doc | Index document for the module generation documentation archive explaining scope, approach, and Phases 1-3 status. |
| docs/internal/two_sum_qa/MODULE_COMPARISON_ANALYSIS.md | doc | Systematic comparison of sorting reference module versus two_sum generated module across all BUILD/JUSTIFY/HARDEN stage files. |
| docs/internal/two_sum_qa/MODULE_COMPLETENESS_VERIFICATION.md | doc | Checklist verifying that the two_sum module contains all required files for each BUILD/JUSTIFY/HARDEN stage. |
| docs/internal/two_sum_qa/README.md | doc | Index document for Two Sum QA documentation describing seven systematic testing phases achieving a 94/100 production-ready score. |
| docs/internal/two_sum_qa/TWO_SUM_COMPLETION_SUMMARY.md | doc | Completion summary declaring the Two Sum LC-1 module production-ready with a quality score of 94/100 on November 18, 2025. |
| docs/internal/two_sum_qa/TWO_SUM_E2E_WORKFLOW_TEST.md | doc | End-to-end test report verifying the Two Sum module across all three pedagogical stages and confirming production readiness. |
| docs/internal/two_sum_qa/TWO_SUM_FINAL_QUALITY_AUDIT.md | doc | Final production readiness audit by Cascade AI approving the Two Sum LC-1 module for production deployment. |
| docs/user-guide/MASTERY_COMMAND_REFERENCE.md | doc | User-facing reference for all 14 mastery CLI commands covering usage, flags, deprecation status, and installation instructions. |
| engine/__init__.py | source | Empty package initializer making 'engine' a Python package. |
| engine/ast_harden/__init__.py | source | Package docstring declaring the ast_harden sub-package as the AST-based bug injection engine for the Harden stage. |
| engine/ast_harden/generic_injector.py | source | GenericBugInjector class that reads declarative JSON bug definitions and executes multi-pass AST find-and-replace transformations on student code. |
| engine/ast_harden/pattern_matcher.py | source | PatternMatcher, FindAndTrackVisitor, and FindAndReplaceTransformer classes implementing the JSON-driven AST pattern matching and transformation system. |
| engine/ast_harden/softmax_poc.py | source | Phase-1 proof-of-concept hardcoded softmax bug injector (SoftmaxCanonicalizer + SoftmaxBugInjector) with a runnable __main__ test harness. |
| engine/ast_harden/softmax_v2_1.py | source | v2.1 two-phase mapping-based softmax bug injector that canonicalizes for matching but transforms the original AST to preserve student variable names, with a runnable __main__ test harness. |
| engine/curriculum.py | source | CurriculumManager class that loads and validates curriculum manifests and provides path accessors for module/problem/pattern assets in both LINEAR and LIBRARY curriculum types. |
| engine/dev_tools/__init__.py | source | Empty package initializer making 'engine/dev_tools' a Python package. |
| engine/dev_tools/bug_author.py | source | BugAuthor class that uses an LLM with few-shot golden examples to automatically generate v2.1 JSON bug definitions from legacy .patch files, with a validation loop. |
| engine/main.py | source | Typer-based CLI entry point defining all Mastery Engine commands (init, status, show, submit, start-challenge, select, create-bug, and legacy variants) for the Build-Justify-Harden learning loop. |
| engine/schemas.py | source | Pydantic data models for all engine contracts: CurriculumManifest, UserProgress, JustifyQuestion, LLMEvaluationResponse, ValidationResult, and BugDefinition schemas. |
| engine/services/__init__.py | source | Empty package initializer making 'engine/services' a Python package. |
| engine/services/ast_service.py | source | Canonicalizer, SoftmaxBugInjector, CanonicalPatternMatcher, and OriginalASTTransformer implementing the softmax-specific AST canonicalize-match-transform pipeline used by the harden service. |
| engine/services/llm_service.py | source | LLMService wrapping the OpenAI API for Chain-of-Thought justification evaluation and general completions, with mock mode when no API key is present. |
| engine/stages/__init__.py | source | Empty package initializer making 'engine/stages' a Python package. |
| engine/stages/harden.py | source | HardenRunner class that selects a bug, injects it (AST or patch), writes the buggy file to the shadow worktree, and returns the symptom for both LINEAR and LIBRARY curriculum modes. |
| engine/stages/justify.py | source | JustifyRunner class that loads justify questions from curriculum and applies a fast keyword failure-mode filter before delegating to LLM semantic evaluation. |
| engine/state.py | source | StateManager class that atomically reads and writes user progress to ~/.mastery_progress.json using a write-then-rename pattern. |
| engine/utils.py | source | find_project_root() utility that walks up the directory tree looking for pyproject.toml, .git, or curricula+engine markers to locate the repository root. |
| engine/validator.py | source | ValidationSubsystem class that executes validator.sh scripts in a controlled subprocess with timeout, captures exit code and output, and parses optional PERFORMANCE_SECONDS metrics. |
| engine/workspace.py | source | WorkspaceManager class providing file system abstractions for workspace paths, harden workspace isolation via file copy, and patch application via the system 'patch' command. |
| maintenance/PROJECT_STRUCTURE.md | doc | Describes the repository's dual-mode (student/developer) directory layout, key directories, mode-switching commands, and development workflows. |
| maintenance/README_ORIGINAL.md | doc | Original CS336 Assignment 1 README covering the Mastery Engine workflow, bug-injection architecture for the Harden stage, and environment setup instructions. |
| maintenance/RoadmapResources.md | doc | Curated list of competitive-programming learning resources (videos, blogs, practice sites) organized by Codeforces rating range. |
| maintenance/VERIFICATION_REPORT.md | doc | Post-fix verification report dated November 18, 2025, documenting TOML syntax fix, entry-point installation, test suite results (129/145 passing), and outstanding work items. |
| maintenance/make_submission.sh | source | Bash script that runs pytest and packages the project into a zip archive for CS336 assignment submission, excluding caches, binaries, and generated files. |
| modes/README.md | doc | Design rationale and curriculum overview for student/developer modes covering all 21 transformer LM modules. |
| modes/developer/cs336_basics/__init__.py | source | Package version initializer reading version from importlib.metadata for the developer reference package. |
| modes/developer/cs336_basics/bpe.py | source | Complete BPE tokenizer training using a doubly-linked-list with max-heap for O(n log n) merge selection and incremental pair-count updates. |
| modes/developer/cs336_basics/layers.py | source | Complete reference implementations of all transformer architecture components: Linear, Embedding, RMSNorm, SwiGLU, RoPE, scaled dot-product attention, multi-head attention, transformer_block, and transformer_lm. |
| modes/developer/cs336_basics/optimizer.py | source | Complete AdamW optimizer with bias-corrected moment estimates and decoupled weight decay applied before gradient steps. |
| modes/developer/cs336_basics/pretokenization_example.py | source | Provides find_chunk_boundaries helper for parallel corpus pre-tokenization plus an inline (non-importable) usage example using Ellipsis as a filename placeholder. |
| modes/developer/cs336_basics/tokenizer.py | source | Complete byte-level BPE Tokenizer wrapping tiktoken's GPT-2 encoding with greedy special-token segmentation for encode, decode, and encode_iterable. |
| modes/developer/cs336_basics/utils.py | source | Complete utility functions: numerically-stable softmax and cross-entropy, gradient clipping, cosine LR schedule with warmup, random batch sampler, and checkpoint save/load. |
| modes/student/cs336_basics/__init__.py | source | Package version initializer reading version from importlib.metadata for the student stub package. |
| modes/student/cs336_basics/bpe.py | source | Stub for train_bpe BPE training function with detailed implementation hints; raises NotImplementedError until the student implements it. |
| modes/student/cs336_basics/generation.py | source | Stub for autoregressive text generation function with temperature/top-k/top-p sampling; raises NotImplementedError. |
| modes/student/cs336_basics/layers.py | source | Stubs for all 10 transformer layer components (Linear, Embedding, silu, RMSNorm, SwiGLU, attention, RoPE, MHA, transformer_block, transformer_lm) each raising NotImplementedError with implementation guidance. |
| modes/student/cs336_basics/optimizer.py | source | Stub AdamW optimizer class with detailed docstring on decoupled weight decay; both __init__ and step raise NotImplementedError. |
| modes/student/cs336_basics/pretokenization_example.py | source | Provided-complete find_chunk_boundaries helper for parallel corpus chunking with inline usage example (identical to developer version). |
| modes/student/cs336_basics/tokenizer.py | source | Complete tiktoken-backed Tokenizer provided to students as a working helper; not a stub—uses GPT-2 encoding with greedy special-token matching. |
| modes/student/cs336_basics/tokenizer_stub.py | source | Stub Tokenizer class skeleton with all four methods (init, encode, decode, encode_iterable) raising NotImplementedError for students to implement from scratch. |
| modes/student/cs336_basics/utils.py | source | Stubs for all 7 utility functions (softmax, cross_entropy, gradient_clipping, cosine LR schedule, get_batch, save_checkpoint, load_checkpoint) each raising NotImplementedError with step-by-step hints. |
| scripts/add_successful_to_golden.py | source | Interactive CLI that reads LLM evaluation results from /tmp and prompts the developer to add successful bug definitions into the golden dataset. |
| scripts/auto_fix_drafts.py | source | Applies hardcoded fix functions for four known-bad draft AST injection patterns, tests each via GenericBugInjector, and saves corrected JSON files. |
| scripts/enrich_problems.py | source | Fetches full LeetCode problem details (description, examples, hints) from a third-party API and merges them into canonical_curriculum.json. |
| scripts/fetch_sources.sh | source | Downloads third-party curriculum source materials (30-Days-Of-Python repo, CP accelerator taxonomy placeholder) into the .sources/ directory. |
| scripts/fix_draft_pattern.py | source | Interactive tool that shows each draft AST injection pattern alongside its patch transformation, tests it, and lets the developer fix and promote it. |
| scripts/generate_ground_truth.py | source | Uses gpt-4o via BugAuthor to generate golden AST injection pattern JSON files for all 21 CS336-A1 curriculum modules in batch. |
| scripts/generate_manifest.py | source | Reads canonical_curriculum.json, validates the dependency DAG via topological sort, and writes manifest.json for linear or library curriculum types. |
| scripts/generate_module.py | source | Generates per-problem module assets (build_prompt.txt, test_cases.json, validator.sh) from enriched curriculum data, supporting single-problem and batch modes. |
| scripts/migrate_bugs_llm.py | source | Batch migrates legacy .patch bug files to v2.1 JSON format by invoking the LLM-based BugAuthor for each unprocessed patch in the cs336_a1 curriculum. |
| scripts/mode | source | Bash mode manager that switches the cs336_basics workspace symlink between student (stub) and developer (full) modes and can run commands in a mode temporarily. |
| scripts/parse_sources.py | source | Parses DSA taxonomy markdown files and a CP roadmap document to produce a verified canonical_curriculum.json for the cp_accelerator curriculum. |
| scripts/systematic_llm_evaluation.py | source | Benchmarks the LLM bug authoring tool across all 21 CS336-A1 bugs, collecting success rates, failure modes, pattern accuracy, and regression checks. |
| scripts/templates/build_prompt.jinja2 | asset | Jinja2 template that renders the build_prompt.txt challenge file shown to students, embedding problem statement, examples, constraints, hints, and resources. |
| scripts/test_ci.sh | source | Local CI runner that executes the exact pytest command used in GitHub Actions (tests/engine/, excluding integration marks) so developers can verify before pushing. |
| scripts/test_library_loading.py | test | Functional test that exercises CurriculumManager loading a LIBRARY manifest, verifying pattern/problem path resolution and on-disk file existence. |
| scripts/validate_student_stubs.py | source | Checks every Python file under modes/student/ for NotImplementedError stubs or TODO markers, exiting non-zero if complete implementations are found. |
| scripts/verify_curriculum_manifests.py | source | Statically verifies that all modules declared in a curriculum manifest have their required files (build_prompt.txt, validator.sh, etc.) present on disk. |
| scripts/verify_ground_truth.py | source | Runs every golden AST injection pattern against its corresponding patch to confirm the pattern produces the expected buggy code, reporting a pass/fail summary. |
| tests/__init__.py | test | Empty package init making tests/ a Python package. |
| tests/_snapshots/test_4d_scaled_dot_product_attention.npz | asset | Pre-committed NumPy snapshot array for 4D scaled-dot-product attention snapshot test. |
| tests/_snapshots/test_adamw.npz | asset | Pre-committed NumPy snapshot array for AdamW optimizer snapshot test. |
| tests/_snapshots/test_embedding.npz | asset | Pre-committed NumPy snapshot array for Embedding layer snapshot test. |
| tests/_snapshots/test_linear.npz | asset | Pre-committed NumPy snapshot array for Linear layer snapshot test. |
| tests/_snapshots/test_multihead_self_attention.npz | asset | Pre-committed NumPy snapshot array for multi-head self-attention snapshot test. |
| tests/_snapshots/test_multihead_self_attention_with_rope.npz | asset | Pre-committed NumPy snapshot array for multi-head self-attention with RoPE snapshot test. |
| tests/_snapshots/test_positionwise_feedforward.npz | asset | Pre-committed NumPy snapshot array for position-wise feedforward snapshot test. |
| tests/_snapshots/test_rmsnorm.npz | asset | Pre-committed NumPy snapshot array for RMSNorm snapshot test. |
| tests/_snapshots/test_rope.npz | asset | Pre-committed NumPy snapshot array for RoPE snapshot test. |
| tests/_snapshots/test_scaled_dot_product_attention.npz | asset | Pre-committed NumPy snapshot array for scaled-dot-product attention snapshot test. |
| tests/_snapshots/test_swiglu.npz | asset | Pre-committed NumPy snapshot array for SwiGLU feedforward snapshot test. |
| tests/_snapshots/test_train_bpe_special_tokens.pkl | asset | Pre-committed pickle snapshot of expected BPE vocab and merges for special-token training test. |
| tests/_snapshots/test_transformer_block.npz | asset | Pre-committed NumPy snapshot array for Transformer block snapshot test. |
| tests/_snapshots/test_transformer_lm.npz | asset | Pre-committed NumPy snapshot array for Transformer language model snapshot test. |
| tests/_snapshots/test_transformer_lm_truncated_input.npz | asset | Pre-committed NumPy snapshot array for Transformer LM with truncated input snapshot test. |
| tests/adapters.py | test | Thin adapter wrappers around cs336_basics implementations, used by test files to invoke student code through a stable interface. |
| tests/common.py | test | Shared test helpers: FIXTURES_PATH constant and gpt2_bytes_to_unicode utility used across tokenizer tests. |
| tests/conftest.py | test | Pytest conftest providing shared fixtures: numpy_snapshot, snapshot, ts_state_dict, and model-dimension fixtures for the test suite. |
| tests/e2e/E2E_TEST_STATUS.md | doc | Status report documenting current E2E test coverage (95%), known gaps, and rationale for v1.0 ship decision. |
| tests/e2e/__init__.py | test | Package init for E2E tests with module docstring describing end-to-end test scope. |
| tests/e2e/debug_shadow_worktree.py | test | Debug script that inspects shadow worktree structure and runs pytest collection diagnostics to trace import failures. |
| tests/e2e/test_adversarial_stress.py | test | Adversarial stress tests probing engine resilience: massive output, timeouts, corrupted patches, permission errors, and LLM prompt injection. |
| tests/e2e/test_build_only.py | test | Minimal E2E test verifying that the BUILD stage completes successfully when developer mode is active. |
| tests/e2e/test_complete_bjh_loop.py | test | Comprehensive E2E regression test for the full Build-Justify-Harden loop of the softmax module using isolated subprocess calls. |
| tests/e2e/test_error_handling.py | test | E2E tests validating engine behavior for error paths: uninitialized commands, stale worktree, wrong-stage usage, and corrupted state. |
| tests/e2e/test_full_softmax_loop.py | test | E2E test of the complete softmax BJH loop using the Typer CLI runner and mocked LLM, verifying all state transitions. |
| tests/engine/__init__.py | test | Empty package init making tests/engine/ a Python package. |
| tests/engine/test_curriculum.py | test | Unit tests for CurriculumManager: init, loading, path resolution, and error handling to achieve 100% line coverage. |
| tests/engine/test_error_handling.py | test | Unit tests targeting uncovered error-handling paths in submit, show, status, and start-challenge CLI commands. |
| tests/engine/test_harden_additional.py | test | Additional edge-case and error-path tests for HardenRunner to increase harden.py coverage. |
| tests/engine/test_init_cleanup.py | test | Unit tests for the init and cleanup CLI commands covering success paths and all documented error conditions. |
| tests/engine/test_legacy_commands.py | test | Tests for backward-compatible legacy submit commands (submit_build, submit_justification, submit_fix) in main.py. |
| tests/engine/test_llm_service.py | test | Unit tests for LLMService achieving 100% coverage using mocked OpenAI API responses. |
| tests/engine/test_main.py | test | Unit tests for main CLI commands with mocked state and curriculum managers, focusing on command behavior and error messages. |
| tests/engine/test_main_comprehensive_coverage.py | test | Aggressive coverage tests for submit-stage helpers _submit_build_stage, _submit_harden_stage, _submit_linear_workflow, and _submit_library_workflow. |
| tests/engine/test_main_console_paths.py | test | Tests verifying Rich console output produced by _submit_build_stage across validation success and failure paths. |
| tests/engine/test_main_error_paths.py | test | Tests for error and exception paths in require_shadow_worktree, _check_curriculum_complete, and _submit_linear_workflow. |
| tests/engine/test_main_helpers.py | test | Unit tests for isolated helper functions in main.py: require_shadow_worktree, _load_curriculum_state, _show_linear_status. |
| tests/engine/test_main_workflows_real.py | test | Systematic tests for workflow orchestration in _submit_linear_workflow and _submit_library_workflow. |
| tests/engine/test_new_cli_commands.py | test | Coverage tests for P1/P2 CLI commands: show, start_challenge, next (deprecated), curriculum_list, and progress_reset. |
| tests/engine/test_stages.py | test | Tests for harden and justify stage modules covering challenge setup, bug selection, question loading, and fast-filter logic. |
| tests/engine/test_state.py | test | Unit tests for StateManager achieving 100% line coverage including corrupted-file and write-error paths. |
| tests/engine/test_submit_handlers.py | test | Tests for the unified submit command and its underlying handler helpers to maximize main.py coverage. |
| tests/engine/test_utils_complete.py | test | Complete coverage tests for engine/utils.py find_project_root function. |
| tests/engine/test_validator.py | test | Unit tests for ValidationSubsystem achieving 100% coverage with focus on security-critical timeout enforcement. |
| tests/engine/test_workspace.py | test | Unit tests for WorkspaceManager covering workspace init, file operations, patch application, and error paths. |
| tests/fixtures/address.txt | asset | Gettysburg Address plaintext used as a BPE tokenizer training corpus fixture. |
| tests/fixtures/corpus.en | asset | English translation corpus sentences used as a tokenizer test fixture. |
| tests/fixtures/german.txt | asset | German-language text used as a non-ASCII BPE tokenizer test fixture. |
| tests/fixtures/gpt2_merges.txt | asset | GPT-2 BPE merge rules used to construct a reference tokenizer in tests. |
| tests/fixtures/gpt2_vocab.json | asset | GPT-2 vocabulary JSON mapping token strings to IDs, used to construct a reference tokenizer in tests. |
| tests/fixtures/special_token_double_newlines_non_whitespace.txt | asset | Short text with special token followed by double newlines and non-whitespace, used to test special-token boundary handling. |
| tests/fixtures/special_token_trailing_newlines.txt | asset | Short text with special token followed by trailing newlines, used to test special-token boundary handling. |
| tests/fixtures/tinystories_sample.txt | asset | Small TinyStories excerpt used as a fast BPE training corpus fixture. |
| tests/fixtures/tinystories_sample_5M.txt | asset | Larger 5M-token TinyStories sample used for heavier BPE training corpus tests. |
| tests/fixtures/train-bpe-reference-merges.txt | asset | Reference BPE merge list expected output for the train_bpe test on the TinyStories corpus. |
| tests/fixtures/train-bpe-reference-vocab.json | asset | Reference BPE vocabulary JSON expected output for the train_bpe test on the TinyStories corpus. |
| tests/fixtures/ts_tests/model.pt | asset | Serialized PyTorch model checkpoint (zip/pt format) used as a golden reference for layer output snapshot tests. |
| tests/fixtures/ts_tests/model_config.json | asset | JSON config (vocab_size, context_length, d_model, num_layers, etc.) for the ts_tests reference model. |
| tests/integration/README.md | doc | Documentation for integration tests: setup, cost per run, when to execute, and best practices for the live LLM API test suite. |
| tests/integration/__init__.py | test | Empty package init making tests/integration/ a Python package. |
| tests/integration/test_llm_service.py | test | Integration tests that make real OpenAI API calls to validate LLMService prompt formatting, response parsing, and error handling. |
| tests/one_d_probes.py | source | Standalone research script that trains a small Transformer on a 1D binary-sequence task to probe model internals; not a pytest test file. |
| tests/test_data.py | test | Tests for get_batch data-sampling utility: shape, randomness, and correct offset between input and label sequences. |
| tests/test_model.py | test | Snapshot-based tests for all neural-network layer implementations (Linear, Embedding, RoPE, MHA, SwiGLU, TransformerBlock, TransformerLM). |
| tests/test_nn_utils.py | test | Tests for nn utility functions: softmax numerical stability, cross-entropy, and gradient clipping against PyTorch references. |
| tests/test_optimizer.py | test | Tests for AdamW optimizer correctness and cosine learning-rate schedule via snapshot and arithmetic checks. |
| tests/test_serialization.py | test | Tests for checkpoint save/load round-trip fidelity for model weights and optimizer state. |
| tests/test_tokenizer.py | test | Tests for BPE Tokenizer encode/decode correctness, GPT-2 parity, special-token handling, memory limits, and train_bpe output. |
| tests/test_train_bpe.py | test | Pytest tests for BPE tokenizer training: validates speed (<1.5s), merge/vocab correctness against GPT-2 reference, and special-token isolation. |

---
### machine-readable artifact
```json
{
  "files": [
    {
      "path": ".env.example",
      "role": "config",
      "oneLiner": "Template for required environment variables (OPENAI_API_KEY and optional debug flags); copy to .env before running."
    },
    {
      "path": ".gitignore",
      "role": "config",
      "oneLiner": "Git ignore rules for Python build artifacts, virtual envs, test caches, IDE files, and project-specific derived files like the shadow worktree and mode symlink."
    },
    {
      "path": "LICENSE",
      "role": "doc",
      "oneLiner": "MIT License for original engine code with third-party attribution notices for Stanford CS336, LeetCode, and 30 Days of Python curriculum content."
    },
    {
      "path": "NOTICE",
      "role": "doc",
      "oneLiner": "NOTICE file listing all third-party content attributions and clarifying which components are original engineering work under MIT License."
    },
    {
      "path": "README.md",
      "role": "doc",
      "oneLiner": "Primary project documentation covering quick-start demo, CLI usage, architecture overview, curriculum descriptions, and the Build-Justify-Harden pedagogical loop."
    },
    {
      "path": "pyproject.toml",
      "role": "config",
      "oneLiner": "Python project manifest defining package metadata, all runtime dependencies, the `mastery` CLI entry point, pytest/ruff tool configuration, and uv build system settings."
    },
    {
      "path": ".github/workflows/tests.yml",
      "role": "config",
      "oneLiner": "GitHub Actions CI workflow that runs pytest unit tests and ruff lint/format checks for the engine package on push/PR to main."
    },
    {
      "path": ".github/workflows/validate_cp_manifest.yml",
      "role": "config",
      "oneLiner": "GitHub Actions CI workflow that validates and regenerates the cp_accelerator manifest.json, checks schema integrity, and analyzes the dependency graph on changes to curricula/cp_accelerator."
    },
    {
      "path": "audits/META_AUDIT_DEC_18.md",
      "role": "doc",
      "oneLiner": "Meta-audit identifying coverage gaps and blind spots in QUALITY_AUDIT.md, with a prioritized follow-up audit backlog for unreviewed repository surfaces."
    },
    {
      "path": "audits/QUALITY_AUDIT.md",
      "role": "doc",
      "oneLiner": "Primary quality and resilience audit artifact for the Mastery Engine, cataloguing 34 findings (9 high/22 medium/3 low) across architecture, security, CI, curricula, and scripts."
    },
    {
      "path": "curricula/cp_accelerator/IMPLEMENTATION_STATUS.md",
      "role": "doc",
      "oneLiner": "Documents architectural decisions, solved design flaws, and implementation milestones for the CP Accelerator curriculum pipeline."
    },
    {
      "path": "curricula/cp_accelerator/README.md",
      "role": "doc",
      "oneLiner": "Overview of the Competitive Programming Accelerator curriculum with attribution, content ownership, and original engineering contributions."
    },
    {
      "path": "curricula/cp_accelerator/STATUS.md",
      "role": "doc",
      "oneLiner": "Current completion status showing all 19 DSA taxonomy patterns parsed and resources partially extracted as of 2025-11-17."
    },
    {
      "path": "curricula/cp_accelerator/canonical_curriculum.json",
      "role": "config",
      "oneLiner": "Canonical source-of-truth JSON defining the full curriculum structure, patterns, resources, and rating brackets for the CP Accelerator."
    },
    {
      "path": "curricula/cp_accelerator/manifest.json",
      "role": "config",
      "oneLiner": "Curriculum manifest JSON with metadata, version, sources, and pattern list for the cp_accelerator curriculum type."
    },
    {
      "path": "curricula/cp_accelerator/patterns/backtracking/problems/lc_78/bugs/missing_copy.json",
      "role": "config",
      "oneLiner": "AST-based bug injection spec that replaces the list shallow copy `current[:]` with a reference `current` in the Subsets backtracking solution."
    },
    {
      "path": "curricula/cp_accelerator/patterns/backtracking/problems/lc_78/bugs/missing_copy_symptom.txt",
      "role": "doc",
      "oneLiner": "Human-readable symptom description and debugging guide for the missing list copy bug in the Subsets backtracking problem."
    },
    {
      "path": "curricula/cp_accelerator/patterns/backtracking/problems/lc_78/build_prompt.txt",
      "role": "asset",
      "oneLiner": "Problem statement, constraints, and implementation instructions presented to the learner for LeetCode 78 Subsets."
    },
    {
      "path": "curricula/cp_accelerator/patterns/backtracking/problems/lc_78/justify_questions.json",
      "role": "asset",
      "oneLiner": "Structured Socratic Q&A with model answers and failure-mode feedback for the Subsets backtracking justification phase."
    },
    {
      "path": "curricula/cp_accelerator/patterns/backtracking/problems/lc_78/solution.py",
      "role": "source",
      "oneLiner": "Reference solution for LeetCode 78 Subsets using backtracking with shallow-copy result accumulation; exports `solve` alias for test runner."
    },
    {
      "path": "curricula/cp_accelerator/patterns/backtracking/problems/lc_78/test_cases.json",
      "role": "test",
      "oneLiner": "Example test cases from the LeetCode 78 Subsets problem statement used by the local validator."
    },
    {
      "path": "curricula/cp_accelerator/patterns/backtracking/problems/lc_78/validator.sh",
      "role": "test",
      "oneLiner": "Shell script that imports solution.py and runs it against test_cases.json to validate the learner's Subsets implementation."
    },
    {
      "path": "curricula/cp_accelerator/patterns/backtracking/problems/lc_90/build_prompt.txt",
      "role": "asset",
      "oneLiner": "Problem statement and implementation instructions presented to the learner for LeetCode 90 Subsets II (with duplicates)."
    },
    {
      "path": "curricula/cp_accelerator/patterns/backtracking/problems/lc_90/test_cases.json",
      "role": "test",
      "oneLiner": "Example test cases from the LeetCode 90 Subsets II problem statement used by the local validator."
    },
    {
      "path": "curricula/cp_accelerator/patterns/backtracking/problems/lc_90/validator.sh",
      "role": "test",
      "oneLiner": "Shell script that imports solution.py and runs it against test_cases.json to validate the learner's Subsets II implementation."
    },
    {
      "path": "curricula/cp_accelerator/patterns/binary_search/problems/lc_34/build_prompt.txt",
      "role": "asset",
      "oneLiner": "Problem statement and implementation instructions presented to the learner for LeetCode 34 Find First and Last Position of Element."
    },
    {
      "path": "curricula/cp_accelerator/patterns/binary_search/problems/lc_34/test_cases.json",
      "role": "test",
      "oneLiner": "Example test cases from the LeetCode 34 problem statement used by the local validator."
    },
    {
      "path": "curricula/cp_accelerator/patterns/binary_search/problems/lc_34/validator.sh",
      "role": "test",
      "oneLiner": "Shell script that imports solution.py and validates the learner's Find First and Last Position implementation."
    },
    {
      "path": "curricula/cp_accelerator/patterns/binary_search/problems/lc_704/bugs/wrong_loop_condition.json",
      "role": "config",
      "oneLiner": "AST-based bug injection spec that replaces `left <= right` with `left < right` in the Binary Search while loop condition."
    },
    {
      "path": "curricula/cp_accelerator/patterns/binary_search/problems/lc_704/bugs/wrong_loop_condition_symptom.txt",
      "role": "doc",
      "oneLiner": "Human-readable symptom description showing wrong loop condition causes Binary Search to miss targets when left equals right."
    },
    {
      "path": "curricula/cp_accelerator/patterns/binary_search/problems/lc_704/build_prompt.txt",
      "role": "asset",
      "oneLiner": "Problem statement and implementation instructions presented to the learner for LeetCode 704 Binary Search."
    },
    {
      "path": "curricula/cp_accelerator/patterns/binary_search/problems/lc_704/justify_questions.json",
      "role": "asset",
      "oneLiner": "Structured Socratic Q&A with model answers and failure-mode feedback for the Binary Search justification phase."
    },
    {
      "path": "curricula/cp_accelerator/patterns/binary_search/problems/lc_704/solution.py",
      "role": "source",
      "oneLiner": "Reference solution for LeetCode 704 Binary Search using iterative halving; exports `solve` alias for the test runner."
    },
    {
      "path": "curricula/cp_accelerator/patterns/binary_search/problems/lc_704/test_cases.json",
      "role": "test",
      "oneLiner": "Example test cases from the LeetCode 704 Binary Search problem statement used by the local validator."
    },
    {
      "path": "curricula/cp_accelerator/patterns/binary_search/problems/lc_704/validator.sh",
      "role": "test",
      "oneLiner": "Shell script that imports solution.py and validates the learner's Binary Search implementation against example test cases."
    },
    {
      "path": "curricula/cp_accelerator/patterns/bit_manipulation/problems/lc_1342/build_prompt.txt",
      "role": "asset",
      "oneLiner": "Problem statement and implementation instructions presented to the learner for LeetCode 1342 Number of Steps to Reduce a Number to Zero."
    },
    {
      "path": "curricula/cp_accelerator/patterns/bit_manipulation/problems/lc_1342/test_cases.json",
      "role": "test",
      "oneLiner": "Example test cases from the LeetCode 1342 problem statement used by the local validator."
    },
    {
      "path": "curricula/cp_accelerator/patterns/bit_manipulation/problems/lc_1342/validator.sh",
      "role": "test",
      "oneLiner": "Shell script that imports solution.py and validates the learner's Number of Steps to Zero implementation."
    },
    {
      "path": "curricula/cp_accelerator/patterns/bit_manipulation/problems/lc_1486/build_prompt.txt",
      "role": "asset",
      "oneLiner": "Problem statement and implementation instructions presented to the learner for LeetCode 1486 XOR Operation in an Array."
    },
    {
      "path": "curricula/cp_accelerator/patterns/bit_manipulation/problems/lc_1486/test_cases.json",
      "role": "test",
      "oneLiner": "Example test cases from the LeetCode 1486 XOR Operation problem statement used by the local validator."
    },
    {
      "path": "curricula/cp_accelerator/patterns/bit_manipulation/problems/lc_1486/validator.sh",
      "role": "test",
      "oneLiner": "Shell script that imports solution.py and validates the learner's XOR Operation in an Array implementation."
    },
    {
      "path": "curricula/cp_accelerator/patterns/combinatorics_and_number_theory/problems/lc_46/build_prompt.txt",
      "role": "asset",
      "oneLiner": "Problem statement and implementation instructions presented to the learner for LeetCode 46 Permutations."
    },
    {
      "path": "curricula/cp_accelerator/patterns/combinatorics_and_number_theory/problems/lc_46/test_cases.json",
      "role": "test",
      "oneLiner": "Example test cases from the LeetCode 46 Permutations problem statement used by the local validator."
    },
    {
      "path": "curricula/cp_accelerator/patterns/combinatorics_and_number_theory/problems/lc_46/validator.sh",
      "role": "test",
      "oneLiner": "Shell script that imports solution.py and validates the learner's Permutations implementation."
    },
    {
      "path": "curricula/cp_accelerator/patterns/combinatorics_and_number_theory/problems/lc_47/build_prompt.txt",
      "role": "asset",
      "oneLiner": "Problem statement and implementation instructions presented to the learner for LeetCode 47 Permutations II (with duplicates)."
    },
    {
      "path": "curricula/cp_accelerator/patterns/combinatorics_and_number_theory/problems/lc_47/test_cases.json",
      "role": "test",
      "oneLiner": "Example test cases from the LeetCode 47 Permutations II problem statement used by the local validator."
    },
    {
      "path": "curricula/cp_accelerator/patterns/combinatorics_and_number_theory/problems/lc_47/validator.sh",
      "role": "test",
      "oneLiner": "Shell script that imports solution.py and validates the learner's Permutations II implementation."
    },
    {
      "path": "curricula/cp_accelerator/patterns/design_patterns/problems/lc_146/build_prompt.txt",
      "role": "asset",
      "oneLiner": "Problem statement and implementation instructions presented to the learner for LeetCode 146 LRU Cache."
    },
    {
      "path": "curricula/cp_accelerator/patterns/design_patterns/problems/lc_146/test_cases.json",
      "role": "test",
      "oneLiner": "Example test cases from the LeetCode 146 LRU Cache problem statement used by the local validator."
    },
    {
      "path": "curricula/cp_accelerator/patterns/design_patterns/problems/lc_146/validator.sh",
      "role": "test",
      "oneLiner": "Shell script that imports solution.py and validates the learner's LRU Cache implementation."
    },
    {
      "path": "curricula/cp_accelerator/patterns/design_patterns/problems/lc_460/build_prompt.txt",
      "role": "asset",
      "oneLiner": "Problem statement and implementation instructions presented to the learner for LeetCode 460 LFU Cache."
    },
    {
      "path": "curricula/cp_accelerator/patterns/design_patterns/problems/lc_460/test_cases.json",
      "role": "test",
      "oneLiner": "Example test cases from the LeetCode 460 LFU Cache problem statement used by the local validator."
    },
    {
      "path": "curricula/cp_accelerator/patterns/design_patterns/problems/lc_460/validator.sh",
      "role": "test",
      "oneLiner": "Shell script that imports solution.py and validates the learner's LFU Cache implementation."
    },
    {
      "path": "curricula/cp_accelerator/patterns/divide_and_conquer/problems/lc_148/build_prompt.txt",
      "role": "asset",
      "oneLiner": "Problem statement and implementation instructions presented to the learner for LeetCode 148 Sort List."
    },
    {
      "path": "curricula/cp_accelerator/patterns/divide_and_conquer/problems/lc_148/test_cases.json",
      "role": "test",
      "oneLiner": "Example test cases from the LeetCode 148 Sort List problem statement used by the local validator."
    },
    {
      "path": "curricula/cp_accelerator/patterns/divide_and_conquer/problems/lc_148/validator.sh",
      "role": "test",
      "oneLiner": "Shell script that imports solution.py and validates the learner's Sort List implementation."
    },
    {
      "path": "curricula/cp_accelerator/patterns/divide_and_conquer/problems/lc_912/build_prompt.txt",
      "role": "asset",
      "oneLiner": "Problem statement and implementation instructions presented to the learner for LeetCode 912 Sort an Array."
    },
    {
      "path": "curricula/cp_accelerator/patterns/divide_and_conquer/problems/lc_912/test_cases.json",
      "role": "test",
      "oneLiner": "Example test cases from the LeetCode 912 Sort an Array problem statement used by the local validator."
    },
    {
      "path": "curricula/cp_accelerator/patterns/divide_and_conquer/problems/lc_912/validator.sh",
      "role": "test",
      "oneLiner": "Shell script that imports solution.py and validates the learner's Sort an Array implementation."
    },
    {
      "path": "curricula/cp_accelerator/patterns/dynamic_programming/problems/lc_198/build_prompt.txt",
      "role": "asset",
      "oneLiner": "Problem statement and implementation instructions presented to the learner for LeetCode 198 House Robber."
    },
    {
      "path": "curricula/cp_accelerator/patterns/dynamic_programming/problems/lc_198/test_cases.json",
      "role": "test",
      "oneLiner": "Example test cases from the LeetCode 198 House Robber problem statement used by the local validator."
    },
    {
      "path": "curricula/cp_accelerator/patterns/dynamic_programming/problems/lc_198/validator.sh",
      "role": "test",
      "oneLiner": "Shell script that imports solution.py and validates the learner's House Robber implementation."
    },
    {
      "path": "curricula/cp_accelerator/patterns/dynamic_programming/problems/lc_70/bugs/wrong_base_case.json",
      "role": "config",
      "oneLiner": "AST-based bug injection spec that changes the DP base case from `n <= 2` to `n <= 1` in the Climbing Stairs solution."
    },
    {
      "path": "curricula/cp_accelerator/patterns/dynamic_programming/problems/lc_70/bugs/wrong_base_case_symptom.txt",
      "role": "doc",
      "oneLiner": "Human-readable symptom description showing wrong base case causes Climbing Stairs to return incorrect values for n=2 and all larger n."
    },
    {
      "path": "curricula/cp_accelerator/patterns/dynamic_programming/problems/lc_70/build_prompt.txt",
      "role": "asset",
      "oneLiner": "Problem statement and implementation instructions presented to the learner for LeetCode 70 Climbing Stairs."
    },
    {
      "path": "curricula/cp_accelerator/patterns/dynamic_programming/problems/lc_70/justify_questions.json",
      "role": "asset",
      "oneLiner": "Structured Socratic Q&A with model answers and failure-mode feedback for the Climbing Stairs DP justification phase."
    },
    {
      "path": "curricula/cp_accelerator/patterns/dynamic_programming/problems/lc_70/solution.py",
      "role": "source",
      "oneLiner": "Reference solution for LeetCode 70 Climbing Stairs using O(1)-space DP with Fibonacci recurrence; exports `solve` alias for the test runner."
    },
    {
      "path": "curricula/cp_accelerator/patterns/dynamic_programming/problems/lc_70/test_cases.json",
      "role": "test",
      "oneLiner": "Example test cases from the LeetCode 70 Climbing Stairs problem statement used by the local validator."
    },
    {
      "path": "curricula/cp_accelerator/patterns/dynamic_programming/problems/lc_70/validator.sh",
      "role": "test",
      "oneLiner": "Shell script that imports solution.py and validates the learner's Climbing Stairs implementation."
    },
    {
      "path": "curricula/cp_accelerator/patterns/greedy/problems/lc_435/bugs/sort_by_start.json",
      "role": "config",
      "oneLiner": "AST-based bug injection spec that replaces sort-by-end-time with sort-by-start-time in the Non-overlapping Intervals greedy solution."
    },
    {
      "path": "curricula/cp_accelerator/patterns/greedy/problems/lc_435/bugs/sort_by_start_symptom.txt",
      "role": "doc",
      "oneLiner": "Human-readable symptom description showing sort-by-start causes the greedy interval algorithm to keep long blocking intervals."
    },
    {
      "path": "curricula/cp_accelerator/patterns/greedy/problems/lc_435/build_prompt.txt",
      "role": "asset",
      "oneLiner": "Problem statement and implementation instructions presented to the learner for LeetCode 435 Non-overlapping Intervals."
    },
    {
      "path": "curricula/cp_accelerator/patterns/greedy/problems/lc_435/justify_questions.json",
      "role": "asset",
      "oneLiner": "Structured Socratic Q&A with model answers and failure-mode feedback for the greedy interval scheduling justification phase."
    },
    {
      "path": "curricula/cp_accelerator/patterns/greedy/problems/lc_435/solution.py",
      "role": "source",
      "oneLiner": "Reference solution for LeetCode 435 Non-overlapping Intervals using greedy sort-by-end-time; exports `solve` alias for the test runner."
    },
    {
      "path": "curricula/cp_accelerator/patterns/greedy/problems/lc_435/test_cases.json",
      "role": "test",
      "oneLiner": "Example test cases from the LeetCode 435 Non-overlapping Intervals problem statement used by the local validator."
    },
    {
      "path": "curricula/cp_accelerator/patterns/greedy/problems/lc_435/validator.sh",
      "role": "test",
      "oneLiner": "Shell script that imports solution.py and validates the learner's Non-overlapping Intervals implementation."
    },
    {
      "path": "curricula/cp_accelerator/patterns/greedy/problems/lc_452/build_prompt.txt",
      "role": "asset",
      "oneLiner": "Problem statement and implementation instructions presented to the learner for LeetCode 452 Minimum Number of Arrows to Burst Balloons."
    },
    {
      "path": "curricula/cp_accelerator/patterns/greedy/problems/lc_452/test_cases.json",
      "role": "test",
      "oneLiner": "Example test cases from the LeetCode 452 Minimum Arrows problem statement used by the local validator."
    },
    {
      "path": "curricula/cp_accelerator/patterns/greedy/problems/lc_452/validator.sh",
      "role": "test",
      "oneLiner": "Shell script that imports solution.py and validates the learner's Minimum Arrows to Burst Balloons implementation."
    },
    {
      "path": "curricula/cp_accelerator/patterns/hash_table/problems/lc_1/bugs/insert_before_check.json",
      "role": "config",
      "oneLiner": "AST-based bug injection spec that replaces complement lookup with current-number lookup in the Two Sum hash table solution."
    },
    {
      "path": "curricula/cp_accelerator/patterns/hash_table/problems/lc_1/bugs/insert_before_check_symptom.txt",
      "role": "doc",
      "oneLiner": "Human-readable symptom description for the insert-before-check bug in Two Sum, with step-by-step walkthrough and debugging hint."
    },
    {
      "path": "curricula/cp_accelerator/patterns/hash_table/problems/lc_1/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Problem statement and build challenge for LeetCode 1 (Two Sum) with hints, learning resources, and mastery submit instructions."
    },
    {
      "path": "curricula/cp_accelerator/patterns/hash_table/problems/lc_1/justify_questions.json",
      "role": "asset",
      "oneLiner": "Justification Q&A set with model answers and failure-mode feedback for evaluating learner understanding of Two Sum and hash table trade-offs."
    },
    {
      "path": "curricula/cp_accelerator/patterns/hash_table/problems/lc_1/solution.py",
      "role": "source",
      "oneLiner": "O(n) reference solution for Two Sum using a hash map (seen dict) to find complement pairs in a single pass."
    },
    {
      "path": "curricula/cp_accelerator/patterns/hash_table/problems/lc_1/solution_buggy.py",
      "role": "asset",
      "oneLiner": "Empty placeholder (0 bytes) for a generated buggy Two Sum solution; populated by the bug-injection engine at exercise time."
    },
    {
      "path": "curricula/cp_accelerator/patterns/hash_table/problems/lc_1/test_cases.json",
      "role": "asset",
      "oneLiner": "Eight JSON test cases (including negatives, zeros, large array) for Two Sum consumed by validator.sh."
    },
    {
      "path": "curricula/cp_accelerator/patterns/hash_table/problems/lc_1/validator.sh",
      "role": "source",
      "oneLiner": "Bash CLI script that imports twoSum from solution.py via inline Python and reports pass/fail against test_cases.json."
    },
    {
      "path": "curricula/cp_accelerator/patterns/hash_table/problems/lc_217/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Problem statement and build challenge for LeetCode 217 (Contains Duplicate) with hash-table pattern overview and submit instructions."
    },
    {
      "path": "curricula/cp_accelerator/patterns/hash_table/problems/lc_217/test_cases.json",
      "role": "asset",
      "oneLiner": "JSON test cases for Contains Duplicate consumed by validator.sh."
    },
    {
      "path": "curricula/cp_accelerator/patterns/hash_table/problems/lc_217/validator.sh",
      "role": "source",
      "oneLiner": "Bash CLI script that imports containsDuplicate from solution.py and runs it against test_cases.json."
    },
    {
      "path": "curricula/cp_accelerator/patterns/hash_table/problems/lc_219/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Problem statement and build challenge for LeetCode 219 (Contains Duplicate II) with hash-table pattern overview and submit instructions."
    },
    {
      "path": "curricula/cp_accelerator/patterns/hash_table/problems/lc_219/test_cases.json",
      "role": "asset",
      "oneLiner": "JSON test cases for Contains Duplicate II consumed by validator.sh."
    },
    {
      "path": "curricula/cp_accelerator/patterns/hash_table/problems/lc_219/validator.sh",
      "role": "source",
      "oneLiner": "Bash CLI script that imports containsNearbyDuplicate from solution.py and runs it against test_cases.json."
    },
    {
      "path": "curricula/cp_accelerator/patterns/hash_table/theory/justify_questions.json",
      "role": "asset",
      "oneLiner": "Theory-level justification Q&A for hash table patterns, covering advantage over brute force, complexity analysis, and edge cases."
    },
    {
      "path": "curricula/cp_accelerator/patterns/heap_and_priority_queue/problems/lc_215/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Problem statement and build challenge for LeetCode 215 (Kth Largest Element in an Array) with submit instructions."
    },
    {
      "path": "curricula/cp_accelerator/patterns/heap_and_priority_queue/problems/lc_215/test_cases.json",
      "role": "asset",
      "oneLiner": "JSON test cases for Kth Largest Element in an Array consumed by validator.sh."
    },
    {
      "path": "curricula/cp_accelerator/patterns/heap_and_priority_queue/problems/lc_215/validator.sh",
      "role": "source",
      "oneLiner": "Bash CLI script that imports findKthLargest from solution.py and runs it against test_cases.json."
    },
    {
      "path": "curricula/cp_accelerator/patterns/heap_and_priority_queue/problems/lc_703/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Problem statement and build challenge for LeetCode 703 (Kth Largest Element in a Stream) with submit instructions."
    },
    {
      "path": "curricula/cp_accelerator/patterns/heap_and_priority_queue/problems/lc_703/test_cases.json",
      "role": "asset",
      "oneLiner": "JSON test cases for Kth Largest Element in a Stream consumed by validator.sh."
    },
    {
      "path": "curricula/cp_accelerator/patterns/heap_and_priority_queue/problems/lc_703/validator.sh",
      "role": "source",
      "oneLiner": "Bash CLI script that imports the KthLargest class from solution.py and runs it against test_cases.json."
    },
    {
      "path": "curricula/cp_accelerator/patterns/linked_list/problems/lc_203/bugs/skip_consecutive.json",
      "role": "config",
      "oneLiner": "AST bug-injection spec (engine v2.1) that replaces != with > in the removeElements filter to cause wrong comparison operator bug."
    },
    {
      "path": "curricula/cp_accelerator/patterns/linked_list/problems/lc_203/bugs/skip_consecutive_symptom.txt",
      "role": "doc",
      "oneLiner": "Symptom description for the wrong comparison operator bug in Remove Linked List Elements, with expected vs actual output and debugging guide."
    },
    {
      "path": "curricula/cp_accelerator/patterns/linked_list/problems/lc_203/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Problem statement and build challenge for LeetCode 203 (Remove Linked List Elements) with submit instructions."
    },
    {
      "path": "curricula/cp_accelerator/patterns/linked_list/problems/lc_203/justify_questions.json",
      "role": "asset",
      "oneLiner": "Justification Q&A set with model answers and failure-mode feedback for evaluating learner understanding of Remove Linked List Elements."
    },
    {
      "path": "curricula/cp_accelerator/patterns/linked_list/problems/lc_203/solution.py",
      "role": "source",
      "oneLiner": "O(n) reference solution for Remove Linked List Elements using array representation compatible with the test runner."
    },
    {
      "path": "curricula/cp_accelerator/patterns/linked_list/problems/lc_203/test_cases.json",
      "role": "asset",
      "oneLiner": "JSON test cases for Remove Linked List Elements consumed by validator.sh."
    },
    {
      "path": "curricula/cp_accelerator/patterns/linked_list/problems/lc_203/validator.sh",
      "role": "source",
      "oneLiner": "Bash CLI script that imports removeElements from solution.py and runs it against test_cases.json."
    },
    {
      "path": "curricula/cp_accelerator/patterns/linked_list/problems/lc_237/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Problem statement and build challenge for LeetCode 237 (Delete Node in a Linked List) with submit instructions."
    },
    {
      "path": "curricula/cp_accelerator/patterns/linked_list/problems/lc_237/test_cases.json",
      "role": "asset",
      "oneLiner": "JSON test cases for Delete Node in a Linked List consumed by validator.sh."
    },
    {
      "path": "curricula/cp_accelerator/patterns/linked_list/problems/lc_237/validator.sh",
      "role": "source",
      "oneLiner": "Bash CLI script that imports deleteNode from solution.py and runs it against test_cases.json."
    },
    {
      "path": "curricula/cp_accelerator/patterns/prefix_sum/problems/lc_1480/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Problem statement and build challenge for LeetCode 1480 (Running Sum of 1d Array) with submit instructions."
    },
    {
      "path": "curricula/cp_accelerator/patterns/prefix_sum/problems/lc_1480/test_cases.json",
      "role": "asset",
      "oneLiner": "JSON test cases for Running Sum of 1d Array consumed by validator.sh."
    },
    {
      "path": "curricula/cp_accelerator/patterns/prefix_sum/problems/lc_1480/validator.sh",
      "role": "source",
      "oneLiner": "Bash CLI script that imports runningSum from solution.py and runs it against test_cases.json."
    },
    {
      "path": "curricula/cp_accelerator/patterns/prefix_sum/problems/lc_303/bugs/off_by_one_prefix.json",
      "role": "config",
      "oneLiner": "AST bug-injection spec (engine v2.1) that introduces an off-by-one error in the prefix sum range query boundary."
    },
    {
      "path": "curricula/cp_accelerator/patterns/prefix_sum/problems/lc_303/bugs/off_by_one_prefix_symptom.txt",
      "role": "doc",
      "oneLiner": "Symptom description for the off-by-one prefix sum bug in Range Sum Query, with expected vs actual output and debugging guide."
    },
    {
      "path": "curricula/cp_accelerator/patterns/prefix_sum/problems/lc_303/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Problem statement and build challenge for LeetCode 303 (Range Sum Query - Immutable) with submit instructions."
    },
    {
      "path": "curricula/cp_accelerator/patterns/prefix_sum/problems/lc_303/justify_questions.json",
      "role": "asset",
      "oneLiner": "Justification Q&A set with model answers for evaluating learner understanding of Range Sum Query and prefix sum technique."
    },
    {
      "path": "curricula/cp_accelerator/patterns/prefix_sum/problems/lc_303/solution.py",
      "role": "source",
      "oneLiner": "O(n) build / O(1) query reference solution for Range Sum Query using a prefix sum array."
    },
    {
      "path": "curricula/cp_accelerator/patterns/prefix_sum/problems/lc_303/test_cases.json",
      "role": "asset",
      "oneLiner": "JSON test cases for Range Sum Query - Immutable consumed by validator.sh."
    },
    {
      "path": "curricula/cp_accelerator/patterns/prefix_sum/problems/lc_303/validator.sh",
      "role": "source",
      "oneLiner": "Bash CLI script that imports sumRange from solution.py and runs it against test_cases.json."
    },
    {
      "path": "curricula/cp_accelerator/patterns/segment_tree_and_fenwick_tree/problems/lc_307/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Problem statement and build challenge for LeetCode 307 (Range Sum Query - Mutable) with submit instructions."
    },
    {
      "path": "curricula/cp_accelerator/patterns/segment_tree_and_fenwick_tree/problems/lc_307/test_cases.json",
      "role": "asset",
      "oneLiner": "JSON test cases for Range Sum Query - Mutable consumed by validator.sh."
    },
    {
      "path": "curricula/cp_accelerator/patterns/segment_tree_and_fenwick_tree/problems/lc_307/validator.sh",
      "role": "source",
      "oneLiner": "Bash CLI script that imports the NumArray class from solution.py and runs it against test_cases.json."
    },
    {
      "path": "curricula/cp_accelerator/patterns/sorting/problems/lc_148/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Problem statement and build challenge for LeetCode 148 (Sort List) with submit instructions."
    },
    {
      "path": "curricula/cp_accelerator/patterns/sorting/problems/lc_148/test_cases.json",
      "role": "asset",
      "oneLiner": "JSON test cases for Sort List consumed by validator.sh."
    },
    {
      "path": "curricula/cp_accelerator/patterns/sorting/problems/lc_148/validator.sh",
      "role": "source",
      "oneLiner": "Bash CLI script that imports sortList from solution.py and runs it against test_cases.json."
    },
    {
      "path": "curricula/cp_accelerator/patterns/sorting/problems/lc_912/bugs/incomplete_merge.json",
      "role": "config",
      "oneLiner": "AST bug-injection spec (engine v2.1) that deletes the result.extend(right[j:]) statement from the merge function, omitting trailing right-array elements."
    },
    {
      "path": "curricula/cp_accelerator/patterns/sorting/problems/lc_912/bugs/incomplete_merge.patch",
      "role": "doc",
      "oneLiner": "Unified diff showing the incomplete_merge mutation against the lc_912 reference solution for audit and review."
    },
    {
      "path": "curricula/cp_accelerator/patterns/sorting/problems/lc_912/bugs/incomplete_merge.py",
      "role": "asset",
      "oneLiner": "Empty placeholder (0 bytes) for the generated incomplete-merge buggy solution; populated by the bug-injection engine at exercise time."
    },
    {
      "path": "curricula/cp_accelerator/patterns/sorting/problems/lc_912/bugs/incomplete_merge_symptom.txt",
      "role": "doc",
      "oneLiner": "Symptom description for the incomplete merge bug in Sort an Array: output shorter than input with debugging hint about leftover elements."
    },
    {
      "path": "curricula/cp_accelerator/patterns/sorting/problems/lc_912/bugs/missing_base_case.json",
      "role": "config",
      "oneLiner": "AST bug-injection spec (engine v2.1) that deletes the base case from sortArray, causing infinite recursion on single-element arrays."
    },
    {
      "path": "curricula/cp_accelerator/patterns/sorting/problems/lc_912/bugs/missing_base_case.py",
      "role": "asset",
      "oneLiner": "Empty placeholder (0 bytes) for the generated missing-base-case buggy solution; populated by the bug-injection engine at exercise time."
    },
    {
      "path": "curricula/cp_accelerator/patterns/sorting/problems/lc_912/bugs/missing_base_case_symptom.txt",
      "role": "doc",
      "oneLiner": "Symptom description for the missing base case bug in merge sort: RecursionError with debugging guide explaining why base cases are essential."
    },
    {
      "path": "curricula/cp_accelerator/patterns/sorting/problems/lc_912/bugs/off_by_one.py",
      "role": "asset",
      "oneLiner": "Empty placeholder (0 bytes) for the generated off-by-one buggy merge sort; populated by the bug-injection engine at exercise time."
    },
    {
      "path": "curricula/cp_accelerator/patterns/sorting/problems/lc_912/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Problem statement and build challenge for LeetCode 912 (Sort an Array) with merge-sort pattern overview and submit instructions."
    },
    {
      "path": "curricula/cp_accelerator/patterns/sorting/problems/lc_912/justify_questions.json",
      "role": "asset",
      "oneLiner": "Justification Q&A set with model answers for evaluating learner understanding of merge sort invariants and complexity."
    },
    {
      "path": "curricula/cp_accelerator/patterns/sorting/problems/lc_912/solution.py",
      "role": "source",
      "oneLiner": "O(n log n) reference solution for Sort an Array using merge sort with a separate merge helper function."
    },
    {
      "path": "curricula/cp_accelerator/patterns/sorting/problems/lc_912/test_cases.json",
      "role": "asset",
      "oneLiner": "JSON test cases for Sort an Array consumed by validator.sh."
    },
    {
      "path": "curricula/cp_accelerator/patterns/sorting/problems/lc_912/validator.sh",
      "role": "source",
      "oneLiner": "Bash CLI script that imports sortArray from solution.py and runs it against test_cases.json."
    },
    {
      "path": "curricula/cp_accelerator/patterns/sorting/theory/justify_questions.json",
      "role": "asset",
      "oneLiner": "Theory-level justification Q&A for sorting patterns, covering merge sort invariant, divide-and-conquer correctness, and complexity."
    },
    {
      "path": "curricula/cp_accelerator/patterns/stack_and_queue/problems/lc_1003/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Problem statement and build challenge for LeetCode 1003 (Check If Word Is Valid After Substitutions) with submit instructions."
    },
    {
      "path": "curricula/cp_accelerator/patterns/stack_and_queue/problems/lc_1003/test_cases.json",
      "role": "asset",
      "oneLiner": "JSON test cases for Check If Word Is Valid After Substitutions consumed by validator.sh."
    },
    {
      "path": "curricula/cp_accelerator/patterns/stack_and_queue/problems/lc_1003/validator.sh",
      "role": "source",
      "oneLiner": "Bash CLI script that imports isValid from solution.py and runs it against test_cases.json for lc_1003."
    },
    {
      "path": "curricula/cp_accelerator/patterns/stack_and_queue/problems/lc_20/bugs/missing_empty_check.json",
      "role": "config",
      "oneLiner": "AST bug-injection spec (engine v2.1) that removes the empty-stack guard before stack[-1], causing IndexError on unmatched closing brackets."
    },
    {
      "path": "curricula/cp_accelerator/patterns/stack_and_queue/problems/lc_20/bugs/missing_empty_check_symptom.txt",
      "role": "doc",
      "oneLiner": "Symptom description for the missing empty-stack check bug in Valid Parentheses: IndexError on inputs like ')(' with debugging guide."
    },
    {
      "path": "curricula/cp_accelerator/patterns/stack_and_queue/problems/lc_20/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Problem statement and build challenge for LeetCode 20 (Valid Parentheses) with stack pattern overview and submit instructions."
    },
    {
      "path": "curricula/cp_accelerator/patterns/stack_and_queue/problems/lc_20/justify_questions.json",
      "role": "asset",
      "oneLiner": "Justification Q&A set with model answers for evaluating learner understanding of Valid Parentheses and stack-based matching."
    },
    {
      "path": "curricula/cp_accelerator/patterns/stack_and_queue/problems/lc_20/solution.py",
      "role": "source",
      "oneLiner": "O(n) reference solution for Valid Parentheses using a stack to match bracket pairs with closing-bracket lookup."
    },
    {
      "path": "curricula/cp_accelerator/patterns/stack_and_queue/problems/lc_20/test_cases.json",
      "role": "asset",
      "oneLiner": "JSON test cases for Valid Parentheses consumed by validator.sh."
    },
    {
      "path": "curricula/cp_accelerator/patterns/stack_and_queue/problems/lc_20/validator.sh",
      "role": "source",
      "oneLiner": "Bash CLI script that imports isValid from solution.py and runs it against test_cases.json for lc_20."
    },
    {
      "path": "curricula/cp_accelerator/patterns/traversal/problems/lc_144/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Problem statement, constraints, and implementation instructions for LeetCode 144 (Binary Tree Preorder Traversal)."
    },
    {
      "path": "curricula/cp_accelerator/patterns/traversal/problems/lc_144/test_cases.json",
      "role": "config",
      "oneLiner": "Example test cases (input/expected pairs) used by validator.sh to check LC-144 solutions."
    },
    {
      "path": "curricula/cp_accelerator/patterns/traversal/problems/lc_144/validator.sh",
      "role": "source",
      "oneLiner": "Shell script that imports student solution.py and runs it against test_cases.json, reporting pass/fail for LC-144."
    },
    {
      "path": "curricula/cp_accelerator/patterns/traversal/problems/lc_589/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Problem statement, constraints, and implementation instructions for LeetCode 589 (N-ary Tree Preorder Traversal)."
    },
    {
      "path": "curricula/cp_accelerator/patterns/traversal/problems/lc_589/test_cases.json",
      "role": "config",
      "oneLiner": "Example test cases used by validator.sh to check LC-589 N-ary preorder traversal solutions."
    },
    {
      "path": "curricula/cp_accelerator/patterns/traversal/problems/lc_589/validator.sh",
      "role": "source",
      "oneLiner": "Shell script that imports student solution.py and runs it against test_cases.json for LC-589."
    },
    {
      "path": "curricula/cp_accelerator/patterns/trie/problems/lc_1804/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Problem statement, constraints, and implementation instructions for LeetCode 1804 (Implement Trie II with prefix counts)."
    },
    {
      "path": "curricula/cp_accelerator/patterns/trie/problems/lc_1804/test_cases.json",
      "role": "config",
      "oneLiner": "Example test cases used by validator.sh to check LC-1804 Trie II solutions."
    },
    {
      "path": "curricula/cp_accelerator/patterns/trie/problems/lc_1804/validator.sh",
      "role": "source",
      "oneLiner": "Shell script that imports student solution.py and runs it against test_cases.json for LC-1804."
    },
    {
      "path": "curricula/cp_accelerator/patterns/trie/problems/lc_208/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Problem statement, constraints, and implementation instructions for LeetCode 208 (Implement Trie with insert/search/startsWith)."
    },
    {
      "path": "curricula/cp_accelerator/patterns/trie/problems/lc_208/test_cases.json",
      "role": "config",
      "oneLiner": "Example test cases used by validator.sh to check LC-208 Trie solutions."
    },
    {
      "path": "curricula/cp_accelerator/patterns/trie/problems/lc_208/validator.sh",
      "role": "source",
      "oneLiner": "Shell script that imports student solution.py and runs it against test_cases.json for LC-208."
    },
    {
      "path": "curricula/cp_accelerator/patterns/two_pointers/problems/lc_1099/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Problem statement, constraints, and implementation instructions for LeetCode 1099 (Two Sum Less Than K)."
    },
    {
      "path": "curricula/cp_accelerator/patterns/two_pointers/problems/lc_1099/test_cases.json",
      "role": "config",
      "oneLiner": "Example test cases used by validator.sh to check LC-1099 Two Sum Less Than K solutions."
    },
    {
      "path": "curricula/cp_accelerator/patterns/two_pointers/problems/lc_1099/validator.sh",
      "role": "source",
      "oneLiner": "Shell script that imports student solution.py and runs it against test_cases.json for LC-1099."
    },
    {
      "path": "curricula/cp_accelerator/patterns/two_pointers/problems/lc_167/bugs/wrong_pointer_move.json",
      "role": "config",
      "oneLiner": "AST-based bug injection descriptor that swaps the two-pointer convergence condition (< vs >) in twoSum, making pointers diverge."
    },
    {
      "path": "curricula/cp_accelerator/patterns/two_pointers/problems/lc_167/bugs/wrong_pointer_move_symptom.txt",
      "role": "doc",
      "oneLiner": "Describes the observable symptom (empty-list return) and debugging guide for the wrong_pointer_move injected bug."
    },
    {
      "path": "curricula/cp_accelerator/patterns/two_pointers/problems/lc_167/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Problem statement, constraints, and implementation instructions for LeetCode 167 (Two Sum II – sorted array)."
    },
    {
      "path": "curricula/cp_accelerator/patterns/two_pointers/problems/lc_167/justify_questions.json",
      "role": "config",
      "oneLiner": "Conceptual Justify-stage questions (with model answers and failure modes) about the two-pointer technique for LC-167."
    },
    {
      "path": "curricula/cp_accelerator/patterns/two_pointers/problems/lc_167/solution.py",
      "role": "source",
      "oneLiner": "Reference implementation of Two Sum II using O(n)/O(1) two-pointer approach; exposes twoSum and solve entry points."
    },
    {
      "path": "curricula/cp_accelerator/patterns/two_pointers/problems/lc_167/test_cases.json",
      "role": "config",
      "oneLiner": "Example and edge-case test cases (including negatives, boundaries) for validating LC-167 solutions."
    },
    {
      "path": "curricula/cp_accelerator/patterns/two_pointers/problems/lc_167/validator.sh",
      "role": "source",
      "oneLiner": "Shell script that imports student solution.py and runs it against test_cases.json for LC-167."
    },
    {
      "path": "curricula/cp_accelerator/patterns/union_find_disjoint_set_union/problems/lc_547/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Problem statement, constraints, and implementation instructions for LeetCode 547 (Number of Provinces via Union-Find)."
    },
    {
      "path": "curricula/cp_accelerator/patterns/union_find_disjoint_set_union/problems/lc_547/test_cases.json",
      "role": "config",
      "oneLiner": "Example test cases used by validator.sh to check LC-547 Number of Provinces solutions."
    },
    {
      "path": "curricula/cp_accelerator/patterns/union_find_disjoint_set_union/problems/lc_547/validator.sh",
      "role": "source",
      "oneLiner": "Shell script that imports student solution.py and runs it against test_cases.json for LC-547."
    },
    {
      "path": "curricula/cp_accelerator/patterns/union_find_disjoint_set_union/problems/lc_684/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Problem statement, constraints, and implementation instructions for LeetCode 684 (Redundant Connection via Union-Find)."
    },
    {
      "path": "curricula/cp_accelerator/patterns/union_find_disjoint_set_union/problems/lc_684/test_cases.json",
      "role": "config",
      "oneLiner": "Example test cases used by validator.sh to check LC-684 Redundant Connection solutions."
    },
    {
      "path": "curricula/cp_accelerator/patterns/union_find_disjoint_set_union/problems/lc_684/validator.sh",
      "role": "source",
      "oneLiner": "Shell script that imports student solution.py and runs it against test_cases.json for LC-684."
    },
    {
      "path": "curricula/cs336_a1/README.md",
      "role": "doc",
      "oneLiner": "Attribution, curriculum overview, module structure, usage commands, and educational philosophy for the CS336 A1 curriculum."
    },
    {
      "path": "curricula/cs336_a1/manifest.json",
      "role": "config",
      "oneLiner": "Module registry listing all 22 CS336 A1 modules in dependency order with paths, types, and performance baselines."
    },
    {
      "path": "curricula/cs336_a1/modules/adamw/bugs/missing_bias_correction.json",
      "role": "config",
      "oneLiner": "AST-based 4-pass bug injection descriptor removing bias-correction terms from the AdamW optimizer step."
    },
    {
      "path": "curricula/cs336_a1/modules/adamw/bugs/missing_bias_correction.patch",
      "role": "config",
      "oneLiner": "Git-unified-diff patch showing the missing_bias_correction bug as applied to cs336_basics/optimizer.py."
    },
    {
      "path": "curricula/cs336_a1/modules/adamw/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Build challenge instructions and mathematical specification for implementing the AdamW optimizer from scratch."
    },
    {
      "path": "curricula/cs336_a1/modules/adamw/justify_questions.json",
      "role": "config",
      "oneLiner": "Conceptual Justify-stage questions (with model answers and failure modes) about AdamW optimizer internals."
    },
    {
      "path": "curricula/cs336_a1/modules/adamw/validator.sh",
      "role": "source",
      "oneLiner": "Shell script running pytest for the AdamW optimizer implementation inside a shadow worktree."
    },
    {
      "path": "curricula/cs336_a1/modules/attention/bugs/missing_scale.json",
      "role": "config",
      "oneLiner": "AST-based bug injection descriptor that removes the 1/sqrt(d_k) scaling factor from scaled dot-product attention."
    },
    {
      "path": "curricula/cs336_a1/modules/attention/bugs/missing_scale.patch",
      "role": "config",
      "oneLiner": "Git-unified-diff patch showing the missing_scale bug as applied to the attention implementation."
    },
    {
      "path": "curricula/cs336_a1/modules/attention/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Build challenge instructions and mathematical specification for implementing scaled dot-product attention."
    },
    {
      "path": "curricula/cs336_a1/modules/attention/justify_questions.json",
      "role": "config",
      "oneLiner": "Conceptual Justify-stage questions (with model answers and failure modes) about the attention mechanism."
    },
    {
      "path": "curricula/cs336_a1/modules/attention/validator.sh",
      "role": "source",
      "oneLiner": "Shell script running pytest for the scaled dot-product attention implementation in a shadow worktree."
    },
    {
      "path": "curricula/cs336_a1/modules/bpe_tokenizer/bugs/wrong_merge_order.json",
      "role": "config",
      "oneLiner": "AST-based bug injection descriptor reversing BPE merge insertion order (prepend instead of append)."
    },
    {
      "path": "curricula/cs336_a1/modules/bpe_tokenizer/bugs/wrong_merge_order.patch",
      "role": "config",
      "oneLiner": "Git-unified-diff patch showing the wrong_merge_order bug as applied to the BPE tokenizer implementation."
    },
    {
      "path": "curricula/cs336_a1/modules/bpe_tokenizer/bugs/wrong_merge_order_draft.json",
      "role": "config",
      "oneLiner": "Draft version of the wrong_merge_order bug descriptor with more verbose description field."
    },
    {
      "path": "curricula/cs336_a1/modules/bpe_tokenizer/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Build challenge instructions and algorithmic specification for implementing BPE tokenizer training."
    },
    {
      "path": "curricula/cs336_a1/modules/bpe_tokenizer/justify_questions.json",
      "role": "config",
      "oneLiner": "Conceptual Justify-stage questions (with model answers and failure modes) about BPE tokenization."
    },
    {
      "path": "curricula/cs336_a1/modules/bpe_tokenizer/validator.sh",
      "role": "source",
      "oneLiner": "Shell script running pytest for the BPE tokenizer training implementation in a shadow worktree."
    },
    {
      "path": "curricula/cs336_a1/modules/checkpointing/bugs/missing_optimizer_state.json",
      "role": "config",
      "oneLiner": "AST-based bug injection descriptor that omits optimizer state from the checkpoint save call."
    },
    {
      "path": "curricula/cs336_a1/modules/checkpointing/bugs/missing_optimizer_state.patch",
      "role": "config",
      "oneLiner": "Git-unified-diff patch showing the missing_optimizer_state bug as applied to the checkpointing code."
    },
    {
      "path": "curricula/cs336_a1/modules/checkpointing/bugs/missing_optimizer_state_draft.json",
      "role": "config",
      "oneLiner": "Draft version of the missing_optimizer_state bug descriptor."
    },
    {
      "path": "curricula/cs336_a1/modules/checkpointing/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Build challenge instructions and specification for implementing model checkpoint save and load."
    },
    {
      "path": "curricula/cs336_a1/modules/checkpointing/justify_questions.json",
      "role": "config",
      "oneLiner": "Conceptual Justify-stage questions (with model answers and failure modes) about model checkpointing."
    },
    {
      "path": "curricula/cs336_a1/modules/checkpointing/validator.sh",
      "role": "source",
      "oneLiner": "Shell script running pytest for the checkpointing save/load implementation in a shadow worktree."
    },
    {
      "path": "curricula/cs336_a1/modules/cosine_schedule/bugs/wrong_cosine_range.json",
      "role": "config",
      "oneLiner": "AST-based bug injection descriptor introducing the wrong cosine oscillation range in the LR schedule."
    },
    {
      "path": "curricula/cs336_a1/modules/cosine_schedule/bugs/wrong_cosine_range.patch",
      "role": "config",
      "oneLiner": "Git-unified-diff patch showing the wrong_cosine_range bug as applied to the scheduler code."
    },
    {
      "path": "curricula/cs336_a1/modules/cosine_schedule/bugs/wrong_cosine_range_draft.json",
      "role": "config",
      "oneLiner": "Draft version of the wrong_cosine_range bug descriptor."
    },
    {
      "path": "curricula/cs336_a1/modules/cosine_schedule/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Build challenge instructions and mathematical specification for implementing cosine LR schedule with linear warmup."
    },
    {
      "path": "curricula/cs336_a1/modules/cosine_schedule/justify_questions.json",
      "role": "config",
      "oneLiner": "Conceptual Justify-stage questions (with model answers and failure modes) about cosine LR scheduling."
    },
    {
      "path": "curricula/cs336_a1/modules/cosine_schedule/validator.sh",
      "role": "source",
      "oneLiner": "Shell script running pytest for the cosine LR schedule implementation in a shadow worktree."
    },
    {
      "path": "curricula/cs336_a1/modules/cross_entropy/bugs/no_logsumexp.json",
      "role": "config",
      "oneLiner": "AST-based bug injection descriptor removing the logsumexp numerical stability trick from cross-entropy loss."
    },
    {
      "path": "curricula/cs336_a1/modules/cross_entropy/bugs/no_logsumexp.patch",
      "role": "config",
      "oneLiner": "Git-unified-diff patch showing the no_logsumexp bug as applied to the loss function code."
    },
    {
      "path": "curricula/cs336_a1/modules/cross_entropy/bugs/no_logsumexp_draft.json",
      "role": "config",
      "oneLiner": "Draft version of the no_logsumexp bug descriptor."
    },
    {
      "path": "curricula/cs336_a1/modules/cross_entropy/bugs/no_logsumexp_symptom.txt",
      "role": "doc",
      "oneLiner": "Describes the NaN/inf symptom and debugging guide for the missing logsumexp numerical stability bug."
    },
    {
      "path": "curricula/cs336_a1/modules/cross_entropy/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Build challenge instructions and mathematical specification for implementing numerically stable cross-entropy loss."
    },
    {
      "path": "curricula/cs336_a1/modules/cross_entropy/justify_questions.json",
      "role": "config",
      "oneLiner": "Conceptual Justify-stage questions (with model answers and failure modes) about numerically stable cross-entropy."
    },
    {
      "path": "curricula/cs336_a1/modules/cross_entropy/validator.sh",
      "role": "source",
      "oneLiner": "Shell script running pytest for the numerically stable cross-entropy implementation in a shadow worktree."
    },
    {
      "path": "curricula/cs336_a1/modules/data_loader/bugs/wrong_sampling_range.json",
      "role": "config",
      "oneLiner": "AST-based bug injection descriptor using an off-by-one high bound in randint, causing out-of-bounds token sampling."
    },
    {
      "path": "curricula/cs336_a1/modules/data_loader/bugs/wrong_sampling_range.patch",
      "role": "config",
      "oneLiner": "Git-unified-diff patch showing the wrong_sampling_range bug as applied to the data loader code."
    },
    {
      "path": "curricula/cs336_a1/modules/data_loader/bugs/wrong_sampling_range_draft.json",
      "role": "config",
      "oneLiner": "Draft version of the wrong_sampling_range bug descriptor."
    },
    {
      "path": "curricula/cs336_a1/modules/data_loader/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Build challenge instructions and specification for implementing the language model get_batch data loader."
    },
    {
      "path": "curricula/cs336_a1/modules/data_loader/justify_questions.json",
      "role": "config",
      "oneLiner": "Conceptual Justify-stage questions (with model answers and failure modes) about LM data loading and sampling."
    },
    {
      "path": "curricula/cs336_a1/modules/data_loader/validator.sh",
      "role": "source",
      "oneLiner": "Shell script that copies layers and runs pytest tests/test_training.py::test_get_batch in a shadow worktree to validate the data loader implementation."
    },
    {
      "path": "curricula/cs336_a1/modules/embedding/bugs/wrong_dimension_order.json",
      "role": "config",
      "oneLiner": "Production AST bug-injection spec (engine_version 2.1) that swaps num_embeddings and embedding_dim in the nn.Embedding constructor call."
    },
    {
      "path": "curricula/cs336_a1/modules/embedding/bugs/wrong_dimension_order.patch",
      "role": "asset",
      "oneLiner": "Unified diff showing the dimension-swap bug as applied to cs336_basics/layers.py for the Embedding module."
    },
    {
      "path": "curricula/cs336_a1/modules/embedding/bugs/wrong_dimension_order_draft.json",
      "role": "config",
      "oneLiner": "Draft (v2.0, multi-pass) AST bug-injection spec for swapping embedding dimensions; predecessor to the production wrong_dimension_order.json."
    },
    {
      "path": "curricula/cs336_a1/modules/embedding/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Student-facing instructional guide covering embedding theory and specifying how to implement the Embedding class from scratch using nn.Parameter."
    },
    {
      "path": "curricula/cs336_a1/modules/embedding/justify_questions.json",
      "role": "asset",
      "oneLiner": "Five conceptual Q&A pairs (with model answers and required concepts) for post-build assessment of the token embedding module."
    },
    {
      "path": "curricula/cs336_a1/modules/embedding/validator.sh",
      "role": "source",
      "oneLiner": "Shell script that copies layers.py and runs pytest tests/test_model.py::test_embedding in a shadow worktree to validate the Embedding implementation."
    },
    {
      "path": "curricula/cs336_a1/modules/gradient_clipping/bugs/per_parameter_clipping.json",
      "role": "config",
      "oneLiner": "Production AST bug-injection spec (engine_version 2.1) that replaces global-norm gradient clipping with per-parameter clipping."
    },
    {
      "path": "curricula/cs336_a1/modules/gradient_clipping/bugs/per_parameter_clipping.patch",
      "role": "asset",
      "oneLiner": "Unified diff showing the per-parameter clipping bug applied to the clip_gradients_by_global_norm function."
    },
    {
      "path": "curricula/cs336_a1/modules/gradient_clipping/bugs/per_parameter_clipping_draft.json",
      "role": "config",
      "oneLiner": "Draft AST bug-injection spec (LLM-generated, tier: complex) for the per-parameter clipping bug; predecessor to the production spec."
    },
    {
      "path": "curricula/cs336_a1/modules/gradient_clipping/bugs/per_parameter_clipping_symptom.txt",
      "role": "doc",
      "oneLiner": "Student-facing description of gradient direction distortion symptoms caused by the per-parameter clipping bug, with debugging hints and correct algorithm."
    },
    {
      "path": "curricula/cs336_a1/modules/gradient_clipping/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Student-facing instructional guide for implementing global L2 gradient norm clipping, including mathematical derivation and implementation steps."
    },
    {
      "path": "curricula/cs336_a1/modules/gradient_clipping/justify_questions.json",
      "role": "asset",
      "oneLiner": "Conceptual Q&A pairs with model answers for post-build assessment of the gradient clipping module."
    },
    {
      "path": "curricula/cs336_a1/modules/gradient_clipping/validator.sh",
      "role": "source",
      "oneLiner": "Shell script that runs pytest tests/test_nn_utils.py::test_gradient_clipping in a shadow worktree to validate the gradient clipping implementation."
    },
    {
      "path": "curricula/cs336_a1/modules/linear/bugs/missing_transpose.json",
      "role": "config",
      "oneLiner": "Production AST bug-injection spec (engine_version 2.1) that removes the .t() weight transpose in the Linear layer forward pass."
    },
    {
      "path": "curricula/cs336_a1/modules/linear/bugs/missing_transpose.patch",
      "role": "asset",
      "oneLiner": "Unified diff showing the missing weight transpose bug as applied to the Linear layer in cs336_basics/layers.py."
    },
    {
      "path": "curricula/cs336_a1/modules/linear/bugs/missing_transpose_draft.json",
      "role": "config",
      "oneLiner": "First draft AST bug-injection spec for the missing weight transpose bug in the Linear layer forward pass."
    },
    {
      "path": "curricula/cs336_a1/modules/linear/bugs/missing_transpose_draft_v2.json",
      "role": "config",
      "oneLiner": "Second draft (v2.1, author: auto_fixed) AST spec for the missing transpose bug; uses replace_value_with instead of replace_with."
    },
    {
      "path": "curricula/cs336_a1/modules/linear/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Student-facing instructional guide for implementing the Linear (fully connected) layer with optional bias from scratch."
    },
    {
      "path": "curricula/cs336_a1/modules/linear/justify_questions.json",
      "role": "asset",
      "oneLiner": "Conceptual Q&A pairs with model answers for post-build assessment of the linear layer module."
    },
    {
      "path": "curricula/cs336_a1/modules/linear/validator.sh",
      "role": "source",
      "oneLiner": "Shell script that runs pytest tests/test_model.py::test_linear in a shadow worktree to validate the Linear layer implementation."
    },
    {
      "path": "curricula/cs336_a1/modules/multihead_attention/bugs/missing_transpose_back.json",
      "role": "config",
      "oneLiner": "Production AST bug-injection spec for the missing transpose-back operation in multihead self-attention output reshaping."
    },
    {
      "path": "curricula/cs336_a1/modules/multihead_attention/bugs/missing_transpose_back.patch",
      "role": "asset",
      "oneLiner": "Unified diff showing the missing transpose_back bug applied to the multihead attention implementation in cs336_basics/layers.py."
    },
    {
      "path": "curricula/cs336_a1/modules/multihead_attention/bugs/missing_transpose_back_draft.json",
      "role": "config",
      "oneLiner": "Draft AST bug-injection spec for the missing transpose_back bug in multihead attention; predecessor to the production spec."
    },
    {
      "path": "curricula/cs336_a1/modules/multihead_attention/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Student-facing instructional guide for implementing multi-head self-attention with RoPE positional encoding."
    },
    {
      "path": "curricula/cs336_a1/modules/multihead_attention/justify_questions.json",
      "role": "asset",
      "oneLiner": "Conceptual Q&A pairs with model answers for post-build assessment of the multihead attention module."
    },
    {
      "path": "curricula/cs336_a1/modules/multihead_attention/validator.sh",
      "role": "source",
      "oneLiner": "Shell script that runs pytest tests/test_model.py::test_multihead_self_attention_with_rope in a shadow worktree to validate multihead attention."
    },
    {
      "path": "curricula/cs336_a1/modules/rmsnorm/bugs/missing_keepdim.json",
      "role": "config",
      "oneLiner": "Production AST bug-injection spec that removes keepdim=True from the mean computation in RMSNorm, causing a broadcasting shape error."
    },
    {
      "path": "curricula/cs336_a1/modules/rmsnorm/bugs/missing_keepdim.patch",
      "role": "asset",
      "oneLiner": "Unified diff showing the missing keepdim=True bug applied to the RMSNorm layer in cs336_basics/layers.py."
    },
    {
      "path": "curricula/cs336_a1/modules/rmsnorm/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Student-facing instructional guide for implementing the RMSNorm normalization layer used in modern LLMs."
    },
    {
      "path": "curricula/cs336_a1/modules/rmsnorm/justify_questions.json",
      "role": "asset",
      "oneLiner": "Conceptual Q&A pairs with model answers for post-build assessment of the RMSNorm module."
    },
    {
      "path": "curricula/cs336_a1/modules/rmsnorm/validator.sh",
      "role": "source",
      "oneLiner": "Shell script that runs pytest tests/test_model.py::test_rmsnorm in a shadow worktree to validate the RMSNorm implementation."
    },
    {
      "path": "curricula/cs336_a1/modules/rope/bugs/wrong_rotation.json",
      "role": "config",
      "oneLiner": "Production AST bug-injection spec for injecting a wrong rotation bug into the RoPE positional encoding implementation."
    },
    {
      "path": "curricula/cs336_a1/modules/rope/bugs/wrong_rotation.patch",
      "role": "asset",
      "oneLiner": "Unified diff showing the wrong rotation bug applied to the RoPE positional encoding in cs336_basics/layers.py."
    },
    {
      "path": "curricula/cs336_a1/modules/rope/bugs/wrong_rotation_draft.json",
      "role": "config",
      "oneLiner": "Draft AST bug-injection spec for the wrong rotation bug in RoPE; predecessor to the production spec."
    },
    {
      "path": "curricula/cs336_a1/modules/rope/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Student-facing instructional guide for implementing Rotary Positional Encoding (RoPE) for Transformer attention."
    },
    {
      "path": "curricula/cs336_a1/modules/rope/justify_questions.json",
      "role": "asset",
      "oneLiner": "Conceptual Q&A pairs with model answers for post-build assessment of the RoPE module."
    },
    {
      "path": "curricula/cs336_a1/modules/rope/validator.sh",
      "role": "source",
      "oneLiner": "Shell script that runs pytest tests/test_model.py::test_rope in a shadow worktree to validate the RoPE implementation."
    },
    {
      "path": "curricula/cs336_a1/modules/silu/bugs/missing_multiply.json",
      "role": "config",
      "oneLiner": "Production AST bug-injection spec that removes the element-wise multiply in the SiLU activation, reducing it to identity or sigmoid only."
    },
    {
      "path": "curricula/cs336_a1/modules/silu/bugs/missing_multiply.patch",
      "role": "asset",
      "oneLiner": "Unified diff showing the missing multiply bug applied to the SiLU activation function in cs336_basics/layers.py."
    },
    {
      "path": "curricula/cs336_a1/modules/silu/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Student-facing instructional guide for implementing the SiLU (Swish) activation function x * sigmoid(x)."
    },
    {
      "path": "curricula/cs336_a1/modules/silu/justify_questions.json",
      "role": "asset",
      "oneLiner": "Conceptual Q&A pairs with model answers for post-build assessment of the SiLU module."
    },
    {
      "path": "curricula/cs336_a1/modules/silu/validator.sh",
      "role": "source",
      "oneLiner": "Shell script that runs pytest tests/test_model.py::test_silu_matches_pytorch in a shadow worktree to validate the SiLU implementation."
    },
    {
      "path": "curricula/cs336_a1/modules/softmax/bugs/no_subtract_max.json",
      "role": "config",
      "oneLiner": "Production AST bug-injection spec (two-pass) that removes the subtract-max numerical stability trick from the softmax implementation."
    },
    {
      "path": "curricula/cs336_a1/modules/softmax/bugs/no_subtract_max.patch",
      "role": "asset",
      "oneLiner": "Unified diff showing the no-subtract-max bug applied to the softmax function, exposing it to numerical overflow."
    },
    {
      "path": "curricula/cs336_a1/modules/softmax/bugs/no_subtract_max_symptom.txt",
      "role": "doc",
      "oneLiner": "Student-facing description of NaN overflow symptoms from the missing subtract-max trick, with failing test case and fix guidance."
    },
    {
      "path": "curricula/cs336_a1/modules/softmax/bugs/no_subtract_max_v2.json",
      "role": "config",
      "oneLiner": "Second version of the AST spec for removing subtract-max from softmax, using an alternative two-pass find_and_track then find_and_replace pattern."
    },
    {
      "path": "curricula/cs336_a1/modules/softmax/bugs/no_subtract_max_v2_symptom.txt",
      "role": "doc",
      "oneLiner": "Student-facing symptom description for the v2 no-subtract-max softmax bug variant, matching the v2 injection spec."
    },
    {
      "path": "curricula/cs336_a1/modules/softmax/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Student-facing instructional guide for implementing numerically stable softmax using the subtract-max trick."
    },
    {
      "path": "curricula/cs336_a1/modules/softmax/justify_questions.json",
      "role": "asset",
      "oneLiner": "Conceptual Q&A pairs with model answers for post-build assessment of the numerically stable softmax module."
    },
    {
      "path": "curricula/cs336_a1/modules/softmax/validator.sh",
      "role": "source",
      "oneLiner": "Shell script that runs pytest tests/test_nn_utils.py::test_softmax_matches_pytorch in a shadow worktree to validate the softmax implementation."
    },
    {
      "path": "curricula/cs336_a1/modules/swiglu/bugs/missing_gate.json",
      "role": "config",
      "oneLiner": "Production AST bug-injection spec that removes the gate computation from the SwiGLU gated activation forward pass."
    },
    {
      "path": "curricula/cs336_a1/modules/swiglu/bugs/missing_gate.patch",
      "role": "asset",
      "oneLiner": "Unified diff showing the missing gate bug applied to the SwiGLU activation in cs336_basics/layers.py."
    },
    {
      "path": "curricula/cs336_a1/modules/swiglu/bugs/missing_gate_draft.json",
      "role": "config",
      "oneLiner": "First draft AST bug-injection spec for the missing gate bug in SwiGLU; predecessor to the production spec."
    },
    {
      "path": "curricula/cs336_a1/modules/swiglu/bugs/missing_gate_draft_v2.json",
      "role": "config",
      "oneLiner": "Second draft (v2.1) AST bug-injection spec for the missing gate bug in SwiGLU; intermediate version between draft and production."
    },
    {
      "path": "curricula/cs336_a1/modules/swiglu/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Student-facing instructional guide for implementing the SwiGLU gated activation function used in modern LLM feed-forward blocks."
    },
    {
      "path": "curricula/cs336_a1/modules/swiglu/justify_questions.json",
      "role": "asset",
      "oneLiner": "Conceptual Q&A pairs with model answers for post-build assessment of the SwiGLU module."
    },
    {
      "path": "curricula/cs336_a1/modules/swiglu/validator.sh",
      "role": "source",
      "oneLiner": "Shell script that runs pytest tests/test_model.py::test_swiglu in a shadow worktree to validate the SwiGLU implementation."
    },
    {
      "path": "curricula/cs336_a1/modules/text_generation/bugs/temperature_after_softmax.json",
      "role": "config",
      "oneLiner": "Production AST bug-injection spec that misplaces temperature scaling to after softmax instead of before it in text generation."
    },
    {
      "path": "curricula/cs336_a1/modules/text_generation/bugs/temperature_after_softmax.patch",
      "role": "asset",
      "oneLiner": "Unified diff showing the temperature_after_softmax bug applied to the text generation function."
    },
    {
      "path": "curricula/cs336_a1/modules/text_generation/bugs/temperature_after_softmax_draft.json",
      "role": "config",
      "oneLiner": "Draft AST bug-injection spec for the temperature_after_softmax bug; predecessor to the production spec."
    },
    {
      "path": "curricula/cs336_a1/modules/text_generation/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Student-facing instructional guide for implementing temperature-based autoregressive text generation."
    },
    {
      "path": "curricula/cs336_a1/modules/text_generation/justify_questions.json",
      "role": "asset",
      "oneLiner": "Conceptual Q&A pairs with model answers for post-build assessment of the text generation module."
    },
    {
      "path": "curricula/cs336_a1/modules/text_generation/validator.sh",
      "role": "source",
      "oneLiner": "Shell script that runs pytest tests/test_generation.py::test_generate in a shadow worktree to validate the text generation implementation."
    },
    {
      "path": "curricula/cs336_a1/modules/tokenizer_class/bugs/wrong_merge_order.json",
      "role": "config",
      "oneLiner": "Production AST bug-injection spec for the wrong merge order bug in the BPE tokenizer class encode/merge logic."
    },
    {
      "path": "curricula/cs336_a1/modules/tokenizer_class/bugs/wrong_merge_order.patch",
      "role": "asset",
      "oneLiner": "Unified diff showing the wrong merge order bug applied to the BPE Tokenizer class in cs336_basics/."
    },
    {
      "path": "curricula/cs336_a1/modules/tokenizer_class/bugs/wrong_merge_order_draft.json",
      "role": "config",
      "oneLiner": "Draft AST bug-injection spec for the wrong merge order bug in the tokenizer class; predecessor to the production spec."
    },
    {
      "path": "curricula/cs336_a1/modules/tokenizer_class/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Student-facing instructional guide for implementing the BPE Tokenizer class with encode, decode, and train_from_iterator methods."
    },
    {
      "path": "curricula/cs336_a1/modules/tokenizer_class/justify_questions.json",
      "role": "asset",
      "oneLiner": "Conceptual Q&A pairs with model answers for post-build assessment of the BPE tokenizer class module."
    },
    {
      "path": "curricula/cs336_a1/modules/tokenizer_class/validator.sh",
      "role": "source",
      "oneLiner": "Shell script that runs pytest tests/test_tokenizer.py::test_tokenizer_class in a shadow worktree to validate the Tokenizer class implementation."
    },
    {
      "path": "curricula/cs336_a1/modules/training_loop/bugs/missing_zero_grad.json",
      "role": "config",
      "oneLiner": "Finalized AST bug-injection spec that deletes the optimizer.zero_grad() call in the training loop (engine_version 2.1)."
    },
    {
      "path": "curricula/cs336_a1/modules/training_loop/bugs/missing_zero_grad.patch",
      "role": "asset",
      "oneLiner": "Unified diff that removes optimizer.zero_grad() and adds an explanatory bug comment in cs336_basics/training.py."
    },
    {
      "path": "curricula/cs336_a1/modules/training_loop/bugs/missing_zero_grad_draft.json",
      "role": "config",
      "oneLiner": "Draft v1 AST bug-injection spec (pass_ key, extra null fields) for deleting the zero_grad call in training_loop."
    },
    {
      "path": "curricula/cs336_a1/modules/training_loop/bugs/missing_zero_grad_draft_v2.json",
      "role": "config",
      "oneLiner": "Draft v2 AST bug-injection spec (auto_fixed metadata, engine_version 2.1) for deleting zero_grad in train_step."
    },
    {
      "path": "curricula/cs336_a1/modules/training_loop/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Student-facing instructional prompt explaining the full training loop (forward, backward, clip, step) for cs336_a1."
    },
    {
      "path": "curricula/cs336_a1/modules/training_loop/justify_questions.json",
      "role": "config",
      "oneLiner": "Five Q&A assessment items with model answers testing conceptual understanding of the PyTorch training loop."
    },
    {
      "path": "curricula/cs336_a1/modules/training_loop/validator.sh",
      "role": "test",
      "oneLiner": "Shell validator that copies training.py to a shadow worktree and runs pytest tests/test_training.py::test_train_loop."
    },
    {
      "path": "curricula/cs336_a1/modules/transformer_block/bugs/missing_residual.json",
      "role": "config",
      "oneLiner": "Finalized AST bug-injection spec that replaces x = x + attn_out with x = attn_out to remove the residual connection."
    },
    {
      "path": "curricula/cs336_a1/modules/transformer_block/bugs/missing_residual.patch",
      "role": "asset",
      "oneLiner": "Unified diff that drops the residual addition in TransformerBlock.forward() and adds an explanatory bug comment."
    },
    {
      "path": "curricula/cs336_a1/modules/transformer_block/bugs/missing_residual_draft.json",
      "role": "config",
      "oneLiner": "Draft v1 AST bug-injection spec (pass_ key, extra null fields) for removing the residual connection in transformer_block."
    },
    {
      "path": "curricula/cs336_a1/modules/transformer_block/bugs/missing_residual_draft_v2.json",
      "role": "config",
      "oneLiner": "Draft v2 AST bug-injection spec (auto_fixed metadata) for replacing x = x + attn_out with x = attn_out."
    },
    {
      "path": "curricula/cs336_a1/modules/transformer_block/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Student-facing instructional prompt for implementing a complete pre-norm Transformer block with RMSNorm, attention, and SwiGLU."
    },
    {
      "path": "curricula/cs336_a1/modules/transformer_block/justify_questions.json",
      "role": "config",
      "oneLiner": "Q&A assessment items with model answers testing understanding of residual connections, pre-norm vs post-norm, and attention."
    },
    {
      "path": "curricula/cs336_a1/modules/transformer_block/validator.sh",
      "role": "test",
      "oneLiner": "Shell validator that copies layers.py to a shadow worktree and runs pytest tests/test_model.py::test_transformer_block."
    },
    {
      "path": "curricula/cs336_a1/modules/transformer_lm/bugs/missing_final_norm.json",
      "role": "config",
      "oneLiner": "Finalized AST bug-injection spec that deletes the ln_final RMSNorm application before the LM head."
    },
    {
      "path": "curricula/cs336_a1/modules/transformer_lm/bugs/missing_final_norm.patch",
      "role": "asset",
      "oneLiner": "Unified diff that removes the final RMSNorm block in transformer_lm and replaces it with an explanatory bug comment."
    },
    {
      "path": "curricula/cs336_a1/modules/transformer_lm/bugs/missing_final_norm_draft.json",
      "role": "config",
      "oneLiner": "Misidentified draft AST spec (id: silu-missing-multiply) that removes x*sigmoid(x) multiplication; likely a wrong-module draft."
    },
    {
      "path": "curricula/cs336_a1/modules/transformer_lm/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Student-facing instructional prompt for assembling the full TransformerLM with embedding, stacked blocks, and final norm."
    },
    {
      "path": "curricula/cs336_a1/modules/transformer_lm/justify_questions.json",
      "role": "config",
      "oneLiner": "Q&A assessment items with model answers on autoregressive target shifting, weight tying, and final norm necessity."
    },
    {
      "path": "curricula/cs336_a1/modules/transformer_lm/validator.sh",
      "role": "test",
      "oneLiner": "Shell validator that copies layers.py to a shadow worktree and runs pytest tests/test_model.py::test_transformer_lm."
    },
    {
      "path": "curricula/cs336_a1/modules/unicode/README.md",
      "role": "doc",
      "oneLiner": "Explains that the unicode module is theory-only (justify stage only), covering UTF-8 encoding, normalization, and grapheme clusters."
    },
    {
      "path": "curricula/cs336_a1/modules/unicode/justify_questions.json",
      "role": "config",
      "oneLiner": "Five comprehensive Q&A items with model answers covering UTF-8 variable-length encoding, normalization, and grapheme clusters."
    },
    {
      "path": "curricula/dummy_hello_world/manifest.json",
      "role": "config",
      "oneLiner": "Curriculum manifest for dummy_hello_world defining the single hello_world module with baseline_perf_seconds=0.001."
    },
    {
      "path": "curricula/dummy_hello_world/modules/hello_world/bugs/typo.patch",
      "role": "asset",
      "oneLiner": "Unified diff that introduces a typo ('Enginne') in greet()'s return string to demonstrate the Harden stage."
    },
    {
      "path": "curricula/dummy_hello_world/modules/hello_world/bugs/typo_symptom.txt",
      "role": "doc",
      "oneLiner": "Student-facing description of the typo bug symptom and instructions to fix the spelling in hello_world.py."
    },
    {
      "path": "curricula/dummy_hello_world/modules/hello_world/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Student-facing build challenge to create a hello_world.py function returning 'Hello, Mastery Engine!'."
    },
    {
      "path": "curricula/dummy_hello_world/modules/hello_world/justify_questions.json",
      "role": "config",
      "oneLiner": "Single justify Q&A item asking why the implementation uses a function rather than a direct print statement."
    },
    {
      "path": "curricula/dummy_hello_world/modules/hello_world/validator.sh",
      "role": "test",
      "oneLiner": "Shell validator that checks for the existence of hello_world.py in the workspace and exits 0 if found."
    },
    {
      "path": "curricula/job_prep_data_annotation/README.md",
      "role": "doc",
      "oneLiner": "Overview of the job_prep_data_annotation curriculum: three modules teaching HTTP, HTML parsing, and 2D grids for DataAnnotation assessments."
    },
    {
      "path": "curricula/job_prep_data_annotation/manifest.json",
      "role": "config",
      "oneLiner": "Curriculum manifest defining http_transport, data_parsing_extraction, and grid_visualization modules with dependencies and metadata."
    },
    {
      "path": "curricula/job_prep_data_annotation/modules/data_parsing_extraction/bugs/fragile_split.patch",
      "role": "asset",
      "oneLiner": "Unified diff replacing BeautifulSoup+regex coordinate parser with a brittle split()-based parser that breaks on whitespace."
    },
    {
      "path": "curricula/job_prep_data_annotation/modules/data_parsing_extraction/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Student-facing build challenge to implement extract_coordinates() using BeautifulSoup and regex for robust HTML parsing."
    },
    {
      "path": "curricula/job_prep_data_annotation/modules/data_parsing_extraction/justify_questions.json",
      "role": "config",
      "oneLiner": "Q&A assessment items on why split() is brittle for HTML parsing and how regex handles whitespace variations."
    },
    {
      "path": "curricula/job_prep_data_annotation/modules/data_parsing_extraction/validator.sh",
      "role": "test",
      "oneLiner": "Shell validator with embedded Python test suite for extract_coordinates() covering consistent/inconsistent formats and performance."
    },
    {
      "path": "curricula/job_prep_data_annotation/modules/grid_visualization/bugs/reference_copying.patch",
      "role": "asset",
      "oneLiner": "Unified diff replacing the list-comprehension 2D grid init with aliased [[' ']*width]*height to introduce the reference-copying bug."
    },
    {
      "path": "curricula/job_prep_data_annotation/modules/grid_visualization/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Student-facing build challenge to implement render_grid() converting sparse coordinates to a dense 2D list without aliasing."
    },
    {
      "path": "curricula/job_prep_data_annotation/modules/grid_visualization/justify_questions.json",
      "role": "config",
      "oneLiner": "Q&A assessment items on Python's shallow vs deep copy semantics and why [[' ']*w]*h creates aliased rows."
    },
    {
      "path": "curricula/job_prep_data_annotation/modules/grid_visualization/validator.sh",
      "role": "test",
      "oneLiner": "Shell validator with embedded Python test suite for render_grid() covering sparse grids, reference-copying detection, and performance."
    },
    {
      "path": "curricula/job_prep_data_annotation/modules/http_transport/bugs/open_trap.patch",
      "role": "asset",
      "oneLiner": "Unified diff replacing requests.get() with open() in fetch_document() to simulate the file-vs-network I/O confusion bug."
    },
    {
      "path": "curricula/job_prep_data_annotation/modules/http_transport/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Student-facing build challenge to implement fetch_document() using requests.get() with proper status-code error handling."
    },
    {
      "path": "curricula/job_prep_data_annotation/modules/http_transport/justify_questions.json",
      "role": "config",
      "oneLiner": "Q&A assessment items on why open() fails for URLs and the fundamental distinction between file I/O and network I/O."
    },
    {
      "path": "curricula/job_prep_data_annotation/modules/http_transport/validator.sh",
      "role": "test",
      "oneLiner": "Shell validator with embedded Python test suite for fetch_document() testing HTTP GET, error handling, and status codes."
    },
    {
      "path": "curricula/python_for_cp/manifest.json",
      "role": "config",
      "oneLiner": "Curriculum manifest for python_for_cp defining pythonic_structures, concise_logic, and std_lib_augmentation modules."
    },
    {
      "path": "curricula/python_for_cp/modules/std_lib_augmentation/bugs/list_pop_performance.patch",
      "role": "asset",
      "oneLiner": "Unified diff replacing deque.popleft() with list.pop(0) in BFS to introduce O(n) per-operation performance regression."
    },
    {
      "path": "curricula/python_for_cp/modules/std_lib_augmentation/bugs/missing_visited_set.patch",
      "role": "asset",
      "oneLiner": "Unified diff removing the visited set from Dijkstra, causing nodes to be processed multiple times in the heap."
    },
    {
      "path": "curricula/python_for_cp/modules/std_lib_augmentation/build_prompt.txt",
      "role": "doc",
      "oneLiner": "Student-facing build challenge to implement BFS with deque, Dijkstra with heapq, and range-count with bisect."
    },
    {
      "path": "curricula/python_for_cp/modules/std_lib_augmentation/justify_questions.json",
      "role": "config",
      "oneLiner": "Q&A assessment items on deque vs list complexity, Dijkstra visited-set necessity, and bisect binary search use cases."
    },
    {
      "path": "curricula/python_for_cp/modules/std_lib_augmentation/validator.sh",
      "role": "test",
      "oneLiner": "Shell validator with embedded Python test suite for shortest_path_bfs, dijkstra_shortest_path, and count_in_range functions."
    },
    {
      "path": "docs/INDEX.md",
      "role": "doc",
      "oneLiner": "Navigation index listing all current, architecture, and internal Mastery Engine documentation with audience labels."
    },
    {
      "path": "docs/README.md",
      "role": "doc",
      "oneLiner": "Top-level documentation overview describing the three engineering pillars and directing users/contributors to the right sub-docs."
    },
    {
      "path": "docs/STRANGER_TEST_RESULTS.md",
      "role": "doc",
      "oneLiner": "End-to-end verification report from a clean-slate README Quick Start test on Nov 19 2025, finding and fixing 3 critical bugs."
    },
    {
      "path": "docs/architecture/AI_CODEBASE_DECONSTRUCTION.md",
      "role": "doc",
      "oneLiner": "Design analysis (blueprint, not shipped) for applying the Mastery Engine to force comprehension of AI-generated codebases."
    },
    {
      "path": "docs/architecture/MASTERY_ENGINE.md",
      "role": "doc",
      "oneLiner": "Comprehensive v5.0 technical blueprint of the Mastery Engine's Build-Justify-Harden pedagogy and layered architecture."
    },
    {
      "path": "docs/architecture/REPO_ANALYSIS.md",
      "role": "doc",
      "oneLiner": "Auto-generated repository analysis for CS336 Assignment 1, covering structure, tests, and implementation contracts."
    },
    {
      "path": "docs/internal/CLEANUP_SUMMARY.md",
      "role": "doc",
      "oneLiner": "Session log summarising the Nov 17 2025 docs cleanup that reduced root .md files from 71 to 2 and added clear hierarchy."
    },
    {
      "path": "docs/internal/CP_ACCELERATOR_QUICKSTART.md",
      "role": "doc",
      "oneLiner": "Quick-start guide and rationale for the CP Accelerator canonical-source-of-truth architecture using expert-curated JSON."
    },
    {
      "path": "docs/internal/CP_SOURCE_VERIFICATION.md",
      "role": "doc",
      "oneLiner": "Verification report documenting the fix of placeholder URLs and titles in canonical_curriculum.json with real parsed data."
    },
    {
      "path": "docs/internal/CRITICAL_REVIEW_RESPONSE.md",
      "role": "doc",
      "oneLiner": "Response to pre-deployment critical review documenting mitigations for BeautifulSoup fluency gap and one other risk."
    },
    {
      "path": "docs/internal/DOCS_CLEANUP_2025-11-18.md",
      "role": "doc",
      "oneLiner": "Change log for Nov 18 2025 reorganisation of 12 scattered docs into two_sum_qa/ and module_generation/ subdirectories."
    },
    {
      "path": "docs/internal/ENGINE_CRITICAL_FIXES_2025-11-18.md",
      "role": "doc",
      "oneLiner": "Post-mortem and fix log for three critical engine bugs (wrong test cases, path fragility, init idempotency) found in session logs."
    },
    {
      "path": "docs/internal/PHASE_8_BATCH_GENERATION_COMPLETE.md",
      "role": "doc",
      "oneLiner": "Completion report for Phase 8 breadth-first content population populating 2 problems per pattern across the 959-problem CP taxonomy."
    },
    {
      "path": "docs/internal/PUBLIC_RELEASE_COMPLETE.md",
      "role": "doc",
      "oneLiner": "Release engineering completion report (commit d694ff3) transforming the repo from course assignment to public portfolio piece."
    },
    {
      "path": "docs/internal/PYTHON_CURRICULA_IMPLEMENTATION.md",
      "role": "doc",
      "oneLiner": "Implementation report for two new linear Python curricula (job_prep_data_annotation and one other) targeting DataAnnotation skill gaps."
    },
    {
      "path": "docs/internal/README.md",
      "role": "doc",
      "oneLiner": "Index and purpose statement for the internal/ directory, noting these are journey artifacts not required for product use."
    },
    {
      "path": "docs/internal/REAL_CLI_TRANSFORMATION.md",
      "role": "doc",
      "oneLiner": "Fix log addressing three issues that made the Mastery Engine feel like a script rather than a real CLI tool like git or npm."
    },
    {
      "path": "docs/internal/archive/README.md",
      "role": "doc",
      "oneLiner": "Overview and navigation guide for the historical archive directory, directing readers to current docs for up-to-date information."
    },
    {
      "path": "docs/internal/archive/deprecated/BATCH_MIGRATION_GUIDE.md",
      "role": "doc",
      "oneLiner": "Deprecated execution guide for batch migration of legacy .patch bug files to v2.1 JSON format with pre-flight checklist."
    },
    {
      "path": "docs/internal/archive/deprecated/EXPERIMENT_MODULE_DESIGN.md",
      "role": "doc",
      "oneLiner": "Deprecated design document for an experimental-investigation module type extending BJH with scientific hypothesis and ablation structure."
    },
    {
      "path": "docs/internal/archive/deprecated/JUSTIFY_ONLY_MODULE_DESIGN.md",
      "role": "doc",
      "oneLiner": "Deprecated design document for a theory-only justify module type assessing conceptual understanding without requiring implementation."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-08_systematic_improvements/SYSTEMATIC_FIXING_SESSION.md",
      "role": "doc",
      "oneLiner": "Session report on systematic analysis of multi-attempt patterns to diagnose and eliminate first-attempt failure root causes."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-08_systematic_improvements/SYSTEMATIC_IMPROVEMENT_FINAL.md",
      "role": "doc",
      "oneLiner": "Final ~4.5-hour session summary confirming all 4 training examples work correctly and next bottleneck (patch extraction) identified."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-08_systematic_improvements/SYSTEMATIC_IMPROVEMENT_SESSION.md",
      "role": "doc",
      "oneLiner": "Progress report for Session 1 adding regression checks, manual LLM analysis, and permanent evaluation script improvements."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-08_systematic_improvements/SYSTEMATIC_IMPROVEMENT_SESSION_2.md",
      "role": "doc",
      "oneLiner": "Completion report for Session 2 using manual analysis to surface issues that statistics missed, diagnosing P0/P1 bottlenecks."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-08_systematic_improvements/SYSTEMATIC_SESSION_FINAL.md",
      "role": "doc",
      "oneLiner": "Complete ~7-hour systematic session analysis confirming 4/4 training data correctness after evidence-based fixes."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-09_verification/FINAL_VERIFICATION_SUMMARY.md",
      "role": "doc",
      "oneLiner": "Final pre-launch verification summary declaring Mastery Engine v1.0 production-ready with 197 passing automated tests."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-09_verification/LAYER2_E2E_SUCCESS.md",
      "role": "doc",
      "oneLiner": "E2E test fix completion report confirming the full BJH loop test passes in 16.51s after shadow worktree symlink fix."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-09_verification/LAYER4_UAT_EXECUTION_GUIDE.md",
      "role": "doc",
      "oneLiner": "Layer 4 UAT execution guide for the Student Zero Gauntlet clean-slate test (later self-invalidated due to methodology flaws)."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-09_verification/LAYER4_UAT_FINDINGS.md",
      "role": "doc",
      "oneLiner": "Invalidated UAT findings report: tester copied from developer mode instead of implementing stubs, rendering all results invalid."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-09_verification/REAL_STUDENT_UAT_MODULE1.md",
      "role": "doc",
      "oneLiner": "Real student UAT report for Module 1 (softmax) build stage using a genuine clean-slate setup with rm of all prior state."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-09_verification/VERIFICATION_PROTOCOL_FINAL_STATUS.md",
      "role": "doc",
      "oneLiner": "Verification Protocol v3.0 pre-launch status report showing 93% foundation complete and cleared for Layer 4 UAT."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-09_verification/VERIFICATION_PROTOCOL_FINAL_STATUS_V3.md",
      "role": "doc",
      "oneLiner": "5-hour verification session completion report with Layers 1-3 fully done, Layer 4 partial, rated exceptional quality."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-09_verification/VERIFICATION_PROTOCOL_LAYER1_STATUS.md",
      "role": "doc",
      "oneLiner": "Layer 1 static verification status report showing 135/145 engine tests passing with 10 minor failures listed."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-09_verification/VERIFICATION_PROTOCOL_LAYER2_COMPLETE.md",
      "role": "doc",
      "oneLiner": "Layer 2 completion report documenting the shadow worktree symlink fix that enabled successful E2E Build stage validation."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-09_verification/VERIFICATION_PROTOCOL_LAYER2_STATUS.md",
      "role": "doc",
      "oneLiner": "Layer 2 partial-status report showing test infrastructure updated but validation blocked by pytest configuration issues."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-09_verification/VERIFICATION_PROTOCOL_LAYER3_COMPLETE.md",
      "role": "doc",
      "oneLiner": "Layer 3 completion report adding multi-module progression validation and 7 adversarial stress tests to the regression suite."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-09_verification/VERIFICATION_PROTOCOL_LAYERS_1_2_COMPLETE.md",
      "role": "doc",
      "oneLiner": "Combined Layers 1-2 completion report confirming production-ready foundation with full E2E BJH loop automated coverage."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-10_bug_system/AST_HARDEN_PHASE2_COMPLETE.md",
      "role": "doc",
      "oneLiner": "Phase 2 completion report integrating AST-based bug injection into production HardenRunner for the softmax module."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-10_bug_system/AST_HARDEN_PHASE2_FINAL.md",
      "role": "doc",
      "oneLiner": "Phase 2 final verification confirming end-to-end AST bug injection with student variable name preservation battle-tested."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-10_bug_system/BOTTLENECK_DIAGNOSIS.md",
      "role": "doc",
      "oneLiner": "Manual analysis report identifying wrong replacement strategy as the bottleneck causing 0% LLM success despite 95.8% node accuracy."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-10_bug_system/COMPLETE_SUCCESS_SUMMARY.md",
      "role": "doc",
      "oneLiner": "Summary confirming all systematic improvements implemented, with AdamW bias-correction bug injection producing correct buggy code."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-10_bug_system/CRITICAL_BUG_RESOLUTION.md",
      "role": "doc",
      "oneLiner": "Critical bug resolution report for student mode containing complete implementations in 10 of 22 modules, enabling bypassing of validation."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-10_bug_system/EVALUATION_FIXED_ANALYSIS.md",
      "role": "doc",
      "oneLiner": "Analysis showing evaluation success improved from 0% to 50% by switching from text comparison to AST-based functional comparison."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-10_bug_system/GPT4O_TEST_RESULTS.md",
      "role": "doc",
      "oneLiner": "Test results comparing gpt-4o vs gpt-4o-mini for bug authoring, showing 100% improvement in first-try success (25% to 50%)."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-10_bug_system/HARDEN_FIX_VERIFICATION.md",
      "role": "doc",
      "oneLiner": "Verification report confirming the fatal harden.py flaw (copying student code instead of reference for patching) was fixed."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-10_bug_system/HARDEN_STAGE_CRITICAL_BUG.md",
      "role": "doc",
      "oneLiner": "Critical bug report identifying the fatal harden.py architectural flaw where student code was patched instead of the reference."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-10_bug_system/LLM_PROMPT_REVIEW.md",
      "role": "doc",
      "oneLiner": "Review of the LLM prompt structure (system + user template) used for evaluating student justify-stage answers in JSON format."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-10_bug_system/LLM_TOOL_DIAGNOSTIC_ANALYSIS.md",
      "role": "doc",
      "oneLiner": "Comprehensive diagnostic analysis of LLM bug generation failures, establishing a 4-bug golden dataset and validated transformation types."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-10_bug_system/MANUAL_LLM_TEST.md",
      "role": "doc",
      "oneLiner": "Manual test procedure guide for one-time live OpenAI API integration validation of the Justify stage (~$0.01 cost)."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-10_bug_system/NEXT_BOTTLENECK_IDENTIFIED.md",
      "role": "doc",
      "oneLiner": "Manual analysis report pinpointing the silu wrong-replacement-strategy bottleneck after statistics showed 0% unknown failures."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-10_bug_system/PATTERN_MATCHER_DEBUG_SESSION.md",
      "role": "doc",
      "oneLiner": "3-hour debug session log resolving multiple AST pattern matching bugs including canonical variable renaming and indentation issues."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-10_bug_system/PHASE2_SIGNOFF.md",
      "role": "doc",
      "oneLiner": "Formal sign-off approving Phase 2 AST-based bug injection for production use, rated exceptional quality."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-10_bug_system/PHASE3_COMPLETION_REPORT.md",
      "role": "doc",
      "oneLiner": "Phase 3 completion report confirming the generic data-driven JSON bug engine replaced hardcoded AST injection, validated on 3 bug types."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-10_bug_system/PHASE3_IMPLEMENTATION_PLAN.md",
      "role": "doc",
      "oneLiner": "Incremental test-driven implementation plan for Phase 3 generalization of SoftmaxBugInjector into a JSON-driven engine."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-10_bug_system/PHASE4_FINAL_SIGNOFF.md",
      "role": "doc",
      "oneLiner": "Phase 4 final sign-off approving LLM-powered bug authoring tool with 83% bug-creation time reduction (13h to 2.3h for 17 bugs)."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-10_bug_system/PHASE4_LLM_TOOL.md",
      "role": "doc",
      "oneLiner": "Design and architecture document for Phase 4 LLM-powered bug authoring tool converting legacy .patch files to v2.1 JSON format."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-10_bug_system/SESSION_COMPLETE_SUMMARY.md",
      "role": "doc",
      "oneLiner": "Complete session summary confirming all systematic improvement requirements met, with silu bottleneck diagnosed as the next P0 issue."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-10_bug_system/TRAINING_DATA_VALIDATION.md",
      "role": "doc",
      "oneLiner": "Validation report confirming all 4/4 golden training examples (silu, attention, rmsnorm, adamw) produce correct buggy code."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/BPE_TEST_FIX_SUMMARY.md",
      "role": "doc",
      "oneLiner": "Fix summary for BPE test that required exact merge order matching, causing false failures due to tie-breaking ambiguity in BPE algorithms."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/COMPREHENSIVE_21_MODULE_EVALUATION.md",
      "role": "doc",
      "oneLiner": "Evaluation report for all 21 curriculum modules using gpt-4o, finding 10/21 (48%) actual success including 5 false-negative corrections."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/COMPREHENSIVE_FIX_SUMMARY.md",
      "role": "doc",
      "oneLiner": "Session summary confirming all blockers fixed across 21 modules, with 3 critical bugs resolved via manual and statistical analysis."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/COMPREHENSIVE_REMEDIATION_SUMMARY.md",
      "role": "doc",
      "oneLiner": "Multi-session (~20h) remediation summary spanning curriculum quality (98/100) and CLI interface improvements across 3 sessions."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/CP_ACCELERATOR_IMPLEMENTATION_GUIDE.md",
      "role": "doc",
      "oneLiner": "Implementation guide for the CP Accelerator curriculum pack synthesising DSA Pattern Taxonomy and CP Roadmap into a BJH curriculum."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/CP_ACCELERATOR_QUICKSTART.md",
      "role": "doc",
      "oneLiner": "Quick-start guide for CP Accelerator describing rating-driven progression from 0 to 1900+ through 19 algorithmic patterns."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/CURRICULUM_COVERAGE.md",
      "role": "doc",
      "oneLiner": "Coverage map proving 100% alignment of all 21 curriculum modules to CS336 Assignment 1 PDF spec with from-scratch ethos."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/CURRICULUM_GAP_ANALYSIS.md",
      "role": "doc",
      "oneLiner": "Gap analysis identifying only 3 of ~19 CS336 components initially covered (16%) and listing all unimplemented modules."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/EINOPS_VIOLATIONS_AUDIT.md",
      "role": "doc",
      "oneLiner": "Audit finding 5 einops violations in multihead_self_attention reference implementation contrary to CS336 PDF §3.3 requirement."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/GROUND_TRUTH_COMPLETE.md",
      "role": "doc",
      "oneLiner": "Completion report for creating and validating 21 AST-based bug.json golden patterns covering 100% of curriculum modules."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/LITERATURE_VERIFICATION.md",
      "role": "doc",
      "oneLiner": "Literature verification guide mapping each curriculum module to ground-truth CS paper sources for pedagogical claim validation."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/MASTER_REMEDIATION_STATUS.md",
      "role": "doc",
      "oneLiner": "Archived status summary tracking curriculum (98/100) and CLI P0 (100%) remediation progress across three work sessions."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/PROJECT_STATUS.md",
      "role": "doc",
      "oneLiner": "Archived project status report declaring Phases 1-4 complete and production-ready for the AST bug injection engine."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/QUALITY_REMEDIATION_PLAN.md",
      "role": "doc",
      "oneLiner": "Archived remediation plan for fixing curriculum internal consistency flaws identified against the CS336 Assignment 1 PDF ground truth."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/REMEDIATION_PROGRESS.md",
      "role": "doc",
      "oneLiner": "Archived progress tracker for the 2025-11-12 curriculum quality remediation session recording completed audit and planning phases."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/REMEDIATION_SUMMARY.md",
      "role": "doc",
      "oneLiner": "Archived summary confirming completion of Priority 1 and 2 curriculum remediation with engine support pending."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/SESSION_3_SUMMARY.md",
      "role": "doc",
      "oneLiner": "Archived summary of Session 3 covering CLI interface systematic analysis and Phase 1 implementation planning."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/STUDENT_MODE_AUDIT.md",
      "role": "doc",
      "oneLiner": "Archived audit mapping all student-mode module files that incorrectly contained full implementations instead of required stubs."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/STUDENT_MODE_FIX_SUMMARY.md",
      "role": "doc",
      "oneLiner": "Archived summary documenting the critical fix that replaced full implementations with proper stubs in student mode files."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/TOKENIZER_VIOLATIONS_AUDIT.md",
      "role": "doc",
      "oneLiner": "Archived audit confirming developer-mode BPE and Tokenizer reference implementations critically violate the from-scratch pedagogical constraint."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-11_curriculum_quality/VERIFICATION_FINDINGS.md",
      "role": "doc",
      "oneLiner": "Archived findings from comparing curriculum build prompts against ground-truth literature to verify accuracy of the RoPE module and others."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-12_cli_remediation/CLI_INTERFACE_AUDIT.md",
      "role": "doc",
      "oneLiner": "Archived audit report of engine/main.py CLI command interface identifying gaps, inconsistencies, and remediation priorities."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-12_cli_remediation/CLI_P0_FINAL_STATUS.md",
      "role": "doc",
      "oneLiner": "Archived final status report confirming 100% feature parity achieved after approximately five hours of P0 CLI implementation work."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-12_cli_remediation/CLI_P0_IMPLEMENTATION_COMPLETE.md",
      "role": "doc",
      "oneLiner": "Archived completion notice for the P0 CLI-001 command proliferation fix delivering the unified submit command in approximately 1.5 hours."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-12_cli_remediation/CLI_P0_IMPLEMENTATION_PLAN.md",
      "role": "doc",
      "oneLiner": "Archived implementation plan describing the strategy for adding a unified submit command to resolve CLI-001 command proliferation."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-12_cli_remediation/CLI_P0_PROGRESS.md",
      "role": "doc",
      "oneLiner": "Archived progress tracker for the P0 unified submit command implementation recording completion of Phases 1-3."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-12_cli_remediation/CLI_P1_IMPLEMENTATION_COMPLETE.md",
      "role": "doc",
      "oneLiner": "Archived completion notice for the P1 CLI-002 inconsistent command behavior fix declared complete on 2025-11-12."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-12_cli_remediation/CLI_REMEDIATION_COMPLETE.md",
      "role": "doc",
      "oneLiner": "Archived final report confirming all P0, P1, and P2 CLI remediation priorities completed in approximately six total hours."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-12_cli_remediation/CLI_REMEDIATION_PLAN.md",
      "role": "doc",
      "oneLiner": "Archived remediation plan for the engine/main.py CLI interface specifying issues to fix and the execution-ready approach."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-12_cli_remediation/CLI_REMEDIATION_STATUS.md",
      "role": "doc",
      "oneLiner": "Archived status report confirming planning complete and P0 core implementation complete for the CLI remediation session."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-12_test_coverage/COMPLETE_COMPREHENSIVE_REPORT.md",
      "role": "doc",
      "oneLiner": "Archived comprehensive report covering CLI remediation and test coverage improvements across five systematic phases over ~10 hours."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-12_test_coverage/COMPLETE_SESSION_SUMMARY.md",
      "role": "doc",
      "oneLiner": "Archived session summary for the combined CLI and test coverage work completed in approximately eight hours with all objectives exceeded."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-12_test_coverage/COVERAGE_70_80_ACHIEVEMENT.md",
      "role": "doc",
      "oneLiner": "Archived report documenting the engine test coverage increase from 64% to 76%, exceeding the 70-80% target."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-12_test_coverage/COVERAGE_80_ACHIEVEMENT.md",
      "role": "doc",
      "oneLiner": "Archived report documenting the engine test coverage increase from 76% to 78%, approaching the 80% threshold."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-12_test_coverage/EXCEPTIONAL_RIGOR_FINAL_REPORT.md",
      "role": "doc",
      "oneLiner": "Archived report detailing five engine bug fixes, seven permanent diagnostics, and twelve manual LLM analyses completed with exceptional rigor."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-12_test_coverage/FINAL_SESSION_REPORT.md",
      "role": "doc",
      "oneLiner": "Archived final session report for the combined CLI plus test coverage work across three major phases, declared production-ready."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-12_test_coverage/FINAL_SESSION_SUMMARY.md",
      "role": "doc",
      "oneLiner": "Archived final summary confirming all systematic improvement objectives satisfied including permanent improvements and regression guards."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-12_test_coverage/TEST_COVERAGE_FINAL_REPORT.md",
      "role": "doc",
      "oneLiner": "Archived final test coverage report after the three-hour systematic measurement and improvement session on 2025-11-12."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-12_test_coverage/TEST_COVERAGE_IMPROVEMENT_SESSION.md",
      "role": "doc",
      "oneLiner": "Archived session record documenting the BPE test fix and coverage improvement rated five-star quality, completed November 13, 2025."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-12_test_coverage/TEST_COVERAGE_SESSION_SUMMARY.md",
      "role": "doc",
      "oneLiner": "Archived session summary covering OpenAI SDK dependency fix, coverage baseline measurement, and incremental coverage improvements on 2025-11-12."
    },
    {
      "path": "docs/internal/archive/sessions/2025-11-12_test_coverage/TEST_FIX_SUMMARY.md",
      "role": "doc",
      "oneLiner": "Archived fix summary confirming all engine tests passing after resolving one failing test in the engine test suite."
    },
    {
      "path": "docs/internal/assignment/cs336_spring2025_assignment1_basics.pdf",
      "role": "asset",
      "oneLiner": "CS336 Spring 2025 Assignment 1 PDF handout serving as the ground-truth pedagogical specification for curriculum content."
    },
    {
      "path": "docs/internal/coverage/CURRENT_REPORT.md",
      "role": "doc",
      "oneLiner": "Current test coverage report showing 78% overall coverage across 145 tests with 100% pass rate and production-ready status."
    },
    {
      "path": "docs/internal/coverage/FINAL_COVERAGE_REPORT.txt",
      "role": "generated",
      "oneLiner": "Captured pytest-cov output showing final per-module statement/miss/cover numbers including engine/main.py at 36%."
    },
    {
      "path": "docs/internal/coverage/baselines/baseline_20251112_180245.txt",
      "role": "generated",
      "oneLiner": "Captured pytest output baseline recording e2e test failures from a macOS environment on 2025-11-12T18:02."
    },
    {
      "path": "docs/internal/coverage/baselines/baseline_no_e2e_20251112_180450.txt",
      "role": "generated",
      "oneLiner": "Captured pytest output baseline of passing non-e2e engine test results captured at 2025-11-12T18:04."
    },
    {
      "path": "docs/internal/coverage/baselines/baseline_no_e2e_20251112_180628.txt",
      "role": "generated",
      "oneLiner": "Captured pytest output baseline for non-e2e tests from a second run at 2025-11-12T18:06."
    },
    {
      "path": "docs/internal/coverage/baselines/baseline_no_e2e_full_20251112_180753.txt",
      "role": "generated",
      "oneLiner": "Captured full pytest session output baseline for 132 selected non-e2e tests on Python 3.13.1 at 2025-11-12T18:07."
    },
    {
      "path": "docs/internal/coverage/baselines/stages_baseline.txt",
      "role": "generated",
      "oneLiner": "Captured pytest-cov baseline showing 31% combined coverage for engine/stages/harden.py and engine/stages/justify.py."
    },
    {
      "path": "docs/internal/coverage/reports/coverage_after_cli_additions.txt",
      "role": "generated",
      "oneLiner": "Captured pytest-cov report for engine/main.py showing 48% coverage after CLI test additions."
    },
    {
      "path": "docs/internal/coverage/reports/coverage_final_phase2.txt",
      "role": "generated",
      "oneLiner": "Captured pytest-cov full-engine coverage report after Phase 2 improvements showing engine/main.py at 48%."
    },
    {
      "path": "docs/internal/coverage/reports/coverage_report_engine.txt",
      "role": "generated",
      "oneLiner": "Captured pytest-cov coverage report for the engine package showing engine/main.py at only 3% before test improvements."
    },
    {
      "path": "docs/internal/coverage/reports/coverage_report_engine_final.txt",
      "role": "generated",
      "oneLiner": "Captured pytest-cov final full-engine coverage report showing engine/main.py at 34% after remediation."
    },
    {
      "path": "docs/internal/coverage/reports/coverage_report_main_final.txt",
      "role": "generated",
      "oneLiner": "Captured pytest-cov coverage report for engine/main.py showing 28% coverage near end of remediation work."
    },
    {
      "path": "docs/internal/coverage/reports/coverage_report_main_partial.txt",
      "role": "generated",
      "oneLiner": "Captured pytest-cov coverage report for engine/main.py at 15% coverage during partial mid-session remediation."
    },
    {
      "path": "docs/internal/coverage/reports/coverage_report_no_e2e.txt",
      "role": "generated",
      "oneLiner": "Captured full pytest-cov report excluding e2e tests, including psutil and external macOS package paths from the original dev machine."
    },
    {
      "path": "docs/internal/coverage/reports/coverage_with_new_cli_tests.txt",
      "role": "generated",
      "oneLiner": "Captured pytest-cov report for engine/main.py showing 48% coverage after adding new CLI-focused tests."
    },
    {
      "path": "docs/internal/current/BUG_INJECTION_GUIDE.md",
      "role": "doc",
      "oneLiner": "Curriculum-author guide explaining the two-tier runtime AST bug injection architecture and how to define bug descriptors."
    },
    {
      "path": "docs/internal/current/CURRICULUM_STATUS.md",
      "role": "doc",
      "oneLiner": "Current production status for two curricula: cs336_a1 (21 modules, 98/100) and cp_accelerator with module counts and quality ratings."
    },
    {
      "path": "docs/internal/current/TEST_COVERAGE_REPORT.md",
      "role": "doc",
      "oneLiner": "Current test coverage report showing 78% overall coverage with 145 passing tests and production-ready status."
    },
    {
      "path": "docs/internal/development/CHANGELOG.md",
      "role": "doc",
      "oneLiner": "Changelog tracking versioned code and handout changes to the CS336 assignment starting from v1.0.6 on 2025-08-28."
    },
    {
      "path": "docs/internal/development/IMPLEMENTATION_PLAN.md",
      "role": "doc",
      "oneLiner": "Strategic implementation guide for completing CS336 Assignment 1, synthesizing best practices and providing an ordered implementation plan."
    },
    {
      "path": "docs/internal/development/MASTERY_WORKLOG.md",
      "role": "doc",
      "oneLiner": "Reverse-chronological worklog documenting the systematic transformation of the CS336 repository into Mastery Engine v1.0."
    },
    {
      "path": "docs/internal/development/MVP_COMPLETION_STATUS.md",
      "role": "doc",
      "oneLiner": "Production readiness declaration for Mastery Engine v1.0 MVP completed on November 12, 2025."
    },
    {
      "path": "docs/internal/development/WORKLOG.md",
      "role": "doc",
      "oneLiner": "Reverse-chronological research worklog with scientific hypothesis/evidence entries for ML experiments starting 2025-09-16."
    },
    {
      "path": "docs/internal/module_generation/MODULE_GENERATION_COMPREHENSIVE_SUMMARY.md",
      "role": "doc",
      "oneLiner": "Summary of the automated module generation system covering Phase 1 and Phase 2 results for 874 LeetCode problems."
    },
    {
      "path": "docs/internal/module_generation/MODULE_GENERATION_PHASE2_RESULTS.md",
      "role": "doc",
      "oneLiner": "Results report for Phase 2 automated build prompt generation validated against LC-912 Sort an Array with quality improvements over manual creation."
    },
    {
      "path": "docs/internal/module_generation/MODULE_GENERATION_PHASE3_DIAGNOSTIC.md",
      "role": "doc",
      "oneLiner": "Diagnostic report for Phase 3.1 module generation robustness testing on LC-200 Number of Islands."
    },
    {
      "path": "docs/internal/module_generation/MODULE_GENERATION_PHASE3_RESULTS.md",
      "role": "doc",
      "oneLiner": "Results report for Phase 3.1 robustness testing on LC-200 confirming graceful degradation and automatic fallback mechanisms."
    },
    {
      "path": "docs/internal/module_generation/MODULE_GENERATION_POC_RESULTS.md",
      "role": "doc",
      "oneLiner": "Proof-of-concept results for automated test case generation validated against LC-912, producing results equivalent to manual creation."
    },
    {
      "path": "docs/internal/module_generation/MODULE_GENERATION_PROGRESS.md",
      "role": "doc",
      "oneLiner": "Progress summary for automating module asset generation for 874 LeetCode problems via a Curriculum-as-Code pipeline."
    },
    {
      "path": "docs/internal/module_generation/MODULE_GENERATION_REFACTORING_PLAN.md",
      "role": "doc",
      "oneLiner": "Refactoring plan to evolve ingest_cp_content.py into generate_module.py for structured canonical_curriculum.json data consumption."
    },
    {
      "path": "docs/internal/module_generation/README.md",
      "role": "doc",
      "oneLiner": "Index document for the module generation documentation archive explaining scope, approach, and Phases 1-3 status."
    },
    {
      "path": "docs/internal/two_sum_qa/MODULE_COMPARISON_ANALYSIS.md",
      "role": "doc",
      "oneLiner": "Systematic comparison of sorting reference module versus two_sum generated module across all BUILD/JUSTIFY/HARDEN stage files."
    },
    {
      "path": "docs/internal/two_sum_qa/MODULE_COMPLETENESS_VERIFICATION.md",
      "role": "doc",
      "oneLiner": "Checklist verifying that the two_sum module contains all required files for each BUILD/JUSTIFY/HARDEN stage."
    },
    {
      "path": "docs/internal/two_sum_qa/README.md",
      "role": "doc",
      "oneLiner": "Index document for Two Sum QA documentation describing seven systematic testing phases achieving a 94/100 production-ready score."
    },
    {
      "path": "docs/internal/two_sum_qa/TWO_SUM_COMPLETION_SUMMARY.md",
      "role": "doc",
      "oneLiner": "Completion summary declaring the Two Sum LC-1 module production-ready with a quality score of 94/100 on November 18, 2025."
    },
    {
      "path": "docs/internal/two_sum_qa/TWO_SUM_E2E_WORKFLOW_TEST.md",
      "role": "doc",
      "oneLiner": "End-to-end test report verifying the Two Sum module across all three pedagogical stages and confirming production readiness."
    },
    {
      "path": "docs/internal/two_sum_qa/TWO_SUM_FINAL_QUALITY_AUDIT.md",
      "role": "doc",
      "oneLiner": "Final production readiness audit by Cascade AI approving the Two Sum LC-1 module for production deployment."
    },
    {
      "path": "docs/user-guide/MASTERY_COMMAND_REFERENCE.md",
      "role": "doc",
      "oneLiner": "User-facing reference for all 14 mastery CLI commands covering usage, flags, deprecation status, and installation instructions."
    },
    {
      "path": "engine/__init__.py",
      "role": "source",
      "oneLiner": "Empty package initializer making 'engine' a Python package."
    },
    {
      "path": "engine/ast_harden/__init__.py",
      "role": "source",
      "oneLiner": "Package docstring declaring the ast_harden sub-package as the AST-based bug injection engine for the Harden stage."
    },
    {
      "path": "engine/ast_harden/generic_injector.py",
      "role": "source",
      "oneLiner": "GenericBugInjector class that reads declarative JSON bug definitions and executes multi-pass AST find-and-replace transformations on student code."
    },
    {
      "path": "engine/ast_harden/pattern_matcher.py",
      "role": "source",
      "oneLiner": "PatternMatcher, FindAndTrackVisitor, and FindAndReplaceTransformer classes implementing the JSON-driven AST pattern matching and transformation system."
    },
    {
      "path": "engine/ast_harden/softmax_poc.py",
      "role": "source",
      "oneLiner": "Phase-1 proof-of-concept hardcoded softmax bug injector (SoftmaxCanonicalizer + SoftmaxBugInjector) with a runnable __main__ test harness."
    },
    {
      "path": "engine/ast_harden/softmax_v2_1.py",
      "role": "source",
      "oneLiner": "v2.1 two-phase mapping-based softmax bug injector that canonicalizes for matching but transforms the original AST to preserve student variable names, with a runnable __main__ test harness."
    },
    {
      "path": "engine/curriculum.py",
      "role": "source",
      "oneLiner": "CurriculumManager class that loads and validates curriculum manifests and provides path accessors for module/problem/pattern assets in both LINEAR and LIBRARY curriculum types."
    },
    {
      "path": "engine/dev_tools/__init__.py",
      "role": "source",
      "oneLiner": "Empty package initializer making 'engine/dev_tools' a Python package."
    },
    {
      "path": "engine/dev_tools/bug_author.py",
      "role": "source",
      "oneLiner": "BugAuthor class that uses an LLM with few-shot golden examples to automatically generate v2.1 JSON bug definitions from legacy .patch files, with a validation loop."
    },
    {
      "path": "engine/main.py",
      "role": "source",
      "oneLiner": "Typer-based CLI entry point defining all Mastery Engine commands (init, status, show, submit, start-challenge, select, create-bug, and legacy variants) for the Build-Justify-Harden learning loop."
    },
    {
      "path": "engine/schemas.py",
      "role": "source",
      "oneLiner": "Pydantic data models for all engine contracts: CurriculumManifest, UserProgress, JustifyQuestion, LLMEvaluationResponse, ValidationResult, and BugDefinition schemas."
    },
    {
      "path": "engine/services/__init__.py",
      "role": "source",
      "oneLiner": "Empty package initializer making 'engine/services' a Python package."
    },
    {
      "path": "engine/services/ast_service.py",
      "role": "source",
      "oneLiner": "Canonicalizer, SoftmaxBugInjector, CanonicalPatternMatcher, and OriginalASTTransformer implementing the softmax-specific AST canonicalize-match-transform pipeline used by the harden service."
    },
    {
      "path": "engine/services/llm_service.py",
      "role": "source",
      "oneLiner": "LLMService wrapping the OpenAI API for Chain-of-Thought justification evaluation and general completions, with mock mode when no API key is present."
    },
    {
      "path": "engine/stages/__init__.py",
      "role": "source",
      "oneLiner": "Empty package initializer making 'engine/stages' a Python package."
    },
    {
      "path": "engine/stages/harden.py",
      "role": "source",
      "oneLiner": "HardenRunner class that selects a bug, injects it (AST or patch), writes the buggy file to the shadow worktree, and returns the symptom for both LINEAR and LIBRARY curriculum modes."
    },
    {
      "path": "engine/stages/justify.py",
      "role": "source",
      "oneLiner": "JustifyRunner class that loads justify questions from curriculum and applies a fast keyword failure-mode filter before delegating to LLM semantic evaluation."
    },
    {
      "path": "engine/state.py",
      "role": "source",
      "oneLiner": "StateManager class that atomically reads and writes user progress to ~/.mastery_progress.json using a write-then-rename pattern."
    },
    {
      "path": "engine/utils.py",
      "role": "source",
      "oneLiner": "find_project_root() utility that walks up the directory tree looking for pyproject.toml, .git, or curricula+engine markers to locate the repository root."
    },
    {
      "path": "engine/validator.py",
      "role": "source",
      "oneLiner": "ValidationSubsystem class that executes validator.sh scripts in a controlled subprocess with timeout, captures exit code and output, and parses optional PERFORMANCE_SECONDS metrics."
    },
    {
      "path": "engine/workspace.py",
      "role": "source",
      "oneLiner": "WorkspaceManager class providing file system abstractions for workspace paths, harden workspace isolation via file copy, and patch application via the system 'patch' command."
    },
    {
      "path": "maintenance/PROJECT_STRUCTURE.md",
      "role": "doc",
      "oneLiner": "Describes the repository's dual-mode (student/developer) directory layout, key directories, mode-switching commands, and development workflows."
    },
    {
      "path": "maintenance/README_ORIGINAL.md",
      "role": "doc",
      "oneLiner": "Original CS336 Assignment 1 README covering the Mastery Engine workflow, bug-injection architecture for the Harden stage, and environment setup instructions."
    },
    {
      "path": "maintenance/RoadmapResources.md",
      "role": "doc",
      "oneLiner": "Curated list of competitive-programming learning resources (videos, blogs, practice sites) organized by Codeforces rating range."
    },
    {
      "path": "maintenance/VERIFICATION_REPORT.md",
      "role": "doc",
      "oneLiner": "Post-fix verification report dated November 18, 2025, documenting TOML syntax fix, entry-point installation, test suite results (129/145 passing), and outstanding work items."
    },
    {
      "path": "maintenance/make_submission.sh",
      "role": "source",
      "oneLiner": "Bash script that runs pytest and packages the project into a zip archive for CS336 assignment submission, excluding caches, binaries, and generated files."
    },
    {
      "path": "modes/README.md",
      "role": "doc",
      "oneLiner": "Design rationale and curriculum overview for student/developer modes covering all 21 transformer LM modules."
    },
    {
      "path": "modes/developer/cs336_basics/__init__.py",
      "role": "source",
      "oneLiner": "Package version initializer reading version from importlib.metadata for the developer reference package."
    },
    {
      "path": "modes/developer/cs336_basics/bpe.py",
      "role": "source",
      "oneLiner": "Complete BPE tokenizer training using a doubly-linked-list with max-heap for O(n log n) merge selection and incremental pair-count updates."
    },
    {
      "path": "modes/developer/cs336_basics/layers.py",
      "role": "source",
      "oneLiner": "Complete reference implementations of all transformer architecture components: Linear, Embedding, RMSNorm, SwiGLU, RoPE, scaled dot-product attention, multi-head attention, transformer_block, and transformer_lm."
    },
    {
      "path": "modes/developer/cs336_basics/optimizer.py",
      "role": "source",
      "oneLiner": "Complete AdamW optimizer with bias-corrected moment estimates and decoupled weight decay applied before gradient steps."
    },
    {
      "path": "modes/developer/cs336_basics/pretokenization_example.py",
      "role": "source",
      "oneLiner": "Provides find_chunk_boundaries helper for parallel corpus pre-tokenization plus an inline (non-importable) usage example using Ellipsis as a filename placeholder."
    },
    {
      "path": "modes/developer/cs336_basics/tokenizer.py",
      "role": "source",
      "oneLiner": "Complete byte-level BPE Tokenizer wrapping tiktoken's GPT-2 encoding with greedy special-token segmentation for encode, decode, and encode_iterable."
    },
    {
      "path": "modes/developer/cs336_basics/utils.py",
      "role": "source",
      "oneLiner": "Complete utility functions: numerically-stable softmax and cross-entropy, gradient clipping, cosine LR schedule with warmup, random batch sampler, and checkpoint save/load."
    },
    {
      "path": "modes/student/cs336_basics/__init__.py",
      "role": "source",
      "oneLiner": "Package version initializer reading version from importlib.metadata for the student stub package."
    },
    {
      "path": "modes/student/cs336_basics/bpe.py",
      "role": "source",
      "oneLiner": "Stub for train_bpe BPE training function with detailed implementation hints; raises NotImplementedError until the student implements it."
    },
    {
      "path": "modes/student/cs336_basics/generation.py",
      "role": "source",
      "oneLiner": "Stub for autoregressive text generation function with temperature/top-k/top-p sampling; raises NotImplementedError."
    },
    {
      "path": "modes/student/cs336_basics/layers.py",
      "role": "source",
      "oneLiner": "Stubs for all 10 transformer layer components (Linear, Embedding, silu, RMSNorm, SwiGLU, attention, RoPE, MHA, transformer_block, transformer_lm) each raising NotImplementedError with implementation guidance."
    },
    {
      "path": "modes/student/cs336_basics/optimizer.py",
      "role": "source",
      "oneLiner": "Stub AdamW optimizer class with detailed docstring on decoupled weight decay; both __init__ and step raise NotImplementedError."
    },
    {
      "path": "modes/student/cs336_basics/pretokenization_example.py",
      "role": "source",
      "oneLiner": "Provided-complete find_chunk_boundaries helper for parallel corpus chunking with inline usage example (identical to developer version)."
    },
    {
      "path": "modes/student/cs336_basics/tokenizer.py",
      "role": "source",
      "oneLiner": "Complete tiktoken-backed Tokenizer provided to students as a working helper; not a stub—uses GPT-2 encoding with greedy special-token matching."
    },
    {
      "path": "modes/student/cs336_basics/tokenizer_stub.py",
      "role": "source",
      "oneLiner": "Stub Tokenizer class skeleton with all four methods (init, encode, decode, encode_iterable) raising NotImplementedError for students to implement from scratch."
    },
    {
      "path": "modes/student/cs336_basics/utils.py",
      "role": "source",
      "oneLiner": "Stubs for all 7 utility functions (softmax, cross_entropy, gradient_clipping, cosine LR schedule, get_batch, save_checkpoint, load_checkpoint) each raising NotImplementedError with step-by-step hints."
    },
    {
      "path": "scripts/add_successful_to_golden.py",
      "role": "source",
      "oneLiner": "Interactive CLI that reads LLM evaluation results from /tmp and prompts the developer to add successful bug definitions into the golden dataset."
    },
    {
      "path": "scripts/auto_fix_drafts.py",
      "role": "source",
      "oneLiner": "Applies hardcoded fix functions for four known-bad draft AST injection patterns, tests each via GenericBugInjector, and saves corrected JSON files."
    },
    {
      "path": "scripts/enrich_problems.py",
      "role": "source",
      "oneLiner": "Fetches full LeetCode problem details (description, examples, hints) from a third-party API and merges them into canonical_curriculum.json."
    },
    {
      "path": "scripts/fetch_sources.sh",
      "role": "source",
      "oneLiner": "Downloads third-party curriculum source materials (30-Days-Of-Python repo, CP accelerator taxonomy placeholder) into the .sources/ directory."
    },
    {
      "path": "scripts/fix_draft_pattern.py",
      "role": "source",
      "oneLiner": "Interactive tool that shows each draft AST injection pattern alongside its patch transformation, tests it, and lets the developer fix and promote it."
    },
    {
      "path": "scripts/generate_ground_truth.py",
      "role": "source",
      "oneLiner": "Uses gpt-4o via BugAuthor to generate golden AST injection pattern JSON files for all 21 CS336-A1 curriculum modules in batch."
    },
    {
      "path": "scripts/generate_manifest.py",
      "role": "source",
      "oneLiner": "Reads canonical_curriculum.json, validates the dependency DAG via topological sort, and writes manifest.json for linear or library curriculum types."
    },
    {
      "path": "scripts/generate_module.py",
      "role": "source",
      "oneLiner": "Generates per-problem module assets (build_prompt.txt, test_cases.json, validator.sh) from enriched curriculum data, supporting single-problem and batch modes."
    },
    {
      "path": "scripts/migrate_bugs_llm.py",
      "role": "source",
      "oneLiner": "Batch migrates legacy .patch bug files to v2.1 JSON format by invoking the LLM-based BugAuthor for each unprocessed patch in the cs336_a1 curriculum."
    },
    {
      "path": "scripts/mode",
      "role": "source",
      "oneLiner": "Bash mode manager that switches the cs336_basics workspace symlink between student (stub) and developer (full) modes and can run commands in a mode temporarily."
    },
    {
      "path": "scripts/parse_sources.py",
      "role": "source",
      "oneLiner": "Parses DSA taxonomy markdown files and a CP roadmap document to produce a verified canonical_curriculum.json for the cp_accelerator curriculum."
    },
    {
      "path": "scripts/systematic_llm_evaluation.py",
      "role": "source",
      "oneLiner": "Benchmarks the LLM bug authoring tool across all 21 CS336-A1 bugs, collecting success rates, failure modes, pattern accuracy, and regression checks."
    },
    {
      "path": "scripts/templates/build_prompt.jinja2",
      "role": "asset",
      "oneLiner": "Jinja2 template that renders the build_prompt.txt challenge file shown to students, embedding problem statement, examples, constraints, hints, and resources."
    },
    {
      "path": "scripts/test_ci.sh",
      "role": "source",
      "oneLiner": "Local CI runner that executes the exact pytest command used in GitHub Actions (tests/engine/, excluding integration marks) so developers can verify before pushing."
    },
    {
      "path": "scripts/test_library_loading.py",
      "role": "test",
      "oneLiner": "Functional test that exercises CurriculumManager loading a LIBRARY manifest, verifying pattern/problem path resolution and on-disk file existence."
    },
    {
      "path": "scripts/validate_student_stubs.py",
      "role": "source",
      "oneLiner": "Checks every Python file under modes/student/ for NotImplementedError stubs or TODO markers, exiting non-zero if complete implementations are found."
    },
    {
      "path": "scripts/verify_curriculum_manifests.py",
      "role": "source",
      "oneLiner": "Statically verifies that all modules declared in a curriculum manifest have their required files (build_prompt.txt, validator.sh, etc.) present on disk."
    },
    {
      "path": "scripts/verify_ground_truth.py",
      "role": "source",
      "oneLiner": "Runs every golden AST injection pattern against its corresponding patch to confirm the pattern produces the expected buggy code, reporting a pass/fail summary."
    },
    {
      "path": "tests/__init__.py",
      "role": "test",
      "oneLiner": "Empty package init making tests/ a Python package."
    },
    {
      "path": "tests/_snapshots/test_4d_scaled_dot_product_attention.npz",
      "role": "asset",
      "oneLiner": "Pre-committed NumPy snapshot array for 4D scaled-dot-product attention snapshot test."
    },
    {
      "path": "tests/_snapshots/test_adamw.npz",
      "role": "asset",
      "oneLiner": "Pre-committed NumPy snapshot array for AdamW optimizer snapshot test."
    },
    {
      "path": "tests/_snapshots/test_embedding.npz",
      "role": "asset",
      "oneLiner": "Pre-committed NumPy snapshot array for Embedding layer snapshot test."
    },
    {
      "path": "tests/_snapshots/test_linear.npz",
      "role": "asset",
      "oneLiner": "Pre-committed NumPy snapshot array for Linear layer snapshot test."
    },
    {
      "path": "tests/_snapshots/test_multihead_self_attention.npz",
      "role": "asset",
      "oneLiner": "Pre-committed NumPy snapshot array for multi-head self-attention snapshot test."
    },
    {
      "path": "tests/_snapshots/test_multihead_self_attention_with_rope.npz",
      "role": "asset",
      "oneLiner": "Pre-committed NumPy snapshot array for multi-head self-attention with RoPE snapshot test."
    },
    {
      "path": "tests/_snapshots/test_positionwise_feedforward.npz",
      "role": "asset",
      "oneLiner": "Pre-committed NumPy snapshot array for position-wise feedforward snapshot test."
    },
    {
      "path": "tests/_snapshots/test_rmsnorm.npz",
      "role": "asset",
      "oneLiner": "Pre-committed NumPy snapshot array for RMSNorm snapshot test."
    },
    {
      "path": "tests/_snapshots/test_rope.npz",
      "role": "asset",
      "oneLiner": "Pre-committed NumPy snapshot array for RoPE snapshot test."
    },
    {
      "path": "tests/_snapshots/test_scaled_dot_product_attention.npz",
      "role": "asset",
      "oneLiner": "Pre-committed NumPy snapshot array for scaled-dot-product attention snapshot test."
    },
    {
      "path": "tests/_snapshots/test_swiglu.npz",
      "role": "asset",
      "oneLiner": "Pre-committed NumPy snapshot array for SwiGLU feedforward snapshot test."
    },
    {
      "path": "tests/_snapshots/test_train_bpe_special_tokens.pkl",
      "role": "asset",
      "oneLiner": "Pre-committed pickle snapshot of expected BPE vocab and merges for special-token training test."
    },
    {
      "path": "tests/_snapshots/test_transformer_block.npz",
      "role": "asset",
      "oneLiner": "Pre-committed NumPy snapshot array for Transformer block snapshot test."
    },
    {
      "path": "tests/_snapshots/test_transformer_lm.npz",
      "role": "asset",
      "oneLiner": "Pre-committed NumPy snapshot array for Transformer language model snapshot test."
    },
    {
      "path": "tests/_snapshots/test_transformer_lm_truncated_input.npz",
      "role": "asset",
      "oneLiner": "Pre-committed NumPy snapshot array for Transformer LM with truncated input snapshot test."
    },
    {
      "path": "tests/adapters.py",
      "role": "test",
      "oneLiner": "Thin adapter wrappers around cs336_basics implementations, used by test files to invoke student code through a stable interface."
    },
    {
      "path": "tests/common.py",
      "role": "test",
      "oneLiner": "Shared test helpers: FIXTURES_PATH constant and gpt2_bytes_to_unicode utility used across tokenizer tests."
    },
    {
      "path": "tests/conftest.py",
      "role": "test",
      "oneLiner": "Pytest conftest providing shared fixtures: numpy_snapshot, snapshot, ts_state_dict, and model-dimension fixtures for the test suite."
    },
    {
      "path": "tests/e2e/E2E_TEST_STATUS.md",
      "role": "doc",
      "oneLiner": "Status report documenting current E2E test coverage (95%), known gaps, and rationale for v1.0 ship decision."
    },
    {
      "path": "tests/e2e/__init__.py",
      "role": "test",
      "oneLiner": "Package init for E2E tests with module docstring describing end-to-end test scope."
    },
    {
      "path": "tests/e2e/debug_shadow_worktree.py",
      "role": "test",
      "oneLiner": "Debug script that inspects shadow worktree structure and runs pytest collection diagnostics to trace import failures."
    },
    {
      "path": "tests/e2e/test_adversarial_stress.py",
      "role": "test",
      "oneLiner": "Adversarial stress tests probing engine resilience: massive output, timeouts, corrupted patches, permission errors, and LLM prompt injection."
    },
    {
      "path": "tests/e2e/test_build_only.py",
      "role": "test",
      "oneLiner": "Minimal E2E test verifying that the BUILD stage completes successfully when developer mode is active."
    },
    {
      "path": "tests/e2e/test_complete_bjh_loop.py",
      "role": "test",
      "oneLiner": "Comprehensive E2E regression test for the full Build-Justify-Harden loop of the softmax module using isolated subprocess calls."
    },
    {
      "path": "tests/e2e/test_error_handling.py",
      "role": "test",
      "oneLiner": "E2E tests validating engine behavior for error paths: uninitialized commands, stale worktree, wrong-stage usage, and corrupted state."
    },
    {
      "path": "tests/e2e/test_full_softmax_loop.py",
      "role": "test",
      "oneLiner": "E2E test of the complete softmax BJH loop using the Typer CLI runner and mocked LLM, verifying all state transitions."
    },
    {
      "path": "tests/engine/__init__.py",
      "role": "test",
      "oneLiner": "Empty package init making tests/engine/ a Python package."
    },
    {
      "path": "tests/engine/test_curriculum.py",
      "role": "test",
      "oneLiner": "Unit tests for CurriculumManager: init, loading, path resolution, and error handling to achieve 100% line coverage."
    },
    {
      "path": "tests/engine/test_error_handling.py",
      "role": "test",
      "oneLiner": "Unit tests targeting uncovered error-handling paths in submit, show, status, and start-challenge CLI commands."
    },
    {
      "path": "tests/engine/test_harden_additional.py",
      "role": "test",
      "oneLiner": "Additional edge-case and error-path tests for HardenRunner to increase harden.py coverage."
    },
    {
      "path": "tests/engine/test_init_cleanup.py",
      "role": "test",
      "oneLiner": "Unit tests for the init and cleanup CLI commands covering success paths and all documented error conditions."
    },
    {
      "path": "tests/engine/test_legacy_commands.py",
      "role": "test",
      "oneLiner": "Tests for backward-compatible legacy submit commands (submit_build, submit_justification, submit_fix) in main.py."
    },
    {
      "path": "tests/engine/test_llm_service.py",
      "role": "test",
      "oneLiner": "Unit tests for LLMService achieving 100% coverage using mocked OpenAI API responses."
    },
    {
      "path": "tests/engine/test_main.py",
      "role": "test",
      "oneLiner": "Unit tests for main CLI commands with mocked state and curriculum managers, focusing on command behavior and error messages."
    },
    {
      "path": "tests/engine/test_main_comprehensive_coverage.py",
      "role": "test",
      "oneLiner": "Aggressive coverage tests for submit-stage helpers _submit_build_stage, _submit_harden_stage, _submit_linear_workflow, and _submit_library_workflow."
    },
    {
      "path": "tests/engine/test_main_console_paths.py",
      "role": "test",
      "oneLiner": "Tests verifying Rich console output produced by _submit_build_stage across validation success and failure paths."
    },
    {
      "path": "tests/engine/test_main_error_paths.py",
      "role": "test",
      "oneLiner": "Tests for error and exception paths in require_shadow_worktree, _check_curriculum_complete, and _submit_linear_workflow."
    },
    {
      "path": "tests/engine/test_main_helpers.py",
      "role": "test",
      "oneLiner": "Unit tests for isolated helper functions in main.py: require_shadow_worktree, _load_curriculum_state, _show_linear_status."
    },
    {
      "path": "tests/engine/test_main_workflows_real.py",
      "role": "test",
      "oneLiner": "Systematic tests for workflow orchestration in _submit_linear_workflow and _submit_library_workflow."
    },
    {
      "path": "tests/engine/test_new_cli_commands.py",
      "role": "test",
      "oneLiner": "Coverage tests for P1/P2 CLI commands: show, start_challenge, next (deprecated), curriculum_list, and progress_reset."
    },
    {
      "path": "tests/engine/test_stages.py",
      "role": "test",
      "oneLiner": "Tests for harden and justify stage modules covering challenge setup, bug selection, question loading, and fast-filter logic."
    },
    {
      "path": "tests/engine/test_state.py",
      "role": "test",
      "oneLiner": "Unit tests for StateManager achieving 100% line coverage including corrupted-file and write-error paths."
    },
    {
      "path": "tests/engine/test_submit_handlers.py",
      "role": "test",
      "oneLiner": "Tests for the unified submit command and its underlying handler helpers to maximize main.py coverage."
    },
    {
      "path": "tests/engine/test_utils_complete.py",
      "role": "test",
      "oneLiner": "Complete coverage tests for engine/utils.py find_project_root function."
    },
    {
      "path": "tests/engine/test_validator.py",
      "role": "test",
      "oneLiner": "Unit tests for ValidationSubsystem achieving 100% coverage with focus on security-critical timeout enforcement."
    },
    {
      "path": "tests/engine/test_workspace.py",
      "role": "test",
      "oneLiner": "Unit tests for WorkspaceManager covering workspace init, file operations, patch application, and error paths."
    },
    {
      "path": "tests/fixtures/address.txt",
      "role": "asset",
      "oneLiner": "Gettysburg Address plaintext used as a BPE tokenizer training corpus fixture."
    },
    {
      "path": "tests/fixtures/corpus.en",
      "role": "asset",
      "oneLiner": "English translation corpus sentences used as a tokenizer test fixture."
    },
    {
      "path": "tests/fixtures/german.txt",
      "role": "asset",
      "oneLiner": "German-language text used as a non-ASCII BPE tokenizer test fixture."
    },
    {
      "path": "tests/fixtures/gpt2_merges.txt",
      "role": "asset",
      "oneLiner": "GPT-2 BPE merge rules used to construct a reference tokenizer in tests."
    },
    {
      "path": "tests/fixtures/gpt2_vocab.json",
      "role": "asset",
      "oneLiner": "GPT-2 vocabulary JSON mapping token strings to IDs, used to construct a reference tokenizer in tests."
    },
    {
      "path": "tests/fixtures/special_token_double_newlines_non_whitespace.txt",
      "role": "asset",
      "oneLiner": "Short text with special token followed by double newlines and non-whitespace, used to test special-token boundary handling."
    },
    {
      "path": "tests/fixtures/special_token_trailing_newlines.txt",
      "role": "asset",
      "oneLiner": "Short text with special token followed by trailing newlines, used to test special-token boundary handling."
    },
    {
      "path": "tests/fixtures/tinystories_sample.txt",
      "role": "asset",
      "oneLiner": "Small TinyStories excerpt used as a fast BPE training corpus fixture."
    },
    {
      "path": "tests/fixtures/tinystories_sample_5M.txt",
      "role": "asset",
      "oneLiner": "Larger 5M-token TinyStories sample used for heavier BPE training corpus tests."
    },
    {
      "path": "tests/fixtures/train-bpe-reference-merges.txt",
      "role": "asset",
      "oneLiner": "Reference BPE merge list expected output for the train_bpe test on the TinyStories corpus."
    },
    {
      "path": "tests/fixtures/train-bpe-reference-vocab.json",
      "role": "asset",
      "oneLiner": "Reference BPE vocabulary JSON expected output for the train_bpe test on the TinyStories corpus."
    },
    {
      "path": "tests/fixtures/ts_tests/model.pt",
      "role": "asset",
      "oneLiner": "Serialized PyTorch model checkpoint (zip/pt format) used as a golden reference for layer output snapshot tests."
    },
    {
      "path": "tests/fixtures/ts_tests/model_config.json",
      "role": "asset",
      "oneLiner": "JSON config (vocab_size, context_length, d_model, num_layers, etc.) for the ts_tests reference model."
    },
    {
      "path": "tests/integration/README.md",
      "role": "doc",
      "oneLiner": "Documentation for integration tests: setup, cost per run, when to execute, and best practices for the live LLM API test suite."
    },
    {
      "path": "tests/integration/__init__.py",
      "role": "test",
      "oneLiner": "Empty package init making tests/integration/ a Python package."
    },
    {
      "path": "tests/integration/test_llm_service.py",
      "role": "test",
      "oneLiner": "Integration tests that make real OpenAI API calls to validate LLMService prompt formatting, response parsing, and error handling."
    },
    {
      "path": "tests/one_d_probes.py",
      "role": "source",
      "oneLiner": "Standalone research script that trains a small Transformer on a 1D binary-sequence task to probe model internals; not a pytest test file."
    },
    {
      "path": "tests/test_data.py",
      "role": "test",
      "oneLiner": "Tests for get_batch data-sampling utility: shape, randomness, and correct offset between input and label sequences."
    },
    {
      "path": "tests/test_model.py",
      "role": "test",
      "oneLiner": "Snapshot-based tests for all neural-network layer implementations (Linear, Embedding, RoPE, MHA, SwiGLU, TransformerBlock, TransformerLM)."
    },
    {
      "path": "tests/test_nn_utils.py",
      "role": "test",
      "oneLiner": "Tests for nn utility functions: softmax numerical stability, cross-entropy, and gradient clipping against PyTorch references."
    },
    {
      "path": "tests/test_optimizer.py",
      "role": "test",
      "oneLiner": "Tests for AdamW optimizer correctness and cosine learning-rate schedule via snapshot and arithmetic checks."
    },
    {
      "path": "tests/test_serialization.py",
      "role": "test",
      "oneLiner": "Tests for checkpoint save/load round-trip fidelity for model weights and optimizer state."
    },
    {
      "path": "tests/test_tokenizer.py",
      "role": "test",
      "oneLiner": "Tests for BPE Tokenizer encode/decode correctness, GPT-2 parity, special-token handling, memory limits, and train_bpe output."
    },
    {
      "path": "tests/test_train_bpe.py",
      "role": "test",
      "oneLiner": "Pytest tests for BPE tokenizer training: validates speed (<1.5s), merge/vocab correctness against GPT-2 reference, and special-token isolation."
    }
  ],
  "entryPoints": [
    {
      "name": "mastery",
      "kind": "cli_command",
      "location": "pyproject.toml:34",
      "description": "Top-level CLI command installed by pip/uv that maps to engine.main:main, exposing all mastery subcommands (init, show, submit, start-challenge, cleanup, etc.)."
    },
    {
      "name": "Engine Tests / test",
      "kind": "github-actions-job",
      "location": ".github/workflows/tests.yml:10",
      "description": "CI job that installs dependencies and runs pytest over tests/engine/ (excluding integration tests) with coverage reporting."
    },
    {
      "name": "Engine Tests / lint",
      "kind": "github-actions-job",
      "location": ".github/workflows/tests.yml:60",
      "description": "CI job that runs ruff linter and formatter check over engine/ and tests/ directories."
    },
    {
      "name": "CP Accelerator - Manifest Integrity Check / validate-manifest",
      "kind": "github-actions-job",
      "location": ".github/workflows/validate_cp_manifest.yml:15",
      "description": "CI job that validates canonical_curriculum.json, regenerates manifest.json via scripts/generate_manifest.py, and asserts no manual edits were made to manifest.json."
    },
    {
      "name": "CP Accelerator - Manifest Integrity Check / schema-validation",
      "kind": "github-actions-job",
      "location": ".github/workflows/validate_cp_manifest.yml:112",
      "description": "CI job that validates the structure of canonical_curriculum.json and manifest.json against required field schemas using inline Python."
    },
    {
      "name": "CP Accelerator - Manifest Integrity Check / dependency-graph-analysis",
      "kind": "github-actions-job",
      "location": ".github/workflows/validate_cp_manifest.yml:201",
      "description": "CI job that generates a dependency graph statistics report for curriculum patterns (roots, leaves, totals) from manifest.json."
    },
    {
      "name": "subsets",
      "kind": "function",
      "location": "curricula/cp_accelerator/patterns/backtracking/problems/lc_78/solution.py:7",
      "description": "Backtracking function generating all subsets (power set) of a unique-element array."
    },
    {
      "name": "solve",
      "kind": "exported_api",
      "location": "curricula/cp_accelerator/patterns/backtracking/problems/lc_78/solution.py:36",
      "description": "Public alias for `subsets` used by the mastery engine test runner to invoke the lc_78 solution."
    },
    {
      "name": "validator.sh (lc_78)",
      "kind": "script",
      "location": "curricula/cp_accelerator/patterns/backtracking/problems/lc_78/validator.sh:1",
      "description": "Shell entry point that loads solution.py and runs it against lc_78 test cases; exits non-zero on failure."
    },
    {
      "name": "validator.sh (lc_90)",
      "kind": "script",
      "location": "curricula/cp_accelerator/patterns/backtracking/problems/lc_90/validator.sh:1",
      "description": "Shell entry point that loads solution.py and runs it against lc_90 Subsets II test cases."
    },
    {
      "name": "search",
      "kind": "function",
      "location": "curricula/cp_accelerator/patterns/binary_search/problems/lc_704/solution.py:7",
      "description": "Iterative binary search returning the index of target in a sorted array, or -1 if not found."
    },
    {
      "name": "solve",
      "kind": "exported_api",
      "location": "curricula/cp_accelerator/patterns/binary_search/problems/lc_704/solution.py:38",
      "description": "Public alias for `search` used by the mastery engine test runner to invoke the lc_704 solution."
    },
    {
      "name": "validator.sh (lc_34)",
      "kind": "script",
      "location": "curricula/cp_accelerator/patterns/binary_search/problems/lc_34/validator.sh:1",
      "description": "Shell entry point that loads solution.py and runs it against lc_34 Find First and Last Position test cases."
    },
    {
      "name": "validator.sh (lc_704)",
      "kind": "script",
      "location": "curricula/cp_accelerator/patterns/binary_search/problems/lc_704/validator.sh:1",
      "description": "Shell entry point that loads solution.py and runs it against lc_704 Binary Search test cases."
    },
    {
      "name": "validator.sh (lc_1342)",
      "kind": "script",
      "location": "curricula/cp_accelerator/patterns/bit_manipulation/problems/lc_1342/validator.sh:1",
      "description": "Shell entry point that loads solution.py and validates it against lc_1342 Number of Steps test cases."
    },
    {
      "name": "validator.sh (lc_1486)",
      "kind": "script",
      "location": "curricula/cp_accelerator/patterns/bit_manipulation/problems/lc_1486/validator.sh:1",
      "description": "Shell entry point that loads solution.py and validates it against lc_1486 XOR Operation test cases."
    },
    {
      "name": "validator.sh (lc_46)",
      "kind": "script",
      "location": "curricula/cp_accelerator/patterns/combinatorics_and_number_theory/problems/lc_46/validator.sh:1",
      "description": "Shell entry point that loads solution.py and validates it against lc_46 Permutations test cases."
    },
    {
      "name": "validator.sh (lc_47)",
      "kind": "script",
      "location": "curricula/cp_accelerator/patterns/combinatorics_and_number_theory/problems/lc_47/validator.sh:1",
      "description": "Shell entry point that loads solution.py and validates it against lc_47 Permutations II test cases."
    },
    {
      "name": "validator.sh (lc_146)",
      "kind": "script",
      "location": "curricula/cp_accelerator/patterns/design_patterns/problems/lc_146/validator.sh:1",
      "description": "Shell entry point that loads solution.py and validates it against lc_146 LRU Cache test cases."
    },
    {
      "name": "validator.sh (lc_460)",
      "kind": "script",
      "location": "curricula/cp_accelerator/patterns/design_patterns/problems/lc_460/validator.sh:1",
      "description": "Shell entry point that loads solution.py and validates it against lc_460 LFU Cache test cases."
    },
    {
      "name": "validator.sh (lc_148)",
      "kind": "script",
      "location": "curricula/cp_accelerator/patterns/divide_and_conquer/problems/lc_148/validator.sh:1",
      "description": "Shell entry point that loads solution.py and validates it against lc_148 Sort List test cases."
    },
    {
      "name": "validator.sh (lc_912)",
      "kind": "script",
      "location": "curricula/cp_accelerator/patterns/divide_and_conquer/problems/lc_912/validator.sh:1",
      "description": "Shell entry point that loads solution.py and validates it against lc_912 Sort an Array test cases."
    },
    {
      "name": "validator.sh (lc_198)",
      "kind": "script",
      "location": "curricula/cp_accelerator/patterns/dynamic_programming/problems/lc_198/validator.sh:1",
      "description": "Shell entry point that loads solution.py and validates it against lc_198 House Robber test cases."
    },
    {
      "name": "climbStairs",
      "kind": "function",
      "location": "curricula/cp_accelerator/patterns/dynamic_programming/problems/lc_70/solution.py:7",
      "description": "O(1)-space DP function counting distinct ways to climb n stairs taking 1 or 2 steps at a time."
    },
    {
      "name": "solve",
      "kind": "exported_api",
      "location": "curricula/cp_accelerator/patterns/dynamic_programming/problems/lc_70/solution.py:36",
      "description": "Public alias for `climbStairs` used by the mastery engine test runner to invoke the lc_70 solution."
    },
    {
      "name": "validator.sh (lc_70)",
      "kind": "script",
      "location": "curricula/cp_accelerator/patterns/dynamic_programming/problems/lc_70/validator.sh:1",
      "description": "Shell entry point that loads solution.py and validates it against lc_70 Climbing Stairs test cases."
    },
    {
      "name": "eraseOverlapIntervals",
      "kind": "function",
      "location": "curricula/cp_accelerator/patterns/greedy/problems/lc_435/solution.py:7",
      "description": "Greedy function returning the minimum number of intervals to remove to make the rest non-overlapping."
    },
    {
      "name": "solve",
      "kind": "exported_api",
      "location": "curricula/cp_accelerator/patterns/greedy/problems/lc_435/solution.py:42",
      "description": "Public alias for `eraseOverlapIntervals` used by the mastery engine test runner to invoke the lc_435 solution."
    },
    {
      "name": "validator.sh (lc_435)",
      "kind": "script",
      "location": "curricula/cp_accelerator/patterns/greedy/problems/lc_435/validator.sh:1",
      "description": "Shell entry point that loads solution.py and validates it against lc_435 Non-overlapping Intervals test cases."
    },
    {
      "name": "validator.sh (lc_452)",
      "kind": "script",
      "location": "curricula/cp_accelerator/patterns/greedy/problems/lc_452/validator.sh:1",
      "description": "Shell entry point that loads solution.py and validates it against lc_452 Minimum Arrows test cases."
    },
    {
      "name": "twoSum",
      "kind": "exported-function",
      "location": "curricula/cp_accelerator/patterns/hash_table/problems/lc_1/solution.py:6",
      "description": "Returns indices of two numbers in nums that sum to target; O(n) time using a hash map."
    },
    {
      "name": "validator (lc_1)",
      "kind": "cli-script",
      "location": "curricula/cp_accelerator/patterns/hash_table/problems/lc_1/validator.sh:1",
      "description": "Bash entry point that imports twoSum and validates it against test_cases.json, exiting non-zero on failure."
    },
    {
      "name": "validator (lc_217)",
      "kind": "cli-script",
      "location": "curricula/cp_accelerator/patterns/hash_table/problems/lc_217/validator.sh:1",
      "description": "Bash entry point that imports containsDuplicate and validates it against test_cases.json."
    },
    {
      "name": "validator (lc_219)",
      "kind": "cli-script",
      "location": "curricula/cp_accelerator/patterns/hash_table/problems/lc_219/validator.sh:1",
      "description": "Bash entry point that imports containsNearbyDuplicate and validates it against test_cases.json."
    },
    {
      "name": "validator (lc_215)",
      "kind": "cli-script",
      "location": "curricula/cp_accelerator/patterns/heap_and_priority_queue/problems/lc_215/validator.sh:1",
      "description": "Bash entry point that imports findKthLargest and validates it against test_cases.json."
    },
    {
      "name": "validator (lc_703)",
      "kind": "cli-script",
      "location": "curricula/cp_accelerator/patterns/heap_and_priority_queue/problems/lc_703/validator.sh:1",
      "description": "Bash entry point that imports KthLargest class and validates it against test_cases.json."
    },
    {
      "name": "removeElements",
      "kind": "exported-function",
      "location": "curricula/cp_accelerator/patterns/linked_list/problems/lc_203/solution.py:8",
      "description": "Removes all linked list elements equal to val; O(n) using array representation for test-runner compatibility."
    },
    {
      "name": "validator (lc_203)",
      "kind": "cli-script",
      "location": "curricula/cp_accelerator/patterns/linked_list/problems/lc_203/validator.sh:1",
      "description": "Bash entry point that imports removeElements and validates it against test_cases.json."
    },
    {
      "name": "validator (lc_237)",
      "kind": "cli-script",
      "location": "curricula/cp_accelerator/patterns/linked_list/problems/lc_237/validator.sh:1",
      "description": "Bash entry point that imports deleteNode and validates it against test_cases.json."
    },
    {
      "name": "validator (lc_1480)",
      "kind": "cli-script",
      "location": "curricula/cp_accelerator/patterns/prefix_sum/problems/lc_1480/validator.sh:1",
      "description": "Bash entry point that imports runningSum and validates it against test_cases.json."
    },
    {
      "name": "sumRange",
      "kind": "exported-function",
      "location": "curricula/cp_accelerator/patterns/prefix_sum/problems/lc_303/solution.py:7",
      "description": "Returns prefix-sum range query result for indices [left, right] inclusive; O(1) after O(n) build."
    },
    {
      "name": "validator (lc_303)",
      "kind": "cli-script",
      "location": "curricula/cp_accelerator/patterns/prefix_sum/problems/lc_303/validator.sh:1",
      "description": "Bash entry point that imports sumRange and validates it against test_cases.json."
    },
    {
      "name": "validator (lc_307)",
      "kind": "cli-script",
      "location": "curricula/cp_accelerator/patterns/segment_tree_and_fenwick_tree/problems/lc_307/validator.sh:1",
      "description": "Bash entry point that imports NumArray class and validates it against test_cases.json."
    },
    {
      "name": "validator (lc_148)",
      "kind": "cli-script",
      "location": "curricula/cp_accelerator/patterns/sorting/problems/lc_148/validator.sh:1",
      "description": "Bash entry point that imports sortList and validates it against test_cases.json."
    },
    {
      "name": "sortArray",
      "kind": "exported-function",
      "location": "curricula/cp_accelerator/patterns/sorting/problems/lc_912/solution.py:6",
      "description": "Sorts an integer array using merge sort; O(n log n) time, O(n) space."
    },
    {
      "name": "validator (lc_912)",
      "kind": "cli-script",
      "location": "curricula/cp_accelerator/patterns/sorting/problems/lc_912/validator.sh:1",
      "description": "Bash entry point that imports sortArray and validates it against test_cases.json."
    },
    {
      "name": "validator (lc_1003)",
      "kind": "cli-script",
      "location": "curricula/cp_accelerator/patterns/stack_and_queue/problems/lc_1003/validator.sh:1",
      "description": "Bash entry point that imports isValid and validates it against test_cases.json for lc_1003."
    },
    {
      "name": "isValid",
      "kind": "exported-function",
      "location": "curricula/cp_accelerator/patterns/stack_and_queue/problems/lc_20/solution.py:7",
      "description": "Validates bracket nesting in string s using a stack; O(n) time and space."
    },
    {
      "name": "validator (lc_20)",
      "kind": "cli-script",
      "location": "curricula/cp_accelerator/patterns/stack_and_queue/problems/lc_20/validator.sh:1",
      "description": "Bash entry point that imports isValid and validates it against test_cases.json for lc_20."
    },
    {
      "name": "validator.sh (lc_144)",
      "kind": "shell-script",
      "location": "curricula/cp_accelerator/patterns/traversal/problems/lc_144/validator.sh:1",
      "description": "Runs student solution.py against LC-144 example test cases and reports pass/fail counts."
    },
    {
      "name": "validator.sh (lc_589)",
      "kind": "shell-script",
      "location": "curricula/cp_accelerator/patterns/traversal/problems/lc_589/validator.sh:1",
      "description": "Runs student solution.py against LC-589 example test cases and reports pass/fail counts."
    },
    {
      "name": "validator.sh (lc_1804)",
      "kind": "shell-script",
      "location": "curricula/cp_accelerator/patterns/trie/problems/lc_1804/validator.sh:1",
      "description": "Runs student solution.py against LC-1804 example test cases and reports pass/fail counts."
    },
    {
      "name": "validator.sh (lc_208)",
      "kind": "shell-script",
      "location": "curricula/cp_accelerator/patterns/trie/problems/lc_208/validator.sh:1",
      "description": "Runs student solution.py against LC-208 example test cases and reports pass/fail counts."
    },
    {
      "name": "validator.sh (lc_1099)",
      "kind": "shell-script",
      "location": "curricula/cp_accelerator/patterns/two_pointers/problems/lc_1099/validator.sh:1",
      "description": "Runs student solution.py against LC-1099 example test cases and reports pass/fail counts."
    },
    {
      "name": "validator.sh (lc_167)",
      "kind": "shell-script",
      "location": "curricula/cp_accelerator/patterns/two_pointers/problems/lc_167/validator.sh:1",
      "description": "Runs student solution.py against LC-167 example and edge-case test cases, reports pass/fail counts."
    },
    {
      "name": "validator.sh (lc_547)",
      "kind": "shell-script",
      "location": "curricula/cp_accelerator/patterns/union_find_disjoint_set_union/problems/lc_547/validator.sh:1",
      "description": "Runs student solution.py against LC-547 example test cases and reports pass/fail counts."
    },
    {
      "name": "validator.sh (lc_684)",
      "kind": "shell-script",
      "location": "curricula/cp_accelerator/patterns/union_find_disjoint_set_union/problems/lc_684/validator.sh:1",
      "description": "Runs student solution.py against LC-684 example test cases and reports pass/fail counts."
    },
    {
      "name": "twoSum",
      "kind": "function",
      "location": "curricula/cp_accelerator/patterns/two_pointers/problems/lc_167/solution.py:7",
      "description": "Reference O(n)/O(1) two-pointer implementation returning 1-indexed pair indices for LC-167."
    },
    {
      "name": "solve",
      "kind": "function",
      "location": "curricula/cp_accelerator/patterns/two_pointers/problems/lc_167/solution.py:38",
      "description": "Alias for twoSum used by the generic test runner (validator.sh calls solve(**test['input']))."
    },
    {
      "name": "validator.sh (adamw)",
      "kind": "shell-script",
      "location": "curricula/cs336_a1/modules/adamw/validator.sh:1",
      "description": "Runs pytest for the AdamW optimizer implementation inside a shadow worktree."
    },
    {
      "name": "validator.sh (attention)",
      "kind": "shell-script",
      "location": "curricula/cs336_a1/modules/attention/validator.sh:1",
      "description": "Runs pytest for the scaled dot-product attention implementation inside a shadow worktree."
    },
    {
      "name": "validator.sh (bpe_tokenizer)",
      "kind": "shell-script",
      "location": "curricula/cs336_a1/modules/bpe_tokenizer/validator.sh:1",
      "description": "Runs pytest for the BPE tokenizer training implementation inside a shadow worktree."
    },
    {
      "name": "validator.sh (checkpointing)",
      "kind": "shell-script",
      "location": "curricula/cs336_a1/modules/checkpointing/validator.sh:1",
      "description": "Runs pytest for the model checkpointing save/load implementation inside a shadow worktree."
    },
    {
      "name": "validator.sh (cosine_schedule)",
      "kind": "shell-script",
      "location": "curricula/cs336_a1/modules/cosine_schedule/validator.sh:1",
      "description": "Runs pytest for the cosine LR schedule implementation inside a shadow worktree."
    },
    {
      "name": "validator.sh (cross_entropy)",
      "kind": "shell-script",
      "location": "curricula/cs336_a1/modules/cross_entropy/validator.sh:1",
      "description": "Runs pytest for the numerically stable cross-entropy implementation inside a shadow worktree."
    },
    {
      "name": "validator.sh (data_loader)",
      "kind": "shell-script",
      "location": "curricula/cs336_a1/modules/data_loader/validator.sh:1",
      "description": "Runs pytest for the LM data loader (get_batch) implementation inside a shadow worktree."
    },
    {
      "name": "validate_data_loader",
      "kind": "shell-script",
      "location": "curricula/cs336_a1/modules/data_loader/validator.sh:1",
      "description": "Mastery engine validation entry point: copies utils.py to shadow worktree and runs pytest tests/test_training.py::test_get_batch."
    },
    {
      "name": "validate_embedding",
      "kind": "shell-script",
      "location": "curricula/cs336_a1/modules/embedding/validator.sh:1",
      "description": "Mastery engine validation entry point: copies layers.py to shadow worktree and runs pytest tests/test_model.py::test_embedding."
    },
    {
      "name": "validate_gradient_clipping",
      "kind": "shell-script",
      "location": "curricula/cs336_a1/modules/gradient_clipping/validator.sh:1",
      "description": "Mastery engine validation entry point: runs pytest tests/test_nn_utils.py::test_gradient_clipping in shadow worktree."
    },
    {
      "name": "validate_linear",
      "kind": "shell-script",
      "location": "curricula/cs336_a1/modules/linear/validator.sh:1",
      "description": "Mastery engine validation entry point: runs pytest tests/test_model.py::test_linear in shadow worktree."
    },
    {
      "name": "validate_multihead_attention",
      "kind": "shell-script",
      "location": "curricula/cs336_a1/modules/multihead_attention/validator.sh:1",
      "description": "Mastery engine validation entry point: runs pytest tests/test_model.py::test_multihead_self_attention_with_rope in shadow worktree."
    },
    {
      "name": "validate_rmsnorm",
      "kind": "shell-script",
      "location": "curricula/cs336_a1/modules/rmsnorm/validator.sh:1",
      "description": "Mastery engine validation entry point: runs pytest tests/test_model.py::test_rmsnorm in shadow worktree."
    },
    {
      "name": "validate_rope",
      "kind": "shell-script",
      "location": "curricula/cs336_a1/modules/rope/validator.sh:1",
      "description": "Mastery engine validation entry point: runs pytest tests/test_model.py::test_rope in shadow worktree."
    },
    {
      "name": "validate_silu",
      "kind": "shell-script",
      "location": "curricula/cs336_a1/modules/silu/validator.sh:1",
      "description": "Mastery engine validation entry point: runs pytest tests/test_model.py::test_silu_matches_pytorch in shadow worktree."
    },
    {
      "name": "validate_softmax",
      "kind": "shell-script",
      "location": "curricula/cs336_a1/modules/softmax/validator.sh:1",
      "description": "Mastery engine validation entry point: runs pytest tests/test_nn_utils.py::test_softmax_matches_pytorch in shadow worktree."
    },
    {
      "name": "validate_swiglu",
      "kind": "shell-script",
      "location": "curricula/cs336_a1/modules/swiglu/validator.sh:1",
      "description": "Mastery engine validation entry point: runs pytest tests/test_model.py::test_swiglu in shadow worktree."
    },
    {
      "name": "validate_text_generation",
      "kind": "shell-script",
      "location": "curricula/cs336_a1/modules/text_generation/validator.sh:1",
      "description": "Mastery engine validation entry point: runs pytest tests/test_generation.py::test_generate in shadow worktree."
    },
    {
      "name": "validate_tokenizer_class",
      "kind": "shell-script",
      "location": "curricula/cs336_a1/modules/tokenizer_class/validator.sh:1",
      "description": "Mastery engine validation entry point: runs pytest tests/test_tokenizer.py::test_tokenizer_class in shadow worktree."
    },
    {
      "name": "training_loop_validator",
      "kind": "script",
      "location": "curricula/cs336_a1/modules/training_loop/validator.sh:1",
      "description": "Invoked by the ValidationSubsystem to run pytest test_training.py::test_train_loop against a shadow worktree copy of training.py."
    },
    {
      "name": "transformer_block_validator",
      "kind": "script",
      "location": "curricula/cs336_a1/modules/transformer_block/validator.sh:1",
      "description": "Invoked by the ValidationSubsystem to run pytest test_model.py::test_transformer_block against a shadow worktree copy of layers.py."
    },
    {
      "name": "transformer_lm_validator",
      "kind": "script",
      "location": "curricula/cs336_a1/modules/transformer_lm/validator.sh:1",
      "description": "Invoked by the ValidationSubsystem to run pytest test_model.py::test_transformer_lm against a shadow worktree copy of layers.py."
    },
    {
      "name": "hello_world_validator",
      "kind": "script",
      "location": "curricula/dummy_hello_world/modules/hello_world/validator.sh:1",
      "description": "Invoked by the ValidationSubsystem to check that hello_world.py exists in the student workspace, returning PERFORMANCE_SECONDS."
    },
    {
      "name": "data_parsing_extraction_validator",
      "kind": "script",
      "location": "curricula/job_prep_data_annotation/modules/data_parsing_extraction/validator.sh:1",
      "description": "Invoked by the ValidationSubsystem; runs embedded Python tests for extract_coordinates() including format variation and performance checks."
    },
    {
      "name": "grid_visualization_validator",
      "kind": "script",
      "location": "curricula/job_prep_data_annotation/modules/grid_visualization/validator.sh:1",
      "description": "Invoked by the ValidationSubsystem; runs embedded Python tests for render_grid() including aliasing detection and edge coordinates."
    },
    {
      "name": "http_transport_validator",
      "kind": "script",
      "location": "curricula/job_prep_data_annotation/modules/http_transport/validator.sh:1",
      "description": "Invoked by the ValidationSubsystem; runs embedded Python tests for fetch_document() covering HTTP GET success and error-handling paths."
    },
    {
      "name": "std_lib_augmentation_validator",
      "kind": "script",
      "location": "curricula/python_for_cp/modules/std_lib_augmentation/validator.sh:1",
      "description": "Invoked by the ValidationSubsystem; runs embedded Python tests for shortest_path_bfs, dijkstra_shortest_path, and count_in_range."
    },
    {
      "name": "main",
      "kind": "script_entry_point",
      "location": "engine/main.py:2937",
      "description": "Top-level entry point that calls the Typer app; registered as the 'engine' CLI command in pyproject.toml."
    },
    {
      "name": "submit",
      "kind": "cli_command",
      "location": "engine/main.py:815",
      "description": "Auto-detects the current BJH stage (build/justify/harden) and runs the appropriate validation workflow."
    },
    {
      "name": "show",
      "kind": "cli_command",
      "location": "engine/main.py:1100",
      "description": "Read-only display of the current or specified module/problem challenge content (build prompt, justify question, or harden instructions)."
    },
    {
      "name": "start-challenge",
      "kind": "cli_command",
      "location": "engine/main.py:1156",
      "description": "Initializes the Harden stage by injecting a bug into the shadow worktree and displaying the symptom description."
    },
    {
      "name": "next",
      "kind": "cli_command",
      "location": "engine/main.py:1324",
      "description": "Deprecated command that forwards to 'show'; retained for backward compatibility."
    },
    {
      "name": "submit-build",
      "kind": "cli_command",
      "location": "engine/main.py:1354",
      "description": "Deprecated legacy command to validate a Build stage implementation; replaced by 'submit'."
    },
    {
      "name": "submit-justification",
      "kind": "cli_command",
      "location": "engine/main.py:1529",
      "description": "Deprecated legacy command to submit an inline Justify stage answer string; replaced by 'submit'."
    },
    {
      "name": "submit-fix",
      "kind": "cli_command",
      "location": "engine/main.py:1724",
      "description": "Deprecated legacy command to validate a Harden stage bug fix; replaced by 'submit'."
    },
    {
      "name": "init",
      "kind": "cli_command",
      "location": "engine/main.py:1917",
      "description": "Initializes the Mastery Engine: verifies git repo, validates curriculum, creates shadow git worktree, syncs uncommitted files, and writes initial state."
    },
    {
      "name": "curriculum-list",
      "kind": "cli_command",
      "location": "engine/main.py:2160",
      "description": "Displays all modules in the current curriculum with their completion status in a Rich table."
    },
    {
      "name": "progress-reset",
      "kind": "cli_command",
      "location": "engine/main.py:2256",
      "description": "Resets a specific module's progress back to the build stage after user confirmation."
    },
    {
      "name": "reset",
      "kind": "cli_command",
      "location": "engine/main.py:2384",
      "description": "Resets module or entire curriculum progress; --hard restores all files from the shadow worktree pristine state."
    },
    {
      "name": "cleanup",
      "kind": "cli_command",
      "location": "engine/main.py:2487",
      "description": "Removes the shadow git worktree to free disk space when finished with the curriculum."
    },
    {
      "name": "select",
      "kind": "cli_command",
      "location": "engine/main.py:2684",
      "description": "LIBRARY-mode command to set the active pattern and problem, resetting the user to the Build stage."
    },
    {
      "name": "status",
      "kind": "cli_command",
      "location": "engine/main.py:2804",
      "description": "Displays current learning progress (curriculum, module/problem, stage, completions) for both LINEAR and LIBRARY curricula."
    },
    {
      "name": "create-bug",
      "kind": "cli_command",
      "location": "engine/main.py:2862",
      "description": "Developer tool that uses LLM few-shot learning to generate a v2.1 JSON bug definition from a .patch file."
    },
    {
      "name": "inject_softmax_bug",
      "kind": "exported_function",
      "location": "engine/ast_harden/softmax_poc.py:147",
      "description": "Complete Phase-1 PoC pipeline (parse → canonicalize → inject → unparse) for the softmax no-subtract-max bug; returns (buggy_code, success)."
    },
    {
      "name": "inject_softmax_bug_v2_1",
      "kind": "exported_function",
      "location": "engine/ast_harden/softmax_v2_1.py:229",
      "description": "v2.1 two-phase pipeline that canonicalizes for matching but transforms the original AST, preserving student variable names; returns (buggy_code, success)."
    },
    {
      "name": "__main__",
      "kind": "__main__",
      "location": "engine/ast_harden/softmax_poc.py:209",
      "description": "Runnable test harness for inject_softmax_bug with two standard/alternative-naming softmax implementations."
    },
    {
      "name": "__main__",
      "kind": "__main__",
      "location": "engine/ast_harden/softmax_v2_1.py:310",
      "description": "Runnable test harness for inject_softmax_bug_v2_1 demonstrating the two-phase mapping approach on two softmax variants."
    },
    {
      "name": "make_submission.sh",
      "kind": "script",
      "location": "maintenance/make_submission.sh:1",
      "description": "Runs the test suite via pytest then zips the project directory into cs336-spring2025-assignment-1-submission.zip for submission."
    },
    {
      "name": "train_bpe",
      "kind": "exported_function",
      "location": "modes/developer/cs336_basics/bpe.py:10",
      "description": "Trains a BPE tokenizer from a text corpus and returns (vocab dict, merges list)."
    },
    {
      "name": "Tokenizer",
      "kind": "exported_class",
      "location": "modes/developer/cs336_basics/tokenizer.py:8",
      "description": "Byte-level BPE tokenizer with encode, decode, and encode_iterable backed by tiktoken GPT-2 encoding."
    },
    {
      "name": "AdamW",
      "kind": "exported_class",
      "location": "modes/developer/cs336_basics/optimizer.py:9",
      "description": "AdamW optimizer with decoupled weight decay extending torch.optim.Optimizer."
    },
    {
      "name": "transformer_lm",
      "kind": "exported_function",
      "location": "modes/developer/cs336_basics/layers.py:339",
      "description": "Full transformer language model forward pass returning logits given input token indices and a weights dict."
    },
    {
      "name": "transformer_block",
      "kind": "exported_function",
      "location": "modes/developer/cs336_basics/layers.py:223",
      "description": "Single pre-norm transformer block applying RoPE multi-head attention and SwiGLU FFN with residual connections."
    },
    {
      "name": "rope",
      "kind": "exported_function",
      "location": "modes/developer/cs336_basics/layers.py:285",
      "description": "Applies Rotary Positional Embeddings in-place to a query or key tensor."
    },
    {
      "name": "scaled_dot_product_attention",
      "kind": "exported_function",
      "location": "modes/developer/cs336_basics/layers.py:129",
      "description": "Numerically stable scaled dot-product attention with optional boolean causal mask."
    },
    {
      "name": "multihead_self_attention_with_rope",
      "kind": "exported_function",
      "location": "modes/developer/cs336_basics/layers.py:398",
      "description": "Batched causal multi-head self-attention with per-head RoPE applied to Q and K."
    },
    {
      "name": "find_chunk_boundaries",
      "kind": "exported_function",
      "location": "modes/developer/cs336_basics/pretokenization_example.py:5",
      "description": "Splits a binary file into N byte-aligned chunks at special-token boundaries for parallel pre-tokenization."
    },
    {
      "name": "softmax",
      "kind": "exported_function",
      "location": "modes/developer/cs336_basics/utils.py:5",
      "description": "Numerically stable softmax using subtract-max trick with float32 upcasting."
    },
    {
      "name": "cross_entropy",
      "kind": "exported_function",
      "location": "modes/developer/cs336_basics/utils.py:25",
      "description": "Numerically stable cross-entropy loss using log-sum-exp trick, averaged over batch."
    },
    {
      "name": "gradient_clipping",
      "kind": "exported_function",
      "location": "modes/developer/cs336_basics/utils.py:50",
      "description": "Clips gradients in-place so global L2 norm does not exceed max_l2_norm."
    },
    {
      "name": "get_lr_cosine_schedule",
      "kind": "exported_function",
      "location": "modes/developer/cs336_basics/utils.py:75",
      "description": "Returns scalar LR for a given iteration following linear warmup then cosine decay schedule."
    },
    {
      "name": "get_batch",
      "kind": "exported_function",
      "location": "modes/developer/cs336_basics/utils.py:114",
      "description": "Randomly samples (x, y) LongTensor batches from a 1D token-ID array for language model training."
    },
    {
      "name": "save_checkpoint",
      "kind": "exported_function",
      "location": "modes/developer/cs336_basics/utils.py:142",
      "description": "Serializes model and optimizer state dicts plus iteration count to a file path or file-like."
    },
    {
      "name": "load_checkpoint",
      "kind": "exported_function",
      "location": "modes/developer/cs336_basics/utils.py:154",
      "description": "Restores model and optimizer state from a checkpoint and returns the saved iteration number."
    },
    {
      "name": "train_bpe",
      "kind": "exported_function",
      "location": "modes/student/cs336_basics/bpe.py:8",
      "description": "Student stub for BPE training; raises NotImplementedError with step-by-step implementation guide."
    },
    {
      "name": "generate",
      "kind": "exported_function",
      "location": "modes/student/cs336_basics/generation.py:5",
      "description": "Student stub for autoregressive text generation with temperature/top-k/top-p; raises NotImplementedError."
    },
    {
      "name": "AdamW",
      "kind": "exported_class",
      "location": "modes/student/cs336_basics/optimizer.py:9",
      "description": "Student stub AdamW class; both __init__ and step raise NotImplementedError."
    },
    {
      "name": "Tokenizer",
      "kind": "exported_class",
      "location": "modes/student/cs336_basics/tokenizer.py:5",
      "description": "Complete Tokenizer provided to students as a working helper using tiktoken GPT-2 encoding."
    },
    {
      "name": "Tokenizer",
      "kind": "exported_class",
      "location": "modes/student/cs336_basics/tokenizer_stub.py:5",
      "description": "Student stub Tokenizer class for from-scratch BPE tokenizer implementation exercise."
    },
    {
      "name": "find_chunk_boundaries",
      "kind": "exported_function",
      "location": "modes/student/cs336_basics/pretokenization_example.py:5",
      "description": "Provided-complete file chunking helper for parallel corpus pre-tokenization (student copy)."
    },
    {
      "name": "main",
      "kind": "__main__",
      "location": "scripts/add_successful_to_golden.py:101",
      "description": "CLI entry: reads /tmp/llm_evaluation_results.json and interactively promotes successful bug definitions to the golden dataset."
    },
    {
      "name": "main",
      "kind": "__main__",
      "location": "scripts/auto_fix_drafts.py:295",
      "description": "CLI entry: applies four hardcoded fixes to draft AST injection patterns and saves corrected JSON files."
    },
    {
      "name": "main",
      "kind": "__main__",
      "location": "scripts/enrich_problems.py:373",
      "description": "CLI entry (--rate-limit, --input, --output): fetches LeetCode problem details and writes enriched canonical_curriculum.json."
    },
    {
      "name": "fetch_sources.sh",
      "kind": "shell_script",
      "location": "scripts/fetch_sources.sh:1",
      "description": "Shell entry: clones 30-Days-Of-Python and creates CP accelerator taxonomy placeholder into .sources/."
    },
    {
      "name": "main",
      "kind": "__main__",
      "location": "scripts/fix_draft_pattern.py:175",
      "description": "CLI entry: interactively walks draft AST patterns, tests them against patches, and helps the developer fix and promote each."
    },
    {
      "name": "main",
      "kind": "__main__",
      "location": "scripts/generate_ground_truth.py:274",
      "description": "CLI entry: uses gpt-4o to generate golden AST injection pattern JSON for all CS336-A1 modules that lack one."
    },
    {
      "name": "main",
      "kind": "__main__",
      "location": "scripts/generate_manifest.py:305",
      "description": "CLI entry (--validate-only, --canonical, --output): validates and regenerates manifest.json from canonical_curriculum.json."
    },
    {
      "name": "main",
      "kind": "__main__",
      "location": "scripts/generate_module.py:581",
      "description": "CLI entry (--problem-id, --all, --force, --limit-per-pattern): generates module assets for one or all problems in the cp_accelerator curriculum."
    },
    {
      "name": "batch_generate_all",
      "kind": "function",
      "location": "scripts/generate_module.py:691",
      "description": "Batch sub-entry invoked by main() when --all flag is set; generates problem directories for every pattern in the library curriculum."
    },
    {
      "name": "main",
      "kind": "__main__",
      "location": "scripts/migrate_bugs_llm.py:141",
      "description": "CLI entry: scans curricula/cs336_a1 for .patch files without JSON counterparts and generates them via BugAuthor LLM."
    },
    {
      "name": "mode",
      "kind": "shell_cli",
      "location": "scripts/mode:149",
      "description": "CLI entry: dispatches to status/switch/test sub-commands managing student↔developer workspace symlink via .active-mode."
    },
    {
      "name": "main",
      "kind": "__main__",
      "location": "scripts/parse_sources.py:604",
      "description": "CLI entry (--validate-urls): parses DSA taxonomy files and RoadmapResources.md to produce curricula/cp_accelerator/canonical_curriculum.json."
    },
    {
      "name": "main",
      "kind": "__main__",
      "location": "scripts/systematic_llm_evaluation.py:1267",
      "description": "CLI entry: runs all 21 CS336-A1 bugs through the LLM evaluator (3 attempts each) and prints statistics plus regression check."
    },
    {
      "name": "check_regression",
      "kind": "function",
      "location": "scripts/systematic_llm_evaluation.py:1222",
      "description": "Post-evaluation regression gate: compares current success count against a baseline of 3/4 bugs and warns on regressions."
    },
    {
      "name": "test_ci.sh",
      "kind": "shell_script",
      "location": "scripts/test_ci.sh:1",
      "description": "Shell entry: runs uv run pytest tests/engine/ -m 'not integration' to replicate GitHub Actions CI locally."
    },
    {
      "name": "test_library_loading",
      "kind": "__main__",
      "location": "scripts/test_library_loading.py:132",
      "description": "Test entry: exercises CurriculumManager LIBRARY loading, path resolution, and on-disk file existence assertions."
    },
    {
      "name": "main",
      "kind": "__main__",
      "location": "scripts/validate_student_stubs.py:177",
      "description": "CLI entry: validates all modes/student/**/*.py files contain NotImplementedError stubs; exits 1 if complete implementations found."
    },
    {
      "name": "verify_curriculum_manifest",
      "kind": "__main__",
      "location": "scripts/verify_curriculum_manifests.py:92",
      "description": "CLI entry: verifies curricula/cs336_a1 manifest by checking all declared module directories have required files."
    },
    {
      "name": "main",
      "kind": "__main__",
      "location": "scripts/verify_ground_truth.py:162",
      "description": "CLI entry: tests every golden AST injection pattern against its patch transformation and exits 1 if any pattern fails."
    },
    {
      "name": "__main__ (debug_shadow_worktree)",
      "kind": "script",
      "location": "tests/e2e/debug_shadow_worktree.py:81",
      "description": "Runs pytest on the debug_shadow_worktree script itself when executed directly to exercise shadow-worktree inspection logic."
    },
    {
      "name": "test_train_bpe_speed",
      "kind": "test_function",
      "location": "tests/test_train_bpe.py:8",
      "description": "Asserts BPE training on corpus.en with vocab_size=500 completes in under 1.5 seconds."
    },
    {
      "name": "test_train_bpe",
      "kind": "test_function",
      "location": "tests/test_train_bpe.py:27",
      "description": "Validates learned merges count (243-245) and vocabulary coverage (>=98%) against GPT-2 reference files."
    },
    {
      "name": "test_train_bpe_special_tokens",
      "kind": "test_function",
      "location": "tests/test_train_bpe.py:87",
      "description": "Verifies special tokens appear in vocab and are never merged into other tokens, with snapshot assertion."
    }
  ],
  "architecture": "The Mastery Engine is a Python 3.11+ CLI learning platform implementing a three-stage \"Build-Justify-Harden\" pedagogical loop, packaged as the `mastery` console script (pyproject.toml:34 -> engine.main:main at engine/main.py:2937). The CLI layer is a Typer app (engine/main.py:56) with Rich-rendered output that dispatches subcommands (init, show/next, submit, start-challenge, cleanup, flag-issue, plus legacy submit-build/justification/fix). It wires together six core subsystems imported at engine/main.py:38-51: (1) StateManager (engine/state.py) persists per-user UserProgress (engine/schemas.py CurriculumType/UserProgress) to a JSON state file with corruption handling; (2) CurriculumManager (engine/curriculum.py:31) loads manifest.json-described curricula in LINEAR (sequential modules) or LIBRARY (freeform) modes, raising CurriculumNotFound/Invalid errors; (3) WorkspaceManager (engine/workspace.py:24) implements process isolation via an ephemeral Git \"shadow worktree\" (.mastery_engine_worktree, configured engine/main.py:70-77) so user code runs against an isolated filesystem copy, applying bug patches (PatchApplicationError at engine/workspace.py:182); (4) ValidationSubsystem (engine/validator.py:26) shells out to per-problem validator.sh / pytest harnesses with timeout and execution-error handling; (5) the Harden stage (engine/stages/harden.py HardenRunner) drives runtime AST mutation through engine/ast_harden/ — GenericBugInjector (engine/ast_harden/generic_injector.py:19), a pattern_matcher, and the softmax v2.1 injector — which parse solutions into an AST, match semantic patterns, and surgically inject logic bugs described by per-problem bugs/*.json specs (with .patch and _symptom.txt companions); and (6) the Justify stage (engine/stages/justify.py JustifyRunner) plus LLMService (engine/services/llm_service.py:27) which evaluates natural-language justifications against rubrics using the OpenAI client (GPT-4o, Chain-of-Thought), degrading to a mock auto-pass mode when no API key is configured (ConfigurationError at engine/services/llm_service.py:329). Supporting code: engine/services/ast_service.py and engine/dev_tools/bug_author.py author/apply mutations, engine/utils.py provides find_project_root and helpers. Content lives under curricula/ — cs336_a1 (Stanford CS336 transformer-from-scratch, 21-22 modules listed in curricula/cs336_a1/manifest.json) and cp_accelerator (competitive-programming LeetCode patterns with a canonical_curriculum.json source-of-truth regenerated into manifest.json by scripts/generate_manifest.py) — each problem/module bundling build_prompt.txt, justify_questions.json, solution.py, test_cases.json, validator.sh and bug specs. A scripts/ directory provides the content pipeline (generate_module.py, parse_sources.py, enrich_problems.py, generate_ground_truth.py, manifest validation). Two GitHub Actions workflows (.github/workflows/tests.yml, validate_cp_manifest.yml) run pytest+ruff and enforce cp_accelerator manifest integrity. Tests under tests/ split into engine unit tests (tests/engine/), e2e BJH-loop tests (tests/e2e/), integration LLM tests, and the inherited CS336 model/tokenizer/optimizer tests. Extensive docs/ (much under docs/internal/archive session logs) and audits/ capture development history.",
  "provisionalIntent": "PROVISIONAL INTENT (Stage-1 hypothesis, to be refined in Stage 4): This project exists to be a curriculum-agnostic \"pedagogical operating system\" CLI that teaches deep technical mastery of software/ML concepts by forcing learners through a Build-Justify-Harden loop: implement a component from a spec (Build, validated by automated test harnesses), defend their conceptual understanding in natural language (Justify, graded by an LLM), and debug a deliberately AST-injected semantic bug in their own working code under Git shadow-worktree isolation (Harden). It appears built primarily as an engineering portfolio / self-study platform showcasing runtime AST mutation, process isolation, LLM-as-evaluator, and an automated content-generation pipeline, shipping with two concrete curricula (Stanford CS336 language-modeling and a competitive-programming/LeetCode accelerator). This intent is PROVISIONAL and serves as the coverage-of-intent denominator against which Stage 2 judges defects until Stage 4 confirms or revises it."
}
```
