# Mastery Engine Quality & Resilience Audit

**Auditor:** `Cascade`
**Date Started:** `2025-12-18`

This document serves as the primary artifact for a systematic code audit of the `Mastery Engine` repository. Its purpose is to validate the architectural integrity, pedagogical efficacy, security, and resilience of the system, identifying technical debt and risks before they compound.

---

## Audit Findings Log

*Log observations here as they are discovered during the execution of the checklist below.*

| Component | Category | Finding / Observation | Severity | Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| `engine/main.py` | Architecture | The `submit` command logic (`_submit_*_stage`) contains significant business logic mixed with CLI presentation code, rather than delegating fully to `stages/` runners. | Medium | Refactor `_submit_build_stage` logic into a dedicated `BuildRunner` class in `engine/stages/build.py` to match `HardenRunner` pattern. |
| `engine/workspace.py` | Resilience | `create_harden_workspace` relies on `shutil.copy2` which may not preserve all file attributes/permissions needed for complex setups, though sufficient for current Python files. | Low | Verify if execution permissions are needed for any copied artifacts; documented assumption. |
| `engine/ast_harden` | Cleanup | Presence of `softmax_poc.py`, `softmax_v2_1.py` alongside `generic_injector.py` suggests prototype code mixed with production code. | Low | Move PoC files to a `prototypes/` directory or delete them to avoid confusion about which injector is active. |
| `scripts/mode` | UX/Safety | The symlink swapping mechanism is effective but relies on the user not having open file handles or running processes inside the `cs336_basics` directory, which can cause confusion on some OSs. | Medium | Add a check or warning in the script if the directory is currently in use/locked (especially relevant for Windows WSL2). |
| `engine/schemas.py` | Correctness | `UserProgress.mark_stage_complete("harden")` appends placeholder IDs like `module_0` instead of real module IDs, causing `completed_modules`-based features (`curriculum-list`, `progress-reset`) to misreport. | High | Pass the real `module_id` into the state transition (or store current module id in state) and append that instead. |
| `engine/main.py` | Resilience | Shadow worktree path handling is inconsistent: `require_shadow_worktree()` checks `SHADOW_WORKTREE_DIR` (project-root-based), but `_submit_harden_stage()` and `HardenRunner` use `Path(".mastery_engine_worktree")`, which can break when running from subdirectories. | High | Centralize all shadow worktree references on `SHADOW_WORKTREE_DIR` (or the `require_shadow_worktree()` return value) and avoid `Path(".mastery_engine_worktree")`. |
| `engine/main.py` | UX/Consistency | User-facing guidance strings mix `engine` and `mastery` command names (e.g. “Run `mastery submit`” inside `engine submit`). | Medium | Standardize messaging to the installed CLI name (or define a single constant/alias and reuse). |
| `engine/main.py` | Error Handling | `_submit_justify_stage()` prints an error Panel then re-raises; the `submit()` wrapper catches and prints another error Panel, duplicating output. | Low | Either raise without printing inside stage handlers, or handle/print once at the top-level. |
| `engine/curriculum.py` | Schema Validation | `CurriculumManifest` does not enforce that `modules` is present for `type="linear"` (or `patterns` for `type="library"`), so malformed curricula may validate but later crash. | Medium | Add Pydantic validation enforcing conditional required fields based on `type`. |
| `engine/services/llm_service.py` | Security/Cost | LLM prompt embeds raw user answer without strong delimiting; `evaluate_justification()` does not set `max_tokens`, risking prompt injection and unbounded cost. | Medium | Delimit user content (e.g., fenced blocks / XML tags) and set a conservative `max_tokens` for evaluation responses. |
| `engine/state.py` | Resilience | State file schema has no versioning/migration path; future schema changes can hard-fail old state loads. | Medium | Add `schema_version` to state file with migration/compat handling (or a graceful reset path). |
| `engine/ast_harden/pattern_matcher.py` | Correctness | Unknown `replacement.type` values fall through and are treated as deletion (`None`), so unsupported operations can silently delete nodes and still count as a “successful replacement”. | High | Validate `replacement.type` and raise on unsupported values; do not treat unknown types as deletion. |
| `curricula/cp_accelerator/.../insert_before_check.json` | Correctness | Bug definition uses `replacement.type: "move_after"`, which is not implemented by the AST injector, so injection behavior will be incorrect (currently interpreted as deletion). | High | Either implement `move_after` (statement reorder) or rewrite the bug to use supported replacement operations. |
| `engine/ast_harden/generic_injector.py` | Architecture | `find_and_replace` passes currently match directly in the original AST (`transformer.visit(original_ast)`) despite comments describing a canonical two-phase approach, reducing robustness to variable renaming. | Medium | Use the two-phase `transform_original()` with the canonical AST (or update docs/comments and ensure patterns are authored for original AST only). |
| `engine/ast_harden/generic_injector.py` | Correctness | `target_function` is only checked for existence; pattern matching/transforms are not scoped to that function body, so matches elsewhere could inject bugs in the wrong location. | Medium | Scope matching/transforms to the target function node (or add an explicit scoping mechanism). |
| `engine/main.py` | Resilience | `init` syncs only modified tracked files (`git ls-files -m`) into the shadow worktree, ignoring untracked files and deletions; validation env can still diverge from the user workspace. | Medium | Sync untracked files (opt-in) and handle deletions/renames, or document the limitation clearly. |
| `engine/schemas.py` | Schema/Tooling | The Pydantic `BugDefinition` schema (extra=forbid + limited `Pattern` fields) does not match observed bug JSON shapes (e.g., `Compare.ops`, richer `metadata`), and runtime injection does not validate against it. | Medium | Align schema with real bug files and validate at load time, or clearly separate “authoring schema” vs “runtime contract”. |

---

## Component Audit Checklist

### I. Core Architecture & CLI Orchestration (`engine/`)

- [x] **Entry Point (`main.py`)**
    - **Separation of Concerns:** Does `main.py` strictly handle CLI arguments and UI rendering (Rich), or does it leak business logic? (Check `_submit_linear_workflow` and `_submit_library_workflow`).
    - **Error Handling:** Are exceptions from the underlying layers (e.g., `CurriculumError`, `StateError`) caught and converted to user-friendly messages without exposing raw stack traces (unless debug mode is on)?
    - **State Management:** Is the transition logic between `build` -> `justify` -> `harden` explicit and robust? Does it handle edge cases (e.g., user interrupts process mid-transition)?

- [x] **State Persistence (`state.py`)**
    - **Atomicity:** Verify the "write-to-temp-then-rename" pattern in `save()`. Is it truly atomic on all target filesystems (specifically WSL2)?
    - **Schema Evolution:** How does the system handle loading an old `.mastery_progress.json` format? Is there versioning or migration logic?
    - **Corruption Recovery:** If the state file is unparseable (e.g., half-written), does `load()` fail gracefully or offer a reset path?

- [x] **Curriculum Loading (`curriculum.py`)**
    - **Path Resolution:** Does `find_project_root` robustly locate the root when running from deep subdirectories (e.g., inside a module folder)?
    - **Schema Validation:** Are `manifest.json` files strictly validated against `CurriculumManifest`? Does it fail fast on invalid dependencies?
    - **Caching:** In `LIBRARY` mode, are pattern/problem lookups efficient (`O(1)`)? Is the cache invalidation logic handling curriculum updates correctly?

### II. The Pedagogical Loop (`engine/stages/`)

- [x] **Build Stage**
    - **Logic Location:** *Critical Check:* Is the build logic properly encapsulated? Current code analysis suggests it lives inside `main.py`. Verify if `stages/build.py` exists and if logic should be moved there.
    - **Validator Interface:** Does the system correctly parse both exit codes and `PERFORMANCE_SECONDS` from standard output?

- [x] **Justify Stage (`justify.py`)**
    - **Fast Filter:** Is the regex/keyword matching in `check_fast_filter` case-insensitive and robust against minor variations?
    - **LLM Fallback:** Does the system seamlessly handle network failures during LLM evaluation? Is the Mock Mode trigger reliable?

- [x] **Harden Stage (`harden.py`)**
    - **Isolation:** Verify that `present_challenge` correctly copies the *developer* reference implementation to the shadow worktree, not the *student's* potentially broken code, to ensure the patch applies cleanly.
    - **Patch Reliability:** Does the system handle cases where the `patch` utility is missing or fails due to whitespace issues?

### III. AST Mutation Engine (`engine/ast_harden/`)

- [x] **Generic Injector (`generic_injector.py`)**
     - **Parsing Robustness:** Can the injector handle code with comments, decorators, or unusual formatting without breaking the AST?
     - **Round-Trip Fidelity:** Does `ast.unparse` (or `astor`) preserve code structure sufficiently well? Does it strip comments that might be pedagogically useful?
     - **Pattern Matching (`pattern_matcher.py`)**: Are the node matching rules specific enough to avoid false positives (e.g., matching the wrong `Assign` node)?

- [x] **Bug Definitions (`.json`)**
     - **Schema Compliance:** Do all JSON files in `curricula` match the v2.1 schema expected by `GenericBugInjector`?
     - **Target Function:** Is the `target_function` field correctly used to scope the injection, or does it scan the whole file?

### IV. Workspace & Isolation (`engine/workspace.py`)

- [x] **Shadow Worktree Strategy**
     - **Symlink Handling:** *Critical:* Verify the fix for symlink copying in `git worktree`. Does `engine init` correctly recreate symlinks (e.g., `cs336_basics`) inside `.mastery_engine_worktree`?
     - **Dirty State:** How does the engine handle uncommitted changes in the main repo when synchronizing to the shadow worktree? (See `main.py` -> `init`).
     - **Cleanup:** Does `cleanup` leave the repo in a clean git state? Does it prune worktree metadata?

- [x] **File Operations**
     - **Permissions:** Does `apply_patch` require specific file permissions?
     - **Path Traversal:** Are inputs like `module_id` sanitized to prevent writing files outside the workspace?

### V. External Services & Integration (`engine/services/`)

- [x] **LLM Service (`llm_service.py`)**
    - **JSON Mode:** Is `response_format={"type": "json_object"}` strictly enforced to prevent parsing errors?
    - **Prompt Injection:** Are user answers sanitized or delimited (e.g., XML tags) to prevent prompt injection attacks against the evaluator?
    - **Cost Control:** Are token limits (`max_tokens`) set appropriately to prevent runaway costs?

### VI. Content Integrity (`curricula/`)

- [ ] **Canonical Source (CP Accelerator)**
    - **Synchronization:** Is there a guarantee that `manifest.json` is perfectly synced with `canonical_curriculum.json`? (Check CI workflow `validate_cp_manifest.yml`).
    - **Dependency Cycles:** Does the topological sort in `generate_manifest.py` correctly catch all cycles?

- [ ] **Validator Scripts (`validator.sh`)**
    - **Execution Environment:** Do they rely on environment variables (like `PYTHONPATH`) that might differ between user shells?
    - **Timeout Safety:** Does `engine/validator.py` enforce a strict timeout to prevent infinite loops in student code from hanging the engine?

### VII. Testing Infrastructure (`tests/`)

- [ ] **Test Isolation**
    - **Mocking:** Do engine unit tests correctly mock the file system and `subprocess` calls?
    - **Integration:** Do E2E tests (`test_main_workflows_real.py`) actually exercise the file system logic, or are they mocking too much?

- [ ] **Stranger Testing**
    - **Reproducibility:** Does the test suite run cleanly on a fresh clone without existing virtual environments or config files? (Reference `STRANGER_TEST_RESULTS.md`).