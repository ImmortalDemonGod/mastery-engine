# Meta Audit: Coverage Gaps in `QUALITY_AUDIT.md`

**Auditor:** `Cascade`
**Date:** `2025-12-18`

## Purpose

This document exists for one reason:

- Identify areas of the repository that were **not explicitly audited** (or were only **lightly sampled**) by `audits/QUALITY_AUDIT.md`.

It is intentionally **not** a re-audit of code quality. It is a map of **audit blind spots**.

## Coverage Labels

- **Explicitly covered**
  - Named in `QUALITY_AUDIT.md` checklist/findings, with clear evidence of review.
- **Partially covered**
  - Falls under a checked checklist item, but only representative samples were inspected (not systematic across all instances).
- **Not covered**
  - Not referenced in `QUALITY_AUDIT.md` and no clear evidence of review.

## Quick Summary (Highest-Risk Coverage Gaps)

- **`scripts/` automation toolchain is largely unaudited**
  - Many scripts can mutate curriculum/code, call LLMs, or touch external resources.
- **Developer tooling and bug-authoring pipeline**
  - `engine/dev_tools/bug_author.py` and `engine create-bug` path were not reviewed for safety, schema compatibility, or cost controls.
- **Most curriculum packs were not audited**
  - The audit focused on `cs336_a1` + `cp_accelerator` concerns; other curricula were not evaluated.
- **`modes/` content not audited**
  - Student/developer code in `modes/*/cs336_basics` is a primary product surface but was not reviewed.
- **Docs accuracy not audited**
  - Large architecture/internal docs were not cross-checked against current code.
- **Supply chain / dependency / secrets posture not audited**
  - `uv.lock`, `.env`, license/compliance checks, dependency pinning strategy not systematically assessed.

## Coverage Map (Repository Inventory vs Quality Audit)

| Repository Area | Examples | Coverage | Why This Matters | Follow-up Audit |
| --- | --- | --- | --- | --- |
| `engine/` core runtime | `engine/main.py`, `state.py`, `curriculum.py`, `validator.py`, `workspace.py` | Explicitly covered | This is the execution path users hit every run. | None (already covered); only re-audit if major refactor lands. |
| `engine/stages/` runners | `engine/stages/harden.py`, `engine/stages/justify.py` | Explicitly covered | Pedagogical loop behavior + file mutation points. | Add follow-up only if new stages are added. |
| `engine/ast_harden/` | `generic_injector.py`, `pattern_matcher.py` | Explicitly covered | Drives AST bug injection correctness/safety. | Follow-up: unit tests for unsupported operations + scoping. |
| `engine/services/` beyond `llm_service.py` | `engine/services/ast_service.py` | Partially covered | Critical to canonicalization and injection correctness, but not explicitly discussed. | Add explicit audit section for canonicalization strategy + invariants. |
| `engine/dev_tools/` | `engine/dev_tools/bug_author.py` | Not covered | Dev tool can generate/overwrite bug definitions via LLM; high correctness/cost risk. | Audit schema alignment, prompt delimiting, cost limits, and file write safety. |
| `scripts/` (most utilities) | `enrich_problems.py`, `parse_sources.py`, `generate_manifest.py`, `generate_module.py`, `migrate_bugs_llm.py`, `systematic_llm_evaluation.py` | Not covered | Scripts can be destructive and/or call network/LLM; often run manually without CI guardrails. | Safety audit: inputs/outputs, idempotency, dry-run support, backups, secrets handling. |
| `curricula/cs336_a1/` content depth | per-module `build_prompt.txt`, `justify_questions.json`, `bugs/*.json`, `validator.sh` | Partially covered | Only sampled validators/bugs; no systematic consistency check across all modules. | Curriculum integrity audit: schema conformance, stage asset presence, dependency correctness, validator contract consistency. |
| `curricula/cp_accelerator/` content depth | `patterns/*/theory`, `problems/*`, all bug definitions/validators | Partially covered | Manifest + CI issues noted, but not systematic check of all patterns/problems. | Library-mode audit: unique problem identity, validator env consistency, bug op support, content completeness. |
| Other curricula packs | `curricula/python_for_cp`, `job_prep_data_annotation`, `dummy_hello_world` | Not covered | Users can initialize these curricula; failures become user-facing. | Quick manifest + asset presence audit; verify validators exist and run. |
| `modes/` code | `modes/student/cs336_basics`, `modes/developer/cs336_basics` | Not covered | This is the code users actually edit; mismatch breaks pedagogy/tests. | Audit mode switching guarantees, file parity, and “reference” expectations used in harden. |
| `tests/` outside `tests/engine/` | `tests/test_*` (assignment tests), `tests/integration/` | Not covered | CI currently excludes these; regressions could slip in learning modules. | Define test scope policy; optionally add CI jobs for non-engine tests and/or nightly full suite. |
| E2E execution in CI | `tests/e2e/` | Partially covered | E2E exists but not executed in CI; only documented. | Add CI smoke E2E subset or nightly run; stabilize temp-repo edge cases. |
| Docs accuracy vs code | `docs/architecture/MASTERY_ENGINE.md`, `docs/user-guide/*`, `docs/internal/*` | Not covered | Docs can drift; users follow docs, not code. | Documentation drift audit: verify commands/paths match current CLI behavior. |
| `maintenance/` scripts | `maintenance/make_submission.sh` | Not covered | Release/support tooling can break silently and block deliverables. | Audit for safety, portability, and correct outputs. |
| Root packaging + dependency management | `pyproject.toml`, `uv.lock` | Partially covered | Python version mismatch was noted; deeper supply-chain posture not covered. | Audit dependency policy: pinning, reproducibility, license checks, and CI alignment. |
| Secrets/config hygiene | `.env`, `.env.example`, logging to `~/.mastery_engine.log` | Not covered | Potential leak vectors and portability issues. | Audit secret handling, logging hygiene, and “safe defaults”. |

## Detailed Gap Inventory (Actionable)

### 1) `scripts/` safety + correctness audit (Not covered)

Focus areas:
- Which scripts mutate files in-place vs generate outputs.
- Whether scripts are idempotent.
- External calls (network, LLM): do they require keys, do they respect rate/cost controls.
- Whether scripts have a safe “dry-run” mode.

#### Preliminary file-level inventory (for follow-up audit scoping)

This is **not** a quality judgment of each script. It is an inventory of **side effects and hazards** that indicate why the `scripts/` toolchain needs its own dedicated audit.

- **Network / External APIs**
  - `scripts/enrich_problems.py`
    - Calls `https://leetcode-api-pied.vercel.app` via `requests`.
    - Default `--output` overwrites `curricula/cp_accelerator/canonical_curriculum.json` in-place.
    - No dry-run/backup by default; no caching.
  - `scripts/fetch_sources.sh`
    - Uses `git clone` (network) and deletes nested `.git` folders.

- **LLM / Cost-bearing operations**
  - `scripts/migrate_bugs_llm.py`
    - Uses `BugAuthor()` (LLM-backed) to batch-convert `*.patch` → `*.json` and writes into `curricula/`.
    - No dry-run; overwrite behavior is implicit (skips only if JSON exists).
  - `scripts/systematic_llm_evaluation.py`
    - Uses `LLMService` + `BugAuthor`; writes results to `/tmp/llm_evaluation_results.json`.
    - Uses hardcoded absolute curriculum paths under `/Volumes/Totallynotaharddrive/...`.
  - `scripts/generate_ground_truth.py`
    - Uses `LLMService` + `BugAuthor`; writes/overwrites JSON bug files and draft files.
    - Uses hardcoded absolute curriculum paths under `/Volumes/Totallynotaharddrive/...`.

- **Destructive file writes into `curricula/`**
  - `scripts/generate_manifest.py`
    - Writes `curricula/cp_accelerator/manifest.json`.
    - Has `--validate-only`, but generation is destructive if run without it.
    - Does not enforce uniqueness of LIBRARY problem IDs (known cache ambiguity).
  - `scripts/generate_module.py`
    - Creates directories and writes `build_prompt.txt`, `test_cases.json`, `justify_questions.json`, `validator.sh`.
    - Uses `chmod(0o755)` on validators.

- **Hardcoded absolute paths (non-portable)**
  - `scripts/systematic_llm_evaluation.py`, `scripts/generate_ground_truth.py`, `scripts/auto_fix_drafts.py`, `scripts/fix_draft_pattern.py`, `scripts/verify_ground_truth.py`, `scripts/add_successful_to_golden.py`
    - Use absolute paths rooted at `/Volumes/Totallynotaharddrive/assignment1-basics/...`.
    - Follow-up audit should check portability expectations: are these scripts intended only for one workstation?

- **Unsafe parsing (`eval`)**
  - `scripts/generate_module.py`
    - Uses `eval()` to parse example input/output strings. This is unsafe if inputs can be attacker-controlled (or if canonical/enriched JSON is tampered).
    - Follow-up audit should require `ast.literal_eval` (or a purpose-built parser) with strict input validation.

- **Interactive scripts (not CI-safe; can cause accidental overwrites)**
  - `scripts/fix_draft_pattern.py`, `scripts/add_successful_to_golden.py`
    - Use `input()` prompting and can write/overwrite “golden” artifacts.

- **Inconsistencies / drift risks**
  - `scripts/parse_sources.py`
    - Documents `--validate-urls` but does not implement URL validation logic.
    - References `RoadmapResources.md` at repo root, but the file lives at `maintenance/RoadmapResources.md`.
  - `scripts/verify_curriculum_manifests.py`
    - Validates only LINEAR-style `manifest["modules"]` and does not handle LIBRARY curricula (`patterns`).

### 2) `engine/dev_tools/bug_author.py` + `engine create-bug` audit (Not covered)

Focus areas:
- Output schema compatibility: ensure generated JSON matches runtime injector expectations.
- Prompt injection/cost controls (especially since it uses `gpt-4o`).
- File write safety (path traversal, overwrite policy).

### 3) LIBRARY mode uniqueness + addressing audit (Partially covered)

Even if CI and caching issues are fixed, follow-up audit should verify:
- Problem addressing strategy is consistent across:
  - manifest generation (`scripts/generate_manifest.py`)
  - engine caches (`engine/curriculum.py`)
  - CLI (`engine/main.py select`)
  - file layout (`curricula/cp_accelerator/patterns/...`)

### 4) Curriculum pack audits beyond cs336_a1/cp_accelerator (Not covered)

Minimum bar for each curriculum:
- `manifest.json` validates against engine schema.
- Each module/problem has required stage artifacts.
- Each validator runs under `ValidationSubsystem` contracts.

### 5) Mode parity audit (Not covered)

Validate:
- The `scripts/mode` switch guarantees a coherent package layout.
- Developer reference implementations exist for all patch-based bugs.
- Student mode provides the expected stubs and doesn’t drift from tests.

### 6) CI scope audit (Partially covered)

Validate:
- CI runs with a Python version consistent with `pyproject.toml`.
- CI covers the intended test matrix (engine-only vs full repo) and documents what is excluded.

## Recommended Follow-up Audit Backlog (Prioritized)

1. **Audit `scripts/` safety + external side effects** (high)
2. **Audit bug authoring toolchain (`engine/dev_tools` + `create-bug`)** (high)
3. **Audit LIBRARY mode uniqueness constraints end-to-end** (high)
4. **Audit other curricula packs for baseline operability** (medium)
5. **Audit `modes/` parity and guarantees** (medium)
6. **Audit docs drift vs current CLI behavior** (medium)
7. **Audit dependency/supply-chain posture (`uv.lock`, licensing, pinning)** (medium)

