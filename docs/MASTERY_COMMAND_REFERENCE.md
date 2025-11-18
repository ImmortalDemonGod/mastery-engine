# Mastery Command Reference

**Last Updated:** November 18, 2025  
**Entry Point:** `mastery` (via `uv run mastery`)

---

## ✅ Verified Working Commands (14 Total)

All commands verified functional as of Nov 18, 2025.

### Primary Commands

#### 1. `mastery submit`
**Status:** ✅ Working  
**Usage:** `uv run mastery submit`  
**Description:** Auto-detects current stage (build/justify/harden) and runs appropriate validation.

#### 2. `mastery init <curriculum_id>`
**Status:** ✅ Working  
**Usage:** `uv run mastery init cp_accelerator [--force]`  
**Description:** Initialize engine with a curriculum. Creates shadow worktree, syncs uncommitted changes.  
**Flags:**
- `--force` / `-f` - Force re-initialization (skip already-initialized check)

#### 3. `mastery status`
**Status:** ✅ Working  
**Usage:** `uv run mastery status`  
**Description:** Display current learning progress.

#### 4. `mastery show [module_id]`
**Status:** ✅ Working  
**Usage:** `uv run mastery show` or `uv run mastery show two_sum`  
**Description:** Display challenge content (read-only, never modifies files).

#### 5. `mastery cleanup`
**Status:** ✅ Working  
**Usage:** `uv run mastery cleanup`  
**Description:** Remove shadow worktree (use when completely done with curriculum).

---

### Secondary Commands

#### 6. `mastery start-challenge`
**Status:** ✅ Working  
**Usage:** `uv run mastery start-challenge`  
**Description:** Initialize Harden stage workspace (applies bug patch).

#### 7. `mastery curriculum-list`
**Status:** ✅ Working  
**Usage:** `uv run mastery curriculum-list`  
**Description:** List all modules with status (✅ complete, 🔵 in progress, ⚪ not started).

#### 8. `mastery progress-reset <module_id>`
**Status:** ✅ Working  
**Usage:** `uv run mastery progress-reset two_sum`  
**Description:** Reset specific module to start over (requires confirmation).

#### 9. `mastery reset`
**Status:** ✅ Working  
**Usage:** `uv run mastery reset`  
**Description:** Reset entire curriculum or specific module.

---

### Legacy Commands (Still Work)

#### 10. `mastery submit-build`
**Status:** ✅ Working (use `submit` instead)  
**Usage:** `uv run mastery submit-build`  
**Description:** Submit Build stage implementation.  
**Note:** `mastery submit` auto-detects and does this automatically.

#### 11. `mastery submit-justification <answer>`
**Status:** ✅ Working (use `submit` instead)  
**Usage:** `uv run mastery submit-justification "Your answer"`  
**Description:** Submit Justify stage answer.  
**Note:** `mastery submit` opens editor automatically.

#### 12. `mastery submit-fix`
**Status:** ✅ Working (use `submit` instead)  
**Usage:** `uv run mastery submit-fix`  
**Description:** Submit Harden stage bug fix.  
**Note:** `mastery submit` auto-detects and does this automatically.

#### 13. `mastery next`
**Status:** ✅ Working (DEPRECATED - use `show` instead)  
**Usage:** `uv run mastery next`  
**Description:** Display next challenge.  
**Note:** Use `mastery show` for read-only display.

---

### Developer Tools

#### 14. `mastery create-bug <module>`
**Status:** ✅ Working  
**Usage:** `uv run mastery create-bug <module>`  
**Description:** [DEV TOOL] Generate JSON bug definition from patch file using LLM.

---

## Installation

```bash
# Install package in editable mode
uv pip install -e .

# Verify installation
uv run mastery --help
```

---

## Important Notes

### Command Name: `mastery` NOT `engine`

**OLD (Documentation):** `engine submit`  
**ACTUAL (Command):** `uv run mastery submit`

The entry point is `mastery`, not `engine`. Documentation using `engine <command>` is **outdated**.

### Recommended Invocation

```bash
# Recommended (works from any directory, venv-aware)
uv run mastery <command>

# Alternative (requires venv activation)
source .venv/bin/activate
mastery <command>
```

### Path Independence

All commands work from **any subdirectory** in the project:

```bash
cd curricula/cp_accelerator/modules/two_sum
uv run mastery status  # ✅ Works!
```

---

## Complete Workflow Example

```bash
# 1. Initialize with curriculum
uv run mastery init cp_accelerator

# 2. Check status
uv run mastery status

# 3. View current challenge
uv run mastery show

# 4. Work on your solution...
# (edit files in workspace/)

# 5. Submit (auto-detects stage)
uv run mastery submit

# 6. Move to next module (if passed)
uv run mastery status

# 7. Repeat steps 3-6

# 8. When completely done
uv run mastery cleanup
```

---

## Common Issues

### Issue 1: "command not found: mastery"

**Solution:** Use `uv run mastery` instead of bare `mastery`.

### Issue 2: Old docs say "engine"

**Solution:** Replace `engine` with `uv run mastery` in all commands.

### Issue 3: Command fails with "Not Initialized"

**Solution:** Run `uv run mastery init <curriculum_id>` first.

---

## Summary

### ✅ All Commands Functional

All 14 commands are registered and working:
- 5 primary commands (submit, init, status, show, cleanup)
- 4 secondary commands (start-challenge, curriculum-list, progress-reset, reset)
- 4 legacy commands (submit-build, submit-justification, submit-fix, next)
- 1 dev tool (create-bug)

### 📝 Documentation Discrepancy

**Old docs:** Use `engine <command>`  
**Actual command:** Use `uv run mastery <command>`

### 🎯 Recommended Usage

```bash
uv run mastery <command>
```

This works from any directory and is venv-aware.

---

**Verification Date:** November 18, 2025  
**Test Coverage:** All 14 commands tested and functional  
**Entry Point:** `mastery` (defined in pyproject.toml)
