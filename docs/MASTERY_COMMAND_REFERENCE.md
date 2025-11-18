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

### Legacy Commands (Soft-Deprecated, DO NOT USE)

**⚠️ These commands are maintained for backward compatibility only.**  
**Use the modern equivalents instead.**

#### ~~10. `mastery submit-build`~~ → Use `mastery submit`
**Status:** 🟡 Deprecated (still works, but don't use)  
**Modern Alternative:** `uv run mastery submit` (auto-detects build stage)  
**Why deprecated:** Replaced by unified `submit` command

#### ~~11. `mastery submit-justification`~~ → Use `mastery submit`
**Status:** 🟡 Deprecated (still works, but don't use)  
**Modern Alternative:** `uv run mastery submit` (auto-detects justify stage)  
**Why deprecated:** Replaced by unified `submit` command

#### ~~12. `mastery submit-fix`~~ → Use `mastery submit`
**Status:** 🟡 Deprecated (still works, but don't use)  
**Modern Alternative:** `uv run mastery submit` (auto-detects harden stage)  
**Why deprecated:** Replaced by unified `submit` command

#### ~~13. `mastery next`~~ → Use `mastery show`
**Status:** 🟡 Deprecated (still works, but don't use)  
**Modern Alternative:** `uv run mastery show` (read-only, always safe)  
**Why deprecated:** Had inconsistent behavior (read vs write)

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

### Commands You Should Use (10 Active Commands)

**Primary (5):**
- `submit` - Auto-detecting submission
- `init` - Initialize curriculum
- `status` - Check progress
- `show` - View challenge content
- `cleanup` - Remove shadow worktree

**Secondary (4):**
- `start-challenge` - Initialize harden stage
- `curriculum-list` - List all modules
- `progress-reset` - Reset specific module
- `reset` - Reset curriculum

**Dev Tools (1):**
- `create-bug` - Generate bug definitions

### Commands You Should NOT Use (4 Deprecated)

**⚠️ Soft-deprecated (kept for backward compatibility):**
- ~~`submit-build`~~ → Use `submit` instead
- ~~`submit-justification`~~ → Use `submit` instead
- ~~`submit-fix`~~ → Use `submit` instead
- ~~`next`~~ → Use `show` instead

**Why they still exist:** Prevents breaking existing scripts/workflows.  
**Should you use them:** NO - Use the modern equivalents.

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
