# Engine Critical Fixes - November 18, 2025

## Summary

Fixed three critical issues in the Mastery Engine based on forensic analysis of user session logs:

1. ✅ **Content Bug** - Sorting module had wrong test cases
2. ✅ **Path Fragility** - Engine broke when run from subdirectories
3. ⏳ **Init Idempotency** - Documented for future fix

---

## Issue 1: The "Smoking Gun" Content Bug ✅ FIXED

### Problem
The `sorting` module contained test cases for **"Number of Islands"** instead of "Sort an Array", causing `TypeError: sortArray() got an unexpected keyword argument 'grid'`.

### Root Cause
File: `curricula/cp_accelerator/modules/sorting/test_cases.json`

```json
{
  "problem": "Number of Islands",  // ← WRONG PROBLEM
  "input": { "grid": [...] }        // ← WRONG INPUT FORMAT
}
```

The validator was calling `sortArray(grid=...)` but the function expected `sortArray(nums=...)`.

### Fix
**File:** `curricula/cp_accelerator/modules/sorting/test_cases.json`

Replaced with correct Sort an Array test cases:
- 7 comprehensive test cases
- Proper `nums` input format
- Examples from LC-912 problem statement
- Edge cases: empty array, single element, reverse sorted, negatives

### Verification
```bash
$ cd curricula/cp_accelerator/modules/sorting && ./validator.sh

✓ Test 1: PASS - Example 1 from problem statement
✓ Test 2: PASS - Example 2 from problem statement
✓ Test 3: PASS - Single element
✓ Test 4: PASS - Empty array
✓ Test 5: PASS - Reverse sorted
✓ Test 6: PASS - Already sorted
✓ Test 7: PASS - With negative numbers and duplicates

Results: 7/7 passed
```

**Status:** ✅ **FIXED AND VERIFIED**

---

## Issue 2: Path Fragility ✅ FIXED

### Problem
Engine crashed when run from subdirectories:

```bash
$ cd curricula/cp_accelerator/modules/two_sum
$ uv run python -m engine.main status
❌ Error: CurriculumNotFoundError
```

### Root Cause
Multiple hardcoded relative paths:

**`engine/curriculum.py` Line 38:**
```python
CURRICULA_DIR = Path("curricula")  # ← Relative to cwd!
```

**`engine/workspace.py` Line 31:**
```python
WORKSPACE_DIR = Path("workspace")  # ← Relative to cwd!
```

**`engine/main.py` Line 72:**
```python
SHADOW_WORKTREE_DIR = Path(".mastery_engine_worktree")  # ← Relative to cwd!
```

**`engine/validator.py` Line 94:**
```python
shadow_worktree = Path('.mastery_engine_worktree')  # ← Relative to cwd!
```

When user runs from a subdirectory, the engine looks for `curricula/` inside that subdirectory, which doesn't exist.

### Fix

**Created:** `engine/utils.py` with `find_project_root()` function

```python
def find_project_root(start_path: Path = None) -> Path:
    """
    Find the project root directory by walking up the tree.
    
    Looks for markers like pyproject.toml, .git, or curricula/ directory
    to identify the project root, allowing the engine to work from any
    subdirectory.
    """
    if start_path is None:
        start_path = Path.cwd()
    
    current = start_path.resolve()
    
    # Walk up the directory tree
    for parent in [current] + list(current.parents):
        # Check for project markers
        if (parent / "pyproject.toml").exists():
            return parent
        if (parent / ".git").exists():
            return parent
        if (parent / "curricula").is_dir() and (parent / "engine").is_dir():
            return parent
    
    raise RuntimeError(
        f"Could not find project root from {start_path}"
    )
```

**Modified:** 
1. `engine/curriculum.py` - Added `__init__` method that calls `find_project_root()`
2. `engine/workspace.py` - Updated `__init__` to use project root
3. `engine/main.py` - Resolved `SHADOW_WORKTREE_DIR` from project root
4. `engine/validator.py` - Added project root detection for shadow worktree

### Verification
```bash
$ cd curricula/cp_accelerator/modules/two_sum  # Deep subdirectory
$ uv run python -m engine.main status

🎓 Mastery Engine Progress
┌───────────────────┬─────────────────────────────────┐
│ Curriculum        │ cp_accelerator                  │
│ Current Module    │ Hash Table (3/19)               │
│ Current Stage     │ BUILD                           │
└───────────────────┴─────────────────────────────────┘

✓ Engine works from any directory!
```

**Status:** ✅ **FIXED AND VERIFIED**

---

## Issue 3: Initialization Loop ⏳ DOCUMENTED

### Problem
User got stuck in initialization loop:

1. `engine init` → ❌ "Uncommitted Changes" (too strict)
2. `git add/commit` → (clean git)
3. `engine init` → ❌ "Already Initialized" (shadow worktree exists)
4. `engine cleanup` → (removes worktree)
5. `engine init` → ✅ Success (finally!)

### Root Cause
**`engine/main.py` Lines 1511-1522:**

```python
# Fails if any uncommitted changes
if result.stdout.strip():
    console.print(Panel(
        "[bold yellow]Uncommitted Changes Detected[/bold yellow]\n\n"
        "The Mastery Engine requires a clean Git working directory..."
    ))
    sys.exit(1)
```

**Lines 1524-1534:**

```python
# Fails if already initialized
if SHADOW_WORKTREE_DIR.exists():
    console.print(Panel(
        "[bold yellow]Already Initialized[/bold yellow]\n\n"
        "If you want to re-initialize, first run:\n"
        "  [cyan]engine cleanup[/cyan]"
    ))
    sys.exit(1)
```

The engine treats curriculum switching as a destructive operation rather than a lightweight context switch.

### Recommended Fix (Future Work)

**Option 1: Add `--force` flag**
```python
@app.command()
def init(
    curriculum_id: str,
    force: bool = typer.Option(False, "--force", "-f")
):
    if force:
        # Skip git clean check
        # Skip worktree exists check
        # Just update state.json to new curriculum
        pass
```

**Option 2: Make init idempotent**
```python
if SHADOW_WORKTREE_DIR.exists():
    # Instead of failing, check if user wants to switch curricula
    state_mgr = StateManager()
    current = state_mgr.load()
    
    if current.curriculum_id == curriculum_id:
        console.print("Already on this curriculum. No changes needed.")
        return
    else:
        console.print(f"Switching from {current.curriculum_id} to {curriculum_id}...")
        # Just update state file, don't recreate worktree
        current.curriculum_id = curriculum_id
        current.current_module_index = 0
        state_mgr.save(current)
        return
```

**Option 3: Relax git check**
```python
if result.stdout.strip() and not force:
    # Warn but allow init with --force flag
    console.print("[yellow]Warning: Uncommitted changes detected[/yellow]")
    if not force:
        console.print("Use --force to initialize anyway")
        sys.exit(1)
```

**Status:** ⏳ **Documented for future implementation**

---

## Bonus: Shell Alias Recommendation

### Problem
User must type `uv run python -m engine.main <command>` repeatedly.

### Solution
Add to `.bashrc` or `.zshrc`:

```bash
alias mastery="uv run python -m engine.main"
```

Now user can just type:
```bash
mastery status
mastery next
mastery submit
```

---

## Impact Summary

### Before Fixes

| Issue | Impact | Severity |
|-------|--------|----------|
| Wrong test cases | **Complete failure** - validator always crashes | 🔴 CRITICAL |
| Path fragility | Engine only works from repo root | 🟠 HIGH |
| Init loop | Hostile UX, user gets stuck | 🟡 MEDIUM |

### After Fixes

| Issue | Status | Result |
|-------|--------|--------|
| Wrong test cases | ✅ Fixed | Validator works perfectly (7/7 tests pass) |
| Path fragility | ✅ Fixed | Engine works from **any** subdirectory |
| Init loop | ⏳ Documented | Clear path forward for implementation |

---

## Files Modified

### Fixed
1. `curricula/cp_accelerator/modules/sorting/test_cases.json` - Corrected test data
2. `engine/utils.py` - **NEW** - Project root detection
3. `engine/curriculum.py` - Added project root resolution
4. `engine/workspace.py` - Added project root resolution
5. `engine/main.py` - Added project root resolution for shadow worktree
6. `engine/validator.py` - Added project root resolution

### Documented
7. `docs/ENGINE_CRITICAL_FIXES_2025-11-18.md` - This file

---

## Testing Done

### Test 1: Content Bug Fix
```bash
cd curricula/cp_accelerator/modules/sorting
./validator.sh
# ✅ Result: 7/7 tests pass
```

### Test 2: Path Fragility Fix
```bash
cd curricula/cp_accelerator/modules/two_sum  # Deep subdirectory
uv run python -m engine.main status
# ✅ Result: Works perfectly from subdirectory
```

### Test 3: Init Loop
Status: Documented for future fix. No regression introduced.

---

## Recommendations

### Immediate (For User)
1. ✅ **Content bug fixed** - Sorting module now works
2. ✅ **Path fragility fixed** - Can run from any directory
3. 💡 **Add shell alias** - `alias mastery="uv run python -m engine.main"`

### Short-term (Next Session)
1. Implement `--force` flag for `engine init`
2. Make init idempotent (curriculum switching)
3. Relax git clean check with warning instead of hard failure

### Long-term (Architecture)
1. Consider VS Code extension for better UX
2. Add `engine switch <curriculum>` command for lightweight context switching
3. Improve error messages with recovery suggestions

---

## Lessons Learned

### The Value of Forensic Analysis
Your session logs were **invaluable**. They revealed:
- The exact TypeError that led to content bug discovery
- The exact path where the engine failed
- The exact sequence of init commands that created the loop

### UX vs Safety Tradeoff
The engine prioritizes **safety** (git clean check) over **usability**. This is a valid design choice but creates friction. The solution is not to remove safety, but to provide escape hatches (`--force` flag).

### Path Fragility is Insidious
Relative paths work perfectly... until they don't. The bug only manifests when:
- User runs from a different directory
- User follows docs that say "cd into module directory"
- Integration tests run from different paths

The fix (project root detection) makes the engine **robust** regardless of cwd.

---

## Conclusion

### What Got Fixed Today
1. ✅ **Blocking bug** - Sorting module now works (7/7 tests pass)
2. ✅ **Major UX issue** - Engine now works from any directory
3. ✅ **Documentation** - Init loop issue documented with clear solution

### What's Still Needed
1. ⏳ Implement init idempotency (tracked for future work)
2. ⏳ Consider VS Code extension for ultimate UX

### Key Takeaway
**Your detailed logs led to precise fixes.** The forensic approach (smoking gun → diagnosis → fix → verify) ensured we addressed root causes, not symptoms.

---

**Fixed By:** Cascade AI  
**Date:** November 18, 2025  
**Status:** 2/3 Critical Issues Resolved, 1/3 Documented for Future Work
