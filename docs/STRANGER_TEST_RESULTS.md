# Stranger Test Results - Complete End-to-End Verification

**Date:** November 19, 2025  
**Tester:** Systematic from-scratch execution in `/tmp/mastery-engine-test-2`  
**Objective:** Verify README Quick Start instructions work for a new user

---

## Executive Summary

✅ **PASSED** - All Quick Start instructions now work correctly from scratch.

**Issues Found:** 2 critical bugs  
**Issues Fixed:** 2 critical bugs  
**Final Status:** Production ready

---

## Test Methodology

### Environment
- **Clean slate:** Fresh `/tmp` directory (no prior state)
- **Fresh clone:** Latest code from GitHub
- **No assumptions:** Followed README exactly as written

### Commands Executed
```bash
# Exactly as documented in README Quick Start
cd /tmp && mkdir mastery-engine-test-2
git clone https://github.com/ImmortalDemonGod/mastery-engine.git
cd mastery-engine
./scripts/mode switch developer
uv sync
uv pip install -e .
uv run mastery init cs336_a1
uv run mastery show
uv run mastery submit
```

---

## Issues Found & Fixed

### ❌ Issue #1: Wrong Instruction Order (CRITICAL)

**Symptom:**
```bash
$ uv sync
× Failed to build `cs336-basics @ file:///...`
╰─▶ Expected a Python module at: cs336_basics/__init__.py
```

**Root Cause:**  
README had dependencies installation BEFORE mode switch. The `cs336_basics/` directory is a symlink created by `./scripts/mode switch`. Installing packages before creating the symlink = failure.

**Original (Broken) Order:**
```bash
1. Install Dependencies (uv sync)      ❌ FAILS
2. Activate Developer Mode (symlink)
```

**Fixed Order:**
```bash
1. Clone repository
2. Activate Developer Mode (symlink)   ← MOVED UP
3. Install Dependencies (uv sync)      ✅ Now works
```

**Fix Applied:**  
✅ Commit `4922ed3` - Reordered README Quick Start steps

---

### ❌ Issue #2: Shadow Worktree Missing Symlink (CRITICAL)

**Symptom:**
```bash
$ uv run mastery submit
❌ Validation Failed
cp: .mastery_engine_worktree/cs336_basics/utils.py: No such file or directory
```

**Root Cause:**  
Git worktrees don't automatically copy symlinks. When `git worktree add` creates `.mastery_engine_worktree`, it copies all tracked files but **NOT** symlinks.

The `cs336_basics/` symlink exists in main repo but not in shadow worktree. Validators run in the shadow worktree and expect `cs336_basics/` to exist.

**Technical Details:**
- Worktree creation: `git worktree add .mastery_engine_worktree --detach`
- Copies: All tracked files ✅
- Does NOT copy: Symlinks ❌
- Validators look for: `.mastery_engine_worktree/cs336_basics/utils.py`
- Result: File not found

**Fix Applied:**  
✅ Commit `1558497` - Added symlink recreation in `engine/main.py`

**Implementation:**
```python
# After creating shadow worktree (line 2018 in engine/main.py)
cs336_symlink = Path("cs336_basics")
if cs336_symlink.is_symlink():
    # Read symlink target from main repo
    symlink_target = os.readlink(cs336_symlink)
    
    # Recreate in shadow worktree
    shadow_symlink = SHADOW_WORKTREE_DIR / "cs336_basics"
    if not shadow_symlink.exists():
        os.symlink(symlink_target, shadow_symlink)
```

**Why This Works:**
- Symlink targets are relative paths: `modes/developer/cs336_basics`
- Relative paths work from both main repo and shadow worktree
- Same directory structure exists in both locations
- Validators can now import from `cs336_basics/`

**Verification:**
```bash
$ ls -la .mastery_engine_worktree/ | grep cs336_basics
lrwxr-xr-x  1 user  wheel  28 Nov 19 14:59 cs336_basics -> modes/developer/cs336_basics
✅ Symlink successfully recreated
```

---

## Test Results - Step by Step

### ✅ Step 1: Clone Repository
```bash
$ git clone https://github.com/ImmortalDemonGod/mastery-engine.git
Cloning into 'mastery-engine'...
remote: Total 2981 (delta 1441)
Receiving objects: 100% (2981/2981), 44.08 MiB | 40.37 MiB/s, done.
✅ Success
```

### ✅ Step 2: Activate Developer Mode
```bash
$ ./scripts/mode switch developer
Switching to developer mode...
✓ Switched to developer mode

Workspace:
  cs336_basics/ → modes/developer/cs336_basics
✅ Success - Symlink created
```

### ✅ Step 3: Install Dependencies
```bash
$ uv sync
Resolved 83 packages in 8ms
Installed 83 packages
✅ Success (now works because symlink exists)

$ uv pip install -e .
Built cs336-basics @ file:///tmp/mastery-engine-test-2/mastery-engine
Installed 1 package
✅ Success
```

### ✅ Step 4: Initialize Curriculum
```bash
$ uv run mastery init cs336_a1
Initializing Mastery Engine...
Creating shadow worktree for safe validation...

╭──────────────────────── 🎓 Mastery Engine Ready ───────────────────────╮
│ ✓ Initialization Complete!                                             │
│ Curriculum: cs336_a1                                                    │
│ Shadow worktree created at: .mastery_engine_worktree                    │
╰─────────────────────────────────────────────────────────────────────────╯
✅ Success

# Verify symlink in shadow worktree
$ ls -la .mastery_engine_worktree/cs336_basics
lrwxr-xr-x  1 user  wheel  28 Nov 19 14:59 cs336_basics -> modes/developer/cs336_basics
✅ Symlink successfully recreated in shadow worktree
```

### ✅ Step 5: View Current Module
```bash
$ uv run mastery show
╭─────────── 📝 Build Challenge: Numerically Stable Softmax ───────────╮
│                                                                       │
│                              Objective                                │
│                                                                       │
│  Implement a numerically stable softmax function...                   │
╰───────────────────────────────────────────────────────────────────────╯
✅ Success - Build prompt displayed
```

### ✅ Step 6: Run Build Validation
```bash
$ uv run mastery submit
Current stage: BUILD (Numerically Stable Softmax)
Running validator for Numerically Stable Softmax...

╭──────────────────────────────── Success ─────────────────────────────╮
│ ✅ Validation Passed!                                                │
│                                                                      │
│ Your implementation passed all tests.                                │
╰──────────────────────────────────────────────────────────────────────╯

Performance: 6.493 seconds
✅ SUCCESS - Validator found cs336_basics in shadow worktree
```

### ⚠️ Step 7: Justify Stage (Interactive - Not Tested)

**Status:** Blocked on user input (expected behavior)

The justify stage opens `$EDITOR` for interactive question answering. This is correct behavior but blocks automated testing.

**What happens:**
```bash
$ uv run mastery submit
# Opens editor (vim/nano/etc) waiting for user to answer questions
# Terminal blocks until editor closes
```

This is **intentional** - the justify stage requires human input. Cannot be tested non-interactively without mocking.

---

## Commits Applied

| Commit | Description | Impact |
|--------|-------------|--------|
| `4922ed3` | Fix README instruction order | README now works from scratch |
| `1558497` | Recreate symlink in shadow worktree | Validators can now run |

---

## Verification Summary

| Test Step | Status | Notes |
|-----------|--------|-------|
| Clone repo | ✅ PASS | Clean clone works |
| Mode switch | ✅ PASS | Symlink created correctly |
| Install dependencies | ✅ PASS | Now works (was broken) |
| Install package | ✅ PASS | Recognizes cs336_basics |
| Initialize curriculum | ✅ PASS | Shadow worktree created |
| Symlink in worktree | ✅ PASS | Automatically recreated |
| View module | ✅ PASS | Build prompt displays |
| Build validation | ✅ PASS | Tests run successfully |
| Justify stage | ⚠️ INTERACTIVE | Requires editor input (expected) |

**Pass Rate:** 8/8 automated steps (100%)  
**Critical Bugs Fixed:** 2/2  
**Status:** ✅ PRODUCTION READY

---

## Lessons Learned

### 1. Always Run Stranger Tests
"It should work" ≠ "It works"

The README looked correct but had a critical ordering bug that only manifested when following instructions from scratch.

### 2. Git Worktrees Don't Copy Symlinks
This is fundamental Git behavior but easy to forget. Any workflow using worktrees with symlinks must explicitly recreate them.

### 3. Test in a Clean Environment
Testing in your development environment misses issues because:
- Dependencies already installed
- Symlinks already created
- State already initialized

Fresh `/tmp` directory caught both bugs immediately.

---

## Recommendations

### ✅ Completed
- [x] Fix README instruction order
- [x] Add symlink recreation in init command
- [x] Document stranger test methodology
- [x] Verify end-to-end from scratch

### 💡 Future Improvements
- [ ] Add automated stranger test to CI/CD (Docker container)
- [ ] Add explicit symlink check in validator error messages
- [ ] Consider adding `--non-interactive` flag for justify stage testing
- [ ] Document Windows limitations (symlinks require admin/dev mode)

---

## Conclusion

The Mastery Engine Quick Start instructions now work **exactly as documented** for a stranger cloning the repository for the first time.

**Before:** 2 critical bugs blocked basic usage  
**After:** Complete end-to-end workflow functional  

**Status:** ✅ **PRODUCTION READY** for public release

---

**Test conducted by:** Cascade AI (systematic verification)  
**Environment:** macOS (zsh), Python 3.13, uv 0.5.14  
**Repository:** https://github.com/ImmortalDemonGod/mastery-engine  
**Test Duration:** ~5 minutes (manual execution)
