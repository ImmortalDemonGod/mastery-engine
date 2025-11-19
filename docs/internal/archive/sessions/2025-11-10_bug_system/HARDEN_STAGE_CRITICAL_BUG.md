# HARDEN Stage - Critical Architectural Flaw

**Date**: November 13, 2025, 12:54 PM CST  
**Severity**: 🔴 **CRITICAL** - System-breaking  
**Status**: 🔄 **FIXING**

---

## The Fatal Flaw

### Current Broken Design

```python
# engine/stages/harden.py line 99-108
# 1. Copy STUDENT's implementation to harden workspace
shutil.copy2(source_file_path, harden_file)

# 2. Try to apply patch (created from DEVELOPER code) to STUDENT code
self.workspace_mgr.apply_patch(harden_file, bug_patch)
```

**Problem**: Patch requires **byte-for-byte match** of developer implementation.

Student's code has:
- Different variable names (`input_tensor` vs `x`, `maximum` vs `max_vals`)
- Different comments
- Different spacing
- Different code structure

**Result**: Patch fails 100% of the time.

---

## Proof

### Developer Implementation (patch created from this)
```python
def softmax(in_features, dim):
    x = in_features
    orig_dtype = x.dtype
    x32 = x.float()
    max_vals = x32.max(dim=dim, keepdim=True).values
    shifted = x32 - max_vals
    exps = torch.exp(shifted)
    ...
```

### Student Implementation (perfectly correct, but different)
```python
def softmax(in_features, dim):
    # My implementation with different style
    input_tensor = in_features  # Different name!
    original_dtype = input_tensor.dtype
    input_f32 = input_tensor.float()
    maximum = input_f32.max(dim=dim, keepdim=True).values  # Different name!
    normalized = input_f32 - maximum  # Different name!
    exponentials = torch.exp(normalized)  # Different name!
    ...
```

### Patch Application
```bash
$ patch student_code.py bug.patch
patching file student_code.py
1 out of 1 hunks failed--saving rejects to student_code.py.rej
```

**Success Rate**: 0%

---

## Better Design

### Correct Architecture

```python
# DON'T patch student code (impossible)
# Instead:

# 1. Copy DEVELOPER's correct implementation to harden workspace
shutil.copy2(developer_file_path, harden_file)

# 2. Apply patch to developer code (guaranteed to work)
self.workspace_mgr.apply_patch(harden_file, bug_patch)

# 3. Student debugs the already-buggy code
# They never see their own implementation in harden
```

**Why This Works**:
- Patch created from developer code
- Applied to developer code  
- Guaranteed byte-for-byte match
- Student gets consistent buggy code regardless of their implementation style

**Why This is Better Pedagogically**:
- Student sees the "reference" buggy implementation
- Focuses on debugging, not implementation style
- Consistent experience for all students
- Matches what symptom description says ("Your softmax implementation...")

---

## Implementation Plan

1. ✅ **Identify source of developer implementation**
   - Already tracked in curriculum: `source_file_path`
   - But this points to STUDENT code (via symlink)
   - Need to resolve to DEVELOPER code

2. ✅ **Update harden.py**
   - Get developer implementation path
   - Copy developer code (not student code)
   - Apply patch (will work)

3. ✅ **Test with real student code**
   - Verify patch applies successfully
   - Verify student can debug and fix
   - Verify validation works

4. ✅ **Update documentation**
   - Clarify that harden shows "reference implementation with bug"
   - Student's own code is preserved in main workspace

---

## Status

- [x] Bug discovered
- [x] Root cause identified
- [x] Better design proposed
- [ ] Implementation in progress
- [ ] Testing
- [ ] Documentation update
