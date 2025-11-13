# Einops Violations Audit
**Date**: 2025-11-12  
**Auditor**: Systematic Code Review  
**Scope**: Reference implementations in `modes/student/cs336_basics/`

## Executive Summary

**Critical Finding**: Reference implementation violates PDF §3.3 einops requirement.

**Status**: 
- ❌ **5 violations found** in `layers.py::multihead_self_attention()`
- ✅ **0 einops imports** in student reference code
- 📋 **Impact**: Students see manual tensor reshaping instead of best-practice einops

**PDF Requirement (§3.3)**:
> "Use einops for tensor operations. It makes code more readable and self-documenting."
> Example: `x = rearrange(x, 'b s (h d) -> b h s d', h=num_heads)`

---

## Detailed Violations

### File: `modes/student/cs336_basics/layers.py`

#### Function: `multihead_self_attention` (Lines 187-241)

This is a **reference implementation** provided to students as working code. It contains 5 einops violations:

#### Violation 1: Line 214 - Batch Flattening
```python
# CURRENT (VIOLATION)
x = in_features.reshape(-1, seq_len, d_model)

# SHOULD BE (einops)
x = rearrange(in_features, '... s d -> (...) s d')
```
**Issue**: Manual `reshape` obscures intent. Reader must infer that leading dims are flattened.

---

#### Violation 2: Line 225 - Split to Heads
```python
# CURRENT (VIOLATION)
def to_heads(t: Tensor) -> Tensor:
    return t.view(t.shape[0], seq_len, num_heads, head_dim).transpose(1, 2).contiguous()

# SHOULD BE (einops)
def to_heads(t: Tensor) -> Tensor:
    return rearrange(t, 'b s (h d) -> b h s d', h=num_heads)
```
**Issue**: 
- Chained `.view().transpose().contiguous()` is hard to read
- Requires explicit shape extraction `t.shape[0]`
- Doesn't document the split operation `(h d)`
- **THIS IS THE EXACT EXAMPLE FROM PDF §3.3**

---

#### Violation 3: Line 232 - Causal Mask Broadcasting
```python
# CURRENT (VIOLATION)
causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device)).view(1, 1, seq_len, seq_len)

# SHOULD BE (einops)
causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device))
causal = rearrange(causal, 's1 s2 -> 1 1 s1 s2')
```
**Issue**: Manual `.view(1, 1, ...)` for broadcasting doesn't document intent clearly.

---

#### Violation 4: Line 236 - Combine Heads
```python
# CURRENT (VIOLATION)
context = context.transpose(1, 2).contiguous().view(context.shape[0], seq_len, d_model)

# SHOULD BE (einops)
context = rearrange(context, 'b h s d -> b s (h d)')
```
**Issue**: 
- Chained `.transpose().contiguous().view()` is verbose
- Requires shape extraction `context.shape[0]`
- Doesn't document the concatenation `(h d)`

---

#### Violation 5: Line 241 - Restore Leading Dimensions
```python
# CURRENT (VIOLATION)
return out.reshape(*orig_leading, seq_len, d_model)

# SHOULD BE (einops)
# Note: This is challenging with einops as it requires dynamic shape unpacking
# Acceptable to keep as reshape OR redesign function signature
```
**Issue**: Minor - this is a legitimate edge case where einops is less clean.

---

## Impact Assessment

### On Students
**Severity**: HIGH

1. **Contradicts official guidance**: PDF explicitly recommends einops
2. **Bad example**: Reference code models poor practices
3. **Pedagogical inconsistency**: "Do as I say, not as I do"
4. **Missed learning**: Students don't see einops benefits:
   - Self-documenting code
   - Compile-time shape checking
   - Reduced cognitive load

### On Curriculum Quality
**Severity**: HIGH

1. **Alignment failure**: Implementation doesn't match stated best practices
2. **Trust erosion**: If reference code ignores guidance, why should students follow it?
3. **Maintainability**: Manual reshapes harder to understand and debug

---

## Root Cause Analysis

### Why Violations Exist

1. **Possible legacy code**: May predate einops adoption
2. **Performance myth**: Belief that manual ops are faster (typically negligible difference)
3. **Unfamiliarity**: Developer may not have known einops best practices
4. **No automated checks**: No linting/CI to enforce einops usage

### Why It Matters

The PDF §3.3 explicitly calls out this exact pattern:
```python
# PDF Example (§3.3):
x = rearrange(x, 'b s (h d) -> b h s d', h=num_heads)
```

The reference code does:
```python
# Current code (Line 225):
t.view(t.shape[0], seq_len, num_heads, head_dim).transpose(1, 2).contiguous()
```

**This is a direct contradiction of the curriculum's own example.**

---

## Verification Scope

### Files Audited
- ✅ `layers.py` - **5 violations found**
- ✅ `utils.py` - No violations (all TODOs)
- ✅ `optimizer.py` - No violations (all TODOs)
- ✅ `generation.py` - No violations (all TODOs)
- ✅ `bpe.py` - Not applicable (no tensor ops)
- ✅ `tokenizer.py` - Not applicable (no tensor ops)

### Import Check
```bash
$ grep -r "import einops" modes/student/cs336_basics/
# NO RESULTS
```

**Finding**: Zero einops usage in student reference code.

---

## Recommendations

### Priority 1.5: Refactor to Einops (IMMEDIATE)

**Affected Function**: `multihead_self_attention` in `layers.py`

**Changes Required**:
1. Add import: `from einops import rearrange`
2. Refactor 4 violations (Lines 214, 225, 232, 236)
3. Document einops usage in function docstring
4. Add inline comments explaining rearrange patterns
5. Verify tests still pass after refactoring

**Example Refactor** (Line 225):
```python
# Before
def to_heads(t: Tensor) -> Tensor:
    return t.view(t.shape[0], seq_len, num_heads, head_dim).transpose(1, 2).contiguous()

# After
def to_heads(t: Tensor) -> Tensor:
    """Split concatenated heads: (b, s, h*d) -> (b, h, s, d)"""
    return rearrange(t, 'b s (h d) -> b h s d', h=num_heads)
```

**Benefits**:
- ✅ Aligns with PDF §3.3
- ✅ Self-documenting (no shape extraction needed)
- ✅ More readable (single operation vs chained calls)
- ✅ Better error messages (einops validates shapes)

---

### Follow-Up Actions

1. **Update build_prompts**: Add einops examples where tensor ops are taught
2. **Add linting**: Create ruff/flake8 rule to flag manual reshape/transpose
3. **Student guidance**: Document when einops is preferred vs acceptable manual ops
4. **Performance note**: Add comment that einops compiles to equivalent native ops

---

## Success Criteria

- [ ] All violations in `layers.py` refactored to einops
- [ ] `from einops import rearrange` added to imports
- [ ] All tests pass after refactoring
- [ ] Function docstring updated to mention einops
- [ ] PDF §3.3 guidance now matches implementation
- [ ] No new violations introduced

---

## Appendix: Einops Quick Reference

### Common Patterns

```python
# Split concatenated dimension
x = rearrange(x, 'b s (h d) -> b h s d', h=num_heads)

# Combine/concatenate dimension
x = rearrange(x, 'b h s d -> b s (h d)')

# Transpose (swap dimensions)
x = rearrange(x, 'b h s d -> b s h d')

# Flatten batch dimensions
x = rearrange(x, '... s d -> (...) s d')

# Add broadcast dimensions
x = rearrange(x, 's1 s2 -> 1 1 s1 s2')
```

### Why Einops > Manual Ops

| Manual | Einops | Winner |
|--------|--------|--------|
| `x.view(B, S, H, D).transpose(1, 2)` | `rearrange(x, 'b s h d -> b h s d')` | **Einops** (clearer intent) |
| Shape errors at runtime | Shape errors with helpful messages | **Einops** (better debugging) |
| Requires shape arithmetic | Declarative pattern matching | **Einops** (less mental load) |
| Hard to verify correctness | Self-documenting | **Einops** (maintainability) |

---

**Next Step**: Proceed to Priority 1.5 - Refactor violations to use einops.
