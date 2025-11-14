# Next Bottleneck Diagnosis - Manual Analysis Results

## Regression Check: ✅ PASS
- Current: 3/4 (75%) - attention, rmsnorm, adamw
- Baseline: 3/4 (75%) - attention, rmsnorm, adamw
- **No regression detected**

## Manual Analysis Reveals Critical Issues

### Issue #1: SILU - Missing Variable Names
**Problem:** All 3 attempts generate `BinOp(Mult)` pattern WITHOUT specific variable
**Impact:** Pattern matches too broadly or doesn't match at all
**Evidence:**
```
Pass 1: BinOp(Mult) → replace_with
Issues: ⚠️ NO SPECIFIC VAR
```

**Root Cause:** For silu, the bug is in a RETURN statement, not an Assign
- Pattern: `return in_features * torch.sigmoid(in_features)`
- LLM generates: `BinOp(Mult)` (expression level)
- Should generate: `Return` with value containing `BinOp` (statement level)

**Golden Example Shows:** Pattern should target Return statement, not bare BinOp

### Issue #2: ATTENTION - Over-Specification on Attempt 1
**Problem:** Attempt 1 fails with over-specified pattern, but attempts 2-3 succeed
**Evidence:**
```
Attempt 1: pattern_match (over-specified)
Attempt 2: unknown (100% accurate but fails)
Attempt 3: SUCCESS (100% accurate)
```

**Root Cause:** LLM learns from feedback but not on first try
- Attempt 1: Over-specifies and fails
- Attempt 2: Fixes over-specification but still fails (unknown reason)
- Attempt 3: Works

**This violates user's 100% first-try constraint**

### Issue #3: ADAMW - Wrong Node Type on Attempt 1
**Problem:** Attempt 1 uses wrong node type for `denom` variable
**Evidence:**
```
Attempt 1: 75% accurate
  ❌ denom: expected Call/None, got BinOp/Div
Attempt 2: 100% accurate → SUCCESS
```

**Root Cause:** LLM analyzes wrong code on first attempt
- denom in BEFORE: `denom = exp_avg_sq.sqrt().add_(eps)` (Call)
- LLM thinks: `BinOp(Div)` (wrong!)
- After feedback: Corrects to `Call`

**This is the BEFORE/AFTER confusion we tried to fix!**

### Issue #4: RMSNORM - Success but No Variable Name
**Problem:** Succeeds despite having no specific variable name
**Evidence:**
```
Pass 1: Call .mean() → remove_keyword_arg
Issues: ⚠️ NO SPECIFIC VAR
```

**Why it works:** `remove_keyword_arg` only needs to find `.mean()` call, doesn't need variable name
**Not a bug, just interesting**

## Statistics vs Reality Gap

**Stats say:**
- Node type accuracy: 95%
- Similarity to golden: 0.49

**Manual analysis reveals:**
- silu: Wrong structural level (BinOp vs Return)
- attention: Over-specified on first try (learns after feedback)
- adamw: Wrong node type on first try (Call vs BinOp/Div)

**Stats MISSED the most important issues!**

## Bottleneck Analysis

### Current First-Try Success: 1/4 (25%)
Only rmsnorm succeeds on first attempt.

### Why First-Try Fails:

1. **silu (all 3 attempts):** Wrong structural level
   - Needs: Return statement pattern
   - Gets: BinOp expression pattern

2. **attention (attempt 1):** Over-specification
   - Generated pattern too complex
   - Feedback helps but violates first-try constraint

3. **adamw (attempt 1):** Wrong node type
   - Still confusing BEFORE/AFTER code
   - Thinks denom is BinOp/Div instead of Call

## Next Actions (Prioritized)

### P0: Fix silu (structural mismatch)
**Problem:** LLM doesn't understand Return vs Assign statements
**Solution:** Enhance prompt with Return statement examples
**Impact:** Could fix 1/4 → 2/4 first-try

### P1: Prevent over-specification on first try
**Problem:** attention over-specifies on attempt 1
**Solution:** Stronger simplicity guidance + more examples
**Impact:** Could fix attention first-try

### P2: Fix BEFORE/AFTER confusion completely
**Problem:** adamw still gets wrong node types on first try
**Solution:** Even more explicit BEFORE code emphasis
**Impact:** Could fix adamw first-try

### Success Scenario:
If all 3 fixes work: 1/4 → 4/4 first-try (100%) ✅

## Key Insight

**Manual analysis is ESSENTIAL:**
- Stats said "95% node type accuracy" 
- Reality: Wrong structural level, wrong types, over-specification
- Without manual analysis, we'd never diagnose these issues

**This validates user's requirement to always manually analyze LLM output!**
