# Next Bottleneck Identified via Manual Analysis

## What Statistics Showed
- Success rate: 0%
- Failure mode: "unknown" (not pattern_match)
- Node type accuracy: 95.8%
- Pattern matcher: Working (manual test successful)

## What Manual Analysis Revealed

### The Real Problem: Wrong Replacement Strategy

**Diagnosis via systematic manual analysis:**
1. Ran actual test to get diagnostic message
2. Examined patch to see what SHOULD happen
3. Compared LLM-generated replacements vs expected

**Finding:**
The LLM doesn't understand when to DELETE statements vs REPLACE values!

**Example (adamw bug):**

Patch shows:
```diff
- bias_correction1 = 1 - beta1 ** state['step']  # DELETE THIS LINE
- bias_correction2 = 1 - beta2 ** state['step']  # DELETE THIS LINE  
- step_size = lr / bias_correction1              # DELETE THIS LINE
+ step_size = lr                                  # ADD THIS (simplified)
- denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)  # DELETE
+ denom = exp_avg_sq.sqrt().add_(eps)            # ADD THIS (simplified)
```

LLM generates:
```json
Pass 1: replace_value_with "1 - beta1 ** state['step']"     ❌ KEEPS THE LINE!
Pass 2: replace_value_with "1 - beta2 ** state['step']"     ❌ KEEPS THE LINE!
Pass 3: replace_value_with "lr / bias_correction1"          ❌ KEEPS THE LINE!
Pass 4: replace_value_with "exp_avg_sq.sqrt()..."           ❌ WRONG EXPRESSION!
```

Should generate:
```json
Pass 1: delete_statement (bias_correction1)                 ✅ DELETES
Pass 2: delete_statement (bias_correction2)                 ✅ DELETES
Pass 3: replace_value_with "lr"                             ✅ SIMPLIFIES
Pass 4: replace_value_with "exp_avg_sq.sqrt().add_(eps)"    ✅ SIMPLIFIES
```

## Root Cause

The LLM doesn't understand the difference between:
1. **DELETE**: Remove entire statement (use `delete_statement`)
2. **REPLACE**: Keep statement but change its value (use `replace_value_with`)

## Impact

- Injection succeeds (returns True)
- But output has EXTRA lines (bias_correction1, bias_correction2, etc.)
- Diagnostic: "didn't remove ENOUGH (more lines than expected)"
- Evaluation fails because output doesn't match expected

## Next Steps

### Option 1: Improve Prompt (Teach DELETE vs REPLACE)
Add explicit guidance:
```
When the patch shows a line with ONLY '-' (removed):
  → Use delete_statement

When the patch shows '-' then '+' (changed):
  → Use replace_value_with with the NEW value
```

### Option 2: Add Examples
Include golden example showing delete_statement usage

### Option 3: Update Evaluation  
Add detection of "wrong replacement strategy" pattern

## Quantified

**Current state:**
- 95.8% node type accuracy (LLM understanding AST)  
- 100% patterns match successfully (engine working)
- 0% correct transformations (wrong delete/replace strategy)

**Bottleneck:** Prompt doesn't explain when to delete vs replace

