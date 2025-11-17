# Phase 3 Completion Report: Generic AST Bug Injection Engine

**Version:** 2.1  
**Status:** ✅ COMPLETE - PRODUCTION READY  
**Date:** November 13, 2025

---

## Executive Summary

Phase 3 has successfully transformed the hardcoded AST bug injection system into a fully generic, data-driven engine that interprets declarative JSON bug definitions. The engine has been validated with three diverse bug types and successfully tested end-to-end in the production workflow.

**Key Achievement:** Any developer can now create new bugs by writing JSON files—no Python code required.

---

## Implementation Summary

### Components Delivered

1. **`engine/ast_harden/pattern_matcher.py`** (297 lines)
   - `PatternMatcher`: Recursive AST pattern matching with context substitution
   - `FindAndTrackVisitor`: Reconnaissance pass to extract variable names
   - `FindAndReplaceTransformer`: Transformation pass to inject bugs
   
2. **`engine/ast_harden/generic_injector.py`** (118 lines)
   - `GenericBugInjector`: Orchestrates multi-pass injection from JSON

3. **`engine/stages/harden.py`** (Modified)
   - Conditional dispatch: `.json` → GenericBugInjector, `.patch` → legacy

4. **Three Golden Dataset Files:**
   - `curricula/cs336_a1/modules/softmax/bugs/no_subtract_max.json`
   - `curricula/cs336_a1/modules/silu/bugs/missing_multiply.json`
   - `curricula/cs336_a1/modules/rmsnorm/bugs/missing_keepdim.json`

---

## JSON Schema v2.1 Specification

### Core Structure

```json
{
  "id": "bug-identifier",
  "description": "Human-readable description",
  "injection_type": "ast",
  "engine_version": "2.1",
  "target_function": "function_name",
  "logic": [
    {
      "pass": 1,
      "type": "find_and_track | find_and_replace",
      "description": "Pass description",
      "pattern": { /* AST pattern */ },
      "conditions": [ /* Optional conditions */ ],
      "track_as": { /* Variables to track */ },
      "replacement": { /* Transformation rules */ }
    }
  ],
  "metadata": { /* Optional metadata */ }
}
```

### Supported Pattern Features

#### Node Matching
```json
{
  "node_type": "Assign | BinOp | Call | Attribute | Name",
  "op": "Sub | Mult | Add | ...",
  "attr": "attribute_name",
  "id": "variable_name | {\"from_context\": \"var_key\"}"
}
```

#### Conditions
```json
{
  "check": "targets_length_equals | target_is_name | has_keyword_arg",
  "value": 1,
  "index": 0,
  "name": "arg_name"
}
```

#### Replacement Strategies
```json
{
  "type": "replace_value_with | replace_with | remove_keyword_arg",
  "source": "node.value.left",
  "name": "keepdim"
}
```

---

## Validation Results

### Test 1: Softmax (Multi-Pass, Context Tracking)

**Bug Type:** Remove subtract-max trick  
**Complexity:** Multi-pass with context variables  
**Pattern:** Find `.max().values`, track variables, find subtract, remove it

**Result:** ✅ PASS
- Pattern matched correctly
- Variable names preserved: `tensor_data`, `maximum`, `adjusted`
- Subtract operation removed: `adjusted = tensor_data`

### Test 2: SiLU (Single-Pass, Node Replacement)

**Bug Type:** Remove multiplication by input  
**Complexity:** Simple single-pass replacement  
**Pattern:** Find `x * sigmoid(x)`, replace with `sigmoid(x)`

**Result:** ✅ PASS
- Single-pass logic validated
- `return torch.sigmoid(in_features)` (multiplication removed)

### Test 3: RMSNorm (Keyword Argument Manipulation)

**Bug Type:** Remove `keepdim=True`  
**Complexity:** Function argument manipulation  
**Pattern:** Find `.mean()` call with `keepdim`, remove argument

**Result:** ✅ PASS
- Keyword argument condition works
- `keepdim` argument removed correctly
- `.mean(dim=-1)` (no keepdim)

### Test 4: Full E2E Production Test

**Workflow:**
1. Fresh initialization
2. Student implements with unique names
3. Build stage passes
4. Advance to harden
5. Generic engine injects bug
6. Challenge prompt displayed

**Result:** ✅ PASS
- Log confirmed: `Using AST-based bug injection (generic)`
- All student variable names preserved
- Bug injected successfully
- System ready for student debugging

---

## Critical Architectural Decisions

### 1. Hybrid AST Strategy (Validated ✅)

**Decision:** Use canonical AST for pattern matching, original AST for transformation.

**Rationale:** 
- Canonical AST provides robust pattern matching (variable names normalized)
- Original AST preserves student's exact code style and variable names

**Implementation:** `FindAndTrackVisitor` matches in canonical AST but extracts from original AST via source location mapping.

### 2. Multi-Pass Logic (Validated ✅)

**Decision:** Support multiple sequential passes with shared context.

**Rationale:** 
- Some bugs require reconnaissance before transformation
- Context variables enable cross-pass references
- Flexible enough for complex multi-step bugs

**Implementation:** `GenericBugInjector` executes passes sequentially, maintaining shared context dictionary.

### 3. Declarative JSON Schema (Validated ✅)

**Decision:** Define bugs in JSON, not Python code.

**Rationale:**
- Lower barrier to entry for curriculum designers
- Version-controlled bug definitions
- Easier to review and validate
- Enables LLM-powered authoring tools

**Implementation:** JSON files in `bugs/` directory, loaded and interpreted by generic engine.

---

## Engine Capabilities (Validated)

✅ **Multi-pass transformation** - Sequential passes with context sharing  
✅ **Pattern matching** - Recursive AST matching with wildcards  
✅ **Context variables** - Track and reference across passes  
✅ **Operator matching** - Compare AST operator types (Sub, Mult, etc.)  
✅ **Keyword argument conditions** - Check for specific function arguments  
✅ **Keyword argument removal** - Remove arguments from Call nodes  
✅ **Node replacement** - Multiple replacement strategies  
✅ **Variable name preservation** - Extract from original AST  
✅ **Source location mapping** - Map canonical → original nodes  

---

## Performance & Quality Metrics

### Code Quality
- **Total lines added:** ~600 lines (pattern_matcher + generic_injector + JSON)
- **Complexity:** Moderate (recursive matching, multi-pass logic)
- **Test coverage:** Validated with 3 diverse bugs + E2E test
- **Breaking changes:** Zero (backward compatible with `.patch` files)

### Robustness
- ✅ Handles diverse bug types (proven with 3 examples)
- ✅ Preserves student variable names (critical requirement)
- ✅ Fails gracefully (returns success=False if pattern not found)
- ✅ Schema is stable and extensible

### Developer Experience
- ✅ Declarative JSON format (no Python required)
- ✅ Self-documenting patterns
- ✅ Clear error messages
- ✅ Backward compatible

---

## Strategic Value

### Before Phase 3 (Hardcoded)
- ❌ New bugs require Python code changes
- ❌ Hardcoded logic in `SoftmaxBugInjector`
- ❌ High barrier to entry for curriculum designers
- ❌ Difficult to maintain and extend

### After Phase 3 (Generic)
- ✅ New bugs = JSON file (5 minutes)
- ✅ Zero Python code required
- ✅ Low barrier to entry (declarative patterns)
- ✅ Easy to maintain and extend
- ✅ Ready for LLM-powered authoring

---

## Next Steps: Phase 4 - LLM Authoring Tool

With a stable engine and three golden dataset files, we can now build the LLM authoring tool to automate bug creation for the remaining 17 bugs.

### Phase 4 Plan

**Step 1:** Design LLM prompt with few-shot examples  
**Step 2:** Implement prompt engineering pipeline  
**Step 3:** Validate LLM-generated JSON against schema  
**Step 4:** Batch-convert remaining `.patch` files  
**Step 5:** Human review and refinement  

**Estimated Effort:** 4-6 hours  
**Expected Output:** 17 new `.json` bug files

---

## Lessons Learned

### What Worked Well

1. **Manual validation first:** Creating 3 bugs manually forced us to discover schema gaps
2. **Diverse test cases:** Each bug exposed different requirements (multi-pass, single-pass, arguments)
3. **Hybrid AST strategy:** Proved critical for variable name preservation
4. **Systematic approach:** Testing each bug in isolation before E2E prevented debugging chaos

### Challenges Overcome

1. **AST structure subtlety:** `.max().values` is `Attribute(value=Call(...))`, not `Call(...)`
2. **Operator matching:** AST operators are objects, not strings (required special handling)
3. **Variable name preservation:** Required mapping from canonical to original AST
4. **Keyword argument removal:** Needed new transformation type

### Design Evolution

| Issue | Initial Approach | Final Solution |
|-------|------------------|----------------|
| Variable names | Extract from canonical AST | Map to original AST |
| Operator matching | String comparison | Class name comparison |
| Multi-pass logic | Single pass only | Sequential passes with context |
| Keyword args | Not supported | Added condition + removal |

---

## Production Readiness Checklist

- ✅ Core engine implemented and tested
- ✅ JSON schema stable and validated
- ✅ Three diverse bugs successfully migrated
- ✅ Full E2E test passed in production workflow
- ✅ Variable name preservation verified
- ✅ Backward compatibility maintained
- ✅ Error handling implemented
- ✅ Logging and diagnostics added
- ✅ Documentation complete
- ✅ Ready for Phase 4 (LLM tool)

---

## Conclusion

Phase 3 has successfully delivered a production-ready generic AST bug injection engine. The system is robust, extensible, and validated with diverse test cases. The declarative JSON schema is stable and ready to support LLM-powered authoring tools.

**Status:** ✅ PRODUCTION READY  
**Next:** Phase 4 - LLM Authoring Tool

---

**Document Version:** 1.0  
**Author:** Cascade AI  
**Date:** November 13, 2025
