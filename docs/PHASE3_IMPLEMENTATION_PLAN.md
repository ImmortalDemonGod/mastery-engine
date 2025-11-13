# Phase 3: Generalization Implementation Plan

**Date**: November 13, 2025  
**Status**: 🚀 READY TO BEGIN  
**Approach**: Incremental, test-driven, starting with softmax refactor

---

## Objective

Transform the hardcoded `SoftmaxBugInjector` into a generic, data-driven engine that interprets declarative JSON bug definitions without requiring custom Python code per bug.

---

## Phase 3 Steps

### Step 1: Design Generic JSON Schema ✅ PLANNED
Define the declarative format for multi-pass bug logic.

**Deliverable**: Enhanced `softmax-no-subtract-max.json` with full logic

### Step 2: Build Generic Pattern Matcher 🔄 NEXT
Create a generic AST pattern matching system that works from JSON definitions.

**Deliverable**: `GenericPatternMatcher` class

### Step 3: Build Generic Transformer
Create a generic AST transformation system driven by JSON replacement rules.

**Deliverable**: `GenericASTTransformer` class

### Step 4: Build Generic Bug Injector
Orchestrate the pattern matcher and transformer with multi-pass logic.

**Deliverable**: `GenericBugInjector` class

### Step 5: Update HardenRunner
Remove hardcoded `SoftmaxBugInjector` import, use generic injector.

**Deliverable**: Modified `harden.py`

### Step 6: Test & Validate
Run end-to-end test with generic system on softmax.

**Deliverable**: Validation that generic system works

### Step 7: Create Second Bug Type
Test generalization with a different bug pattern.

**Deliverable**: Second module converted to AST

### Step 8: LLM Authoring Tool (Optional)
Build tool to generate JSON from natural language descriptions.

**Deliverable**: `generate_bug_definition.py`

---

## Step 1 Details: Generic JSON Schema

### Enhanced Bug Definition Format

```json
{
  "id": "softmax-no-subtract-max",
  "description": "Removes the subtract-max trick, causing numerical overflow.",
  "injection_type": "ast",
  "engine_version": "2.1",
  "target_function": "softmax",
  "logic": [
    {
      "pass": 1,
      "type": "find_and_track",
      "description": "Find the variable assigned the result of .max() call",
      "pattern": {
        "node_type": "Assign",
        "value": {
          "node_type": "Call",
          "func": {
            "node_type": "Attribute",
            "attr": "max"
          }
        }
      },
      "conditions": [
        {
          "check": "targets_length_equals",
          "value": 1
        },
        {
          "check": "target_is_name",
          "index": 0
        }
      ],
      "track_as": {
        "max_var_name": "node.targets[0].id",
        "tensor_var_name": "node.value.func.value.id"
      }
    },
    {
      "pass": 2,
      "type": "find_and_replace",
      "description": "Find where tensor is subtracted by max_var and remove subtraction",
      "pattern": {
        "node_type": "Assign",
        "value": {
          "node_type": "BinOp",
          "op": "Sub",
          "left": {
            "node_type": "Name",
            "id": {
              "from_context": "tensor_var_name"
            }
          },
          "right": {
            "node_type": "Name",
            "id": {
              "from_context": "max_var_name"
            }
          }
        }
      },
      "replacement": {
        "type": "replace_value_with",
        "source": "node.value.left"
      }
    }
  ]
}
```

### Key Schema Elements

1. **`target_function`**: Which function to search (limits scope)
2. **`logic` array**: Ordered passes (reconnaissance then strike)
3. **`pattern`**: Declarative AST pattern matching
4. **`conditions`**: Additional validation checks
5. **`track_as`**: Context variables to extract
6. **`from_context`**: Reference tracked variables
7. **`replacement`**: How to construct new node

---

## Implementation Strategy

### Principle: Incremental Validation
- Implement one component at a time
- Test after each component
- Keep previous version working during development

### Risk Mitigation
- Start with softmax (known working case)
- Build generic system alongside hardcoded version
- Switch only after validation
- Keep backward compatibility

### Testing Approach
1. Unit test: Generic pattern matcher in isolation
2. Unit test: Generic transformer in isolation
3. Integration test: Full generic injector
4. E2E test: Complete workflow with softmax
5. Second bug test: Validate generalization

---

## Success Criteria

### Must Have
- [ ] Generic injector works for softmax bug
- [ ] No hardcoded bug-specific logic in engine
- [ ] Backward compatible with .patch files
- [ ] End-to-end test passes

### Should Have
- [ ] At least 2 bug types working on generic system
- [ ] Clear error messages when patterns don't match
- [ ] Documentation updated

### Nice to Have
- [ ] LLM authoring tool for generating JSON
- [ ] Pattern validation at load time
- [ ] Visual debugging for pattern matching

---

## Timeline Estimate

| Step | Estimated Time | Complexity |
|------|---------------|------------|
| 1. JSON Schema | 30 min | Low (design only) |
| 2. Pattern Matcher | 2-3 hours | High (core logic) |
| 3. Transformer | 1-2 hours | Medium |
| 4. Bug Injector | 1 hour | Low (orchestration) |
| 5. HardenRunner | 30 min | Low (remove import) |
| 6. Testing | 1 hour | Medium |
| 7. Second Bug | 1-2 hours | Medium |
| 8. LLM Tool | 2-3 hours | Medium (optional) |
| **Total** | **9-13 hours** | **High** |

---

## Next Action

Begin Step 2: Build Generic Pattern Matcher

**File to create**: `engine/ast_harden/pattern_matcher.py`

**Functionality**:
- Parse JSON pattern definitions
- Match against canonical AST nodes
- Extract context variables
- Handle `from_context` references

---

**Status**: Plan complete, ready to begin implementation.
