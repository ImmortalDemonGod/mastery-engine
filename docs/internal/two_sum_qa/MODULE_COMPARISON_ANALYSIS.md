# Module Completeness Analysis: Sorting vs Two Sum

**Date:** November 18, 2025  
**Purpose:** Systematic comparison to ensure all Build/Justify/Harden stages are properly implemented

## Required Files Per Stage (from MASTERY_ENGINE.md)

According to the architecture documentation, each module requires:

### Stage 1: BUILD
- **`build_prompt.txt`** - The challenge description
- **`test_cases.json`** - Test cases for validation
- **`validator.sh`** - Automated validation script

### Stage 2: JUSTIFY  
- **`justify_questions.json`** - Conceptual questions with model answers and failure modes

### Stage 3: HARDEN
- **`bugs/`** directory containing:
  - **`<bug_name>.json`** - Bug metadata and injection instructions
  - **`<bug_name>.patch`** - Code patch to inject the bug (optional)
  - **`<bug_name>_symptom.txt`** - Description of the bug's observable symptom

## Module Comparison

### Sorting Module (Reference - Manually Created)

```
sorting/
├── build_prompt.txt          ✅ 2,231 bytes
├── test_cases.json           ✅ 479 bytes
├── validator.sh              ✅ 1,283 bytes
├── solution.py               ✅ 1,308 bytes (reference)
├── justify_questions.json    ✅ 5,427 bytes
└── bugs/                     ✅ Directory exists
    ├── incomplete_merge.json         ✅ 1,155 bytes
    ├── incomplete_merge.patch        ✅ 1,719 bytes
    ├── incomplete_merge_symptom.txt  ✅ 444 bytes
    ├── missing_base_case.json        🟡 0 bytes (placeholder)
    ├── missing_base_case.py          🟡 0 bytes (placeholder)
    ├── missing_base_case_symptom.txt 🟡 0 bytes (placeholder)
    └── off_by_one.py                 🟡 0 bytes (placeholder)
```

**Status:** ✅ Complete for all 3 stages (1 bug fully implemented, others are placeholders)

### Two Sum Module (Generated - Needs Completion)

```
two_sum/
├── build_prompt.txt          ✅ 2,960 bytes
├── test_cases.json           ✅ 924 bytes
├── validator.sh              ❌ MISSING
├── solution.py               🟡 70 bytes (stub)
├── justify_questions.json    ❌ MISSING
└── bugs/                     ❌ MISSING
```

**Status:** 🔴 INCOMPLETE - Only BUILD stage partially implemented

## Gap Analysis

### Critical Gaps (Blocking Module Use)

| File | Stage | Status | Impact |
|------|-------|--------|--------|
| `validator.sh` | BUILD | ❌ Missing | Cannot validate student solutions |
| `justify_questions.json` | JUSTIFY | ❌ Missing | Cannot progress past BUILD |
| `bugs/` directory | HARDEN | ❌ Missing | Cannot progress past JUSTIFY |

### Non-Critical Gaps

| File | Stage | Status | Impact |
|------|-------|--------|--------|
| `solution.py` | Reference | 🟡 Stub | CI validation would fail, but not needed for student workflow |

## Required Actions

### Priority 1: Enable BUILD Stage Completion

**Create `validator.sh`** with the following requirements:
- Execute test cases from `test_cases.json`
- Import and call the student's solution function
- Report pass/fail for each test
- Exit with code 0 if all pass, non-zero otherwise

**Template structure:**
```bash
#!/bin/bash
# Validator for Two Sum

set -e

# Run Python validation
python3 << 'EOF'
import json
import sys
from pathlib import Path

# Import student solution
sys.path.insert(0, str(Path(__file__).parent))
from solution import twoSum

# Load test cases
with open("test_cases.json") as f:
    data = json.load(f)

passed = 0
failed = 0

for test in data["tests"]:
    result = twoSum(**test["input"])
    expected = test["expected"]
    
    if sorted(result) == sorted(expected):  # Order doesn't matter
        print(f"✓ Test {test['id']}: PASS")
        passed += 1
    else:
        print(f"✗ Test {test['id']}: FAIL")
        failed += 1

print(f"\nResults: {passed}/{passed + failed} passed")
sys.exit(0 if failed == 0 else 1)
EOF
```

### Priority 2: Enable JUSTIFY Stage

**Create `justify_questions.json`** with:
- 3-5 conceptual questions about hash tables and two sum
- Model answers demonstrating deep understanding
- Failure modes (e.g., "Rote Memorizer", "Hand-Waver")

**Example questions:**
1. "Why is a hash table better than nested loops for this problem?"
2. "What is the time complexity and why?"
3. "What edge cases must be handled?"

### Priority 3: Enable HARDEN Stage

**Create `bugs/` directory** with at least 1 bug:

**Example bug: `missing_complement_check.json`**
```json
{
  "bug_id": "missing_complement_check",
  "title": "Missing Duplicate Index Check",
  "description": "The solution doesn't check if the complement is the same element",
  "difficulty": "easy",
  "symptom": "Returns [0, 0] for input nums=[2, 7, 11, 15], target=4",
  "injection_type": "ast_pattern",
  "pattern": {
    "find": "if complement in seen:",
    "replace": "if complement in seen and seen[complement] != i:"
  }
}
```

## Sorting Module Quality Assessment

### Strengths
- ✅ Complete implementation across all 3 stages
- ✅ Professional `build_prompt.txt` with clear requirements
- ✅ Comprehensive `justify_questions.json` with failure modes
- ✅ AST-based bug injection (sophisticated)

### Weaknesses
- 🟡 Only 1 fully implemented bug (incomplete_merge)
- 🟡 Placeholder files for additional bugs
- 🟡 No performance benchmarking in validator

### Recommendations for Improvement
1. Complete the 2 placeholder bugs
2. Add performance assertions to validator
3. Add edge case tests (very large arrays, negative numbers)

## Two Sum Module Required Improvements

### Immediate (Before Student Use)
1. **Create `validator.sh`** - Required to complete BUILD stage
2. **Create `justify_questions.json`** - Required to progress to JUSTIFY
3. **Create at least 1 bug** - Required to progress to HARDEN

### Medium-term (For Production Quality)
1. Create 2-3 additional bugs for variety
2. Enhance test cases with edge cases
3. Create reference solution (for CI validation)

### Long-term (For Excellence)
1. Add performance benchmarking
2. Add multiple solution strategies (brute force vs optimal)
3. Create advanced harden challenges

## Generation Script Gaps

The `generate_module.py` script currently generates:
- ✅ `build_prompt.txt` (from template)
- ✅ `test_cases.json` (from examples)
- ❌ `validator.sh` (template only, not functional)
- ❌ `justify_questions.json` (template only, generic)
- ❌ `bugs/` (not implemented)

### Recommended Script Enhancements

**Phase 3.4 (Next):** Enhance generation script to create:
1. **Functional `validator.sh`** based on test cases
2. **Scaffolded `justify_questions.json`** with problem-specific placeholders
3. **`bugs/` directory** with at least 1 template bug

**Implementation approach:**
```python
def generate_validator(self, problem_data: Dict) -> str:
    """Generate functional validator.sh from test cases."""
    # Extract function signature from examples
    # Generate validation logic
    # Return executable script

def scaffold_justify_questions(self, problem_data: Dict) -> List[Dict]:
    """Generate problem-specific justify questions."""
    # Use LLM to generate questions based on:
    # - Problem description
    # - Topics (e.g., "Hash Table", "Two Pointers")
    # - Difficulty level
    # Return structured questions

def generate_bug_templates(self, problem_data: Dict) -> List[Dict]:
    """Generate bug templates for common mistakes."""
    # Based on problem type and topics
    # Return at least 1 bug per problem
```

## Success Criteria for Module Completeness

A module is considered **complete** when:

### BUILD Stage
- [x] `build_prompt.txt` exists and is >1000 chars
- [x] `test_cases.json` exists with ≥2 test cases
- [ ] `validator.sh` exists and is executable
- [ ] `validator.sh` correctly validates solutions

### JUSTIFY Stage
- [ ] `justify_questions.json` exists
- [ ] Contains ≥3 questions
- [ ] Each question has a model answer
- [ ] Each question has ≥1 failure mode

### HARDEN Stage
- [ ] `bugs/` directory exists
- [ ] Contains ≥1 fully-specified bug
- [ ] Each bug has `.json` metadata
- [ ] Each bug has `_symptom.txt` description
- [ ] Bug can be successfully injected

## Current Module Status Summary

| Module | BUILD | JUSTIFY | HARDEN | Overall |
|--------|-------|---------|--------|---------|
| **sorting** | ✅ Complete | ✅ Complete | ✅ Complete (1 bug) | ✅ READY |
| **two_sum** | 🟡 Partial | ❌ Missing | ❌ Missing | 🔴 INCOMPLETE |

## Action Plan

### Immediate (Next 30 minutes)
1. Create `two_sum/validator.sh`
2. Create `two_sum/justify_questions.json`
3. Create `two_sum/bugs/` directory
4. Create at least 1 bug for two_sum

### Short-term (Next session)
1. Update `generate_module.py` to auto-generate validators
2. Add justify question generation (LLM-based)
3. Add bug template generation

### Medium-term (Phase 3.5)
1. Complete all placeholder bugs in sorting module
2. Generate modules for 4-6 more diverse problems
3. Validate completeness across all generated modules

## Conclusion

The sorting module serves as a **gold standard** for module completeness, demonstrating all three stages properly implemented. The two_sum module, while having excellent BUILD stage content, requires completion of JUSTIFY and HARDEN stages to be functionally equivalent.

**Critical Next Step:** Create the missing files for two_sum module to enable full BUILD→JUSTIFY→HARDEN workflow.
