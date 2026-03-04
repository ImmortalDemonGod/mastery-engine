# Phase 4: LLM-Powered Bug Authoring Tool

**Version:** 1.0  
**Status:** 🚧 IN PROGRESS  
**Date:** November 13, 2025

---

## Overview

Phase 4 delivers an LLM-powered tool that automates the conversion of legacy `.patch` bug files to the new v2.1 JSON format. The tool uses few-shot learning with the golden dataset (3 validated examples) to generate, validate, and self-correct bug definitions.

---

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM Bug Authoring Tool                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │ Golden       │──────▶│   Prompt     │                    │
│  │ Dataset      │       │   Builder    │                    │
│  │ (3 examples) │       │              │                    │
│  └──────────────┘       └──────┬───────┘                    │
│                                │                             │
│                                ▼                             │
│                         ┌──────────────┐                     │
│  ┌──────────────┐       │    LLM       │                     │
│  │ Patch File   │──────▶│   Service    │                     │
│  │ Parser       │       │              │                     │
│  └──────────────┘       └──────┬───────┘                     │
│                                │                             │
│                                ▼                             │
│                         ┌──────────────┐                     │
│                         │   JSON       │                     │
│                         │   Parser     │                     │
│                         └──────┬───────┘                     │
│                                │                             │
│                                ▼                             │
│  ┌──────────────┐       ┌──────────────┐                     │
│  │ Generic      │◀──────│  Validation  │                     │
│  │ Injector     │       │    Loop      │                     │
│  │ (Test)       │       │              │                     │
│  └──────────────┘       └──────┬───────┘                     │
│                                │                             │
│                          ✅ Success                          │
│                          ❌ Retry (max 3)                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input**: `.patch` file + symptom description
2. **Prompt Construction**: System prompt (schema + golden examples) + user prompt (patch details)
3. **LLM Generation**: Generate JSON bug definition
4. **Validation**: Parse JSON → Test with GenericBugInjector → Compare output
5. **Self-Correction**: If validation fails, retry with error feedback (max 3 attempts)
6. **Output**: Valid `.json` bug definition file

---

## Components

### 1. BugAuthor Class (`engine/dev_tools/bug_author.py`)

Main orchestrator for LLM-powered bug generation.

**Key Methods:**

```python
class BugAuthor:
    def __init__(self, llm_service: Optional[LLMService] = None)
    def generate_bug_definition(
        self,
        module_name: str,
        patch_path: Path,
        symptom: str,
        max_retries: int = 3
    ) -> tuple[Optional[dict], bool]
```

**Features:**
- Golden dataset loader (3 validated examples)
- System prompt builder with complete v2.1 schema
- Patch file parser (unified diff format)
- Validation loop with self-correction
- AST normalization for comparison

### 2. CLI Command (`engine create-bug`)

Developer-facing CLI command for single-bug generation.

**Usage:**
```bash
engine create-bug attention \
  --patch curricula/cs336_a1/modules/attention/bugs/missing_mask.patch \
  --output bugs/missing_mask.json \
  --symptom symptom.txt
```

**Features:**
- Automatic output path inference
- Rich UI with progress indicators
- JSON preview on success
- Error reporting with diagnostics

### 3. Batch Migration Script (`scripts/migrate_bugs_llm.py`)

Automated batch conversion of all remaining .patch files.

**Usage:**
```bash
python scripts/migrate_bugs_llm.py
```

**Features:**
- Automatic module scanning
- Skip existing JSON files
- Progress tracking with summaries
- Failure reporting for manual review

---

## Prompt Engineering

### System Prompt Structure

```
┌────────────────────────────────────────────────┐
│ 1. Role Definition                              │
│    "You are an expert in Python AST..."        │
├────────────────────────────────────────────────┤
│ 2. JSON Schema v2.1                             │
│    Complete schema with all node types,        │
│    operators, conditions, and replacements     │
├────────────────────────────────────────────────┤
│ 3. Pattern Matching Rules                       │
│    - node_type must match AST class            │
│    - Operators use class names                 │
│    - Context references use {"from_context"}   │
├────────────────────────────────────────────────┤
│ 4. Transformation Types                         │
│    - replace_value_with                        │
│    - replace_with                              │
│    - remove_keyword_arg                        │
├────────────────────────────────────────────────┤
│ 5. Multi-Pass Strategy                          │
│    - find_and_track for reconnaissance         │
│    - find_and_replace for transformation       │
├────────────────────────────────────────────────┤
│ 6. Golden Dataset (3 examples)                  │
│    Example 1: Softmax (complex, multi-pass)    │
│    Example 2: SiLU (simple, single-pass)       │
│    Example 3: RMSNorm (medium, arg removal)    │
└────────────────────────────────────────────────┘
```

### User Prompt Structure

```
┌────────────────────────────────────────────────┐
│ 1. Task Description                             │
│    "Generate Bug Definition for 'module_name'"  │
├────────────────────────────────────────────────┤
│ 2. Bug Description (from symptom.txt)           │
├────────────────────────────────────────────────┤
│ 3. Code Transformation                           │
│    BEFORE (Correct): [extracted from patch]    │
│    AFTER (Buggy): [extracted from patch]       │
├────────────────────────────────────────────────┤
│ 4. Critical Requirements                         │
│    - Semantic matching (not variable names)    │
│    - Context tracking when needed              │
│    - Appropriate pass types                    │
├────────────────────────────────────────────────┤
│ 5. Output Format                                 │
│    "Return ONLY valid JSON (no markdown)"      │
└────────────────────────────────────────────────┘
```

---

## Validation Loop

### Three-Stage Validation

```python
def generate_bug_definition(...) -> tuple[Optional[dict], bool]:
    for attempt in range(max_retries):
        # 1. Generate
        response = llm_service.generate_completion(...)
        
        # 2. Parse JSON
        try:
            bug_def = json.loads(response)
        except JSONDecodeError:
            retry_with_error("Invalid JSON")
            continue
        
        # 3. Schema Validation
        if not validate_schema(bug_def):
            retry_with_error("Schema validation failed")
            continue
        
        # 4. Injection Test
        if not test_bug_definition(bug_def, correct_code, buggy_code):
            retry_with_error("Injection test failed")
            continue
        
        # Success!
        return bug_def, True
    
    return None, False
```

### Self-Correction Mechanism

On validation failure, the tool appends error feedback to the user prompt:

```
**Error in attempt 1:** Invalid JSON - Expecting ',' delimiter: line 23 column 5 (char 456)
Please try again with valid JSON.
```

This creates a feedback loop where the LLM can learn from its mistakes.

---

## Golden Dataset

### Three Validated Examples

| Example | Complexity | Pattern Type | Features |
|---------|-----------|--------------|----------|
| **Softmax** | Complex | Multi-pass | Context tracking, variable references |
| **SiLU** | Simple | Single-pass | Node replacement, straightforward |
| **RMSNorm** | Medium | Single-pass | Keyword argument manipulation |

These examples are carefully chosen to represent different tiers of complexity and different transformation strategies, providing comprehensive guidance to the LLM.

---

## Usage Examples

### Single Bug Generation

```bash
# Generate single bug with explicit paths
engine create-bug attention \
  --patch curricula/cs336_a1/modules/attention/bugs/missing_mask.patch \
  --symptom curricula/cs336_a1/modules/attention/symptom.txt \
  --output curricula/cs336_a1/modules/attention/bugs/missing_mask.json

# Auto-infer output path
engine create-bug attention \
  --patch curricula/cs336_a1/modules/attention/bugs/missing_mask.patch
```

### Batch Migration

```bash
# Migrate all remaining patches
python scripts/migrate_bugs_llm.py

# Output:
# 📁 Scanning curricula/cs336_a1...
# ✨ Found 17 patch files to migrate
# 
# [1/17] attention: missing_mask.patch
# 🤖 LLM Attempt 1/3...
#   ✅ Validation passed!
# ✅ Success! Wrote missing_mask.json
# ...
```

---

## Testing Strategy

### Unit Tests

```python
def test_golden_dataset_loads():
    """Verify golden dataset files exist and parse correctly."""
    
def test_patch_parser():
    """Verify patch parsing extracts before/after code."""
    
def test_prompt_construction():
    """Verify system and user prompts are well-formed."""
    
def test_validation_loop():
    """Verify validation catches invalid JSON and failed injections."""
```

### Integration Tests

```python
def test_end_to_end_generation():
    """Generate bug definition from real patch and validate."""
    
def test_self_correction():
    """Verify tool retries on failure with error feedback."""
```

### Manual Validation

After batch migration:
1. Review generated JSON files for sanity
2. Test each with GenericBugInjector on reference code
3. Run full E2E test for critical bugs
4. Commit reviewed and validated definitions

---

## Quality Metrics

### Success Criteria

- ✅ **Parsing Success Rate**: >95% (valid JSON output)
- ✅ **Schema Validation Rate**: >90% (matches v2.1 schema)
- ✅ **Injection Test Pass Rate**: >80% (produces correct buggy code)
- ✅ **Human Review Pass Rate**: >90% (semantically correct after review)

### Failure Modes

| Failure Type | Cause | Mitigation |
|--------------|-------|------------|
| Invalid JSON | LLM hallucination | Lower temperature (0.3) |
| Schema mismatch | Missing required fields | Explicit schema in prompt |
| Wrong pattern | AST structure misunderstanding | Golden examples with diverse patterns |
| Failed injection | Incorrect transformation logic | Test validation with clear error feedback |

---

## Risk Assessment

### Low Risk

- ✅ **Golden dataset proven**: 3 examples validated in production
- ✅ **Schema stable**: No changes expected to v2.1
- ✅ **Validation loop**: Self-correction prevents bad output
- ✅ **Human review**: Final check before commit

### Mitigation Strategies

1. **Start with simple bugs**: Migrate easiest cases first
2. **Manual review required**: No automatic commits
3. **Iterative approach**: Review and adjust prompts after first batch
4. **Fallback plan**: Manual JSON authoring for complex edge cases

---

## Timeline & Milestones

### Phase 4 Plan

| Milestone | Status | Description |
|-----------|--------|-------------|
| 4.1 Tool Architecture | ✅ | BugAuthor class implemented |
| 4.2 CLI Integration | ✅ | `engine create-bug` command |
| 4.3 Batch Script | ✅ | `migrate_bugs_llm.py` script |
| 4.4 Testing | 🚧 | Unit tests for BugAuthor |
| 4.5 Batch Migration | ⏳ | Run on all 17 remaining bugs |
| 4.6 Human Review | ⏳ | Review and validate all JSON |
| 4.7 Documentation | ✅ | This document |

**Estimated Total Time**: 4-6 hours  
**Current Progress**: Foundation complete (60%)

---

## Next Steps

### Immediate Actions

1. ✅ Test CLI command with single bug
2. ⏳ Run batch migration on all patches
3. ⏳ Manual review of generated JSON
4. ⏳ Commit validated definitions
5. ⏳ Update curriculum metadata

### Future Enhancements

- **Prompt refinement**: Improve based on failure analysis
- **Context window optimization**: Compress golden examples if needed
- **Multi-model support**: Test with different LLMs (GPT-4, Claude, etc.)
- **Interactive mode**: Allow human-in-the-loop corrections during generation
- **Diff visualization**: Show before/after AST for verification

---

## Conclusion

Phase 4 builds on the stable foundation of Phase 3 to provide a force multiplier for bug creation. By leveraging LLM few-shot learning with the proven golden dataset, we can rapidly convert the remaining 17 legacy bugs while maintaining quality through validation and human review.

The systematic approach—validate manually first, then automate—has de-risked this phase and positioned us for success.

**Status**: Foundation complete, ready for batch migration.

---

**Document Version:** 1.0  
**Author:** Cascade AI  
**Date:** November 13, 2025
