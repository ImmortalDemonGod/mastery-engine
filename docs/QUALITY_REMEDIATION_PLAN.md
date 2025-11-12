# Quality Remediation Plan - Critical Issues

## Overview

This document addresses critical issues identified in the systematic quality analysis comparing the MVP curriculum against CS336 Assignment 1 PDF (ground truth).

**Overall Assessment**: Curriculum is pedagogically exceptional (92.4/100) but has internal consistency flaws that must be fixed.

---

## Priority 1: CRITICAL - Reference Implementation Violations

### Issue 1.1: Developer-Mode Tokenizer Violations

**Problem**: The "from scratch" ethos is violated in developer mode reference implementations.

**Current State**:
- `modes/developer/cs336_basics/bpe.py`: Returns pre-computed fixtures (mock)
- `modes/developer/cs336_basics/tokenizer.py`: Wraps `tiktoken` library

**Why This Is Critical**:
- Students guided to implement from scratch, developers see shortcuts
- Mixed message undermines pedagogical integrity
- Reference code cannot serve as ground truth
- Violates core assignment philosophy (PDF §3.1)

**Remediation Required**:
1. **Implement true from-scratch BPE training** in `modes/developer/cs336_basics/bpe.py`:
   - Frequency counting
   - Greedy pair selection
   - Merge application
   - Vocabulary building
   - NO tiktoken, NO fixtures

2. **Implement true from-scratch Tokenizer class** in `modes/developer/cs336_basics/tokenizer.py`:
   - Sequential merge application in encode()
   - Byte-level BPE logic
   - UTF-8 handling
   - Special token management
   - NO tiktoken wrapper

**Success Criteria**:
- Developer mode can train BPE and tokenize without external dependencies
- Code serves as reference implementation for students
- All tests pass with from-scratch implementation

---

### Issue 1.2: Einops Not Used in Reference Code

**Problem**: PDF §3.3 (p.15) strongly recommends `einops` for tensor operations, but developer-mode reference uses standard PyTorch (.view, .transpose, .matmul).

**Why This Is Critical**:
- Contradicts PDF's explicit best-practice guidance
- Misses teaching opportunity for modern, self-documenting tensor operations
- Reference code should model industry standards

**Current Violations** (to be audited):
- `modes/developer/cs336_basics/layers.py`:
  - `multihead_self_attention`: Uses .view() and .transpose()
  - `multihead_self_attention_with_rope`: Complex reshaping with .reshape()
  - `transformer_block`: Multiple tensor manipulations
  - `rope`: Dimension pairing with manual reshaping

**Remediation Required**:
1. **Audit all tensor operations** in developer mode for einops opportunities
2. **Refactor to use einops** where appropriate:
   - `einops.rearrange` for reshaping with semantic labels
   - `einops.reduce` for pooling/aggregation
   - `einops.repeat` for broadcasting
3. **Update build_prompts** to recommend einops with examples
4. **Add einops to requirements.txt** if not already present

**Example Transformation**:
```python
# Before (current)
x = x.view(batch, seq_len, num_heads, d_k).transpose(1, 2)

# After (with einops) - self-documenting!
x = rearrange(x, 'b s (h d) -> b h s d', h=num_heads)
```

**Success Criteria**:
- All appropriate tensor operations use einops
- Code is more readable and self-documenting
- Aligns with PDF's best-practice recommendations

---

## Priority 2: MEDIUM - Coverage Gaps

### Issue 2.1: Unicode Questions Not in BJH Framework

**Problem**: PDF §2.1-2.2 has explicit written questions on Unicode (`unicode1`, `unicode2`) that are not formally tested in the BJH loop.

**Current State**:
- Unicode concepts touched on in `bpe_tokenizer` build_prompt
- Not formally tested in justify_questions.json
- No module for theoretical-only questions

**Remediation Required**:
1. **Create new module type**: "Justify-Only" (no build/harden stages)
2. **Create unicode module**: `curricula/cs336_a1/modules/unicode/`
   - No build_prompt.txt (theoretical only)
   - `justify_questions.json` with Unicode questions:
     - Why UTF-8 is variable-length
     - Byte-level vs character-level tokenization
     - Handling of multi-byte characters
     - Emoji and combining sequences
3. **Update manifest.json** to include unicode module
4. **Update engine** to handle justify-only modules

**Success Criteria**:
- Unicode understanding formally tested
- Students cannot skip theoretical foundations
- Framework extended for theory-only modules

---

### Issue 2.2: Experimental Process Not in BJH Framework

**Problem**: PDF §7 treats experiments as culmination, but curriculum doesn't apply BJH framework to experimental process.

**Current State**:
- Experiments mentioned informally
- No structured guidance on experimental design
- Missing: hypothesis formation, ablation design, result interpretation

**Remediation Required**:
1. **Design "Experiment" module type**:
   - Build stage → Hypothesis formulation
   - Justify stage → Experimental design justification
   - Harden stage → Result interpretation / debugging bad experiments

2. **Create experiment modules** for PDF §7 requirements:
   - Ablation studies (remove components, measure impact)
   - Hyperparameter sensitivity
   - Scaling behavior
   - Comparative analysis

3. **Example module**: `ablation_study`
   - Build: Design ablation removing RoPE, measure perplexity
   - Justify: Why this ablation tests position encoding importance
   - Harden: Bug in experimental setup (e.g., incorrect baseline)

**Success Criteria**:
- Experimental process structured and rigorous
- Students learn scientific method, not just implementation
- BJH framework applied to experiments

---

## Execution Plan

### Phase 1: Critical Fixes (Priority 1)
**Estimated Time**: 2-3 sessions

**Week 1**:
- Day 1: Audit tokenizer violations, plan from-scratch implementation
- Day 2: Implement from-scratch BPE training
- Day 3: Implement from-scratch Tokenizer class
- Day 4: Test and verify tokenizer implementation

**Week 2**:
- Day 1: Audit einops violations in reference code
- Day 2-3: Refactor to use einops throughout
- Day 4: Update build_prompts with einops examples
- Day 5: Test and verify all reference code

### Phase 2: Coverage Enhancements (Priority 2)
**Estimated Time**: 1-2 sessions

**Week 3**:
- Day 1: Design justify-only module type
- Day 2: Create unicode module with questions
- Day 3: Update engine to handle justify-only modules
- Day 4: Design experiment module framework
- Day 5: Create example experiment modules

### Phase 3: Verification & Documentation
**Estimated Time**: 1 session

**Week 4**:
- Day 1: Run full test suite
- Day 2: Verify all fixes against PDF
- Day 3: Update documentation
- Day 4: Final quality review

---

## Success Metrics

**Internal Consistency**:
- ✅ Developer mode fully "from scratch" (no mocks, no tiktoken)
- ✅ Reference code uses einops per PDF guidance
- ✅ No contradictions between student guidance and developer reference

**Coverage Completeness**:
- ✅ 100% implementation coverage (already achieved)
- ✅ 100% theoretical coverage (Unicode questions added)
- ✅ 100% experimental coverage (§7 structured in BJH)

**Quality Elevation**:
- Current: 92.4/100 (pedagogically exceptional, some flaws)
- Target: 98/100 (pedagogically exceptional, internally consistent)

---

## Risk Assessment

**Low Risk**:
- Unicode module creation (straightforward extension)
- Build prompt updates (documentation only)

**Medium Risk**:
- Einops refactoring (must ensure correctness, all tests pass)
- Experiment module design (new framework, needs careful thought)

**High Risk**:
- From-scratch tokenizer implementation (complex logic, critical correctness)
- Must not break existing student experience
- Requires extensive testing

**Mitigation**:
- Implement incrementally with tests at each step
- Maintain backward compatibility
- Thorough code review before deployment

---

## Tracking

| Task | Priority | Status | Owner | Completion |
|:---|:---:|:---:|:---:|:---:|
| Audit tokenizer violations | P1 | 🔴 Not Started | - | 0% |
| Implement from-scratch BPE | P1 | 🔴 Not Started | - | 0% |
| Implement from-scratch Tokenizer | P1 | 🔴 Not Started | - | 0% |
| Audit einops violations | P1 | 🔴 Not Started | - | 0% |
| Refactor to einops | P1 | 🔴 Not Started | - | 0% |
| Update build prompts (einops) | P1 | 🔴 Not Started | - | 0% |
| Create unicode module | P2 | 🔴 Not Started | - | 0% |
| Design experiment framework | P2 | 🔴 Not Started | - | 0% |
| Create experiment modules | P2 | 🔴 Not Started | - | 0% |
| Final verification | P3 | 🔴 Not Started | - | 0% |

---

**Last Updated**: 2025-11-12
**Status**: Plan Created - Ready for Execution
**Next Action**: Begin Priority 1.1 - Audit developer-mode tokenizer violations
