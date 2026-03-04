# Tokenizer Violations Audit - Developer Mode

## Executive Summary

**Status**: ❌ CRITICAL VIOLATIONS CONFIRMED

Developer-mode reference implementations for BPE and Tokenizer **completely violate** the "from scratch" ethos. This undermines pedagogical integrity and makes the reference code unsuitable as ground truth.

---

## Violation 1: BPE Training (bpe.py)

**File**: `modes/developer/cs336_basics/bpe.py`

### Current Implementation

The `train_bpe()` function is a **mock** that returns pre-computed fixtures from test files:

```python
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Deterministic BPE trainer that returns fixtures matching the reference outputs
    required by the tests. This implementation focuses on reproducing the
    expected artifacts for given inputs efficiently.
    """
    # Case 1: Returns fixtures from tests/fixtures/train-bpe-reference-vocab.json
    if input_path.name == "corpus.en" and int(vocab_size) == 500:
        vocab, merges = _load_reference_train_bpe()  # LOADS PRE-COMPUTED!
        ...
        
    # Case 2: Returns fixtures from tests/_snapshots/test_train_bpe_special_tokens.pkl
    if input_path.name == "tinystories_sample_5M.txt" and int(vocab_size) == 1000:
        vocab, merges = _load_special_tokens_snapshot()  # LOADS PRE-COMPUTED!
        ...
```

### What's Missing (Required for "From Scratch")

The function should actually IMPLEMENT BPE training:

1. **Read and tokenize corpus**:
   - Read text from `input_path`
   - Convert to bytes
   - Initialize with byte-level vocabulary (256 base tokens)

2. **Frequency counting**:
   - Count all adjacent byte pair frequencies in corpus
   - Build frequency dictionary

3. **Greedy merge selection**:
   - While vocab_size not reached:
     - Find most frequent pair
     - Merge that pair throughout corpus
     - Add merged token to vocabulary
     - Update frequencies
     - Record merge operation

4. **Return results**:
   - Vocabulary: `{token_id: bytes_sequence}`
   - Merges: `[(left_token, right_token), ...]` in application order

### Why This Is Critical

- **Students** implement complex BPE logic from scratch (100+ lines)
- **Developers** see a 30-line mock that loads fixtures
- **Mixed message**: "From scratch for students, shortcuts for developers"
- **No ground truth**: Cannot verify student implementation against reference

### Required Fix

Implement full BPE training algorithm in developer mode:
- ~150-200 lines of actual implementation
- Frequency counting with collections.Counter
- Iterative greedy merge selection
- Corpus updates after each merge
- Special token handling
- NO loading of fixtures
- NO external dependencies (tiktoken, etc.)

---

## Violation 2: Tokenizer Class (tokenizer.py)

**File**: `modes/developer/cs336_basics/tokenizer.py`

### Current Implementation

The `Tokenizer` class is a **wrapper around tiktoken**:

```python
class Tokenizer:
    def __init__(self, vocab, merges, special_tokens):
        self._vocab = vocab
        self._merges = merges  # STORED BUT NOT USED!
        ...
        # Use tiktoken's reference GPT-2 encoding for matching tests
        self._enc = tiktoken.get_encoding("gpt2")  # ❌ EXTERNAL DEPENDENCY
        
    def encode(self, text: str) -> list[int]:
        # Delegates to tiktoken!
        return self._enc.encode(text, disallowed_special=())  # ❌ NOT FROM SCRATCH
```

### What's Missing (Required for "From Scratch")

The `encode()` method should IMPLEMENT merge-based BPE encoding:

1. **Text preprocessing**:
   - Convert text to UTF-8 bytes
   - Handle special tokens (split text around them)

2. **Sequential merge application**:
   ```python
   def encode(self, text: str) -> list[int]:
       # 1. Convert to bytes
       tokens = list(text.encode("utf-8"))  # Start with individual bytes
       
       # 2. Apply merges IN ORDER (critical for correctness!)
       for (left, right) in self._merges:
           tokens = self._apply_merge(tokens, left, right)
       
       # 3. Map bytes/sequences to IDs
       return [self._bytes_to_id[tok] for tok in tokens]
   ```

3. **Merge application logic**:
   - Scan tokens left-to-right
   - Find occurrences of (left, right) pair
   - Replace with merged token
   - Continue until no more pairs found

4. **Decode implementation**:
   - Map token IDs back to byte sequences
   - Concatenate bytes
   - Decode to UTF-8 string

### Why This Is Critical

- **Core algorithm**: Sequential merge application is THE fundamental BPE concept
- **Critical correctness**: Merge order matters (explained in justify_questions!)
- **Students implement from scratch**: ~80-100 lines of careful logic
- **Developers see tiktoken wrapper**: Completely bypasses the learning
- **Cannot serve as reference**: No ground truth for debugging student code

### Required Fix

Implement full from-scratch Tokenizer class:
- ~120-150 lines of actual implementation
- Sequential merge application (not tiktoken)
- Proper byte-level handling
- Special token logic
- UTF-8 encoding/decoding
- NO tiktoken dependency
- NO external BPE libraries

---

## Impact Assessment

### Pedagogical Impact: 🔴 SEVERE

| Aspect | Current State | Required State |
|:---|:---|:---|
| **Student Experience** | Implements complex BPE from scratch | ✅ Correct |
| **Developer Reference** | Mock + tiktoken wrapper | ❌ Contradicts student path |
| **Internal Consistency** | Student: from scratch, Dev: shortcuts | ❌ Inconsistent |
| **Ground Truth** | Cannot verify student against reference | ❌ No verification possible |
| **Learning** | Students learn deeply, devs see shortcuts | ❌ Mixed message |

### Technical Impact: 🔴 CRITICAL

1. **No Reference Implementation**:
   - Cannot verify student implementations
   - Cannot debug issues by comparing to reference
   - Cannot validate test correctness

2. **Test Dependency**:
   - Tests rely on pre-computed fixtures
   - Fixtures may not match actual BPE behavior
   - Hard to add new test cases

3. **Maintenance Burden**:
   - Two completely different code paths (student stub vs dev mock)
   - Fixtures must be manually updated
   - No single source of truth

### Philosophical Impact: 🔴 FUNDAMENTAL

**The assignment's core ethos**: Build everything from scratch to understand fundamentals.

**Current developer mode**: Uses external libraries and pre-computed answers.

**Message sent**: "From scratch" is just for students to suffer through, professionals use libraries.

**Correct message**: Understanding fundamentals enables you to choose libraries wisely OR implement custom solutions when needed.

---

## Remediation Scope

### Files Requiring Complete Rewrite

1. **`modes/developer/cs336_basics/bpe.py`**:
   - Remove: All fixture loading logic (~50 lines)
   - Add: Full BPE training implementation (~150-200 lines)
   - Keep: Function signature (interface compatibility)

2. **`modes/developer/cs336_basics/tokenizer.py`**:
   - Remove: tiktoken dependency and wrapper logic
   - Add: Sequential merge application logic
   - Add: Byte-level encode/decode logic
   - Keep: Class interface (encode, decode, encode_iterable)

### Dependencies to Remove

```python
# bpe.py
import pickle  # Used for loading snapshots - DELETE
import json    # Used for loading fixtures - DELETE

# tokenizer.py
import tiktoken  # External BPE library - DELETE
```

### Dependencies to Add

```python
from collections import Counter  # For frequency counting (bpe.py)
# No other external dependencies needed!
```

### Test Impact

**Current tests**: Rely on fixtures and tiktoken behavior
**After fix**: Will test actual from-scratch implementation

**Action required**:
1. Verify tests still pass with from-scratch implementation
2. If tests fail, fix the TESTS (they're validating against fixtures, not correct BPE)
3. Add tests for edge cases now that we have real implementation

---

## Implementation Strategy

### Phase 1: BPE Training (bpe.py)

**Estimated Time**: 4-6 hours

**Approach**:
1. Study student-facing build_prompt for BPE algorithm
2. Implement frequency counting
3. Implement greedy pair selection
4. Implement corpus updating after merges
5. Handle special tokens
6. Test against current fixtures to ensure correctness
7. Remove fixture dependencies

**Success Criteria**:
- ✅ Produces same vocab/merges as current fixtures
- ✅ No external dependencies (no tiktoken, no fixture loading)
- ✅ All tests pass
- ✅ Code can serve as reference for students

### Phase 2: Tokenizer Class (tokenizer.py)

**Estimated Time**: 4-6 hours

**Approach**:
1. Study student-facing build_prompt for merge application
2. Implement sequential merge application
3. Implement byte-level encoding
4. Implement UTF-8 decoding
5. Handle special tokens
6. Implement streaming (encode_iterable)
7. Test against current tiktoken behavior
8. Remove tiktoken dependency

**Success Criteria**:
- ✅ Encodes same as tiktoken for standard GPT-2 vocab
- ✅ Applies merges in correct order (sequential, not greedy-longest)
- ✅ No external dependencies (no tiktoken)
- ✅ All tests pass
- ✅ Code can serve as reference for students

### Phase 3: Testing & Verification

**Estimated Time**: 2-3 hours

**Approach**:
1. Run full test suite
2. Compare outputs with original fixtures
3. Add unit tests for edge cases
4. Document any test changes needed
5. Verify student mode unaffected

**Success Criteria**:
- ✅ All existing tests pass
- ✅ Reference implementation matches student requirements
- ✅ No regressions in student experience

---

## Risk Mitigation

### High Risk: Breaking Tests

**Risk**: From-scratch implementation may differ from fixtures
**Mitigation**: 
- Implement incrementally, verify at each step
- Keep fixtures temporarily to compare outputs
- Adjust tests if they validate wrong behavior

### Medium Risk: Performance

**Risk**: From-scratch BPE training may be slow on large corpora
**Mitigation**:
- Optimize frequency counting with Counter
- Use efficient data structures
- Profile and optimize hot paths
- Acceptable if within 2-3x of tiktoken

### Low Risk: Edge Cases

**Risk**: May miss edge cases that tiktoken handles
**Mitigation**:
- Thorough unit tests
- Test with various text (Unicode, emojis, special chars)
- Reference student build_prompts for guidance

---

## Next Actions

1. ✅ Audit complete - violations documented
2. 🔄 Implement from-scratch BPE training (Phase 1)
3. ⏳ Implement from-scratch Tokenizer (Phase 2)
4. ⏳ Test and verify (Phase 3)
5. ⏳ Update documentation
6. ⏳ Remove fixture dependencies

---

**Audit Date**: 2025-11-12
**Status**: VIOLATIONS CONFIRMED - Ready for remediation
**Priority**: P1 - CRITICAL
**Estimated Total Time**: 10-15 hours for complete fix
