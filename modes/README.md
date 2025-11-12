# Modes Directory - Design Rationale

## Purpose

This directory contains the source of truth for student and developer modes.

## What Gets Stubbed vs What Stays Complete

### Student Mode (`modes/student/cs336_basics/`)

**Stubbed (TODO implementations):**
- `utils.py` - Contains 3 functions taught by the curriculum:
  - `softmax()` - Numerically stable softmax
  - `cross_entropy()` - Numerically stable cross-entropy loss
  - `gradient_clipping()` - Global gradient clipping

**Complete (scaffolding for LLM):**
- `layers.py` - Neural network layers (Linear, Embedding, RMSNorm, SwiGLU, attention, transformer blocks, RoPE)
- `tokenizer.py` - BPE tokenizer implementation
- `bpe.py` - Byte Pair Encoding algorithm
- `optimizer.py` - AdamW optimizer
- `pretokenization_example.py` - Example preprocessing scripts
- `__init__.py` - Package initialization

### Developer Mode (`modes/developer/cs336_basics/`)

**Complete implementations of everything:**
- `utils.py` - All 3 functions fully implemented
- All other files identical to student mode

## Why This Design?

### The CS336 Curriculum Goal

The original CS336 assignment is about building an LLM from scratch. Students need:

1. **Core utilities** (curriculum teaches these):
   - Numerically stable softmax
   - Cross-entropy loss
   - Gradient clipping

2. **LLM scaffolding** (provided as complete):
   - Transformer architecture (layers.py)
   - Tokenizer (BPE)
   - Optimizer (AdamW)

### What Students Can Do After Finishing

After completing the 3 curriculum modules (softmax, cross_entropy, gradient_clipping), students have:

✅ Complete cs336_basics package  
✅ Can train a language model  
✅ Can use all the transformer components  
✅ Have working implementations of the 3 critical functions they learned

### Mastery Engine Curriculum Scope

The Mastery Engine **only teaches the 3 functions** in the curriculum manifest:
- Module 1: `softmax()`
- Module 2: `cross_entropy()`  
- Module 3: `gradient_clipping()`

Everything else (layers, tokenizer, etc.) is **pre-implemented scaffolding** that enables the LLM training workflow.

## File Verification

Run this to verify modes are set up correctly:

```bash
# All support files should be identical in both modes
diff modes/student/cs336_basics/layers.py modes/developer/cs336_basics/layers.py
# Should output: (no differences)

# Only utils.py should differ
diff modes/student/cs336_basics/utils.py modes/developer/cs336_basics/utils.py
# Should output: differences showing TODO stubs vs implementations
```

## Mode Switching

When you switch modes with `./scripts/mode switch student|developer`:

- **Student mode**: Tests fail until you implement softmax, cross_entropy, gradient_clipping
- **Developer mode**: Tests pass, you can develop engine features

Both modes have complete LLM scaffolding, so students can:
- Run the LLM training after finishing the curriculum
- Use transformer layers, tokenizer, optimizer without implementing them
- Focus on learning the 3 core numerical stability concepts

## Future Curriculum Expansion

To add more modules to the curriculum:

1. Add module to `curricula/cs336_a1/manifest.json`
2. Create module content in `curricula/cs336_a1/modules/{module_name}/`
3. If teaching a new function:
   - Add TODO stub to `modes/student/cs336_basics/{file}.py`
   - Add complete implementation to `modes/developer/cs336_basics/{file}.py`
4. Keep all other files identical in both modes

The dual-mode architecture supports expanding the curriculum without breaking existing scaffolding.
