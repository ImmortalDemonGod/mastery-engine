# CS336 Assignment 1: Deep Learning Basics

## 📜 Attribution

**Source:** Adapted from **[Stanford CS336: Language Modeling from Scratch (Spring 2024)](https://stanford-cs336.github.io/spring2024/)**.

### Content Ownership

- **Problem Statements:** Mathematical specifications and implementation requirements derive from Stanford CS336 course assignments.
- **Reference Implementations:** Build prompts and solution patterns follow the course structure.
- **Intellectual Property:** Course materials remain property of Stanford University and the original course instructors.

### Mastery Engine Extensions (Original Work)

The following components were **engineered specifically for this platform** and do not appear in the original Stanford course:

1. **"Justify" Questions:** Conceptual understanding evaluation with LLM-powered Socratic dialogue
2. **"Harden" Bug Patterns:** Semantic bug injection via AST mutation (not present in original coursework)
3. **Test Harness:** Automated validation with performance metrics
4. **Mock Mode:** Credential-free demo capability
5. **Shadow Worktree:** Process isolation for safe bug debugging

**Critical Distinction:** The reference implementations answer *what to build*. The Mastery Engine provides the *how to validate, explain, and debug*.

## Curriculum Overview

**Target Audience:** Deep learning practitioners building transformer models from scratch

**Topics Covered:**
1. BPE Tokenization
2. Softmax & Activation Functions (SiLU, SwiGLU, RMSNorm)
3. Attention Mechanisms (Scaled Dot-Product, Multihead, RoPE)
4. Loss Functions & Optimization (Cross-Entropy, AdamW, Cosine Schedule)
5. Training Loop & Gradient Clipping
6. Text Generation (Temperature, Top-k, Top-p)

**Completion Time:** ~60 hours

**Tech Stack:**
- PyTorch 2.0+
- einops (declarative tensor operations)
- tiktoken (BPE tokenizer interface)

## Module Structure

Each module follows the Build-Justify-Harden loop:

**Build:** Implement the component (validated by unit tests)
**Justify:** Answer conceptual questions (evaluated by LLM)
**Harden:** Debug a semantic bug injected into YOUR correct code

## Usage

```bash
# Initialize curriculum
uv run mastery init cs336_a1

# Start working
uv run mastery show      # View current module
uv run mastery submit    # Build stage
uv run mastery submit    # Justify stage
uv run mastery start-challenge  # Harden stage
```

## Educational Philosophy

This curriculum emphasizes:
1. **Implementation-first learning** - Build before you theorize
2. **Conceptual understanding** - Justify forces articulation
3. **Production debugging** - Harden simulates real mistakes

## License

See root [LICENSE](../../LICENSE) file for attribution details.

**TL;DR:** Engine is MIT (original work). Curriculum content adapted from Stanford CS336 (educational fair use).
