# CS336 Assignment 1: Deep Learning Basics

## Attribution

This curriculum is **adapted from Stanford CS336 (Language Modeling)** coursework.

**Original Course:**
- Institution: Stanford University
- Course: CS336 - Language Modeling from Scratch
- Website: https://stanford-cs336.github.io/spring2024/
- Assignment: Assignment 1 - Basics

**Content Ownership:**
Course materials, problem descriptions, and pedagogical structure remain property of Stanford University and the original course instructors.

**What is Original (Mastery Engine Contributions):**
- Engine architecture for Build-Justify-Harden pedagogy
- AST-based bug injection system
- Automated test harness and validation
- LLM-powered justify evaluation
- Shadow worktree harden challenges
- Mock mode fallback for demo purposes

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
