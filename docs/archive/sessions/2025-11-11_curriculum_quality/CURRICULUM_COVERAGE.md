# CS336 Assignment 1 - Curriculum Coverage

## Overview

This document maps the complete 21-module curriculum to the CS336 Assignment 1 specification (PDF). It demonstrates **100% coverage** of all required implementations with the **"from scratch" ethos** rigorously maintained.

**Key Principle**: No use of `nn.Linear` or `nn.Embedding` - all components built using `nn.Parameter` directly.

## PDF Requirements Mapping

### §2: Tokenization (PDF §2.1-2.6)

| PDF Section | Requirement | Module | Status |
|:---|:---|:---|:---:|
| §2.1-2.5 | BPE algorithm implementation | `bpe_tokenizer` | ✅ |
| §2.6 | Tokenizer class with encode/decode | `tokenizer_class` | ✅ |

**Implementation Details:**
- `train_bpe()` in `bpe.py` - Learn vocabulary from corpus
- `Tokenizer` class in `tokenizer.py` - Encode/decode with special tokens
- UTF-8 byte-level for multilingual support
- Sequential merge application for deterministic encoding

---

### §3: Model Architecture (PDF §3.1-3.6)

#### §3.4.1: Activation Functions

| Requirement | Module | Implementation | Status |
|:---|:---|:---|:---:|
| SiLU activation | `silu` | `silu(x)` function | ✅ |

**From Scratch**: Hand-coded as `x * torch.sigmoid(x)`, no `F.silu()`.

#### §3.4.2: Linear Layers

| Requirement | Module | Implementation | Status |
|:---|:---|:---|:---:|
| Fully-connected layer | `linear` | `Linear` class | ✅ |

**From Scratch**: 
- Uses `nn.Parameter` for weights and bias
- Kaiming uniform initialization
- Forward: `y = x @ W^T + b`
- **NO use of `nn.Linear`**

#### §3.4.3: Embeddings

| Requirement | Module | Implementation | Status |
|:---|:---|:---|:---:|
| Token embedding table | `embedding` | `Embedding` class | ✅ |

**From Scratch**:
- Uses `nn.Parameter` for embedding matrix
- Truncated normal initialization
- Forward: Direct indexing `weight[token_ids]`
- **NO use of `nn.Embedding`**

#### §3.4.4: Normalization

| Requirement | Module | Implementation | Status |
|:---|:---|:---|:---:|
| RMSNorm (no bias) | `rmsnorm` | `RMSNorm` class | ✅ |

**Implementation**: `x / sqrt(mean(x²) + eps) * weight`

#### §3.4.5: SwiGLU Feed-Forward

| Requirement | Module | Implementation | Status |
|:---|:---|:---|:---:|
| Gated FFN with SiLU | `swiglu` | `SwiGLU` class | ✅ |

**Architecture**: `W2(SiLU(W1(x)) * W3(x))`
- Three linear projections (W1, W2, W3)
- Element-wise gating with SiLU

#### §3.4.6: Attention

| Requirement | Module | Implementation | Status |
|:---|:---|:---|:---:|
| Scaled dot-product attention | `attention` | `scaled_dot_product_attention()` | ✅ |
| RoPE positional embeddings | `rope` | `rope()` function | ✅ |
| Multi-head attention + RoPE | `multihead_attention` | `multihead_self_attention_with_rope()` | ✅ |

**Key Features:**
- Causal masking for autoregressive generation
- Mixed precision (compute in fp32, return original dtype)
- Efficient multi-head computation with reshaping

#### §3.5: Transformer Block

| Requirement | Module | Implementation | Status |
|:---|:---|:---|:---:|
| Pre-norm transformer block | `transformer_block` | `transformer_block()` function | ✅ |

**Architecture**:
```
x = x + MHA(RMSNorm(x))
x = x + FFN(RMSNorm(x))
```

#### §3.6: Complete Model

| Requirement | Module | Implementation | Status |
|:---|:---|:---|:---:|
| Full Transformer LM | `transformer_lm` | `transformer_lm()` function | ✅ |

**Pipeline**:
1. Token embedding lookup
2. N × Transformer blocks
3. Final RMSNorm
4. LM head projection (to vocabulary)

---

### §4: Optimization (PDF §4.1-4.3)

#### §4.1: Loss Functions

| Requirement | Module | Implementation | Status |
|:---|:---|:---|:---:|
| Numerically stable softmax | `softmax` | `softmax()` function | ✅ |
| Numerically stable cross-entropy | `cross_entropy` | `cross_entropy()` function | ✅ |

**Numerical Stability**:
- Softmax: Subtract max for numerical stability
- Cross-entropy: Fused implementation avoiding explicit softmax

#### §4.2: AdamW Optimizer

| Requirement | Module | Implementation | Status |
|:---|:---|:---|:---:|
| AdamW with bias correction | `adamw` | `AdamW` class | ✅ |

**Features**:
- Exponential moving averages (m, v)
- Bias correction (unbiased estimates)
- Decoupled weight decay
- Per-parameter adaptive learning rates

#### §4.3: Learning Rate Schedule

| Requirement | Module | Implementation | Status |
|:---|:---|:---|:---:|
| Cosine schedule with warmup | `cosine_schedule` | `get_lr_cosine_schedule()` | ✅ |

**Schedule**:
- Linear warmup phase
- Cosine decay to min_lr
- Configurable warmup steps and total steps

#### Gradient Clipping

| Requirement | Module | Implementation | Status |
|:---|:---|:---|:---:|
| Global norm clipping | `gradient_clipping` | `gradient_clipping()` | ✅ |

---

### §5: Training (PDF §5.1-5.3)

#### §5.1: Data Loading

| Requirement | Module | Implementation | Status |
|:---|:---|:---|:---:|
| Batch sampling | `data_loader` | `get_batch()` function | ✅ |

**Features**:
- Random subsequence sampling
- Input/target pairs (shifted by 1)
- Efficient fancy indexing
- Device handling

#### §5.2: Checkpointing

| Requirement | Module | Implementation | Status |
|:---|:---|:---|:---:|
| Save checkpoint | `checkpointing` | `save_checkpoint()` function | ✅ |
| Load checkpoint | `checkpointing` | `load_checkpoint()` function | ✅ |

**Complete State**:
- Model weights
- Optimizer state (momentum buffers)
- Iteration count
- Device-agnostic loading

#### §5.3: Training Loop

| Requirement | Module | Implementation | Status |
|:---|:---|:---|:---:|
| Complete training integration | `training_loop` | Integration of all components | ✅ |

**Pipeline**:
1. Data loading (get_batch)
2. Forward pass (transformer_lm)
3. Loss computation (cross_entropy)
4. Backward pass
5. Gradient clipping
6. Optimizer step
7. LR schedule update
8. Periodic checkpointing

---

### §6: Generation (PDF §6.1-6.3)

| Requirement | Module | Implementation | Status |
|:---|:---|:---|:---:|
| Autoregressive generation | `text_generation` | `generate()` function | ✅ |

**Sampling Strategies**:
- Temperature scaling
- Top-k sampling
- Top-p (nucleus) sampling
- Greedy decoding

---

## Module Dependency Graph

```
Foundation (no dependencies):
  softmax, cross_entropy, gradient_clipping
  linear, embedding, silu, rmsnorm, rope
  adamw, cosine_schedule
  data_loader, checkpointing
  bpe_tokenizer

Architecture (depends on foundation):
  swiglu           → linear, silu
  attention        → softmax
  multihead_attention → attention, rope, linear
  transformer_block   → multihead_attention, swiglu, rmsnorm
  transformer_lm      → transformer_block, embedding, rmsnorm, linear

Application (depends on architecture):
  training_loop    → transformer_lm, adamw, cosine_schedule, 
                     data_loader, checkpointing, 
                     cross_entropy, gradient_clipping
  tokenizer_class  → bpe_tokenizer
  text_generation  → transformer_lm, tokenizer_class
```

## Student Implementation Requirements

### Files with Stubs (18 total components)

**`modes/student/cs336_basics/layers.py`** (10 components):
1. `Linear` class - Fully-connected layer
2. `Embedding` class - Token embeddings
3. `silu()` function - Activation
4. `RMSNorm` class - Normalization
5. `SwiGLU` class - Gated FFN
6. `scaled_dot_product_attention()` - Attention mechanism
7. `rope()` - Positional embeddings
8. `multihead_self_attention_with_rope()` - Multi-head attention
9. `transformer_block()` - Complete block
10. `transformer_lm()` - Full model

**`modes/student/cs336_basics/utils.py`** (7 components):
1. `softmax()` - Numerical stability
2. `cross_entropy()` - Loss function
3. `gradient_clipping()` - Gradient norm clipping
4. `get_lr_cosine_schedule()` - LR schedule
5. `get_batch()` - Data loading
6. `save_checkpoint()` - Model persistence
7. `load_checkpoint()` - Model restoration

**`modes/student/cs336_basics/optimizer.py`** (1 component):
1. `AdamW` class - Optimizer

**`modes/student/cs336_basics/bpe.py`** (1 component):
1. `train_bpe()` - Tokenizer training

**`modes/student/cs336_basics/generation.py`** (1 component):
1. `generate()` - Text generation

### Total: 20 Components + 1 Integration Module = 21 Modules

## Verification of "From Scratch" Compliance

### ✅ Compliant Implementations

All student stubs require:
- `nn.Parameter` for learnable weights
- Manual initialization (Kaiming, truncated normal)
- Explicit forward pass logic
- Matrix operations using `.matmul()`, `.t()`, etc.

### ❌ Prohibited Patterns

Students **cannot** use:
- `nn.Linear` - Must implement with `nn.Parameter`
- `nn.Embedding` - Must implement with `nn.Parameter`
- `F.cross_entropy` with logits - Must implement fused version
- `torch.optim.AdamW` - Must implement from scratch
- High-level training wrappers

## Curriculum Quality Metrics

### Content Created

| Component | Count | Total Lines |
|:---|---:|---:|
| Build prompts | 21 | ~5,500 |
| Justify questions | 105 (21 × 5) | ~4,200 |
| Validators | 21 | ~450 |
| Pedagogical bugs | 21 | ~210 |
| **Total** | **168 files** | **~10,360 lines** |

### Pedagogical Features

Each module includes:
- **Build prompt** (~250-300 lines):
  - Background and motivation
  - Mathematical foundations
  - Implementation constraints
  - Step-by-step guidance
  - Common pitfalls
  - Integration examples
  - Performance considerations

- **Justify questions** (5 per module):
  - Deep conceptual understanding
  - Mathematical derivations
  - Design trade-offs
  - Real-world applications
  - Model answers with required concepts

- **Validator** (executable shell script):
  - Runs pytest for specific module
  - Measures performance baseline
  - Shadow worktree execution

- **Pedagogical bug** (patch file):
  - Tests critical concept
  - Common student mistake
  - Diagnostic for understanding

## Coverage Summary

### PDF Requirements: 100% ✅

- ✅ §2 Tokenization (2 modules)
- ✅ §3 Architecture (10 modules)
- ✅ §4 Optimization (4 modules)
- ✅ §5 Training (3 modules)
- ✅ §6 Generation (1 module)
- ✅ Integration (1 module)

### From-Scratch Compliance: 100% ✅

- ✅ No `nn.Linear` usage
- ✅ No `nn.Embedding` usage
- ✅ All components use `nn.Parameter`
- ✅ Manual initialization
- ✅ Explicit forward passes

### Student Starting State: Clean ✅

- ✅ 18 components properly stubbed
- ✅ All raise `NotImplementedError`
- ✅ Clear TODO messages with module references
- ✅ Developer mode has complete implementations

## Curriculum Philosophy

**"Build to Understand"**

The curriculum rejects the "black box" approach common in modern deep learning education. Students don't just *use* PyTorch layers - they *build* them.

**Learning Outcomes:**

After completing all 21 modules, students can:
1. Explain every line of a Transformer LM
2. Debug training issues at any level (data, model, optimizer)
3. Implement architecture variants from papers
4. Make informed decisions about model design
5. Optimize for memory and compute
6. Deploy production LLMs with confidence

**The Goal**: Deep, transferable understanding - not just "it works."

---

## Appendix: Module List (Complete)

1. `softmax` - Numerically stable softmax
2. `cross_entropy` - Numerically stable cross-entropy loss
3. `gradient_clipping` - Global gradient norm clipping
4. `linear` - Fully-connected layer (no nn.Linear)
5. `embedding` - Token embeddings (no nn.Embedding)
6. `silu` - SiLU activation
7. `rmsnorm` - RMS normalization
8. `swiglu` - SwiGLU gated FFN
9. `attention` - Scaled dot-product attention
10. `rope` - Rotary position embeddings
11. `multihead_attention` - Multi-head self-attention with RoPE
12. `transformer_block` - Complete transformer block
13. `transformer_lm` - Full model assembly
14. `adamw` - AdamW optimizer
15. `cosine_schedule` - Cosine LR schedule with warmup
16. `data_loader` - Batch sampling (get_batch)
17. `checkpointing` - Save/load with optimizer state
18. `training_loop` - Complete training integration
19. `bpe_tokenizer` - BPE training algorithm
20. `tokenizer_class` - Tokenizer wrapper
21. `text_generation` - Autoregressive generation

**Total: 21 modules, 18 stubbed components, 100% PDF coverage** 🎯
