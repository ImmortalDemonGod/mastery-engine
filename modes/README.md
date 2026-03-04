# Modes Directory - Design Rationale

## Purpose

This directory contains the source of truth for **student mode** (learning path) and **developer mode** (reference implementations).

The CS336 assignment follows a **"from scratch" ethos**: students implement all critical components of a Transformer language model without using high-level PyTorch abstractions like `nn.Linear` or `nn.Embedding`.

## Complete Curriculum: 21 Modules

The curriculum teaches **21 modules** covering the full LM pipeline:

### Foundation Components (8 modules)
- `softmax` - Numerically stable softmax
- `cross_entropy` - Numerically stable cross-entropy loss  
- `gradient_clipping` - Global gradient clipping
- `linear` - Fully-connected layer (from scratch, no `nn.Linear`)
- `embedding` - Token embeddings (from scratch, no `nn.Embedding`)
- `silu` - SiLU activation function
- `rmsnorm` - Root Mean Square Layer Normalization
- `rope` - Rotary Position Embeddings

### Architecture Components (5 modules)
- `swiglu` - SwiGLU gated feed-forward network
- `attention` - Scaled dot-product attention
- `multihead_attention` - Multi-head self-attention with RoPE
- `transformer_block` - Complete transformer block
- `transformer_lm` - Full model assembly

### Training Infrastructure (5 modules)
- `adamw` - AdamW optimizer with bias correction
- `cosine_schedule` - Cosine learning rate schedule with warmup
- `data_loader` - Batch sampling for training (`get_batch`)
- `checkpointing` - Model save/load with optimizer state
- `training_loop` - Complete training integration

### Application (3 modules)
- `bpe_tokenizer` - Byte Pair Encoding training
- `tokenizer_class` - Tokenizer wrapper with encode/decode
- `text_generation` - Autoregressive generation with sampling

## Student Mode vs Developer Mode

### Student Mode (`modes/student/cs336_basics/`)

**18 Components Stubbed** (must implement):

`layers.py`:
- `Linear` class
- `Embedding` class  
- `silu()` function
- `RMSNorm` class
- `SwiGLU` class
- `scaled_dot_product_attention()` function
- `rope()` function
- `multihead_self_attention_with_rope()` function
- `transformer_block()` function
- `transformer_lm()` function

`utils.py`:
- `softmax()` function
- `cross_entropy()` function
- `gradient_clipping()` function
- `get_lr_cosine_schedule()` function
- `get_batch()` function
- `save_checkpoint()` function
- `load_checkpoint()` function

`optimizer.py`:
- `AdamW` class

`bpe.py`:
- `train_bpe()` function

`generation.py`:
- `generate()` function

**Helper Functions** (provided complete):
- `multihead_self_attention()` - Non-RoPE variant (not in curriculum)
- Utility helpers in `pretokenization_example.py`

### Developer Mode (`modes/developer/cs336_basics/`)

**All components fully implemented** - serves as reference and enables engine development.

## Why "From Scratch"?

The curriculum philosophy: **No magic black boxes.**

❌ **Not allowed:**
```python
self.linear = nn.Linear(in_features, out_features)  # Too high-level!
self.embed = nn.Embedding(vocab_size, d_model)     # Hides the mechanism!
```

✅ **Required:**
```python
# Implement using nn.Parameter directly
self.weight = nn.Parameter(torch.empty(out_features, in_features))
nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

def forward(self, x):
    return x.matmul(self.weight.t()) + self.bias  # y = x @ W^T + b
```

**Why this matters:**
- Understand weight initialization (Kaiming uniform, truncated normal)
- Understand matrix shapes and transposes
- Understand parameter counting and memory usage
- Build intuition for backpropagation and gradients
- No "automagic" - every line has clear meaning

## What Students Build

After completing all 21 modules, students have:

✅ **Complete Transformer LM from scratch**
- No `nn.Linear` or `nn.Embedding` used
- Every component hand-built with `nn.Parameter`
- Deep understanding of architecture internals

✅ **Full training pipeline**
- Data loading with random sampling
- AdamW optimizer with momentum
- Cosine schedule with warmup
- Checkpointing for resilience

✅ **Production-ready tokenizer**
- BPE training from corpus
- Encode/decode with special tokens
- UTF-8 byte-level for multilingual

✅ **Text generation**
- Autoregressive sampling
- Temperature, top-k, top-p strategies
- Real LLM interaction

## File Structure

```
modes/
├── student/              # Learning mode - stubs for curriculum
│   └── cs336_basics/
│       ├── layers.py     # 10 components stubbed
│       ├── utils.py      # 7 functions stubbed
│       ├── optimizer.py  # AdamW stubbed
│       ├── bpe.py        # train_bpe stubbed
│       ├── generation.py # generate stubbed
│       ├── tokenizer.py  # Complete (helper)
│       └── __init__.py
│
└── developer/            # Reference mode - all complete
    └── cs336_basics/
        └── [all files complete]
```

## Mode Switching

Use the mode script:

```bash
./scripts/mode switch student    # Activate learning mode
./scripts/mode switch developer  # Activate reference mode
```

**Student mode:**
- 18 components raise `NotImplementedError`
- Tests fail until implementations complete
- Forces learning through implementation

**Developer mode:**  
- All components fully implemented
- Tests pass immediately
- Enables engine development without waiting for curriculum completion

## Verification

Check alignment between modes:

```bash
# Student mode should have stubs
grep -n "NotImplementedError" modes/student/cs336_basics/*.py

# Developer mode should be complete (no stubs)
grep -n "NotImplementedError" modes/developer/cs336_basics/*.py
# Should return nothing

# Files in both modes
ls modes/student/cs336_basics/
ls modes/developer/cs336_basics/
# Should have identical file lists
```

## Curriculum Expansion

To add new modules:

1. **Update manifest**: Add to `curricula/cs336_a1/manifest.json`
2. **Create module content**:
   - `curricula/cs336_a1/modules/{module_id}/build_prompt.txt`
   - `curricula/cs336_a1/modules/{module_id}/justify_questions.json`
   - `curricula/cs336_a1/modules/{module_id}/validator.sh`
   - `curricula/cs336_a1/modules/{module_id}/bugs/*.patch`
3. **Update student mode**: Add stub to appropriate file
4. **Update developer mode**: Add complete implementation

## Design Philosophy

**Minimal scaffolding, maximum learning:**
- Students implement what matters (architecture, training, tokenization)
- No busywork (pretokenization scripts provided)
- Progressive complexity (foundation → components → integration)
- Real understanding through implementation

**The goal:** Students who complete this curriculum can confidently say "I built a Transformer language model from scratch" and mean it literally.
