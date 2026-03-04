# CS336 Assignment 1: Curriculum Gap Analysis

## Current Mastery Engine Coverage

**Modules Implemented: 3 / ~19 components (16% coverage)**

### ✅ Currently Taught by Mastery Engine
1. **softmax** - Numerically stable softmax (§4.2)
2. **cross_entropy** - Numerically stable cross-entropy loss (§4.1)  
3. **gradient_clipping** - Global gradient clipping by L2 norm (§4.5)

## Full CS336 Assignment 1 Requirements

### Section 2: BPE Tokenizer (NOT IN CURRICULUM)
**Training:**
- `train_bpe()` - Train byte-level BPE tokenizer with:
  - Pre-tokenization with regex
  - Iterative byte-pair merging
  - Special token handling
  - Vocabulary and merge list output

**Tokenizer Class:**
- `encode()` - Text → token IDs
- `decode()` - Token IDs → text
- `encode_iterable()` - Memory-efficient streaming encoding
- Special token support
- Handle Unicode replacement character for invalid sequences

**Points:** 15 (train_bpe) + 15 (tokenizer) = 30 points

---

### Section 3: Transformer Architecture (NOT IN CURRICULUM)

**3.4 Basic Building Blocks:**
1. **Linear** (1 point) - Linear transformation y = Wx
   - Custom nn.Module implementation
   - Truncated normal initialization
   - No bias term

2. **Embedding** (1 point) - Token ID → dense vectors
   - Vocabulary → d_model embedding lookup
   - Truncated normal initialization

**3.5 Normalization:**
3. **RMSNorm** (1 point) - Root Mean Square Layer Normalization
   - Per-position normalization
   - Learnable scale parameter
   - Used in pre-norm Transformer blocks

**3.6 Feed-Forward Networks:**
4. **SiLU** (1 point) - Sigmoid Linear Unit activation
   - σ(x) · x activation function

5. **SwiGLU** (2 points) - Gated Linear Unit with SiLU
   - W₂(SiLU(W₁x) ⊙ W₃x)
   - Three weight matrices
   - Dimension: d_model → d_ff → d_model

**3.7 Attention:**
6. **scaled_dot_product_attention** (4 points)
   - QK^T / √d_k attention scores
   - Causal masking for autoregressive generation
   - Softmax over keys
   - Weighted sum over values

7. **RoPE** (3 points) - Rotary Position Embeddings
   - Apply rotation matrices to queries and keys
   - Position-aware without absolute position embeddings
   - Frequency-based position encoding (θ parameter)

8. **multihead_self_attention** (4 points)
   - Batched multi-head computation
   - Q, K, V projections per head
   - Scaled dot-product attention per head
   - Output projection

9. **multihead_self_attention_with_rope** (2 points)
   - Multi-head attention + RoPE integration
   - Apply RoPE to Q and K before attention

**3.8 Transformer Block:**
10. **transformer_block** (3 points) - Pre-norm Transformer block
    - x + MultiHeadSelfAttention(RMSNorm(x))
    - z + SwiGLU(RMSNorm(z))
    - Residual connections

**3.9 Full Model:**
11. **transformer_lm** (4 points) - Complete Transformer language model
    - Token embedding
    - num_layers Transformer blocks
    - Final RMSNorm
    - Output linear projection (LM head)

**Architecture Points:** 1+1+1+1+2+4+3+4+2+3+4 = 26 points

---

### Section 4: Loss, Optimizer, Scheduler (PARTIAL)

12. **softmax** (DONE) ✅ - Already in curriculum
13. **cross_entropy** (DONE) ✅ - Already in curriculum

14. **AdamW** (6 points) - Optimizer with decoupled weight decay
    - Momentum estimates (β₁, β₂)
    - Bias correction
    - Decoupled weight decay
    - Learning rate scheduling integration

15. **get_lr_cosine_schedule** (1 point) - Cosine annealing with warmup
    - Linear warmup phase
    - Cosine decay phase
    - Constant minimum learning rate

16. **gradient_clipping** (DONE) ✅ - Already in curriculum

**Loss/Optimizer Points:** 6 (AdamW) + 1 (scheduler) = 7 points

---

### Section 5: Training Loop (NOT IN CURRICULUM)

17. **get_batch** (2 points) - Data loading
    - Sample random sequences from dataset
    - Create input/target pairs
    - Memory-mapped file support (np.memmap)
    - Device placement (CPU/CUDA/MPS)

18. **save_checkpoint** (1 point) - Serialize training state
    - Model state_dict
    - Optimizer state_dict
    - Iteration counter
    - torch.save to file

19. **load_checkpoint** (1 point) - Restore training state
    - Load serialized checkpoint
    - Restore model and optimizer
    - Return iteration number

20. **Training Loop Script** (4 points)
    - Hyperparameter configuration
    - Training and validation loops
    - Periodic checkpointing
    - Logging (console / W&B)

**Training Points:** 2 + 1 + 1 + 4 = 8 points

---

### Section 6: Text Generation (NOT IN CURRICULUM)

21. **Decoder** (3 points) - Text generation from LM
    - Temperature scaling
    - Top-p (nucleus) sampling
    - Stop on 
