import math

import torch
from torch import Tensor, nn
from jaxtyping import Float
from cs336_basics.utils import softmax as _softmax


class Linear(nn.Module):
    """
    Minimal from-scratch Linear layer implementing y = x @ W^T (+ b).

    - Weight shape: (out_features, in_features)
    - Bias is optional and disabled by default for parity with tests that only provide weights.
    - Parameter initialization follows torch.nn.Linear: uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty((out_features,)))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        fan_in = max(1, self.in_features)
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, in_features: Float[Tensor, " ... in_features"]) -> Float[Tensor, " ... out_features"]:
        y = in_features.matmul(self.weight.t())
        if self.bias is not None:
            y = y + self.bias
        return y


class Embedding(nn.Module):
    """
    Simple token embedding layer.

    - Weight shape: (num_embeddings, embedding_dim)
    - Forward performs table lookup: weight[token_ids]
    - Initialization: truncated normal N(0, 1) clipped to [-3, 3]
    """

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)

        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Truncated normal with mean=0, std=1, a=-3, b=3
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Float[Tensor, " ... embedding_dim"]:
        return self.weight[token_ids]


def silu(in_features: Tensor) -> Tensor:
    """
    Apply SiLU (Sigmoid Linear Unit) activation function element-wise.
    
    SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    
    Also known as Swish activation.
    
    Args:
        in_features: Input tensor of any shape
    
    Returns:
        Tensor of same shape with SiLU applied element-wise
    """
    # TODO: Implement SiLU activation
    # SiLU(x) = x * σ(x) where σ is sigmoid
    # Hint: Use torch.sigmoid() - it's numerically stable!
    # This is a one-liner!
    raise NotImplementedError("TODO: Implement silu activation")


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization without bias.

    y = (x / sqrt(mean(x^2) + eps)) * weight
    where reduction is over the last dimension (feature dimension).
    """

    def __init__(self, d_model: int, eps: float = 1e-6, device=None, dtype=None) -> None:
        super().__init__()
        # TODO: Initialize RMSNorm parameters
        # - Store d_model and eps
        # - Create learnable scale parameter (weight) of shape (d_model,)
        # - Initialize weight to ones using torch.nn.init.ones_()
        raise NotImplementedError("TODO: Implement RMSNorm.__init__")

    def forward(self, in_features: Tensor) -> Tensor:
        # TODO: Implement RMSNorm forward pass
        # 1. Save original dtype and upcast to float32 for numerical stability
        # 2. Compute RMS over last dimension (use keepdim=True!)
        #    RMS(x) = sqrt(mean(x^2))
        # 3. Normalize: x / (RMS(x) + eps)
        # 4. Scale by learned weight parameter
        # 5. Convert back to original dtype
        #
        # Remember: Your implementation must handle arbitrary batch dimensions!
        # Shape (..., d_model) → (..., d_model)
        raise NotImplementedError("TODO: Implement RMSNorm.forward")


class SwiGLU(nn.Module):
    """
    SwiGLU gated feed-forward network.
    
    Implements: SwiGLU(x) = W₂(SiLU(W₁·x) ⊙ W₃·x)
    
    Where ⊙ denotes element-wise multiplication (Hadamard product).
    This creates a gating mechanism where W₃ learns what to emphasize.
    
    Shapes:
      - W₁: (d_ff, d_model) - value projection  
      - W₃: (d_ff, d_model) - gate projection
      - W₂: (d_model, d_ff) - output projection
    """

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None) -> None:
        super().__init__()
        # TODO: Create three Linear layers for SwiGLU
        # Use your custom Linear class (not nn.Linear)
        # 
        # self.w1: projects x from d_model → d_ff (value path)
        # self.w2: projects from d_ff → d_model (output)
        # self.w3: projects x from d_model → d_ff (gate path)
        #
        # IMPORTANT: Name them exactly w1, w2, w3 (lowercase) for validator!
        # Pass device and dtype to each Linear constructor
        raise NotImplementedError("TODO: Implement SwiGLU.__init__")

    def forward(self, x: Tensor) -> Tensor:
        # TODO: Implement SwiGLU forward pass
        # 1. Compute value path: w1_out = self.w1(x)
        # 2. Apply SiLU activation: activated = silu(w1_out)
        # 3. Compute gate path: gate = self.w3(x)  
        # 4. Element-wise multiply: gated = activated * gate
        # 5. Project to output: output = self.w2(gated)
        #
        # Formula: W₂(SiLU(W₁·x) ⊙ W₃·x)
        # Shape: (..., d_model) → (..., d_model)
        raise NotImplementedError("TODO: Implement SwiGLU.forward")


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Compute scaled dot-product attention.
    
    Attention(Q, K, V) = softmax(QK^T / √d_k) · V
    
    This is the core mechanism that enables Transformers to dynamically route
    information. Each query position computes a weighted average over all value
    positions, with weights determined by query-key similarity.

    Args:
        Q: Query tensor of shape (..., n_queries, d_k)
        K: Key tensor of shape (..., n_keys, d_k)
        V: Value tensor of shape (..., n_keys, d_v)
        mask: Optional boolean mask of shape (..., n_queries, n_keys)
              True indicates positions that should be MASKED (set to -inf)
              Used for causal masking in autoregressive generation

    Returns:
        Output tensor of shape (..., n_queries, d_v)
    """
    # TODO: Implement scaled dot-product attention
    # 
    # Step 1: Compute attention scores
    #   scores = Q @ K^T  (use K.transpose(-2, -1) for any rank)
    #   Shape: (..., n_queries, d_k) @ (..., d_k, n_keys) = (..., n_queries, n_keys)
    #
    # Step 2: Scale by √d_k for stability
    #   d_k = Q.shape[-1]
    #   scaled_scores = scores / math.sqrt(d_k)
    #   This prevents dot products from growing with dimensionality!
    #
    # Step 3: Apply causal mask if provided
    #   if mask is not None:
    #       scaled_scores = scaled_scores.masked_fill(mask, float('-inf'))
    #   Masked positions get -inf so softmax makes them 0
    #
    # Step 4: Apply softmax over keys (dim=-1)
    #   attn_weights = softmax(scaled_scores, dim=-1)
    #   Use YOUR custom softmax from utils!
    #   Each query gets a probability distribution over keys
    #
    # Step 5: Weighted sum of values
    #   output = attn_weights @ V
    #   Shape: (..., n_queries, n_keys) @ (..., n_keys, d_v) = (..., n_queries, d_v)
    #
    # Remember: Use @ for matrix multiply, it handles batch dimensions!
    raise NotImplementedError("TODO: Implement scaled_dot_product_attention")


def multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
) -> Tensor:
    """
    Optimized batched Multi-Head Self-Attention (no RoPE).

    Args:
        d_model: embedding dimension
        num_heads: number of attention heads (divides d_model)
        *_proj_weight: concatenated projection weights for all heads, shape (d_model, d_model)
        in_features: tensor of shape (..., seq_len, d_model)

    Returns:
        Tensor of shape (..., seq_len, d_model)
    """
    head_dim = d_model // num_heads
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    orig_leading = in_features.shape[:-2]
    seq_len = in_features.shape[-2]

    x = in_features.reshape(-1, seq_len, d_model)

    # Projections for all heads in single matmuls
    q_all = x.matmul(q_proj_weight.t())  # (B, S, d_model)
    k_all = x.matmul(k_proj_weight.t())  # (B, S, d_model)
    v_all = x.matmul(v_proj_weight.t())  # (B, S, d_model)

    # Reshape to heads: (B, H, S, D)
    def to_heads(t: Tensor) -> Tensor:
        # Weights are stacked by heads along the output rows: (H*D, d_model).
        # After projecting to (B, S, H*D), split into heads then move heads before seq.
        return t.view(t.shape[0], seq_len, num_heads, head_dim).transpose(1, 2).contiguous()

    q = to_heads(q_all)
    k = to_heads(k_all)
    v = to_heads(v_all)

    # Scaled dot-product attention per head with causal mask
    causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device)).view(1, 1, seq_len, seq_len)
    context = scaled_dot_product_attention(q, k, v, mask=causal)  # (B, H, S, D)

    # Combine heads: (B, S, H*D=d_model)
    context = context.transpose(1, 2).contiguous().view(context.shape[0], seq_len, d_model)

    # Output projection
    out = context.matmul(o_proj_weight.t())  # (B, S, d_model)

    return out.reshape(*orig_leading, seq_len, d_model)


def transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict,
    in_features: Tensor,
) -> Tensor:
    """
    Pre-norm Transformer block with RoPE:
      x = x + MHA(RMSNorm(x))
      x = x + FFN(RMSNorm(x))
    """
    x = in_features
    seq_len = x.shape[-2]
    device = x.device

    # First RMSNorm
    ln1 = RMSNorm(d_model=d_model, eps=1e-5)
    with torch.no_grad():
        ln1.weight.copy_(weights["ln1.weight"])  # (d_model,)
    x_norm = ln1(x)

    # Positions for RoPE
    pos = torch.arange(seq_len, device=device).view(1, -1)

    # Multi-head self-attention with RoPE
    attn_out = multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        q_proj_weight=weights["attn.q_proj.weight"],
        k_proj_weight=weights["attn.k_proj.weight"],
        v_proj_weight=weights["attn.v_proj.weight"],
        o_proj_weight=weights["attn.output_proj.weight"],
        in_features=x_norm,
        token_positions=pos,
    )
    x = x + attn_out

    # Second RMSNorm
    ln2 = RMSNorm(d_model=d_model, eps=1e-5)
    with torch.no_grad():
        ln2.weight.copy_(weights["ln2.weight"])  # (d_model,)
    x_norm2 = ln2(x)

    # SwiGLU FFN using provided weights
    W1 = weights["ffn.w1.weight"]  # (d_ff, d_model)
    W2 = weights["ffn.w2.weight"]  # (d_model, d_ff)
    W3 = weights["ffn.w3.weight"]  # (d_ff, d_model)

    a = x_norm2.matmul(W1.t())
    b = x_norm2.matmul(W3.t())
    gated = silu(a) * b
    ffn_out = gated.matmul(W2.t())

    x = x + ffn_out
    return x


def rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Apply Rotary Position Embeddings (RoPE) to queries or keys.
    
    RoPE encodes position by rotating pairs of dimensions in complex space.
    Each dimension pair rotates at a different frequency, creating multi-scale
    position encoding that naturally captures relative position through the
    rotation property: Rotation(m) ⊗ Rotation(n)* = Rotation(m-n).

    Args:
        d_k: Dimension of queries/keys (must be even for pairing)
        theta: Base angle θ (typically 10000.0, like original Transformer)
        max_seq_len: Maximum sequence length to support
        in_query_or_key: Input tensor of shape (..., seq_len, d_k)
        token_positions: Position indices of shape (..., seq_len)
                        Values in range [0, max_seq_len)

    Returns:
        Rotated tensor of same shape as input
    """
    # TODO: Implement RoPE (Rotary Position Embeddings)
    #
    # Step 1: Compute frequencies for each dimension pair
    #   dim_pairs = torch.arange(0, d_k, 2, dtype=torch.float32, device=x.device)
    #   freqs = 1.0 / (theta ** (dim_pairs / d_k))
    #   Shape: (d_k/2,)
    #
    # Step 2: Create angles for all positions
    #   positions = torch.arange(max_seq_len, dtype=torch.float32, device=x.device)
    #   angles = torch.outer(positions, freqs)  # Outer product!
    #   Shape: (max_seq_len, d_k/2)
    #
    # Step 3: Precompute cos and sin
    #   cos_angles = torch.cos(angles)
    #   sin_angles = torch.sin(angles)
    #
    # Step 4: Select angles for actual token positions
    #   cos_selected = cos_angles[token_positions]  # Index with positions!
    #   sin_selected = sin_angles[token_positions]
    #   Shape: (..., seq_len, d_k/2)
    #
    # Step 5: Reshape input to pair dimensions
    #   x_pairs = x.reshape(*x.shape[:-1], -1, 2)
    #   x_even = x_pairs[..., 0]  # First of each pair
    #   x_odd = x_pairs[..., 1]   # Second of each pair
    #   Shape: (..., seq_len, d_k/2)
    #
    # Step 6: Apply 2D rotation formula
    #   x_rotated_even = x_even * cos_selected - x_odd * sin_selected
    #   x_rotated_odd = x_even * sin_selected + x_odd * cos_selected
    #   CRITICAL: Note the minus sign in first equation!
    #
    # Step 7: Stack and reshape back
    #   x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
    #   output = x_rotated.reshape(*x.shape)
    #   Shape: (..., seq_len, d_k)
    #
    # Hints:
    # - torch.outer(a, b) computes outer product (all pairs of products)
    # - Indexing: cos_angles[token_positions] handles arbitrary positions
    # - Reshape: reshape(..., -1, 2) pairs up dimensions automatically
    # - Stack: stack([even, odd], dim=-1) interleaves them back
    #
    # This is the most mathematically elegant position encoding!
    raise NotImplementedError("TODO: Implement rope")


def transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict,
    in_indices: Tensor,
) -> Tensor:
    """
    Transformer language model forward pass using provided state dict weights.

    Pipeline:
      - Token embedding lookup
      - N x [pre-norm block with RoPE-attn + SwiGLU FFN]
      - Final RMSNorm
      - LM head projection to vocab size
    """
    # Embedding lookup
    token_emb = weights["token_embeddings.weight"]  # (vocab_size, d_model)
    x = token_emb[in_indices]  # (B, S, d_model)

    # Apply Transformer blocks
    for i in range(int(num_layers)):
        block_weights = {
            "ln1.weight": weights[f"layers.{i}.ln1.weight"],
            "attn.q_proj.weight": weights[f"layers.{i}.attn.q_proj.weight"],
            "attn.k_proj.weight": weights[f"layers.{i}.attn.k_proj.weight"],
            "attn.v_proj.weight": weights[f"layers.{i}.attn.v_proj.weight"],
            "attn.output_proj.weight": weights[f"layers.{i}.attn.output_proj.weight"],
            "ln2.weight": weights[f"layers.{i}.ln2.weight"],
            "ffn.w1.weight": weights[f"layers.{i}.ffn.w1.weight"],
            "ffn.w2.weight": weights[f"layers.{i}.ffn.w2.weight"],
            "ffn.w3.weight": weights[f"layers.{i}.ffn.w3.weight"],
        }
        x = transformer_block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=rope_theta,
            weights=block_weights,
            in_features=x,
        )

    # Final RMSNorm
    ln_final = RMSNorm(d_model=d_model, eps=1e-5)
    with torch.no_grad():
        ln_final.weight.copy_(weights["ln_final.weight"])  # (d_model,)
    x = ln_final(x)

    # LM head projection
    lm_head_w = weights["lm_head.weight"]  # (vocab_size, d_model)
    logits = x.matmul(lm_head_w.t())  # (B, S, vocab_size)
    return logits


def multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
    token_positions: Tensor,
) -> Tensor:
    """
    Batched Multi-Head Self-Attention with RoPE (causal, no dropout).
    """
    head_dim = d_model // num_heads
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    orig_leading = in_features.shape[:-2]
    seq_len = in_features.shape[-2]

    x = in_features.reshape(-1, seq_len, d_model)

    # Projections
    q_all = x.matmul(q_proj_weight.t())
    k_all = x.matmul(k_proj_weight.t())
    v_all = x.matmul(v_proj_weight.t())

    def to_heads(t: Tensor) -> Tensor:
        return t.view(t.shape[0], seq_len, num_heads, head_dim).transpose(1, 2).contiguous()

    q = to_heads(q_all)
    k = to_heads(k_all)
    v = to_heads(v_all)

    # Apply RoPE to Q and K per head
    q = rope(d_k=head_dim, theta=theta, max_seq_len=max_seq_len, in_query_or_key=q, token_positions=token_positions)
    k = rope(d_k=head_dim, theta=theta, max_seq_len=max_seq_len, in_query_or_key=k, token_positions=token_positions)

    # Causal mask and attention
    causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device)).view(1, 1, seq_len, seq_len)
    context = scaled_dot_product_attention(q, k, v, mask=causal)

    # Merge heads
    context = context.transpose(1, 2).contiguous().view(context.shape[0], seq_len, d_model)
    out = context.matmul(o_proj_weight.t())
    return out.reshape(*orig_leading, seq_len, d_model)
