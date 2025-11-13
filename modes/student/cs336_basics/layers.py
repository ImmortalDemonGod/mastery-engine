import math

import torch
from torch import Tensor, nn
from jaxtyping import Float
from einops import rearrange
from cs336_basics.utils import softmax as _softmax


class Linear(nn.Module):
    """
    TODO: Implement a from-scratch Linear (fully-connected) layer: y = x @ W^T + b
    
    This is a foundational building block used by SwiGLU and other components.
    
    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        bias: If True, adds a learnable bias. Default: False
    
    Implementation requirements:
    - Weight parameter shape: (out_features, in_features)
    - Optional bias parameter shape: (out_features,)
    - Initialize parameters with uniform(-bound, bound) where bound = 1/sqrt(in_features)
    - Forward: y = x @ W^T + b (use matmul and transpose)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        # TODO: Initialize weight parameter and optional bias parameter
        raise NotImplementedError("TODO: Implement Linear.__init__")

    def forward(self, in_features: Float[Tensor, " ... in_features"]) -> Float[Tensor, " ... out_features"]:
        # TODO: Implement forward pass: y = x @ W^T + b
        raise NotImplementedError("TODO: Implement Linear.forward")


class Embedding(nn.Module):
    """
    TODO: Implement a from-scratch token embedding layer.
    
    See the 'embedding' curriculum module for detailed implementation guidance.
    
    This layer maps discrete token IDs to continuous vector representations.
    IMPORTANT: Do NOT use nn.Embedding - implement from scratch!
    
    Args:
        num_embeddings: Size of vocabulary (number of possible tokens)
        embedding_dim: Size of each embedding vector
    
    Implementation requirements:
    - Create weight parameter of shape (num_embeddings, embedding_dim)
    - Initialize with truncated normal: N(0,1) clipped to [-3, 3]
    - Forward: simple table lookup using weight[token_ids]
    """

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        # TODO: Create weight parameter and initialize it
        raise NotImplementedError("TODO: Implement Embedding.__init__")

    def forward(self, token_ids: Tensor) -> Float[Tensor, " ... embedding_dim"]:
        # TODO: Implement forward pass: table lookup
        raise NotImplementedError("TODO: Implement Embedding.forward")


def silu(in_features: Tensor) -> Tensor:
    """
    TODO: Implement SiLU (Swish) activation function.
    
    See the 'silu' curriculum module for detailed implementation guidance.
    
    SiLU(x) = x * sigmoid(x)
    
    This is also known as the Swish activation and is used in modern architectures.
    
    Args:
        in_features: Input tensor of any shape
    
    Returns:
        Output tensor of same shape as input
    """
    raise NotImplementedError("TODO: Implement silu activation: x * sigmoid(x)")


class RMSNorm(nn.Module):
    """
    TODO: Implement Root Mean Square Layer Normalization (RMSNorm).
    
    See the 'rmsnorm' curriculum module for detailed implementation guidance.
    
    RMSNorm normalizes using RMS instead of mean/variance:
    y = (x / RMS(x)) * weight
    where RMS(x) = sqrt(mean(x^2) + eps)
    
    Args:
        d_model: Model dimension (feature size)
        eps: Small constant for numerical stability (default: 1e-5)
    
    Implementation requirements:
    - Learnable weight parameter of shape (d_model,)
    - Compute RMS over last dimension with keepdim=True
    - Normalize: x / sqrt(mean(x^2) + eps)
    - Scale by weight: normalized * weight
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        # TODO: Initialize weight parameter
        raise NotImplementedError("TODO: Implement RMSNorm.__init__")

    def forward(self, in_features: Tensor) -> Tensor:
        # TODO: Implement forward pass: normalize by RMS and scale by weight
        raise NotImplementedError("TODO: Implement RMSNorm.forward")


class SwiGLU(nn.Module):
    """
    TODO: Implement SwiGLU feed-forward network.
    
    See the 'swiglu' curriculum module for detailed implementation guidance.
    
    SwiGLU applies gated activation:
      out = W2( SiLU(W1(x)) * W3(x) )
    
    This is the modern FFN used in LLaMA and other state-of-the-art models.
    
    Args:
        d_model: Model dimension (input/output size)
        d_ff: Feed-forward hidden dimension (typically 4×d_model)
    
    Implementation requirements:
    - Three Linear layers: W1, W2, W3 (all without bias)
    - W1 and W3 project from d_model to d_ff
    - W2 projects from d_ff back to d_model
    - Forward: Compute SiLU(W1(x)) * W3(x), then apply W2
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        # TODO: Initialize three Linear layers (W1, W3, W2)
        raise NotImplementedError("TODO: Implement SwiGLU.__init__")

    def forward(self, in_features: Tensor) -> Tensor:
        # TODO: Implement forward: W2(SiLU(W1(x)) * W3(x))
        raise NotImplementedError("TODO: Implement SwiGLU.forward")


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: torch.Tensor | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    TODO: Implement scaled dot-product attention.
    
    See the 'attention' curriculum module for detailed implementation guidance.
    
    This is the core attention mechanism:
      Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    
    Args:
        Q: Query tensor of shape (..., queries, d_k)
        K: Key tensor of shape (..., keys, d_k)
        V: Value tensor of shape (..., values, d_v), values == keys
        mask: Optional boolean mask (..., queries, keys). True=keep, False=mask
    
    Returns:
        Output tensor of shape (..., queries, d_v)
    
    Implementation requirements:
    - Compute scores: Q @ K^T  (transpose last two dims of K)
    - Scale by 1/sqrt(d_k) for numerical stability
    - Apply mask if provided: masked_fill with -inf where mask is False
    - Apply softmax over keys dimension (dim=-1)
    - Compute output: attention_weights @ V
    - Handle mixed precision: upcast to float32, return in V's dtype
    """
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

    Uses einops for clear, self-documenting tensor operations (per PDF §3.3).

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

    # Flatten leading batch dimensions: (..., s, d) -> (batch, s, d)
    x = rearrange(in_features, '... s d -> (...) s d')

    # Projections for all heads in single matmuls
    q_all = x.matmul(q_proj_weight.t())  # (B, S, d_model)
    k_all = x.matmul(k_proj_weight.t())  # (B, S, d_model)
    v_all = x.matmul(v_proj_weight.t())  # (B, S, d_model)

    # Split into heads and move head dimension before sequence: (B, H, S, D)
    def to_heads(t: Tensor) -> Tensor:
        """Split concatenated heads: (b, s, h*d) -> (b, h, s, d)"""
        return rearrange(t, 'b s (h d) -> b h s d', h=num_heads)

    q = to_heads(q_all)
    k = to_heads(k_all)
    v = to_heads(v_all)

    # Scaled dot-product attention per head with causal mask
    # Create lower-triangular causal mask and add broadcast dimensions
    causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device))
    causal = rearrange(causal, 's1 s2 -> 1 1 s1 s2')  # Add batch and head dims
    context = scaled_dot_product_attention(q, k, v, mask=causal)  # (B, H, S, D)

    # Combine heads: (B, H, S, D) -> (B, S, H*D=d_model)
    context = rearrange(context, 'b h s d -> b s (h d)')

    # Output projection
    out = context.matmul(o_proj_weight.t())  # (B, S, d_model)

    # Restore original leading dimensions
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
    TODO: Implement a pre-norm Transformer block with residual connections.
    
    See the 'transformer_block' curriculum module for detailed implementation guidance.
    
    Architecture (pre-norm with RoPE):
      x = x + MultiHeadSelfAttentionWithRoPE(RMSNorm(x))
      x = x + SwiGLU_FFN(RMSNorm(x))
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        max_seq_len: Maximum sequence length
        theta: RoPE base frequency
        weights: Dictionary containing layer weights:
            - "ln1.weight": First RMSNorm weights
            - "attn.q_proj.weight", "attn.k_proj.weight", etc.: Attention weights
            - "ln2.weight": Second RMSNorm weights
            - "ffn.w1.weight", "ffn.w2.weight", "ffn.w3.weight": FFN weights
        in_features: Input tensor (..., seq_len, d_model)
    
    Returns:
        Output tensor of same shape as input
    
    Implementation requirements:
    - Apply RMSNorm, then multi-head attention with RoPE, then residual add
    - Apply RMSNorm, then SwiGLU FFN, then residual add
    - Use weights from dictionary (load with torch.no_grad())
    - Generate position indices for RoPE
    """
    raise NotImplementedError("TODO: Implement transformer_block")


def rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Tensor,
    token_positions: Tensor,
) -> Tensor:
    """
    TODO: Implement Rotary Positional Embedding (RoPE).
    
    See the 'rope' curriculum module for detailed implementation guidance.
    
    RoPE encodes position by rotating pairs of dimensions in complex space.
    This enables relative position encoding without adding parameters.
    
    Args:
        d_k: Embedding dimension (must be even)
        theta: RoPE base frequency (typically 10000.0)
        max_seq_len: Maximum sequence length (for API compatibility)
        in_query_or_key: Tensor of shape (..., seq_len, d_k)
        token_positions: Position indices of shape (..., seq_len) or (seq_len,)
    
    Returns:
        Rotated tensor of same shape as input
    
    Implementation requirements:
    - Compute inverse frequencies: inv_freq_i = 1 / (theta^(i/d_k)) for i in [0, d_k/2)
    - Compute rotation angles: angles = positions * inv_freq
    - Compute cos and sin of angles
    - Split input into even/odd pairs: [x0, x1, x2, x3, ...] → [x0, x2, ...], [x1, x3, ...]
    - Apply rotation: x_even' = x_even*cos - x_odd*sin, x_odd' = x_even*sin + x_odd*cos
    - Interleave back: [x0', x1', x2', x3', ...]
    """
    raise NotImplementedError("TODO: Implement RoPE")


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
    TODO: Implement complete Transformer language model assembly.
    
    This is the final integration - assembling all components into a full LM.
    NOTE: A 'transformer_lm' curriculum module needs to be created for this.
    
    Pipeline:
      1. Token embedding lookup
      2. Stack of N transformer blocks
      3. Final RMSNorm
      4. LM head projection to vocabulary
    
    Args:
        vocab_size: Size of vocabulary
        context_length: Maximum sequence length
        d_model: Model dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads per block
        d_ff: Feed-forward hidden dimension
        rope_theta: RoPE base frequency
        weights: Dictionary containing all model weights
        in_indices: Input token IDs (..., seq_len)
    
    Returns:
        Logits tensor (..., seq_len, vocab_size)
    
    Implementation requirements:
    - Lookup embeddings: weights["token_embeddings.weight"][in_indices]
    - Apply num_layers transformer blocks with appropriate weight extraction
    - Apply final RMSNorm with weights["ln_final.weight"]
    - Project to vocab: x @ weights["lm_head.weight"]^T
    """
    raise NotImplementedError("TODO: Implement transformer_lm - full model assembly")


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
    TODO: Implement Multi-Head Self-Attention with RoPE.
    
    See the 'multihead_attention' curriculum module for detailed implementation guidance.
    
    This combines multiple attention heads with rotary positional embeddings.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads (must divide d_model)
        max_seq_len: Maximum sequence length
        theta: RoPE base frequency
        q_proj_weight: Query projection weight (d_model, d_model)
        k_proj_weight: Key projection weight (d_model, d_model)
        v_proj_weight: Value projection weight (d_model, d_model)
        o_proj_weight: Output projection weight (d_model, d_model)
        in_features: Input tensor (..., seq_len, d_model)
        token_positions: Position tensor (..., seq_len) or (seq_len,)
    
    Returns:
        Output tensor of same shape as in_features
    
    Implementation requirements:
    - Project input to Q, K, V using provided weights
    - Split into num_heads: reshape to (..., num_heads, seq_len, head_dim)
    - Apply RoPE to Q and K (per head)
    - Create causal mask (lower triangular)
    - Apply scaled_dot_product_attention with causal mask
    - Concatenate heads back: reshape to (..., seq_len, d_model)
    - Apply output projection
    """
    raise NotImplementedError("TODO: Implement multihead_self_attention_with_rope")
