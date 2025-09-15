import torch
from jaxtyping import Float


def softmax(in_features: Float[torch.Tensor, " ..."], dim: int) -> Float[torch.Tensor, " ..."]:
    """
    Numerically-stable softmax over the specified dimension using the subtract-max trick.

    Policy:
    - Upcast intermediates to float32 for stability.
    - Subtract the max along `dim` before exponentiation to avoid overflow.
    - Cast the final probabilities back to the original dtype of the input tensor.
    """
    x = in_features
    orig_dtype = x.dtype
    x32 = x.float()
    max_vals = x32.max(dim=dim, keepdim=True).values
    shifted = x32 - max_vals
    exps = torch.exp(shifted)
    sums = exps.sum(dim=dim, keepdim=True)
    out = exps / sums
    return out.to(orig_dtype)


def cross_entropy(
    inputs: Float[torch.Tensor, " batch_size vocab_size"], targets: torch.Tensor
) -> Float[torch.Tensor, ""]:
    """
    Numerically-stable cross-entropy loss averaged over the batch.

    Args:
        inputs: Unnormalized logits of shape (batch_size, vocab_size).
        targets: Class indices tensor of shape (batch_size,), int dtype.

    Returns:
        Scalar tensor: mean cross-entropy over the batch.
    """
    logits = inputs
    orig_dtype = logits.dtype
    x32 = logits.float()
    t = targets.long()
    # log-sum-exp for stability
    lse = torch.logsumexp(x32, dim=-1)
    # pick the logit for the correct class
    correct = x32.gather(dim=-1, index=t.unsqueeze(-1)).squeeze(-1)
    loss = (lse - correct).mean()
    return loss.to(orig_dtype)


def gradient_clipping(parameters, max_l2_norm: float) -> None:
    """
    Clip gradients in-place so that the global L2 norm across all parameter gradients
    does not exceed `max_l2_norm`.

    Semantics mirror torch.nn.utils.clip_grad.clip_grad_norm_ (eps=1e-6).

    Args:
        parameters: Iterable of torch.nn.Parameter (or objects with .grad tensors).
        max_l2_norm: Maximum allowed global L2 norm.
    """
    # Collect grads that exist
    grads = [p.grad for p in parameters if getattr(p, "grad", None) is not None]
    if not grads:
        return
    # Compute global L2 norm (same as norm of concatenation)
    norms = torch.stack([g.detach().norm(2) for g in grads])
    total_norm = norms.norm(2)
    # Only scale if norm exceeds the threshold
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + 1e-6)
        for g in grads:
            g.mul_(clip_coef)


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Cosine learning rate with linear warmup.

    Behavior matches tests:
    - Linear warmup from 0 to max_learning_rate over [0, warmup_iters).
    - Cosine decay from max to min over the interval [warmup_iters, cosine_cycle_iters].
      Progress is computed as (it - warmup_iters) / (cosine_cycle_iters - warmup_iters) and
      clamped to [0, 1]. The cosine formula is: min + 0.5*(max-min)*(1 + cos(pi * progress)).
    - For it > cosine_cycle_iters, hold at min_learning_rate.
    """
    # Warmup
    if it < warmup_iters:
        if warmup_iters <= 0:
            return float(max_learning_rate)
        return float(max_learning_rate * (it / warmup_iters))

    # Cosine decay window
    if it <= cosine_cycle_iters:
        # Handle degenerate case where window length is zero
        denom = max(1, cosine_cycle_iters - warmup_iters)
        progress = (it - warmup_iters) / denom
        from math import cos, pi

        return float(
            min_learning_rate
            + 0.5 * (max_learning_rate - min_learning_rate) * (1.0 + cos(pi * progress))
        )

    # After cycle: clamp to min
    return float(min_learning_rate)
