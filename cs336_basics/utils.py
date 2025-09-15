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
