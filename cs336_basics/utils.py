import torch
from jaxtyping import Float


def softmax(in_features: Float[torch.Tensor, " ..."], dim: int) -> Float[torch.Tensor, " ..."]:
    """
    Numerically-stable softmax over the specified dimension.

    Policy:
    - Upcast intermediates to float32 for stability.
    - Use log-sum-exp for strong numerical stability (handles very large/small logits).
    - Cast the final probabilities back to the original dtype of the input tensor.
    """
    x = in_features
    orig_dtype = x.dtype
    x32 = x.float()
    # logsumexp is stable even for extreme values and degenerate cases
    lse = torch.logsumexp(x32, dim=dim, keepdim=True)
    out = torch.exp(x32 - lse)
    return out.to(orig_dtype)
