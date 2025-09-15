import math

import torch
from torch import Tensor, nn
from jaxtyping import Float


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
    """SiLU activation: x * sigmoid(x)."""
    return in_features * torch.sigmoid(in_features)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization without bias.

    y = (x / sqrt(mean(x^2) + eps)) * weight
    where reduction is over the last dimension (feature dimension).
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.d_model))

    def forward(self, in_features: Tensor) -> Tensor:
        x = in_features
        # Compute rms over last dim
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        y = x / rms
        return y * self.weight


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward block:
      out = W2( SiLU(W1(x)) * W3(x) )
    Shapes:
      - W1: (d_ff, d_model)
      - W3: (d_ff, d_model)
      - W2: (d_model, d_ff)
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        # Use our minimal Linear layers without bias
        self.w1 = Linear(in_features=self.d_model, out_features=self.d_ff, bias=False)
        self.w2 = Linear(in_features=self.d_ff, out_features=self.d_model, bias=False)
        self.w3 = Linear(in_features=self.d_model, out_features=self.d_ff, bias=False)

    def forward(self, in_features: Tensor) -> Tensor:
        a = self.w1(in_features)
        b = self.w3(in_features)
        gated = silu(a) * b
        return self.w2(gated)
