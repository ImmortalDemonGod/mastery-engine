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
