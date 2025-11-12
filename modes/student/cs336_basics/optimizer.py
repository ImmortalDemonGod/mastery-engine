from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    """
    TODO: Implement AdamW optimizer (Decoupled Weight Decay Regularization).
    
    This optimizer combines:
    1. Adam's momentum and adaptive learning rates
    2. Decoupled weight decay (not L2 regularization!)
    
    See the 'adamw' curriculum module for detailed implementation guidance.
    
    References:
    - Loshchilov & Hutter, 2019: Decoupled Weight Decay Regularization
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (alpha)
        betas: (beta1, beta2) for first and second moment estimates
        eps: Small constant for numerical stability
        weight_decay: Weight decay coefficient (applied to parameters directly)
    
    Implementation requirements:
    - Maintain exponential moving averages (EMA) of gradients and squared gradients
    - Apply bias correction to moments
    - Use decoupled weight decay (subtract from parameters, not gradients)
    - Store state per-parameter (step count, m, v)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        # TODO: Validate hyperparameters and call super().__init__
        raise NotImplementedError("TODO: Implement AdamW.__init__ - validate params and initialize optimizer state")

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        """
        TODO: Perform a single optimization step.
        
        For each parameter with gradients:
        1. Initialize state if needed (step=0, m=0, v=0)
        2. Increment step counter
        3. Apply decoupled weight decay: p = p - lr * weight_decay * p
        4. Update biased first moment: m = beta1 * m + (1-beta1) * grad
        5. Update biased second moment: v = beta2 * v + (1-beta2) * grad²
        6. Compute bias-corrected moments: m_hat = m / (1 - beta1^t), v_hat = v / (1 - beta2^t)
        7. Update parameters: p = p - lr * m_hat / (sqrt(v_hat) + eps)
        
        Args:
            closure: Optional closure that reevaluates model and returns loss
        
        Returns:
            loss (if closure provided, else None)
        """
        raise NotImplementedError("TODO: Implement AdamW.step - perform optimization update")
