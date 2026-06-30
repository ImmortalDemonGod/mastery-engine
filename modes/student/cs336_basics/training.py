"""Language-model training loop (student stub).

Implement ``train_loop`` for the `training_loop` module: run the forward -> loss ->
backward -> grad-clip -> optimizer-step cycle over the dataloader and return loss
statistics. See the module's build prompt for the full specification.
"""
from __future__ import annotations

from torch import nn


def train_loop(
    model: nn.Module,
    train_dataloader,
    optimizer,
    num_epochs: int,
    max_grad_norm: float = 1.0,
    log_interval: int = 100,
    device: str = "cpu",
) -> dict:
    """TODO: Implement the training loop. See the training_loop module build prompt."""
    raise NotImplementedError("TODO: Implement train_loop function")
