"""Language-model training loop tying the components together.

Runs the standard forward -> loss -> backward -> clip -> step cycle over a
dataloader of ``{'input_ids', 'target_ids'}`` batches and returns per-step loss
statistics.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
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
    """Train ``model`` for ``num_epochs`` passes over ``train_dataloader``.

    Each batch is a mapping with ``input_ids`` and ``target_ids`` tensors of shape
    ``(batch, seq_len)``. Returns ``{'losses': [...], 'steps': [...]}`` sampled every
    ``log_interval`` steps.
    """
    model.to(device)
    model.train()

    step = 0
    losses: list[float] = []
    steps: list[int] = []

    for _epoch in range(num_epochs):
        for batch in train_dataloader:
            step += 1
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)

            logits = model(input_ids)  # (batch, seq_len, vocab)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            if step % log_interval == 0:
                losses.append(loss.item())
                steps.append(step)

    return {"losses": losses, "steps": steps}
