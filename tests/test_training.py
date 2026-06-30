"""Tests for the language-model training loop (`cs336_basics.training.train_loop`).

Uses a tiny model with DETERMINISTIC weights (no RNG, so identical across torch
versions/platforms) trained to overfit a fixed, learnable batch (predict token+1).
A correct loop drives the loss down sharply; a broken one (e.g. missing zero_grad)
does not — fast, no real corpus/GPU.
"""
import torch
from torch import nn

from .adapters import run_train_loop

VOCAB = 8
D_MODEL = 16


class _TinyModel(nn.Module):
    def __init__(self, vocab: int = VOCAB, d_model: int = D_MODEL):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.head = nn.Linear(d_model, vocab)
        with torch.no_grad():
            # Deterministic, small, structured init (no RNG).
            self.emb.weight.copy_(
                torch.linspace(-0.5, 0.5, vocab * d_model).reshape(vocab, d_model)
            )
            self.head.weight.copy_(
                torch.linspace(-0.1, 0.1, vocab * d_model).reshape(vocab, d_model)
            )
            self.head.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T) -> (B, T, vocab)
        return self.head(self.emb(x))


def _fixed_batches():
    # A learnable pattern: predict the next token id (i+1) % vocab. Deterministic.
    x = (torch.arange(4 * 6) % VOCAB).reshape(4, 6)
    y = (x + 1) % VOCAB
    return [{"input_ids": x, "target_ids": y}]


def test_train_loop():
    model = _TinyModel()
    batches = _fixed_batches()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-2)

    stats = run_train_loop(
        model, batches, optimizer, num_epochs=80, log_interval=1, device="cpu"
    )

    losses = stats["losses"]
    assert losses, "train_loop returned no loss statistics"
    assert len(stats["steps"]) == len(losses)
    # A correct loop overfits the single fixed batch: loss must fall sharply.
    assert losses[-1] < 0.5 * losses[0], (
        f"loss did not decrease as expected: {losses[0]:.3f} -> {losses[-1]:.3f}"
    )
