"""Tests for the language-model training loop (`cs336_basics.training.train_loop`).

Uses a tiny model that overfits a single fixed batch, so a correct loop drives the
loss down monotonically-ish — no real corpus, GPU, or long training required.
"""
import torch
from torch import nn

from .adapters import run_train_loop


class _TinyModel(nn.Module):
    def __init__(self, vocab: int = 8, d_model: int = 16, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.emb = nn.Embedding(vocab, d_model)
        self.head = nn.Linear(d_model, vocab)
        with torch.no_grad():
            self.emb.weight.copy_(torch.randn(vocab, d_model, generator=gen))
            self.head.weight.copy_(torch.randn(vocab, d_model, generator=gen) * 0.1)
            self.head.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T) -> (B, T, vocab)
        return self.head(self.emb(x))


def _fixed_batches(vocab: int, batch: int, seq: int, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randint(0, vocab, (batch, seq), generator=gen)
    y = torch.randint(0, vocab, (batch, seq), generator=gen)
    return [{"input_ids": x, "target_ids": y}]


def test_train_loop():
    torch.manual_seed(0)
    vocab = 8
    model = _TinyModel(vocab=vocab)
    batches = _fixed_batches(vocab, batch=4, seq=6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    stats = run_train_loop(
        model, batches, optimizer, num_epochs=60, log_interval=1, device="cpu"
    )

    losses = stats["losses"]
    assert losses, "train_loop returned no loss statistics"
    assert len(stats["steps"]) == len(losses)
    # A correct loop overfits the single fixed batch: loss must fall substantially.
    assert losses[-1] < 0.5 * losses[0], (
        f"loss did not decrease as expected: {losses[0]:.3f} -> {losses[-1]:.3f}"
    )
