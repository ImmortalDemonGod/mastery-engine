"""Tests for autoregressive text generation (`cs336_basics.generation.generate`).

Uses a tiny LM whose weights are constructed deterministically (no RNG, so behaviour
is identical across torch versions/platforms): token ``i`` deterministically predicts
token ``(i+1) % vocab``. Greedy decoding (``top_k=1``) therefore walks a known,
non-degenerate sequence — exactly checkable, no trained model/GPU/snapshots needed.
"""
import torch
from torch import nn

from .adapters import run_generate


class _CounterLM(nn.Module):
    """Deterministic toy LM: the next-token argmax for token i is (i+1) % vocab."""

    def __init__(self, vocab: int = 12):
        super().__init__()
        self.vocab = vocab
        weight = torch.zeros(vocab, vocab)
        for i in range(vocab):
            weight[i, (i + 1) % vocab] = 1.0
        self.table = nn.Embedding(vocab, vocab)
        with torch.no_grad():
            self.table.weight.copy_(weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # (1, T) -> (1, T, vocab)
        return self.table(tokens)


class _IdTokenizer:
    """Whitespace-separated integer ids; no EOS."""

    eos_token_id = None

    def encode(self, text: str) -> list[int]:
        return [int(t) for t in text.split()] or [0]

    def decode(self, ids) -> str:
        return " ".join(str(int(i)) for i in ids)


def test_generate():
    vocab = 12
    model = _CounterLM(vocab=vocab)
    tok = _IdTokenizer()

    # top_k=1 is greedy/argmax; with the counter model the walk is 3,4,5,6,7,8,9.
    out = run_generate(model, tok, "3", max_length=6, top_k=1, device="cpu")
    expected = " ".join(str((3 + i) % vocab) for i in range(7))
    assert out == expected, f"greedy output {out!r} != expected {expected!r}"

    # The greedy walk genuinely moves (so a wrong-position bug is detectable).
    assert len(set(out.split())) == 7

    # Determinism: identical inputs -> identical output.
    assert run_generate(model, tok, "3", max_length=6, top_k=1, device="cpu") == out

    # Length bound: prompt (1) + max_length (6) = 7 tokens.
    assert len(out.split()) == 7

    # Other strategies run and stay in-vocab.
    for kwargs in ({"temperature": 0.8}, {"top_k": 3}, {"top_p": 0.9}):
        result = run_generate(model, tok, "3", max_length=4, device="cpu", **kwargs)
        ids = [int(x) for x in result.split()]
        assert all(0 <= i < vocab for i in ids)
