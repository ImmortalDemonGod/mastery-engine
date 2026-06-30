"""Tests for autoregressive text generation (`cs336_basics.generation.generate`).

Uses a tiny deterministic toy LM whose next-token logits depend only on the current
token, so greedy decoding (`top_k=1`) has a single, checkable answer — no trained
model, GPU, or snapshots required.
"""
import torch
from torch import nn

from .adapters import run_generate


class _TinyLM(nn.Module):
    """Deterministic toy LM: logits for the next token depend only on the last token."""

    def __init__(self, vocab: int = 12, seed: int = 1):
        super().__init__()
        self.vocab = vocab
        gen = torch.Generator().manual_seed(seed)
        self.table = nn.Embedding(vocab, vocab)
        with torch.no_grad():
            self.table.weight.copy_(torch.randn(vocab, vocab, generator=gen))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # (1, T) -> (1, T, vocab)
        return self.table(tokens)


class _IdTokenizer:
    """Whitespace-separated integer ids; no EOS."""

    eos_token_id = None

    def encode(self, text: str) -> list[int]:
        return [int(t) for t in text.split()] or [0]

    def decode(self, ids) -> str:
        return " ".join(str(int(i)) for i in ids)


def _expected_greedy(model: nn.Module, start: list[int], steps: int) -> list[int]:
    toks = list(start)
    for _ in range(steps):
        with torch.no_grad():
            logits = model(torch.tensor([toks]))
        toks.append(int(logits[0, -1].argmax()))
    return toks


def test_generate():
    model = _TinyLM(vocab=12, seed=1)
    tok = _IdTokenizer()

    # top_k=1 is greedy/argmax -> deterministic and exactly checkable.
    out = run_generate(model, tok, "3", max_length=6, top_k=1, device="cpu")
    expected = _IdTokenizer().decode(_expected_greedy(model, [3], 6))
    assert out == expected, f"greedy output {out!r} != expected {expected!r}"

    # The reference greedy walk must actually move (not a constant token), otherwise
    # a "wrong-position" bug could not be detected.
    assert len(set(out.split())) > 1, "toy model degenerate; pick another seed"

    # Determinism: identical inputs -> identical output.
    assert run_generate(model, tok, "3", max_length=6, top_k=1, device="cpu") == out

    # Length bound: prompt (1) + max_length (6) = 7 tokens.
    assert len(out.split()) == 7

    # Other strategies run and stay in-vocab.
    for kwargs in ({"temperature": 0.8}, {"top_k": 3}, {"top_p": 0.9}):
        result = run_generate(model, tok, "3", max_length=4, device="cpu", **kwargs)
        ids = [int(x) for x in result.split()]
        assert all(0 <= i < 12 for i in ids)
