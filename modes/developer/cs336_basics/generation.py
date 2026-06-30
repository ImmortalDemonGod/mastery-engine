"""Autoregressive text generation with selectable sampling strategies.

Implements greedy / temperature / top-k / nucleus (top-p) decoding on top of any
language model that maps a ``(1, seq_len)`` LongTensor of token ids to
``(1, seq_len, vocab_size)`` logits.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _top_k_sample(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Sample one token id from the top-k logits (renormalized)."""
    k = min(k, logits.shape[-1])
    top_logits, top_indices = torch.topk(logits, k)
    top_probs = F.softmax(top_logits, dim=-1)
    choice = torch.multinomial(top_probs, num_samples=1)
    return top_indices[choice].squeeze()


def _nucleus_sample(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Sample one token id from the smallest set with cumulative prob >= p."""
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    # Keep the smallest prefix whose cumulative prob >= p, INCLUDING the token that
    # crosses the threshold. A token is kept iff the cumulative mass strictly BEFORE
    # it is < p (so probs [0.6, 0.3, 0.1] with p=0.7 keeps {0.6, 0.3}).
    keep = (cumsum - sorted_probs) < p
    keep[0] = True
    filtered = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
    filtered = filtered / filtered.sum()
    choice = torch.multinomial(filtered, num_samples=1)
    return sorted_indices[choice].squeeze()


@torch.no_grad()
def generate(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    device: str = "cpu",
) -> str:
    """Generate text autoregressively from ``prompt``.

    Sampling priority: top-p > top-k > plain temperature sampling. Temperature
    always scales the logits BEFORE softmax. Stops at ``max_length`` new tokens or
    when the tokenizer's EOS id is produced (if it exposes ``eos_token_id``).
    """
    model.eval()
    model.to(device)

    token_ids = tokenizer.encode(prompt)
    tokens = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
    eos_id = getattr(tokenizer, "eos_token_id", None)

    for _ in range(max_length):
        logits = model(tokens)              # (1, T, vocab)
        next_logits = logits[0, -1, :]      # logits for the NEXT token (last position)
        next_logits = next_logits / temperature

        if top_p is not None:
            next_token = _nucleus_sample(next_logits, top_p)
        elif top_k is not None:
            next_token = _top_k_sample(next_logits, top_k)
        else:
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze()

        tokens = torch.cat([tokens, next_token.view(1, 1)], dim=1)
        if eos_id is not None and int(next_token) == eos_id:
            break

    return tokenizer.decode(tokens[0].tolist())
