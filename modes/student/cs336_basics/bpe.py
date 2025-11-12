from __future__ import annotations
import os

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """TODO: Implement BPE tokenizer training. See bpe_tokenizer module."""
    raise NotImplementedError("TODO: Implement train_bpe function")
