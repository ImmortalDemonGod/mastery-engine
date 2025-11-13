from __future__ import annotations
from collections.abc import Iterable


class Tokenizer:
    """
    Byte-level BPE tokenizer with special token support.
    
    Wraps BPE vocabulary and merges to provide encode/decode functionality.
    """
    
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """Initialize tokenizer with BPE vocab and merges."""
        # TODO: Store vocab, merges, special_tokens
        # TODO: Build reverse lookup: bytes -> id
        # TODO: Sort special tokens by length for greedy matching
        raise NotImplementedError("TODO: Implement Tokenizer.__init__")
    
    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        # TODO: Segment text around special tokens
        # TODO: Apply BPE merges to non-special segments
        # TODO: Map bytes to token IDs
        raise NotImplementedError("TODO: Implement Tokenizer.encode")
    
    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
        # TODO: Map each ID to bytes using vocab
        # TODO: Concatenate bytes and decode as UTF-8
        raise NotImplementedError("TODO: Implement Tokenizer.decode")
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """Stream-encode chunks of text."""
        # TODO: Yield token IDs for each chunk in iterable
        raise NotImplementedError("TODO: Implement Tokenizer.encode_iterable")
