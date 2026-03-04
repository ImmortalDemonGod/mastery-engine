from __future__ import annotations

import os
from collections import Counter
from pathlib import Path


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a Byte Pair Encoding (BPE) tokenizer from a text corpus.
    
    This is a true from-scratch implementation of the BPE algorithm:
    1. Initialize vocabulary with 256 bytes
    2. Iteratively merge most frequent adjacent token pairs
    3. Build vocabulary up to vocab_size
    
    Args:
        input_path: Path to training corpus text file
        vocab_size: Target vocabulary size (includes 256 initial bytes)
        special_tokens: List of special tokens to add to vocabulary
        **kwargs: Additional arguments (unused, for compatibility)
    
    Returns:
        vocab: Dictionary mapping token ID to bytes sequence
               {0: b'\x00', ..., 256: b'th', 257: b'the', ...}
        merges: List of merge operations in order applied
                [(b'h', b'e'), (b'he', b'l'), ...]
    """
    # TODO: Implement from-scratch BPE training
    # 1. Initialize vocab with 256 bytes: {0: b'\x00', 1: b'\x01', ..., 255: b'\xff'}
    # 2. Read corpus and convert to byte-level tokens
    # 3. For (vocab_size - 256) iterations:
    #    a. Count all adjacent token pairs
    #    b. Find most frequent pair
    #    c. Create new token for that pair
    #    d. Replace all occurrences in token stream
    #    e. Add (left_bytes, right_bytes) to merges list
    # 4. Add special_tokens to vocab
    # 5. Return (vocab, merges)
    raise NotImplementedError("TODO: Implement train_bpe")
