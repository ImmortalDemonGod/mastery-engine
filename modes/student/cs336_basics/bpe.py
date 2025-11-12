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
    input_path = Path(input_path)
    
    # Step 1: Read corpus
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Step 2: Initialize vocabulary with all 256 bytes
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_token_id = 256
    
    # Step 3: Convert text to byte-level tokens (initial encoding)
    text_bytes = text.encode('utf-8')
    tokens = list(text_bytes)  # Each byte is a token ID (0-255)
    
    # Step 4: Iterative merging with optimized pair counting
    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - 256  # Number of merges to perform
    
    for _ in range(num_merges):
        # Count all adjacent token pairs (optimized with Counter)
        pair_counts = _count_pairs(tokens)
        
        # If no pairs left, stop early
        if not pair_counts:
            break
        
        # Find most frequent pair (deterministic tie-breaking: lexicographic)
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], -p[0], -p[1]))
        
        # Record merge operation (store as bytes for consistency)
        left_bytes = vocab[best_pair[0]]
        right_bytes = vocab[best_pair[1]]
        merges.append((left_bytes, right_bytes))
        
        # Create new token by concatenating
        new_token_bytes = left_bytes + right_bytes
        vocab[next_token_id] = new_token_bytes
        
        # Replace all occurrences of pair with new token ID (optimized)
        tokens = _replace_pair_fast(tokens, best_pair, next_token_id)
        
        next_token_id += 1
    
    # Step 5: Add special tokens to vocabulary (if provided)
    if special_tokens:
        # Add special tokens that aren't already in vocab
        existing_bytes = set(vocab.values())
        for special in special_tokens:
            special_bytes = special.encode('utf-8')
            if special_bytes not in existing_bytes:
                vocab[next_token_id] = special_bytes
                next_token_id += 1
    
    return vocab, merges


def _count_pairs(tokens: list[int]) -> Counter[tuple[int, int]]:
    """
    Count all adjacent token pairs efficiently.
    
    Args:
        tokens: List of token IDs
    
    Returns:
        Counter mapping pairs to their frequencies
    """
    pairs = [
        (tokens[i], tokens[i + 1])
        for i in range(len(tokens) - 1)
    ]
    return Counter(pairs)


def _replace_pair_fast(
    tokens: list[int],
    pair: tuple[int, int],
    new_token_id: int,
) -> list[int]:
    """
    Replace all occurrences of a token pair with a new merged token (optimized).
    
    Uses list comprehension and slicing for better performance.
    
    Args:
        tokens: List of token IDs
        pair: Tuple of (left_id, right_id) to replace
        new_token_id: New token ID to use for the merged pair
    
    Returns:
        Updated token list with pairs replaced
    """
    if len(tokens) < 2:
        return tokens
    
    result = []
    i = 0
    while i < len(tokens):
        # Check if current position matches the pair
        if (i < len(tokens) - 1 and 
            tokens[i] == pair[0] and 
            tokens[i + 1] == pair[1]):
            result.append(new_token_id)
            i += 2  # Skip both tokens in the pair
        else:
            result.append(tokens[i])
            i += 1
    return result
