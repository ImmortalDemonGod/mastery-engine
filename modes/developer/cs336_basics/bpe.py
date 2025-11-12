from __future__ import annotations

import os
from collections import Counter, defaultdict
import heapq
import re
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
    
    # Step 3: Pre-tokenize using GPT-2 regex (word-like, numbers, punctuation, whitespace)
    # This matches OpenAI's original GPT-2 tokenization scheme for BPE training
    pat_str = (
        r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"
    )
    regex = re.compile(pat_str)
    pieces = regex.findall(text)
    # Build byte-level token stream with sentinel (-1) between pieces to prevent cross-boundary merges
    tokens: list[int] = []
    first = True
    for piece in pieces:
        if not first:
            tokens.append(-1)  # boundary sentinel
        tokens.extend(piece.encode('utf-8'))
        first = False
    
    # Step 4: Iterative merging with incremental updates
    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - 256  # Number of merges to perform

    # Build doubly-linked list over token indices for efficient merges
    n = len(tokens)
    if n == 0:
        return vocab, merges
    prev = [-1] * n
    nxt = [-1] * n
    for i in range(n):
        if i > 0:
            prev[i] = i - 1
        if i < n - 1:
            nxt[i] = i + 1
    alive = [True] * n

    # Occurrence sets for each pair: pair -> set of left indices
    occ: dict[tuple[int, int], set[int]] = defaultdict(set)
    for i in range(n - 1):
        a, b2 = tokens[i], tokens[i + 1]
        if a < 0 or b2 < 0:  # respect boundaries
            continue
        if a == 10 or b2 == 10:  # skip newline-adjacent pairs
            continue
        occ[(a, b2)].add(i)

    # Build max-heap of (-count, first_index, left_bytes, right_bytes, pair)
    # for fast best-pair selection with deterministic tie-breaking by earliest
    # occurrence and then lexicographic bytes.
    heap: list[tuple[int, int, bytes, bytes, tuple[int, int]]] = []
    for p, s in occ.items():
        if not s:
            continue
        l, r = p
        lb, rb = vocab[l], vocab[r]
        if 10 in lb or 10 in rb:
            continue
        first_idx = min(s)
        heap.append((-len(s), first_idx, lb, rb, p))
    heapq.heapify(heap)

    # If special tokens include GPT-style tokens beginning with "<|",
    # forbid merges that would create byte sequences containing b"<|".
    forbid_bytes = b"<|" if any(isinstance(st, str) and st.startswith("<|") for st in (special_tokens or [])) else None


    for _ in range(num_merges):
        # Extract best pair, skipping stale heap entries
        best_pair: tuple[int, int] | None = None
        while heap:
            negc, first_idx, lb, rb, p = heapq.heappop(heap)
            c = -negc
            cur_set = occ.get(p, set())
            cur = len(cur_set)
            if cur == 0:
                continue
            # validate both count and earliest index for freshness
            if cur != c or (cur_set and min(cur_set) != first_idx):
                continue
            if forbid_bytes is not None and forbid_bytes in (lb + rb):
                # Skip forbidden pair and continue searching
                continue
            best_pair = p
            break
        if best_pair is None:
            break

        left_id, right_id = best_pair
        left_bytes = vocab[left_id]
        right_bytes = vocab[right_id]
        merges.append((left_bytes, right_bytes))

        # Create new token
        new_token = next_token_id
        vocab[new_token] = left_bytes + right_bytes
        next_token_id += 1

        # Merge all non-overlapping occurrences of best_pair from left to right
        positions = sorted(occ[best_pair])
        occ[best_pair].clear()
        for i in positions:
            # Validate current occurrence
            if not (0 <= i < n and alive[i]):
                continue
            j = nxt[i]
            if j == -1 or not alive[j]:
                continue
            if tokens[i] != left_id or tokens[j] != right_id:
                continue

            # Remove affected neighbor pair occurrences
            # Left neighbor pair at index prev[i]
            li = prev[i]
            if li != -1 and alive[li] and tokens[li] >= 0:
                old_lp = (tokens[li], left_id)
                s = occ.get(old_lp)
                if s is not None and li in s:
                    s.discard(li)
                    if s:
                        lb, rb = vocab[old_lp[0]], vocab[old_lp[1]]
                        heapq.heappush(heap, (-len(s), min(s), lb, rb, old_lp))

            # Right neighbor pair at index j
            rj = nxt[j]
            if rj != -1 and alive[rj] and tokens[rj] >= 0:
                old_rp = (right_id, tokens[rj])
                s = occ.get(old_rp)
                if s is not None and j in s:
                    s.discard(j)
                    if s:
                        lb, rb = vocab[old_rp[0]], vocab[old_rp[1]]
                        heapq.heappush(heap, (-len(s), min(s), lb, rb, old_rp))

            # Perform the merge at (i, j)
            tokens[i] = new_token
            alive[j] = False
            # Link i to rj
            nxt[i] = rj
            if rj != -1:
                prev[rj] = i

            # Add new left pair occurrence at li
            if li != -1 and alive[li] and tokens[li] >= 0:
                nl = (tokens[li], tokens[i])
                occ[nl].add(li)
                lb, rb = vocab[nl[0]], vocab[nl[1]]
                s2 = occ[nl]
                heapq.heappush(heap, (-len(s2), min(s2), lb, rb, nl))

            # Add new right pair occurrence at i
            if nxt[i] != -1 and tokens[nxt[i]] >= 0:
                nr = (tokens[i], tokens[nxt[i]])
                occ[nr].add(i)
                lb, rb = vocab[nr[0]], vocab[nr[1]]
                s3 = occ[nr]
                heapq.heappush(heap, (-len(s3), min(s3), lb, rb, nr))

    
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
    Count all adjacent token pairs efficiently using Counter over zipped pairs.

    Args:
        tokens: List of token IDs

    Returns:
        Counter mapping pairs to their frequencies
    """
    if len(tokens) < 2:
        return Counter()
    # Counter(zip(...)) is implemented in C and is faster than Python loops
    return Counter(zip(tokens, tokens[1:]))


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
