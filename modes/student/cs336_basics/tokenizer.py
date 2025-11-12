from __future__ import annotations

from typing import Iterable, Optional


class Tokenizer:
    """
    From-scratch byte-level BPE Tokenizer that applies merges sequentially.

    This implementation uses the provided vocabulary (id -> bytes) and the
    ordered list of merges (bytes, bytes). It handles special tokens by
    greedy longest-match segmentation and provides encode/decode and
    streaming encode_iterable.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: Optional[list[str]] = None,
    ) -> None:
        # Persist inputs
        self._vocab = dict(vocab)
        self._merges = list(merges)
        self._special_tokens = tuple(special_tokens or [])

        # Build byte<->id maps for fast lookup
        self._id_to_bytes = dict(self._vocab)
        self._bytes_to_id = {b: i for i, b in self._vocab.items()}

        # Precompute special-tokens (bytes) and sort by length (greedy, longest-first)
        self._special_tokens_bytes = [s.encode("utf-8") for s in self._special_tokens]
        self._special_tokens_sorted = sorted(self._special_tokens, key=len, reverse=True)

    # ------------------------- Public API -------------------------
    def encode(self, text: str) -> list[int]:
        """Encode a string into token IDs by applying merges sequentially.

        - Segments around special tokens greedily (longest match wins)
        - Applies merges to non-special spans only
        - Maps final byte-sequences to ids via provided vocabulary
        """
        if not self._special_tokens:
            return self._encode_span(text)

        ids: list[int] = []
        i = 0
        n = len(text)
        while i < n:
            # Greedy longest-match for special tokens
            matched = False
            for tok in self._special_tokens_sorted:
                if tok and text.startswith(tok, i):
                    tok_b = tok.encode("utf-8")
                    tok_id = self._bytes_to_id.get(tok_b)
                    if tok_id is not None:
                        ids.append(tok_id)
                        i += len(tok)
                        matched = True
                        break
            if matched:
                continue

            # Encode up to next special token occurrence
            next_pos = n
            for tok in self._special_tokens_sorted:
                if not tok:
                    continue
                j = text.find(tok, i)
                if j != -1:
                    next_pos = min(next_pos, j)
            span = text[i:next_pos]
            if span:
                ids.extend(self._encode_span(span))
            i = next_pos
        return ids

    def decode(self, ids: list[int]) -> str:
        # Concatenate bytes from vocab, then decode as UTF-8
        out = bytearray()
        for _id in ids:
            b = self._id_to_bytes.get(int(_id))
            if b is None:
                # Unknown id: skip to preserve robustness
                continue
            out.extend(b)
        return out.decode("utf-8", errors="ignore")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for chunk in iterable:
            for _id in self.encode(chunk):
                yield _id

    # ------------------------ Internal API ------------------------
    def _encode_span(self, text: str) -> list[int]:
        """Encode a non-special span (no special tokens inside)."""
        # Start with byte-level tokens as bytes objects
        b = text.encode("utf-8")
        tokens: list[bytes] = [bytes([x]) for x in b]

        # Apply merges sequentially
        for left, right in self._merges:
            if not tokens or len(tokens) == 1:
                break
            tokens = self._apply_merge(tokens, left, right)

        # Map final byte-sequences to ids
        ids: list[int] = []
        for t in tokens:
            tid = self._bytes_to_id.get(t)
            if tid is None:
                # If a token sequence isn't present (shouldn't happen with correct vocab),
                # fall back to splitting to bytes to ensure progress.
                for by in t:
                    tid_b = self._bytes_to_id.get(bytes([by]))
                    if tid_b is not None:
                        ids.append(tid_b)
            else:
                ids.append(tid)
        return ids

    @staticmethod
    def _apply_merge(
        tokens: list[bytes], left: bytes, right: bytes
    ) -> list[bytes]:
        """Replace consecutive (left, right) pairs with left+right throughout tokens."""
        if len(tokens) < 2:
            return tokens

        merged = []
        i = 0
        L = len(tokens)
        while i < L:
            if i < L - 1 and tokens[i] == left and tokens[i + 1] == right:
                merged.append(left + right)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged
