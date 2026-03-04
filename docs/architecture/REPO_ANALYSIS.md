# CS336 Assignment 1 (Basics) — Repository Analysis

Generated: 2025-09-15 15:23:33 -05:00

This document summarizes the purpose, structure, tools, tests, and required implementation contracts for the repository. It is intended as a quick but thorough reference while you implement the assignment.

## Summary

- Course/Assignment: CS336 Spring 2025 — Assignment 1: Basics.
- Paradigm: Test-driven. You implement core functionality and wire it up via `tests/adapters.py`.
- Scope:
  - Tokenization and BPE training matching GPT-2 behavior.
  - Core NN utilities (softmax, cross-entropy, gradient clipping).
  - Optimizer and schedule (AdamW, cosine LR with warmup).
  - Transformer components: Linear, Embedding, SwiGLU FFN, RMSNorm, SDPA, MHA (+RoPE), TransformerBlock, and full LM.
  - Data batching and checkpointing (save/load).

## Repository Structure

- `README.md`
  - Setup with `uv`, running tests, and optional data downloads.
  - Points to full spec: `cs336_spring2025_assignment1_basics.pdf`.
- `cs336_spring2025_assignment1_basics.pdf`
  - Handout/spec for algorithms and implementation details.
- `pyproject.toml`
  - Project metadata, dependencies, pytest and ruff configuration.
  - PyTorch version is platform-conditioned (mac x86_64 uses 2.2.2; otherwise 2.6.0).
- `cs336_basics/`
  - `__init__.py` — package version.
  - `pretokenization_example.py` — example utility for parallelizable pre-tokenization chunking by splitting at a special token boundary.
- `tests/`
  - `adapters.py` — the single integration surface you must implement; tests call its functions.
  - `test_*.py` — unit and snapshot tests defining desired behavior.
  - `fixtures/` — GPT-2 vocab/merges, sample corpora, and a teacher model state dict/config.
  - `_snapshots/` — expected numeric outputs for multiple tests.
- `make_submission.sh`
  - Runs tests and packages a submission zip while excluding caches, dotfiles, large artifacts, and fixture/snapshot data.

## Environment and Tooling

- Python: `>=3.11`
- Environment manager: `uv`
  - Run code/tests with `uv run <cmd>`, e.g. `uv run pytest`.
- Key dependencies:
  - `torch` (`2.2.2` on mac x86_64, `2.6.0` elsewhere), `einops`, `einx`, `jaxtyping`, `tiktoken`, `pytest`, `tqdm`, `wandb`, `ty`, `regex`, `submitit`, `psutil`.
- Pytest config (`[tool.pytest.ini_options]`):
  - `log_cli = true`, `log_cli_level = WARNING`, `addopts = -s`.
- Linting: ruff (120 char width, selected/ignored rules; extra ignores for `__init__.py`).

## Running and Submitting

- Run tests:
  - `uv run pytest`
- Download data (optional for your own experimentation; not required for unit tests): see `README.md` section “Download data”.
- Submission packaging:
  - `./make_submission.sh` creates `cs336-spring2025-assignment-1-submission.zip`.
  - Excludes dotfiles, caches, common large text/binary artifacts, and `tests/fixtures` and `tests/_snapshots`.

## Tests and Adapters: Implementation Contracts

All tests call functions defined in `tests/adapters.py`. Implement your own modules (e.g., under `cs336_basics/`) and have the adapters call into them. The contracts below summarize inputs/outputs and key semantics validated by the tests.

### Linear and Embedding

- `run_linear(d_in, d_out, weights: Float[Tensor, "d_out d_in"], in_features: Float[Tensor, "... d_in"]) -> ... d_out`
  - Compute a batched linear transform (biasless) using provided weights.
  - Validated via snapshot with teacher weights.
- `run_embedding(vocab_size, d_model, weights: Float[Tensor, "vocab_size d_model"], token_ids: Int[Tensor, "..."]) -> ... d_model`
  - Gather embeddings by id; validated via snapshot with teacher weights.

### Activation and Normalization

- `run_silu(in_features) -> same shape`
  - Must match `torch.nn.functional.silu` numerically.
- `run_rmsnorm(d_model, eps, weights: Float[Tensor, "d_model"], in_features) -> same shape`
  - RMSNorm with learned scale (gamma), no bias. Apply epsilon for stability.

### SwiGLU Feed-Forward

- `run_swiglu(d_model, d_ff, w1_weight, w2_weight, w3_weight, in_features) -> ... d_model`
  - Pattern: up-projection to `d_ff`, gated via SiLU/GLU variant, then down-projection to `d_model`.
  - A common implementation: `gate = silu(x @ W1^T) * (x @ W3^T)` then `out = gate @ W2^T`.
  - Snapshot-checked.

### Softmax, Cross-Entropy, Gradient Clipping

- `run_softmax(x, dim) -> same shape`
  - Must match `F.softmax`; ensure numerical stability (subtract max across `dim`).
- `run_cross_entropy(inputs: [B, V], targets: [B]) -> scalar`
  - Must match `F.cross_entropy` (use log-sum-exp for stability).
- `run_gradient_clipping(parameters, max_l2_norm) -> None`
  - Clip global L2 norm of grads in-place to match `torch.nn.utils.clip_grad.clip_grad_norm_`.
  - Skip params with no grad; handle some with `requires_grad=False`.

### Optimizer and Learning Rate Schedule

- `get_adamw_cls() -> OptimizerClass`
  - Return an optimizer implementing AdamW. Returning `torch.optim.AdamW` is acceptable.
- `run_get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters) -> float`
  - Linear warmup (0 → max) for `warmup_iters` steps.
  - Cosine decay (max → min) over `cosine_cycle_iters`.
  - Clamp at `min` afterwards.
  - Tests enumerate exact expected values for 25 iters.

### Attention

- `run_scaled_dot_product_attention(Q, K, V, mask=None) -> output`
  - Compute attention: `softmax((Q @ K^T) / sqrt(d_k) + mask_bias) @ V`.
  - Support 3D/4D shapes (batch/head dims folded or explicit). Mask is boolean shaped like `... queries keys`; masked positions must not contribute (add −inf before softmax).
- `run_multihead_self_attention(d_model, num_heads, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, in_features) -> ...`
  - Fused projections for all heads, split to heads, SDPA, merge, output projection. No RoPE here.
  - Test weights are concatenated across heads (e.g., `[d_model, d_model]`), rows ordered by head.
- `run_multihead_self_attention_with_rope(..., max_seq_len, theta, ..., token_positions) -> ...`
  - Same as above, but apply RoPE to Q and K only.
  - RoPE dimension equals head size: `d_model // num_heads`.
  - `token_positions` provided; handle broadcasting.
- `run_rope(d_k, theta, max_seq_len, in_query_or_key, token_positions) -> ...`
  - Apply rotary positional embeddings to Q or K. Precompute cos/sin up to `max_seq_len` as needed.

### Transformer Block and Language Model

- `run_transformer_block(d_model, num_heads, d_ff, max_seq_len, theta, weights, in_features) -> ...`
  - Pre-norm: `x + MHA(RMSNorm(x))` with RoPE; then `x + FFN(RMSNorm(x))` with SwiGLU.
  - `weights` keys include: `attn.{q,k,v}_proj.weight`, `attn.output_proj.weight`, `ln1.weight`, `ffn.{w1,w2,w3}.weight`, `ln2.weight`.
  - Snapshot-checked.
- `run_transformer_lm(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, weights, in_indices) -> logits`
  - Embed tokens → N Transformer blocks (with RoPE) → final RMSNorm → LM head → logits `[B, T, V]`.
  - Must also handle truncated inputs (half sequence); snapshot-checked.

### Data Batching

- `run_get_batch(dataset: 1D np.ndarray[int], batch_size, context_length, device) -> (x, y)`
  - Randomly draw valid start indices uniformly from `[0, len(dataset) - context_length - 1]`.
  - `x.shape = y.shape = [batch_size, context_length]`; `y = x + 1` elementwise (next token labels).
  - Place tensors on requested device; invalid device should trigger torch errors (e.g., `cuda:99`).
  - The test checks sampling distribution statistics across many draws.

### Checkpointing

- `run_save_checkpoint(model, optimizer, iteration, out)`
  - Serialize model state dict, optimizer state dict, and the `iteration` counter to path or file-like.
- `run_load_checkpoint(src, model, optimizer) -> iteration`
  - Restore states and return saved iteration. The test compares model params numerically and optimizer state dicts for logical equality.

### Tokenizer and BPE Training

- `get_tokenizer(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None)`
  - Must provide:
    - `encode(str) -> list[int]`
    - `decode(list[int]) -> str`
    - `encode_iterable(iterable_of_text_lines_or_file) -> Iterator[int]`
  - Behavior must match GPT-2 encoding (`tiktoken`) for many inputs (with and without special tokens), including round-trip correctness.
  - Preserve special tokens exactly (never split), including overlapping specials (e.g., `<|endoftext|>` and `<|endoftext|><|endoftext|>`).
  - `encode_iterable` must be memory-efficient (Linux-only memory limit tests).
- `run_train_bpe(input_path, vocab_size, special_tokens, **kwargs) -> (vocab, merges)`
  - Must train quickly on small corpus (`< 1.5s` in test environment).
  - Merges must exactly match reference; vocab keys and values must match as sets.
  - Special tokens should not be merged into other tokens. Ensure pre-tokenization and counting respect this.
  - See `cs336_basics/pretokenization_example.py` for chunk boundary logic enabling parallel pre-tokenization by splitting chunks at special token boundaries.

## Pre-tokenization Chunking Utility

- File: `cs336_basics/pretokenization_example.py`
  - Function: `find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]`
  - Purpose: Identify byte offsets to split a large file into chunks such that boundaries fall on the next occurrence of `split_special_token` (e.g., `b"<|endoftext|>"`).
  - Usage example:
    ```python
    with open(..., "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on this chunk and count
    ```

## Snapshot Testing

- Snapshot utilities live in `tests/conftest.py` (`NumpySnapshot`, `Snapshot`).
- Numeric tests load expected arrays from `tests/_snapshots/*.npz` and compare using specified tolerances (`rtol`, `atol`).
- Some tests relax tolerances slightly where appropriate (e.g., transformer LM).

## Fixtures and Teacher Weights

- Teacher state dict: `tests/fixtures/ts_tests/model.pt` with `model_config.json`.
- Tests use these to validate your component outputs (e.g., `Linear`, `Embedding`, `SwiGLU`, `MHA`, `TransformerBlock`, `TransformerLM`).

## Performance, Determinism, and Stability

- Numerical stability:
  - `softmax` and `cross_entropy` must be stable under large logits; implement subtract-max and log-sum-exp.
- Determinism:
  - Tests seed torch and use fixed fixtures for reproducible comparisons.
- Performance:
  - `run_train_bpe` must be reasonably fast; efficient pre-token counting and data structures are important. Consider chunk-based or streaming processing.

## Common Pitfalls

- RoPE:
  - Apply to Q and K only; use per-head dimension (`d_model // num_heads`) for rotary embedding.
  - Ensure positions align with token indices (`token_positions`).
- Masking:
  - Masks should zero out contributions by adding `-inf` to those logits before softmax.
- Gradient clipping:
  - Compute a single global grad norm; skip params with `grad is None`.
- `run_get_batch`:
  - Uniform sampling across valid start indices; place tensors on the requested device; handle invalid device ordinals properly.
- Tokenizer:
  - Preserve special tokens exactly; support overlapping specials; ensure `encode_iterable` streams without materializing full contents.

## CHANGELOG Highlights (relevant items)

- `1.0.6` (2025-08-28): RoPE formulation fixes; adapters typing fixes; dependency relock.
- `1.0.5` (2025-04-15): Submission script added; switch to `uv_build`; RoPE indexing fix; test fixes; typing and formatting improvements.
- `1.0.4` (2025-04-08): Pretokenization parallelization guidance; split on special tokens; MPS compile fix.
- `1.0.3` (2025-04-07): Test for removing special tokens in BPE; fix RoPE off-by-one; mac Intel support for PyTorch 2.2.2; Python 3.11 support.
- `1.0.2` (2025-04-03): Added missing tests; RMSNorm interface clarifications; SwiGLU hints; BPE stylized example clarification.

## Recommended Implementation Order

1. `run_softmax`, `run_cross_entropy`, `run_gradient_clipping` — quick wins and well-specified.
2. `get_adamw_cls` (return `torch.optim.AdamW`), `run_get_lr_cosine_schedule` — deterministic numeric checks.
3. `run_get_batch`, `run_save_checkpoint`, `run_load_checkpoint` — infrastructure pieces.
4. `run_linear`, `run_embedding`, `run_silu`, `run_rmsnorm` — building blocks validated against teacher weights.
5. `run_scaled_dot_product_attention`, `run_multihead_self_attention` — attention core.
6. `run_rope`, `run_multihead_self_attention_with_rope` — position handling.
7. `run_swiglu`, `run_transformer_block`, `run_transformer_lm` — full model path.
8. `get_tokenizer`, `run_train_bpe` — most intricate; use handout specs and `pretokenization_example.py` for chunked counting.

## Useful File References

- `tests/adapters.py` — adapter function signatures and shape annotations.
- `tests/test_model.py` — linear, embedding, RMSNorm, SwiGLU, attention, RoPE, transformer block/LM tests.
- `tests/test_nn_utils.py` — softmax, cross-entropy, gradient clipping tests.
- `tests/test_optimizer.py` — AdamW equivalence and cosine schedule tests.
- `tests/test_data.py` — batching test including sampling distribution.
- `tests/test_serialization.py` — save/load checkpoint roundtrip.
- `tests/test_tokenizer.py` — rigorous tokenizer/streaming memory behavior vs `tiktoken`.
- `tests/test_train_bpe.py` — BPE training speed and equivalence to reference merges/vocab.

## Notes on Wiring the Adapters

- Create your implementation modules (e.g., `cs336_basics/nn.py`, `cs336_basics/attn.py`, `cs336_basics/rope.py`, `cs336_basics/ffn.py`, `cs336_basics/model.py`, `cs336_basics/tokenizer.py`, `cs336_basics/bpe.py`).
- In `tests/adapters.py`, import your functions/classes and forward the calls. Keep tensor shapes/types consistent with annotations.
- For attention/MHA weights, remember the test convention: concatenated rows per head for q/k/v projections.

---

If you maintain this document as your single source of truth during implementation, you should be able to systematically satisfy each test. Update this file as you make architectural decisions or note implementation-specific choices.
