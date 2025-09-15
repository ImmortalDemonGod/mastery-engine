
---

# Project Worklog: `cs336-basics`

This document is the single source of truth for the development process of this project. It is a living document, updated incrementally to chronicle objectives, actions taken, decisions made, bugs encountered, and discoveries found.

The primary goal of this log is to create a reproducible and auditable trail of work, transforming development from a series of ad-hoc actions into a systematic, evidence-driven process.

---

## Principles for Effective Worklog Entries

Each entry in this log should adhere to the following principles. They are designed to maximize the log's value for debugging, reflection, and collaboration.

#### 1. **Reverse Chronological Order**
*   **What:** New entries are always added to the top of the file.
*   **Why:** This convention ensures that the most recent work—which is almost always the most relevant context for current tasks—is immediately visible without scrolling. It provides a natural "stack trace" of your recent development history.

#### 2. **One Entry Per Logical Task or Commit**
*   **What:** An entry should correspond to a single, logical unit of work that results in a Git commit. Always include the commit hash.
*   **Why:** This creates a powerful, auditable link between your thought process (the log) and the resulting code (the commit). If you ever need to understand the exact state of the code when a decision was made, you can simply `git show <commit_hash>` to see the precise changes.

#### 3. **Be Scientific: Document Hypotheses and Evidence**
*   **What:** Frame your work as a series of small experiments. State your objective (your hypothesis), document your actions (the experiment), and record the results objectively (the evidence).
*   **Why:** This mindset shifts you from just "writing code" to "validating a hypothesis." It forces clarity of thought and ensures that your decisions are based on evidence from test outputs and observations, not just gut feelings. This is the core of effective debugging.

#### 4. **Treat Artifacts as First-Class Citizens**
*   **What:** Explicitly list the artifacts related to your work—the exact commands you ran, the names of log files you generated, or the diffs you reviewed.
*   **Why:** A log entry without evidence is just an opinion. By treating your logs, commands, and diffs as formal artifacts, you make your work **reproducible and verifiable**. Anyone (including your future self, weeks from now) can re-run your commands and see the same results, which is the gold standard for technical investigation.

#### 5. **Document Failures, Not Just Successes**
*   **What:** Give equal, if not more, attention to documenting bugs, failed attempts, and incorrect hypotheses. Detail the error, your debugging process, and the final solution.
*   **Why:** The most valuable learning and the most critical debugging information come from understanding what *didn't* work. A well-documented failure saves you from solving the same problem twice and builds a project-specific knowledge base of potential pitfalls and their solutions.

---

## Log Entry Template

*Copy and paste this template for each new entry at the top of the file.*

```markdown
---
### **YYYY-MM-DD HH:MM**

**Objective:**
*   (State the clear, single goal for this work session. What hypothesis are you testing or what task are you completing?)

**Actions & Command(s):**
*   (List the high-level steps taken to achieve the objective. Include the *exact* shell commands used for testing, data processing, etc.)
*   e.g., `uv run pytest tests/test_nn_utils.py::test_softmax_matches_pytorch`

**Observations & Results:**
*   (Describe the outcome. Did tests pass? Did they fail? Include key snippets of error messages, performance metrics, or any surprising behavior.)

**Analysis & Decisions:**
*   (Synthesize the results. What do the observations mean? What was the root cause of the bug? Based on the evidence, what is the next logical step?)

**Artifacts:**
*   **Commit:** `[Paste the full commit hash here]`
*   **Log File(s):** `[Optional: path/to/relevant/log.txt]`
*   **Diff File(s):** `[Optional: path/to/relevant/diff.txt]`
---
```

---

### **2025-09-15 18:54**

**Objective:**
*   Implement Scaled Dot-Product Attention (SDPA) and wire the adapter. Validate via targeted tests.

**Actions & Command(s):**
1.  Implemented `scaled_dot_product_attention(Q, K, V, mask)` in `cs336_basics/layers.py` using numerically stable softmax (delegates to `cs336_basics.utils.softmax`) and boolean mask support via `masked_fill(~mask, -inf)`.
2.  Wired `tests/adapters.py::run_scaled_dot_product_attention` to delegate to our implementation.
3.  Ran targeted tests: `uv run pytest -q tests/test_model.py::test_scaled_dot_product_attention tests/test_model.py::test_4d_scaled_dot_product_attention`

**Observations & Results:**
*   Both SDPA tests passed for 3D and 4D inputs.

**Analysis & Decisions:**
*   SDPA verified for batched and multi-head-shaped inputs. Next: implement Multi-Head Self-Attention (without RoPE).

**Artifacts:**
*   **Command:** `uv run pytest -q tests/test_model.py::test_scaled_dot_product_attention tests/test_model.py::test_4d_scaled_dot_product_attention`
*   **Commit:** `[Paste the full commit hash here]`
---

---

### **2025-09-15 18:47**

**Objective:**
*   Implement a from-scratch `SwiGLU` module and wire the `run_swiglu` adapter. Validate via targeted test.

**Actions & Command(s):**
1.  Extended `cs336_basics/layers.py` with `SwiGLU` composing three `Linear` layers (bias=False) with SiLU gating: `out = W2( SiLU(W1(x)) * W3(x) )`.
2.  Wired `tests/adapters.py::run_swiglu` to instantiate `_SwiGLU`, copy `w1`, `w2`, `w3` weights, and forward `in_features`.
3.  Ran targeted test: `uv run pytest tests/test_model.py::test_swiglu -q`

**Observations & Results:**
*   Test passed. Output matched snapshot (tolerances per test).

**Analysis & Decisions:**
*   `SwiGLU` matches reference semantics and shapes. Next: proceed to attention components or checkpoint adapters.

**Artifacts:**
*   **Command:** `uv run pytest tests/test_model.py::test_swiglu -q`
*   **Commit:** `[Paste the full commit hash here]`
---

---

### **2025-09-15 18:40**

**Objective:**
*   Implement `RMSNorm` and wire `run_rmsnorm`. Validate via targeted test.

**Actions & Command(s):**
1.  Implemented `RMSNorm(d_model, eps)` with affine weight (no bias): `y = (x / sqrt(mean(x^2) + eps)) * weight` over last dim in `cs336_basics/layers.py`.
2.  Wired `tests/adapters.py::run_rmsnorm` to instantiate `_RMSNorm`, copy `weights`, and forward inputs.
3.  Ran targeted test: `uv run pytest tests/test_model.py::test_rmsnorm -q`

**Observations & Results:**
*   Test passed; snapshot match for RMSNorm.

**Analysis & Decisions:**
*   Normalization primitive verified; proceed to activation (SiLU) next.

**Artifacts:**
*   **Command:** `uv run pytest tests/test_model.py::test_rmsnorm -q`
*   **Commit:** `[Paste the full commit hash here]`
---

---

### **2025-09-15 18:45**

**Objective:**
*   Implement `SiLU` activation and wire `run_silu`. Validate via targeted test.

**Actions & Command(s):**
1.  Added `silu(x) = x * sigmoid(x)` in `cs336_basics/layers.py`.
2.  Wired `tests/adapters.py::run_silu` to call `_silu_impl`.
3.  Ran targeted test: `uv run pytest tests/test_model.py::test_silu_matches_pytorch -q`

**Observations & Results:**
*   Test passed with parity against `torch.nn.functional.silu`.

**Analysis & Decisions:**
*   Activation verified; proceed to RMSNorm next.

**Artifacts:**
*   **Command:** `uv run pytest tests/test_model.py::test_silu_matches_pytorch -q`
*   **Commit:** `[Paste the full commit hash here]`
---

---

### **2025-09-15 18:38**

**Objective:**
*   Implement a from-scratch `Embedding` module and wire the `run_embedding` adapter. Validate via targeted test.

**Actions & Command(s):**
1.  Extended `cs336_basics/layers.py` with `Embedding`:
    - Weight shape `(vocab_size, d_model)`.
    - Forward performs table lookup `weight[token_ids]`.
    - Initialization: `trunc_normal_(mean=0.0, std=1.0, a=-3.0, b=3.0)`.
2.  Wired `tests/adapters.py::run_embedding` to instantiate `Embedding`, copy provided weights, and forward `token_ids`.
3.  Ran targeted test: `uv run pytest tests/test_model.py::test_embedding -q`

**Observations & Results:**
*   Test passed. Output matched snapshot exactly.

**Analysis & Decisions:**
*   Embedding semantics and initialization align with the reference. Next, continue with simple `layers` components (e.g., `SiLU`, `RMSNorm`) before circling back to checkpoint adapters.

**Artifacts:**
*   **Command:** `uv run pytest tests/test_model.py::test_embedding -q`
*   **Commit:** `[Paste the full commit hash here]`
---

---

### **2025-09-15 18:27**

**Objective:**
*   Implement a from-scratch `Linear` module and wire the `run_linear` adapter. Validate via targeted test.

**Actions & Command(s):**
1.  Created `cs336_basics/layers.py` and implemented `Linear`:
    - Weight shape `(d_out, d_in)`, optional bias (disabled by default for snapshot parity).
    - Initialization: uniform in `[-1/sqrt(fan_in), 1/sqrt(fan_in)]`.
2.  Wired `tests/adapters.py::run_linear` to instantiate our `Linear`, copy provided weights, and forward the input.
3.  Ran targeted test: `uv run pytest tests/test_model.py::test_linear -q`

**Observations & Results:**
*   Test passed. Output matched snapshot exactly.

**Analysis & Decisions:**
*   The `Linear` implementation matches the reference orientation and initialization.
*   Next: proceed to the next core building block (e.g., `Embedding`) and/or wire checkpoint adapters for serialization tests.

**Artifacts:**
*   **Command:** `uv run pytest tests/test_model.py::test_linear -q`
*   **Commit:** `[Paste the full commit hash here]`
---

---

### **2025-09-15 18:22**

**Objective:**
*   Implement the `save_checkpoint` and `load_checkpoint` utility functions in preparation for testing.

**Actions & Command(s):**
1.  Implemented `save_checkpoint` in `cs336_basics/utils.py` to serialize a dictionary containing the model state, optimizer state, and iteration number using `torch.save`.
2.  Implemented `load_checkpoint` in `cs336_basics/utils.py` to restore the state to a given model and optimizer, ensuring the checkpoint is loaded to the CPU via `map_location="cpu"` for portability.

**Observations & Results:**
*   The core logic for serialization and deserialization is now complete in the `cs336_basics` package.
*   The corresponding adapters (`run_save_checkpoint`, `run_load_checkpoint`) have not yet been wired up, and the validation test has not been run.

**Analysis & Decisions:**
*   The utility functions are ready for integration. The next immediate step is to wire them up in `tests/adapters.py` and run the `test_serialization.py` suite to validate their correctness in a full round-trip.


---

### **2025-09-15 17:36**

**Objective:**
*   Implement `get_batch` to sample uniform LM sequences and wire the adapter.

**Actions & Command(s):**
1.  Implemented `get_batch(dataset, batch_size, context_length, device)` in `cs336_basics/utils.py`:
    - Uses PyTorch RNG (`torch.randint`) for uniform start indices.
    - Builds `x` and `y` where `y = x + 1`.
    - Moves tensors to the requested device, letting PyTorch surface invalid-device errors.
2.  Wired `run_get_batch` in `tests/adapters.py` to delegate to the implementation.
3.  Ran targeted test: `uv run pytest tests/test_data.py::test_get_batch -q`

**Observations & Results:**
*   Test passed. Shapes correct, `y = x+1` verified, sampling distribution within expected statistical bounds, invalid device path raises.

**Analysis & Decisions:**
*   Confirms uniform sampling and device handling. Next: checkpointing save/load.

**Artifacts:**
*   **Command:** `uv run pytest tests/test_data.py::test_get_batch -q`

---

### **2025-09-15 17:23**

**Objective:**
*   Implement a from-scratch `AdamW` optimizer and wire it through the adapters.

**Actions & Command(s):**
1.  Created `cs336_basics/optimizer.py` and implemented `AdamW` inheriting from `torch.optim.Optimizer`:
    - Decoupled weight decay applied directly to parameters.
    - Maintains per-parameter `exp_avg`, `exp_avg_sq`, and `step` in `state` with bias correction.
2.  Wired `get_adamw_cls` in `tests/adapters.py` to return our custom class.
3.  Ran targeted test: `uv run pytest tests/test_optimizer.py::test_adamw -q`

**Observations & Results:**
*   Test passed. Our implementation either matches PyTorch’s `AdamW` closely or snapshot expectations.

**Analysis & Decisions:**
*   Implementation satisfies decoupled weight decay semantics and test expectations. Move on to data batching next.

**Artifacts:**
*   **Command:** `uv run pytest tests/test_optimizer.py::test_adamw -q`

---

### **2025-09-15 17:22**

**Objective:**
*   Implement `get_lr_cosine_schedule` (linear warmup + cosine decay) and wire the adapter.

**Actions & Command(s):**
1.  Implemented `get_lr_cosine_schedule` in `cs336_basics/utils.py` with:
    - Linear warmup from 0→max across `warmup_iters`.
    - Cosine decay from max→min over `[warmup_iters, cosine_cycle_iters]` using `min + 0.5*(max-min)*(1+cos(pi*progress))` with clamped progress.
    - Truncate to `min` beyond cycle end; safeguarded degenerate cases.
2.  Wired `run_get_lr_cosine_schedule` in `tests/adapters.py` to call the implementation.
3.  Ran targeted test: `uv run pytest tests/test_optimizer.py::test_get_lr_cosine_schedule -q`

**Observations & Results:**
*   Test passed, matching the exact expected LR sequence in the unit test.

**Analysis & Decisions:**
*   The schedule matches spec and unit expectations. Next: implement custom AdamW.

**Artifacts:**
*   **Command:** `uv run pytest tests/test_optimizer.py::test_get_lr_cosine_schedule -q`

---

### **2025-09-15 17:18**

**Objective:**
*   Implement in-place global L2 `gradient_clipping` and wire it through `tests/adapters.py`.

**Actions & Command(s):**
1.  Implemented `gradient_clipping(parameters, max_l2_norm)` in `cs336_basics/utils.py`:
    - Skip `None` grads, compute global L2 norm across all grads, scale in-place if exceeding `max_l2_norm` (epsilon 1e-6 in denominator).
2.  Wired `run_gradient_clipping` in `tests/adapters.py` to call `cs336_basics.utils.gradient_clipping`.
3.  Ran targeted test: `uv run pytest tests/test_nn_utils.py::test_gradient_clipping -q`

**Observations & Results:**
*   Test passed and matched `torch.nn.utils.clip_grad.clip_grad_norm_` behavior across parameters, skipping frozen ones.

**Analysis & Decisions:**
*   Implementation meets the global-norm requirement and in-place semantics. Proceed to LR schedule next.

**Artifacts:**
*   **Command:** `uv run pytest tests/test_nn_utils.py::test_gradient_clipping -q`

---

### **2025-09-15 17:08**

**Objective:**
*   Implement a numerically stable `cross_entropy` and wire it through `tests/adapters.py`, validating parity with PyTorch.

**Actions & Command(s):**
1.  Implemented `cross_entropy` in `cs336_basics/utils.py` using the log-sum-exp trick with float32 upcasting for stability.
2.  Wired `run_cross_entropy` in `tests/adapters.py` to call `cs336_basics.utils.cross_entropy`.
3.  Corrected earlier accidental adapter edits to keep `run_linear` and `run_embedding` as `NotImplementedError`.
4.  Ran targeted test: `uv run pytest tests/test_nn_utils.py::test_cross_entropy -q`

**Observations & Results:**
*   Test passed and matched `torch.nn.functional.cross_entropy` results, including for large-magnitude logits.

**Analysis & Decisions:**
*   The log-sum-exp formulation with float32 intermediates is robust and deterministic under strict tolerances.
*   Next: implement `gradient_clipping` and validate via `tests/test_nn_utils.py::test_gradient_clipping`.

**Artifacts:**
*   **Command:** `uv run pytest tests/test_nn_utils.py::test_cross_entropy -q`

---

### **2025-09-15 16:56**

**Objective:**
*   Implement a numerically stable `softmax` and wire it through `tests/adapters.py`, establishing the TDD loop.

**Actions & Command(s):**
1.  Created `cs336_basics/utils.py` for stateless utilities.
2.  Implemented `softmax` with float32 upcasting and the subtract-max trick for numerical stability.
3.  Wired `run_softmax` in `tests/adapters.py` to call `cs336_basics.utils.softmax`.
4.  Ran targeted test: `uv run pytest tests/test_nn_utils.py::test_softmax_matches_pytorch -q`

**Observations & Results:**
*   First attempt using `logsumexp` produced tiny discrepancies beyond `atol=1e-6` for the `x+100` case.
*   Switched to the subtract-max formulation; the test passed.

**Analysis & Decisions:**
*   Both formulations are stable; the subtract-max version matched PyTorch closer under the strict tolerance in this test.
*   Adopt project-wide policy to upcast intermediates to float32 for sensitive ops.
*   Next: implement `cross_entropy` using the log-sum-exp trick (float32 intermediates) and wire `run_cross_entropy`.

**Artifacts:**
*   **Command:** `uv run pytest tests/test_nn_utils.py::test_softmax_matches_pytorch -q`
---

