



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

### **2025-09-17 11:45**

**Objective:**
*   Implement and pass tests for the `RMSNorm` layer, paying close attention to the numerical stability requirements.

**Actions & Command(s):**
1.  Created a new `RMSNorm` class inheriting from `nn.Module` in `cs336_basics/layers.py`.
2.  Implemented the forward pass according to the formula in the PDF, including the `float32` upcasting for the variance calculation.
3.  Wired the `run_rmsnorm` adapter in `tests/adapters.py`.
4.  `uv run pytest tests/test_model.py::test_rmsnorm`

**Observations & Results:**
*   The test passed successfully on the first attempt.
*   Confirmed that using `.to(torch.float32)` for the intermediate mean-square calculation and then casting back to the original dtype worked as expected. Without this, tests might have failed on a `bfloat16` setup.

**Analysis & Decisions:**
*   The implementation is correct and numerically stable. The upcasting strategy noted in the `IMPLEMENTATION_PLAN.md` was critical. The foundational layers (`Linear`, `Embedding`, `RMSNorm`) are now complete. The next step is to begin the attention mechanism, starting with `scaled_dot_product_attention`.

**Artifacts:**
*   **Commit:** `c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4`
---
### **2025-09-17 10:30**

**Objective:**
*   Implement the `softmax` utility function and ensure it passes all tests, especially for numerical stability.

**Actions & Command(s):**
1.  Created `cs336_basics/utils.py`.
2.  Implemented a naive `softmax` function: `exp(x) / exp(x).sum(...)`.
3.  Wired the adapter in `tests/adapters.py`.
4.  `uv run pytest tests/test_nn_utils.py::test_softmax_matches_pytorch`

**Observations & Results:**
*   The test failed. The basic case passed, but the high-magnitude input (`x + 100`) resulted in `tensor([[nan, nan, nan, nan, nan], ...])` due to `torch.exp()` returning `inf`.
*   **Error Snippet:** `AssertionError: Mismatched elements: ...`

**Analysis & Decisions:**
*   **Analysis:** The naive implementation is numerically unstable, as predicted. The `inf` values from `torch.exp()` lead to `inf / inf`, which results in `nan`.
*   **Decision:** Re-implemented the function using the **subtract-max trick**. The new implementation first calculates `stable_x = x - x.max(dim, keepdim=True).values`. This centers the largest value at 0, preventing overflow in the exponentiation while yielding the same final probabilities. After