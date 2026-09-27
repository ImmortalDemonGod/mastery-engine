# Oracle Corrections — mastery-corr-001-impl

Finding: CORR-001 (`audit/02-static-audit.md` L17, SHA `7f6610a902befcb84fc47e5c82a161e3d3184ce4`)

Each section below names the changed test and justifies why the **original oracle was wrong**,
anchored to the finding itself — not to the implementation.

---

## 1. `tests/engine/test_state.py::test_mark_stage_complete_harden_advances_module`

### Original oracle (origin/main)

```python
def test_mark_stage_complete_harden_advances_module(self):
    progress = UserProgress(curriculum_id="test", current_stage="harden")
    progress.mark_stage_complete("harden")          # no module_id argument
    assert progress.current_stage == "build"
    assert progress.current_module_index == 1
    assert len(progress.completed_modules) == 1     # only length — no ID value
```

### Why the old oracle was wrong

The test called `mark_stage_complete("harden")` **without a `module_id` argument** and then
asserted only `len(progress.completed_modules) == 1`.

This oracle was wrong on two independent grounds, each rooted in the finding rather than in
the fix:

**Ground 1 — The oracle accepted the synthetic ID without verifying it.**
`engine/schemas.py:168` (origin/main) generates `module_id = f"module_{self.current_module_index}"` —
a synthetic zero-based index string — and appends it to `completed_modules`.  The assertion
`len() == 1` passes regardless of *what* was appended; it would pass equally for `"module_0"`,
`"softmax"`, or any other string.  An oracle that does not assert the stored *value* cannot
distinguish correct from incorrect behavior.  The finding explicitly states that
`"module_0"` / `"module_1"` never match real `module.id` values like `"softmax"` at the
downstream consumers (`main.py:2196`, `main.py:2335`), so the oracle must verify the real ID
is stored — a length check is insufficient.

**Ground 2 — The oracle exercised the buggy call convention.**
The production callers (`main.py:511`, `main.py:1825`) have `current_module.id` available and
*should* pass it.  The old test called the method with no second argument, so it silently
confirmed the broken one-arg calling convention instead of verifying the correct two-arg one.
An oracle that only exercises the buggy signature encodes the defect rather than the invariant.

The correct oracle must: (a) call `mark_stage_complete("harden", "softmax")` to match the
required caller contract, and (b) assert `"softmax" in progress.completed_modules` to verify
the real ID is stored.

---

## 2. `tests/engine/test_submit_handlers.py::test_harden_success_advances_module`

### Original oracle (origin/main)

```python
progress.mark_stage_complete.assert_called_once_with("harden")
```
(at line 451; `progress` is a `MagicMock(spec=UserProgress)`)

### Why the old oracle was wrong

The fixture at line 415 creates a manifest with a single module `ModuleMetadata(id="softmax", ...)`.
The function under test (`_submit_harden_stage`) resolves `current_module` from
`manifest.modules[progress.current_module_index]`, giving `current_module.id == "softmax"`.

The *correct* behavior of `_submit_harden_stage` is to call
`progress.mark_stage_complete("harden", "softmax")` — passing the real module ID that is
unambiguously in scope at the call site.

The original oracle `assert_called_once_with("harden")` asserted the **buggy** call
signature: no module_id argument.  This is not a judgment call or a stricter-than-necessary
check; it is a direct encoding of the defect.  A mock assertion on a unit-under-test's
output call is testing the unit's *behavior* — if the unit is called incorrectly (one arg
instead of two), the mock assertion must reflect the correct call to catch regressions.

The old oracle would pass with the buggy code and fail with the correct code, which means it
was guarding the wrong behavior.  The invariant the oracle must enforce is: the handler passes
the real module ID to `mark_stage_complete`, which requires `assert_called_once_with("harden", "softmax")`.

---

## 3. `tests/e2e/test_complete_bjh_loop.py::test_complete_softmax_bjh_loop`

### Original oracle (origin/main) — relevant lines

```python
# Line 414: direct assertion against synthetic ID
assert state["completed_modules"][0] == "module_0", "Completed module should be module_0 (softmax)"

# Line 446: synthetic ID constructed identically to the bug site
module_id = f"module_{state['current_module_index']}"
if module_id not in state["completed_modules"]:
    state["completed_modules"].append(module_id)

# Lines 459-460: final assertions against both synthetic IDs
assert "module_0" in state["completed_modules"], "Module_0 (softmax) should be in completed modules"
assert "module_1" in state["completed_modules"], "Module_1 (cross_entropy) should be in completed modules"
```

### Why the old oracle was wrong

**Line 414 / Lines 459–460 — asserting the buggy ID directly.**
The finding states unambiguously that `"module_0"` / `"module_1"` are synthetic artifacts of
the defect in `schemas.py:168`, not valid module identifiers.  The downstream consumers at
`main.py:2196` and `main.py:2335` compare against `module.id` — real values like `"softmax"`
and `"cross_entropy"` from `curricula/cs336_a1/manifest.json`.  An oracle that asserts
`completed_modules[0] == "module_0"` is asserting the presence of the *bug artifact*, not the
correct system behavior.  When the bug is fixed, the engine writes `"softmax"` — and the old
oracle would falsely flag this as a regression.

**Line 446 — test-internal state manipulation replicating the bug.**
The E2E test shortcuts the cross_entropy harden stage by directly writing to the state JSON.
The original code computed `module_id = f"module_{state['current_module_index']}"` — the
exact same synthetic expression as `schemas.py:168` — then appended it.  This means the test
was not only asserting the wrong value; it was *actively injecting* the synthetic ID into the
state, perpetuating the defect even after the engine itself is fixed.  An E2E test that
manually constructs state must use the same IDs the engine would produce.  After the fix, the
engine writes `"cross_entropy"` for module index 1; the test's manual injection must mirror
this (`module_id = "cross_entropy"`) to produce consistent state.

In all three cases the old oracle encoded the defect (`"module_0"` / `"module_1"` / `f"module_{index}"`)
rather than the invariant (real IDs from the manifest).  The corrections replace the defect-
encoding assertions with the correct-behavior assertions.
