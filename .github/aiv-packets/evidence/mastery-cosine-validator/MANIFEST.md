# AIV Evidence Manifest — mastery-cosine-validator

**Finding:** cosine-validator-wrong-file  
**Baseline SHA (cited):** 7f6610a902befcb84fc47e5c82a161e3d3184ce4 (origin/main)  
**Fix HEAD SHA:** d4970aa82ba5392c23f199169ad15c73ccabf902 (fix/mastery-cosine-validator)  
**Produced:** 2026-06-21  
**Independent assessor verdict:** OVERALL READY (all 4 artifacts CONFIRMED)

## Claims

| # | Claim | Baseline ref | Artifact type |
|---|-------|-------------|---------------|
| C1 | Baseline validator.sh:18 copies `optimizer.py` instead of `utils.py` (B1 bug present) | 7f6610a | design tests (static + integration) + live-fire subprocess |
| C2 | Baseline validator.sh uses pytest node `test_lr_cosine_schedule` (non-existent; B2 bug present) | 7f6610a | design tests (static) + live-fire exit code 4 |
| C3 | HEAD validator.sh copies `utils.py` (B1 fixed) and propagates developer's implementation to shadow | d4970aa | design tests PASS + live-fire subprocess propagation confirmed |
| C4 | HEAD validator.sh uses pytest node `test_get_lr_cosine_schedule` and it PASSES | d4970aa | live-fire: 1 passed, exit 0 |

## Artifacts

| File | SHA256 | Proves | Cited ref | AIV class |
|------|--------|--------|-----------|-----------|
| `baseline_red.txt` | `c9b6fbcd57c372143ab0859f09ac3ce12341e27d296427b794acd9be50d9086d` | C1, C2 — 3/3 design tests FAIL on 7f6610a | 7f6610a | A (execution), B (referential), D (differential) |
| `head_green.txt` | `629a749d97870ca4ea0507c7df549348e32ab2b6a9910cc672016350df4caeca` | C3, C4 — 3/3 design tests PASS at d4970aa | d4970aa | A (execution), B (referential), D (differential) |
| `live_fire_base_defect.txt` | `6c6dd77ab19ece7d474d252f52bb58e46005f88296530862bc36b5819fbaeebc` | C1, C2 — live subprocess: shadow stale + exit 4 | 7f6610a | A (live execution on real composed system) |
| `live_fire_head_e2e.txt` | `fc856bc5ccd820961641219b2690e56eca6788ddeac51abb958c2ebcb301065d` | C3, C4 — live subprocess: utils.py propagated + test PASSED | d4970aa | A (live execution), B (referential) |
| `validator_sh_diff.txt` | `3560aecf0143b26d4b4b903cd6dd1c70d56294c3d80b88fa6ef59a42f945f0ff` | Before/after diff of the changed file vs cited baseline | 7f6610a vs d4970aa | D (differential) |

## Class coverage

| Class | Status | Evidence |
|-------|--------|---------|
| A — Execution | REAL | live_fire_base_defect.txt (exit 4, stale confirmed) + live_fire_head_e2e.txt (1 passed, exit 0) |
| B — Referential | REAL | All artifacts include SHA-pinned refs; claim→artifact map above |
| C — Negative | N/A — scope searched: no other validator.sh in `curricula/cs336_a1/modules/cosine_schedule/` copies `optimizer.py` in the BUILD stage path; the disallowed pattern (`cp cs336_basics/optimizer.py`) is absent from HEAD validator.sh |
| D — Differential | REAL | validator_sh_diff.txt (git diff origin/main); baseline_red.txt vs head_green.txt pair |
| E — Intent | REAL | Finding: https://github.com/ImmortalDemonGod/mastery-engine/blob/7f6610a902befcb84fc47e5c82a161e3d3184ce4/audit/02-static-audit.md#L8 |
| F — Provenance | REAL | SHA256 manifest in this file; artifacts under .github/aiv-packets/evidence/ (untracked, pending orchestrator commit) |
