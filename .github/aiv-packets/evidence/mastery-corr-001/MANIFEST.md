# CORR-001 Evidence Manifest

Finding: `engine/schemas.py:168` — `mark_stage_complete()` appends synthetic `f"module_{self.current_module_index}"` instead of caller-supplied `module_id`  
Audit URL: https://github.com/ImmortalDemonGod/mastery-engine/blob/7f6610a902befcb84fc47e5c82a161e3d3184ce4/audit/02-static-audit.md#L17  
Baseline SHA: `7f6610a902befcb84fc47e5c82a161e3d3184ce4` (origin/main)  
HEAD SHA: `ea4a3540abc16e6274681a599273150ae9b962ff` (fix/mastery-corr-001)  
Evidence dir: `.github/aiv-packets/evidence/mastery-corr-001/`

## Artifacts

| File | sha256 | Claim proven | Baseline ref | AIV class |
|---|---|---|---|---|
| `baseline_red.txt` | `e5eca4ea3904c817e91b37e86fcd2e488495fbbbbb3dab327c5b97910807065f` | Defect EXISTS at `7f6610a`: 5 BUG-A/C/D tests FAIL with `TypeError: got unexpected keyword argument 'module_id'` | `7f6610a` (origin/main worktree at `/tmp/mastery-corr-001_base`) | A (execution), D (differential before) |
| `head_green.txt` | `9b0ecfe862ed92f35f66c12536c136ec05ee96657fc9d425804e264871fe5818` | Fix is PRESENT at HEAD: all 10 tests PASS against `engine/schemas.py:168` | HEAD `ea4a354` | A (execution), D (differential after) |
| `schemas_diff_base_to_head.txt` | `87f83029ee6ebedaa924d25decbe0062f90c5d6bf6547dcde3ebdb2872615a2d` | Static diff binding the synthetic-ID removal + `module_id` param addition to the exact lines cited in the finding | `7f6610a` → `ea4a354` | B (referential), D (differential) |

## Change at finding location

**Before** (`7f6610a:engine/schemas.py:156,168`):
```python
def mark_stage_complete(self, stage: str) -> None:          # no module_id param
    ...
    elif stage == "harden":
        module_id = f"module_{self.current_module_index}"   # synthetic — BUG
        if module_id not in self.completed_modules:
            self.completed_modules.append(module_id)
```

**After** (`ea4a354:engine/schemas.py:156,168`):
```python
def mark_stage_complete(self, stage: str, module_id: Optional[str] = None) -> None:
    ...
    elif stage == "harden":
        if module_id is None:
            raise ValueError("module_id is required for harden stage")
        if module_id not in self.completed_modules:
            self.completed_modules.append(module_id)        # real id stored
```
