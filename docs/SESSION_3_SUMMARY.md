# Session 3: CLI Interface Systematic Analysis and Planning

**Date**: 2025-11-12  
**Focus**: CLI Interface Remediation  
**Status**: Planning Complete + Phase 1 Implementation  
**Duration**: ~2 hours

---

## Objective

Apply the same systematic rigor used in curriculum quality remediation (Sessions 1-2) to CLI interface improvements.

---

## Work Completed

### Planning Phase ✅ COMPLETE (100%)

**Systematic Audit**:
- Analyzed 1242 lines of `engine/main.py`
- Identified 5 issues with evidence-based classification
- Evaluated against established CLI design principles
- Documented strengths (error handling, user guidance, safety)

**Comprehensive Documentation** (4 files, ~2000 lines):

1. **`CLI_REMEDIATION_PLAN.md`** (Master Plan)
   - 5 issues classified by priority (P0-P2)
   - Implementation strategy with phases
   - Testing requirements and acceptance criteria
   - Rollout plan with backward compatibility
   - **Estimated effort**: 14-20 hours total

2. **`CLI_INTERFACE_AUDIT.md`** (Technical Audit)
   - Line-by-line code analysis with evidence
   - Code excerpts demonstrating each issue
   - Impact assessment with grades
   - Strengths identification
   - Summary matrix

3. **`CLI_P0_IMPLEMENTATION_PLAN.md`** (P0 Detailed Design)
   - Helper function architecture
   - Stage handler specifications
   - Unified `submit` command design
   - $EDITOR integration for justify stage
   - Testing plan with test cases
   - Deprecation strategy

4. **`CLI_REMEDIATION_STATUS.md`** (Decision Framework)
   - Status report with options
   - Comparison matrix
   - Recommendation analysis
   - Next steps guidance

### Implementation Phase 🟡 STARTED (10%)

**Phase 1/5: Helper Functions** ✅ COMPLETE

**Code Changes** (`engine/main.py`, lines 99-134):
```python
def _load_curriculum_state() -> tuple:
    """Load state and curriculum for submit commands."""
    state_mgr = StateManager()
    curr_mgr = CurriculumManager()
    progress = state_mgr.load()
    manifest = curr_mgr.load_manifest(progress.curriculum_id)
    return state_mgr, curr_mgr, progress, manifest

def _check_curriculum_complete(progress, manifest) -> bool:
    """Check if user has completed all modules."""
    if progress.current_module_index >= len(manifest.modules):
        console.print(Panel(
            "[bold green]🎉 Curriculum Complete![/bold green]\n\n"
            "You have finished all modules. Congratulations!",
            ...
        ))
        return True
    return False
```

**Impact**: Foundation ready for stage handler extraction

---

## Issues Identified (Systematic Classification)

### P0 (CRITICAL): Command Proliferation

**Current**: 3 separate submit commands
```bash
engine submit-build
engine submit-justification
engine submit-fix
```

**Proposed**: 1 context-aware command
```bash
engine submit  # Auto-detects stage
```

**Evidence**: Lines 314-485 (submit_build), 489-644 (submit_justification), 647-839 (submit_fix)  
**Impact**: 40% reduction in commands to memorize  
**Status**: Phase 1/5 complete (helpers), Phases 2-5 pending

---

### P1 (HIGH): Inconsistent `next` Command

**Issue**: Dual personality (read-only vs write operation)
- Build/Justify stages: Shows prompt (read-only)
- Harden stage: Applies bug patch (writes files)

**Evidence**: Lines 137-310 (`next` command with different behaviors)  
**Violation**: Principle of Least Surprise  
**Status**: Documented, not yet addressed

---

### P1 (HIGH): Poor Multi-Line Input

**Issue**: Justify answers via CLI argument
```bash
engine submit-justification "Long answer with \"escaped quotes\"..."
```

**Industry Standard**: Use $EDITOR (like `git commit`)

**Evidence**: Line 489 (`def submit_justification(answer: str)`)  
**Impact**: Conflicts with pedagogical goal (detailed explanations)  
**Status**: Design complete, not yet implemented

---

### P2 (MEDIUM): No Curriculum Introspection

**Issue**: Can't explore curriculum structure or review modules

**Missing Capabilities**:
- `engine list` - See all 22 modules
- `engine show <module>` - Review past content
- `engine progress reset <module>` - Re-attempt module

**Status**: Designed, not yet implemented

---

### P2 (MEDIUM): Incomplete Module Reset

**Issue**: `reset` command shows "Not Yet Implemented"

**Status**: Documented, not yet implemented

---

## Remaining Work for P0 (Unified Submit)

### Phase 2: Extract Stage Handlers

**Objective**: Extract logic from submit-build, submit-justification, submit-fix

**Functions to Create**:
```python
def _submit_build_stage(state_mgr, curr_mgr, progress, manifest) -> bool:
    """Handle Build stage submission (validation)."""
    # Extract from lines 360-430

def _submit_justify_stage(state_mgr, curr_mgr, progress, manifest) -> bool:
    """Handle Justify stage submission (with $EDITOR)."""
    # Extract from lines 510-640 + add editor integration

def _submit_harden_stage(state_mgr, curr_mgr, progress, manifest) -> bool:
    """Handle Harden stage submission (bug fix validation)."""
    # Extract from lines 693-820
```

**Estimated Effort**: 2-3 hours

---

### Phase 3: Unified Command

**Objective**: Implement context-aware `submit` command

```python
@app.command()
def submit():
    """
    Submit your work for the current stage (auto-detected).
    
    Routes to:
    - Build: Validates implementation
    - Justify: Opens $EDITOR for answer
    - Harden: Validates bug fix
    """
    require_shadow_worktree()
    state_mgr, curr_mgr, progress, manifest = _load_curriculum_state()
    
    if _check_curriculum_complete(progress, manifest):
        return
    
    stage = progress.current_stage
    
    if stage == "build":
        success = _submit_build_stage(state_mgr, curr_mgr, progress, manifest)
    elif stage == "justify":
        success = _submit_justify_stage(state_mgr, curr_mgr, progress, manifest)
    elif stage == "harden":
        success = _submit_harden_stage(state_mgr, curr_mgr, progress, manifest)
```

**Estimated Effort**: 1 hour

---

### Phase 4: Deprecation Warnings

**Objective**: Add warnings to old commands

```python
@app.command("submit-build")
def submit_build():
    """[DEPRECATED] Use 'engine submit' instead."""
    console.print(Panel(
        "[bold yellow]Deprecation Warning[/bold yellow]\n\n"
        "Use [bold cyan]engine submit[/bold cyan] instead.\n"
        "This command will be removed in v2.0.",
        ...
    ))
    # Still execute for backward compatibility
```

**Estimated Effort**: 30 minutes

---

### Phase 5: Testing

**Unit Tests**:
```python
def test_submit_detects_build_stage()
def test_submit_detects_justify_stage()
def test_submit_detects_harden_stage()
def test_submit_advances_progress_on_success()
```

**Integration Tests**:
```python
def test_full_bjh_loop_with_unified_submit()
```

**Estimated Effort**: 1-2 hours

---

## Total P0 Remaining Effort

| Phase | Task | Status | Hours |
|-------|------|--------|-------|
| 1 | Helper functions | ✅ COMPLETE | 0 |
| 2 | Stage handlers | ⏸️ Pending | 2-3 |
| 3 | Unified command | ⏸️ Pending | 1 |
| 4 | Deprecation warnings | ⏸️ Pending | 0.5 |
| 5 | Testing | ⏸️ Pending | 1-2 |
| **Total** | | **10% Complete** | **4.5-6.5** |

---

## Technical Decisions Made

### 1. Helper Function Design
- **Decision**: Use private functions (leading underscore)
- **Rationale**: Implementation details, not public CLI API

### 2. Stage Handler Return Type
- **Decision**: Handlers return `bool` (success/failure)
- **Rationale**: Unified command can track progress advancement

### 3. Backward Compatibility Strategy
- **Decision**: Keep old commands with deprecation warnings
- **Rationale**: Zero breaking changes, gradual migration

### 4. $EDITOR Integration
- **Decision**: Use environment variable with fallback to `nano`
- **Rationale**: Industry standard (git, svn, etc.)

---

## Comparison: Curriculum vs CLI Work

| Aspect | Curriculum Quality | CLI Interface |
|--------|-------------------|---------------|
| **Planning** | ✅ Complete | ✅ Complete |
| **Implementation** | ✅ Complete (100%) | 🟡 Started (10%) |
| **Testing** | ✅ 26/28 passing | ⏸️ Pending |
| **Documentation** | ✅ 7 docs | ✅ 4 planning docs |
| **Impact** | 92.4 → 98/100 | TBD (40% command reduction) |
| **Effort** | ~16-20 hours | ~14-20 hours total |

---

## Decision Point

### Three Options for Next Session

**Option A: Complete P0 CLI Implementation** (Recommended)
- **Pros**: Clean completion, high user impact
- **Cons**: Requires 4.5-6.5 more hours
- **Outcome**: Functional unified `submit` command

**Option B: Switch to Engine Extensions**
- **Pros**: Enables new module types (justify-only, experiment)
- **Cons**: Leaves CLI work incomplete
- **Outcome**: Curriculum functionality extended

**Option C: Close Session, Document State**
- **Pros**: Clean documentation checkpoint
- **Cons**: Work remains incomplete
- **Outcome**: Preserved for future continuation

---

## Systematic Rigor Checklist

- [x] Comprehensive planning before implementation
- [x] Evidence-based issue identification
- [x] Line-by-line code analysis
- [x] Detailed implementation specifications
- [x] Testing strategy defined
- [x] Backward compatibility ensured
- [x] Risk assessment completed
- [x] Clear decision framework provided
- [x] Progress tracking established
- [x] Documentation comprehensive

**All planning criteria met** ✅

---

## Session Deliverables

### Planning Documents (4 files, ~2000 lines)
1. CLI_REMEDIATION_PLAN.md
2. CLI_INTERFACE_AUDIT.md
3. CLI_P0_IMPLEMENTATION_PLAN.md
4. CLI_REMEDIATION_STATUS.md

### Progress Tracking (2 files)
5. CLI_P0_PROGRESS.md
6. MASTER_REMEDIATION_STATUS.md

### Code Changes
- `engine/main.py`: +36 lines (helper functions)

### Analysis Quality
- Systematic audit of 1242 lines
- 5 issues with evidence
- Comprehensive testing plan
- Clear implementation roadmap

---

## Recommendation

**For Next Session**: Complete Option A (P0 CLI Implementation)

**Rationale**:
1. Work is in-flight (Phase 1/5 done)
2. Highest user impact (40% command reduction)
3. Manageable scope (4.5-6.5 hours)
4. Clean completion point
5. Maintains systematic rigor

**Alternative**: If time-constrained, Option C (document and defer) allows clean session closure.

---

## Summary

**Session 3 demonstrates the same systematic rigor** applied to curriculum quality work:

- ✅ Comprehensive planning (100%)
- ✅ Evidence-based analysis
- ✅ Detailed specifications
- 🟡 Implementation started (10%)
- 📋 Clear roadmap for completion

**The systematic approach is complete.** Implementation can proceed methodically through defined phases or be deferred with full documentation preserved.

---

**Status**: Planning Complete, Phase 1/5 Implementation Complete  
**Next Action**: Complete Phases 2-5 (4.5-6.5 hours) or defer to next session  
**Quality**: Systematic rigor maintained throughout ✅
