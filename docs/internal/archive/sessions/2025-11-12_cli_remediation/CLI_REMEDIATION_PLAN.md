# CLI Interface Remediation Plan

**Date**: 2025-11-12  
**Scope**: Engine CLI Interface (`engine/main.py`)  
**Status**: Planning Complete, Ready for Execution

---

## Executive Summary

This document provides a systematic remediation plan for the Mastery Engine CLI interface based on comprehensive analysis against established CLI design principles. The current CLI is a functional MVP but exhibits several sub-optimal design choices that create unnecessary cognitive load and violate the Principle of Least Surprise.

**Goal**: Transform the CLI from a "functional MVP" to an "optimal interface" through targeted refactoring that improves simplicity, predictability, and ergonomics while maintaining all existing strengths.

---

## Priority Classification

Issues are classified by impact on user experience and violation severity:

- **P0 (CRITICAL)**: Fundamental design flaws causing significant cognitive load
- **P1 (HIGH)**: Serious UX issues violating established principles
- **P2 (MEDIUM)**: Quality-of-life improvements
- **P3 (LOW)**: Nice-to-have enhancements

---

## Issue Inventory

### P0 (CRITICAL) - Command Proliferation

**Issue ID**: CLI-001  
**Category**: Context-Awareness  
**Severity**: CRITICAL

**Problem**:
Students must use three different commands for conceptually the same action:
- `engine submit-build` (Build stage)
- `engine submit-justification` (Justify stage)
- `engine submit-fix` (Harden stage)

The engine already tracks current stage in `~/.mastery_progress.json`. Forcing users to specify the stage manually adds unnecessary cognitive load and creates opportunities for error.

**Current Code** (`engine/main.py`):
```python
@app.command("submit-build")
def submit_build():
    """Submit your implementation for the Build stage."""
    ...

@app.command("submit-justification")
def submit_justification(answer: str):
    """Submit your justification answer."""
    ...

@app.command("submit-fix")
def submit_fix():
    """Submit your bug fix for the Harden stage."""
    ...
```

**Impact**:
- User must remember 3 commands instead of 1
- Error-prone (running wrong submit-* for current stage)
- Violates DRY principle (duplicate validation logic)

**Proposed Solution**:
Single context-aware `submit` command:
```python
@app.command("submit")
def submit():
    """Submit your work for the current stage (auto-detected)."""
    progress = state_mgr.load()
    
    if progress.current_stage == "build":
        # Run build validation
    elif progress.current_stage == "justify":
        # Run justify validation (with editor)
    elif progress.current_stage == "harden":
        # Run harden validation
```

**Acceptance Criteria**:
- ✅ Single `engine submit` command works for all stages
- ✅ Auto-detects current stage from state file
- ✅ Provides stage-specific validation
- ✅ Clear error if called in invalid state
- ✅ Backward compatibility maintained during transition

**Estimated Effort**: 4-6 hours

---

### P1 (HIGH) - Inconsistent Command Behavior

**Issue ID**: CLI-002  
**Category**: Principle of Least Surprise  
**Severity**: HIGH

**Problem**:
The `engine next` command has dual, inconsistent behavior:
- **Build/Justify stages**: Read-only (displays prompt/question)
- **Harden stage**: Write operation (modifies files by applying bug patch)

A command named "next" or "show" should be idempotent with no side effects. Users expecting to re-read the bug symptom may be surprised to find their partial fix overwritten.

**Current Code** (`engine/main.py`):
```python
@app.command("next")
def show_next():
    """Display the next challenge in your current stage."""
    # ... reads state ...
    
    if progress.current_stage == "build":
        prompt_path = curr_mgr.get_build_prompt_path(...)
        # Read-only: display prompt
        
    elif progress.current_stage == "justify":
        # Read-only: display question
        
    elif progress.current_stage == "harden":
        # WRITE OPERATION: applies bug patch!
        harden_runner.present_challenge(...)
```

**Impact**:
- Unpredictable behavior (sometimes safe, sometimes destructive)
- Loss of work (partial fixes overwritten on re-run)
- Violates user trust and expectations

**Proposed Solution**:
Split into two commands:

1. **`engine show`** - Always read-only
```python
@app.command("show")
def show(module_id: Optional[str] = None):
    """Display prompt/question for current or specified module (read-only)."""
    # Always safe, never modifies files
```

2. **`engine start-challenge`** - Explicit write action
```python
@app.command("start-challenge")
def start_challenge():
    """Prepare the Harden stage workspace (creates buggy file)."""
    # Only works in Harden stage
    # Explicit write action with clear intent
```

**Acceptance Criteria**:
- ✅ `engine show` is guaranteed read-only (idempotent)
- ✅ `engine start-challenge` is explicit write action
- ✅ Clear error messages if commands used in wrong stage
- ✅ `engine next` deprecated with helpful migration message

**Estimated Effort**: 3-4 hours

---

### P1 (HIGH) - Poor Multi-Line Input Ergonomics

**Issue ID**: CLI-003  
**Category**: Ergonomics  
**Severity**: HIGH

**Problem**:
The `submit-justification` command requires passing entire answer as CLI argument:
```bash
engine submit-justification "A long answer with \"quotes\" to escape..."
```

This is extremely poor UX for multi-line text, forcing users to wrestle with shell quoting. The universally accepted pattern is to open the user's editor (like `git commit`).

**Current Code** (`engine/main.py`):
```python
@app.command("submit-justification")
def submit_justification(answer: str):
    """Submit your justification answer."""
    # answer comes from CLI argument - awkward for multi-line
```

**Impact**:
- Hostile UX for the pedagogical goal (detailed explanations)
- Shell quoting/escaping frustration
- Discourages well-formatted answers

**Proposed Solution**:
Use `$EDITOR` pattern (like `git commit`):
```python
import tempfile
import subprocess
import os

def get_justification_answer():
    """Open user's editor for multi-line answer."""
    editor = os.getenv('EDITOR', 'nano')
    
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as tf:
        tf.write("# Enter your justification answer below\n")
        tf.write("# Lines starting with # are ignored\n\n")
        temp_path = tf.name
    
    subprocess.run([editor, temp_path])
    
    with open(temp_path, 'r') as f:
        answer = '\n'.join(line for line in f if not line.startswith('#'))
    
    os.unlink(temp_path)
    return answer.strip()
```

**Acceptance Criteria**:
- ✅ `engine submit` (justify stage) opens $EDITOR
- ✅ Supports EDITOR environment variable
- ✅ Falls back to sensible default (nano/vim)
- ✅ Strips comment lines from answer
- ✅ Clear instructions in template

**Estimated Effort**: 2-3 hours

---

### P2 (MEDIUM) - Lack of Curriculum Introspection

**Issue ID**: CLI-004  
**Category**: Discoverability  
**Severity**: MEDIUM

**Problem**:
No way for students to:
- See all modules in curriculum
- Review past module content
- Understand their progress visually

The `status` command shows current position but not the full curriculum structure.

**Current Limitation**:
- Can't run `engine list` to see all modules
- Can't run `engine show <module_id>` to review completed modules
- Limited visibility into overall journey

**Proposed Solution**:
Add introspection commands:

1. **`engine curriculum list`**
```python
@app.command("list")
def list_modules():
    """List all modules in the current curriculum."""
    # Display table with: ID, Name, Status (✅/🔵/⚪), Dependencies
```

2. **`engine curriculum show <module_id>`**
```python
@app.command("show")
def show_module(module_id: str):
    """Display full content for a specific module."""
    # Show: build_prompt, justify_questions, bug descriptions
```

**Acceptance Criteria**:
- ✅ `engine curriculum list` shows all modules with status
- ✅ Status indicators: ✅ Complete, 🔵 In Progress, ⚪ Not Started
- ✅ `engine curriculum show <id>` displays full module content
- ✅ Works for both completed and future modules (preview)

**Estimated Effort**: 3-4 hours

---

### P2 (MEDIUM) - Incomplete Module Reset

**Issue ID**: CLI-005  
**Category**: Flexibility  
**Severity**: MEDIUM

**Problem**:
The `engine reset` command is partially implemented with placeholder message:
> "[bold yellow]Module Reset Not Yet Implemented[/bold yellow]"

Students cannot re-attempt modules to solidify understanding.

**Current Code** (`engine/main.py`):
```python
@app.command("reset")
def reset(
    module_id: Optional[str] = None,
    hard: bool = typer.Option(False, "--hard"),
):
    if module_id is not None:
        console.print("[bold yellow]Module Reset Not Yet Implemented[/bold yellow]")
        raise typer.Exit(1)
```

**Proposed Solution**:
Implement full module reset:
```python
@app.command("reset")
def reset_module(module_id: str):
    """Reset progress for a specific module to start over."""
    # 1. Validate module_id exists
    # 2. Confirm with user (interactive prompt)
    # 3. Update state: remove from completed, set as current
    # 4. Clean up workspace for that module
    # 5. Set stage to "build"
```

**Acceptance Criteria**:
- ✅ `engine progress reset <module_id>` fully functional
- ✅ Interactive confirmation (prevent accidental reset)
- ✅ Cleans workspace appropriately
- ✅ Updates state file correctly
- ✅ Clear success/error messages

**Estimated Effort**: 2-3 hours

---

## Implementation Strategy

### Phase 1: Critical Path (P0)
**Goal**: Fix fundamental design flaw (command proliferation)

1. Implement unified `submit` command
2. Add stage detection logic
3. Migrate validation routing
4. Test all three stages
5. Add deprecation warnings for old commands

**Duration**: 1 session (~4-6 hours)

### Phase 2: Safety and Ergonomics (P1)
**Goal**: Fix Principle of Least Surprise violations

1. Split `next` into `show` and `start-challenge`
2. Implement editor-based justification input
3. Update all stage logic
4. Test idempotency and safety

**Duration**: 1 session (~5-7 hours)

### Phase 3: Introspection (P2)
**Goal**: Add discoverability and flexibility

1. Implement `curriculum list`
2. Implement `curriculum show <module_id>`
3. Complete `progress reset <module_id>`
4. Add comprehensive tests

**Duration**: 1 session (~5-7 hours)

---

## Testing Requirements

### Unit Tests
- [ ] `test_submit_auto_detects_stage()`
- [ ] `test_submit_build_validation()`
- [ ] `test_submit_justify_with_editor()`
- [ ] `test_submit_harden_validation()`
- [ ] `test_show_is_idempotent()`
- [ ] `test_start_challenge_only_in_harden()`
- [ ] `test_curriculum_list_all_modules()`
- [ ] `test_curriculum_show_module()`
- [ ] `test_progress_reset_module()`

### Integration Tests
- [ ] Complete BJH loop with new commands
- [ ] Editor integration (mock $EDITOR)
- [ ] State file updates
- [ ] Error handling for all edge cases

### Backward Compatibility
- [ ] Old commands still work (deprecated)
- [ ] Clear migration messages
- [ ] No breaking changes to state file format

---

## Rollout Plan

### Phase 1: Soft Launch
- Implement new commands alongside old
- Add deprecation warnings to old commands
- Update documentation with migration guide

### Phase 2: Transition Period
- Monitor usage
- Collect user feedback
- Fix any discovered issues

### Phase 3: Hard Cutover
- Remove deprecated commands
- Update all documentation
- Release as "Engine v2.0"

---

## Success Metrics

### Simplicity
- **Before**: 5 core loop commands (init, status, next, submit-build, submit-justification, submit-fix)
- **After**: 3 core loop commands (init, status, show, submit, start-challenge)
- **Improvement**: 40% reduction in commands to memorize

### Predictability
- **Before**: `next` has inconsistent side effects
- **After**: All commands have consistent, predictable behavior
- **Metric**: Zero unexpected file modifications

### Ergonomics
- **Before**: Multi-line justifications via CLI argument
- **After**: Professional editor experience
- **Metric**: Student satisfaction survey

### Discoverability
- **Before**: No way to explore curriculum
- **After**: Full introspection via `curriculum` subcommands
- **Metric**: Reduced "how do I...?" support requests

---

## Risk Assessment

### Low Risk
- Adding new commands (non-breaking)
- Improving existing command behavior

### Medium Risk
- Changing `next` behavior (may surprise existing users)
- **Mitigation**: Clear migration guide, deprecation warnings

### High Risk
- Breaking backward compatibility
- **Mitigation**: Maintain old commands during transition, version bump

---

## Documentation Updates Required

1. **README.md**: Update command examples
2. **CLI_GUIDE.md**: Create comprehensive CLI reference
3. **MIGRATION_V2.md**: Guide for transitioning from v1.0
4. **engine/main.py**: Update all docstrings

---

## Tracking

| Task | Priority | Status | Estimated Hours | Completion |
|:---|:---:|:---:|:---:|:---:|
| Unified `submit` command | P0 | 🔴 Not Started | 4-6 | 0% |
| Split `next` into `show`/`start-challenge` | P1 | 🔴 Not Started | 3-4 | 0% |
| Editor-based justification input | P1 | 🔴 Not Started | 2-3 | 0% |
| `curriculum list` command | P2 | 🔴 Not Started | 2-3 | 0% |
| `curriculum show <module>` command | P2 | 🔴 Not Started | 1-2 | 0% |
| Complete `progress reset` | P2 | 🔴 Not Started | 2-3 | 0% |
| Unit tests | All | 🔴 Not Started | 4-6 | 0% |
| Integration tests | All | 🔴 Not Started | 3-4 | 0% |
| Documentation updates | All | 🔴 Not Started | 2-3 | 0% |

**Total Estimated Effort**: 23-37 hours (3-5 sessions)

---

**Last Updated**: 2025-11-12  
**Status**: Plan Created - Ready for Execution  
**Next Action**: Begin P0 - Implement unified `submit` command
