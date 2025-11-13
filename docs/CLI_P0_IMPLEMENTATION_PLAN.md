# P0 Implementation: Unified Submit Command

**Priority**: P0 (CRITICAL)  
**Issue**: CLI-001 - Command Proliferation  
**Status**: Implementation Ready

---

## Implementation Strategy

### Phase 1: Create Unified Command (Non-Breaking)

**Approach**: Add new `submit` command alongside existing commands

**Benefits**:
- Zero risk to existing users
- Allows incremental testing
- Provides migration path

### Phase 2: Deprecation Warnings

**Approach**: Add warnings to old commands

### Phase 3: Hard Cutover (Future)

**Approach**: Remove deprecated commands in v2.0

---

## Detailed Implementation

### Step 1: Create Helper Functions

Extract common logic from existing submit-* commands:

```python
def _load_state_and_curriculum():
    """Common initialization for all submit commands."""
    state_mgr = StateManager()
    curr_mgr = CurriculumManager()
    progress = state_mgr.load()
    manifest = curr_mgr.load_manifest(progress.curriculum_id)
    return state_mgr, curr_mgr, progress, manifest

def _check_curriculum_complete(progress, manifest):
    """Check if all modules completed."""
    if progress.current_module_index >= len(manifest.modules):
        console.print()
        console.print(Panel(
            "[bold green]All modules completed![/bold green]\n\n"
            "Congratulations! You have finished the curriculum.",
            title="Curriculum Complete",
            border_style="green"
        ))
        console.print()
        return True
    return False
```

### Step 2: Implement Stage-Specific Handlers

```python
def _submit_build_handler(state_mgr, curr_mgr, progress, manifest):
    """Handle Build stage submission."""
    # Extract from existing submit_build() function
    # Lines 322-392 of current main.py
    current_module = manifest.modules[progress.current_module_index]
    workspace_mgr = WorkspaceManager()
    validator_subsys = ValidationSubsystem()
    
    validator_path = curr_mgr.get_validator_path(progress.curriculum_id, current_module)
    workspace_path = Path.cwd()
    
    console.print()
    console.print(f"[bold cyan]Running validator for {current_module.name}...[/bold cyan]")
    console.print()
    
    result = validator_subsys.execute(validator_path, workspace_path)
    
    if result.exit_code == 0:
        console.print(Panel(
            "[bold green]✅ Validation Passed![/bold green]\n\n"
            "Your implementation passed all tests.",
            title="Success",
            border_style="green"
        ))
        
        if result.performance_seconds is not None:
            console.print()
            console.print(f"[dim]Performance: {result.performance_seconds:.3f} seconds[/dim]")
            
            if current_module.baseline_perf_seconds:
                speedup = current_module.baseline_perf_seconds / result.performance_seconds
                if speedup > 2.0:
                    console.print(f"[yellow]⚡ Impressive! {speedup:.1f}x faster than baseline![/yellow]")
        
        console.print()
        
        progress.mark_stage_complete("build")
        state_mgr.save(progress)
        
        logger.info(f"Build stage completed for module '{current_module.id}'")
        
        console.print(Panel(
            "Next step: Answer conceptual questions.\n\n"
            "Run [bold cyan]engine submit[/bold cyan] to open your editor and answer.",
            title="Next Action",
            border_style="blue"
        ))
        console.print()
        return True
    else:
        console.print(Panel(
            "[bold red]❌ Validation Failed[/bold red]\n\n"
            "See test output below:",
            title="Failure",
            border_style="red"
        ))
        console.print()
        console.print("[bold]Test Output:[/bold]")
        console.print(result.stderr if result.stderr else result.stdout)
        console.print()
        return False

def _submit_justify_handler(state_mgr, curr_mgr, progress, manifest):
    """Handle Justify stage submission (with editor)."""
    # New implementation with $EDITOR
    import tempfile
    import os
    
    current_module = manifest.modules[progress.current_module_index]
    justify_runner = JustifyRunner(curr_mgr)
    questions = justify_runner.load_questions(progress.curriculum_id, current_module)
    
    if not questions:
        raise CurriculumInvalidError(f"No justify questions for '{current_module.id}'")
    
    question = questions[0]
    
    # Open editor for answer
    editor = os.getenv('EDITOR', os.getenv('VISUAL', 'nano'))
    
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.md', delete=False) as tf:
        tf.write("# Justify Question\n\n")
        tf.write(f"{question.question}\n\n")
        tf.write("---\n\n")
        tf.write("# Your Answer\n\n")
        tf.write("Write your answer below. Lines starting with # are ignored.\n\n")
        temp_path = tf.name
    
    try:
        subprocess.run([editor, temp_path], check=True)
        
        with open(temp_path, 'r') as f:
            lines = f.readlines()
        
        # Extract answer (skip comment lines and header)
        answer_lines = []
        in_answer = False
        for line in lines:
            if line.strip() == "# Your Answer":
                in_answer = True
                continue
            if in_answer and not line.strip().startswith('#'):
                answer_lines.append(line)
        
        answer = ''.join(answer_lines).strip()
        
        if not answer:
            console.print()
            console.print(Panel(
                "[bold yellow]Empty Answer[/bold yellow]\n\n"
                "You didn't provide an answer. Try again.",
                title="No Answer",
                border_style="yellow"
            ))
            console.print()
            return False
        
    finally:
        os.unlink(temp_path)
    
    # Now validate answer (existing logic from submit-justification)
    console.print()
    console.print(f"[bold cyan]Evaluating your answer...[/bold cyan]")
    console.print()
    
    # Keyword filter + LLM evaluation
    # (existing logic from lines 500-600)
    # ... validation logic ...
    
    # For now, stub:
    console.print("[yellow]Note: Full justify validation not yet integrated with editor[/yellow]")
    return False

def _submit_harden_handler(state_mgr, curr_mgr, progress, manifest):
    """Handle Harden stage submission."""
    # Extract from existing submit_fix() function
    current_module = manifest.modules[progress.current_module_index]
    workspace_mgr = WorkspaceManager()
    validator_subsys = ValidationSubsystem()
    
    shadow_worktree = Path('.mastery_engine_worktree')
    harden_workspace = shadow_worktree / "workspace" / "harden"
    
    if not harden_workspace.exists():
        console.print()
        console.print(Panel(
            "[bold red]Harden Workspace Not Found[/bold red]\n\n"
            "You must first prepare the challenge using [bold cyan]engine start-challenge[/bold cyan].",
            title="ERROR",
            border_style="red"
        ))
        console.print()
        return False
    
    validator_path = curr_mgr.get_validator_path(progress.curriculum_id, current_module)
    
    console.print()
    console.print(f"[bold cyan]Running validator on your fix for {current_module.name}...[/bold cyan]")
    console.print()
    
    if current_module.id == "softmax":
        harden_file = harden_workspace / "utils.py"
        shadow_dest = shadow_worktree / "cs336_basics" / "utils.py"
    else:
        harden_file = harden_workspace / f"{current_module.id}.py"
        shadow_dest = shadow_worktree / f"{current_module.id}.py"
    
    import shutil
    shadow_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(harden_file, shadow_dest)
    
    result = validator_subsys.execute(validator_path, shadow_worktree)
    
    if result.exit_code == 0:
        console.print(Panel(
            "[bold green]✅ Bug Fixed![/bold green]\n\n"
            "Your fix passes all tests.",
            title="Success",
            border_style="green"
        ))
        console.print()
        
        progress.mark_stage_complete("harden")
        state_mgr.save(progress)
        
        logger.info(f"Harden stage completed for module '{current_module.id}'")
        
        if progress.current_module_index < len(manifest.modules):
            console.print(Panel(
                f"Module '{current_module.name}' complete!\n\n"
                "Run [bold cyan]engine show[/bold cyan] to start the next module.",
                title="Next Action",
                border_style="blue"
            ))
        else:
            console.print(Panel(
                "[bold green]🎉 Curriculum Complete![/bold green]",
                title="Congratulations",
                border_style="green"
            ))
        console.print()
        return True
    else:
        console.print(Panel(
            "[bold red]❌ Fix Incomplete[/bold red]\n\n"
            "See test output below:",
            title="Failure",
            border_style="red"
        ))
        console.print()
        console.print("[bold]Test Output:[/bold]")
        console.print(result.stderr if result.stderr else result.stdout)
        console.print()
        return False
```

### Step 3: Implement Unified Command

```python
@app.command()
def submit():
    """
    Submit your work for the current stage (auto-detected).
    
    The engine automatically determines which stage you're in and
    runs the appropriate validation:
    
    - Build stage: Validates your implementation
    - Justify stage: Opens your editor for conceptual answers
    - Harden stage: Validates your bug fix
    """
    try:
        require_shadow_worktree()
        
        state_mgr, curr_mgr, progress, manifest = _load_state_and_curriculum()
        
        if _check_curriculum_complete(progress, manifest):
            return
        
        current_module = manifest.modules[progress.current_module_index]
        stage = progress.current_stage
        
        console.print()
        console.print(f"[dim]Detected stage: {stage.upper()}[/dim]")
        console.print()
        
        # Route to appropriate handler
        if stage == "build":
            success = _submit_build_handler(state_mgr, curr_mgr, progress, manifest)
        elif stage == "justify":
            success = _submit_justify_handler(state_mgr, curr_mgr, progress, manifest)
        elif stage == "harden":
            success = _submit_harden_handler(state_mgr, curr_mgr, progress, manifest)
        else:
            console.print(Panel(
                f"[bold red]Unknown Stage[/bold red]\n\n"
                f"Current stage '{stage}' is not recognized.",
                title="ERROR",
                border_style="red"
            ))
            sys.exit(1)
        
        if success:
            logger.info(f"Successfully completed {stage} stage for module '{current_module.id}'")
    
    except (StateFileCorruptedError, CurriculumNotFoundError, CurriculumInvalidError,
            ValidatorNotFoundError, ValidatorTimeoutError, ValidatorExecutionError,
            JustifyQuestionsError, HardenChallengeError) as e:
        # Handle all known exceptions
        error_type = type(e).__name__.replace('Error', '')
        console.print(Panel(
            f"[bold red]{error_type}[/bold red]\n\n{str(e)}",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error in submit command")
        console.print(Panel(
            f"[bold red]Unexpected Error[/bold red]\n\n{str(e)}\n\n"
            f"Check log: {Path.home() / '.mastery_engine.log'}",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)
```

---

## Testing Plan

### Unit Tests

```python
def test_submit_detects_build_stage():
    """Verify submit routes to build handler when in build stage."""
    # Mock state file with stage="build"
    # Call submit()
    # Assert build validator was called

def test_submit_detects_justify_stage():
    """Verify submit routes to justify handler when in justify stage."""
    # Mock state file with stage="justify"
    # Mock $EDITOR
    # Call submit()
    # Assert editor was opened

def test_submit_detects_harden_stage():
    """Verify submit routes to harden handler when in harden stage."""
    # Mock state file with stage="harden"
    # Call submit()
    # Assert harden validator was called

def test_submit_advances_progress_on_success():
    """Verify submit advances to next stage after successful validation."""
    # Mock successful validation
    # Call submit()
    # Assert progress.current_stage changed
```

### Integration Tests

```python
def test_full_bjh_loop_with_unified_submit():
    """Complete BJH loop using only submit command."""
    # init
    # show (build)
    # submit (build - should pass)
    # show (justify)
    # submit (justify - should open editor)
    # show (harden)
    # start-challenge
    # submit (harden - should pass)
    # Assert module complete
```

---

## Deprecation Strategy

### Phase 1: Add Deprecation Warnings

```python
@app.command("submit-build")
def submit_build():
    """
    [DEPRECATED] Use 'engine submit' instead.
    
    This command will be removed in v2.0.
    """
    console.print()
    console.print(Panel(
        "[bold yellow]Deprecation Warning[/bold yellow]\n\n"
        "The [bold]submit-build[/bold] command is deprecated.\n\n"
        "Please use [bold cyan]engine submit[/bold cyan] instead.\n"
        "The engine will auto-detect your current stage.\n\n"
        "This command will be removed in v2.0.",
        title="Deprecated Command",
        border_style="yellow"
    ))
    console.print()
    
    # Still execute for backward compatibility
    # ... existing logic ...
```

### Phase 2: Documentation Updates

- Update README.md with new workflow
- Create MIGRATION_V2.md guide
- Update all build_prompt.txt "Next Action" panels

### Phase 3: Hard Removal (v2.0)

- Remove submit-build, submit-justification, submit-fix
- Keep only submit
- Update version to 2.0.0

---

## Success Criteria

✅ **Simplicity**: Single `submit` command replaces 3  
✅ **Context-Awareness**: Auto-detects stage from state file  
✅ **Backward Compatible**: Old commands still work (with warnings)  
✅ **Ergonomics**: Editor integration for justify stage  
✅ **Tests Passing**: All unit and integration tests pass  
✅ **Documentation**: Clear migration guide provided

---

## Implementation Checklist

- [ ] Extract common helper functions
- [ ] Implement stage-specific handlers
- [ ] Create unified `submit` command
- [ ] Add comprehensive error handling
- [ ] Implement $EDITOR integration for justify
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Add deprecation warnings to old commands
- [ ] Update documentation
- [ ] Test end-to-end workflow

---

**Status**: Ready for implementation  
**Estimated Effort**: 4-6 hours  
**Next Step**: Begin implementation of helper functions
