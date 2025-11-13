"""
Simple E2E test for just the BUILD stage to verify our fix works.
"""
from pathlib import Path
from tests.e2e.test_complete_bjh_loop import isolated_repo, run_engine_command, get_state
import subprocess


def test_build_stage_passes(isolated_repo: Path):
    """Test that BUILD stage completes successfully with developer mode."""
    
    # Init
    result = run_engine_command(isolated_repo, "init", "cs336_a1")
    assert result.returncode == 0, f"Init failed: {result.stderr}"
    
    state = get_state(isolated_repo)
    assert state["current_stage"] == "build"
    
    # Switch to developer mode
    mode_script = isolated_repo / "scripts" / "mode"
    result = subprocess.run(
        [str(mode_script), "switch", "developer"],
        cwd=isolated_repo,
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    
    # Update shadow worktree symlink
    shadow_worktree = isolated_repo / ".mastery_engine_worktree"
    shadow_symlink = shadow_worktree / "cs336_basics"
    if shadow_symlink.is_symlink():
        shadow_symlink.unlink()
    shadow_symlink.symlink_to("modes/developer/cs336_basics")
    
    # Submit build
    result = run_engine_command(isolated_repo, "submit")
    assert result.returncode == 0, f"Build failed: {result.stderr}"
    assert ("Validation Passed" in result.stdout or "passed" in result.stdout.lower()), \
        f"Build validation didn't pass:\n{result.stdout}"
    
    # Verify state advanced
    state = get_state(isolated_repo)
    assert state["current_stage"] == "justify", f"Expected justify stage, got {state['current_stage']}"
    
    print("\n✅ BUILD STAGE PASSED!")
