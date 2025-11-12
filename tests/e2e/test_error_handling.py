"""
End-to-End Tests for Error Handling and Edge Cases

These tests validate the Mastery Engine's behavior when users make mistakes,
encounter edge cases, or use commands incorrectly. They are the "fortress"
protecting against regression in user-facing error messages and recovery flows.

Test Coverage:
- Commands used without initialization
- Stale Git worktree recovery
- Wrong stage command usage
- Corrupted state file handling
- Missing curriculum detection
- Empty/malformed user inputs
"""

import json
import subprocess
import shutil
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def isolated_repo(tmp_path: Path) -> Generator[Path, None, None]:
    """
    Create a minimal isolated Git repository for testing.
    
    This is a lighter-weight version of the full isolated_repo fixture,
    containing only what's needed for command-level error testing.
    """
    # Get the real repository root
    real_repo = Path(__file__).parent.parent.parent
    
    # Create test repository
    test_repo = tmp_path / "test_repo"
    test_repo.mkdir()
    
    # Initialize Git repository
    subprocess.run(["git", "init"], cwd=test_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=test_repo, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=test_repo, check=True, capture_output=True
    )
    
    # Copy only essential files
    shutil.copytree(real_repo / "engine", test_repo / "engine")
    shutil.copytree(real_repo / "curricula", test_repo / "curricula")
    
    # Copy minimal support files
    if (real_repo / "pyproject.toml").exists():
        shutil.copy2(real_repo / "pyproject.toml", test_repo / "pyproject.toml")
    if (real_repo / "README.md").exists():
        shutil.copy2(real_repo / "README.md", test_repo / "README.md")
    if (real_repo / ".gitignore").exists():
        shutil.copy2(real_repo / ".gitignore", test_repo / ".gitignore")
    
    # Create initial commit
    subprocess.run(["git", "add", "-A"], cwd=test_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=test_repo, check=True, capture_output=True
    )
    
    yield test_repo
    
    # Cleanup shadow worktree if it exists
    shadow_worktree = test_repo / ".mastery_engine_worktree"
    if shadow_worktree.exists():
        try:
            subprocess.run(
                ["git", "worktree", "remove", str(shadow_worktree), "--force"],
                cwd=test_repo,
                capture_output=True
            )
        except subprocess.CalledProcessError:
            pass


def run_engine_command(repo_path: Path, *args: str) -> subprocess.CompletedProcess:
    """Run an engine command in the isolated repository."""
    import sys
    cmd = [sys.executable, "-m", "engine.main"] + list(args)
    return subprocess.run(
        cmd,
        cwd=repo_path,
        capture_output=True,
        text=True
    )


def test_submit_build_without_init(isolated_repo: Path):
    """
    Test that submit-build fails gracefully when engine is not initialized.
    
    This validates Issue #1 fix: users should get clear guidance to run init.
    """
    # Try to submit build without initializing
    result = run_engine_command(isolated_repo, "submit-build")
    
    # Should fail with clear error message
    assert result.returncode == 1
    assert "Not Initialized" in result.stdout
    assert "engine init" in result.stdout
    assert "INITIALIZATION REQUIRED" in result.stdout


def test_status_without_init(isolated_repo: Path):
    """Test that status command fails gracefully when not initialized."""
    result = run_engine_command(isolated_repo, "status")
    
    assert result.returncode == 1
    assert "Not Initialized" in result.stdout
    assert "engine init" in result.stdout


def test_next_without_init(isolated_repo: Path):
    """Test that next command fails gracefully when not initialized."""
    result = run_engine_command(isolated_repo, "next")
    
    assert result.returncode == 1
    assert "Not Initialized" in result.stdout


def test_submit_justification_without_init(isolated_repo: Path):
    """Test that submit-justification fails gracefully when not initialized."""
    result = run_engine_command(isolated_repo, "submit-justification", "test answer")
    
    assert result.returncode == 1
    assert "Not Initialized" in result.stdout


def test_submit_fix_without_init(isolated_repo: Path):
    """Test that submit-fix fails gracefully when not initialized."""
    result = run_engine_command(isolated_repo, "submit-fix")
    
    assert result.returncode == 1
    assert "Not Initialized" in result.stdout


def test_stale_worktree_auto_recovery(isolated_repo: Path):
    """
    Test that init auto-recovers from stale worktree registration.
    
    This validates Issue #3 fix: auto-prune prevents user confusion.
    """
    # Simulate stale worktree: register it with git but remove the directory
    shadow_dir = isolated_repo / ".mastery_engine_worktree"
    
    # Create and register worktree
    subprocess.run(
        ["git", "worktree", "add", str(shadow_dir), "--detach"],
        cwd=isolated_repo,
        check=True,
        capture_output=True
    )
    
    # Remove the directory (simulating manual deletion)
    shutil.rmtree(shadow_dir)
    
    # Try to init - should auto-prune and succeed
    result = run_engine_command(isolated_repo, "init", "cs336_a1")
    
    # Should succeed (auto-prune handles stale registration)
    assert result.returncode == 0, f"Init failed with stale worktree: {result.stdout}\n{result.stderr}"
    assert "Initialization Complete" in result.stdout
    assert shadow_dir.exists(), "Shadow worktree should be created"


def test_init_with_nonexistent_curriculum(isolated_repo: Path):
    """Test that init fails gracefully with clear error for invalid curriculum."""
    result = run_engine_command(isolated_repo, "init", "nonexistent_curriculum_xyz")
    
    assert result.returncode == 1
    assert "Invalid Curriculum" in result.stdout or "not found" in result.stdout
    assert "nonexistent_curriculum_xyz" in result.stdout


def test_init_with_dirty_git_state(isolated_repo: Path):
    """Test that init requires clean Git working directory."""
    # Create an uncommitted file
    (isolated_repo / "dirty_file.txt").write_text("uncommitted change")
    
    result = run_engine_command(isolated_repo, "init", "cs336_a1")
    
    assert result.returncode == 1
    assert "Uncommitted Changes Detected" in result.stdout
    assert "git commit" in result.stdout or "git stash" in result.stdout


def test_double_init_prevention(isolated_repo: Path):
    """Test that init cannot be run twice without cleanup."""
    # First init should succeed
    result1 = run_engine_command(isolated_repo, "init", "cs336_a1")
    assert result1.returncode == 0
    
    # Second init should fail with clear message
    result2 = run_engine_command(isolated_repo, "init", "cs336_a1")
    assert result2.returncode == 1
    assert "Already Initialized" in result2.stdout
    assert "engine cleanup" in result2.stdout


def test_wrong_stage_command_usage(isolated_repo: Path):
    """Test that commands fail gracefully when used in wrong stage."""
    # Initialize and get to build stage
    run_engine_command(isolated_repo, "init", "cs336_a1")
    
    # Try to submit justification in build stage
    result = run_engine_command(isolated_repo, "submit-justification", "test")
    
    assert result.returncode == 0  # Command runs but rejects action
    assert "Not in Justify Stage" in result.stdout or "Wrong Stage" in result.stdout
    assert "engine status" in result.stdout  # Suggests how to check current stage


def test_empty_justification_rejection(isolated_repo: Path):
    """Test that empty answers are rejected with helpful message."""
    # Set up state file for justify stage
    run_engine_command(isolated_repo, "init", "cs336_a1")
    
    state_file = Path.home() / ".mastery_progress.json"
    state = {
        "curriculum_id": "cs336_a1",
        "current_module_index": 0,
        "current_stage": "justify",
        "completed_modules": []
    }
    with open(state_file, 'w') as f:
        json.dump(state, f)
    
    # Try to submit empty answer
    result = run_engine_command(isolated_repo, "submit-justification", "")
    
    assert result.returncode == 0  # Command runs
    assert "Empty Answer" in result.stdout or "non-empty" in result.stdout


def test_cleanup_when_not_initialized(isolated_repo: Path):
    """Test that cleanup handles gracefully when nothing to clean."""
    result = run_engine_command(isolated_repo, "cleanup")
    
    # Should succeed gracefully
    assert result.returncode == 0
    assert "No shadow worktree found" in result.stdout or "Nothing to clean" in result.stdout


def test_init_cleanup_init_cycle(isolated_repo: Path):
    """Test full lifecycle: init -> cleanup -> init again."""
    # First init
    result1 = run_engine_command(isolated_repo, "init", "cs336_a1")
    assert result1.returncode == 0
    assert (isolated_repo / ".mastery_engine_worktree").exists()
    
    # Cleanup
    result2 = run_engine_command(isolated_repo, "cleanup")
    assert result2.returncode == 0
    assert not (isolated_repo / ".mastery_engine_worktree").exists()
    
    # Second init should work
    result3 = run_engine_command(isolated_repo, "init", "cs336_a1")
    assert result3.returncode == 0
    assert (isolated_repo / ".mastery_engine_worktree").exists()


def test_state_file_corruption_handling(isolated_repo: Path):
    """Test that corrupted state file is handled with clear error."""
    # Initialize normally
    run_engine_command(isolated_repo, "init", "cs336_a1")
    
    # Corrupt the state file
    state_file = Path.home() / ".mastery_progress.json"
    state_file.write_text("{ this is not valid json {{{")
    
    # Try to run status
    result = run_engine_command(isolated_repo, "status")
    
    assert result.returncode == 1
    assert "State File Corrupted" in result.stdout or "corrupted" in result.stdout.lower()
    assert str(state_file) in result.stdout  # Shows path for recovery
