"""
Debug script to inspect shadow worktree structure during E2E test.
Run this to understand why pytest can't collect tests.
"""
import subprocess
import sys
from pathlib import Path
from tests.e2e.test_complete_bjh_loop import isolated_repo, run_engine_command, get_state

def debug_shadow_worktree(isolated_repo: Path):
    """Inspect shadow worktree structure after init."""
    
    # Initialize engine
    result = run_engine_command(isolated_repo, "init", "cs336_a1")
    assert result.returncode == 0, f"Init failed: {result.stderr}"
    
    shadow_worktree = isolated_repo / ".mastery_engine_worktree"
    
    print("\n=== MAIN REPO STRUCTURE ===")
    print(f"Main repo: {isolated_repo}")
    print(f"cs336_basics exists: {(isolated_repo / 'cs336_basics').exists()}")
    print(f"cs336_basics is symlink: {(isolated_repo / 'cs336_basics').is_symlink()}")
    if (isolated_repo / 'cs336_basics').is_symlink():
        target = (isolated_repo / 'cs336_basics').readlink()
        print(f"cs336_basics -> {target}")
    
    print("\n=== SHADOW WORKTREE STRUCTURE ===")
    print(f"Shadow worktree: {shadow_worktree}")
    print(f"cs336_basics exists: {(shadow_worktree / 'cs336_basics').exists()}")
    print(f"cs336_basics is symlink: {(shadow_worktree / 'cs336_basics').is_symlink()}")
    if (shadow_worktree / 'cs336_basics').is_symlink():
        target = (shadow_worktree / 'cs336_basics').readlink()
        print(f"cs336_basics -> {target}")
    
    # List what's actually in shadow worktree
    print("\n=== FILES IN SHADOW WORKTREE ===")
    for item in sorted(shadow_worktree.iterdir()):
        if item.name.startswith('.'):
            continue
        print(f"  {item.name} ({'dir' if item.is_dir() else 'symlink' if item.is_symlink() else 'file'})")
    
    # Try to see if cs336_basics is importable from shadow worktree
    print("\n=== PYTHON IMPORT TEST ===")
    test_script = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
print(f"CWD: {Path.cwd()}")
print(f"sys.path[0]: {sys.path[0]}")
try:
    import cs336_basics
    print(f"✅ cs336_basics imported: {cs336_basics.__file__}")
except ImportError as e:
    print(f"❌ Import failed: {e}")
"""
    result = subprocess.run(
        [sys.executable, "-c", test_script],
        cwd=shadow_worktree,
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr}")
    
    # Try to run pytest
    print("\n=== PYTEST TEST ===")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_nn_utils.py::test_softmax_matches_pytorch", "--collect-only"],
        cwd=shadow_worktree,
        capture_output=True,
        text=True,
        env={**subprocess.os.environ, "PYTHONPATH": str(shadow_worktree)}
    )
    print(f"Exit code: {result.returncode}")
    print(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        print(f"STDERR:\n{result.stderr}")


if __name__ == "__main__":
    import pytest
    
    # Use pytest's tmp_path fixture
    pytest.main([__file__, "-v", "-s"])
