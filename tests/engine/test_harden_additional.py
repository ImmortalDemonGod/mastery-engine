"""Additional tests for engine/stages/harden.py to increase coverage."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from engine.stages.harden import HardenRunner, HardenChallengeError
from engine.schemas import ModuleMetadata, ProblemMetadata
from engine.workspace import WorkspaceError


class TestHardenRunnerEdgeCases:
    """Tests for HardenRunner edge cases and error paths."""
    
    def test_select_bug_picks_random_bug(self, tmp_path):
        """Should randomly select a bug file."""
        mock_curr_mgr = MagicMock()
        mock_workspace_mgr = MagicMock()
        runner = HardenRunner(mock_curr_mgr, mock_workspace_mgr)
        
        # Create bug files
        bugs_dir = tmp_path / "bugs"
        bugs_dir.mkdir()
        (bugs_dir / "bug1.json").write_text("{}")
        (bugs_dir / "bug1_symptom.txt").write_text("Symptom 1")
        (bugs_dir / "bug2.json").write_text("{}")
        (bugs_dir / "bug2_symptom.txt").write_text("Symptom 2")
        
        bug_file, symptom_file = runner._select_bug(bugs_dir)
        
        assert bug_file.exists()
        assert symptom_file.exists()
        assert bug_file.suffix == ".json"
    
    def test_present_challenge_workspace_error_wrapped(self, tmp_path):
        """Should wrap WorkspaceError in HardenChallengeError."""
        mock_curr_mgr = MagicMock()
        mock_workspace_mgr = MagicMock()
        runner = HardenRunner(mock_curr_mgr, mock_workspace_mgr)
        
        module = ModuleMetadata(id="mod1", name="Module 1", path="mod1")
        source_file = Path("solution.py")
        
        # Create shadow worktree and developer file
        shadow_worktree = tmp_path / ".mastery_engine_worktree"
        shadow_worktree.mkdir()
        
        modes_dir = tmp_path / "modes" / "developer"
        modes_dir.mkdir(parents=True)
        dev_file = modes_dir / "solution.py"
        dev_file.write_text("def solution(): pass")
        
        # Mock bug selection - patch file
        bugs_dir = MagicMock()
        mock_curr_mgr.get_bugs_dir.return_value = bugs_dir
        
        mock_bug_file = MagicMock()
        mock_bug_file.suffix = ".patch"
        mock_symptom_file = MagicMock()
        mock_symptom_file.read_text.return_value = "Symptom"
        runner._select_bug = MagicMock(return_value=(mock_bug_file, mock_symptom_file))
        
        # Mock workspace to raise WorkspaceError
        mock_workspace_mgr.apply_patch.side_effect = WorkspaceError("Patch failed")
        
        import os
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            with pytest.raises(HardenChallengeError, match="Failed to set up harden challenge"):
                runner.present_challenge("test_curr", module, source_file)
        finally:
            os.chdir(old_cwd)
    
    def test_present_library_challenge_invalid_bug_type(self, tmp_path):
        """Should handle unexpected bug file types."""
        mock_curr_mgr = MagicMock()
        mock_workspace_mgr = MagicMock()
        runner = HardenRunner(mock_curr_mgr, mock_workspace_mgr)
        
        problem = ProblemMetadata(
            id="prob1",
            name="Problem 1",
            title="Problem 1",
            path="pattern1/prob1",
            difficulty="easy"
        )
        source_file = Path("solution.py")
        
        # Create shadow worktree and student file
        shadow_worktree = tmp_path / ".mastery_engine_worktree"
        shadow_worktree.mkdir()
        student_file = tmp_path / "solution.py"
        student_file.write_text("def solve(): return 1")
        
        # Mock bug selection - unknown type
        bugs_dir = MagicMock()
        mock_curr_mgr.get_problem_bugs_dir.return_value = bugs_dir
        
        mock_bug_file = MagicMock()
        mock_bug_file.suffix = ".unknown"  # Not .json or .patch
        mock_symptom_file = MagicMock()
        mock_symptom_file.read_text.return_value = "Symptom"
        runner._select_bug = MagicMock(return_value=(mock_bug_file, mock_symptom_file))
        
        import os
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            # Should raise error for unknown bug type
            with pytest.raises(HardenChallengeError):
                runner.present_library_challenge("test_curr", "pattern1", problem, Path("/tmp/problem"))
        finally:
            os.chdir(old_cwd)
