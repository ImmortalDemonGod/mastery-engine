"""
Comprehensive tests for init and cleanup commands.

These tests target uncovered code in engine/main.py:
- init command (lines 1454-1601, 137 lines)
- cleanup command (lines 1931-1982, 45 lines)

Coverage targets:
- Init success path
- Init error paths (not in git, dirty working dir, already initialized, invalid curriculum)
- Cleanup success path
- Cleanup error paths (no worktree, git errors)
"""

import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
import subprocess

from typer.testing import CliRunner
from engine.main import app
from engine.schemas import UserProgress, CurriculumManifest, ModuleMetadata
from engine.curriculum import CurriculumNotFoundError, CurriculumInvalidError


runner = CliRunner()


class TestInitCommand:
    """Tests for the init command."""
    
    @patch('engine.main.StateManager')
    @patch('engine.main.CurriculumManager')
    @patch('subprocess.run')
    @patch('engine.main.SHADOW_WORKTREE_DIR')
    def test_init_success(self, mock_shadow_dir, mock_subprocess, mock_curr_cls, mock_state_cls):
        """Should successfully initialize engine with valid curriculum."""
        # Setup
        mock_shadow_dir.exists.return_value = False
        mock_shadow_dir.__str__.return_value = ".mastery_engine_worktree"
        mock_shadow_dir.__truediv__ = lambda self, other: MagicMock()
        
        # Mock git commands
        mock_subprocess.side_effect = [
            MagicMock(returncode=0),  # git rev-parse (check is git repo)
            MagicMock(returncode=0, stdout=""),  # git status (clean)
            MagicMock(returncode=0),  # git worktree prune
            MagicMock(returncode=0),  # git worktree add
        ]
        
        # Mock curriculum
        mock_curr_mgr = MagicMock()
        mock_curr_cls.return_value = mock_curr_mgr
        mock_curr_mgr.load_manifest.return_value = CurriculumManifest(
            curriculum_name="Test Curriculum",
            author="Test",
            version="1.0.0",
            modules=[
                ModuleMetadata(id="mod1", name="Module 1", path="modules/mod1")
            ]
        )
        
        # Mock state manager
        mock_state_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        
        # Execute
        result = runner.invoke(app, ["init", "test_curriculum"])
        
        # Verify
        assert result.exit_code == 0
        assert "Initialization Complete" in result.stdout
        assert "Test Curriculum" in result.stdout
        mock_state_mgr.save.assert_called_once()
        saved_state = mock_state_mgr.save.call_args[0][0]
        assert saved_state.curriculum_id == "test_curriculum"
        assert saved_state.current_stage == "build"
    
    @patch('subprocess.run')
    def test_init_not_git_repository(self, mock_subprocess):
        """Should fail if not in a git repository."""
        # Setup - git rev-parse fails
        mock_subprocess.side_effect = subprocess.CalledProcessError(
            128, ["git", "rev-parse", "--git-dir"]
        )
        
        # Execute
        result = runner.invoke(app, ["init", "test_curriculum"])
        
        # Verify
        assert result.exit_code == 1
        assert "Not a Git Repository" in result.stdout
    
    @patch('engine.main.StateManager')
    @patch('engine.main.CurriculumManager')
    @patch('subprocess.run')
    @patch('engine.main.SHADOW_WORKTREE_DIR')
    def test_init_dirty_working_directory(self, mock_shadow_dir, mock_subprocess, mock_curr_cls, mock_state_cls):
        """Init succeeds with dirty working directory (snapshot sync behavior)."""
        # Setup
        mock_shadow_dir.exists.return_value = False
        
        # Mock git commands - dirty state with snapshot sync
        mock_subprocess.side_effect = [
            MagicMock(returncode=0, stdout="/tmp/repo", stderr=""),  # git rev-parse
            MagicMock(returncode=0, stdout="M  file.py\n", stderr=""),  # git status (dirty)
            MagicMock(returncode=0, stdout="file.py\n", stderr=""),  # git ls-files -m
            MagicMock(returncode=0, stdout="", stderr=""),  # git worktree prune
            MagicMock(returncode=0, stdout="", stderr=""),  # git worktree add
        ]
        
        # Mock curriculum
        mock_curr_mgr = MagicMock()
        mock_curr_cls.return_value = mock_curr_mgr
        mock_curr_mgr.load_manifest.return_value = CurriculumManifest(
            curriculum_name="Test",
            author="Test",
            version="1.0.0",
            modules=[ModuleMetadata(id="m1", name="M1", path="modules/m1")]
        )
        
        # Mock state - return None for load() to simulate fresh init
        mock_state_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_state_mgr.load.return_value = None  # No existing state
        mock_state_mgr.progress = None
        
        result = runner.invoke(app, ["init", "test_curriculum"])
        
        # New behavior: init succeeds with snapshot sync
        assert result.exit_code == 0
        assert "Initialization Complete" in result.stdout
    
    @patch('engine.main.StateManager')
    @patch('engine.main.CurriculumManager')
    @patch('subprocess.run')
    @patch('engine.main.SHADOW_WORKTREE_DIR')
    def test_init_already_initialized(self, mock_shadow_dir, mock_subprocess, mock_curr_cls, mock_state_cls):
        """Init is idempotent - succeeds gracefully when already initialized."""
        # Setup
        mock_shadow_dir.exists.return_value = True
        
        # Mock git commands
        mock_subprocess.side_effect = [
            MagicMock(returncode=0, stdout="/tmp/repo"),  # git rev-parse
            MagicMock(returncode=0, stdout=""),  # git status (clean)
            MagicMock(returncode=0),  # git worktree prune
            MagicMock(returncode=0, stdout=".mastery_engine_worktree"),  # git worktree list
        ]
        
        # Mock curriculum
        mock_curr_mgr = MagicMock()
        mock_curr_cls.return_value = mock_curr_mgr
        mock_curr_mgr.load_manifest.return_value = CurriculumManifest(
            curriculum_name="Test",
            author="Test",
            version="1.0.0",
            modules=[ModuleMetadata(id="m1", name="M1", path="modules/m1")]
        )
        
        # Mock state with existing progress - same curriculum
        mock_state_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        existing_progress = UserProgress(curriculum_id="test_curriculum", current_module_index=0, current_stage="build")
        mock_state_mgr.load.return_value = existing_progress
        mock_state_mgr.progress = existing_progress
        
        result = runner.invoke(app, ["init", "test_curriculum"])
        
        # New behavior: init is idempotent
        assert result.exit_code == 0
        assert "Already" in result.stdout or "No changes needed" in result.stdout
    
    @patch('engine.main.CurriculumManager')
    @patch('subprocess.run')
    @patch('engine.main.SHADOW_WORKTREE_DIR')
    def test_init_invalid_curriculum(self, mock_shadow_dir, mock_subprocess, mock_curr_cls):
        """Should fail if curriculum doesn't exist."""
        # Setup
        mock_shadow_dir.exists.return_value = False
        
        mock_subprocess.side_effect = [
            MagicMock(returncode=0),  # git rev-parse
            MagicMock(returncode=0, stdout=""),  # git status (clean)
            MagicMock(returncode=0),  # git worktree prune
        ]
        
        mock_curr_mgr = MagicMock()
        mock_curr_cls.return_value = mock_curr_mgr
        mock_curr_mgr.load_manifest.side_effect = CurriculumNotFoundError(
            "Curriculum 'invalid' not found"
        )
        
        # Execute
        result = runner.invoke(app, ["init", "invalid"])
        
        # Verify
        assert result.exit_code == 1
        assert "Invalid Curriculum" in result.stdout
        assert "not found" in result.stdout
    
    @patch('engine.main.StateManager')
    @patch('engine.main.CurriculumManager')
    @patch('subprocess.run')
    @patch('engine.main.SHADOW_WORKTREE_DIR')
    def test_init_git_worktree_fails(self, mock_shadow_dir, mock_subprocess, mock_curr_cls, mock_state_cls):
        """Should handle git worktree creation failure."""
        # Setup
        mock_shadow_dir.exists.return_value = False
        mock_shadow_dir.__str__.return_value = ".mastery_engine_worktree"
        
        # Mock git commands - worktree add fails
        mock_subprocess.side_effect = [
            MagicMock(returncode=0),  # git rev-parse
            MagicMock(returncode=0, stdout=""),  # git status
            MagicMock(returncode=0),  # git worktree prune
            subprocess.CalledProcessError(1, ["git", "worktree", "add"], stderr="error"),
        ]
        
        # Mock curriculum
        mock_curr_mgr = MagicMock()
        mock_curr_cls.return_value = mock_curr_mgr
        mock_curr_mgr.load_manifest.return_value = CurriculumManifest(
            curriculum_name="Test",
            author="Test",
            version="1.0.0",
            modules=[ModuleMetadata(id="m1", name="M1", path="m/m1")]
        )
        
        # Execute
        result = runner.invoke(app, ["init", "test"])
        
        # Verify
        assert result.exit_code == 1
        assert "Git Error" in result.stdout
        assert "Failed to create shadow worktree" in result.stdout


class TestCleanupCommand:
    """Tests for the cleanup command."""
    
    @patch('subprocess.run')
    @patch('engine.main.SHADOW_WORKTREE_DIR')
    def test_cleanup_success(self, mock_shadow_dir, mock_subprocess):
        """Should successfully remove shadow worktree."""
        # Setup
        mock_shadow_dir.exists.return_value = True
        mock_shadow_dir.__str__.return_value = ".mastery_engine_worktree"
        
        mock_subprocess.return_value = MagicMock(returncode=0)
        
        # Execute
        result = runner.invoke(app, ["cleanup"])
        
        # Verify
        assert result.exit_code == 0
        assert "Cleanup Complete" in result.stdout
        assert "Shadow worktree removed" in result.stdout
        
        # Verify git worktree remove was called
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0][0]
        assert "git" in call_args
        assert "worktree" in call_args
        assert "remove" in call_args
    
    @patch('engine.main.SHADOW_WORKTREE_DIR')
    def test_cleanup_no_worktree(self, mock_shadow_dir):
        """Should handle case where no worktree exists."""
        # Setup
        mock_shadow_dir.exists.return_value = False
        
        # Execute
        result = runner.invoke(app, ["cleanup"])
        
        # Verify
        assert result.exit_code == 0
        assert "No shadow worktree found" in result.stdout
        assert "Nothing to clean up" in result.stdout
    
    @patch('subprocess.run')
    @patch('engine.main.SHADOW_WORKTREE_DIR')
    def test_cleanup_git_error(self, mock_shadow_dir, mock_subprocess):
        """Should handle git worktree remove failure."""
        # Setup
        mock_shadow_dir.exists.return_value = True
        mock_shadow_dir.__str__.return_value = ".mastery_engine_worktree"
        
        mock_subprocess.side_effect = subprocess.CalledProcessError(
            1, ["git", "worktree", "remove"], stderr="locked"
        )
        
        # Execute
        result = runner.invoke(app, ["cleanup"])
        
        # Verify
        assert result.exit_code == 1
        assert "Git Error" in result.stdout
        assert "Failed to remove shadow worktree" in result.stdout
        assert "git worktree remove" in result.stdout


class TestRequireShadowWorktree:
    """Tests for the require_shadow_worktree helper function."""
    
    @patch('engine.main.SHADOW_WORKTREE_DIR')
    def test_require_shadow_worktree_missing(self, mock_shadow_dir):
        """Should exit if shadow worktree doesn't exist."""
        # Setup
        mock_shadow_dir.exists.return_value = False
        
        # Execute - use a command that requires shadow worktree
        result = runner.invoke(app, ["status"])
        
        # Verify
        assert result.exit_code == 1
        assert "Not Initialized" in result.stdout
        assert "engine init" in result.stdout
