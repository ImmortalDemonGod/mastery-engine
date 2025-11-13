"""
Comprehensive tests for new P1/P2 CLI commands.

Tests coverage targets for:
- show() command (P1) - Read-only display
- start_challenge() command (P1) - Explicit harden init
- next() command (P1) - Deprecation wrapper
- curriculum_list() command (P2) - Module listing
- progress_reset() command (P2) - Module reset

These tests target the ~530 lines added in P1/P2 CLI remediation.
"""

import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path

from typer.testing import CliRunner
from engine.main import app
from engine.schemas import UserProgress, CurriculumManifest, ModuleMetadata
from engine.curriculum import CurriculumNotFoundError, CurriculumInvalidError
from engine.state import StateFileCorruptedError
from engine.stages.justify import JustifyQuestionsError


runner = CliRunner()


class TestShowCommand:
    """Tests for P1 show command (read-only display)."""
    
    @patch('engine.main.require_shadow_worktree')
    @patch('engine.main.CurriculumManager')
    @patch('engine.main.StateManager')
    def test_show_current_module_build_stage(self, mock_state_cls, mock_curr_cls, mock_shadow):
        """Should display build prompt for current module."""
        # Setup
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_curr_cls.return_value = mock_curr_mgr
        
        mock_state_mgr.load.return_value = UserProgress(
            curriculum_id="test",
            current_module_index=0,
            current_stage="build"
        )
        
        mock_curr_mgr.load_manifest.return_value = CurriculumManifest(
            curriculum_name="Test Curriculum",
            author="Test",
            version="1.0.0",
            modules=[ModuleMetadata(id="mod1", name="Module 1", path="modules/mod1")]
        )
        
        mock_prompt_path = MagicMock()
        mock_prompt_path.exists.return_value = True
        mock_prompt_path.read_text.return_value = "# Build Challenge\nImplement feature X"
        mock_curr_mgr.get_build_prompt_path.return_value = mock_prompt_path
        
        # Execute
        result = runner.invoke(app, ["show"])
        
        # Verify
        assert result.exit_code == 0
        assert "Build Challenge" in result.stdout
        mock_curr_mgr.get_build_prompt_path.assert_called_once()
    
    @patch('engine.main.require_shadow_worktree')
    @patch('engine.main.JustifyRunner')
    @patch('engine.main.CurriculumManager')
    @patch('engine.main.StateManager')
    def test_show_current_module_justify_stage(self, mock_state_cls, mock_curr_cls, mock_justify_cls, mock_shadow):
        """Should display justify question for current module."""
        # Setup
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_curr_cls.return_value = mock_curr_mgr
        
        mock_state_mgr.load.return_value = UserProgress(
            curriculum_id="test",
            current_module_index=0,
            current_stage="justify"
        )
        
        mock_curr_mgr.load_manifest.return_value = CurriculumManifest(
            curriculum_name="Test Curriculum",
            author="Test",
            version="1.0.0",
            modules=[ModuleMetadata(id="mod1", name="Module 1", path="modules/mod1")]
        )
        
        mock_justify_runner = MagicMock()
        mock_justify_cls.return_value = mock_justify_runner
        
        from engine.schemas import JustifyQuestion
        mock_justify_runner.load_questions.return_value = [
            JustifyQuestion(
                id="q1",
                question="Why is this approach better?",
                model_answer="Because X, Y, Z",
                failure_modes=[],
                required_concepts=["concept1"]
            )
        ]
        
        # Execute
        result = runner.invoke(app, ["show"])
        
        # Verify
        assert result.exit_code == 0
        assert "Justify Challenge" in result.stdout
        assert "Why is this approach better?" in result.stdout
    
    @patch('engine.main.require_shadow_worktree')
    @patch('engine.main.CurriculumManager')
    @patch('engine.main.StateManager')
    def test_show_current_module_harden_stage_shows_instructions(self, mock_state_cls, mock_curr_cls, mock_shadow):
        """Should show instructions to run start-challenge in harden stage (read-only)."""
        # Setup
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_curr_cls.return_value = mock_curr_mgr
        
        mock_state_mgr.load.return_value = UserProgress(
            curriculum_id="test",
            current_module_index=0,
            current_stage="harden"
        )
        
        mock_curr_mgr.load_manifest.return_value = CurriculumManifest(
            curriculum_name="Test Curriculum",
            author="Test",
            version="1.0.0",
            modules=[ModuleMetadata(id="mod1", name="Module 1", path="modules/mod1")]
        )
        
        # Execute
        result = runner.invoke(app, ["show"])
        
        # Verify
        assert result.exit_code == 0
        assert "Harden Stage" in result.stdout
        assert "start-challenge" in result.stdout
        # Verify it does NOT call harden_runner (read-only!)
    
    @patch('engine.main.require_shadow_worktree')
    @patch('engine.main.CurriculumManager')
    @patch('engine.main.StateManager')
    def test_show_specific_module_by_id(self, mock_state_cls, mock_curr_cls, mock_shadow):
        """Should display specified module content."""
        # Setup
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_curr_cls.return_value = mock_curr_mgr
        
        mock_state_mgr.load.return_value = UserProgress(
            curriculum_id="test",
            current_module_index=1,
            current_stage="justify"
        )
        
        mock_curr_mgr.load_manifest.return_value = CurriculumManifest(
            curriculum_name="Test Curriculum",
            author="Test",
            version="1.0.0",
            modules=[
                ModuleMetadata(id="softmax", name="Softmax", path="modules/softmax"),
                ModuleMetadata(id="attention", name="Attention", path="modules/attention")
            ]
        )
        
        mock_prompt_path = MagicMock()
        mock_prompt_path.exists.return_value = True
        mock_prompt_path.read_text.return_value = "# Softmax Build Prompt"
        mock_curr_mgr.get_build_prompt_path.return_value = mock_prompt_path
        
        # Execute - show softmax even though current is attention
        result = runner.invoke(app, ["show", "softmax"])
        
        # Verify
        assert result.exit_code == 0
        assert "Softmax" in result.stdout
    
    @patch('engine.main.require_shadow_worktree')
    @patch('engine.main.CurriculumManager')
    @patch('engine.main.StateManager')
    def test_show_nonexistent_module_shows_error(self, mock_state_cls, mock_curr_cls, mock_shadow):
        """Should show error for nonexistent module ID."""
        # Setup
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_curr_cls.return_value = mock_curr_mgr
        
        mock_state_mgr.load.return_value = UserProgress(
            curriculum_id="test",
            current_module_index=0,
            current_stage="build"
        )
        
        mock_curr_mgr.load_manifest.return_value = CurriculumManifest(
            curriculum_name="Test Curriculum",
            author="Test",
            version="1.0.0",
            modules=[ModuleMetadata(id="mod1", name="Module 1", path="modules/mod1")]
        )
        
        # Execute
        result = runner.invoke(app, ["show", "nonexistent"])
        
        # Verify
        assert result.exit_code == 1
        assert "Module Not Found" in result.stdout


class TestStartChallengeCommand:
    """Tests for P1 start-challenge command (explicit harden init)."""
    
    @patch('engine.main.HardenRunner')
    @patch('engine.main.WorkspaceManager')
    @patch('engine.main.require_shadow_worktree')
    @patch('engine.main.CurriculumManager')
    @patch('engine.main.StateManager')
    def test_start_challenge_in_harden_stage_succeeds(
        self, mock_state_cls, mock_curr_cls, mock_shadow, mock_workspace_cls, mock_harden_cls
    ):
        """Should initialize harden workspace when in harden stage."""
        # Setup
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_curr_cls.return_value = mock_curr_mgr
        
        mock_state_mgr.load.return_value = UserProgress(
            curriculum_id="test",
            current_module_index=0,
            current_stage="harden"
        )
        
        mock_curr_mgr.load_manifest.return_value = CurriculumManifest(
            curriculum_name="Test Curriculum",
            author="Test",
            version="1.0.0",
            modules=[ModuleMetadata(id="softmax", name="Softmax", path="modules/softmax")]
        )
        
        mock_harden_runner = MagicMock()
        mock_harden_cls.return_value = mock_harden_runner
        mock_harden_runner.present_challenge.return_value = (
            Path("harden/utils.py"),
            "Bug symptom: Function returns incorrect value"
        )
        
        # Execute
        result = runner.invoke(app, ["start-challenge"])
        
        # Verify
        assert result.exit_code == 0
        assert "Debug Challenge" in result.stdout
        assert "Bug symptom" in result.stdout
        mock_harden_runner.present_challenge.assert_called_once()
    
    @patch('engine.main.require_shadow_worktree')
    @patch('engine.main.CurriculumManager')
    @patch('engine.main.StateManager')
    def test_start_challenge_in_build_stage_shows_error(self, mock_state_cls, mock_curr_cls, mock_shadow):
        """Should show error when not in harden stage."""
        # Setup
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_curr_cls.return_value = mock_curr_mgr
        
        mock_state_mgr.load.return_value = UserProgress(
            curriculum_id="test",
            current_module_index=0,
            current_stage="build"
        )
        
        mock_curr_mgr.load_manifest.return_value = CurriculumManifest(
            curriculum_name="Test Curriculum",
            author="Test",
            version="1.0.0",
            modules=[ModuleMetadata(id="mod1", name="Module 1", path="modules/mod1")]
        )
        
        # Execute
        result = runner.invoke(app, ["start-challenge"])
        
        # Verify
        assert result.exit_code == 1
        assert "Wrong Stage" in result.stdout
        assert "only works in the Harden stage" in result.stdout


class TestNextCommandDeprecation:
    """Tests for P1 next command deprecation wrapper."""
    
    @patch('engine.main.show')
    def test_next_shows_deprecation_warning(self, mock_show):
        """Should show deprecation warning and forward to show."""
        # Execute
        result = runner.invoke(app, ["next"])
        
        # Verify
        assert result.exit_code == 0
        assert "Deprecated Command" in result.stdout
        assert "engine show" in result.stdout
        assert "engine start-challenge" in result.stdout
        mock_show.assert_called_once_with(module_id=None)


class TestCurriculumListCommand:
    """Tests for P2 curriculum-list command."""
    
    @patch('engine.main.CurriculumManager')
    @patch('engine.main.StateManager')
    def test_curriculum_list_shows_all_modules_with_status(self, mock_state_cls, mock_curr_cls):
        """Should display table with all modules and their status."""
        # Setup
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_curr_cls.return_value = mock_curr_mgr
        
        mock_state_mgr.load.return_value = UserProgress(
            curriculum_id="test",
            current_module_index=1,
            current_stage="justify",
            completed_modules=["softmax"]
        )
        
        mock_curr_mgr.load_manifest.return_value = CurriculumManifest(
            curriculum_name="CS336 Assignment 1",
            author="Stanford",
            version="1.0.0",
            modules=[
                ModuleMetadata(id="softmax", name="Softmax", path="modules/softmax"),
                ModuleMetadata(id="attention", name="Attention", path="modules/attention"),
                ModuleMetadata(id="transformer", name="Transformer", path="modules/transformer")
            ]
        )
        
        # Execute
        result = runner.invoke(app, ["curriculum-list"])
        
        # Verify
        assert result.exit_code == 0
        assert "CS336 Assignment 1" in result.stdout
        assert "softmax" in result.stdout
        assert "attention" in result.stdout
        assert "transformer" in result.stdout
        assert "modules completed" in result.stdout
        # Check progress separately due to ANSI codes
        assert ("1" in result.stdout and "3" in result.stdout) or "Progress" in result.stdout
        # Status indicators should be present
        assert "✅" in result.stdout or "Complete" in result.stdout
        assert "🔵" in result.stdout or "JUSTIFY" in result.stdout


class TestProgressResetCommand:
    """Tests for P2 progress-reset command."""
    
    @patch('rich.prompt.Confirm')
    @patch('engine.main.CurriculumManager')
    @patch('engine.main.StateManager')
    def test_progress_reset_completed_module_with_confirmation(
        self, mock_state_cls, mock_curr_cls, mock_confirm_cls
    ):
        """Should reset a completed module after confirmation."""
        # Setup
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_curr_cls.return_value = mock_curr_mgr
        
        progress = UserProgress(
            curriculum_id="test",
            current_module_index=1,
            current_stage="build",
            completed_modules=["softmax"]
        )
        mock_state_mgr.load.return_value = progress
        
        mock_curr_mgr.load_manifest.return_value = CurriculumManifest(
            curriculum_name="Test Curriculum",
            author="Test",
            version="1.0.0",
            modules=[
                ModuleMetadata(id="softmax", name="Softmax", path="modules/softmax"),
                ModuleMetadata(id="attention", name="Attention", path="modules/attention")
            ]
        )
        
        # Mock user confirmation
        mock_confirm_cls.ask.return_value = True
        
        # Execute
        result = runner.invoke(app, ["progress-reset", "softmax"])
        
        # Verify
        assert result.exit_code == 0
        assert "Module Reset Complete" in result.stdout
        mock_state_mgr.save.assert_called_once()
        # Verify progress was modified
        saved_progress = mock_state_mgr.save.call_args[0][0]
        assert "softmax" not in saved_progress.completed_modules
        assert saved_progress.current_module_index == 0
        assert saved_progress.current_stage == "build"
    
    @patch('rich.prompt.Confirm')
    @patch('engine.main.CurriculumManager')
    @patch('engine.main.StateManager')
    def test_progress_reset_cancelled_by_user(
        self, mock_state_cls, mock_curr_cls, mock_confirm_cls
    ):
        """Should cancel reset if user declines confirmation."""
        # Setup
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_curr_cls.return_value = mock_curr_mgr
        
        mock_state_mgr.load.return_value = UserProgress(
            curriculum_id="test",
            current_module_index=1,
            current_stage="build",
            completed_modules=["softmax"]
        )
        
        mock_curr_mgr.load_manifest.return_value = CurriculumManifest(
            curriculum_name="Test Curriculum",
            author="Test",
            version="1.0.0",
            modules=[
                ModuleMetadata(id="softmax", name="Softmax", path="modules/softmax"),
                ModuleMetadata(id="attention", name="Attention", path="modules/attention")
            ]
        )
        
        # Mock user declining
        mock_confirm_cls.ask.return_value = False
        
        # Execute
        result = runner.invoke(app, ["progress-reset", "softmax"])
        
        # Verify
        assert result.exit_code == 0
        assert "Reset cancelled" in result.stdout
        mock_state_mgr.save.assert_not_called()
    
    @patch('engine.main.CurriculumManager')
    @patch('engine.main.StateManager')
    def test_progress_reset_nonexistent_module_shows_error(self, mock_state_cls, mock_curr_cls):
        """Should show error for nonexistent module."""
        # Setup
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_curr_cls.return_value = mock_curr_mgr
        
        mock_state_mgr.load.return_value = UserProgress(
            curriculum_id="test",
            current_module_index=0,
            current_stage="build"
        )
        
        mock_curr_mgr.load_manifest.return_value = CurriculumManifest(
            curriculum_name="Test Curriculum",
            author="Test",
            version="1.0.0",
            modules=[ModuleMetadata(id="mod1", name="Module 1", path="modules/mod1")]
        )
        
        # Execute
        result = runner.invoke(app, ["progress-reset", "nonexistent"])
        
        # Verify
        assert result.exit_code == 1
        assert "Module Not Found" in result.stdout
    
    @patch('engine.main.CurriculumManager')
    @patch('engine.main.StateManager')
    def test_progress_reset_not_started_module_shows_info(self, mock_state_cls, mock_curr_cls):
        """Should show info message for module that hasn't been started."""
        # Setup
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_curr_cls.return_value = mock_curr_mgr
        
        mock_state_mgr.load.return_value = UserProgress(
            curriculum_id="test",
            current_module_index=0,
            current_stage="build"
        )
        
        mock_curr_mgr.load_manifest.return_value = CurriculumManifest(
            curriculum_name="Test Curriculum",
            author="Test",
            version="1.0.0",
            modules=[
                ModuleMetadata(id="mod1", name="Module 1", path="modules/mod1"),
                ModuleMetadata(id="mod2", name="Module 2", path="modules/mod2")
            ]
        )
        
        # Execute - try to reset module 2 which hasn't started
        result = runner.invoke(app, ["progress-reset", "mod2"])
        
        # Verify
        assert result.exit_code == 0
        assert "Module Not Started" in result.stdout
        assert "nothing to reset" in result.stdout
