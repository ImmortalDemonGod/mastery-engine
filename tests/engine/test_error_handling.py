"""
Tests for error handling paths in main CLI commands.

These tests target uncovered error handling code in engine/main.py:
- submit command error paths (lines 555-575)
- show command error paths (lines 715-743)
- status command error paths (lines 2031-2070)
- start-challenge error paths (lines 803, 830-857)

Coverage targets: Error handling, exception propagation, user messaging
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from typer.testing import CliRunner
from engine.main import app
from engine.schemas import UserProgress, CurriculumManifest, ModuleMetadata
from engine.curriculum import CurriculumNotFoundError, CurriculumInvalidError
from engine.state import StateFileCorruptedError
from engine.stages.justify import JustifyQuestionsError
from engine.stages.harden import HardenChallengeError
from engine.validator import ValidatorNotFoundError, ValidatorTimeoutError, ValidatorExecutionError
from engine.services.llm_service import LLMAPIError, LLMResponseError, ConfigurationError


runner = CliRunner()


class TestSubmitCommandErrorHandling:
    """Tests for error handling in the unified submit command."""
    
    @patch('engine.main.StateManager')
    @patch('engine.main.require_shadow_worktree')
    def test_submit_state_file_corrupted(self, mock_req_wt, mock_state_cls):
        """Should handle StateFileCorruptedError gracefully."""
        mock_req_wt.return_value = Path(".mastery_engine_worktree")
        
        mock_state_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_state_mgr.load.side_effect = StateFileCorruptedError("Corrupted state file")
        
        result = runner.invoke(app, ["submit"])
        
        assert result.exit_code == 1
        assert "StateFileCorrupted" in result.stdout or "State File Corrupted" in result.stdout
    
    @patch('engine.main.CurriculumManager')
    @patch('engine.main.StateManager')
    @patch('engine.main.require_shadow_worktree')
    def test_submit_curriculum_not_found(self, mock_req_wt, mock_state_cls, mock_curr_cls):
        """Should handle CurriculumNotFoundError gracefully."""
        mock_req_wt.return_value = Path(".mastery_engine_worktree")
        
        mock_state_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_state_mgr.load.return_value = UserProgress(
            curriculum_id="nonexistent",
            current_module_index=0,
            current_stage="build"
        )
        
        mock_curr_mgr = MagicMock()
        mock_curr_cls.return_value = mock_curr_mgr
        mock_curr_mgr.load_manifest.side_effect = CurriculumNotFoundError("Curriculum not found")
        
        result = runner.invoke(app, ["submit"])
        
        assert result.exit_code == 1
        assert "CurriculumNotFound" in result.stdout or "Curriculum Not Found" in result.stdout
    
    @patch('engine.main.CurriculumManager')
    @patch('engine.main.StateManager')
    @patch('engine.main.require_shadow_worktree')
    def test_submit_curriculum_invalid(self, mock_req_wt, mock_state_cls, mock_curr_cls):
        """Should handle CurriculumInvalidError gracefully."""
        mock_req_wt.return_value = Path(".mastery_engine_worktree")
        
        mock_state_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_state_mgr.load.return_value = UserProgress(
            curriculum_id="invalid",
            current_module_index=0,
            current_stage="build"
        )
        
        mock_curr_mgr = MagicMock()
        mock_curr_cls.return_value = mock_curr_mgr
        mock_curr_mgr.load_manifest.side_effect = CurriculumInvalidError("Invalid curriculum format")
        
        result = runner.invoke(app, ["submit"])
        
        assert result.exit_code == 1
        assert "CurriculumInvalid" in result.stdout or "Invalid Curriculum" in result.stdout
    
    @patch('engine.main.ValidationSubsystem')
    @patch('engine.main.WorkspaceManager')
    @patch('engine.main.CurriculumManager')
    @patch('engine.main.StateManager')
    @patch('engine.main.require_shadow_worktree')
    def test_submit_validator_timeout(self, mock_req_wt, mock_state_cls, mock_curr_cls, mock_workspace_cls, mock_val_cls):
        """Should handle ValidatorTimeoutError gracefully."""
        mock_req_wt.return_value = Path(".mastery_engine_worktree")
        
        mock_state_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_state_mgr.load.return_value = UserProgress(
            curriculum_id="test",
            current_module_index=0,
            current_stage="build"
        )
        
        mock_curr_mgr = MagicMock()
        mock_curr_cls.return_value = mock_curr_mgr
        mock_curr_mgr.load_manifest.return_value = CurriculumManifest(
            curriculum_name="Test",
            author="Test",
            version="1.0",
            modules=[ModuleMetadata(id="m1", name="M1", path="m/m1")]
        )
        
        mock_validator = MagicMock()
        mock_val_cls.return_value = mock_validator
        mock_validator.execute.side_effect = ValidatorTimeoutError("Validator timed out")
        
        result = runner.invoke(app, ["submit"])
        
        assert result.exit_code == 1
        assert "ValidatorTimeout" in result.stdout or "timed out" in result.stdout.lower()
    
    @patch('engine.main._submit_build_stage')
    @patch('engine.main._check_curriculum_complete')
    @patch('engine.main._load_curriculum_state')
    @patch('engine.main.require_shadow_worktree')
    def test_submit_unexpected_error(self, mock_req_wt, mock_load, mock_check, mock_submit_build):
        """Should handle unexpected exceptions gracefully."""
        mock_req_wt.return_value = Path(".mastery_engine_worktree")
        
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        mock_progress = UserProgress(
            curriculum_id="test",
            current_module_index=0,
            current_stage="build"
        )
        mock_manifest = CurriculumManifest(
            curriculum_name="Test",
            author="Test",
            version="1.0",
            modules=[ModuleMetadata(id="m1", name="M1", path="m/m1")]
        )
        
        mock_load.return_value = (mock_state_mgr, mock_curr_mgr, mock_progress, mock_manifest)
        mock_check.return_value = False
        mock_submit_build.side_effect = RuntimeError("Unexpected internal error")
        
        result = runner.invoke(app, ["submit"])
        
        assert result.exit_code == 1
        assert "Unexpected Error" in result.stdout or "ENGINE ERROR" in result.stdout


class TestShowCommandErrorHandling:
    """Tests for error handling in the show command."""
    
    @patch('engine.main.StateManager')
    @patch('engine.main.require_shadow_worktree')
    def test_show_state_file_corrupted(self, mock_req_wt, mock_state_cls):
        """Should handle StateFileCorruptedError gracefully."""
        mock_req_wt.return_value = Path(".mastery_engine_worktree")
        
        mock_state_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_state_mgr.load.side_effect = StateFileCorruptedError("Corrupted state")
        
        result = runner.invoke(app, ["show"])
        
        assert result.exit_code == 1
        assert "State File Corrupted" in result.stdout
    
    @patch('engine.main.CurriculumManager')
    @patch('engine.main.StateManager')
    @patch('engine.main.require_shadow_worktree')
    def test_show_curriculum_not_found(self, mock_req_wt, mock_state_cls, mock_curr_cls):
        """Should handle CurriculumNotFoundError gracefully."""
        mock_req_wt.return_value = Path(".mastery_engine_worktree")
        
        mock_state_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_state_mgr.load.return_value = UserProgress(
            curriculum_id="missing",
            current_module_index=0,
            current_stage="build"
        )
        
        mock_curr_mgr = MagicMock()
        mock_curr_cls.return_value = mock_curr_mgr
        mock_curr_mgr.load_manifest.side_effect = CurriculumNotFoundError("Not found")
        
        result = runner.invoke(app, ["show"])
        
        assert result.exit_code == 1
        assert "Curriculum Not Found" in result.stdout
    
    @patch('engine.main.JustifyRunner')
    @patch('engine.main.CurriculumManager')
    @patch('engine.main.StateManager')
    @patch('engine.main.require_shadow_worktree')
    def test_show_justify_questions_error(self, mock_req_wt, mock_state_cls, mock_curr_cls, mock_justify_cls):
        """Should handle JustifyQuestionsError gracefully."""
        mock_req_wt.return_value = Path(".mastery_engine_worktree")
        
        mock_state_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_state_mgr.load.return_value = UserProgress(
            curriculum_id="test",
            current_module_index=0,
            current_stage="justify"
        )
        
        mock_curr_mgr = MagicMock()
        mock_curr_cls.return_value = mock_curr_mgr
        mock_curr_mgr.load_manifest.return_value = CurriculumManifest(
            curriculum_name="Test",
            author="Test",
            version="1.0",
            modules=[ModuleMetadata(id="m1", name="M1", path="m/m1")]
        )
        
        mock_justify_runner = MagicMock()
        mock_justify_cls.return_value = mock_justify_runner
        mock_justify_runner.load_questions.side_effect = JustifyQuestionsError("Questions not found")
        
        result = runner.invoke(app, ["show"])
        
        assert result.exit_code == 1
        assert "Justify Questions Error" in result.stdout


class TestStartChallengeErrorHandling:
    """Tests for error handling in start-challenge command."""
    
    @patch('engine.main.CurriculumManager')
    @patch('engine.main.StateManager')
    @patch('engine.main.require_shadow_worktree')
    def test_start_challenge_wrong_stage_error(self, mock_req_wt, mock_state_cls, mock_curr_cls):
        """Should reject start-challenge if not in harden stage."""
        mock_req_wt.return_value = Path(".mastery_engine_worktree")
        
        mock_state_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_state_mgr.load.return_value = UserProgress(
            curriculum_id="test",
            current_module_index=0,
            current_stage="build"  # Wrong stage
        )
        
        mock_curr_mgr = MagicMock()
        mock_curr_cls.return_value = mock_curr_mgr
        mock_curr_mgr.load_manifest.return_value = CurriculumManifest(
            curriculum_name="Test",
            author="Test",
            version="1.0",
            modules=[ModuleMetadata(id="m1", name="M1", path="m/m1")]
        )
        
        result = runner.invoke(app, ["start-challenge"])
        
        assert result.exit_code == 1  # Exits with error
        assert "Not in Harden Stage" in result.stdout or "Only available" in result.stdout
    
    @patch('engine.main.HardenRunner')
    @patch('engine.main.WorkspaceManager')
    @patch('engine.main.CurriculumManager')
    @patch('engine.main.StateManager')
    @patch('engine.main.require_shadow_worktree')
    def test_start_challenge_harden_error(self, mock_req_wt, mock_state_cls, mock_curr_cls, mock_workspace_cls, mock_harden_cls):
        """Should handle HardenChallengeError gracefully."""
        mock_req_wt.return_value = Path(".mastery_engine_worktree")
        
        mock_state_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_state_mgr.load.return_value = UserProgress(
            curriculum_id="test",
            current_module_index=0,
            current_stage="harden"
        )
        
        mock_curr_mgr = MagicMock()
        mock_curr_cls.return_value = mock_curr_mgr
        mock_curr_mgr.load_manifest.return_value = CurriculumManifest(
            curriculum_name="Test",
            author="Test",
            version="1.0",
            modules=[ModuleMetadata(id="m1", name="M1", path="m/m1")]
        )
        
        mock_harden_runner = MagicMock()
        mock_harden_cls.return_value = mock_harden_runner
        mock_harden_runner.present_challenge.side_effect = HardenChallengeError("No bugs available")
        
        result = runner.invoke(app, ["start-challenge"])
        
        assert result.exit_code == 1
        assert "Harden Challenge Error" in result.stdout or "ENGINE ERROR" in result.stdout


class TestStatusCommandErrorHandling:
    """Tests for error handling in status command."""
    
    @patch('engine.main.StateManager')
    @patch('engine.main.require_shadow_worktree')
    def test_status_state_file_corrupted(self, mock_req_wt, mock_state_cls):
        """Should handle StateFileCorruptedError gracefully."""
        mock_req_wt.return_value = Path(".mastery_engine_worktree")
        
        mock_state_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_state_mgr.load.side_effect = StateFileCorruptedError("Corrupted")
        
        result = runner.invoke(app, ["status"])
        
        assert result.exit_code == 1
        assert "State File Corrupted" in result.stdout
    
    @patch('engine.main.CurriculumManager')
    @patch('engine.main.StateManager')
    @patch('engine.main.require_shadow_worktree')
    def test_status_curriculum_not_found(self, mock_req_wt, mock_state_cls, mock_curr_cls):
        """Should handle CurriculumNotFoundError gracefully."""
        mock_req_wt.return_value = Path(".mastery_engine_worktree")
        
        mock_state_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_state_mgr.load.return_value = UserProgress(
            curriculum_id="missing",
            current_module_index=0,
            current_stage="build"
        )
        
        mock_curr_mgr = MagicMock()
        mock_curr_cls.return_value = mock_curr_mgr
        mock_curr_mgr.load_manifest.side_effect = CurriculumNotFoundError("Not found")
        
        result = runner.invoke(app, ["status"])
        
        assert result.exit_code == 1
        assert "Curriculum Not Found" in result.stdout
