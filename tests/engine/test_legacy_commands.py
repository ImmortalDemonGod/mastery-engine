"""
Tests for legacy submit commands (backward compatibility).

These tests target uncovered code in engine/main.py:
- submit_build command (lines 889-1061, 165 lines)
- submit_justification command (lines 1064-1257, 181 lines)  
- submit_fix command (lines 1260-1451, 184 lines)

These commands are maintained for backward compatibility but the unified
`submit` command is now preferred. Testing key paths to maintain coverage.
"""

import pytest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

from typer.testing import CliRunner
from engine.main import app
from engine.schemas import UserProgress, CurriculumManifest, ModuleMetadata, JustifyQuestion, ValidationResult, LLMEvaluationResponse


runner = CliRunner()


class TestLegacySubmitBuild:
    """Tests for legacy submit-build command."""
    
    @patch('engine.main.ValidationSubsystem')
    @patch('engine.main.WorkspaceManager')
    @patch('engine.main.CurriculumManager')
    @patch('engine.main.StateManager')
    @patch('engine.main.require_shadow_worktree')
    def test_submit_build_success(self, mock_req_wt, mock_state_cls, mock_curr_cls, mock_workspace_cls, mock_val_cls):
        """Should successfully submit and validate build stage."""
        # Setup
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
            modules=[ModuleMetadata(id="mod1", name="Module 1", path="m/m1")]
        )
        mock_curr_mgr.get_validator_path.return_value = Path("validator.py")
        
        mock_validator = MagicMock()
        mock_val_cls.return_value = mock_validator
        mock_validator.execute.return_value = ValidationResult(
            exit_code=0,
            stdout="All tests passed",
            stderr="",
            performance_seconds=1.5
        )
        
        # Execute
        result = runner.invoke(app, ["submit-build"])
        
        # Verify
        assert result.exit_code == 0
        assert "Validation Passed" in result.stdout
        mock_state_mgr.save.assert_called_once()
    
    @patch('engine.main.StateManager')
    @patch('engine.main.require_shadow_worktree')
    def test_submit_build_curriculum_complete(self, mock_req_wt, mock_state_cls):
        """Should handle case where all modules are complete."""
        # Setup
        mock_req_wt.return_value = Path(".mastery_engine_worktree")
        
        mock_state_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_state_mgr.load.return_value = UserProgress(
            curriculum_id="test",
            current_module_index=5,  # Beyond modules
            current_stage="build"
        )
        
        with patch('engine.main.CurriculumManager') as mock_curr_cls:
            mock_curr_mgr = MagicMock()
            mock_curr_cls.return_value = mock_curr_mgr
            mock_curr_mgr.load_manifest.return_value = CurriculumManifest(
                curriculum_name="Test",
                author="Test",
                version="1.0",
                modules=[ModuleMetadata(id="m1", name="M1", path="m/m1")]
            )
            
            # Execute
            result = runner.invoke(app, ["submit-build"])
            
            # Verify
            assert result.exit_code == 0
            assert "All modules completed" in result.stdout or "Curriculum Complete" in result.stdout
    
    @patch('engine.main.CurriculumManager')
    @patch('engine.main.StateManager')
    @patch('engine.main.require_shadow_worktree')
    def test_submit_build_wrong_stage(self, mock_req_wt, mock_state_cls, mock_curr_cls):
        """Should reject submission if not in build stage."""
        # Setup
        mock_req_wt.return_value = Path(".mastery_engine_worktree")
        
        mock_state_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_state_mgr.load.return_value = UserProgress(
            curriculum_id="test",
            current_module_index=0,
            current_stage="justify"  # Wrong stage
        )
        
        mock_curr_mgr = MagicMock()
        mock_curr_cls.return_value = mock_curr_mgr
        mock_curr_mgr.load_manifest.return_value = CurriculumManifest(
            curriculum_name="Test",
            author="Test",
            version="1.0",
            modules=[ModuleMetadata(id="m1", name="M1", path="m/m1")]
        )
        
        # Execute
        result = runner.invoke(app, ["submit-build"])
        
        # Verify
        assert result.exit_code == 0
        assert "Not in Build Stage" in result.stdout
        assert "JUSTIFY" in result.stdout
    
    @patch('engine.main.ValidationSubsystem')
    @patch('engine.main.WorkspaceManager')
    @patch('engine.main.CurriculumManager')
    @patch('engine.main.StateManager')
    @patch('engine.main.require_shadow_worktree')
    def test_submit_build_validation_fails(self, mock_req_wt, mock_state_cls, mock_curr_cls, mock_workspace_cls, mock_val_cls):
        """Should handle validation failure."""
        # Setup
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
        mock_curr_mgr.get_validator_path.return_value = Path("validator.py")
        
        mock_validator = MagicMock()
        mock_val_cls.return_value = mock_validator
        mock_validator.execute.return_value = ValidationResult(
            exit_code=1,
            stdout="Test failed",
            stderr="Error: assertion failed",
            performance_seconds=None
        )
        
        # Execute
        result = runner.invoke(app, ["submit-build"])
        
        # Verify
        assert result.exit_code == 0
        assert "Validation Failed" in result.stdout
        mock_state_mgr.save.assert_not_called()


class TestLegacySubmitJustification:
    """Tests for legacy submit-justification command."""
    
    @patch('engine.main.LLMService')
    @patch('engine.main.JustifyRunner')
    @patch('engine.main.CurriculumManager')
    @patch('engine.main.StateManager')
    @patch('engine.main.require_shadow_worktree')
    def test_submit_justification_correct_answer(self, mock_req_wt, mock_state_cls, mock_curr_cls, mock_justify_cls, mock_llm_cls):
        """Should accept correct justification answer."""
        # Setup
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
        mock_justify_runner.load_questions.return_value = [
            JustifyQuestion(
                id="q1",
                question="Why?",
                model_answer="Because X",
                failure_modes=[],
                required_concepts=["test"]
            )
        ]
        mock_justify_runner.check_fast_filter.return_value = (False, None)
        
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm
        mock_llm.evaluate_justification.return_value = LLMEvaluationResponse(
            is_correct=True,
            feedback="Good answer!"
        )
        
        # Execute
        result = runner.invoke(app, ["submit-justification", "My detailed answer"])
        
        # Verify
        assert result.exit_code == 0
        assert "Correct" in result.stdout or "accepted" in result.stdout.lower()
        mock_state_mgr.save.assert_called_once()
    
    @patch('engine.main.CurriculumManager')
    @patch('engine.main.StateManager')
    @patch('engine.main.require_shadow_worktree')
    def test_submit_justification_wrong_stage(self, mock_req_wt, mock_state_cls, mock_curr_cls):
        """Should reject submission if not in justify stage."""
        # Setup
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
        
        # Execute
        result = runner.invoke(app, ["submit-justification", "My answer"])
        
        # Verify
        assert result.exit_code == 0
        assert "Not in Justify Stage" in result.stdout


class TestLegacySubmitFix:
    """Tests for legacy submit-fix command."""
    
    # Note: submit_fix success path requires extensive filesystem mocking (shutil.copy2, file operations)
    # and is not included here. The key validation logic (stage checking) is covered in other tests.
    
    @patch('engine.main.CurriculumManager')
    @patch('engine.main.StateManager')
    @patch('engine.main.require_shadow_worktree')
    def test_submit_fix_wrong_stage(self, mock_req_wt, mock_state_cls, mock_curr_cls):
        """Should reject submission if not in harden stage."""
        # Setup
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
        
        # Execute
        result = runner.invoke(app, ["submit-fix"])
        
        # Verify
        assert result.exit_code == 0
        assert "Not in Harden Stage" in result.stdout
