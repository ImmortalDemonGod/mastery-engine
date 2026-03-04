"""
Comprehensive tests for CLI submit handlers and unified submit command.

These tests target the P0 CLI implementation (unified submit command) to
achieve maximum coverage of engine/main.py (currently at 3%).

Test coverage targets:
- _load_curriculum_state()
- _check_curriculum_complete()
- _submit_build_stage()
- _submit_justify_stage()
- _submit_harden_stage()
- submit() unified command routing
"""

import pytest
from unittest.mock import MagicMock, patch, mock_open, call
from pathlib import Path
import tempfile
import os

from engine.schemas import UserProgress, CurriculumManifest, ModuleMetadata, JustifyQuestion, LLMEvaluationResponse
from engine.curriculum import CurriculumNotFoundError, CurriculumInvalidError
from engine.validator import ValidationResult, ValidatorTimeoutError, ValidatorExecutionError
from engine.stages.justify import JustifyQuestionsError
from engine.stages.harden import HardenChallengeError
from engine.services.llm_service import ConfigurationError, LLMAPIError, LLMResponseError


class TestLoadCurriculumState:
    """Tests for _load_curriculum_state helper."""
    
    @patch('engine.main.CurriculumManager')
    @patch('engine.main.StateManager')
    def test_loads_state_and_curriculum_successfully(self, mock_state_cls, mock_curr_cls):
        """Should load state and curriculum and return tuple."""
        from engine.main import _load_curriculum_state
        
        # Setup mocks
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_curr_cls.return_value = mock_curr_mgr
        
        mock_progress = UserProgress(
            curriculum_id="test_curriculum",
            current_module_index=0,
            current_stage="build"
        )
        mock_manifest = CurriculumManifest(
            curriculum_name="test",
            author="test",
            version="1.0.0",
            modules=[ModuleMetadata(id="mod1", name="Module 1", path="modules/mod1")]
        )
        
        mock_state_mgr.load.return_value = mock_progress
        mock_curr_mgr.load_manifest.return_value = mock_manifest
        
        # Execute
        state_mgr, curr_mgr, progress, manifest = _load_curriculum_state()
        
        # Verify
        assert state_mgr == mock_state_mgr
        assert curr_mgr == mock_curr_mgr
        assert progress == mock_progress
        assert manifest == mock_manifest
        mock_state_mgr.load.assert_called_once()
        mock_curr_mgr.load_manifest.assert_called_once_with("test_curriculum")


class TestCheckCurriculumComplete:
    """Tests for _check_curriculum_complete helper."""
    
    def test_returns_false_when_modules_remaining(self):
        """Should return False when current_module_index < len(modules)."""
        from engine.main import _check_curriculum_complete
        
        progress = UserProgress(
            curriculum_id="test",
            current_module_index=0,
            current_stage="build"
        )
        manifest = CurriculumManifest(
            curriculum_name="test",
            author="test",
            version="1.0.0",
            modules=[
                ModuleMetadata(id="mod1", name="Module 1", path="modules/mod1"),
                ModuleMetadata(id="mod2", name="Module 2", path="modules/mod2")
            ]
        )
        
        result = _check_curriculum_complete(progress, manifest)
        assert result is False
    
    @patch('engine.main.console')
    def test_returns_true_when_all_modules_complete(self, mock_console):
        """Should return True and print completion message when done."""
        from engine.main import _check_curriculum_complete
        
        progress = UserProgress(
            curriculum_id="test",
            current_module_index=2,  # Beyond last module
            current_stage="build"
        )
        manifest = CurriculumManifest(
            curriculum_name="test",
            author="test",
            version="1.0.0",
            modules=[
                ModuleMetadata(id="mod1", name="Module 1", path="modules/mod1"),
                ModuleMetadata(id="mod2", name="Module 2", path="modules/mod2")
            ]
        )
        
        result = _check_curriculum_complete(progress, manifest)
        assert result is True
        # Verify completion panel was printed
        assert mock_console.print.called


class TestSubmitBuildStage:
    """Tests for _submit_build_stage handler."""
    
    @patch('engine.main.ValidationSubsystem')
    def test_build_success_advances_stage(self, mock_validator_cls):
        """Should advance to justify stage on successful validation."""
        from engine.main import _submit_build_stage
        
        # Setup
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        progress = MagicMock(spec=UserProgress)
        progress.curriculum_id = "test"
        progress.current_module_index = 0
        progress.current_stage = "build"
        
        manifest = CurriculumManifest(
            curriculum_name="test",
            author="test",
            version="1.0.0",
            modules=[ModuleMetadata(id="softmax", name="Softmax", path="modules/softmax")]
        )
        
        mock_validator = MagicMock()
        mock_validator_cls.return_value = mock_validator
        mock_validator.execute.return_value = ValidationResult(
            exit_code=0,
            stdout="All tests passed",
            stderr="",
            performance_seconds=1.5
        )
        
        # Execute
        result = _submit_build_stage(mock_state_mgr, mock_curr_mgr, progress, manifest)
        
        # Verify
        assert result is True
        progress.mark_stage_complete.assert_called_once_with("build")
        mock_state_mgr.save.assert_called_once_with(progress)
    
    @patch('engine.main.ValidationSubsystem')
    @patch('engine.main.console')
    def test_build_failure_does_not_advance(self, mock_console, mock_validator_cls):
        """Should not advance stage on failed validation."""
        from engine.main import _submit_build_stage
        
        # Setup
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        progress = UserProgress(
            curriculum_id="test",
            current_module_index=0,
            current_stage="build"
        )
        manifest = CurriculumManifest(
            curriculum_name="test",
            author="test",
            version="1.0.0",
            modules=[ModuleMetadata(id="softmax", name="Softmax", path="modules/softmax")]
        )
        
        mock_validator = MagicMock()
        mock_validator_cls.return_value = mock_validator
        mock_validator.execute.return_value = ValidationResult(
            exit_code=1,
            stdout="",
            stderr="FAILED test_softmax",
            performance_seconds=0.5
        )
        
        # Execute
        result = _submit_build_stage(mock_state_mgr, mock_curr_mgr, progress, manifest)
        
        # Verify
        assert result is False
        mock_state_mgr.save.assert_not_called()


class TestSubmitJustifyStage:
    """Tests for _submit_justify_stage handler with $EDITOR and LLM."""
    
    @patch('engine.main.LLMService')
    @patch('subprocess.run')
    @patch('engine.main.JustifyRunner')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.unlink')
    def test_justify_fast_filter_rejects_shallow_answer(
        self, mock_unlink, mock_file, mock_justify_cls, mock_subprocess, mock_llm_cls
    ):
        """Should reject answer caught by fast keyword filter."""
        from engine.main import _submit_justify_stage
        
        # Setup
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        progress = MagicMock(spec=UserProgress)
        progress.curriculum_id = "test"
        progress.current_module_index = 0
        progress.current_stage = "justify"
        
        manifest = CurriculumManifest(
            curriculum_name="test",
            author="test",
            version="1.0.0",
            modules=[ModuleMetadata(id="softmax", name="Softmax", path="modules/softmax")]
        )
        
        mock_justify_runner = MagicMock()
        mock_justify_cls.return_value = mock_justify_runner
        mock_justify_runner.load_questions.return_value = [
            JustifyQuestion(
                id="q1",
                question="Why softmax?",
                model_answer="Softmax applies exp then normalizes",
                failure_modes=[],
                required_concepts=["exp", "normalization"]
            )
        ]
        mock_justify_runner.check_fast_filter.return_value = (True, "Your answer is too vague.")
        
        # Mock LLM service to not be in mock mode
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm
        mock_llm.use_mock = False  # Ensure production path
        
        # Mock editor writes answer
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_file.return_value.readlines.return_value = [
            "# Justify Question\n",
            "Why softmax?\n",
            "# Your Answer\n",
            "It's good.\n"  # Vague answer
        ]
        
        # Execute
        result = _submit_justify_stage(mock_state_mgr, mock_curr_mgr, progress, manifest)
        
        # Verify
        assert result is False
        mock_justify_runner.check_fast_filter.assert_called_once()
        mock_state_mgr.save.assert_not_called()
    
    @patch('engine.main.LLMService')
    @patch('subprocess.run')
    @patch('engine.main.JustifyRunner')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.unlink')
    def test_justify_llm_correct_advances_stage(
        self, mock_unlink, mock_file, mock_justify_cls, mock_subprocess, mock_llm_cls
    ):
        """Should advance to harden stage when LLM marks answer correct."""
        from engine.main import _submit_justify_stage
        
        # Setup
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        progress = MagicMock(spec=UserProgress)
        progress.curriculum_id = "test"
        progress.current_module_index = 0
        progress.current_stage = "justify"
        
        manifest = CurriculumManifest(
            curriculum_name="test",
            author="test",
            version="1.0.0",
            modules=[ModuleMetadata(id="softmax", name="Softmax", path="modules/softmax")]
        )
        
        mock_justify_runner = MagicMock()
        mock_justify_cls.return_value = mock_justify_runner
        mock_justify_runner.load_questions.return_value = [
            JustifyQuestion(
                id="q1",
                question="Why softmax?",
                model_answer="Softmax applies exp then normalizes",
                failure_modes=[],
                required_concepts=["exp", "normalization"]
            )
        ]
        mock_justify_runner.check_fast_filter.return_value = (False, "")
        
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm
        mock_llm.use_mock = False  # Ensure production path (not auto-pass)
        mock_llm.evaluate_justification.return_value = LLMEvaluationResponse(
            is_correct=True,
            feedback="Excellent explanation!"
        )
        
        # Mock editor writes answer
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_file.return_value.readlines.return_value = [
            "# Justify Question\n",
            "Why softmax?\n",
            "# Your Answer\n",
            "Softmax applies exp to convert logits to positive values, then normalizes by the sum.\n"
        ]
        
        # Execute
        result = _submit_justify_stage(mock_state_mgr, mock_curr_mgr, progress, manifest)
        
        # Verify
        assert result is True
        progress.mark_stage_complete.assert_called_once_with("justify")
        mock_state_mgr.save.assert_called_once()
    
    @patch('engine.main.LLMService')
    @patch('subprocess.run')
    @patch('engine.main.JustifyRunner')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.unlink')
    def test_justify_llm_incorrect_does_not_advance(
        self, mock_unlink, mock_file, mock_justify_cls, mock_subprocess, mock_llm_cls
    ):
        """Should not advance when LLM marks answer incorrect."""
        from engine.main import _submit_justify_stage
        
        # Setup
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        progress = MagicMock(spec=UserProgress)
        progress.curriculum_id = "test"
        progress.current_module_index = 0
        progress.current_stage = "justify"
        
        manifest = CurriculumManifest(
            curriculum_name="test",
            author="test",
            version="1.0.0",
            modules=[ModuleMetadata(id="softmax", name="Softmax", path="modules/softmax")]
        )
        
        mock_justify_runner = MagicMock()
        mock_justify_cls.return_value = mock_justify_runner
        mock_justify_runner.load_questions.return_value = [
            JustifyQuestion(
                id="q1",
                question="Why softmax?",
                model_answer="Softmax applies exp then normalizes",
                failure_modes=[],
                required_concepts=["exp", "normalization"]
            )
        ]
        mock_justify_runner.check_fast_filter.return_value = (False, "")
        
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm
        mock_llm.use_mock = False  # Ensure production path (not auto-pass)
        mock_llm.evaluate_justification.return_value = LLMEvaluationResponse(
            is_correct=False,
            feedback="You're missing the normalization step."
        )
        
        # Mock editor writes answer
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_file.return_value.readlines.return_value = [
            "# Justify Question\n",
            "Why softmax?\n",
            "# Your Answer\n",
            "Softmax just applies exp.\n"
        ]
        
        # Execute
        result = _submit_justify_stage(mock_state_mgr, mock_curr_mgr, progress, manifest)
        
        # Verify
        assert result is False
        mock_state_mgr.save.assert_not_called()


class TestSubmitHardenStage:
    """Tests for _submit_harden_stage handler."""
    
    @patch('engine.main.Path')
    @patch('engine.main.ValidationSubsystem')
    @patch('shutil.copy2')
    def test_harden_success_advances_module(self, mock_copy, mock_validator_cls, mock_path_cls):
        """Should advance to next module on successful harden fix."""
        from engine.main import _submit_harden_stage
        
        # Setup
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        progress = MagicMock(spec=UserProgress)
        progress.curriculum_id = "test"
        progress.current_module_index = 0
        progress.current_stage = "harden"
        
        manifest = CurriculumManifest(
            curriculum_name="test",
            author="test",
            version="1.0.0",
            modules=[ModuleMetadata(id="softmax", name="Softmax", path="modules/softmax")]
        )
        
        # Mock filesystem paths
        mock_shadow_worktree = MagicMock()
        mock_harden_workspace = MagicMock()
        mock_harden_workspace.exists.return_value = True
        mock_shadow_dest = MagicMock()
        mock_shadow_dest.parent.mkdir = MagicMock()
        
        def path_side_effect(arg):
            if arg == '.mastery_engine_worktree':
                return mock_shadow_worktree
            elif 'harden' in str(arg):
                return mock_harden_workspace
            else:
                return mock_shadow_dest
        
        mock_path_cls.side_effect = path_side_effect
        mock_shadow_worktree.__truediv__ = lambda self, other: mock_harden_workspace if 'workspace' in str(other) else mock_shadow_dest
        mock_harden_workspace.__truediv__ = lambda self, other: MagicMock()
        
        mock_validator = MagicMock()
        mock_validator_cls.return_value = mock_validator
        mock_validator.execute.return_value = ValidationResult(
            exit_code=0,
            stdout="All tests passed",
            stderr="",
            performance_seconds=1.0
        )
        
        # Execute
        result = _submit_harden_stage(mock_state_mgr, mock_curr_mgr, progress, manifest)
        
        # Verify
        assert result is True
        progress.mark_stage_complete.assert_called_once_with("harden")
        mock_state_mgr.save.assert_called_once()


class TestUnifiedSubmitCommand:
    """Tests for unified submit() command routing."""
    
    @patch('engine.main.require_shadow_worktree')
    @patch('engine.main._submit_build_stage')
    @patch('engine.main._load_curriculum_state')
    @patch('engine.main._check_curriculum_complete')
    def test_submit_routes_to_build_handler(
        self, mock_check_complete, mock_load, mock_build, mock_shadow
    ):
        """Should route to build handler when in build stage."""
        from engine.main import submit
        
        # Setup
        mock_check_complete.return_value = False
        mock_load.return_value = (
            MagicMock(),
            MagicMock(),
            UserProgress(curriculum_id="test", current_module_index=0, current_stage="build"),
            CurriculumManifest(
                curriculum_name="test",
                author="test",
                version="1.0.0",
                modules=[ModuleMetadata(id="mod1", name="Module 1", path="modules/mod1")]
            )
        )
        mock_build.return_value = True
        
        # Execute
        submit()
        
        # Verify
        mock_build.assert_called_once()
    
    @patch('engine.main.require_shadow_worktree')
    @patch('engine.main._submit_justify_stage')
    @patch('engine.main._load_curriculum_state')
    @patch('engine.main._check_curriculum_complete')
    def test_submit_routes_to_justify_handler(
        self, mock_check_complete, mock_load, mock_justify, mock_shadow
    ):
        """Should route to justify handler when in justify stage."""
        from engine.main import submit
        
        # Setup
        mock_check_complete.return_value = False
        mock_load.return_value = (
            MagicMock(),
            MagicMock(),
            UserProgress(curriculum_id="test", current_module_index=0, current_stage="justify"),
            CurriculumManifest(
                curriculum_name="test",
                author="test",
                version="1.0.0",
                modules=[ModuleMetadata(id="mod1", name="Module 1", path="modules/mod1")]
            )
        )
        mock_justify.return_value = True
        
        # Execute
        submit()
        
        # Verify
        mock_justify.assert_called_once()
    
    @patch('engine.main.require_shadow_worktree')
    @patch('engine.main._submit_harden_stage')
    @patch('engine.main._load_curriculum_state')
    @patch('engine.main._check_curriculum_complete')
    def test_submit_routes_to_harden_handler(
        self, mock_check_complete, mock_load, mock_harden, mock_shadow
    ):
        """Should route to harden handler when in harden stage."""
        from engine.main import submit
        
        # Setup
        mock_check_complete.return_value = False
        mock_load.return_value = (
            MagicMock(),
            MagicMock(),
            UserProgress(curriculum_id="test", current_module_index=0, current_stage="harden"),
            CurriculumManifest(
                curriculum_name="test",
                author="test",
                version="1.0.0",
                modules=[ModuleMetadata(id="mod1", name="Module 1", path="modules/mod1")]
            )
        )
        mock_harden.return_value = True
        
        # Execute
        submit()
        
        # Verify
        mock_harden.assert_called_once()
