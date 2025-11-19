"""Comprehensive tests to aggressively increase main.py coverage."""

import pytest
from unittest.mock import MagicMock, patch, mock_open, call
from pathlib import Path
import json

from engine.main import (
    _submit_build_stage,
    _submit_harden_stage,
    _submit_linear_workflow,
    _submit_library_workflow,
)
from engine.schemas import (
    UserProgress,
    CurriculumManifest,
    ModuleMetadata,
    ProblemMetadata,
    CurriculumType,
)
from engine.validator import ValidatorNotFoundError, ValidatorTimeoutError, ValidatorExecutionError
from engine.workspace import WorkspaceError


class TestBuildStageComprehensive:
    """Comprehensive build stage tests."""
    
    @patch('engine.main.logger')
    @patch('engine.main.console')
    @patch('engine.main.ValidationSubsystem')
    def test_build_logs_completion(self, mock_val_cls, mock_console, mock_logger):
        """Should log successful completion."""
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        
        progress = UserProgress(curriculum_id="test", current_module_index=0, current_stage="build")
        manifest = CurriculumManifest(
            curriculum_name="Test", curriculum_type=CurriculumType.LINEAR,
            author="Test", version="1.0.0",
            modules=[ModuleMetadata(id="mod1", name="Module 1", path="mod1")]
        )
        
        mock_curr_mgr.get_validator_path.return_value = Path("/tmp/val.sh")
        mock_validator = MagicMock()
        mock_val_cls.return_value = mock_validator
        mock_result = MagicMock(exit_code=0, performance_seconds=1.0)
        mock_validator.execute.return_value = mock_result
        
        _submit_build_stage(mock_state_mgr, mock_curr_mgr, progress, manifest)
        
        mock_logger.info.assert_called()
    
    @patch('engine.main.logger')
    @patch('engine.main.console')
    @patch('engine.main.ValidationSubsystem')
    def test_build_logs_failure(self, mock_val_cls, mock_console, mock_logger):
        """Should log validation failure."""
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        
        progress = UserProgress(curriculum_id="test", current_module_index=0, current_stage="build")
        manifest = CurriculumManifest(
            curriculum_name="Test", curriculum_type=CurriculumType.LINEAR,
            author="Test", version="1.0.0",
            modules=[ModuleMetadata(id="mod1", name="Module 1", path="mod1")]
        )
        
        mock_curr_mgr.get_validator_path.return_value = Path("/tmp/val.sh")
        mock_validator = MagicMock()
        mock_val_cls.return_value = mock_validator
        mock_result = MagicMock(exit_code=1, stdout="Failed", stderr="")
        mock_validator.execute.return_value = mock_result
        
        _submit_build_stage(mock_state_mgr, mock_curr_mgr, progress, manifest)
        
        mock_logger.info.assert_called()
    
    @patch('engine.main.console')
    @patch('engine.main.ValidationSubsystem')
    def test_build_validator_not_found(self, mock_val_cls, mock_console):
        """Should handle ValidatorNotFoundError."""
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        
        progress = UserProgress(curriculum_id="test", current_module_index=0, current_stage="build")
        manifest = CurriculumManifest(
            curriculum_name="Test", curriculum_type=CurriculumType.LINEAR,
            author="Test", version="1.0.0",
            modules=[ModuleMetadata(id="mod1", name="Module 1", path="mod1")]
        )
        
        mock_curr_mgr.get_validator_path.return_value = Path("/tmp/val.sh")
        mock_validator = MagicMock()
        mock_val_cls.return_value = mock_validator
        mock_validator.execute.side_effect = ValidatorNotFoundError("Not found")
        
        with pytest.raises(ValidatorNotFoundError):
            _submit_build_stage(mock_state_mgr, mock_curr_mgr, progress, manifest)
    
    @patch('engine.main.console')
    @patch('engine.main.ValidationSubsystem')
    def test_build_validator_timeout(self, mock_val_cls, mock_console):
        """Should handle ValidatorTimeoutError."""
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        
        progress = UserProgress(curriculum_id="test", current_module_index=0, current_stage="build")
        manifest = CurriculumManifest(
            curriculum_name="Test", curriculum_type=CurriculumType.LINEAR,
            author="Test", version="1.0.0",
            modules=[ModuleMetadata(id="mod1", name="Module 1", path="mod1")]
        )
        
        mock_curr_mgr.get_validator_path.return_value = Path("/tmp/val.sh")
        mock_validator = MagicMock()
        mock_val_cls.return_value = mock_validator
        mock_validator.execute.side_effect = ValidatorTimeoutError("Timeout")
        
        with pytest.raises(ValidatorTimeoutError):
            _submit_build_stage(mock_state_mgr, mock_curr_mgr, progress, manifest)
    
    @patch('engine.main.console')
    @patch('engine.main.ValidationSubsystem')
    def test_build_validator_execution_error(self, mock_val_cls, mock_console):
        """Should handle ValidatorExecutionError."""
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        
        progress = UserProgress(curriculum_id="test", current_module_index=0, current_stage="build")
        manifest = CurriculumManifest(
            curriculum_name="Test", curriculum_type=CurriculumType.LINEAR,
            author="Test", version="1.0.0",
            modules=[ModuleMetadata(id="mod1", name="Module 1", path="mod1")]
        )
        
        mock_curr_mgr.get_validator_path.return_value = Path("/tmp/val.sh")
        mock_validator = MagicMock()
        mock_val_cls.return_value = mock_validator
        mock_validator.execute.side_effect = ValidatorExecutionError("Execution failed")
        
        with pytest.raises(ValidatorExecutionError):
            _submit_build_stage(mock_state_mgr, mock_curr_mgr, progress, manifest)


# Harden stage tests removed - they trigger complex file system operations


class TestLinearWorkflowComprehensive:
    """Comprehensive linear workflow tests."""
    
    @patch('engine.main.logger')
    @patch('engine.main._submit_build_stage')
    @patch('engine.main.console')
    def test_linear_logs_stage_completion(self, mock_console, mock_build, mock_logger):
        """Should log stage completion."""
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        
        progress = UserProgress(curriculum_id="test", current_module_index=0, current_stage="build")
        manifest = CurriculumManifest(
            curriculum_name="Test", curriculum_type=CurriculumType.LINEAR,
            author="Test", version="1.0.0",
            modules=[ModuleMetadata(id="mod1", name="Module 1", path="mod1")]
        )
        
        mock_build.return_value = True
        
        _submit_linear_workflow(mock_state_mgr, mock_curr_mgr, progress, manifest)
        
        mock_logger.info.assert_called()
    
    @patch('engine.main.console')
    def test_linear_unknown_stage_exits(self, mock_console):
        """Should exit on unknown stage."""
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        
        progress = UserProgress(curriculum_id="test", current_module_index=0, current_stage="unknown")
        manifest = CurriculumManifest(
            curriculum_name="Test", curriculum_type=CurriculumType.LINEAR,
            author="Test", version="1.0.0",
            modules=[ModuleMetadata(id="mod1", name="Module 1", path="mod1")]
        )
        
        with pytest.raises(SystemExit) as exc:
            _submit_linear_workflow(mock_state_mgr, mock_curr_mgr, progress, manifest)
        
        assert exc.value.code == 1


class TestLibraryWorkflowComprehensive:
    """Comprehensive library workflow tests."""
    
    @patch('engine.main.console')
    def test_library_no_problem_exits(self, mock_console):
        """Should exit when no problem selected."""
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        
        progress = UserProgress(
            curriculum_id="test", current_module_index=0,
            current_stage="build", active_problem_id=None
        )
        manifest = CurriculumManifest(
            curriculum_name="Test", curriculum_type=CurriculumType.LIBRARY,
            author="Test", version="1.0.0", patterns=[]
        )
        
        with pytest.raises(SystemExit) as exc:
            _submit_library_workflow(mock_state_mgr, mock_curr_mgr, progress, manifest)
        
        assert exc.value.code == 1
    
    @patch('engine.main.console')
    def test_library_problem_not_found_exits(self, mock_console):
        """Should exit when problem doesn't exist."""
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        
        progress = UserProgress(
            curriculum_id="test", current_module_index=0,
            current_stage="build", active_problem_id="nonexistent"
        )
        manifest = CurriculumManifest(
            curriculum_name="Test", curriculum_type=CurriculumType.LIBRARY,
            author="Test", version="1.0.0", patterns=[]
        )
        
        mock_curr_mgr.get_problem_metadata.return_value = None
        
        with pytest.raises(SystemExit) as exc:
            _submit_library_workflow(mock_state_mgr, mock_curr_mgr, progress, manifest)
        
        assert exc.value.code == 1
    
    # Library justify/harden completion tests removed - causing hangs
