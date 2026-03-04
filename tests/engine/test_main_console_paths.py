"""Tests for console output paths in main.py."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from engine.main import _submit_build_stage
from engine.schemas import UserProgress, CurriculumManifest, ModuleMetadata, CurriculumType


class TestBuildStageConsoleOutput:
    """Tests for console output in build stage."""
    
    @patch('engine.main.console')
    @patch('engine.main.ValidationSubsystem')
    def test_performance_speedup_message(self, mock_val_cls, mock_console):
        """Should show speedup message when performance is good."""
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        
        progress = UserProgress(
            curriculum_id="test",
            current_module_index=0,
            current_stage="build"
        )
        manifest = CurriculumManifest(
            curriculum_name="Test",
            curriculum_type=CurriculumType.LINEAR,
            author="Test",
            version="1.0.0",
            modules=[ModuleMetadata(
                id="mod1",
                name="Module 1",
                path="mod1",
                baseline_perf_seconds=10.0  # Baseline is 10 seconds
            )]
        )
        
        mock_curr_mgr.get_validator_path.return_value = Path("/tmp/validator.sh")
        mock_validator = MagicMock()
        mock_val_cls.return_value = mock_validator
        
        # Mock result with great performance (2.5x speedup)
        mock_result = MagicMock()
        mock_result.exit_code = 0
        mock_result.performance_seconds = 4.0  # 2.5x faster than baseline
        mock_validator.execute.return_value = mock_result
        
        result = _submit_build_stage(mock_state_mgr, mock_curr_mgr, progress, manifest)
        
        assert result is True
        # Should print speedup message
        assert any('faster' in str(call).lower() or 'speedup' in str(call).lower() 
                   for call in mock_console.print.call_args_list)
    
    @patch('engine.main.console')
    @patch('engine.main.ValidationSubsystem')
    def test_performance_display_without_baseline(self, mock_val_cls, mock_console):
        """Should show performance without comparison when no baseline."""
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        
        progress = UserProgress(
            curriculum_id="test",
            current_module_index=0,
            current_stage="build"
        )
        manifest = CurriculumManifest(
            curriculum_name="Test",
            curriculum_type=CurriculumType.LINEAR,
            author="Test",
            version="1.0.0",
            modules=[ModuleMetadata(
                id="mod1",
                name="Module 1",
                path="mod1"
                # No baseline_perf_seconds
            )]
        )
        
        mock_curr_mgr.get_validator_path.return_value = Path("/tmp/validator.sh")
        mock_validator = MagicMock()
        mock_val_cls.return_value = mock_validator
        
        mock_result = MagicMock()
        mock_result.exit_code = 0
        mock_result.performance_seconds = 2.5
        mock_validator.execute.return_value = mock_result
        
        result = _submit_build_stage(mock_state_mgr, mock_curr_mgr, progress, manifest)
        
        assert result is True
        # Should still print performance
        assert mock_console.print.called
    
    @patch('engine.main.console')
    @patch('engine.main.ValidationSubsystem')
    def test_failure_shows_stderr(self, mock_val_cls, mock_console):
        """Should show stderr on validation failure."""
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        
        progress = UserProgress(
            curriculum_id="test",
            current_module_index=0,
            current_stage="build"
        )
        manifest = CurriculumManifest(
            curriculum_name="Test",
            curriculum_type=CurriculumType.LINEAR,
            author="Test",
            version="1.0.0",
            modules=[ModuleMetadata(id="mod1", name="Module 1", path="mod1")]
        )
        
        mock_curr_mgr.get_validator_path.return_value = Path("/tmp/validator.sh")
        mock_validator = MagicMock()
        mock_val_cls.return_value = mock_validator
        
        mock_result = MagicMock()
        mock_result.exit_code = 1
        mock_result.stderr = "Error: test failed\n"
        mock_result.stdout = ""
        mock_validator.execute.return_value = mock_result
        
        result = _submit_build_stage(mock_state_mgr, mock_curr_mgr, progress, manifest)
        
        assert result is False
        # Should have printed error output
        print_calls = [str(call) for call in mock_console.print.call_args_list]
        assert any('failed' in call.lower() for call in print_calls)
