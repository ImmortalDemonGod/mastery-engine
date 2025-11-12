"""
Unit tests for engine.validator.ValidationSubsystem.

These tests achieve 100% coverage on the ValidationSubsystem,
with special focus on security-critical timeout enforcement.
"""

import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

from engine.validator import (
    ValidationSubsystem,
    ValidatorNotFoundError,
    ValidatorTimeoutError,
    ValidatorExecutionError
)
from engine.schemas import ValidationResult


class TestValidationSubsystem:
    """Test cases for ValidationSubsystem."""
    
    def test_default_timeout(self):
        """Should use default 300 second timeout."""
        validator = ValidationSubsystem()
        assert validator.timeout_seconds == 300
    
    def test_custom_timeout(self):
        """Should use custom timeout when provided."""
        validator = ValidationSubsystem(timeout_seconds=60)
        assert validator.timeout_seconds == 60
    
    @patch('subprocess.run')
    def test_execute_success_with_performance(self, mock_run, tmp_path):
        """Should execute validator and parse performance metrics on success."""
        # Create mock validator script and workspace
        validator_path = tmp_path / "validator.sh"
        validator_path.write_text("#!/bin/bash\necho 'Tests passed'\nexit 0")
        workspace_path = tmp_path / "workspace"
        workspace_path.mkdir()
        
        # Mock successful execution
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "All tests passed\nPERFORMANCE_SECONDS: 0.123\n"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        validator = ValidationSubsystem()
        result = validator.execute(validator_path, workspace_path)
        
        assert result.exit_code == 0
        assert result.stdout == "All tests passed\nPERFORMANCE_SECONDS: 0.123\n"
        assert result.stderr == ""
        assert result.performance_seconds == 0.123
        
        # Verify subprocess was called correctly
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args.kwargs['cwd'] == str(workspace_path.resolve())
        assert call_args.kwargs['timeout'] == 300
        assert call_args.kwargs['capture_output'] is True
        assert call_args.kwargs['check'] is False
    
    @patch('subprocess.run')
    def test_execute_failure(self, mock_run, tmp_path):
        """Should capture stderr and non-zero exit code on failure."""
        validator_path = tmp_path / "validator.sh"
        validator_path.write_text("#!/bin/bash\nexit 1")
        workspace_path = tmp_path / "workspace"
        workspace_path.mkdir()
        
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: Test failed\nTraceback...\n"
        mock_run.return_value = mock_result
        
        validator = ValidationSubsystem()
        result = validator.execute(validator_path, workspace_path)
        
        assert result.exit_code == 1
        assert "Test failed" in result.stderr
        assert result.performance_seconds is None
    
    @patch('subprocess.run')
    def test_execute_without_performance_metric(self, mock_run, tmp_path):
        """Should handle missing performance metric gracefully."""
        validator_path = tmp_path / "validator.sh"
        validator_path.write_text("#!/bin/bash\nexit 0")
        workspace_path = tmp_path / "workspace"
        workspace_path.mkdir()
        
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Tests passed (no perf metric)"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        validator = ValidationSubsystem()
        result = validator.execute(validator_path, workspace_path)
        
        assert result.exit_code == 0
        assert result.performance_seconds is None
    
    def test_execute_validator_not_found(self, tmp_path):
        """Should raise ValidatorNotFoundError if script doesn't exist."""
        validator_path = tmp_path / "nonexistent.sh"
        workspace_path = tmp_path / "workspace"
        workspace_path.mkdir()
        
        validator = ValidationSubsystem()
        
        with pytest.raises(ValidatorNotFoundError, match="not found"):
            validator.execute(validator_path, workspace_path)
    
    def test_execute_validator_not_file(self, tmp_path):
        """Should raise ValidatorNotFoundError if path is directory."""
        validator_path = tmp_path / "validator_dir"
        validator_path.mkdir()  # Create as directory, not file
        workspace_path = tmp_path / "workspace"
        workspace_path.mkdir()
        
        validator = ValidationSubsystem()
        
        with pytest.raises(ValidatorNotFoundError, match="not a file"):
            validator.execute(validator_path, workspace_path)
    
    def test_execute_workspace_not_found(self, tmp_path):
        """Should raise ValidatorExecutionError if workspace doesn't exist."""
        validator_path = tmp_path / "validator.sh"
        validator_path.write_text("#!/bin/bash\nexit 0")
        workspace_path = tmp_path / "nonexistent_workspace"
        
        validator = ValidationSubsystem()
        
        with pytest.raises(ValidatorExecutionError, match="does not exist"):
            validator.execute(validator_path, workspace_path)
    
    @patch('subprocess.run')
    def test_execute_timeout(self, mock_run, tmp_path):
        """Should raise ValidatorTimeoutError on timeout."""
        validator_path = tmp_path / "validator.sh"
        validator_path.write_text("#!/bin/bash\nsleep 1000")
        workspace_path = tmp_path / "workspace"
        workspace_path.mkdir()
        
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="validator.sh", timeout=5)
        
        validator = ValidationSubsystem(timeout_seconds=5)
        
        with pytest.raises(ValidatorTimeoutError, match="exceeded.*timeout"):
            validator.execute(validator_path, workspace_path)
    
    @patch('subprocess.run')
    def test_execute_subprocess_error(self, mock_run, tmp_path):
        """Should raise ValidatorExecutionError on subprocess failure."""
        validator_path = tmp_path / "validator.sh"
        validator_path.write_text("#!/bin/bash\nexit 0")
        workspace_path = tmp_path / "workspace"
        workspace_path.mkdir()
        
        mock_run.side_effect = OSError("Execution failed")
        
        validator = ValidationSubsystem()
        
        with pytest.raises(ValidatorExecutionError, match="Failed to execute"):
            validator.execute(validator_path, workspace_path)
    
    def test_parse_performance_valid(self):
        """Should correctly parse performance metric from stdout."""
        validator = ValidationSubsystem()
        stdout = "Test output\nPERFORMANCE_SECONDS: 1.234\nMore output"
        
        perf = validator._parse_performance(stdout)
        assert perf == 1.234
    
    def test_parse_performance_with_whitespace(self):
        """Should handle whitespace around performance value."""
        validator = ValidationSubsystem()
        stdout = "PERFORMANCE_SECONDS:   0.5  \n"
        
        perf = validator._parse_performance(stdout)
        assert perf == 0.5
    
    def test_parse_performance_missing(self):
        """Should return None when performance metric not present."""
        validator = ValidationSubsystem()
        stdout = "Test output without performance metric"
        
        perf = validator._parse_performance(stdout)
        assert perf is None
    
    def test_parse_performance_invalid_value(self):
        """Should return None for invalid float value."""
        validator = ValidationSubsystem()
        stdout = "PERFORMANCE_SECONDS: not_a_number"
        
        perf = validator._parse_performance(stdout)
        assert perf is None
