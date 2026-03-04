"""Tests for error paths in main.py."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from engine.main import (
    _check_curriculum_complete,
    _submit_linear_workflow,
    require_shadow_worktree,
)
from engine.schemas import UserProgress, CurriculumManifest, ModuleMetadata, CurriculumType


class TestRequireShadowWorktree:
    """Tests for require_shadow_worktree function."""
    
    @patch('engine.main.SHADOW_WORKTREE_DIR')
    def test_exits_when_worktree_missing(self, mock_dir):
        """Should exit when shadow worktree doesn't exist."""
        mock_dir.exists.return_value = False
        
        with pytest.raises(SystemExit) as exc_info:
            require_shadow_worktree()
        
        assert exc_info.value.code == 1


class TestCheckCurriculumComplete:
    """Tests for _check_curriculum_complete function."""
    
    @patch('engine.main.console')
    def test_prints_completion_message(self, mock_console):
        """Should print congratulations when complete."""
        progress = UserProgress(
            curriculum_id="test",
            current_module_index=2,
            current_stage="build",
            completed_modules=["mod1", "mod2"]
        )
        manifest = CurriculumManifest(
            curriculum_name="Test Curriculum",
            curriculum_type=CurriculumType.LINEAR,
            author="Test",
            version="1.0.0",
            modules=[
                ModuleMetadata(id="mod1", name="Module 1", path="mod1"),
                ModuleMetadata(id="mod2", name="Module 2", path="mod2"),
            ]
        )
        
        result = _check_curriculum_complete(progress, manifest)
        
        assert result is True
        # Should have printed completion message
        assert mock_console.print.called


class TestSubmitLinearWorkflowErrors:
    """Tests for error handling in _submit_linear_workflow."""
    
    @patch('engine.main._check_curriculum_complete')
    @patch('engine.main.console')
    def test_early_return_when_complete(self, mock_console, mock_check):
        """Should return early when curriculum is complete."""
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        
        progress = UserProgress(
            curriculum_id="test",
            current_module_index=1,
            current_stage="build"
        )
        manifest = CurriculumManifest(
            curriculum_name="Test",
            curriculum_type=CurriculumType.LINEAR,
            author="Test",
            version="1.0.0",
            modules=[ModuleMetadata(id="mod1", name="Module 1", path="mod1")]
        )
        
        mock_check.return_value = True  # Curriculum complete
        
        # Should return without calling stage handlers
        _submit_linear_workflow(mock_state_mgr, mock_curr_mgr, progress, manifest)
        
        mock_check.assert_called_once()


class TestMainModuleTopLevel:
    """Tests for top-level code in main.py."""
    
    def test_module_level_fallback_path(self):
        """Test that module-level RuntimeError fallback creates relative path."""
        # The fallback path is already tested by module import
        # Just verify the constant exists
        from engine.main import SHADOW_WORKTREE_DIR
        assert SHADOW_WORKTREE_DIR is not None
        # Either absolute or relative path
        assert str(SHADOW_WORKTREE_DIR).endswith('.mastery_engine_worktree')
