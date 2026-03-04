"""
Unit tests for main.py helper functions.

Targets isolated, testable functions without command workflows.
"""

import pytest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

from engine.main import (
    require_shadow_worktree,
    _load_curriculum_state,
    _check_curriculum_complete,
   _show_linear_status,
    _show_library_status,
)
from engine.schemas import UserProgress, CurriculumManifest, ModuleMetadata, CurriculumType
from engine.state import StateFileCorruptedError


class TestRequireShadowWorktree:
    """Tests for require_shadow_worktree helper."""
    
    @patch('engine.main.SHADOW_WORKTREE_DIR')
    def test_worktree_exists(self, mock_dir):
        """Should return path when worktree exists."""
        mock_dir.exists.return_value = True
        result = require_shadow_worktree()
        assert result == mock_dir
    
    @patch('engine.main.SHADOW_WORKTREE_DIR')
    def test_worktree_missing_raises(self, mock_dir):
        """Should raise SystemExit when worktree doesn't exist."""
        mock_dir.exists.return_value = False
        with pytest.raises(SystemExit):
            require_shadow_worktree()


class TestLoadCurriculumState:
    """Tests for _load_curriculum_state helper."""
    
    @patch('engine.main.CurriculumManager')
    @patch('engine.main.StateManager')
    def test_load_success(self, mock_state_cls, mock_curr_cls):
        """Should successfully load state and curriculum."""
        # Setup state
        mock_state_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        progress = UserProgress(
            curriculum_id="test_curr",
            current_module_index=0,
            current_stage="build"
        )
        mock_state_mgr.load.return_value = progress
        mock_state_mgr.progress = progress
        
        # Setup curriculum
        mock_curr_mgr = MagicMock()
        mock_curr_cls.return_value = mock_curr_mgr
        manifest = CurriculumManifest(
            curriculum_name="Test",
            curriculum_type=CurriculumType.LINEAR,
            author="Test",
            version="1.0.0",
            modules=[ModuleMetadata(id="mod1", name="Module 1", path="mod1")]
        )
        mock_curr_mgr.load_manifest.return_value = manifest
        
        state_mgr, curr_mgr, progress_out, manifest_out = _load_curriculum_state()
        
        assert state_mgr == mock_state_mgr
        assert curr_mgr == mock_curr_mgr
        assert progress_out == progress
        assert manifest_out == manifest
    
    # Note: _load_curriculum_state doesn't handle None progress - would need fix in main.py


class TestCheckCurriculumComplete:
    """Tests for _check_curriculum_complete helper."""
    
    def test_all_modules_completed(self):
        """Should return True when all modules are done."""
        progress = UserProgress(
            curriculum_id="test",
            current_module_index=2,
            current_stage="build",
            completed_modules=["mod1", "mod2"]
        )
        manifest = CurriculumManifest(
            curriculum_name="Test",
            curriculum_type=CurriculumType.LINEAR,
            author="Test",
            version="1.0.0",
            modules=[
                ModuleMetadata(id="mod1", name="Module 1", path="mod1"),
                ModuleMetadata(id="mod2", name="Module 2", path="mod2"),
            ]
        )
        
        assert _check_curriculum_complete(progress, manifest) is True
    
    def test_modules_remaining(self):
        """Should return False when modules remain."""
        progress = UserProgress(
            curriculum_id="test",
            current_module_index=1,
            current_stage="build",
            completed_modules=["mod1"]
        )
        manifest = CurriculumManifest(
            curriculum_name="Test",
            curriculum_type=CurriculumType.LINEAR,
            author="Test",
            version="1.0.0",
            modules=[
                ModuleMetadata(id="mod1", name="Module 1", path="mod1"),
                ModuleMetadata(id="mod2", name="Module 2", path="mod2"),
            ]
        )
        
        assert _check_curriculum_complete(progress, manifest) is False


class TestShowLinearStatus:
    """Tests for _show_linear_status display helper."""
    
    @patch('engine.main.console')
    def test_display_current_module(self, mock_console):
        """Should display status for current module."""
        progress = UserProgress(
            curriculum_id="test",
            current_module_index=1,
            current_stage="justify",
            completed_modules=["mod1"]
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
        
        _show_linear_status(progress, manifest)
        
        # Should have printed something
        assert mock_console.print.called
    
    @patch('engine.main.console')
    def test_display_completed_curriculum(self, mock_console):
        """Should display completion message."""
        progress = UserProgress(
            curriculum_id="test",
            current_module_index=2,
            current_stage="build",
            completed_modules=["mod1", "mod2"]
        )
        manifest = CurriculumManifest(
            curriculum_name="Test",
            curriculum_type=CurriculumType.LINEAR,
            author="Test",
            version="1.0.0",
            modules=[
                ModuleMetadata(id="mod1", name="Module 1", path="mod1"),
                ModuleMetadata(id="mod2", name="Module 2", path="mod2"),
            ]
        )
        
        _show_linear_status(progress, manifest)
        
        assert mock_console.print.called


class TestShowLibraryStatus:
    """Tests for _show_library_status display helper."""
    
    @patch('engine.main.console')
    @patch('engine.main.CurriculumManager')
    def test_no_problem_selected(self, mock_curr_cls, mock_console):
        """Should prompt to select a problem."""
        mock_curr_mgr = MagicMock()
        progress = UserProgress(
            curriculum_id="test",
            current_module_index=0,
            current_stage="build",
            active_problem_id=None
        )
        manifest = CurriculumManifest(
            curriculum_name="Test",
            curriculum_type=CurriculumType.LIBRARY,
            author="Test",
            version="1.0.0",
            patterns=[]
        )
        
        _show_library_status(progress, manifest, mock_curr_mgr)
        
        assert mock_console.print.called
