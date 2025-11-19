"""
Systematic tests for main.py workflow orchestration.

Targets: _submit_linear_workflow, _submit_library_workflow, and workflow paths.
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from engine.main import _submit_linear_workflow, _submit_library_workflow
from engine.schemas import UserProgress, CurriculumManifest, ModuleMetadata, CurriculumType


class TestLinearWorkflow:
    """Tests for _submit_linear_workflow orchestration."""
    
    @patch('engine.main._submit_build_stage')
    @patch('engine.main.console')
    def test_routes_to_build_stage(self, mock_console, mock_build):
        """Should route to build stage handler when in build."""
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
        
        mock_build.return_value = True
        
        _submit_linear_workflow(mock_state_mgr, mock_curr_mgr, progress, manifest)
        
        mock_build.assert_called_once_with(mock_state_mgr, mock_curr_mgr, progress, manifest)
    
    @patch('engine.main._submit_justify_stage')
    @patch('engine.main.console')
    def test_routes_to_justify_stage(self, mock_console, mock_justify):
        """Should route to justify stage handler when in justify."""
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        
        progress = UserProgress(
            curriculum_id="test",
            current_module_index=0,
            current_stage="justify"
        )
        manifest = CurriculumManifest(
            curriculum_name="Test",
            curriculum_type=CurriculumType.LINEAR,
            author="Test",
            version="1.0.0",
            modules=[ModuleMetadata(id="mod1", name="Module 1", path="mod1")]
        )
        
        mock_justify.return_value = True
        
        _submit_linear_workflow(mock_state_mgr, mock_curr_mgr, progress, manifest)
        
        mock_justify.assert_called_once_with(mock_state_mgr, mock_curr_mgr, progress, manifest)
    
    @patch('engine.main._submit_harden_stage')
    @patch('engine.main.console')
    def test_routes_to_harden_stage(self, mock_console, mock_harden):
        """Should route to harden stage handler when in harden."""
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        
        progress = UserProgress(
            curriculum_id="test",
            current_module_index=0,
            current_stage="harden"
        )
        manifest = CurriculumManifest(
            curriculum_name="Test",
            curriculum_type=CurriculumType.LINEAR,
            author="Test",
            version="1.0.0",
            modules=[ModuleMetadata(id="mod1", name="Module 1", path="mod1")]
        )
        
        mock_harden.return_value = True
        
        _submit_linear_workflow(mock_state_mgr, mock_curr_mgr, progress, manifest)
        
        mock_harden.assert_called_once_with(mock_state_mgr, mock_curr_mgr, progress, manifest)
    
    @patch('engine.main.console')
    def test_exits_on_unknown_stage(self, mock_console):
        """Should exit with error on unknown stage."""
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        
        progress = UserProgress(
            curriculum_id="test",
            current_module_index=0,
            current_stage="invalid_stage"
        )
        manifest = CurriculumManifest(
            curriculum_name="Test",
            curriculum_type=CurriculumType.LINEAR,
            author="Test",
            version="1.0.0",
            modules=[ModuleMetadata(id="mod1", name="Module 1", path="mod1")]
        )
        
        with pytest.raises(SystemExit):
            _submit_linear_workflow(mock_state_mgr, mock_curr_mgr, progress, manifest)


class TestLibraryWorkflow:
    """Tests for _submit_library_workflow orchestration."""
    
    @patch('engine.main.console')
    def test_exits_when_no_problem_selected(self, mock_console):
        """Should exit when no problem is selected."""
        mock_state_mgr = MagicMock()
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
        
        with pytest.raises(SystemExit):
            _submit_library_workflow(mock_state_mgr, mock_curr_mgr, progress, manifest)
    
    # Removed test that goes too deep into stack - would need full validator mocking
    
    @patch('engine.main.console')
    def test_exits_when_problem_not_found(self, mock_console):
        """Should exit when selected problem doesn't exist."""
        mock_state_mgr = MagicMock()
        mock_curr_mgr = MagicMock()
        
        progress = UserProgress(
            curriculum_id="test",
            current_module_index=0,
            current_stage="build",
            active_problem_id="nonexistent"
        )
        manifest = CurriculumManifest(
            curriculum_name="Test",
            curriculum_type=CurriculumType.LIBRARY,
            author="Test",
            version="1.0.0",
            patterns=[]
        )
        
        # Mock problem lookup returning None
        mock_curr_mgr.get_problem_metadata.return_value = None
        
        with pytest.raises(SystemExit):
            _submit_library_workflow(mock_state_mgr, mock_curr_mgr, progress, manifest)
