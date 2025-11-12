"""
Unit tests for engine.main CLI commands.

Tests use mocking to isolate CLI logic from state/curriculum management.
Focus is on command behavior, error handling, and user messaging.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from engine.main import app
from engine.schemas import UserProgress, CurriculumManifest, ModuleMetadata
from engine.state import StateFileCorruptedError
from engine.curriculum import CurriculumNotFoundError, CurriculumInvalidError


runner = CliRunner()


class TestNextCommand:
    """Test cases for the `engine next` command."""
    
    @patch('engine.main.CurriculumManager')
    @patch('engine.main.StateManager')
    def test_next_displays_build_prompt(self, mock_state_cls, mock_curr_cls):
        """Should display build prompt when in build stage."""
        # Mock state manager
        mock_state_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_state_mgr.load.return_value = UserProgress(
            curriculum_id="test_curriculum",
            current_module_index=0,
            current_stage="build"
        )
        
        # Mock curriculum manager
        mock_curr_mgr = MagicMock()
        mock_curr_cls.return_value = mock_curr_mgr
        mock_curr_mgr.load_manifest.return_value = CurriculumManifest(
            curriculum_name="test_curriculum",
            author="Test",
            version="1.0.0",
            modules=[
                ModuleMetadata(id="mod1", name="Module 1", path="modules/mod1")
            ]
        )
        
        mock_prompt_path = MagicMock()
        mock_prompt_path.exists.return_value = True
        mock_prompt_path.read_text.return_value = "# Test Build Prompt\n\nImplement something."
        mock_curr_mgr.get_build_prompt_path.return_value = mock_prompt_path
        
        result = runner.invoke(app, ["next"])
        
        assert result.exit_code == 0
        assert "Build Challenge: Module 1" in result.stdout
        assert "Test Build Prompt" in result.stdout
    
    @patch('engine.main.CurriculumManager')
    @patch('engine.main.StateManager')
    def test_next_when_curriculum_complete(self, mock_state_cls, mock_curr_cls):
        """Should show completion message when all modules finished."""
        mock_state_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_state_mgr.load.return_value = UserProgress(
            curriculum_id="test_curriculum",
            current_module_index=5,  # Beyond last module
            current_stage="build"
        )
        
        mock_curr_mgr = MagicMock()
        mock_curr_cls.return_value = mock_curr_mgr
        mock_curr_mgr.load_manifest.return_value = CurriculumManifest(
            curriculum_name="test_curriculum",
            author="Test",
            version="1.0.0",
            modules=[
                ModuleMetadata(id="mod1", name="Module 1", path="modules/mod1")
            ]
        )
        
        result = runner.invoke(app, ["next"])
        
        assert result.exit_code == 0
        assert "Congratulations" in result.stdout
        assert "completed all modules" in result.stdout
    
    @patch('engine.main.CurriculumManager')
    @patch('engine.main.StateManager')
    def test_next_when_wrong_stage(self, mock_state_cls, mock_curr_cls):
        """Should show error message when not in build stage."""
        mock_state_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_state_mgr.load.return_value = UserProgress(
            curriculum_id="test_curriculum",
            current_module_index=0,
            current_stage="justify"  # Wrong stage
        )
        
        mock_curr_mgr = MagicMock()
        mock_curr_cls.return_value = mock_curr_mgr
        mock_curr_mgr.load_manifest.return_value = CurriculumManifest(
            curriculum_name="test_curriculum",
            author="Test",
            version="1.0.0",
            modules=[
                ModuleMetadata(id="mod1", name="Module 1", path="modules/mod1")
            ]
        )
        
        result = runner.invoke(app, ["next"])
        
        assert result.exit_code == 0
        assert "Not in Build Stage" in result.stdout
        assert "JUSTIFY" in result.stdout
    
    @patch('engine.main.CurriculumManager')
    @patch('engine.main.StateManager')
    def test_next_handles_missing_prompt(self, mock_state_cls, mock_curr_cls):
        """Should raise error when build_prompt.txt doesn't exist."""
        mock_state_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_state_mgr.load.return_value = UserProgress(
            curriculum_id="test_curriculum",
            current_module_index=0,
            current_stage="build"
        )
        
        mock_curr_mgr = MagicMock()
        mock_curr_cls.return_value = mock_curr_mgr
        mock_curr_mgr.load_manifest.return_value = CurriculumManifest(
            curriculum_name="test_curriculum",
            author="Test",
            version="1.0.0",
            modules=[
                ModuleMetadata(id="mod1", name="Module 1", path="modules/mod1")
            ]
        )
        
        mock_prompt_path = MagicMock()
        mock_prompt_path.exists.return_value = False  # Prompt missing
        mock_curr_mgr.get_build_prompt_path.return_value = mock_prompt_path
        
        result = runner.invoke(app, ["next"])
        
        assert result.exit_code == 1
        assert "Invalid Curriculum" in result.stdout
        assert "Build prompt missing" in result.stdout
    
    @patch('engine.main.StateManager')
    def test_next_handles_corrupted_state(self, mock_state_cls):
        """Should handle StateFileCorruptedError gracefully."""
        mock_state_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_state_mgr.load.side_effect = StateFileCorruptedError("Corrupted!")
        
        result = runner.invoke(app, ["next"])
        
        assert result.exit_code == 1
        assert "State File Corrupted" in result.stdout


class TestStatusCommand:
    """Test cases for the `engine status` command."""
    
    @patch('engine.main.CurriculumManager')
    @patch('engine.main.StateManager')
    def test_status_displays_progress(self, mock_state_cls, mock_curr_cls):
        """Should display progress table with current state."""
        mock_state_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_state_mgr.load.return_value = UserProgress(
            curriculum_id="test_curriculum",
            current_module_index=0,
            current_stage="build",
            completed_modules=[]
        )
        
        mock_curr_mgr = MagicMock()
        mock_curr_cls.return_value = mock_curr_mgr
        mock_curr_mgr.load_manifest.return_value = CurriculumManifest(
            curriculum_name="test_curriculum",
            author="Test Author",
            version="1.0.0",
            modules=[
                ModuleMetadata(id="mod1", name="Module 1", path="modules/mod1")
            ]
        )
        
        result = runner.invoke(app, ["status"])
        
        assert result.exit_code == 0
        assert "Mastery Engine Progress" in result.stdout
        assert "test_curriculum" in result.stdout
        assert "Test Author" in result.stdout
        assert "Module 1" in result.stdout
        assert "BUILD" in result.stdout
    
    @patch('engine.main.CurriculumManager')
    @patch('engine.main.StateManager')
    def test_status_shows_completion_message(self, mock_state_cls, mock_curr_cls):
        """Should show completion message when all modules done."""
        mock_state_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_state_mgr.load.return_value = UserProgress(
            curriculum_id="test_curriculum",
            current_module_index=1,  # Beyond last module
            current_stage="build"
        )
        
        mock_curr_mgr = MagicMock()
        mock_curr_cls.return_value = mock_curr_mgr
        mock_curr_mgr.load_manifest.return_value = CurriculumManifest(
            curriculum_name="test_curriculum",
            author="Test",
            version="1.0.0",
            modules=[
                ModuleMetadata(id="mod1", name="Module 1", path="modules/mod1")
            ]
        )
        
        result = runner.invoke(app, ["status"])
        
        assert result.exit_code == 0
        assert "All modules completed" in result.stdout
    
    @patch('engine.main.StateManager')
    def test_status_handles_curriculum_not_found(self, mock_state_cls):
        """Should handle CurriculumNotFoundError gracefully."""
        mock_state_mgr = MagicMock()
        mock_state_cls.return_value = mock_state_mgr
        mock_state_mgr.load.return_value = UserProgress(curriculum_id="nonexistent")
        
        with patch('engine.main.CurriculumManager') as mock_curr_cls:
            mock_curr_mgr = MagicMock()
            mock_curr_cls.return_value = mock_curr_mgr
            mock_curr_mgr.load_manifest.side_effect = CurriculumNotFoundError("Not found!")
            
            result = runner.invoke(app, ["status"])
            
            assert result.exit_code == 1
            assert "Curriculum Not Found" in result.stdout
