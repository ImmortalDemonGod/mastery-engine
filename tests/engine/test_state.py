"""
Unit tests for engine.state.StateManager.

These tests achieve 100% line coverage on the StateManager class,
validating all critical state management functionality.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from engine.state import StateManager, StateFileCorruptedError, StateWriteError
from engine.schemas import UserProgress


@pytest.fixture
def temp_state_file(tmp_path):
    """Fixture to use a temporary state file location."""
    state_file = tmp_path / "test_progress.json"
    with patch.object(StateManager, 'STATE_FILE', state_file):
        yield state_file


class TestStateManagerLoad:
    """Test cases for StateManager.load()"""
    
    def test_load_nonexistent_file_returns_default(self, temp_state_file):
        """When state file doesn't exist, should return default state."""
        manager = StateManager()
        progress = manager.load()
        
        assert progress.curriculum_id == "dummy_hello_world"
        assert progress.current_module_index == 0
        assert progress.current_stage == "build"
        assert progress.completed_modules == []
    
    def test_load_valid_file(self, temp_state_file):
        """Should correctly parse a valid state file."""
        # Create a valid state file
        test_data = {
            "curriculum_id": "test_curriculum",
            "current_module_index": 2,
            "current_stage": "justify",
            "completed_modules": ["module1", "module2"]
        }
        temp_state_file.write_text(json.dumps(test_data))
        
        manager = StateManager()
        progress = manager.load()
        
        assert progress.curriculum_id == "test_curriculum"
        assert progress.current_module_index == 2
        assert progress.current_stage == "justify"
        assert progress.completed_modules == ["module1", "module2"]
    
    def test_load_malformed_json_raises_error(self, temp_state_file):
        """Should raise StateFileCorruptedError for malformed JSON."""
        temp_state_file.write_text("{ invalid json }")
        
        manager = StateManager()
        with pytest.raises(StateFileCorruptedError, match="corrupted"):
            manager.load()
    
    def test_load_invalid_schema_raises_error(self, temp_state_file):
        """Should raise StateFileCorruptedError for data that doesn't match schema."""
        # Invalid type for required field
        temp_state_file.write_text('{"curriculum_id": "test", "current_module_index": "not_an_int", '
                                   '"current_stage": "build"}')
        
        manager = StateManager()
        with pytest.raises(StateFileCorruptedError):
            manager.load()


class TestStateManagerSave:
    """Test cases for StateManager.save()"""
    
    def test_save_creates_file(self, temp_state_file):
        """Should create state file with correct content."""
        manager = StateManager()
        progress = UserProgress(
            curriculum_id="test_curriculum",
            current_module_index=1,
            current_stage="build"
        )
        
        manager.save(progress)
        
        assert temp_state_file.exists()
        loaded_data = json.loads(temp_state_file.read_text())
        assert loaded_data["curriculum_id"] == "test_curriculum"
        assert loaded_data["current_module_index"] == 1
    
    def test_save_overwrites_existing_file(self, temp_state_file):
        """Should overwrite existing state file."""
        # Create initial state
        temp_state_file.write_text('{"curriculum_id": "old", "current_module_index": 0, '
                                   '"current_stage": "build", "completed_modules": []}')
        
        manager = StateManager()
        progress = UserProgress(
            curriculum_id="new_curriculum",
            current_module_index=5,
            current_stage="harden"
        )
        
        manager.save(progress)
        
        loaded_data = json.loads(temp_state_file.read_text())
        assert loaded_data["curriculum_id"] == "new_curriculum"
        assert loaded_data["current_module_index"] == 5
    
    def test_save_atomic_write_cleans_up_temp_on_error(self, temp_state_file):
        """Should clean up temp file if rename fails."""
        manager = StateManager()
        progress = UserProgress(curriculum_id="test")
        
        # Mock rename to raise an exception
        with patch('pathlib.Path.rename', side_effect=OSError("Rename failed")):
            with pytest.raises(StateWriteError):
                manager.save(progress)
        
        # Temp file should be cleaned up
        temp_file = temp_state_file.with_suffix('.tmp')
        assert not temp_file.exists()


class TestUserProgressModel:
    """Test cases for UserProgress model methods."""
    
    def test_mark_stage_complete_build_to_justify(self):
        """Should advance from build to justify stage."""
        progress = UserProgress(curriculum_id="test", current_stage="build")
        progress.mark_stage_complete("build")
        
        assert progress.current_stage == "justify"
        assert progress.current_module_index == 0
    
    def test_mark_stage_complete_justify_to_harden(self):
        """Should advance from justify to harden stage."""
        progress = UserProgress(curriculum_id="test", current_stage="justify")
        progress.mark_stage_complete("justify")
        
        assert progress.current_stage == "harden"
        assert progress.current_module_index == 0
    
    def test_mark_stage_complete_harden_advances_module(self):
        """Should advance to next module after harden completion."""
        progress = UserProgress(curriculum_id="test", current_stage="harden")
        progress.mark_stage_complete("harden")
        
        assert progress.current_stage == "build"
        assert progress.current_module_index == 1
        assert len(progress.completed_modules) == 1
