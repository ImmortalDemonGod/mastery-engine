"""
Unit tests for engine.curriculum.CurriculumManager.

These tests achieve 100% line coverage on the CurriculumManager class,
validating curriculum loading and path resolution functionality.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from engine.curriculum import CurriculumManager, CurriculumNotFoundError, CurriculumInvalidError
from engine.schemas import CurriculumManifest, ModuleMetadata


class TestCurriculumManagerInit:
    """Tests for CurriculumManager initialization."""
    
    @patch('engine.curriculum.find_project_root')
    def test_init_fallback_on_find_root_error(self, mock_find_root):
        """Should fall back to cwd when find_project_root raises RuntimeError."""
        mock_find_root.side_effect = RuntimeError("No project root")
        manager = CurriculumManager()
        assert manager._project_root == Path.cwd()


@pytest.fixture
def temp_curricula_dir(tmp_path):
    """Fixture to use a temporary curricula directory."""
    curricula_dir = tmp_path / "curricula"
    curricula_dir.mkdir()
    yield curricula_dir


@pytest.fixture
def valid_curriculum(temp_curricula_dir):
    """Create a valid curriculum structure for testing."""
    curriculum_id = "test_curriculum"
    curriculum_path = temp_curricula_dir / curriculum_id
    curriculum_path.mkdir()
    
    manifest = {
        "curriculum_name": "test_curriculum",
        "author": "Test Author",
        "version": "1.0.0",
        "modules": [
            {
                "id": "module1",
                "name": "Module One",
                "path": "modules/module1",
                "baseline_perf_seconds": 0.5
            },
            {
                "id": "module2",
                "name": "Module Two",
                "path": "modules/module2",
                "baseline_perf_seconds": None
            }
        ]
    }
    
    manifest_path = curriculum_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    
    # Create module directories
    (curriculum_path / "modules" / "module1").mkdir(parents=True)
    (curriculum_path / "modules" / "module2").mkdir(parents=True)
    
    return curriculum_id


class TestLoadManifest:
    """Test cases for CurriculumManager.load_manifest()"""
    
    def test_load_valid_manifest(self, temp_curricula_dir, valid_curriculum):
        """Should successfully load and validate a correct manifest."""
        manager = CurriculumManager(curricula_dir=temp_curricula_dir)
        manifest = manager.load_manifest(valid_curriculum)
        
        assert manifest.curriculum_name == "test_curriculum"
        assert manifest.author == "Test Author"
        assert manifest.version == "1.0.0"
        assert len(manifest.modules) == 2
        assert manifest.modules[0].id == "module1"
        assert manifest.modules[0].baseline_perf_seconds == 0.5
        assert manifest.modules[1].baseline_perf_seconds is None
    
    def test_load_nonexistent_curriculum_raises_error(self, temp_curricula_dir):
        """Should raise CurriculumNotFoundError for missing curriculum."""
        manager = CurriculumManager(curricula_dir=temp_curricula_dir)
        
        with pytest.raises(CurriculumNotFoundError, match="not found"):
            manager.load_manifest("nonexistent_curriculum")
    
    def test_load_missing_manifest_raises_error(self, temp_curricula_dir):
        """Should raise CurriculumInvalidError when manifest.json is missing."""
        # Create curriculum directory without manifest
        curriculum_path = temp_curricula_dir / "no_manifest"
        curriculum_path.mkdir()
        
        manager = CurriculumManager(curricula_dir=temp_curricula_dir)
        
        with pytest.raises(CurriculumInvalidError, match="missing manifest.json"):
            manager.load_manifest("no_manifest")
    
    def test_load_malformed_json_raises_error(self, temp_curricula_dir):
        """Should raise CurriculumInvalidError for malformed JSON."""
        curriculum_path = temp_curricula_dir / "bad_json"
        curriculum_path.mkdir()
        (curriculum_path / "manifest.json").write_text("{ invalid json }")
        
        manager = CurriculumManager(curricula_dir=temp_curricula_dir)
        
        with pytest.raises(CurriculumInvalidError, match="malformed JSON"):
            manager.load_manifest("bad_json")
    
    def test_load_invalid_schema_raises_error(self, temp_curricula_dir):
        """Should raise CurriculumInvalidError for data not matching schema."""
        curriculum_path = temp_curricula_dir / "bad_schema"
        curriculum_path.mkdir()
        
        # Missing required fields
        bad_manifest = {"curriculum_name": "test"}
        (curriculum_path / "manifest.json").write_text(json.dumps(bad_manifest))
        
        manager = CurriculumManager(curricula_dir=temp_curricula_dir)
        
        with pytest.raises(CurriculumInvalidError, match="Failed to validate"):
            manager.load_manifest("bad_schema")


class TestPathResolution:
    """Test cases for path resolution methods."""
    
    def test_get_module_path(self, temp_curricula_dir, valid_curriculum):
        """Should return correct absolute path to module directory."""
        manager = CurriculumManager(curricula_dir=temp_curricula_dir)
        manifest = manager.load_manifest(valid_curriculum)
        module = manifest.modules[0]
        
        module_path = manager.get_module_path(valid_curriculum, module)
        
        assert module_path == temp_curricula_dir / valid_curriculum / "modules" / "module1"
        assert module_path.exists()
    
    def test_get_build_prompt_path(self, temp_curricula_dir, valid_curriculum):
        """Should return correct path to build_prompt.txt."""
        manager = CurriculumManager(curricula_dir=temp_curricula_dir)
        manifest = manager.load_manifest(valid_curriculum)
        module = manifest.modules[0]
        
        prompt_path = manager.get_build_prompt_path(valid_curriculum, module)
        
        expected = temp_curricula_dir / valid_curriculum / "modules" / "module1" / "build_prompt.txt"
        assert prompt_path == expected
    
    def test_get_validator_path(self, temp_curricula_dir, valid_curriculum):
        """Should return correct path to validator.sh."""
        manager = CurriculumManager(curricula_dir=temp_curricula_dir)
        manifest = manager.load_manifest(valid_curriculum)
        module = manifest.modules[0]
        
        validator_path = manager.get_validator_path(valid_curriculum, module)
        
        expected = temp_curricula_dir / valid_curriculum / "modules" / "module1" / "validator.sh"
        assert validator_path == expected
    
    def test_get_justify_questions_path(self, temp_curricula_dir, valid_curriculum):
        """Should return correct path to justify_questions.json."""
        manager = CurriculumManager(curricula_dir=temp_curricula_dir)
        manifest = manager.load_manifest(valid_curriculum)
        module = manifest.modules[0]
        
        questions_path = manager.get_justify_questions_path(valid_curriculum, module)
        
        expected = temp_curricula_dir / valid_curriculum / "modules" / "module1" / "justify_questions.json"
        assert questions_path == expected
    
    def test_get_bugs_dir(self, temp_curricula_dir, valid_curriculum):
        """Should return correct path to bugs directory."""
        manager = CurriculumManager(curricula_dir=temp_curricula_dir)
        manifest = manager.load_manifest(valid_curriculum)
        module = manifest.modules[0]
        
        bugs_dir = manager.get_bugs_dir(valid_curriculum, module)
        
        expected = temp_curricula_dir / valid_curriculum / "modules" / "module1" / "bugs"
        assert bugs_dir == expected
