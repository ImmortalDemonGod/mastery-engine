"""
Curriculum management for the Mastery Engine.

The CurriculumManager handles loading and parsing curriculum packs,
providing access to module assets (prompts, validators, questions, bugs).

Key design principles:
- Fail fast with clear error messages if curriculum is malformed
- Validate curriculum structure using Pydantic schemas
- Provide simple API for accessing module paths and metadata
"""

from pathlib import Path
import json
import logging
from typing import Optional

from engine.schemas import CurriculumManifest, ModuleMetadata
from engine.utils import find_project_root


logger = logging.getLogger(__name__)


class CurriculumManager:
    """
    Manages curriculum pack loading and provides access to module assets.
    
    Curriculum packs are stored in the curricula/ directory with the structure:
        curricula/<curriculum_id>/
            manifest.json
            modules/<module_id>/
                build_prompt.txt
                validator.sh
                justify_questions.json
                bugs/...
    
    The curriculum directory is resolved relative to the project root,
    allowing the engine to work from any subdirectory.
    """
    
    def __init__(self):
        """Initialize with project root detection."""
        self._project_root = find_project_root()
        self.CURRICULA_DIR = self._project_root / "curricula"
    
    def load_manifest(self, curriculum_id: str) -> CurriculumManifest:
        """
        Load and validate a curriculum's manifest.json file.
        
        Args:
            curriculum_id: Unique identifier for the curriculum pack
            
        Returns:
            Validated CurriculumManifest object
            
        Raises:
            CurriculumNotFoundError: If curriculum directory doesn't exist
            CurriculumInvalidError: If manifest is malformed or missing
        """
        curriculum_path = self.CURRICULA_DIR / curriculum_id
        
        if not curriculum_path.exists():
            raise CurriculumNotFoundError(
                f"Curriculum '{curriculum_id}' not found in {self.CURRICULA_DIR}"
            )
        
        manifest_path = curriculum_path / "manifest.json"
        
        if not manifest_path.exists():
            raise CurriculumInvalidError(
                f"Curriculum '{curriculum_id}' is missing manifest.json"
            )
        
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            manifest = CurriculumManifest.model_validate(data)
            logger.info(f"Loaded curriculum '{curriculum_id}' with {len(manifest.modules)} modules")
            return manifest
        except json.JSONDecodeError as e:
            raise CurriculumInvalidError(
                f"manifest.json is malformed JSON: {e}"
            ) from e
        except Exception as e:
            raise CurriculumInvalidError(
                f"Failed to validate manifest.json: {e}"
            ) from e
    
    def get_module_path(self, curriculum_id: str, module_metadata: ModuleMetadata) -> Path:
        """
        Get the absolute path to a module's directory.
        
        Args:
            curriculum_id: Unique identifier for the curriculum pack
            module_metadata: Metadata for the specific module
            
        Returns:
            Path object pointing to the module directory
        """
        return self.CURRICULA_DIR / curriculum_id / module_metadata.path
    
    def get_build_prompt_path(self, curriculum_id: str, module_metadata: ModuleMetadata) -> Path:
        """Get path to a module's build_prompt.txt file."""
        return self.get_module_path(curriculum_id, module_metadata) / "build_prompt.txt"
    
    def get_validator_path(self, curriculum_id: str, module_metadata: ModuleMetadata) -> Path:
        """Get path to a module's validator.sh script."""
        return self.get_module_path(curriculum_id, module_metadata) / "validator.sh"
    
    def get_justify_questions_path(self, curriculum_id: str, module_metadata: ModuleMetadata) -> Path:
        """Get path to a module's justify_questions.json file."""
        return self.get_module_path(curriculum_id, module_metadata) / "justify_questions.json"
    
    def get_bugs_dir(self, curriculum_id: str, module_metadata: ModuleMetadata) -> Path:
        """Get path to a module's bugs directory."""
        return self.get_module_path(curriculum_id, module_metadata) / "bugs"


class CurriculumNotFoundError(Exception):
    """Raised when requested curriculum pack doesn't exist."""
    pass


class CurriculumInvalidError(Exception):
    """Raised when curriculum pack structure is invalid or malformed."""
    pass
