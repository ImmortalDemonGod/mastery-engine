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
from typing import Optional, Dict, Tuple

from engine.schemas import (
    CurriculumManifest,
    ModuleMetadata,
    PatternMetadata,
    ProblemMetadata,
    CurriculumType
)
from engine.utils import find_project_root


logger = logging.getLogger(__name__)


class CurriculumManager:
    """
    Manages curriculum pack loading and provides access to module/problem assets.
    
    Supports two curriculum types:
    
    LINEAR (e.g., cs336_a1): Sequential modules
        curricula/<curriculum_id>/
            manifest.json (type: "linear")
            modules/<module_id>/
                build_prompt.txt
                validator.sh
                justify_questions.json
                bugs/...
    
    LIBRARY (e.g., cp_accelerator): Hierarchical patterns+problems
        curricula/<curriculum_id>/
            manifest.json (type: "library")
            patterns/<pattern_id>/
                theory/
                    justify_questions.json
                problems/<problem_id>/
                    build_prompt.txt
                    validator.sh
                    bugs/...
    
    The curriculum directory is resolved relative to the project root,
    allowing the engine to work from any subdirectory.
    """
    
    # Class-level attribute for test compatibility (can be patched)
    CURRICULA_DIR = Path("curricula")
    
    def __init__(self, curricula_dir: Path = None):
        """
        Initialize with project root detection.
        
        Args:
            curricula_dir: Optional override for curricula directory (for testing)
        """
        if curricula_dir is not None:
            # Explicit override (typically from tests)
            self.CURRICULA_DIR = curricula_dir
            self._project_root = curricula_dir.parent if curricula_dir.parent != curricula_dir else Path.cwd()
        else:
            try:
                self._project_root = find_project_root()
                # Only override CURRICULA_DIR if we successfully found project root
                # This allows tests to patch the class-level attribute
                self.CURRICULA_DIR = self._project_root / "curricula"
            except RuntimeError:
                # Fallback - leave CURRICULA_DIR as class default (tests can patch)
                self._project_root = Path.cwd()
        
        # Lookup caches for LIBRARY mode (pattern_id -> PatternMetadata, problem_id -> (pattern_id, ProblemMetadata))
        self._pattern_cache: Dict[str, PatternMetadata] = {}
        self._problem_cache: Dict[str, Tuple[str, ProblemMetadata]] = {}
    
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
            
            # Build lookup caches for LIBRARY mode
            if manifest.type == CurriculumType.LIBRARY:
                self._build_library_caches(manifest)
                logger.info(f"Loaded LIBRARY curriculum '{curriculum_id}' with {len(manifest.patterns)} patterns")
            else:
                logger.info(f"Loaded LINEAR curriculum '{curriculum_id}' with {len(manifest.modules)} modules")
            
            return manifest
        except json.JSONDecodeError as e:
            raise CurriculumInvalidError(
                f"manifest.json is malformed JSON: {e}"
            ) from e
        except Exception as e:
            raise CurriculumInvalidError(
                f"Failed to validate manifest.json: {e}"
            ) from e
    
    def _build_library_caches(self, manifest: CurriculumManifest) -> None:
        """
        Build lookup caches for efficient LIBRARY mode access.
        
        Creates O(1) lookups for:
        - pattern_id -> PatternMetadata
        - problem_id -> (pattern_id, ProblemMetadata)
        """
        self._pattern_cache.clear()
        self._problem_cache.clear()
        
        if manifest.patterns is None:
            return
        
        for pattern in manifest.patterns:
            self._pattern_cache[pattern.id] = pattern
            
            for problem in pattern.problems:
                # Store both pattern_id and problem metadata for context
                self._problem_cache[problem.id] = (pattern.id, problem)
    
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
    
    # --- New LIBRARY Mode Accessors ---
    
    def get_problem_path(self, curriculum_id: str, problem_id: str) -> Optional[Path]:
        """
        Get the absolute path to a problem's directory (LIBRARY mode).
        
        Args:
            curriculum_id: Unique identifier for the curriculum pack
            problem_id: Problem identifier (e.g., "lc_912")
            
        Returns:
            Path object pointing to the problem directory, or None if not found
        """
        if problem_id not in self._problem_cache:
            return None
        
        pattern_id, problem_metadata = self._problem_cache[problem_id]
        return self.CURRICULA_DIR / curriculum_id / problem_metadata.path
    
    def get_pattern_theory_path(self, curriculum_id: str, pattern_id: str) -> Optional[Path]:
        """
        Get the absolute path to a pattern's theory directory (LIBRARY mode).
        
        Args:
            curriculum_id: Unique identifier for the curriculum pack
            pattern_id: Pattern identifier (e.g., "sorting")
            
        Returns:
            Path object pointing to the theory directory, or None if not found
        """
        if pattern_id not in self._pattern_cache:
            return None
        
        pattern_metadata = self._pattern_cache[pattern_id]
        return self.CURRICULA_DIR / curriculum_id / pattern_metadata.theory_path
    
    def get_problem_metadata(self, problem_id: str) -> Optional[Tuple[str, ProblemMetadata]]:
        """
        Get problem metadata with its parent pattern ID (LIBRARY mode).
        
        Args:
            problem_id: Problem identifier (e.g., "lc_912")
            
        Returns:
            Tuple of (pattern_id, ProblemMetadata), or None if not found
        """
        return self._problem_cache.get(problem_id)
    
    def get_pattern_metadata(self, pattern_id: str) -> Optional[PatternMetadata]:
        """
        Get pattern metadata (LIBRARY mode).
        
        Args:
            pattern_id: Pattern identifier (e.g., "sorting")
            
        Returns:
            PatternMetadata, or None if not found
        """
        return self._pattern_cache.get(pattern_id)


class CurriculumNotFoundError(Exception):
    """Raised when requested curriculum pack doesn't exist."""
    pass


class CurriculumInvalidError(Exception):
    """Raised when curriculum pack structure is invalid or malformed."""
    pass
