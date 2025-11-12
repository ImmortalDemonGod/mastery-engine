"""
Harden stage runner for the Mastery Engine.

The HardenRunner implements the "Harden" stage of the BJH pedagogy loop.
It challenges users to debug their own implementations by injecting bugs
and asking them to fix them.

Key responsibilities:
- Select and present a bug for the current module
- Set up isolated harden workspace
- Display symptom description to guide debugging
- Provide interface for bug presentation logic
"""

import logging
import random
from pathlib import Path

from engine.schemas import ModuleMetadata
from engine.workspace import WorkspaceManager, WorkspaceError
from engine.curriculum import CurriculumManager


logger = logging.getLogger(__name__)


class HardenRunner:
    """
    Manages the Harden stage workflow.
    
    The Harden stage tests a user's debugging ability by:
    1. Copying their validated Build submission
    2. Applying a pedagogical bug patch
    3. Presenting symptoms for the user to debug
    4. Validating their fix
    """
    
    def __init__(
        self,
        curriculum_manager: CurriculumManager,
        workspace_manager: WorkspaceManager
    ):
        """
        Initialize harden runner.
        
        Args:
            curriculum_manager: For accessing bug assets
            workspace_manager: For workspace isolation and patch application
        """
        self.curriculum_mgr = curriculum_manager
        self.workspace_mgr = workspace_manager
    
    def present_challenge(
        self,
        curriculum_id: str,
        module: ModuleMetadata,
        source_filename: str
    ) -> tuple[Path, str]:
        """
        Set up and present a harden challenge to the user.
        
        This method:
        1. Selects a bug from the module's bugs directory
        2. Creates isolated harden workspace
        3. Applies the bug patch
        4. Returns the symptom description
        
        Args:
            curriculum_id: ID of the current curriculum
            module: Metadata for the current module
            source_filename: Name of the build submission file
            
        Returns:
            Tuple of (harden_file_path, symptom_description)
            
        Raises:
            HardenChallengeError: If challenge setup fails
        """
        try:
            # Select a bug
            bugs_dir = self.curriculum_mgr.get_bugs_dir(curriculum_id, module)
            bug_patch, symptom_file = self._select_bug(bugs_dir)
            
            logger.info(f"Selected bug: {bug_patch.name}")
            
            # Create isolated harden workspace
            harden_file = self.workspace_mgr.create_harden_workspace(
                module.id,
                source_filename
            )
            
            # Apply bug patch
            self.workspace_mgr.apply_patch(harden_file, bug_patch)
            
            # Read symptom description
            symptom = symptom_file.read_text(encoding='utf-8')
            
            logger.info(f"Harden challenge prepared: {module.id}")
            
            return harden_file, symptom
            
        except WorkspaceError as e:
            raise HardenChallengeError(f"Failed to set up harden challenge: {e}") from e
        except Exception as e:
            raise HardenChallengeError(f"Unexpected error in harden setup: {e}") from e
    
    def _select_bug(self, bugs_dir: Path) -> tuple[Path, Path]:
        """
        Select a bug from the bugs directory.
        
        For now, randomly selects from available bugs. In the future, this could
        use adaptive difficulty selection based on user performance.
        
        Args:
            bugs_dir: Directory containing bug patches and symptoms
            
        Returns:
            Tuple of (patch_file, symptom_file)
            
        Raises:
            HardenChallengeError: If no bugs available or files missing
        """
        if not bugs_dir.exists():
            raise HardenChallengeError(
                f"Bugs directory not found: {bugs_dir}. "
                "This module may not have harden challenges configured."
            )
        
        # Find all patch files
        patch_files = list(bugs_dir.glob("*.patch"))
        
        if not patch_files:
            raise HardenChallengeError(
                f"No bug patches found in {bugs_dir}. "
                "This is a curriculum configuration error."
            )
        
        # Select a patch (random for now)
        selected_patch = random.choice(patch_files)
        
        # Find corresponding symptom file
        # Convention: bug.patch → bug_symptom.txt
        symptom_name = selected_patch.stem + "_symptom.txt"
        symptom_file = bugs_dir / symptom_name
        
        if not symptom_file.exists():
            raise HardenChallengeError(
                f"Symptom file missing for {selected_patch.name}: {symptom_file}"
            )
        
        return selected_patch, symptom_file


class HardenChallengeError(Exception):
    """Raised when harden challenge setup fails."""
    pass
