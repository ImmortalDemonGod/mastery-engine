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
        source_file_path: Path
    ) -> tuple[Path, str]:
        """
        Set up and present a harden challenge to the user in shadow worktree.
        
        SHADOW WORKTREE MODEL:
        1. Selects a bug from the module's bugs directory
        2. Copies DEVELOPER's reference implementation to shadow worktree harden directory
        3. Applies the bug patch to the reference implementation (guaranteed to work)
        4. Returns path to buggy file in shadow worktree for user to debug
        
        Note: We use the developer's implementation (not student's) because patch files
        require exact byte-for-byte matches. Student code may have different variable names,
        comments, or structure, causing patches to fail. This approach ensures consistent,
        debuggable buggy code for all students regardless of their implementation style.
        
        Args:
            curriculum_id: ID of the current curriculum
            module: Metadata for the current module
            source_file_path: Path to user's correct implementation in main directory
            
        Returns:
            Tuple of (harden_file_path_in_shadow_worktree, symptom_description)
            
        Raises:
            HardenChallengeError: If challenge setup fails
        """
        try:
            # Verify shadow worktree exists
            shadow_worktree = Path('.mastery_engine_worktree')
            if not shadow_worktree.exists():
                raise HardenChallengeError(
                    "Shadow worktree not found. Please run 'engine init' first."
                )
            
            # Select a bug
            bugs_dir = self.curriculum_mgr.get_bugs_dir(curriculum_id, module)
            bug_file, symptom_file = self._select_bug(bugs_dir)
            
            logger.info(f"Selected bug: {bug_file.name}")
            
            # Create harden directory in shadow worktree
            harden_dir = shadow_worktree / "workspace" / "harden"
            harden_dir.mkdir(parents=True, exist_ok=True)
            harden_file = harden_dir / source_file_path.name
            
            # Dispatch based on bug type (.json = AST, .patch = legacy)
            if bug_file.suffix == '.json':
                # --- NEW LOGIC: AST-based bug injection (Generic) ---
                from engine.ast_harden.generic_injector import GenericBugInjector
                
                logger.info("Using AST-based bug injection (generic)")
                
                # Load student's OWN code from main workspace
                student_code_path = Path.cwd() / source_file_path
                if not student_code_path.exists():
                    raise HardenChallengeError(
                        f"Could not find student source file at {student_code_path}"
                    )
                
                student_source_code = student_code_path.read_text(encoding='utf-8')
                
                # Inject bug using generic AST transformation
                injector = GenericBugInjector(bug_file)
                buggy_source_code, success = injector.inject(student_source_code)
                
                if not success:
                    raise HardenChallengeError(
                        "The AST bug injector failed to find the required semantic pattern in your code. "
                        "This can happen if your implementation style is highly unusual. "
                        "Please ensure your implementation follows standard practices or flag this as an issue."
                    )
                
                # Write buggy code to harden workspace
                harden_file.write_text(buggy_source_code, encoding='utf-8')
                logger.info(f"AST bug injected successfully into {harden_file}")
                
            else:
                # --- OLD LOGIC: Patch-based bug injection (backward compatibility) ---
                import shutil
                from pathlib import Path as PathLib
                
                logger.info("Using legacy patch-based bug injection")
                
                # Use DEVELOPER implementation for patch-based bugs
                # (patches require exact byte-for-byte match)
                repo_root = PathLib.cwd()
                
                try:
                    rel_path = source_file_path.relative_to(repo_root)
                except ValueError:
                    rel_path = PathLib(source_file_path.name)
                    if len(source_file_path.parts) > 1:
                        rel_path = PathLib(*source_file_path.parts[-2:])
                
                developer_file = repo_root / "modes" / "developer" / rel_path
                
                if not developer_file.exists():
                    raise HardenChallengeError(
                        f"Developer reference implementation not found: {developer_file}"
                    )
                
                shutil.copy2(developer_file, harden_file)
                logger.info(f"Copied developer reference implementation to {harden_file}")
                
                # Apply patch
                self.workspace_mgr.apply_patch(harden_file, bug_file)
                logger.info(f"Applied patch {bug_file.name}")
            
            # Read symptom description
            symptom = symptom_file.read_text(encoding='utf-8')
            
            logger.info(f"Harden challenge prepared in shadow worktree: {module.id}")
            
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
        
        # Find all bug files (.patch or .json)
        patch_files = list(bugs_dir.glob("*.patch"))
        json_files = list(bugs_dir.glob("*.json"))
        bug_files = patch_files + json_files
        
        if not bug_files:
            raise HardenChallengeError(
                f"No bug files found in {bugs_dir}. "
                "This is a curriculum configuration error."
            )
        
        # Select a bug file (random for now)
        selected_bug = random.choice(bug_files)
        
        # Find corresponding symptom file
        # Convention: bug.patch or bug.json → bug_symptom.txt
        symptom_name = selected_bug.stem + "_symptom.txt"
        symptom_file = bugs_dir / symptom_name
        
        if not symptom_file.exists():
            raise HardenChallengeError(
                f"Symptom file missing for {selected_bug.name}: {symptom_file}"
            )
        
        return selected_bug, symptom_file


class HardenChallengeError(Exception):
    """Raised when harden challenge setup fails."""
    pass
