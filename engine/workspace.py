"""
Workspace management for the Mastery Engine.

The WorkspaceManager provides an abstraction layer for file system operations
related to the user's workspace directory where they write their implementations.

Key design principles:
- Centralize workspace path logic in one place
- Make it easy to change workspace location or structure
- Provide clear API for workspace operations
"""

from pathlib import Path
import logging
import shutil
import subprocess


logger = logging.getLogger(__name__)


class WorkspaceManager:
    """
    Manages the user's workspace directory where implementations are written.
    
    The workspace is a dedicated directory where users create their code files.
    This manager abstracts the file system paths to decouple the engine
    from specific directory structures.
    """
    
    WORKSPACE_DIR = Path("workspace")
    
    def __init__(self, workspace_root: Path | None = None):
        """
        Initialize workspace manager.
        
        Args:
            workspace_root: Optional custom workspace root. If None, uses default ./workspace
        """
        self.workspace_root = workspace_root if workspace_root else self.WORKSPACE_DIR
    
    def get_workspace_path(self) -> Path:
        """
        Get the absolute path to the workspace root directory.
        
        Returns:
            Path object pointing to workspace root
        """
        return self.workspace_root.resolve()
    
    def get_submission_path(self, module_id: str, filename: str = None) -> Path:
        """
        Get the path where a user's submission file should be located.
        
        For now, this returns workspace/<filename>. In the future, this could
        be extended to support module-specific subdirectories.
        
        Args:
            module_id: ID of the module (for future use in nested structure)
            filename: Name of the submission file (e.g., "hello_world.py")
            
        Returns:
            Path object pointing to the submission file location
        """
        if filename:
            return self.workspace_root / filename
        else:
            # Return workspace root if no filename specified
            return self.workspace_root
    
    def ensure_workspace_exists(self) -> None:
        """
        Create the workspace directory if it doesn't exist.
        
        Raises:
            WorkspaceError: If workspace cannot be created
        """
        try:
            self.workspace_root.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured workspace exists at {self.workspace_root}")
        except Exception as e:
            raise WorkspaceError(f"Failed to create workspace directory: {e}") from e
    
    def create_harden_workspace(self, module_id: str, source_filename: str) -> Path:
        """
        Create isolated harden workspace by copying build submission.
        
        The harden stage requires workspace isolation to prevent users from
        accidentally modifying their original build submission while debugging.
        
        Args:
            module_id: ID of the module being hardened
            source_filename: Name of the build submission file to copy
            
        Returns:
            Path to the harden workspace file
            
        Raises:
            WorkspaceError: If source file doesn't exist or copy fails
        """
        # Source: user's build submission
        source_file = self.workspace_root / source_filename
        
        if not source_file.exists():
            raise WorkspaceError(
                f"Build submission not found: {source_file}. "
                "You must complete the build stage first."
            )
        
        # Destination: harden subdirectory
        harden_dir = self.workspace_root / "harden"
        harden_dir.mkdir(parents=True, exist_ok=True)
        
        dest_file = harden_dir / source_filename
        
        try:
            shutil.copy2(source_file, dest_file)
            logger.info(f"Created harden workspace: copied {source_file} -> {dest_file}")
            return dest_file
        except Exception as e:
            raise WorkspaceError(f"Failed to create harden workspace: {e}") from e
    
    def apply_patch(self, target_file: Path, patch_file: Path) -> None:
        """
        Apply a patch file to a target file using the `patch` command.
        
        This uses the system `patch` utility to apply unified diffs.
        The patch is applied in-place to the target file.
        
        Args:
            target_file: File to be patched (must exist)
            patch_file: Patch file in unified diff format (must exist)
            
        Raises:
            WorkspaceError: If files don't exist or patch application fails
        """
        if not target_file.exists():
            raise WorkspaceError(f"Target file does not exist: {target_file}")
        
        if not patch_file.exists():
            raise WorkspaceError(f"Patch file does not exist: {patch_file}")
        
        try:
            # Use patch command: patch <target_file> <patch_file>
            result = subprocess.run(
                ["patch", str(target_file), str(patch_file)],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                logger.error(f"Patch application failed: {result.stderr}")
                raise PatchApplicationError(
                    f"Failed to apply patch: {result.stderr}"
                )
            
            logger.info(f"Successfully applied patch {patch_file} to {target_file}")
            
        except PatchApplicationError:
            raise
        except Exception as e:
            raise WorkspaceError(f"Error executing patch command: {e}") from e


class WorkspaceError(Exception):
    """Raised when workspace operations fail."""
    pass


class PatchApplicationError(WorkspaceError):
    """Raised when patch application fails."""
    pass
