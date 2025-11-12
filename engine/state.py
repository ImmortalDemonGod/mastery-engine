"""
State management for Mastery Engine user progress.

The StateManager handles reading and writing the .mastery_progress.json file,
which tracks the user's current position in their learning journey.

Key design principles:
- Atomic writes to prevent state corruption on crashes
- Graceful handling of missing/corrupted state files
- Clear error messages for debugging
"""

from pathlib import Path
import json
import logging
from typing import Optional

from engine.schemas import UserProgress


logger = logging.getLogger(__name__)


class StateManager:
    """
    Manages user progress state with atomic file operations.
    
    The state file is stored in the user's home directory to persist
    across repository clones and working directory changes.
    """
    
    STATE_FILE = Path.home() / ".mastery_progress.json"
    
    def load(self) -> UserProgress:
        """
        Load user progress from state file.
        
        Returns:
            UserProgress object. If file doesn't exist, returns default state
            for the dummy_hello_world curriculum.
            
        Raises:
            StateFileCorruptedError: If state file exists but cannot be parsed
        """
        if not self.STATE_FILE.exists():
            logger.info("No existing progress file found, starting fresh")
            return self._get_default_state()
        
        try:
            with open(self.STATE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            progress = UserProgress.model_validate(data)
            logger.info(f"Loaded progress: curriculum={progress.curriculum_id}, "
                       f"module_index={progress.current_module_index}, "
                       f"stage={progress.current_stage}")
            return progress
        except json.JSONDecodeError as e:
            raise StateFileCorruptedError(
                f"Progress file is corrupted and cannot be parsed: {e}"
            ) from e
        except Exception as e:
            raise StateFileCorruptedError(
                f"Failed to load progress file: {e}"
            ) from e
    
    def save(self, progress: UserProgress) -> None:
        """
        Save user progress to state file using atomic write.
        
        Uses write-to-temp-then-rename pattern to ensure atomicity.
        If the process crashes during write, the original file remains intact.
        
        Args:
            progress: UserProgress object to persist
            
        Raises:
            StateWriteError: If write operation fails
        """
        temp_file = self.STATE_FILE.with_suffix('.tmp')
        
        try:
            # Write to temporary file
            json_data = progress.model_dump_json(indent=2)
            temp_file.write_text(json_data, encoding='utf-8')
            
            # Atomic rename
            temp_file.rename(self.STATE_FILE)
            
            logger.info(f"Saved progress: curriculum={progress.curriculum_id}, "
                       f"module_index={progress.current_module_index}, "
                       f"stage={progress.current_stage}")
        except Exception as e:
            # Clean up temp file if it exists
            if temp_file.exists():
                temp_file.unlink()
            raise StateWriteError(f"Failed to save progress: {e}") from e
    
    def _get_default_state(self) -> UserProgress:
        """
        Get default starting state for new users.
        
        Returns:
            UserProgress configured for the dummy_hello_world curriculum
        """
        return UserProgress(
            curriculum_id="dummy_hello_world",
            current_module_index=0,
            current_stage="build"
        )


class StateFileCorruptedError(Exception):
    """Raised when progress file exists but cannot be parsed."""
    pass


class StateWriteError(Exception):
    """Raised when progress file cannot be written."""
    pass
