"""
Utility functions for the Mastery Engine.

This module provides helper functions used across the engine,
including project root detection for path resolution.
"""

from pathlib import Path
import logging


logger = logging.getLogger(__name__)


def find_project_root(start_path: Path = None) -> Path:
    """
    Find the project root directory by walking up the tree.
    
    Looks for markers like pyproject.toml, .git, or curricula/ directory
    to identify the project root, allowing the engine to work from any
    subdirectory.
    
    Args:
        start_path: Directory to start search from (defaults to cwd)
        
    Returns:
        Path to project root directory
        
    Raises:
        RuntimeError: If project root cannot be found
        
    Example:
        >>> root = find_project_root()
        >>> curricula_dir = root / "curricula"
    """
    if start_path is None:
        start_path = Path.cwd()
    
    current = start_path.resolve()
    
    # Walk up the directory tree
    for parent in [current] + list(current.parents):
        # Check for project markers
        if (parent / "pyproject.toml").exists():
            logger.debug(f"Found project root at {parent} (pyproject.toml)")
            return parent
        if (parent / ".git").exists():
            logger.debug(f"Found project root at {parent} (.git)")
            return parent
        if (parent / "curricula").is_dir() and (parent / "engine").is_dir():
            logger.debug(f"Found project root at {parent} (curricula + engine)")
            return parent
    
    # If we get here, we couldn't find the root
    raise RuntimeError(
        f"Could not find project root from {start_path}. "
        f"Make sure you're inside the Mastery Engine repository."
    )
