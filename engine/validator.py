"""
Validation subsystem for executing curriculum validator scripts.

The ValidationSubsystem provides secure, controlled execution of validator.sh scripts
that check user implementations against test suites.

Key design principles:
- Security: Execute scripts with timeout to prevent infinite loops
- Isolation: Run in controlled environment (future: add Docker sandboxing)
- Parsing: Extract structured data from validator output
- Error handling: Capture and report execution failures clearly
"""

import subprocess
import re
import logging
from pathlib import Path
from typing import Optional

from engine.schemas import ValidationResult


logger = logging.getLogger(__name__)


class ValidationSubsystem:
    """
    Executes validator scripts and parses their output.
    
    Validator scripts must follow the contract:
    - Exit code 0 = validation passed
    - Exit code non-zero = validation failed
    - stdout may contain "PERFORMANCE_SECONDS: <float>" for benchmarking
    - stderr contains detailed error messages/tracebacks
    """
    
    DEFAULT_TIMEOUT_SECONDS = 300  # 5 minutes
    
    def __init__(self, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS):
        """
        Initialize validation subsystem.
        
        Args:
            timeout_seconds: Maximum time to allow validator to run before terminating
        """
        self.timeout_seconds = timeout_seconds
    
    def execute(self, validator_path: Path, workspace_path: Path) -> ValidationResult:
        """
        Execute a validator script and return structured results.
        
        The validator script is run from the workspace directory with:
        - Controlled timeout to prevent infinite loops
        - Captured stdout/stderr for result parsing
        - Environment isolation (current directory = workspace)
        
        Args:
            validator_path: Path to the validator.sh script
            workspace_path: Path to the workspace directory (working directory for execution)
            
        Returns:
            ValidationResult with exit code, output, and parsed performance metrics
            
        Raises:
            ValidatorNotFoundError: If validator script doesn't exist
            ValidatorTimeoutError: If validation exceeds timeout
            ValidatorExecutionError: If execution fails for other reasons
        """
        if not validator_path.exists():
            raise ValidatorNotFoundError(
                f"Validator script not found: {validator_path}"
            )
        
        if not validator_path.is_file():
            raise ValidatorNotFoundError(
                f"Validator path is not a file: {validator_path}"
            )
        
        # Ensure workspace exists
        if not workspace_path.exists():
            raise ValidatorExecutionError(
                f"Workspace directory does not exist: {workspace_path}"
            )
        
        logger.info(f"Executing validator: {validator_path} in workspace: {workspace_path}")
        
        try:
            # Execute the validator script with environment variables
            import os
            env = os.environ.copy()
            
            # SHADOW_WORKTREE: Path to shadow worktree for isolated validation
            shadow_worktree = Path('.mastery_engine_worktree')
            if shadow_worktree.exists():
                env['SHADOW_WORKTREE'] = str(shadow_worktree.resolve())
            
            result = subprocess.run(
                [str(validator_path.resolve())],
                cwd=str(workspace_path.resolve()),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,  # Don't raise exception on non-zero exit
                env=env
            )
            
            logger.info(f"Validator completed with exit code {result.returncode}")
            
            # Parse performance metrics from stdout
            performance_seconds = self._parse_performance(result.stdout)
            
            return ValidationResult(
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                performance_seconds=performance_seconds
            )
            
        except subprocess.TimeoutExpired as e:
            logger.error(f"Validator timed out after {self.timeout_seconds} seconds")
            raise ValidatorTimeoutError(
                f"Validator execution exceeded {self.timeout_seconds} second timeout"
            ) from e
        except Exception as e:
            logger.exception(f"Validator execution failed: {e}")
            raise ValidatorExecutionError(
                f"Failed to execute validator: {e}"
            ) from e
    
    def _parse_performance(self, stdout: str) -> Optional[float]:
        """
        Parse performance metrics from validator stdout.
        
        Looks for pattern: "PERFORMANCE_SECONDS: <float>"
        
        Args:
            stdout: Standard output from validator script
            
        Returns:
            Parsed float value, or None if pattern not found
        """
        match = re.search(r'PERFORMANCE_SECONDS:\s*([\d.]+)', stdout)
        if match:
            try:
                perf = float(match.group(1))
                logger.debug(f"Parsed performance: {perf} seconds")
                return perf
            except ValueError:
                logger.warning(f"Failed to parse performance value: {match.group(1)}")
                return None
        return None


class ValidatorNotFoundError(Exception):
    """Raised when validator script doesn't exist or is not accessible."""
    pass


class ValidatorTimeoutError(Exception):
    """Raised when validator execution exceeds timeout."""
    pass


class ValidatorExecutionError(Exception):
    """Raised when validator execution fails for reasons other than timeout."""
    pass
