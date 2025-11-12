#!/bin/bash
# Validator for rmsnorm module
# Runs pytest for RMSNorm implementation in shadow worktree

set -e  # Exit on error

# SHADOW_WORKTREE environment variable is set by the ValidationSubsystem
if [ -z "$SHADOW_WORKTREE" ]; then
    echo "ERROR: SHADOW_WORKTREE environment variable not set"
    exit 1
fi

# Determine if we're in BUILD or HARDEN stage
# BUILD STAGE: Copy from main directory to shadow worktree
# HARDEN STAGE: File already copied by submit-fix, just cd to shadow worktree
if [ "$(pwd)" != "$SHADOW_WORKTREE" ]; then
    # BUILD STAGE: We're in main directory, copy file and cd to shadow worktree
    cp cs336_basics/layers.py "$SHADOW_WORKTREE/cs336_basics/layers.py"
    cd "$SHADOW_WORKTREE"
else
    # HARDEN STAGE: Already in shadow worktree, file was copied by submit-fix
    # Just cd to ensure we're in the right place (no-op but explicit)
    cd "$SHADOW_WORKTREE"
fi

# Record start time for performance measurement
start_time=$(python3 -c 'import time; print(time.time())')

# Smart pytest execution: use current Python environment if available, else uv
# This allows the validator to work both in production (uv) and testing (active venv)
if [ -n "$MASTERY_PYTHON" ]; then
    # Engine provided its Python executable - use it (works in test and production)
    # Add current directory to PYTHONPATH for imports
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    "$MASTERY_PYTHON" -m pytest tests/test_model.py::test_rmsnorm -v --tb=short --import-mode=importlib
elif [ -n "$VIRTUAL_ENV" ]; then
    # We're in an active virtual environment - use its Python explicitly
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    "$VIRTUAL_ENV/bin/python" -m pytest tests/test_model.py::test_rmsnorm -v --tb=short --import-mode=importlib
else
    # No active environment - use uv to create one
    uv run pytest tests/test_model.py::test_rmsnorm -v --tb=short --import-mode=importlib
fi

# Record end time
end_time=$(python3 -c 'import time; print(time.time())')

# Calculate duration
duration=$(python3 -c "print($end_time - $start_time)")

# Print performance metric in expected format
echo "PERFORMANCE_SECONDS: $duration"
