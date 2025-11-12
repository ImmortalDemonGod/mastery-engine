#!/bin/bash
# Validator for softmax module
# Runs pytest for softmax implementation in shadow worktree

set -e  # Exit on error

# SHADOW_WORKTREE environment variable is set by the ValidationSubsystem
if [ -z "$SHADOW_WORKTREE" ]; then
    echo "ERROR: SHADOW_WORKTREE environment variable not set"
    exit 1
fi

# Determine if we're already in shadow worktree (for harden stage)
# or in main directory (for build stage)
echo "DEBUG: pwd=$(pwd)" >&2
echo "DEBUG: SHADOW_WORKTREE=$SHADOW_WORKTREE" >&2
if [ "$(pwd)" != "$SHADOW_WORKTREE" ]; then
    echo "DEBUG: BUILD STAGE - Copying from main directory" >&2
    # BUILD STAGE: Copy from main directory to shadow worktree
    cp cs336_basics/utils.py "$SHADOW_WORKTREE/cs336_basics/utils.py"
    cd "$SHADOW_WORKTREE"
else
    echo "DEBUG: HARDEN STAGE - Using file already copied by submit-fix" >&2
    # HARDEN STAGE: Already in shadow worktree, file was copied by submit-fix
    # Just stay here and run tests
    true
fi
echo "DEBUG: After check, pwd=$(pwd)" >&2

# Record start time for performance measurement
start_time=$(python3 -c 'import time; print(time.time())')

# Smart pytest execution: use current Python environment if available, else uv
# This allows the validator to work both in production (uv) and testing (active venv)
if [ -n "$MASTERY_PYTHON" ]; then
    # Engine provided its Python executable - use it (works in test and production)
    # Add current directory to PYTHONPATH for imports
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    # Use importlib mode to properly handle relative imports in tests
    "$MASTERY_PYTHON" -m pytest tests/test_nn_utils.py::test_softmax_matches_pytorch -v --tb=short --import-mode=importlib
elif [ -n "$VIRTUAL_ENV" ]; then
    # We're in an active virtual environment - use its Python explicitly
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    "$VIRTUAL_ENV/bin/python" -m pytest tests/test_nn_utils.py::test_softmax_matches_pytorch -v --tb=short --import-mode=importlib
else
    # No active environment - use uv to create one
    uv run pytest tests/test_nn_utils.py::test_softmax_matches_pytorch -v --tb=short --import-mode=importlib
fi

# Record end time
end_time=$(python3 -c 'import time; print(time.time())')

# Calculate duration
duration=$(python3 -c "print($end_time - $start_time)")

# Print performance metric in expected format
echo "PERFORMANCE_SECONDS: $duration"
