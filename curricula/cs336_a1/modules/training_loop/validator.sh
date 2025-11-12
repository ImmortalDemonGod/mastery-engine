#!/bin/bash
# Validator for training_loop module
# Runs pytest for training loop implementation in shadow worktree

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
    cp cs336_basics/training.py "$SHADOW_WORKTREE/cs336_basics/training.py"
    cd "$SHADOW_WORKTREE"
else
    # HARDEN STAGE: Already in shadow worktree, file was copied by submit-fix
    # Just cd to ensure we're in the right place (no-op but explicit)
    cd "$SHADOW_WORKTREE"
fi

# Record start time for performance measurement
start_time=$(python3 -c 'import time; print(time.time())')

# Smart pytest execution: use current Python environment if available, else uv
if [ -n "$MASTERY_PYTHON" ]; then
    # Engine provided its Python executable - use it
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    "$MASTERY_PYTHON" -m pytest tests/test_training.py::test_train_loop -v --tb=short --import-mode=importlib
elif [ -n "$VIRTUAL_ENV" ]; then
    # We're in an active virtual environment - use its Python explicitly
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    "$VIRTUAL_ENV/bin/python" -m pytest tests/test_training.py::test_train_loop -v --tb=short --import-mode=importlib
else
    # No active environment - use uv to create one
    uv run pytest tests/test_training.py::test_train_loop -v --tb=short --import-mode=importlib
fi

# Record end time
end_time=$(python3 -c 'import time; print(time.time())')

# Calculate duration
duration=$(python3 -c "print($end_time - $start_time)")

# Print performance metric in expected format
echo "PERFORMANCE_SECONDS: $duration"
