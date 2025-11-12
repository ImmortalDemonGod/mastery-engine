#!/bin/bash
# Validator for linear module
# Runs pytest for Linear layer implementation in shadow worktree

set -e  # Exit on error

# SHADOW_WORKTREE environment variable is set by the ValidationSubsystem
if [ -z "$SHADOW_WORKTREE" ]; then
    echo "ERROR: SHADOW_WORKTREE environment variable not set"
    exit 1
fi

# Determine if we're in BUILD or HARDEN stage
if [ "$(pwd)" != "$SHADOW_WORKTREE" ]; then
    # BUILD STAGE: Copy file and cd to shadow worktree
    cp cs336_basics/layers.py "$SHADOW_WORKTREE/cs336_basics/layers.py"
    cd "$SHADOW_WORKTREE"
else
    # HARDEN STAGE: Already in shadow worktree
    cd "$SHADOW_WORKTREE"
fi

# Record start time
start_time=$(python3 -c 'import time; print(time.time())')

# Smart pytest execution
if [ -n "$MASTERY_PYTHON" ]; then
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    "$MASTERY_PYTHON" -m pytest tests/test_model.py::test_linear -v --tb=short --import-mode=importlib
elif [ -n "$VIRTUAL_ENV" ]; then
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    "$VIRTUAL_ENV/bin/python" -m pytest tests/test_model.py::test_linear -v --tb=short --import-mode=importlib
else
    uv run pytest tests/test_model.py::test_linear -v --tb=short --import-mode=importlib
fi

# Record end time
end_time=$(python3 -c 'import time; print(time.time())')
duration=$(python3 -c "print($end_time - $start_time)")

echo "PERFORMANCE_SECONDS: $duration"
