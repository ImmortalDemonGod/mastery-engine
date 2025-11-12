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
if [ "$(pwd)" != "$SHADOW_WORKTREE" ]; then
    # BUILD STAGE: Copy from main directory to shadow worktree
    cp cs336_basics/utils.py "$SHADOW_WORKTREE/cs336_basics/utils.py"
    cd "$SHADOW_WORKTREE"
else
    # HARDEN STAGE: Already in shadow worktree, file was copied by submit-fix
    # Just stay here and run tests
    true
fi

# Record start time for performance measurement
start_time=$(python3 -c 'import time; print(time.time())')

# Run the softmax test
uv run pytest tests/test_nn_utils.py::test_softmax_matches_pytorch -v

# Record end time
end_time=$(python3 -c 'import time; print(time.time())')

# Calculate duration
duration=$(python3 -c "print($end_time - $start_time)")

# Print performance metric in expected format
echo "PERFORMANCE_SECONDS: $duration"
