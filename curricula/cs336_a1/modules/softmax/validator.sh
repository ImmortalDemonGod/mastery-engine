#!/bin/bash
# Validator for softmax module
# Runs pytest for softmax implementation and extracts performance metrics

set -e  # Exit on error

# Copy implementation from workspace to actual location
# The workspace is for user isolation, but tests expect code in cs336_basics/
cp cs336_basics/utils.py "$REPO_ROOT/cs336_basics/utils.py" 2>/dev/null || true

# Change to repo root to run tests
cd "$REPO_ROOT"

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
