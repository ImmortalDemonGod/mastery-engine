#!/bin/bash
# Validator for data_loader module
# Runs pytest for get_batch function in shadow worktree

set -e

if [ -z "$SHADOW_WORKTREE" ]; then
    echo "ERROR: SHADOW_WORKTREE environment variable not set"
    exit 1
fi

if [ "$(pwd)" != "$SHADOW_WORKTREE" ]; then
    cp cs336_basics/utils.py "$SHADOW_WORKTREE/cs336_basics/utils.py"
    cd "$SHADOW_WORKTREE"
else
    cd "$SHADOW_WORKTREE"
fi

start_time=$(python3 -c 'import time; print(time.time())')

if [ -n "$MASTERY_PYTHON" ]; then
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    "$MASTERY_PYTHON" -m pytest tests/test_training.py::test_get_batch -v --tb=short --import-mode=importlib
elif [ -n "$VIRTUAL_ENV" ]; then
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    "$VIRTUAL_ENV/bin/python" -m pytest tests/test_training.py::test_get_batch -v --tb=short --import-mode=importlib
else
    uv run pytest tests/test_training.py::test_get_batch -v --tb=short --import-mode=importlib
fi

end_time=$(python3 -c 'import time; print(time.time())')
duration=$(python3 -c "print($end_time - $start_time)")

echo "PERFORMANCE_SECONDS: $duration"
