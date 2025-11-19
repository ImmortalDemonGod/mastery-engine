#!/bin/bash
# Test script that runs the same commands as GitHub Actions CI
# Run this before pushing to verify tests will pass

set -e  # Exit on first error

echo "========================================="
echo "  Running CI Tests Locally"
echo "========================================="
echo ""

echo "📋 This runs the EXACT commands from .github/workflows/tests.yml"
echo ""

# Check if in project root
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Must run from project root directory"
    exit 1
fi

# Check if venv activated or uv available
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv not found. Install with: pip install uv"
    exit 1
fi

echo "Step 1: Running engine tests..."
echo "Command: uv run pytest tests/engine/ -v -m \"not integration\" --tb=short"
echo ""

uv run pytest tests/engine/ -v \
    -m "not integration" \
    --tb=short

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "========================================="
    echo "  ✅ ALL TESTS PASSED"
    echo "========================================="
    echo ""
    echo "Safe to push! GitHub Actions should pass."
else
    echo "========================================="
    echo "  ❌ TESTS FAILED"
    echo "========================================="
    echo ""
    echo "Fix failing tests before pushing."
    echo "GitHub Actions will fail with these results."
    exit 1
fi
