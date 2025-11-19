#!/bin/bash
# Local validator for XOR Operation in an Array
# Runs solution against example test cases

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_CASES="$SCRIPT_DIR/test_cases.json"

# Check if solution exists
if [ ! -f "$SCRIPT_DIR/solution.py" ]; then
    echo "❌ solution.py not found"
    exit 1
fi

# Run tests
python3 << 'EOF'
import json
import sys
from pathlib import Path

# Import user solution
sys.path.insert(0, str(Path(__file__).parent))
from solution import solve

# Load test cases
with open("test_cases.json") as f:
    test_cases = json.load(f)

passed = 0
failed = 0

for i, test in enumerate(test_cases["tests"], 1):
    try:
        result = solve(**test["input"])
        expected = test["expected"]
        
        if result == expected:
            print(f"✓ Test {i}: PASS")
            passed += 1
        else:
            print(f"✗ Test {i}: FAIL")
            print(f"  Input: {test['input']}")
            print(f"  Expected: {expected}")
            print(f"  Got: {result}")
            failed += 1
    except Exception as e:
        print(f"✗ Test {i}: ERROR - {e}")
        failed += 1

print(f"\nResults: {passed}/{passed + failed} passed")
sys.exit(0 if failed == 0 else 1)
EOF
