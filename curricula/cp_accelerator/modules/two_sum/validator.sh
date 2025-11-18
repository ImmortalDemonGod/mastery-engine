#!/bin/bash
# Local validator for Two Sum (LC-1)
# Runs solution against example test cases from the problem statement

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
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
try:
    from curricula.cp_accelerator.modules.two_sum.solution import twoSum
except ImportError:
    print("❌ Could not import twoSum function from solution.py")
    print("   Make sure your solution.py defines: def twoSum(nums, target):")
    sys.exit(1)

# Load test cases
test_cases_path = Path(__file__).parent / "test_cases.json"
with open(test_cases_path) as f:
    test_data = json.load(f)

passed = 0
failed = 0

print(f"\n🧪 Running {len(test_data['tests'])} test cases for Two Sum...\n")

for test in test_data["tests"]:
    try:
        result = twoSum(**test["input"])
        expected = test["expected"]
        
        # Two Sum can return indices in any order, so we need to sort for comparison
        result_sorted = sorted(result) if isinstance(result, (list, tuple)) else result
        expected_sorted = sorted(expected) if isinstance(expected, (list, tuple)) else expected
        
        if result_sorted == expected_sorted:
            print(f"✓ Test {test['id']}: PASS")
            print(f"  Input: nums={test['input']['nums']}, target={test['input']['target']}")
            print(f"  Expected: {expected} | Got: {result}")
            passed += 1
        else:
            print(f"✗ Test {test['id']}: FAIL")
            print(f"  Input: nums={test['input']['nums']}, target={test['input']['target']}")
            print(f"  Expected: {expected}")
            print(f"  Got: {result}")
            failed += 1
    except Exception as e:
        print(f"✗ Test {test['id']}: ERROR - {e}")
        print(f"  Input: nums={test['input']['nums']}, target={test['input']['target']}")
        failed += 1

print(f"\n{'='*60}")
print(f"Results: {passed}/{passed + failed} tests passed")
print(f"{'='*60}\n")

if failed == 0:
    print("🎉 All tests passed! Ready to submit.")
else:
    print(f"❌ {failed} test(s) failed. Keep debugging!")

sys.exit(0 if failed == 0 else 1)
EOF
