#!/usr/bin/env bash
# Validator for Data Parsing & Extraction module

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

echo "=== Data Parsing & Extraction Validator ==="
echo "Testing extract_coordinates() implementation..."
echo ""

TEST_SCRIPT=$(mktemp)
cat > "$TEST_SCRIPT" << 'EOF'
import sys
import time
from cs336_basics.utils import extract_coordinates

def test_consistent_format():
    """Test parsing with consistent formatting"""
    html = """
    <div>
      <span>x=10, y=5, char=A</span>
      <span>x=20, y=10, char=B</span>
    </div>
    """
    result = extract_coordinates(html)
    expected = [(10, 5, 'A'), (20, 10, 'B')]
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ Consistent format test passed")

def test_inconsistent_format():
    """Test parsing with varied spacing"""
    html = """
    <div>
      <span>x = 10, y= 5, char =A</span>
      <span>x=20,y=10,char=B</span>
      <span>x= 30 , y =15, char = C</span>
    </div>
    """
    result = extract_coordinates(html)
    expected = [(10, 5, 'A'), (20, 10, 'B'), (30, 15, 'C')]
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ Inconsistent format test passed")

def test_many_coordinates():
    """Test parsing with many coordinates"""
    spans = [f"<span>x={i}, y={i*2}, char={'ABCDEFGHIJ'[i%10]}</span>" for i in range(15)]
    html = f"<div>{''.join(spans)}</div>"
    result = extract_coordinates(html)
    assert len(result) == 15, f"Expected 15 coordinates, got {len(result)}"
    assert result[0] == (0, 0, 'A')
    assert result[5] == (5, 10, 'F')
    print("✓ Many coordinates test passed")

def test_single_coordinate():
    """Test with single coordinate"""
    html = "<span>x=42, y=17, char=Z</span>"
    result = extract_coordinates(html)
    assert result == [(42, 17, 'Z')]
    print("✓ Single coordinate test passed")

def test_empty_html():
    """Test empty HTML handling"""
    try:
        extract_coordinates("")
        assert False, "Should raise ValueError for empty HTML"
    except ValueError:
        print("✓ Empty HTML validation test passed")

def test_sorted_output():
    """Test that output is sorted by x coordinate"""
    html = """
    <div>
      <span>x=30, y=1, char=C</span>
      <span>x=10, y=2, char=A</span>
      <span>x=20, y=3, char=B</span>
    </div>
    """
    result = extract_coordinates(html)
    x_values = [coord[0] for coord in result]
    assert x_values == sorted(x_values), "Coordinates should be sorted by x"
    print("✓ Sorted output test passed")

def test_performance():
    """Test performance with large input"""
    spans = [f"<span>x={i}, y={i}, char=A</span>" for i in range(100)]
    html = f"<div>{''.join(spans)}</div>"
    
    start = time.time()
    result = extract_coordinates(html)
    elapsed = time.time() - start
    
    assert len(result) == 100
    assert elapsed < 0.5, f"Parsing 100 coordinates took {elapsed:.3f}s (should be < 0.5s)"
    print(f"✓ Performance test passed ({elapsed:.3f}s)")
    return elapsed

if __name__ == "__main__":
    try:
        test_empty_html()
        test_consistent_format()
        test_inconsistent_format()
        test_single_coordinate()
        test_many_coordinates()
        test_sorted_output()
        perf = test_performance()
        
        print("")
        print("=== All Tests Passed ===")
        print(f"PERFORMANCE_SECONDS: {perf:.3f}")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
EOF

cd "$PROJECT_ROOT"
python3 "$TEST_SCRIPT"
TEST_EXIT=$?

rm "$TEST_SCRIPT"
exit $TEST_EXIT
