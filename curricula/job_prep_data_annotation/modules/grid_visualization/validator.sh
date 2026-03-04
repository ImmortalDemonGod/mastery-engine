#!/usr/bin/env bash
# Validator for Grid Visualization module

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

echo "=== Grid Visualization Validator ==="
echo "Testing render_grid() implementation..."
echo ""

TEST_SCRIPT=$(mktemp)
cat > "$TEST_SCRIPT" << 'EOF'
import sys
import time
from cs336_basics.utils import render_grid

def test_small_grid():
    """Test small 2x2 grid"""
    coords = [(0, 0, 'H'), (1, 0, 'I'), (0, 1, 'L'), (1, 1, 'O')]
    result = render_grid(coords)
    expected = ["HI", "LO"]
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ Small grid test passed")

def test_sparse_grid():
    """Test sparse grid with gaps"""
    coords = [(0, 0, 'A'), (4, 2, 'B')]
    result = render_grid(coords)
    assert len(result) == 3, "Should have 3 rows"
    assert len(result[0]) == 5, "Should have 5 columns"
    assert result[0][0] == 'A'
    assert result[2][4] == 'B'
    assert result[1][2] == ' ', "Empty cells should be spaces"
    print("✓ Sparse grid test passed")

def test_single_coordinate():
    """Test grid with single coordinate"""
    coords = [(2, 3, 'X')]
    result = render_grid(coords)
    assert len(result) == 4, "Should have 4 rows (0-3)"
    assert len(result[0]) == 3, "Should have 3 columns (0-2)"
    assert result[3][2] == 'X'
    print("✓ Single coordinate test passed")

def test_reference_copying():
    """Critical test: Verify no reference copying bug"""
    coords = [(0, 0, 'A'), (1, 0, 'B'), (0, 1, 'C'), (1, 1, 'D')]
    result = render_grid(coords)
    
    # If implementation used [[' '] * w] * h, this would fail
    # because modifying [0][0] would affect all rows
    grid = [[' '] * 2 for _ in range(2)]
    for x, y, char in coords:
        grid[y][x] = char
    
    # Verify each position independently
    assert result[0][0] == 'A', "Position (0,0) should be 'A'"
    assert result[0][1] == 'B', "Position (1,0) should be 'B'"
    assert result[1][0] == 'C', "Position (0,1) should be 'C'"
    assert result[1][1] == 'D', "Position (1,1) should be 'D'"
    
    print("✓ Reference copying test passed (no aliasing)")

def test_edge_coordinates():
    """Test coordinates along edges"""
    coords = [(0, 0, 'A'), (5, 0, 'B'), (0, 3, 'C'), (5, 3, 'D')]
    result = render_grid(coords)
    assert result[0][0] == 'A'
    assert result[0][5] == 'B'
    assert result[3][0] == 'C'
    assert result[3][5] == 'D'
    print("✓ Edge coordinates test passed")

def test_empty_input():
    """Test empty coordinate list"""
    try:
        render_grid([])
        assert False, "Should raise ValueError for empty coords"
    except ValueError:
        print("✓ Empty input validation test passed")

def test_large_grid():
    """Test large grid performance"""
    # Create 10x10 grid with diagonal pattern
    coords = [(i, i, chr(65 + i % 26)) for i in range(10)]
    result = render_grid(coords)
    
    assert len(result) == 10
    assert len(result[0]) == 10
    
    # Verify diagonal
    for i in range(10):
        assert result[i][i] == chr(65 + i % 26)
    
    print("✓ Large grid test passed")

def test_performance():
    """Test performance with realistic data"""
    # 50 coordinates in a 20x20 grid
    coords = [(i % 20, i // 20, chr(65 + i % 26)) for i in range(50)]
    
    start = time.time()
    result = render_grid(coords)
    elapsed = time.time() - start
    
    assert len(result) > 0
    assert elapsed < 0.1, f"Rendering took {elapsed:.3f}s (should be < 0.1s)"
    print(f"✓ Performance test passed ({elapsed:.4f}s)")
    return elapsed

if __name__ == "__main__":
    try:
        test_empty_input()
        test_small_grid()
        test_single_coordinate()
        test_sparse_grid()
        test_reference_copying()
        test_edge_coordinates()
        test_large_grid()
        perf = test_performance()
        
        print("")
        print("=== All Tests Passed ===")
        print(f"PERFORMANCE_SECONDS: {perf:.4f}")
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
