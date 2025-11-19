#!/usr/bin/env bash
# Validator for Standard Library Augmentation module

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

echo "=== Standard Library Augmentation Validator ==="
echo "Testing deque, heapq, and bisect implementations..."
echo ""

TEST_SCRIPT=$(mktemp)
cat > "$TEST_SCRIPT" << 'EOF'
import sys
import time
from collections import deque
import heapq
import bisect
from cs336_basics.utils import shortest_path_bfs, dijkstra_shortest_path, count_in_range

# ===== BFS Tests =====
def test_bfs_simple():
    """Test basic BFS"""
    graph = {0: [1, 2], 1: [3], 2: [3], 3: []}
    result = shortest_path_bfs(graph, 0, 3)
    assert result == 2, f"Expected 2, got {result}"
    print("✓ BFS simple path test passed")

def test_bfs_no_path():
    """Test disconnected graph"""
    graph = {0: [1], 1: [], 2: [3], 3: []}
    result = shortest_path_bfs(graph, 0, 3)
    assert result == -1, f"Expected -1 for no path, got {result}"
    print("✓ BFS no path test passed")

def test_bfs_cycle():
    """Test graph with cycle"""
    graph = {0: [1], 1: [2], 2: [0, 3], 3: []}
    result = shortest_path_bfs(graph, 0, 3)
    assert result == 3, f"Expected 3, got {result}"
    print("✓ BFS cycle test passed")

def test_bfs_performance():
    """Test BFS performance on large graph"""
    # Create chain: 0 -> 1 -> 2 -> ... -> 999
    graph = {i: [i+1] if i < 999 else [] for i in range(1000)}
    
    start = time.time()
    result = shortest_path_bfs(graph, 0, 999)
    elapsed = time.time() - start
    
    assert result == 999
    assert elapsed < 0.1, f"BFS too slow: {elapsed:.3f}s"
    print(f"✓ BFS performance test passed ({elapsed:.4f}s)")

# ===== Dijkstra Tests =====
def test_dijkstra_simple():
    """Test basic Dijkstra"""
    graph = {0: [(1, 4), (2, 1)], 1: [(3, 1)], 2: [(3, 5)], 3: []}
    result = dijkstra_shortest_path(graph, 0, 3)
    assert result == 5, f"Expected 5, got {result}"  # 0 -> 1 -> 3
    print("✓ Dijkstra simple path test passed")

def test_dijkstra_unreachable():
    """Test unreachable node"""
    graph = {0: [(1, 1)], 1: [], 2: [(3, 1)], 3: []}
    result = dijkstra_shortest_path(graph, 0, 3)
    assert result == float('inf'), f"Expected inf for unreachable, got {result}"
    print("✓ Dijkstra unreachable test passed")

def test_dijkstra_multiple_paths():
    """Test choosing optimal path"""
    graph = {
        0: [(1, 10), (2, 3)],
        1: [(3, 2)],
        2: [(1, 1), (3, 8)],
        3: []
    }
    result = dijkstra_shortest_path(graph, 0, 3)
    assert result == 6, f"Expected 6 (0->2->1->3), got {result}"
    print("✓ Dijkstra multiple paths test passed")

def test_dijkstra_performance():
    """Test Dijkstra on larger graph"""
    # Create chain with varying weights
    graph = {i: [(i+1, i % 10 + 1)] if i < 500 else [] for i in range(501)}
    
    start = time.time()
    result = dijkstra_shortest_path(graph, 0, 500)
    elapsed = time.time() - start
    
    assert result > 0  # Some path exists
    assert elapsed < 0.2, f"Dijkstra too slow: {elapsed:.3f}s"
    print(f"✓ Dijkstra performance test passed ({elapsed:.4f}s)")

# ===== Bisect Tests =====
def test_bisect_simple():
    """Test basic range count"""
    arr = [1, 2, 4, 4, 5, 7]
    result = count_in_range(arr, 3, 5)
    assert result == 3, f"Expected 3 ([4,4,5]), got {result}"
    print("✓ Bisect simple range test passed")

def test_bisect_exact_bounds():
    """Test exact boundary values"""
    arr = [1, 2, 3, 4, 5]
    result = count_in_range(arr, 2, 4)
    assert result == 3, f"Expected 3 ([2,3,4]), got {result}"
    print("✓ Bisect exact bounds test passed")

def test_bisect_empty():
    """Test empty array"""
    arr = []
    result = count_in_range(arr, 1, 10)
    assert result == 0, f"Expected 0 for empty array, got {result}"
    print("✓ Bisect empty array test passed")

def test_bisect_no_match():
    """Test range with no elements"""
    arr = [1, 2, 5, 6]
    result = count_in_range(arr, 3, 4)
    assert result == 0, f"Expected 0 for no match, got {result}"
    print("✓ Bisect no match test passed")

def test_bisect_duplicates():
    """Test with many duplicates"""
    arr = [1, 3, 3, 3, 3, 3, 5]
    result = count_in_range(arr, 3, 3)
    assert result == 5, f"Expected 5 duplicates of 3, got {result}"
    print("✓ Bisect duplicates test passed")

def test_bisect_performance():
    """Test bisect on large array"""
    arr = sorted(range(0, 100000, 2))  # Even numbers 0 to 99998
    
    start = time.time()
    result = count_in_range(arr, 10000, 20000)
    elapsed = time.time() - start
    
    expected = len([x for x in arr if 10000 <= x <= 20000])
    assert result == expected
    assert elapsed < 0.001, f"Bisect too slow: {elapsed:.5f}s"
    print(f"✓ Bisect performance test passed ({elapsed:.6f}s)")
    return elapsed

if __name__ == "__main__":
    try:
        # BFS tests
        test_bfs_simple()
        test_bfs_no_path()
        test_bfs_cycle()
        test_bfs_performance()
        
        # Dijkstra tests
        test_dijkstra_simple()
        test_dijkstra_unreachable()
        test_dijkstra_multiple_paths()
        test_dijkstra_performance()
        
        # Bisect tests
        test_bisect_simple()
        test_bisect_exact_bounds()
        test_bisect_empty()
        test_bisect_no_match()
        test_bisect_duplicates()
        perf = test_bisect_performance()
        
        print("")
        print("=== All Tests Passed ===")
        print(f"PERFORMANCE_SECONDS: {perf:.6f}")
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
