#!/usr/bin/env bash
# Validator for HTTP Transport module

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

echo "=== HTTP Transport Validator ==="
echo "Testing fetch_document() implementation..."
echo ""

# Create temporary test script
TEST_SCRIPT=$(mktemp)
cat > "$TEST_SCRIPT" << 'EOF'
import sys
import time
import requests
from cs336_basics.utils import fetch_document

def test_valid_request():
    """Test successful HTTP request"""
    url = "https://httpbin.org/html"
    result = fetch_document(url)
    assert isinstance(result, str), "Return value must be a string"
    assert len(result) > 0, "Response should not be empty"
    assert "<html>" in result.lower(), "Should contain HTML content"
    print("✓ Valid HTTP request test passed")

def test_404_error():
    """Test 404 error handling"""
    try:
        fetch_document("https://httpbin.org/status/404")
        assert False, "Should have raised HTTPError for 404"
    except requests.exceptions.HTTPError as e:
        assert "404" in str(e), "Error should mention 404 status"
        print("✓ 404 error handling test passed")

def test_500_error():
    """Test 500 error handling"""
    try:
        fetch_document("https://httpbin.org/status/500")
        assert False, "Should have raised HTTPError for 500"
    except requests.exceptions.HTTPError as e:
        assert "500" in str(e), "Error should mention 500 status"
        print("✓ 500 error handling test passed")

def test_empty_url():
    """Test empty URL validation"""
    try:
        fetch_document("")
        assert False, "Should have raised ValueError for empty URL"
    except ValueError:
        print("✓ Empty URL validation test passed")

def test_performance():
    """Test performance"""
    start = time.time()
    fetch_document("https://httpbin.org/html")
    elapsed = time.time() - start
    assert elapsed < 3.0, f"Request took {elapsed:.2f}s, should be < 3s"
    print(f"✓ Performance test passed ({elapsed:.3f}s)")
    return elapsed

if __name__ == "__main__":
    try:
        test_empty_url()
        test_valid_request()
        test_404_error()
        test_500_error()
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

# Run the test
cd "$PROJECT_ROOT"
python3 "$TEST_SCRIPT"
TEST_EXIT=$?

# Cleanup
rm "$TEST_SCRIPT"

exit $TEST_EXIT
