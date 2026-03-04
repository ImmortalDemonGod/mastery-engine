"""
Range Sum Query - Immutable - Reference Solution
Build: O(n), Query: O(1), Space: O(n)
"""


def sumRange(nums, left, right):
    """
    Return the sum of elements between indices left and right inclusive.
    Uses prefix sum array for O(1) range queries.

    Args:
        nums: List of integers
        left: Left index (inclusive)
        right: Right index (inclusive)

    Returns:
        Sum of nums[left..right]

    Example:
        >>> sumRange([-2, 0, 3, -5, 2, -1], 0, 2)
        1
    """
    # Build prefix sum array: prefix[i] = sum of nums[0..i-1]
    prefix = [0] * (len(nums) + 1)
    for i in range(len(nums)):
        prefix[i + 1] = prefix[i] + nums[i]

    return prefix[right + 1] - prefix[left]


# Alias for compatibility with test runner
solve = sumRange
