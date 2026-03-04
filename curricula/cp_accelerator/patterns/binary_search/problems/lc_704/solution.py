"""
Binary Search - Reference Solution
Time: O(log n), Space: O(1)
"""


def search(nums, target):
    """
    Given a sorted array of integers and a target, return its index or -1.

    Args:
        nums: Sorted list of unique integers
        target: Integer to find

    Returns:
        Index of target in nums, or -1 if not found

    Example:
        >>> search([-1, 0, 3, 5, 9, 12], 9)
        4
    """
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1


# Alias for compatibility with test runner
solve = search
