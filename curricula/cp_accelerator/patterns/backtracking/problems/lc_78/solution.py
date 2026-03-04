"""
Subsets - Reference Solution
Time: O(n * 2^n), Space: O(n * 2^n)
"""


def subsets(nums):
    """
    Generate all possible subsets (the power set).

    Args:
        nums: List of unique integers

    Returns:
        List of all subsets

    Example:
        >>> subsets([1, 2, 3])
        [[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]]
    """
    result = []

    def backtrack(start, current):
        result.append(current[:])

        for i in range(start, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()

    backtrack(0, [])
    return result


# Alias for compatibility with test runner
solve = subsets
