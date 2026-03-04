"""
Two Sum II - Input Array Is Sorted - Reference Solution
Time: O(n), Space: O(1)
"""


def twoSum(numbers, target):
    """
    Given a 1-indexed sorted array, find two numbers that add up to target.

    Args:
        numbers: Sorted list of integers (1-indexed in output)
        target: Target sum

    Returns:
        List of two 1-indexed indices [i, j] where numbers[i-1] + numbers[j-1] == target

    Example:
        >>> twoSum([2, 7, 11, 15], 9)
        [1, 2]
    """
    left, right = 0, len(numbers) - 1

    while left < right:
        current_sum = numbers[left] + numbers[right]

        if current_sum == target:
            return [left + 1, right + 1]
        elif current_sum < target:
            left += 1
        else:
            right -= 1

    return []


# Alias for compatibility with test runner
solve = twoSum
