"""
Climbing Stairs - Reference Solution
Time: O(n), Space: O(1)
"""


def climbStairs(n):
    """
    Count the number of distinct ways to climb n stairs,
    taking 1 or 2 steps at a time.

    Args:
        n: Number of stairs (1 <= n <= 45)

    Returns:
        Number of distinct ways to reach the top

    Example:
        >>> climbStairs(3)
        3
    """
    if n <= 2:
        return n

    prev2, prev1 = 1, 2

    for i in range(3, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current

    return prev1


# Alias for compatibility with test runner
solve = climbStairs
