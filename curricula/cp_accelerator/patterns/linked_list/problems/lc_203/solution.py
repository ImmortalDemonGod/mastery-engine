"""
Remove Linked List Elements - Reference Solution
Time: O(n), Space: O(1)
Note: Uses array representation for compatibility with test runner.
"""


def removeElements(head, val):
    """
    Remove all elements from a list that have a specific value.

    Args:
        head: List of integers (array representation of linked list)
        val: Value to remove

    Returns:
        List with all occurrences of val removed

    Example:
        >>> removeElements([1, 2, 6, 3, 4, 5, 6], 6)
        [1, 2, 3, 4, 5]
    """
    return [x for x in head if x != val]


# Alias for compatibility with test runner
solve = removeElements
