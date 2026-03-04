"""
Valid Parentheses - Reference Solution
Time: O(n), Space: O(n)
"""


def isValid(s):
    """
    Determine if the input string has valid bracket nesting.

    Args:
        s: String containing only '(', ')', '{', '}', '[', ']'

    Returns:
        True if brackets are validly nested, False otherwise

    Example:
        >>> isValid("()[]{}")
        True
        >>> isValid("(]")
        False
    """
    stack = []
    matching = {')': '(', '}': '{', ']': '['}

    for char in s:
        if char in matching:
            if not stack or stack[-1] != matching[char]:
                return False
            stack.pop()
        else:
            stack.append(char)

    return len(stack) == 0


# Alias for compatibility with test runner
solve = isValid
