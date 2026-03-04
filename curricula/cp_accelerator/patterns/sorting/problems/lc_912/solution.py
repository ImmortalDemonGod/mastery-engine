"""
Reference solution for LeetCode 912: Sort an Array
Uses merge sort for guaranteed O(n log n) time complexity.
"""

def sortArray(nums):
    """
    Sort an array using merge sort.
    
    Time Complexity: O(n log n)
    Space Complexity: O(n) for the temporary array during merging
    
    Args:
        nums: List of integers to sort
        
    Returns:
        Sorted list in non-decreasing order
    """
    if len(nums) <= 1:
        return nums
    
    # Divide
    mid = len(nums) // 2
    left = sortArray(nums[:mid])
    right = sortArray(nums[mid:])
    
    # Conquer (merge)
    return merge(left, right)


def merge(left, right):
    """
    Merge two sorted arrays into one sorted array.
    
    This is the key subroutine of merge sort that maintains the invariant:
    the output is sorted if both inputs are sorted.
    """
    result = []
    i = j = 0
    
    # Merge while both arrays have elements
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Append remaining elements
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result


# Alias for compatibility with test runner
solve = sortArray
