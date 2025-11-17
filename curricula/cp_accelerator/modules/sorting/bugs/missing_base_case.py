"""
Buggy solution missing proper base case handling.
"""

def sortArray(nums):
    """
    BUG: Missing check for empty array, only checks len <= 1
    This will cause infinite recursion on empty arrays.
    """
    if len(nums) == 1:  # BUG: Should be <= 1 to handle empty arrays
        return nums
    
    mid = len(nums) // 2
    left = sortArray(nums[:mid])
    right = sortArray(nums[mid:])
    
    return merge(left, right)


def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result
