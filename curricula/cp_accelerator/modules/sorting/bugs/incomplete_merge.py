"""
Buggy solution with incomplete merge logic.
"""

def sortArray(nums):
    if len(nums) <= 1:
        return nums
    
    mid = len(nums) // 2
    left = sortArray(nums[:mid])
    right = sortArray(nums[mid:])
    
    return merge(left, right)


def merge(left, right):
    """
    BUG: Forgets to append remaining elements from both arrays.
    Only appends from left array, ignoring remaining right elements.
    """
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # BUG: Only extends left, forgets right
    result.extend(left[i:])
    # Missing: result.extend(right[j:])
    
    return result
