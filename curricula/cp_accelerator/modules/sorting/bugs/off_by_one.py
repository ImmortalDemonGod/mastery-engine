"""
Buggy solution with off-by-one error in merge function.
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
    BUG: Uses < instead of <= in comparison, causing instability
    with duplicate elements and potential incorrect ordering.
    """
    result = []
    i = j = 0
    
    # BUG: This line should be left[i] <= right[j] for stable sort
    while i < len(left) and j < len(right):
        if left[i] < right[j]:  # BUG HERE
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result
