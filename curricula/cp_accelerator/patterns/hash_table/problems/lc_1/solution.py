"""
Two Sum - Reference Solution
Time: O(n), Space: O(n)
"""

def twoSum(nums, target):
    """
    Given an array of integers nums and an integer target, return indices 
    of the two numbers such that they add up to target.
    
    Args:
        nums: List of integers
        target: Target sum
        
    Returns:
        List of two indices [i, j] where nums[i] + nums[j] == target
        
    Example:
        >>> twoSum([2, 7, 11, 15], 9)
        [0, 1]
    """
    seen = {}  # Maps number -> index
    
    for i, num in enumerate(nums):
        complement = target - num
        
        # Check if complement exists BEFORE inserting current element
        # This ensures we don't use the same index twice
        if complement in seen:
            return [seen[complement], i]
        
        # Insert current element after checking
        seen[num] = i
    
    # Problem guarantees exactly one solution, so we never reach here
    return []