def twoSum(nums, target):
    # Your implementation here
    for i in range(len(nums)):
        for j in range(len(nums)):
            print("i: ", i)
            print("j: ", j)
            if nums[i] + nums[j] == target:
                return [i,j]

print(twoSum([3,3], target=6))