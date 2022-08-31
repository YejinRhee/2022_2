from itertools import permutations

n = int(input())
nums = list(map(int, input().split()))
ops_lst = []
for i, op_num in enumerate(list(map(int, input().split()))):
    ops_lst += [i]*op_num
my_max = -1_000_000_001
my_min = 1_000_000_001
for permu in set(permutations(ops_lst, n-1)):
    res = nums[0]
    for i, op in enumerate(permu):
        if op == 0:
            res += nums[i+1]
        elif op == 1:
            res -= nums[i+1]
        elif op == 2:
            res *= nums[i+1]
        else:
            if res >= 0:
                res //= nums[i+1]
            else:
                res = -(-res//nums[i+1])
    if res > my_max:
        my_max = res
    if res < my_min:
        my_min = res

print(my_max)
print(my_min)
