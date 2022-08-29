'''
인덱스와 원소를 pythonic하게 표현하는 방법 - enumerate 
for entry in enumerate(['A', 'B', 'C']):
...     print(entry)
(0, 'A')
(1, 'B')
(2, 'C')
enumerate() 함수는 기본적으로 인덱스와 원소로 이루어진 터플(tuple)을 만들어줍니다.
'''

from itertools import permutations

n = int(input())
nums = list(map(int, input().split()))
ops_lst = []
for i, op_num in enumerate(list(map(int, input().split()))):
    ops_lst += [i]*op_num

ans_max = -1000000001
ans_min = 1000000001

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

    if res > ans_max:
        ans_max = res

    if res < ans_min:
        ans_min = res


print(ans_max)
print(ans_min)
