# TC 분석 : O(n * 2^(n+1)) 
from itertools import combinations
n,s = map(int,input().split())
items = list(map(int,input().split()))
cnt = 0
for i in range(1,n+1): # n
    lst = list(combinations(items,i)) # 각 nCi개 -> 총 2^n개
    for j in range(len(lst)): # 1~2^n
        sum = 0
        for item in lst[j]:   # 1~i = 1~n
            sum += item
        if sum == s:
            cnt +=1

print(cnt)