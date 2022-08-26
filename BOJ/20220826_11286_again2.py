import heapq as hq, sys

input = sys.stdin.readline
min_heap = []
'''
hint 
min_heap에 튜플로 값을 넣으면 
튜플의 첫번째 인자로 min 하고, 끝나면 두번째 인자로 min 비교해줌
'''

n = int(input())
for _ in range(n):
    x = int(input())
    if x : 
        hq.heappush(min_heap, (abs(x), x))
    else:
        if min_heap:
            print(hq.heappop(min_heap)[1])
        else:
            print("0")
