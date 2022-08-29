import heapq as hq
import sys
import heapq as hq

input = sys.stdin.readline

min_heap = []
for _ in range(int(input())):
    hq.heappush(min_heap, int(input()))

ans = 0
while len(min_heap) > 1:
    val = hq.heappop(min_heap) + hq.heappop(min_heap)
    ans += val
    hq.heappush(min_heap, val)

print(ans)


'''
copyyyyyyyy
시간복잡도 :
초기화 n
for문 (n-1) * 3logn
'''
min_heap = []
n = int(input())
for _ in range(n):
  hq.heappush(min_heap, int(input()))
ans = 0
for _ in range(n-1):
  val = hq.heappop(min_heap)+hq.heappop(min_heap)
  ans += val
  hq.heappush(min_heap, val)

print(ans)
