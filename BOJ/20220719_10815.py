from bisect import bisect_left, bisect_right
n = int(input())
cards = sorted(map(int,input().split()))
m = int(input())
ans = []
for i in map(int,input().split()):
    ans.append(bisect_right(cards,i)-bisect_left(cards,i))

print(*ans)