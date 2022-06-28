from collections import deque
n = int(input())
de = deque()

for i in range(n,0,-1):
    de.appendleft(i)
    for _ in range(i):
        de.appendleft(de.pop())
print(*list(de))
