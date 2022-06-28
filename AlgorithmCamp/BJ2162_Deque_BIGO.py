from collections import deque
n = int(input())
de = deque([]) # de = deque(range(1,n+1))
for i in range(n):
    de.append(i+1)
while len(de) > 1: # 매 수행마다 len은 1씩 줄어
    de.popleft()
    de.append(de.popleft())
print(de)
