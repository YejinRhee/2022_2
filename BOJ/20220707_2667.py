# TC 분석 : O( n^2 * max(size)) 
from collections import deque

def is_valid_coord(y,x):
    return 0<=y<n and 0<=x<n

def bfs(q):
    size = 1
    while q: # O(max(size))
        py,px = q.pop()
        visited[py][px] = True
        for i in range(4):  # 4
            ny = py + dy[i]
            nx = px + dx[i]
            if is_valid_coord(ny,nx) and adj[ny][nx] == '1' and visited[ny][nx] == False:
                visited[ny][nx] = True
                q.append((ny,nx))
                size += 1
    return size
n = int(input())
adj = [list(input()) for _ in range(n)] # n
visited = [[False]*n for _ in range(n)] # n^2
visited[0][0] == True
dx = [1,0,-1,0]
dy = [0,1,0,-1]
sizes = []

# O( n^2 * max(size))
for i in range(n):
    for j in range(n): # n^2
        if visited[i][j] == False and adj[i][j] == '1': # num(연결요소)
            q = deque([])
            q.append((i,j)) 
            sizes.append(bfs(q))  # O(max(size)
sizes.sort()
print(len(sizes))
for size in sizes:
    print(size)
